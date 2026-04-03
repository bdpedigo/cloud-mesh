"""
worker.py — GKE batch worker for condensed HKS feature extraction.

Each pod runs this script in a loop, pulling tasks from the configured
task queue and extracting mesh features for a single (root_id, datastack,
materialization_version) triple.

Configuration: config.toml (job section) + hks_parameters.toml.
Environment variables prefixed with CLOUD_MESH_ override config.toml values.
"""

import logging
import os
import sys
import traceback
from pathlib import Path

import tomllib
from caveclient import CAVEclient, set_session_defaults
from cloudfiles import CloudFiles
from meshmash import condensed_hks_pipeline, save_condensed_features
from taskqueue import TaskQueue, queueable
from taskqueue.queueablefns import REGISTRY as _tq_registry

# ── Load config ───────────────────────────────────────────────────────────────

_config_path = Path(
    os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml")
)
_hks_path = Path(
    os.environ.get("HKS_PARAMS_PATH", Path(__file__).parent / "hks_parameters.toml")
)

with open(_config_path, "rb") as _f:
    _config = tomllib.load(_f)["job"]

with open(_hks_path, "rb") as _f:
    _hks = tomllib.load(_f)


def _cfg(key, env_var, cast=str):
    """Return env var if set, otherwise config.toml value."""
    raw = os.environ.get(env_var)
    if raw is not None:
        return cast(raw)
    return _config[key]


QUEUE_URL = _cfg("queue_url", "CLOUD_MESH_QUEUE_URL")
OUTPUT_BUCKET = _cfg("output_bucket", "CLOUD_MESH_OUTPUT_BUCKET")
LEASE_SECONDS = _cfg("lease_seconds", "CLOUD_MESH_LEASE_SECONDS", int)
MAX_RUNS = _cfg("max_runs", "CLOUD_MESH_MAX_RUNS", int)
N_JOBS = _cfg("n_jobs", "CLOUD_MESH_N_JOBS", int)
RECOMPUTE = (
    _cfg("recompute", "CLOUD_MESH_RECOMPUTE", lambda v: v.lower() == "true")
    if os.environ.get("CLOUD_MESH_RECOMPUTE")
    else bool(_config["recompute"])
)
LOGGING_LEVEL = _cfg("logging_level", "CLOUD_MESH_LOGGING_LEVEL")

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("worker")

# ── Task ──────────────────────────────────────────────────────────────────────


def _output_path(output_bucket: str, datastack: str, root_id: int) -> str:
    return f"{output_bucket.rstrip('/')}/{datastack}/features/{root_id}.npz"


@queueable
def run_for_root(root_id: int, datastack: str) -> None:
    """Extract condensed HKS features for one neuron and write to GCS."""
    log.info("root ID=%s, datastack=%s - starting", root_id, datastack)
    out_path = _output_path(OUTPUT_BUCKET, datastack, root_id)
    cf = CloudFiles(f"{OUTPUT_BUCKET.rstrip('/')}/{datastack}/features")

    if not RECOMPUTE and cf.exists(f"{root_id}.npz"):
        log.info(
            "root ID=%s, datastack=%s - already done, skipping", root_id, datastack
        )
        return

    try:
        set_session_defaults(max_retries=5, backoff_factor=4, backoff_max=240)
        client = CAVEclient(datastack)

        log.info("root ID=%s, datastack=%s - loading mesh", root_id, datastack)
        cv = client.info.segmentation_cloudvolume(progress=False)
        mesh = cv.mesh.get(root_id, **_hks.get("cv_mesh_get", {}))[root_id]
        mesh = (mesh.vertices, mesh.faces)
        log.info(
            "root ID=%s, datastack=%s - mesh loaded: %d vertices, %d faces",
            root_id,
            datastack,
            mesh[0].shape[0],
            mesh[1].shape[0],
        )

        log.info("root ID=%s, datastack=%s - extracting features", root_id, datastack)
        result = condensed_hks_pipeline(
            mesh,
            verbose=False,
            n_jobs=N_JOBS,
            **_hks["condensed_hks_pipeline"],
        )
        log.info("root ID=%s, datastack=%s - features extracted", root_id, datastack)

        log.info(
            "root ID=%s, datastack=%s - saving features → %s",
            root_id,
            datastack,
            out_path,
        )
        save_condensed_features(
            out_path,
            result.condensed_features,
            result.labels,
            **_hks.get("save_condensed_features", {}),
        )
        log.info(
            "root ID=%s, datastack=%s - saved features → %s",
            root_id,
            datastack,
            out_path,
        )

    except Exception:
        exc = traceback.format_exc()
        log.error(
            "root ID=%s, datastack=%s - failed\n%s",
            root_id,
            datastack,
            exc,
        )
        raise


# Ensure the function is findable under ('worker', 'run_for_root') regardless of
# whether this module is run as __main__ or imported as worker.
_tq_registry[("worker", "run_for_root")] = run_for_root

# ── Poll ──────────────────────────────────────────────────────────────────────

_runs = 0


def _stop_fn(executed: int) -> None:
    if executed >= MAX_RUNS:
        log.info("reached max_runs=%d, exiting", MAX_RUNS)
        sys.exit(0)


if __name__ == "__main__":
    log.info("worker starting — queue: %s", QUEUE_URL)
    tq = TaskQueue(QUEUE_URL)
    tq.poll(lease_seconds=LEASE_SECONDS, verbose=False, tally=False, stop_fn=_stop_fn)
