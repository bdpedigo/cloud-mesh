"""
enqueue_pubsub.py — Populate the Pub/Sub task queue with HKS jobs.

Two modes, selected with --mode:

1. whole-cell: each root ID becomes one full-mesh condensed_hks_pipeline task
   that computes HKS over the entire cell mesh.

       python enqueue_pubsub.py --mode whole-cell --ids 864691135436446706
       python enqueue_pubsub.py --mode whole-cell --ids path/to/roots.txt

2. synapse: for each root ID (the "main cell"), one chunked_hks_pipeline task
   is enqueued per synaptic partner — targeted at the synapse center with a
   fixed query radius. Partners are covered on both sides: cells that are
   pre-synaptic to the main cell, and cells that are post-synaptic to it.
   The main cell's own HKS is handled separately via whole-cell mode, so the
   two modes together cover every cell touching each synapse.

       python enqueue_pubsub.py --mode synapse --ids 864691135436446706
       python enqueue_pubsub.py --mode synapse --ids path/to/roots.txt

   Synapse tables are resolved in this order for each root ID:
     1. --synapse-file (explicit path; single root ID only)
     2. --synapse-table-dir/<root_id>_syn.parquet
     3. Live query from CAVE

--ids accepts either a single integer root ID or a path to a plain-text file
with one root ID per line (lines starting with # are ignored).

The synapse dataframe must have ``id``, ``pre_pt_root_id``, ``post_pt_root_id``,
and ``ctr_pt_position`` columns. ``ctr_pt_position`` is in voxel units and is
converted to nm at enqueue time. Voxel size is auto-detected from the CAVE
datastack via ``client.info.viewer_resolution()``; pass ``--voxel-size`` to
override (e.g. ``--voxel-size 15 15 50``).
"""

import argparse
import logging
import os
import sys
import tomllib
from functools import partial
from pathlib import Path

import numpy as np

from PSTaskQueue import PSTaskQueue
from run_hks import run_hks
from slack_utils import post_slack

# Config

_config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml"))

with open(_config_path, "rb") as _f:
    _cfg = tomllib.load(_f)

_PUBSUB_HELP = (
    "config.toml has no [pubsub] section. Pub/Sub mode needs one: copy the "
    "[pubsub] block from config-template.toml and fill in your project's "
    "topic and subscription paths."
)

if "pubsub" not in _cfg:
    raise SystemExit(_PUBSUB_HELP)

QUEUE_URL = _cfg["pubsub"]["topic"]

# Optional local fileshare directory holding per-cell synapse tables named
# ``{root_id}_syn.parquet``. When unset (or a table is missing) the synapse
# table is queried from CAVE instead.
DEFAULT_SYNAPSE_TABLE_DIR = _cfg.get("paths", {}).get("synapse_table_dir")

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)s — %(message)s", stream=sys.stdout
)
log = logging.getLogger("enqueue")


# Helpers

def _describe_source(args):
    """Return a short human-readable description of the enqueue source."""
    if Path(args.ids).is_file():
        return f"file {args.ids} (mode={args.mode})"
    return f"--ids {args.ids} (mode={args.mode})"


def _coerce_position(raw):
    """Normalize a ctr_pt_position cell to a length-3 float array (voxel units).

    Parquet typically preserves it as a list/ndarray. CSV roundtripping turns
    it into a string like ``"[12345 67890 1234]"`` or ``"[12345, 67890, 1234]"``;
    handle both.
    """
    if isinstance(raw, str):
        cleaned = raw.strip().strip("[]()")
        sep = "," if "," in cleaned else None
        arr = np.fromstring(cleaned, sep=sep) if sep else np.fromstring(cleaned, sep=" ")
    else:
        arr = np.asarray(list(raw), dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"ctr_pt_position must be length-3, got shape {arr.shape}: {raw!r}")
    return arr.astype(float)


def _build_synapse_tasks(syn_df, datastack, voxel_size, radius_nm, root_id_col="pre_pt_root_id", exclude_root_id=None):
    """Build one run_hks partial per synapse row.

    Skips rows whose ``root_id_col`` is null or 0 (orphan / unsegmented), and
    rows where the target root ID equals ``exclude_root_id`` (the main cell,
    whose HKS is handled by whole-cell mode).
    Pass ``root_id_col="post_pt_root_id"`` to enqueue tasks for the post side.
    """
    voxel_size = np.asarray(voxel_size, dtype=float)
    if voxel_size.shape != (3,):
        raise ValueError(f"voxel_size must be length-3, got {voxel_size!r}")
    radius_nm = float(radius_nm)
    exclude_root_id = int(exclude_root_id) if exclude_root_id is not None else None

    required = {"id", root_id_col, "ctr_pt_position"}
    missing = required - set(syn_df.columns)
    if missing:
        raise KeyError(f"synapse dataframe is missing required columns: {sorted(missing)}")

    tasks = []
    skipped = 0
    for _, row in syn_df.iterrows():
        pre_root_raw = row[root_id_col]
        try:
            pre_root = int(pre_root_raw)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if pre_root == 0:
            skipped += 1
            continue
        if exclude_root_id is not None and pre_root == exclude_root_id:
            skipped += 1
            continue

        try:
            ctr_voxel = _coerce_position(row["ctr_pt_position"])
        except Exception as e:
            log.warning("skipping synapse id=%s: bad ctr_pt_position (%s)", row.get("id"), e)
            skipped += 1
            continue

        query_point_nm = (ctr_voxel * voxel_size).tolist()
        synapse_id = int(row["id"])

        tasks.append(
            partial(
                run_hks,
                pre_root,
                datastack,
                synapse_id=synapse_id,
                query_point_nm=query_point_nm,
                query_radius_nm=radius_nm,
            )
        )
    if skipped:
        log.info("skipped %d synapse row(s) with missing/zero pre_pt_root_id or bad position", skipped)
    return tasks


# CLI

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enqueue HKS tasks for cloud-mesh workers.")
    p.add_argument(
        "--mode",
        required=True,
        choices=["whole-cell", "synapse"],
        help=(
            "whole-cell: one full-mesh HKS task per root ID. "
            "synapse: one chunked HKS task per synaptic partner of each root ID."
        ),
    )
    p.add_argument(
        "--ids",
        required=True,
        metavar="IDS_OR_FILE",
        help=(
            "Either a path to a text file with one root ID per line, "
            "or a single root ID passed directly."
        ),
    )
    p.add_argument(
        "--datastack",
        required=True,
        metavar="STRING",
        help="CAVE datastack name, e.g. minnie65_phase3_v1",
    )
    p.add_argument(
        "--queue-url",
        default=QUEUE_URL,
        help="Override the queue URL from config.toml",
    )

    # synapse mode options
    p.add_argument(
        "--synapse-file",
        metavar="FILE",
        default=None,
        help=(
            "Saved synapse dataframe (.parquet/.csv/.feather/.pkl) for a single "
            "root ID. Only valid with --mode synapse and a single --ids value. "
            "If omitted, synapse tables are loaded from --synapse-table-dir or "
            "queried from CAVE."
        ),
    )
    p.add_argument(
        "--synapse-table-dir",
        metavar="DIR",
        default=DEFAULT_SYNAPSE_TABLE_DIR,
        help=(
            "Directory containing per-cell synapse tables named "
            "<root_id>_syn.parquet. Used with --mode synapse when no "
            "--synapse-file is given. Falls back to CAVE if no file found. "
            f"(default from config [paths].synapse_table_dir: {DEFAULT_SYNAPSE_TABLE_DIR})"
        ),
    )
    p.add_argument(
        "--query-radius-nm",
        type=float,
        default=1000.0,
        help="Radius (nm) around each synapse center for chunked HKS (default: 1000).",
    )
    p.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help=(
            "Voxel size in nm for ctr_pt_position->nm conversion. If omitted, "
            "auto-detected from the CAVE datastack via "
            "client.info.viewer_resolution(), which reports the segmentation "
            "MIP-0 scale that ctr_pt_position is stored in."
        ),
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N root IDs from the input file (in file order). Default: all.",
    )
    p.add_argument(
        "--cap",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Cap the total number of tasks pushed to the queue. "
            "Useful for smoke tests. Default: no cap."
        ),
    )

    args = p.parse_args()

    if args.cap is not None and args.cap < 0:
        p.error("--cap must be a non-negative integer")
    if args.max_cells is not None and args.max_cells < 1:
        p.error("--max-cells must be a positive integer")
    if args.synapse_file is not None and args.mode != "synapse":
        p.error("--synapse-file requires --mode synapse")

    return args


# Main

def _resolve_root_ids(ids_arg, max_cells=None):
    """Parse --ids: either a path to a txt file or a single integer."""
    ids_path = Path(ids_arg)
    if ids_path.is_file():
        root_ids = [
            int(line.strip())
            for line in ids_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        root_ids = list(dict.fromkeys(root_ids))  # deduplicate, preserve order
        if max_cells is not None:
            root_ids = root_ids[:max_cells]
        return root_ids
    else:
        return [int(ids_arg)]


def _resolve_voxel_size(args, client=None):
    """Return voxel_size tuple, auto-detecting from CAVE if needed."""
    if args.voxel_size is not None:
        return tuple(args.voxel_size)
    try:
        if client is None:
            from caveclient import CAVEclient
            client = CAVEclient(datastack_name=args.datastack)
        voxel_size = tuple(float(v) for v in client.info.viewer_resolution())
        log.info("auto-detected voxel_size=%s nm (datastack=%s)", voxel_size, args.datastack)
        return voxel_size
    except Exception as exc:  # noqa: BLE001
        voxel_size = (15.0, 15.0, 50.0)
        log.warning("could not pull viewer_resolution() (%s); falling back to %s", exc, voxel_size)
        return voxel_size


def _load_synapse_df(root_id, args, client=None):
    """Load synapse table for root_id. Returns (syn_df, source_description)."""
    import pandas as pd
    from generate_queue import get_post_synapse_df

    # Explicit --synapse-file takes priority (single-cell only).
    if args.synapse_file is not None:
        log.info("root_id=%s: loading synapse table from --synapse-file %s", root_id, args.synapse_file)
        return get_post_synapse_df(root_id, file_path=args.synapse_file), args.synapse_file

    # Check synapse-table-dir for a pre-saved parquet.
    if args.synapse_table_dir:
        syn_path = Path(args.synapse_table_dir) / f"{root_id}_syn.parquet"
        if syn_path.exists():
            log.info("root_id=%s: loading synapse table from %s", root_id, syn_path)
            return pd.read_parquet(syn_path), str(syn_path)

    # Fall back to CAVE.
    log.info("root_id=%s: no local synapse table found, querying CAVE", root_id)
    if client is None:
        from caveclient import CAVEclient
        client = CAVEclient(datastack_name=args.datastack)
    return get_post_synapse_df(root_id, client=client), "CAVE"


def main() -> None:
    args = parse_args()
    root_ids = _resolve_root_ids(args.ids, max_cells=args.max_cells)

    if args.mode == "synapse":
        import pandas as pd  # noqa: F401

        voxel_size = _resolve_voxel_size(args)
        tasks = []
        n_failed = 0

        for root_id in root_ids:
            try:
                syn_df, source = _load_synapse_df(root_id, args)
            except Exception as exc:
                log.warning("root_id=%s: failed to load synapse table (%s), skipping", root_id, exc)
                n_failed += 1
                continue

            cell_tasks_pre = _build_synapse_tasks(
                syn_df,
                datastack=args.datastack,
                voxel_size=voxel_size,
                radius_nm=args.query_radius_nm,
                root_id_col="pre_pt_root_id",
                exclude_root_id=root_id,
            )
            cell_tasks_post = _build_synapse_tasks(
                syn_df,
                datastack=args.datastack,
                voxel_size=voxel_size,
                radius_nm=args.query_radius_nm,
                root_id_col="post_pt_root_id",
                exclude_root_id=root_id,
            )
            cell_tasks = cell_tasks_pre + cell_tasks_post
            tasks.extend(cell_tasks)
            log.info(
                "root_id=%s: %d tasks from %d synapses (%d pre-side, %d post-side) [source: %s]",
                root_id, len(cell_tasks), len(syn_df),
                len(cell_tasks_pre), len(cell_tasks_post), source,
            )

        log.info(
            "synapse mode: %d/%d cells loaded, %d total tasks "
            "(radius=%.1f nm, voxel_size=%s)",
            len(root_ids) - n_failed, len(root_ids), len(tasks),
            args.query_radius_nm, tuple(voxel_size),
        )

    else:
        # whole-cell mode: one full-mesh task per root ID.
        log.info(
            "whole-cell mode: enqueueing %d full-mesh tasks → datastack=%s queue=%s",
            len(root_ids), args.datastack, args.queue_url,
        )
        tasks = [partial(run_hks, root_id, args.datastack) for root_id in root_ids]

    if args.cap is not None and len(tasks) > args.cap:
        log.info("capping %d tasks down to %d (--cap)", len(tasks), args.cap)
        tasks = tasks[: args.cap]

    tq = PSTaskQueue(args.queue_url)
    tq.insert(tasks)
    log.info("done — %d tasks inserted", len(tasks))

    post_slack(
        f"HKS {args.datastack} - :inbox_tray: enqueued {len(tasks)} {args.mode} task(s) "
        f"from {_describe_source(args)}"
    )


if __name__ == "__main__":
    main()
