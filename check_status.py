"""
check_status.py — Inspect which HKS outputs exist in GCS.

Mirrors the --mode / --ids interface of enqueue_pubsub.py.

Modes (--mode):

  whole-cell   For each root ID, check whether
               gs://{bucket}/{datastack}/features/{root_id}.npz exists.

                   python check_status.py --mode whole-cell --ids 864691135436446706
                   python check_status.py --mode whole-cell --ids path/roots.txt

  synapse      For each root ID (the "main cell"), check whether the chunked
               HKS file exists for every synaptic partner on both sides
               (pre-side partners where main cell is post, post-side partners
               where main cell is pre). The main cell itself is excluded.
               Synapse tables are resolved from --synapse-file,
               --synapse-table-dir, or CAVE (in that order).

                   python check_status.py --mode synapse --ids 864691135436446706
                   python check_status.py --mode synapse --ids path/roots.txt
                   python check_status.py --mode synapse --ids 864691135436446706 \\
                       --synapse-file path/synapses.parquet

  summary      Like synapse mode but aggregates counts per root ID and
               optionally pushes results to a Google Sheet.

                   python check_status.py --mode summary --ids path/roots.txt
                   python check_status.py --mode summary --ids path/roots.txt \\
                       --update-gsheet

--ids accepts either a single integer root ID or a path to a plain-text file
with one root ID per line (lines starting with # are ignored).

All modes save a CSV report to --out-dir (default: ./status_reports/).
"""

import argparse
import logging
import os
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd
from caveclient import CAVEclient
from google.cloud import storage

from generate_queue import get_all_synapse_df


# Config
_config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml"))

with open(_config_path, "rb") as _f:
    _cfg = tomllib.load(_f)

OUTPUT_BUCKET_URI = _cfg["job"]["output_bucket"]
# Falls back to [monitor].datastack; None means --datastack must be passed.
DEFAULT_DATASTACK = _cfg.get("monitor", {}).get("datastack")

_paths = _cfg.get("paths", {})

# Default destination for CSV reports. Overridable via --out-dir.
DEFAULT_REPORT_DIR = Path(_paths.get("report_dir") or (Path(__file__).parent / "status_reports"))

# Per-cell synapse tables (``{root_id}_syn.parquet``) and presynaptic mesh
# files (``{root_id}_mesh.h5``) on the local fileshare. Both optional: when
# ``synapse_table_dir`` is unset, synapse tables are pulled from CAVE; when
# ``mesh_dir`` is unset, mesh-size annotation requires an explicit --mesh-dir.
DEFAULT_SYNAPSE_TABLE_DIR = (
    Path(_paths["synapse_table_dir"]) if _paths.get("synapse_table_dir") else None
)
DEFAULT_MESH_DIR = Path(_paths["mesh_dir"]) if _paths.get("mesh_dir") else None

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)s — %(message)s", stream=sys.stdout
)
log = logging.getLogger("check_status")


# GCS helpers

def bucket_name_from_uri(uri: str) -> str:
    if uri.startswith("gs://"):
        uri = uri[len("gs://"):]
    return uri.split("/", 1)[0]


def list_existing_outputs(bucket_name: str, datastack: str) -> dict[str, int]:
    """Return a ``{name: size_bytes}`` mapping for every ``.npz`` object under
    ``{datastack}/features/``. One bulk LIST is much cheaper than per-task
    ``blob.exists()``.
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = f"{datastack}/features/"
    sizes = {
        blob.name: int(blob.size or 0)
        for blob in bucket.list_blobs(prefix=prefix)
        if blob.name.endswith(".npz")
    }
    log.info("listed %d existing .npz objects under gs://%s/%s",
             len(sizes), bucket_name, prefix)
    return sizes


# Public API

def check_full_mesh_status(
    root_ids: Iterable[int],
    datastack: str = DEFAULT_DATASTACK,
    bucket_uri: str = OUTPUT_BUCKET_URI,
) -> pd.DataFrame:
    """For each root_id, check whether ``features/{root_id}.npz`` exists.

    Returns a DataFrame with columns: ``root_id``, ``path``, ``exists``.
    """
    bucket_name = bucket_name_from_uri(bucket_uri)
    existing = list_existing_outputs(bucket_name, datastack)

    rows = []
    for rid in root_ids:
        rid = int(rid)
        name = f"{datastack}/features/{rid}.npz"
        rows.append(
            {
                "root_id": rid,
                "path": f"gs://{bucket_name}/{name}",
                "exists": name in existing,
                "size_bytes": existing.get(name),
            }
        )
    return pd.DataFrame(rows)


def check_synapse_status(
    root_id: int,
    datastack: str = DEFAULT_DATASTACK,
    synapse_file: Optional[str] = None,
    bucket_uri: str = OUTPUT_BUCKET_URI,
    synapse_table_dir: Optional[Path] = DEFAULT_SYNAPSE_TABLE_DIR,
) -> pd.DataFrame:
    """For each synaptic partner of ``root_id``, check whether the chunked
    HKS file exists on both sides (pre-side partners where main cell is post,
    post-side partners where main cell is pre). The main cell itself is excluded.

    Synapse table resolution order: ``synapse_file`` arg → ``synapse_table_dir``
    → CAVE live query.

    Returns a DataFrame with columns: ``root_id``, ``partner_root_id``,
    ``main_side``, ``synapse_id``, ``path``, ``exists``, ``size_bytes``.
    """
    root_id = int(root_id)

    # Resolve synapse table.
    if synapse_file is not None:
        log.info("root_id=%s: loading synapse table from %s", root_id, synapse_file)
        syn_df = pd.read_parquet(synapse_file) if str(synapse_file).endswith(".parquet") \
            else pd.read_csv(synapse_file)
        source = synapse_file
    else:
        local_path = (
            Path(synapse_table_dir) / f"{root_id}_syn.parquet"
            if synapse_table_dir is not None
            else None
        )
        if local_path is not None and local_path.exists():
            log.info("root_id=%s: loading synapse table from %s", root_id, local_path)
            syn_df = pd.read_parquet(local_path)
            source = local_path
        else:
            log.info("root_id=%s: no local synapse table found, querying CAVE", root_id)
            client = CAVEclient(datastack_name=datastack)
            syn_df = get_all_synapse_df(root_id, client=client)
            source = "CAVE"

    log.info("root_id=%s: loaded %d synapse rows [source: %s]", root_id, len(syn_df), source)

    bucket_name = bucket_name_from_uri(bucket_uri)
    existing = list_existing_outputs(bucket_name, datastack)

    rows = []
    for _, r in syn_df.iterrows():
        try:
            pre = int(r["pre_pt_root_id"]) if r["pre_pt_root_id"] is not None else 0
            post = int(r["post_pt_root_id"]) if r["post_pt_root_id"] is not None else 0
            sid = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue

        is_pre = (pre == root_id)
        is_post = (post == root_id)
        if is_pre and is_post:
            continue  # autapse — main cell is its own partner, skip
        elif is_post:
            main_side, partner = "post", pre
        elif is_pre:
            main_side, partner = "pre", post
        else:
            continue  # row doesn't involve root_id at all

        if partner == 0:
            rows.append({
                "root_id": root_id, "partner_root_id": 0, "main_side": main_side,
                "synapse_id": sid, "path": None, "exists": False, "size_bytes": None,
            })
            continue

        name = f"{datastack}/features/synapse/{partner}_syn_{sid}.npz"
        rows.append({
            "root_id": root_id,
            "partner_root_id": partner,
            "main_side": main_side,
            "synapse_id": sid,
            "path": f"gs://{bucket_name}/{name}",
            "exists": name in existing,
            "size_bytes": existing.get(name),
        })
    return pd.DataFrame(rows)


# Local mesh-size augmentation

def _mesh_file_size(root_id: int, mesh_dir: Path) -> Optional[int]:
    """Return the size in bytes of ``{mesh_dir}/{root_id}_mesh.h5`` on disk,
    or ``None`` if the file is missing / unreadable.
    """
    if root_id is None:
        return None
    try:
        rid = int(root_id)
    except (TypeError, ValueError):
        return None
    if rid == 0:
        return None
    path = Path(mesh_dir) / f"{rid}_mesh.h5"
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def add_mesh_size_column(
    report_csv: os.PathLike,
    mesh_dir: os.PathLike = DEFAULT_MESH_DIR,
    root_id_col: str = "pre_pt_root_id",
    size_col: str = "mesh_size_bytes",
    out_path: Optional[os.PathLike] = None,
) -> pd.DataFrame:
    """Read a status CSV, add a column with the presynaptic-mesh file size
    on disk, and save the augmented CSV.

    Parameters
    ----------
    report_csv : path
        Existing status report (e.g. ``status_syn_<root>_<ts>.csv``).
    mesh_dir : path
        Folder containing per-neuron mesh files named ``{root_id}_mesh.h5``.
    root_id_col : str
        Column in the CSV holding the (presynaptic) root id to look up.
        Defaults to ``"pre_pt_root_id"`` to match :func:`check_synapse_status`
        output.
    size_col : str
        Name of the new column to add. Defaults to ``"mesh_size_bytes"``.
    out_path : path, optional
        Where to write the augmented CSV. Defaults to overwriting
        ``report_csv`` in place.

    Returns
    -------
    pd.DataFrame
        The augmented dataframe (also written to disk).
    """
    report_csv = Path(report_csv)
    mesh_dir = Path(mesh_dir)
    out_path = Path(out_path) if out_path is not None else report_csv

    df = pd.read_csv(report_csv)
    if root_id_col not in df.columns:
        raise KeyError(
            f"{report_csv.name} has no column {root_id_col!r}; "
            f"available columns: {list(df.columns)}"
        )

    cache: dict[int, Optional[int]] = {}

    def _lookup(rid):
        try:
            key = int(rid) if pd.notna(rid) else None
        except (TypeError, ValueError):
            key = None
        if key is None:
            return None
        if key not in cache:
            cache[key] = _mesh_file_size(key, mesh_dir)
        return cache[key]

    df[size_col] = df[root_id_col].map(_lookup)

    n_total = df[size_col].notna().sum() if size_col in df else 0
    n_unique = sum(1 for v in cache.values() if v is not None)
    log.info(
        "annotated %d / %d rows with %s (from %d unique meshes found in %s)",
        int(n_total), len(df), size_col, n_unique, mesh_dir,
    )

    df.to_csv(out_path, index=False)
    log.info("wrote augmented report to %s", out_path)
    return df


# Per-root summary across many root IDs

def check_one_root_synapses(
    root_id: int,
    *,
    datastack: str,
    existing: dict,
    bucket_name: str,
    client=None,
    synapse_table_dir: Optional[Path] = DEFAULT_SYNAPSE_TABLE_DIR,
) -> pd.DataFrame:
    """Return a per-synapse status frame for one main root id.

    For each synapse touching ``root_id`` (queried from CAVE in BOTH
    directions via :func:`generate_queue.get_all_synapse_df`), determines
    which side is the *partner* (the one that's not ``root_id``), and
    checks whether the chunked HKS file
    ``{datastack}/features/synapse/{partner_root}_syn_{synapse_id}.npz``
    exists in the bucket listing already loaded into ``existing``.

    Columns:
      - ``main_root_id`` (= root_id)
      - ``pre_pt_root_id``, ``post_pt_root_id``, ``synapse_id``
      - ``main_side`` ∈ {"pre", "post", "autapse"}
      - ``partner_root_id`` (0 / NaN -> orphan)
      - ``path``
      - ``exists``, ``size_bytes``
    """
    synapse_file = (
        Path(synapse_table_dir) / f"{root_id}_syn.parquet"
        if synapse_table_dir is not None
        else None
    )
    if synapse_file is not None and synapse_file.exists():
        log.info("loading synaptic partners of root_id=%s from %s", root_id, synapse_file)
        syn_df = pd.read_parquet(synapse_file)
    else:
        log.info("querying CAVE for synaptic partners of root_id=%s", root_id)
        syn_df = get_all_synapse_df(root_id, client=client)

    rows = []
    for _, r in syn_df.iterrows():
        try:
            pre = int(r["pre_pt_root_id"]) if r["pre_pt_root_id"] is not None else 0
        except (TypeError, ValueError):
            pre = 0
        try:
            post = int(r["post_pt_root_id"]) if r["post_pt_root_id"] is not None else 0
        except (TypeError, ValueError):
            post = 0
        try:
            sid = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue

        is_pre = (pre == root_id)
        is_post = (post == root_id)
        if is_pre and is_post:
            main_side = "autapse"
            partner = root_id
        elif is_post:
            main_side = "post"
            partner = pre
        elif is_pre:
            main_side = "pre"
            partner = post
        else:
            continue

        if partner == 0:
            name = None
            exists = False
            size = None
        else:
            name = f"{datastack}/features/synapse/{partner}_syn_{sid}.npz"
            exists = name in existing
            size = existing.get(name)

        rows.append(
            {
                "main_root_id": int(root_id),
                "pre_pt_root_id": pre,
                "post_pt_root_id": post,
                "synapse_id": sid,
                "main_side": main_side,
                "partner_root_id": int(partner),
                "path": (f"gs://{bucket_name}/{name}") if name else None,
                "exists": exists,
                "size_bytes": size,
            }
        )
    return pd.DataFrame(rows)


def check_roots_summary(
    root_ids,
    *,
    datastack: str = DEFAULT_DATASTACK,
    bucket_uri: str = OUTPUT_BUCKET_URI,
    out_dir: Path = DEFAULT_REPORT_DIR,
    label: Optional[str] = None,
    write_per_synapse: bool = False,
    update_gsheet: bool = False,
    sheet_title: str = "HKS Status Dashboard",
    synapse_table_dir: Optional[Path] = DEFAULT_SYNAPSE_TABLE_DIR,
):
    """Build a high-level summary across many root IDs.

    For each ``root_id``:
      1. Pull all synapses where it's pre OR post (CAVE union).
      2. Check whether the chunked HKS file exists for the partner side.
      3. Aggregate counts (total / present / missing / orphan).

    Always writes a per-root summary CSV. If ``write_per_synapse=True``,
    also writes the full per-synapse details for each root_id.
    If ``update_gsheet=True``, pushes the summary to a persistent Google Sheet.

    Returns
    -------
    (summary_df, per_synapse_df)
    """
    bucket_name = bucket_name_from_uri(bucket_uri)
    log.info("listing existing chunked HKS outputs once for all roots...")
    existing = list_existing_outputs(bucket_name, datastack)

    client = CAVEclient(datastack_name=datastack)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or "roots_summary"
    _now = datetime.now(ZoneInfo("America/New_York"))
    ts      = _now.strftime("%m/%d/%Y %H:%M:%S %Z")  # human-readable, for sheet column
    ts_file = _now.strftime("%Y%m%dT%H%M%S_ET")      # filename-safe
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)

    summary_rows = []
    per_syn_frames = []

    for rid in root_ids:
        rid = int(rid)
        try:
            df = check_one_root_synapses(
                rid,
                datastack=datastack,
                existing=existing,
                bucket_name=bucket_name,
                client=client,
                synapse_table_dir=synapse_table_dir,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("root_id=%s: query/check failed (%s: %s); skipping.",
                        rid, type(exc).__name__, exc)
            continue

        n_total = len(df)
        if n_total == 0:
            log.info("root_id=%s: no synapses found.", rid)
            summary_rows.append({
                "root_id": str(rid), "n_synapses_total": 0,
                "n_pre_synapses": 0, "n_post_synapses": 0, "n_autapses": 0,
                "n_orphan_partners": 0, "n_attempted": 0, "n_present": 0,
                "n_missing": 0, "fraction_present": float("nan"),
                "last_updated": ts,
            })
            continue

        n_pre = int((df["main_side"] == "pre").sum())
        n_post = int((df["main_side"] == "post").sum())
        n_autapse = int((df["main_side"] == "autapse").sum())
        n_orphan = int((df["partner_root_id"] == 0).sum())
        non_orphan = df[df["partner_root_id"] != 0]
        n_attempted = len(non_orphan)
        n_present = int(non_orphan["exists"].sum())
        n_missing = n_attempted - n_present
        frac = (n_present / n_attempted) if n_attempted else float("nan")

        log.info(
            "root_id=%s: %d/%d non-orphan synapses present (%.1f%%) — "
            "pre=%d post=%d autapse=%d orphan=%d",
            rid, n_present, n_attempted, 100.0 * frac if n_attempted else 0.0,
            n_pre, n_post, n_autapse, n_orphan,
        )

        summary_rows.append({
            "root_id": str(rid),
            "n_synapses_total": n_total,
            "n_pre_synapses": n_pre,
            "n_post_synapses": n_post,
            "n_autapses": n_autapse,
            "n_orphan_partners": n_orphan,
            "n_attempted": n_attempted,
            "n_present": n_present,
            "n_missing": n_missing,
            "fraction_present": frac,
            "last_updated": ts,
        })

        if write_per_synapse:
            per_root_path = out_dir / f"status_{safe_label}_{rid}_{ts_file}.csv"
            df.to_csv(per_root_path, index=False)
            log.info("  wrote per-synapse detail to %s", per_root_path)

        per_syn_frames.append(df)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"status_{safe_label}_{ts_file}.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("wrote summary (%d roots) to %s", len(summary_df), summary_path)

    if update_gsheet:
        _push_summary_to_gsheet(
            summary_df=summary_df,
            sheet_title=sheet_title,
        )

    per_syn_df = (
        pd.concat(per_syn_frames, ignore_index=True)
        if per_syn_frames else pd.DataFrame()
    )
    return summary_df, per_syn_df


def _push_summary_to_gsheet(
    summary_df: pd.DataFrame,
    sheet_title: str,
) -> None:
    """Push summary_df to the persistent Google Sheet, creating it on first run
    and updating in-place on subsequent runs. The spreadsheet ID is cached in
    ``status_report_state.json`` next to this file.

    Uses the Sheets API only (no Drive API required). On first run the sheet is
    owned by the OAuth user — accessible directly in Google Drive.
    """
    try:
        import gsheet_utils
    except ImportError as exc:
        raise SystemExit(
            "--update-gsheet needs the optional 'gsheet' dependencies. "
            f"Install them with:  uv sync --extra gsheet   ({exc})"
        )

    state = gsheet_utils.load_state()
    gc = gsheet_utils._gspread_client()

    spreadsheet_id = state.get("spreadsheet_id")
    if not spreadsheet_id:
        log.info("No cached spreadsheet; creating '%s'...", sheet_title)
        sh = gc.create(sheet_title)
        spreadsheet_id = sh.id
        state["spreadsheet_id"] = spreadsheet_id
        gsheet_utils.save_state(state)
        log.info("Created spreadsheet id=%s", spreadsheet_id)
    else:
        log.info("Updating existing spreadsheet id=%s", spreadsheet_id)

    gsheet_utils.update_worksheet_with_df(
        spreadsheet_id,
        sheet_name="Summary",
        df=summary_df,
        gc=gc,
    )
    url = gsheet_utils.spreadsheet_url(spreadsheet_id)
    log.info("Google Sheet updated: %s", url)


# CSV writer

def _save_report(df: pd.DataFrame, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now(ZoneInfo("America/New_York")).strftime("%Y%m%dT%H%M%S_ET")
    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)
    out_path = out_dir / f"status_{safe_label}_{ts_file}.csv"
    df.to_csv(out_path, index=False)
    return out_path


# CLI

def _resolve_root_ids(ids_arg, max_cells=None) -> list[int]:
    """Parse --ids: either a path to a txt file or a single integer."""
    ids_path = Path(ids_arg)
    if ids_path.is_file():
        root_ids = [
            int(line.strip())
            for line in ids_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        root_ids = list(dict.fromkeys(root_ids))
        if max_cells is not None:
            root_ids = root_ids[:max_cells]
        return root_ids
    return [int(ids_arg)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Check which HKS outputs exist in the GCS results bucket."
    )
    p.add_argument(
        "--mode",
        required=True,
        choices=["whole-cell", "synapse", "summary", "annotate-mesh-size"],
        help=(
            "whole-cell: check full-mesh HKS outputs per root ID. "
            "synapse: check chunked partner HKS outputs for each root ID. "
            "summary: like synapse but aggregates counts per root ID. "
            "annotate-mesh-size: append local mesh file sizes to an existing report CSV."
        ),
    )
    p.add_argument(
        "--ids",
        metavar="IDS_OR_FILE",
        default=None,
        help=(
            "Either a path to a text file with one root ID per line, "
            "or a single root ID. Required for whole-cell, synapse, and summary modes."
        ),
    )
    p.add_argument(
        "--datastack",
        default=DEFAULT_DATASTACK,
        required=DEFAULT_DATASTACK is None,
        metavar="STRING",
        help=(
            "CAVE datastack name, e.g. minnie65_phase3_v1"
            + (f" (default from config: {DEFAULT_DATASTACK})" if DEFAULT_DATASTACK else "")
        ),
    )
    p.add_argument(
        "--synapse-file",
        metavar="FILE",
        default=None,
        help=(
            "Saved synapse dataframe (.parquet/.csv) for a single root ID. "
            "Only valid with --mode synapse and a single --ids value. "
            "If omitted, resolved from --synapse-table-dir or CAVE."
        ),
    )
    p.add_argument(
        "--synapse-table-dir",
        metavar="DIR",
        default=str(DEFAULT_SYNAPSE_TABLE_DIR) if DEFAULT_SYNAPSE_TABLE_DIR else None,
        help=(
            "Directory containing per-cell synapse tables named "
            "<root_id>_syn.parquet. Falls back to CAVE if not found. "
            f"(default from config [paths].synapse_table_dir: {DEFAULT_SYNAPSE_TABLE_DIR})"
        ),
    )
    p.add_argument(
        "--write-per-synapse",
        action="store_true",
        help="With --mode summary, also save a per-synapse detail CSV for each root ID.",
    )
    p.add_argument(
        "--update-gsheet",
        action="store_true",
        help="With --mode summary, push results to a persistent Google Sheet.",
    )
    p.add_argument(
        "--sheet-title",
        default="HKS Status Dashboard",
        help="Title of the Google Sheet to create or update (default: 'HKS Status Dashboard').",
    )
    p.add_argument(
        "--annotate-report",
        metavar="REPORT_CSV",
        default=None,
        help="Existing status report CSV to annotate. Required for --mode annotate-mesh-size.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        metavar="DIR",
        help=f"Directory for CSV reports (default: {DEFAULT_REPORT_DIR}).",
    )
    p.add_argument(
        "--bucket",
        default=OUTPUT_BUCKET_URI,
        help="Override the GCS output bucket from config.toml.",
    )
    p.add_argument(
        "--mesh-dir",
        type=Path,
        default=DEFAULT_MESH_DIR,
        metavar="DIR",
        help=(
            "Folder of presynaptic mesh files, named <root_id>_mesh.h5. "
            f"(default from config [paths].mesh_dir: {DEFAULT_MESH_DIR})"
        ),
    )
    p.add_argument(
        "--mesh-root-id-col",
        default="partner_root_id",
        help="Column in the status CSV holding the root ID whose mesh size to look up "
             "(default: partner_root_id).",
    )

    args = p.parse_args()

    if args.mode in ("whole-cell", "synapse", "summary") and args.ids is None:
        p.error(f"--ids is required for --mode {args.mode}")
    if args.mode == "annotate-mesh-size" and args.annotate_report is None:
        p.error("--annotate-report is required for --mode annotate-mesh-size")
    if args.synapse_file is not None and args.mode != "synapse":
        p.error("--synapse-file requires --mode synapse")

    return args


def main() -> None:
    args = parse_args()

    syn_table_dir = Path(args.synapse_table_dir) if args.synapse_table_dir else None

    if args.mode == "annotate-mesh-size":
        if args.mesh_dir is None:
            raise SystemExit(
                "annotate-mesh-size needs a mesh folder: pass --mesh-dir or set "
                "config [paths].mesh_dir"
            )
        add_mesh_size_column(
            report_csv=args.annotate_report,
            mesh_dir=args.mesh_dir,
            root_id_col=args.mesh_root_id_col,
        )
        return

    root_ids = _resolve_root_ids(args.ids)

    if args.mode == "summary":
        log.info("summary mode: %d root IDs", len(root_ids))
        ids_label = Path(args.ids).stem if Path(args.ids).is_file() else args.ids
        check_roots_summary(
            root_ids,
            datastack=args.datastack,
            bucket_uri=args.bucket,
            out_dir=args.out_dir,
            label=f"roots_{ids_label}",
            write_per_synapse=args.write_per_synapse,
            update_gsheet=args.update_gsheet,
            sheet_title=args.sheet_title,
            synapse_table_dir=syn_table_dir,
        )
        return

    if args.mode == "synapse":
        frames = []
        for root_id in root_ids:
            df = check_synapse_status(
                root_id,
                datastack=args.datastack,
                synapse_file=args.synapse_file,
                bucket_uri=args.bucket,
                synapse_table_dir=syn_table_dir,
            )
            frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        ids_label = Path(args.ids).stem if Path(args.ids).is_file() else args.ids
        label = f"syn_{ids_label}"
    else:  # whole-cell
        df = check_full_mesh_status(
            root_ids,
            datastack=args.datastack,
            bucket_uri=args.bucket,
        )
        ids_label = Path(args.ids).stem if Path(args.ids).is_file() else args.ids
        label = f"whole_cell_{ids_label}"

    n = len(df)
    n_done = int(df["exists"].sum()) if n else 0
    total_bytes = int(df["size_bytes"].fillna(0).sum()) if n else 0
    log.info(
        "%d / %d outputs present (%.1f%%); total size %.1f MiB",
        n_done, n, (100.0 * n_done / n) if n else 0.0,
        total_bytes / (1024 * 1024),
    )

    out_path = _save_report(df, args.out_dir, label)
    log.info("wrote report to %s", out_path)


if __name__ == "__main__":
    main()
