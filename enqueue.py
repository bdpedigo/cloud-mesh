"""
enqueue.py — Populate the task queue with root IDs to process.

Run locally BEFORE spinning up the GKE cluster (or while it's running).

Usage
-----
# From a CSV with a column of root IDs:
    python enqueue.py --csv path/to/roots.csv --col pt_root_id \\
        --datastack minnie65_phase3_v1

# From a plain text file (one root ID per line):
    python enqueue.py --txt path/to/roots.txt \\
        --datastack minnie65_phase3_v1

# Quick inline list (space-separated):
    python enqueue.py --ids 864691135494786192 864691135851320007 \\
        --datastack minnie65_phase3_v1
"""

import argparse
import logging
import sys
from functools import partial
from pathlib import Path

import tomllib
from taskqueue import TaskQueue

from worker import run_for_root

# ── Config ────────────────────────────────────────────────────────────────────

_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as _f:
    _job = tomllib.load(_f)["job"]

QUEUE_URL = _job["queue_url"]

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)s — %(message)s", stream=sys.stdout
)
log = logging.getLogger("enqueue")

# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Enqueue root IDs for cloud-mesh workers.")
    p.add_argument(
        "--datastack",
        required=True,
        help="CAVE datastack name, e.g. minnie65_phase3_v1",
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", metavar="FILE", help="CSV file containing root IDs")
    src.add_argument(
        "--txt", metavar="FILE", help="Text file with one root ID per line"
    )
    src.add_argument(
        "--ids",
        nargs="+",
        type=int,
        metavar="ROOT_ID",
        help="Root IDs passed directly on the command line",
    )

    p.add_argument(
        "--col",
        default="pt_root_id",
        help="Column name for root IDs when using --csv (default: pt_root_id)",
    )
    p.add_argument(
        "--queue-url", default=QUEUE_URL, help="Override the queue URL from config.toml"
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    if args.csv:
        import pandas as pd

        df = pd.read_csv(args.csv)
        root_ids = df[args.col].dropna().astype(int).unique().tolist()
    elif args.txt:
        lines = Path(args.txt).read_text().splitlines()
        root_ids = [int(line.strip()) for line in lines if line.strip()]
    else:
        root_ids = args.ids

    log.info(
        "enqueueing %d tasks → datastack=%s queue=%s",
        len(root_ids),
        args.datastack,
        args.queue_url,
    )

    tasks = [
        partial(run_for_root, root_id, args.datastack)
        for root_id in root_ids
    ]

    tq = TaskQueue(args.queue_url)
    tq.insert(tasks)
    log.info("done — %d tasks inserted", len(tasks))


if __name__ == "__main__":
    main()
