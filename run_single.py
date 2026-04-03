"""
run_single.py — Run the HKS pipeline for a single root ID, bypassing the task queue.

Useful for smoke-testing your local environment end-to-end before spinning up
a full queue-based deployment. Uses the same config.toml and CloudVolume
secrets as the worker, so the output lands in the same GCS bucket.

Usage
-----
    uv run run_single.py --root-id 864691135307555142 --datastack minnie65_public
"""

import argparse
import logging
import sys

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)s — %(message)s", stream=sys.stdout
)

from worker import run_for_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the HKS pipeline for a single root ID (no task queue)."
    )
    p.add_argument("--root-id", required=True, type=int, help="Root ID to process")
    p.add_argument(
        "--datastack",
        required=True,
        help="CAVE datastack name, e.g. minnie65_public",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.info(
        f"root ID={args.root_id}, datastack={args.datastack} - running HKS pipeline"
    )
    run_for_root(args.root_id, args.datastack)
