"""
purge_queue.py — Drain every pending message from the Pub/Sub subscription.

Pulls and acknowledges in batches until the subscription is empty, then
reports the count. Useful after a bad enqueue, or before re-running a batch
with different parameters.

    uv run purge_queue.py                 # drain the configured subscription
    uv run purge_queue.py --dry-run       # just report the pending count

Project and subscription are read from ``config.toml`` (``[cluster].project``
and ``[pubsub].subscription``); override either with the matching flag.
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

from google.cloud import pubsub_v1

_config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml"))

with open(_config_path, "rb") as _f:
    _cfg = tomllib.load(_f)

DEFAULT_PROJECT = _cfg["cluster"]["project"]

_PUBSUB_HELP = (
    "config.toml has no [pubsub] section. Pub/Sub mode needs one: copy the "
    "[pubsub] block from config-template.toml and fill in your project's "
    "topic and subscription paths."
)

if "pubsub" not in _cfg:
    raise SystemExit(_PUBSUB_HELP)

# config stores the fully-qualified subscription path; we want the bare name.
DEFAULT_SUBSCRIPTION = _cfg["pubsub"]["subscription"].rsplit("/", 1)[-1]

BATCH_SIZE = 1000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drain all pending messages from the Pub/Sub subscription."
    )
    p.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=f"GCP project ID (default from config.toml: {DEFAULT_PROJECT}).",
    )
    p.add_argument(
        "--subscription",
        default=DEFAULT_SUBSCRIPTION,
        help=f"Subscription name (default from config.toml: {DEFAULT_SUBSCRIPTION}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report how many messages are pending without acknowledging any.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(args.project, args.subscription)

    if args.dry_run:
        response = subscriber.pull(subscription=sub_path, max_messages=BATCH_SIZE)
        print(f"{len(response.received_messages)} message(s) pending (dry run, nothing acked)")
        return

    total = 0
    while True:
        response = subscriber.pull(subscription=sub_path, max_messages=BATCH_SIZE)
        if not response.received_messages:
            break
        subscriber.acknowledge(
            subscription=sub_path,
            ack_ids=[m.ack_id for m in response.received_messages],
        )
        total += len(response.received_messages)
        print(f"acked {total} message(s)...", file=sys.stderr)

    print(f"drained {total} message(s) from {sub_path}")


if __name__ == "__main__":
    main()
