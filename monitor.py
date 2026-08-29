#!/usr/bin/env python3
"""
monitor.py - Periodic status reporter for the HKS pipeline.

Designed to run as a Kubernetes CronJob. On each tick:

  1. Reads num_undelivered_messages for the main subscription and the
     dead-letter subscription via Cloud Monitoring.
  2. Counts feature .npz files in gs://{output_bucket}/{datastack}/features/
     (split into full-mesh and per-synapse outputs).
  3. Compares against state stored in
     gs://{output_bucket}/_monitor_state.json
     to derive throughput, ETA, and "what's new" since last tick.
  4. Posts at most one summary line + one "new dead-letters" line +
     one "drained" line per tick to Slack via SLACK_WEBHOOK_URL.

The script is intentionally self-contained: one process, one tick, exit.
The CronJob schedule controls how often it runs.

Environment:
  GCP_PROJECT               (required) GCP project
  CONFIG_PATH               (optional) path to config.toml; default /app/config.toml
  CLOUD_MESH_OUTPUT_BUCKET  (optional) overrides config.toml job.output_bucket
  CLOUD_MESH_DATASTACK      (optional) overrides config.toml monitor.datastack
  SLACK_WEBHOOK_URL         (optional) if missing, log only
  SLACK_CHANNEL             (optional) overrides the webhook's own channel
"""

import json
import logging
import os
import sys
import tomllib
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from google.cloud import monitoring_v3, storage


# Logging

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)s monitor - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("monitor")


# Config

_config_path = Path(os.environ.get("CONFIG_PATH", "/app/config.toml"))
with open(_config_path, "rb") as _f:
    _cfg = tomllib.load(_f)

PROJECT = os.environ.get("GCP_PROJECT", _cfg["cluster"]["project"])
OUTPUT_BUCKET_URI = os.environ.get(
    "CLOUD_MESH_OUTPUT_BUCKET", _cfg["job"]["output_bucket"]
)
_PUBSUB_HELP = (
    "config.toml has no [pubsub] section. Pub/Sub mode needs one: copy the "
    "[pubsub] block from config-template.toml and fill in your project's "
    "topic and subscription paths."
)

if "pubsub" not in _cfg:
    raise SystemExit(_PUBSUB_HELP)

SUBSCRIPTION = _cfg["pubsub"]["subscription"].split("/")[-1]
DEAD_SUB = _cfg["pubsub"]["dead_letter_subscription"].split("/")[-1]
DATASTACK = os.environ.get(
    "CLOUD_MESH_DATASTACK", _cfg.get("monitor", {}).get("datastack")
)
WEBHOOK = os.environ.get("SLACK_WEBHOOK_URL")
# Unset by default: the incoming webhook posts to its own channel.
SLACK_CHANNEL = os.environ.get("SLACK_CHANNEL")

# Strip gs:// and any path tail to get the raw bucket name.
if OUTPUT_BUCKET_URI.startswith("gs://"):
    BUCKET_NAME = OUTPUT_BUCKET_URI[len("gs://") :].split("/", 1)[0]
else:
    BUCKET_NAME = OUTPUT_BUCKET_URI.split("/", 1)[0]

STATE_BLOB_NAME = "_monitor_state.json"


# Metric helpers

def _pubsub_undelivered(subscription_id: str):
    """Return the most recent num_undelivered_messages for ``subscription_id``,
    or None if the metric is unavailable.
    """
    try:
        client = monitoring_v3.MetricServiceClient()
        now = datetime.now(timezone.utc)
        interval = monitoring_v3.TimeInterval(
            {
                "end_time": {"seconds": int(now.timestamp())},
                "start_time": {
                    "seconds": int((now - timedelta(minutes=10)).timestamp())
                },
            }
        )
        results = client.list_time_series(
            name=f"projects/{PROJECT}",
            filter=(
                'metric.type="pubsub.googleapis.com/subscription/'
                'num_undelivered_messages" '
                f'AND resource.labels.subscription_id="{subscription_id}"'
            ),
            interval=interval,
            view=monitoring_v3.ListTimeSeriesRequest.TimeSeriesView.FULL,
        )
        latest_val = None
        latest_ts = None
        for ts in results:
            for pt in ts.points:
                pt_ts = pt.interval.end_time.timestamp()
                if latest_ts is None or pt_ts > latest_ts:
                    latest_val = int(pt.value.int64_value)
                    latest_ts = pt_ts
        return latest_val
    except Exception as e:
        log.warning("metric fetch failed for sub=%s: %s", subscription_id, e)
        return None


_K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
_K8S_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
_K8S_API = "https://kubernetes.default.svc"


def _count_worker_pods(namespace="workers", label="app=cloud-mesh-worker"):
    """Return (running, total) worker pods. (None, None) if the API call fails
    or we're not running in-cluster.
    """
    try:
        with open(_K8S_TOKEN_PATH) as f:
            token = f.read().strip()
        url = (
            f"{_K8S_API}/api/v1/namespaces/{namespace}/pods"
            f"?labelSelector={label}"
        )
        r = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            verify=_K8S_CA_PATH,
            timeout=10,
        )
        r.raise_for_status()
        items = r.json().get("items", [])
        running = sum(1 for p in items if p.get("status", {}).get("phase") == "Running")
        return running, len(items)
    except Exception as e:
        log.warning("pod count fetch failed: %s", e)
        return None, None


def _count_outputs(bucket_name, datastack):
    """Return (full_count, syn_count) for .npz feature files under
    {datastack}/features/.
    """
    sclient = storage.Client()
    bucket = sclient.bucket(bucket_name)
    prefix = f"{datastack}/features/"
    full = 0
    syn = 0
    for blob in bucket.list_blobs(prefix=prefix):
        if not blob.name.endswith(".npz"):
            continue
        if "_syn_" in blob.name:
            syn += 1
        else:
            full += 1
    return full, syn


# State helpers

def _load_state(bucket_name):
    sclient = storage.Client()
    blob = sclient.bucket(bucket_name).blob(STATE_BLOB_NAME)
    if not blob.exists():
        return {}
    try:
        return json.loads(blob.download_as_bytes())
    except Exception as e:
        log.warning("could not parse state file: %s", e)
        return {}


def _save_state(bucket_name, state):
    sclient = storage.Client()
    blob = sclient.bucket(bucket_name).blob(STATE_BLOB_NAME)
    blob.upload_from_string(
        json.dumps(state, indent=2),
        content_type="application/json",
    )


# Slack

def _post_slack(text):
    log.info(text)
    if not WEBHOOK:
        return
    payload = {"text": text}
    if SLACK_CHANNEL:
        payload["channel"] = SLACK_CHANNEL
    try:
        r = requests.post(WEBHOOK, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.warning("slack post failed: %s", e)


# Formatting

def _fmt_eta(minutes):
    if minutes is None or minutes <= 0:
        return "?"
    if minutes < 1:
        return f"{int(minutes * 60)}s"
    if minutes < 60:
        return f"{int(minutes)}m"
    h = int(minutes // 60)
    m = int(minutes - 60 * h)
    return f"{h}h{m:02d}m"


# Main

def main():
    log.info(
        "tick: project=%s bucket=%s sub=%s datastack=%s",
        PROJECT, BUCKET_NAME, SUBSCRIPTION, DATASTACK,
    )

    pending = _pubsub_undelivered(SUBSCRIPTION)
    dead = _pubsub_undelivered(DEAD_SUB)
    full, syn = _count_outputs(BUCKET_NAME, DATASTACK)
    done = full + syn
    pods_running, pods_total = _count_worker_pods()

    now = datetime.now(timezone.utc)
    state = _load_state(BUCKET_NAME)
    last_done = state.get("last_done")
    last_ts_iso = state.get("last_ts")
    last_dead = state.get("last_dead", 0) or 0
    was_active = bool(state.get("was_active", False))

    rate_per_min = None
    if last_done is not None and last_ts_iso:
        try:
            last_ts = datetime.fromisoformat(last_ts_iso)
            dt_min = (now - last_ts).total_seconds() / 60.0
            if dt_min > 0:
                rate_per_min = (done - last_done) / dt_min
        except Exception:
            pass

    eta_min = None
    if pending is not None and rate_per_min and rate_per_min > 0:
        eta_min = pending / rate_per_min

    pending_str = "?" if pending is None else str(pending)
    dead_str = "?" if dead is None else str(dead)
    rate_str = "?" if rate_per_min is None else f"{rate_per_min:.1f}/min"
    eta_str = _fmt_eta(eta_min)
    if pods_running is None:
        pods_str = "?"
    elif pods_total is not None and pods_total != pods_running:
        pods_str = f"{pods_running}/{pods_total}"
    else:
        pods_str = str(pods_running)

    summary = (
        f"HKS {DATASTACK} - pending {pending_str} - done {done} "
        f"({syn} syn / {full} full) - dead {dead_str} - "
        f"pods {pods_str} - rate {rate_str} - eta {eta_str}"
    )

    is_active = (pending or 0) > 0
    new_failures = max(0, (dead or 0) - last_dead)

    # Post a summary while a batch is active, plus one "drained" line
    # on the transition to idle. Stay silent during quiet periods.
    if is_active or was_active:
        _post_slack(summary)

    if new_failures > 0:
        _post_slack(
            f"HKS {DATASTACK} - :warning: {new_failures} new dead-lettered "
            f"task(s) (total: {dead_str})"
        )

    if was_active and not is_active and pending is not None:
        _post_slack(
            f"HKS {DATASTACK} - :white_check_mark: queue drained, "
            f"{done} outputs total"
        )

    new_state = {
        "last_done": done,
        "last_ts": now.isoformat(),
        "last_dead": dead if dead is not None else last_dead,
        "was_active": is_active,
    }
    _save_state(BUCKET_NAME, new_state)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Make sure the traceback ends up in stdout/stderr where kubectl
        # logs can find it, then exit non-zero so the Job is marked failed.
        log.error("monitor tick crashed:\n%s", traceback.format_exc())
        sys.exit(1)
