#!/usr/bin/env python3
import os
import sys
import signal
import json
import logging
import threading
import tomllib
from pathlib import Path
from google.cloud import pubsub_v1

from run_hks import run_hks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s \u2014 %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("pubsub-worker")

# \u2500\u2500 Load config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
_config_path = Path(os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml"))
with open(_config_path, "rb") as _f:
    _config = tomllib.load(_f)

_PUBSUB_HELP = (
    "config.toml has no [pubsub] section. Pub/Sub mode needs one: copy the "
    "[pubsub] block from config-template.toml and fill in your project's "
    "topic and subscription paths."
)

if "pubsub" not in _config:
    raise SystemExit(_PUBSUB_HELP)

_pubsub = _config["pubsub"]

PROJECT          = os.environ.get("GCP_PROJECT", _config["cluster"]["project"])
SUBSCRIPTION     = os.environ.get("PUBSUB_SUBSCRIPTION", _pubsub["subscription"])
ACK_DEADLINE     = int(os.environ.get("PUBSUB_ACK_DEADLINE", _pubsub["ack_deadline_seconds"]))
HEARTBEAT_INTERVAL = int(os.environ.get("PUBSUB_HEARTBEAT_INTERVAL", _pubsub["heartbeat_interval_seconds"]))


# \u2500\u2500 Heartbeat \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _start_heartbeat(message: pubsub_v1.subscriber.message.Message, stop_event: threading.Event):
    """Extend the ack deadline every HEARTBEAT_INTERVAL seconds until stop_event is set."""
    def _beat():
        while not stop_event.wait(timeout=HEARTBEAT_INTERVAL):
            try:
                message.modify_ack_deadline(ACK_DEADLINE)
                log.debug("heartbeat: extended ack deadline by %ds", ACK_DEADLINE)
            except Exception:
                log.warning("heartbeat: failed to extend ack deadline", exc_info=True)
    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    return t


# \u2500\u2500 Main \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def main():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path(
        PROJECT, SUBSCRIPTION.split("/")[-1]  # accept full path or bare name
    )

    def callback(message: pubsub_v1.subscriber.message.Message) -> None:
        stop_heartbeat = threading.Event()
        try:
            payload = json.loads(message.data.decode("utf-8"))
            root_id = int(payload["root_id"])
            datastack = payload["datastack"]

            # Optional targeted-region fields. All three (synapse_id, point, radius)
            # must be supplied together; if any are missing the worker falls back
            # to full-mesh HKS. synapse_id tags the output file.
            raw_synapse = payload.get("synapse_id")
            raw_point = payload.get("query_point_nm")
            raw_radius = payload.get("query_radius_nm")
            synapse_id = int(raw_synapse) if raw_synapse is not None else None
            query_point_nm = (
                [float(c) for c in raw_point] if raw_point is not None else None
            )
            query_radius_nm = (
                float(raw_radius) if raw_radius is not None else None
            )

            if (
                synapse_id is not None
                and query_point_nm is not None
                and query_radius_nm is not None
            ):
                log.info(
                    "received CHUNKED task root_id=%s datastack=%s synapse_id=%s point=%s r=%s nm",
                    root_id, datastack, synapse_id, query_point_nm, query_radius_nm,
                )
            else:
                log.info("received task root_id=%s datastack=%s", root_id, datastack)

            _start_heartbeat(message, stop_heartbeat)
            run_hks(
                root_id,
                datastack,
                synapse_id=synapse_id,
                query_point_nm=query_point_nm,
                query_radius_nm=query_radius_nm,
            )

            message.ack()
            log.info("task done and acked root_id=%s", root_id)
        except Exception as e:
            log.exception("task processing failed; nacking for retry: %s", e)
            try:
                message.nack()
            except Exception:
                log.exception("failed to nack message (continuing)")
        finally:
            stop_heartbeat.set()

    flow_control = pubsub_v1.types.FlowControl(max_messages=1)
    streaming_pull_future = subscriber.subscribe(
        subscription_path, callback=callback, flow_control=flow_control
    )
    log.info("Listening for messages on %s ...", subscription_path)

    def _shutdown(signum, frame):
        log.info("Received signal %s: shutting down...", signum)
        streaming_pull_future.cancel()
        subscriber.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        streaming_pull_future.result()
    except Exception as e:
        log.exception("streaming pull terminated: %s", e)
        streaming_pull_future.cancel()
        subscriber.close()

if __name__ == "__main__":
    main()
