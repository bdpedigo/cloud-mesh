"""
slack_utils.py — Resolve a Slack incoming-webhook URL and post messages.

Extracted from ``enqueue_pubsub.py`` so other scripts (workers, monitor cron,
local CLIs) can reuse the same secret-resolution logic without copying it.

Resolution order for the webhook URL:

  1. ``SLACK_WEBHOOK_URL`` environment variable (overrides everything).
  2. GCP Secret Manager via the ``google-cloud-secret-manager`` Python
     client, using Application Default Credentials. Works on Windows
     without the ``gcloud`` CLI installed.
  3. ``gcloud`` CLI subprocess (legacy fallback for environments where the
     Python client isn't installed but ``gcloud`` is on PATH).

Defaults for ``project`` and ``secret_name`` are read from
``config.toml`` at import time. Pass the kwargs explicitly to override.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)


# ── default project / secret pulled from config.toml at import time ──────────

_config_path = Path(
    os.environ.get("CONFIG_PATH", Path(__file__).parent / "config.toml")
)
try:
    with open(_config_path, "rb") as _f:
        _cfg = tomllib.load(_f)
    DEFAULT_PROJECT: Optional[str] = _cfg.get("cluster", {}).get("project")
    DEFAULT_SECRET_NAME: str = _cfg.get("job", {}).get(
        "slack_secret_name", "slack-webhook"
    )
except FileNotFoundError:
    DEFAULT_PROJECT = None
    DEFAULT_SECRET_NAME = "slack-webhook"

# No default channel: when unset, the incoming webhook posts to whichever
# channel it was created for. Set SLACK_CHANNEL to override.
DEFAULT_CHANNEL: Optional[str] = os.environ.get("SLACK_CHANNEL")


# ── webhook resolution ────────────────────────────────────────────────────────

def get_slack_webhook(
    project: Optional[str] = None,
    secret_name: Optional[str] = None,
) -> Optional[str]:
    """Return the Slack webhook URL, or None if it can't be resolved.

    Logs at INFO/WARNING for every code path so failures aren't silent.
    """
    project = project if project is not None else DEFAULT_PROJECT
    secret_name = secret_name if secret_name is not None else DEFAULT_SECRET_NAME

    env = os.environ.get("SLACK_WEBHOOK_URL")
    if env:
        log.info("slack webhook: using SLACK_WEBHOOK_URL env var")
        return env

    # Preferred path: Python Secret Manager client (works anywhere ADC is
    # available — same credentials caveclient/cloud-volume use, no extra
    # tooling required on Windows).
    try:
        from google.cloud import secretmanager
    except ImportError as e:
        log.warning(
            "slack webhook: google-cloud-secret-manager not installed (%s); "
            "run `uv sync` to install. Trying gcloud CLI fallback.", e,
        )
    else:
        if not project:
            log.warning(
                "slack webhook: no GCP project configured (set "
                "config.toml::cluster.project or pass project=)"
            )
            return None
        try:
            from google.api_core import exceptions as gax_exceptions

            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{project}/secrets/{secret_name}/versions/latest"
            log.info("slack webhook: fetching %s via Secret Manager Python client", name)
            response = client.access_secret_version(request={"name": name})
            secret = response.payload.data.decode("utf-8").strip()
            if secret:
                log.info("slack webhook: fetched OK from Secret Manager (%d chars)", len(secret))
                return secret
            log.warning("slack webhook: Secret Manager returned empty payload for %s", name)
            return None
        except gax_exceptions.GoogleAPICallError as e:
            log.warning(
                "slack webhook: Secret Manager API error for project=%s secret=%s: %s",
                project, secret_name, e,
            )
            return None
        except Exception as e:  # auth / permission / network — fall through
            log.warning(
                "slack webhook: Secret Manager Python client failed (%s: %s); "
                "trying gcloud CLI", type(e).__name__, e,
            )

    # Legacy fallback: shell out to gcloud. Useful inside the worker container.
    if not project:
        log.warning(
            "slack webhook: no GCP project configured for gcloud fallback"
        )
        return None
    try:
        out = subprocess.check_output(
            [
                "gcloud", "secrets", "versions", "access", "latest",
                f"--secret={secret_name}", f"--project={project}",
            ],
            stderr=subprocess.PIPE,
            text=True,
        ).strip()
        if out:
            log.info("slack webhook: fetched OK from gcloud CLI (%d chars)", len(out))
            return out
        log.warning("slack webhook: gcloud returned empty payload")
        return None
    except FileNotFoundError as e:
        log.warning(
            "slack webhook: gcloud CLI not on PATH (%s). Install "
            "google-cloud-secret-manager (`uv sync`) or run "
            "`gcloud auth application-default login` and ensure your account "
            "has roles/secretmanager.secretAccessor on project=%s.",
            e, project,
        )
        return None
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip() if hasattr(e, "stderr") else ""
        log.warning(
            "slack webhook: gcloud CLI failed (rc=%s): %s",
            e.returncode, stderr or e,
        )
        return None


# ── post a message ────────────────────────────────────────────────────────────

def post_slack(
    text: str,
    *,
    channel: Optional[str] = None,
    project: Optional[str] = None,
    secret_name: Optional[str] = None,
) -> bool:
    """Post ``text`` to Slack via the configured incoming webhook.

    Returns True on success, False otherwise. Always logs the outcome —
    even Slack's "200 OK + body=invalid_payload/channel_not_found" failure
    pattern is detected and surfaced as a WARNING.

    Parameters
    ----------
    text : str
        Message body. Echoed at INFO before posting so logs preserve a
        record even if the webhook is unavailable.
    channel : str, optional
        Channel override. Defaults to ``DEFAULT_CHANNEL`` (which honors the
        ``SLACK_CHANNEL`` env var). Pass an empty string ``""`` to use the
        webhook's own default channel without an override.
    project, secret_name : str, optional
        Forwarded to :func:`get_slack_webhook`. Defaults read from
        ``config.toml``.
    """
    log.info(text)
    webhook = get_slack_webhook(project=project, secret_name=secret_name)
    if not webhook:
        log.warning("slack webhook unavailable; skipping post")
        return False

    if channel is None:
        channel = DEFAULT_CHANNEL

    payload = {"text": text}
    if channel:
        payload["channel"] = channel

    try:
        r = requests.post(webhook, json=payload, timeout=10)
    except Exception as e:
        log.warning("slack post failed (network): %s: %s", type(e).__name__, e)
        return False

    body = (r.text or "").strip()
    # Slack incoming webhooks return HTTP 200 + body "ok" on success.
    # On failure they often still return 200 but with a body like
    # "invalid_payload", "no_service", "channel_not_found", "no_team".
    if r.status_code != 200:
        log.warning(
            "slack post failed: HTTP %s — %s (channel=%s)",
            r.status_code, body[:200], channel,
        )
        return False
    if body.lower() != "ok":
        log.warning(
            "slack post returned 200 but body=%r (channel=%s) — message likely "
            "not delivered. If body is 'channel_not_found', drop the channel "
            "override and let the webhook post to its default channel.",
            body[:200], channel,
        )
        return False
    log.info("slack post ok (channel=%s)", channel)
    return True


# ── tiny CLI for ad-hoc testing ──────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        stream=sys.stdout,
    )
    msg = " ".join(sys.argv[1:]) or "slack_utils smoke test"
    ok = post_slack(msg)
    sys.exit(0 if ok else 1)
