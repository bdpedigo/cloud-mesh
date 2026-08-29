"""
gsheet_utils.py — Lightweight Google Sheets + Drive helpers.

Used by ``check_status.py`` to publish status reports to a persistent
Google Sheet (the dashboard) plus a Drive folder of per-root-id detail
CSVs. The IDs of the sheet and folder are cached locally in
``status_report_state.json`` so subsequent runs update the same
artifacts in place.

Authentication
--------------
Uses **Application Default Credentials** with Sheets + Drive scopes.
Before first use, run::

    gcloud auth application-default login \\
        --scopes=openid,https://www.googleapis.com/auth/userinfo.email,\\
        https://www.googleapis.com/auth/cloud-platform,\\
        https://www.googleapis.com/auth/spreadsheets,\\
        https://www.googleapis.com/auth/drive

(That's a single line — line-continued for readability.) The same ADC
file is reused for caveclient / cloud-volume / Secret Manager, so this
only needs to happen once per machine.

Dependencies
------------
``gspread`` and ``google-api-python-client``. Add to pyproject.toml::

    "gspread>=6.0.0",
    "google-api-python-client>=2.100.0",
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Optional, Sequence

log = logging.getLogger(__name__)


SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive.file",  # needed by gspread.create()
]

import os

# Where we cache the spreadsheet/folder IDs created on first run. Stored
# next to this file so the repo can ship without it; a fresh checkout
# will create new artifacts on first invocation. Override with
# CLOUD_MESH_STATUS_STATE env var if you want to put it elsewhere.
DEFAULT_STATE_PATH = Path(
    os.environ.get(
        "CLOUD_MESH_STATUS_STATE",
        Path(__file__).parent / "status_report_state.json",
    )
)

# ── auth config ───────────────────────────────────────────────────────────────
#
# Auth modes, tried in order:
#
# 1. OAuth Desktop app (recommended) — set GOOGLE_OAUTH_CLIENT_FILE to the
#      path of an OAuth 2.0 Client ID JSON (type: Desktop app) downloaded from
#      GCP Console -> APIs & Services -> Credentials.
#      A browser opens once for user consent; the token is cached in
#      GOOGLE_OAUTH_TOKEN_FILE (default: gspread_token.json next to this file).
#
# 2. SA key file — set GOOGLE_SHEETS_SA_KEY=/path/to/key.json
#      Requires key creation (may be blocked by org policy).
#
# 3. SA impersonation — set GOOGLE_SHEETS_SA_EMAIL=sa@project.iam.gserviceaccount.com
#      Blocked on Google Workspace orgs that restrict SA access to Sheets API.

_OAUTH_CLIENT_FILE: Optional[str] = os.environ.get(
    "GOOGLE_OAUTH_CLIENT_FILE",
    str(Path(__file__).parent / "client_secret.json")
    if (Path(__file__).parent / "client_secret.json").exists()
    else None,
)
_OAUTH_TOKEN_FILE: str = os.environ.get(
    "GOOGLE_OAUTH_TOKEN_FILE",
    str(Path(__file__).parent / "gspread_token.json"),
)
_SA_KEY_PATH: Optional[str] = os.environ.get("GOOGLE_SHEETS_SA_KEY")
_SA_EMAIL: Optional[str] = os.environ.get("GOOGLE_SHEETS_SA_EMAIL")


# ── auth ──────────────────────────────────────────────────────────────────────

def _gspread_client():
    import gspread

    if _OAUTH_CLIENT_FILE:
        log.info("auth: OAuth Desktop app (%s)", _OAUTH_CLIENT_FILE)
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from google_auth_oauthlib.flow import InstalledAppFlow

        creds = None
        token_path = Path(_OAUTH_TOKEN_FILE)
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(_OAUTH_TOKEN_FILE, SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(_OAUTH_CLIENT_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
            token_path.write_text(creds.to_json())
        return gspread.Client(auth=creds)

    if _SA_KEY_PATH:
        log.info("auth: service account key file %s", _SA_KEY_PATH)
        return gspread.service_account(filename=_SA_KEY_PATH, scopes=SCOPES)

    if _SA_EMAIL:
        from google.auth import default, impersonated_credentials
        from google.auth.transport.requests import Request
        log.info("auth: impersonating service account %s", _SA_EMAIL)
        source_creds, _ = default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        creds = impersonated_credentials.Credentials(
            source_credentials=source_creds,
            target_principal=_SA_EMAIL,
            target_scopes=SCOPES,
            lifetime=3600,
        )
        creds.refresh(Request())
        return gspread.Client(auth=creds)

    raise RuntimeError(
        "No Google Sheets credentials configured. Set GOOGLE_OAUTH_CLIENT_FILE, "
        "GOOGLE_SHEETS_SA_KEY, or GOOGLE_SHEETS_SA_EMAIL."
    )


def _credentials():
    """Return raw google-auth credentials (used only by Drive service; not needed
    in the OAuth-only path)."""
    if _SA_KEY_PATH:
        from google.oauth2.service_account import Credentials
        return Credentials.from_service_account_file(_SA_KEY_PATH, scopes=SCOPES)
    from google.auth import default
    creds, _ = default(scopes=SCOPES)
    return creds


def _drive_service():
    from googleapiclient.discovery import build

    return build("drive", "v3", credentials=_credentials(), cache_discovery=False)


# ── state file ────────────────────────────────────────────────────────────────

def load_state(path: Path = DEFAULT_STATE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(state: dict, path: Path = DEFAULT_STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# ── Drive ─────────────────────────────────────────────────────────────────────

def get_or_create_folder(
    name: str,
    *,
    parent_id: Optional[str] = None,
    drive=None,
) -> str:
    """Return a Drive folder id whose name matches ``name``. Creates it
    if no such folder exists in ``parent_id`` (root if None)."""
    drive = drive or _drive_service()

    q_parts = [
        "mimeType='application/vnd.google-apps.folder'",
        "trashed=false",
        f"name='{name}'",
    ]
    if parent_id:
        q_parts.append(f"'{parent_id}' in parents")
    q = " and ".join(q_parts)
    resp = drive.files().list(q=q, fields="files(id,name)").execute()
    files = resp.get("files", [])
    if files:
        log.info("Drive folder '%s' already exists (id=%s)", name, files[0]["id"])
        return files[0]["id"]

    body = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        body["parents"] = [parent_id]
    folder = drive.files().create(body=body, fields="id").execute()
    log.info("created Drive folder '%s' (id=%s)", name, folder["id"])
    return folder["id"]


def upload_csv_to_folder(
    folder_id: str,
    filename: str,
    csv_text: str,
    *,
    drive=None,
) -> str:
    """Create-or-replace a CSV file named ``filename`` inside ``folder_id``.

    Returns the file id. Replacement matches by name within the folder
    so re-running the report overwrites the previous detail file rather
    than creating duplicates.
    """
    from googleapiclient.http import MediaIoBaseUpload

    drive = drive or _drive_service()

    # Look up an existing file by name in the folder.
    q = (
        f"name='{filename}' and trashed=false and "
        f"'{folder_id}' in parents"
    )
    resp = drive.files().list(q=q, fields="files(id,name)").execute()
    existing = resp.get("files", [])

    media = MediaIoBaseUpload(
        io.BytesIO(csv_text.encode("utf-8")),
        mimetype="text/csv",
        resumable=False,
    )

    if existing:
        file_id = existing[0]["id"]
        drive.files().update(fileId=file_id, media_body=media).execute()
        return file_id

    body = {"name": filename, "parents": [folder_id]}
    f = drive.files().create(body=body, media_body=media, fields="id").execute()
    return f["id"]


def drive_file_url(file_id: str) -> str:
    """Web-viewable URL for a Drive file id."""
    return f"https://drive.google.com/file/d/{file_id}/view"


# ── Sheets ────────────────────────────────────────────────────────────────────

def get_or_create_spreadsheet(
    title: str,
    *,
    parent_folder_id: Optional[str] = None,
    gc=None,
    drive=None,
) -> str:
    """Return a spreadsheet id whose name matches ``title``. Creates it
    in ``parent_folder_id`` (or My Drive root) if no such sheet exists.
    """
    gc = gc or _gspread_client()
    drive = drive or _drive_service()

    q_parts = [
        "mimeType='application/vnd.google-apps.spreadsheet'",
        "trashed=false",
        f"name='{title}'",
    ]
    if parent_folder_id:
        q_parts.append(f"'{parent_folder_id}' in parents")
    q = " and ".join(q_parts)
    resp = drive.files().list(q=q, fields="files(id,name)").execute()
    files = resp.get("files", [])
    if files:
        log.info("Spreadsheet '%s' already exists (id=%s)", title, files[0]["id"])
        return files[0]["id"]

    sh = gc.create(title)
    if parent_folder_id:
        # Move the newly-created spreadsheet into the target folder.
        file_id = sh.id
        f = drive.files().get(fileId=file_id, fields="parents").execute()
        prev_parents = ",".join(f.get("parents", []))
        drive.files().update(
            fileId=file_id,
            addParents=parent_folder_id,
            removeParents=prev_parents,
            fields="id, parents",
        ).execute()
    log.info("created Spreadsheet '%s' (id=%s)", title, sh.id)
    return sh.id


def update_worksheet_with_df(
    spreadsheet_id: str,
    sheet_name: str,
    df,
    *,
    gc=None,
):
    """Replace the contents of ``sheet_name`` with ``df`` (creating the
    worksheet if missing). ``last_updated`` is expected as a column in ``df``.
    """
    gc = gc or _gspread_client()
    sh = gc.open_by_key(spreadsheet_id)

    try:
        ws = sh.worksheet(sheet_name)
        ws.clear()
    except Exception:  # noqa: BLE001
        ws = sh.add_worksheet(
            title=sheet_name,
            rows=str(max(100, len(df) + 10)),
            cols=str(max(20, len(df.columns) + 2)),
        )

    rows = [list(df.columns)]
    for row in df.itertuples(index=False, name=None):
        rows.append([
            ("" if v is None or (hasattr(v, "__float__") and v != v) else v)  # NaN -> ""
            for v in row
        ])
    ws.update(values=rows, range_name="A1")
    return ws


def spreadsheet_url(spreadsheet_id: str) -> str:
    return f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
