"""Ordnernamen mit Leerzeichen und als Jobfehler gezaehlte Ordnerausfaelle."""

import imaplib
import os
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

from kiara.ingestion._file_utils import _INTERNAL_DATASOURCE_FILES
from kiara.ingestion.imap_scanner import (
    SCAN_FOLDER_ERRORS_FILE,
    IMAPScanner,
    quote_mailbox,
    read_scan_folder_errors,
)
from kiara.web.ingestion_worker import IngestionWorker


def _mail_bytes():
    msg = MIMEText("Body")
    msg["Subject"] = "Test"
    msg["From"] = "a@example.com"
    msg["Date"] = "Mon, 01 Jan 2024 10:00:00 +0100"
    return msg.as_bytes()


def _scanner(folders):
    return IMAPScanner(
        {
            "imap_server": "mail.test.local",
            "imap_username": "user",
            "imap_password": "pass",
            "imap_folders": folders,
        }
    )


def _server(bad_folders=()):
    """Mock-Server, der ungequotete Namen mit Leerzeichen wie Exchange ablehnt."""
    conn = MagicMock()
    raw = _mail_bytes()

    def select(mailbox, readonly=False):
        if " " in mailbox and not mailbox.startswith('"'):
            raise imaplib.IMAP4.error("EXAMINE command error: BAD [b'Command Argument Error. 12']")
        if mailbox in bad_folders:
            return ("NO", [b"Folder not found"])
        return ("OK", [b"1"])

    def uid(cmd, *args):
        if cmd == "search":
            return ("OK", [b"1"])
        header = b"1 (UID 1 BODY[] {" + str(len(raw)).encode() + b"}"
        return ("OK", [(header, raw), b")"])

    conn.select.side_effect = select
    conn.uid.side_effect = uid
    conn.response.return_value = ("UIDVALIDITY", [b"7"])
    return conn


def test_quote_mailbox_escapes_rfc3501():
    assert quote_mailbox("Gesendete Elemente") == '"Gesendete Elemente"'
    assert quote_mailbox('a"b\\c') == '"a\\"b\\\\c"'


def test_manually_quoted_folder_entry_is_normalized():
    scanner = _scanner('"Gesendete Elemente", INBOX, "a\\"b"')
    assert scanner._folders == ["Gesendete Elemente", "INBOX", 'a"b']


@patch.object(IMAPScanner, "connect")
def test_folder_with_spaces_is_scanned(mock_connect, tmp_path):
    conn = _server()
    mock_connect.return_value = conn
    (tmp_path / SCAN_FOLDER_ERRORS_FILE).write_text('{"alt": "stale"}')

    files = _scanner("INBOX, Gesendete Elemente, Tier &- technik St. Gallen").scan(
        str(tmp_path)
    )

    selected = [c.args[0] for c in conn.select.call_args_list]
    assert '"Gesendete Elemente"' in selected
    assert '"Tier &- technik St. Gallen"' in selected
    assert len(files) == 3
    assert read_scan_folder_errors(str(tmp_path)) == {}
    assert (tmp_path / ".scan_done").exists()


@patch.object(IMAPScanner, "connect")
def test_unselectable_folder_is_recorded_and_blocks_scan_cache(mock_connect, tmp_path):
    mock_connect.return_value = _server(bad_folders={'"Weg"'})

    files = _scanner("INBOX, Weg").scan(str(tmp_path))

    assert len(files) == 1
    errors = read_scan_folder_errors(str(tmp_path))
    assert list(errors) == ["Weg"]
    assert "NO" in errors["Weg"]
    assert not (tmp_path / ".scan_done").exists()
    assert (tmp_path / ".uid_registry.json").exists()


def test_folder_errors_file_is_not_ingested():
    assert SCAN_FOLDER_ERRORS_FILE in _INTERNAL_DATASOURCE_FILES
    assert SCAN_FOLDER_ERRORS_FILE + ".tmp" in _INTERNAL_DATASOURCE_FILES


def test_worker_counts_folder_errors(tmp_path):
    (tmp_path / SCAN_FOLDER_ERRORS_FILE).write_text(
        '{"Sent Items": "SELECT NO", "FiBu &ANw-bergabe": "BAD"}'
    )
    logged = []
    callbacks = {"on_file_log": logged.append, "flush_file_logs": lambda: None}
    stats = {"errors": 2}

    IngestionWorker._record_mail_scan_errors(
        {"mail_scan_dir": str(tmp_path)}, stats, callbacks
    )

    assert stats["errors"] == 4
    assert set(stats["mail_scan_folder_errors"]) == {"Sent Items", "FiBu &ANw-bergabe"}
    assert {e.file_name for e in logged} == {"Sent Items", "FiBu Übergabe"}
    assert {e.error_phase for e in logged} == {"mail_scan"}
    assert {e.status for e in logged} == {"error"}


def test_worker_ignores_non_mail_scans(tmp_path):
    (tmp_path / SCAN_FOLDER_ERRORS_FILE).write_text('{"x": "y"}')
    stats = {}
    IngestionWorker._record_mail_scan_errors({"mail_scan_dir": None}, stats, {})
    assert stats == {}
    assert os.path.exists(tmp_path / SCAN_FOLDER_ERRORS_FILE)
