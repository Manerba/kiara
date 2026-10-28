"""Tests fuer IMAP-Scanner Edge-Cases (Phase 6 Erweiterung).

Angepasst an UID-basiertes 3-Phasen-Scan-Protokoll:
    Phase 1: Ordner-Discovery
    Phase 2: UID-Sammlung (conn.uid("search", ...))
    Phase 3: Batch-Fetch (conn.uid("fetch", uid_set, "(UID RFC822)"))
"""

import os
import tempfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

from kiara.ingestion.imap_scanner import IMAPScanner


def _make_mock_mail(
    subject="Test",
    body="Body text",
    sender="test@example.com",
    date="Mon, 01 Jan 2024 10:00:00 +0100",
):
    """Erzeugt eine MIME-Mail fuer Tests."""
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = "empfaenger@firma.de"
    msg["Date"] = date
    msg.attach(MIMEText(body, "plain", "utf-8"))
    return msg


def _setup_uid_conn(mock_conn, mail_msg, uid=b"1", num_uids=1):
    """Konfiguriert mock_conn fuer UID-basiertes IMAP-Scan-Protokoll.

    Setzt select, response(UIDVALIDITY), uid("search"), uid("fetch") korrekt.
    """
    mock_conn.select.return_value = ("OK", [b"1"])
    mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

    # uid-Dispatch: "search" vs "fetch"
    uid_list = b" ".join(bytes(str(i + 1), "ascii") for i in range(num_uids))
    mail_bytes = mail_msg.as_bytes() if mail_msg else b""

    def uid_dispatch(cmd, *args):
        if cmd == "search":
            return ("OK", [uid_list])
        elif cmd == "fetch":
            # Batch-Format: [(b'1 (UID <uid> RFC822 {size}', raw_bytes), b')']
            results = []
            # Parse the uid_set from args[0]
            uid_set = args[0] if args else uid
            uids_requested = uid_set.split(b",") if isinstance(uid_set, bytes) else [uid]
            for u in uids_requested:
                header = u + b" (UID " + u + b" RFC822 {" + str(len(mail_bytes)).encode() + b"}"
                results.append((header, mail_bytes))
                results.append(b")")
            return ("OK", results)
        return ("OK", [b""])

    mock_conn.uid.side_effect = uid_dispatch


class TestScanMultipleFolders:
    """Mehrere IMAP-Ordner werden alle selektiert."""

    @patch.object(IMAPScanner, "connect")
    def test_scan_multiple_folders(self, mock_connect):
        """Scanner mit 3 Ordnern -> conn.select() wird fuer jede Phase aufgerufen."""
        mail = _make_mock_mail("Test Mail", "Body")

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, mail)

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_folders": "INBOX,Sent,Archive",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scanner.scan(tmpdir)
            # Phase 1 (UID-Sammlung): select pro Ordner = 3
            # Phase 2 (Fetch): select pro Ordner mit Mails = 3
            # Total = 6 select-Aufrufe
            assert mock_conn.select.call_count == 6
            # Alle drei Ordner wurden selektiert
            folders_called = [c.args[0] for c in mock_conn.select.call_args_list]
            assert "INBOX" in folders_called
            assert "Sent" in folders_called
            assert "Archive" in folders_called


class TestScanFolderSelectFails:
    """Ordner-Selektion schlaegt fehl -> wird uebersprungen."""

    @patch.object(IMAPScanner, "connect")
    def test_scan_folder_select_fails(self, mock_connect):
        """Ein Ordner gibt NO zurueck -> wird uebersprungen, andere normal."""
        mail = _make_mock_mail("Test", "Body")

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

        mail_bytes = mail.as_bytes()

        def uid_dispatch(cmd, *args):
            if cmd == "search":
                return ("OK", [b"1"])
            elif cmd == "fetch":
                uid_set = args[0] if args else b"1"
                uids_requested = uid_set.split(b",") if isinstance(uid_set, bytes) else [b"1"]
                results = []
                for u in uids_requested:
                    header = u + b" (UID " + u + b" RFC822 {" + str(len(mail_bytes)).encode() + b"}"
                    results.append((header, mail_bytes))
                    results.append(b")")
                return ("OK", results)
            return ("OK", [b""])

        mock_conn.uid.side_effect = uid_dispatch

        # Phase 1 (UID-Sammlung): INBOX OK, Sent NO, Archive OK
        # Phase 2 (Fetch): INBOX OK, Archive OK (nur Ordner die Phase 1 bestanden)
        mock_conn.select.side_effect = [
            ("OK", [b"1"]),
            ("NO", [b"Folder not found"]),
            ("OK", [b"1"]),
            ("OK", [b"1"]),
            ("OK", [b"1"]),
        ]

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_folders": "INBOX,Sent,Archive",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            # Nur 2 Ordner liefern Mails (INBOX + Archive)
            assert len(files) == 2


class TestScanNoMessagesSinceDate:
    """Keine Mails im Zeitraum -> leere Liste, kein Fehler."""

    @patch.object(IMAPScanner, "connect")
    def test_scan_no_messages_since_date(self, mock_connect):
        """uid("search") gibt leeren Ergebnis-String -> leere Liste."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.select.return_value = ("OK", [b"0"])
        mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

        def uid_dispatch(cmd, *args):
            if cmd == "search":
                return ("OK", [b""])
            return ("OK", [b""])

        mock_conn.uid.side_effect = uid_dispatch

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_folders": "INBOX",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            assert files == []


class TestAttachmentTooLarge:
    """Anhang groesser als max -> wird uebersprungen, Text bleibt."""

    @patch.object(IMAPScanner, "connect")
    def test_attachment_too_large(self, mock_connect):
        """Grosser Anhang wird uebersprungen, Mail-Text wird geschrieben."""
        msg = MIMEMultipart()
        msg["Subject"] = "Mail mit grossem Anhang"
        msg["From"] = "test@example.com"
        msg["To"] = "user@firma.de"
        msg["Date"] = "Mon, 15 Jan 2024 12:00:00 +0100"
        msg.attach(MIMEText("Siehe Anhang.", "plain", "utf-8"))

        # Anhang > 1 MB (wir setzen max auf 1 MB)
        big_payload = b"X" * (2 * 1024 * 1024)  # 2 MB
        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(big_payload)
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename="grosses_dokument.pdf")
        msg.attach(attachment)

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, msg)

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_download_attachments": True,
                "imap_max_attachment_mb": 1,  # Nur 1 MB erlaubt
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            # Nur die Text-Datei, kein Anhang
            assert len(files) == 1
            assert files[0].endswith(".txt")
            content = open(files[0], encoding="utf-8").read()
            assert "Siehe Anhang" in content


class TestAttachmentsDisabled:
    """Anhang-Download deaktiviert -> nur Mail-Text."""

    @patch.object(IMAPScanner, "connect")
    def test_attachments_disabled(self, mock_connect):
        """imap_download_attachments=False -> keine Anhaenge."""
        msg = MIMEMultipart()
        msg["Subject"] = "Mail mit Anhang"
        msg["From"] = "test@example.com"
        msg["To"] = "user@firma.de"
        msg["Date"] = "Mon, 15 Jan 2024 12:00:00 +0100"
        msg.attach(MIMEText("Text der Mail.", "plain", "utf-8"))

        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(b"%PDF-1.4 test content")
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename="rechnung.pdf")
        msg.attach(attachment)

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, msg)

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_download_attachments": False,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            # Nur Text-Datei, kein Attachment
            assert len(files) == 1
            assert files[0].endswith(".txt")
            attachment_files = [f for f in files if "attachments" in f]
            assert len(attachment_files) == 0


class TestDateParseFailure:
    """Ungueltiges Datum -> Fallback auf '00000000' Prefix."""

    @patch.object(IMAPScanner, "connect")
    def test_date_parse_failure(self, mock_connect):
        """Ungueltiger Date-Header -> Fallback auf 00000000."""
        msg = MIMEMultipart()
        msg["Subject"] = "Test ohne Datum"
        msg["From"] = "test@example.com"
        msg["To"] = "user@firma.de"
        msg["Date"] = "INVALID DATE STRING"
        msg.attach(MIMEText("Body.", "plain", "utf-8"))

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, msg)

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            assert len(files) == 1
            # Dateiname beginnt mit 00000000 (Fallback)
            filename = os.path.basename(files[0])
            assert filename.startswith("00000000")


class TestScanFetchFailure:
    """uid("fetch") gibt NO zurueck -> leere Liste fuer diese Mail, kein Crash."""

    @patch.object(IMAPScanner, "connect")
    def test_scan_fetch_failure(self, mock_connect):
        """conn.uid("fetch") gibt NO zurueck -> leere Liste, kein Absturz."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.select.return_value = ("OK", [b"1"])
        mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

        def uid_dispatch(cmd, *args):
            if cmd == "search":
                return ("OK", [b"1"])
            elif cmd == "fetch":
                return ("NO", [None])
            return ("OK", [b""])

        mock_conn.uid.side_effect = uid_dispatch

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            assert files == []


class TestTestConnectionReturnsFolders:
    """test_connection() erfolgreich -> Ordnerliste."""

    @patch.object(IMAPScanner, "connect")
    def test_test_connection_returns_folders(self, mock_connect):
        """Erfolgreicher Verbindungstest gibt Ordnerliste zurueck."""
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.list.return_value = (
            "OK",
            [
                b'(\\HasNoChildren) "/" "INBOX"',
                b'(\\HasNoChildren) "/" "Sent"',
                b'(\\HasNoChildren) "/" "Drafts"',
            ],
        )

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
            }
        )

        result = scanner.test_connection()
        assert result["success"] is True
        assert "folders" in result
        assert isinstance(result["folders"], list)
        assert len(result["folders"]) == 3
