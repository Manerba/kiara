"""Tests fuer den IMAP-Scanner (Phase 6).

Angepasst an UID-basiertes 3-Phasen-Scan-Protokoll:
    Phase 1: Ordner-Discovery
    Phase 2: UID-Sammlung (conn.uid("search", ...))
    Phase 3: Batch-Fetch (conn.uid("fetch", uid_set, "(UID RFC822)"))
"""

import json
import os
import tempfile
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import MagicMock, patch

import pytest

from kiara.ingestion.imap_scanner import IMAPScanner


def _setup_uid_conn(mock_conn, mails, uid_start=1):
    """Konfiguriert mock_conn fuer UID-basiertes IMAP-Scan-Protokoll."""
    mock_conn.select.return_value = ("OK", [str(len(mails)).encode()])
    mock_conn.response.return_value = ("UIDVALIDITY", [b"12345"])

    uid_list = b" ".join(str(i + uid_start).encode() for i in range(len(mails)))

    def uid_dispatch(cmd, *args):
        if cmd == "search":
            return ("OK", [uid_list])
        elif cmd == "fetch":
            results = []
            uid_set = args[0] if args else b"1"
            uids_requested = uid_set.split(b",") if isinstance(uid_set, bytes) else [b"1"]
            for u in uids_requested:
                uid_int = int(u) - uid_start
                if 0 <= uid_int < len(mails):
                    mail_bytes = mails[uid_int].as_bytes()
                    header = u + b" (UID " + u + b" RFC822 {" + str(len(mail_bytes)).encode() + b"}"
                    results.append((header, mail_bytes))
                    results.append(b")")
            return ("OK", results)
        return ("OK", [b""])

    mock_conn.uid.side_effect = uid_dispatch


class TestDecodeHeader:
    """MIME-Header-Dekodierung."""

    def test_decode_header_utf8(self):
        encoded = "=?UTF-8?Q?Rechnung_f=C3=BCr_M=C3=A4rz?="
        result = IMAPScanner._decode_header(encoded)
        assert "Rechnung" in result
        assert "rz" in result  # Maerz

    def test_decode_header_plain(self):
        result = IMAPScanner._decode_header("Simple Subject")
        assert result == "Simple Subject"

    def test_decode_header_empty(self):
        assert IMAPScanner._decode_header("") == ""
        assert IMAPScanner._decode_header(None) == ""

    def test_decode_header_base64(self):
        encoded = "=?UTF-8?B?SMO2ZmxpY2hlIEdyw7zDn2U=?="
        result = IMAPScanner._decode_header(encoded)
        assert "Gr" in result


class TestExtractBody:
    """Body-Extraktion aus E-Mail-Nachrichten."""

    def test_extract_body_plain(self):
        msg = MIMEText("Dies ist der Nachrichtentext.", "plain", "utf-8")
        result = IMAPScanner._extract_body(msg)
        assert result == "Dies ist der Nachrichtentext."

    def test_extract_body_html_fallback(self):
        msg = MIMEMultipart()
        html_part = MIMEText("<p>HTML <b>Inhalt</b></p>", "html", "utf-8")
        msg.attach(html_part)
        result = IMAPScanner._extract_body(msg)
        assert "HTML" in result
        assert "Inhalt" in result
        assert "<p>" not in result  # Tags entfernt
        assert "<b>" not in result

    def test_extract_body_multipart_plain_preferred(self):
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("Klartext-Version", "plain", "utf-8"))
        msg.attach(MIMEText("<p>HTML-Version</p>", "html", "utf-8"))
        result = IMAPScanner._extract_body(msg)
        assert result == "Klartext-Version"

    def test_extract_body_empty(self):
        msg = MIMEMultipart()
        result = IMAPScanner._extract_body(msg)
        assert result == ""


class TestSafeFilename:
    """Dateinamen-Bereinigung."""

    def test_safe_filename_special_chars(self):
        result = IMAPScanner._safe_filename("Re: Rechnung #123/2024 (wichtig!)")
        assert "/" not in result
        assert "#" not in result
        assert "!" not in result
        assert "Rechnung" in result

    def test_safe_filename_empty(self):
        assert IMAPScanner._safe_filename("") == "unnamed"
        assert IMAPScanner._safe_filename("!!!") == "unnamed"

    def test_safe_filename_truncate(self):
        long_name = "A" * 200
        result = IMAPScanner._safe_filename(long_name)
        assert len(result) <= 100


class TestFolderKey:
    """Kollisionsfreie IMAP-Ordner-Keys."""

    def test_folder_key_separates_slash_from_underscore(self):
        nested = IMAPScanner._folder_key("A/B")
        flat = IMAPScanner._folder_key("A_B")

        assert nested != flat
        assert "/" not in nested
        assert "/" not in flat


class TestDiscoverFolders:
    """Wildcard-LIST-Discovery."""

    def test_list_status_error_raises(self):
        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "u",
                "imap_password": "p",
                "imap_folders": "*",
            }
        )
        conn = MagicMock()
        conn.list.return_value = ("NO", [])

        with pytest.raises(RuntimeError, match="IMAP LIST fehlgeschlagen"):
            scanner._discover_folders(conn)

    def test_empty_list_payload_raises(self):
        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "u",
                "imap_password": "p",
                "imap_folders": "*",
            }
        )
        conn = MagicMock()
        conn.list.return_value = ("OK", [])

        with pytest.raises(RuntimeError, match="IMAP LIST fehlgeschlagen"):
            scanner._discover_folders(conn)


class TestScanWritesFiles:
    """IMAP-Scan schreibt Dateien korrekt."""

    def _make_mock_mail(
        self,
        subject="Test",
        body="Body text",
        sender="test@example.com",
        date="Mon, 01 Jan 2024 10:00:00 +0100",
    ):
        msg = MIMEMultipart()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = "empfaenger@firma.de"
        msg["Date"] = date
        msg.attach(MIMEText(body, "plain", "utf-8"))
        return msg

    @patch.object(IMAPScanner, "connect")
    def test_scan_writes_files(self, mock_connect):
        """Mock-IMAP liefert 2 Mails -> 2 Textdateien."""
        mail1 = self._make_mock_mail("Test Mail 1", "Body 1")
        mail2 = self._make_mock_mail("Test Mail 2", "Body 2")

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, [mail1, mail2])

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
            assert len(files) == 2
            for f in files:
                assert os.path.exists(f)
                content = open(f, encoding="utf-8").read()
                assert "Betreff:" in content
                assert "Von:" in content
                assert "Body" in content

    @patch.object(IMAPScanner, "connect")
    def test_scan_with_attachment(self, mock_connect):
        """Mail mit Anhang -> Textdatei + Anhangsdatei."""
        msg = MIMEMultipart()
        msg["Subject"] = "Mail mit Anhang"
        msg["From"] = "test@example.com"
        msg["To"] = "user@firma.de"
        msg["Date"] = "Mon, 15 Jan 2024 12:00:00 +0100"
        msg.attach(MIMEText("Siehe Anhang.", "plain", "utf-8"))

        # PDF-Anhang simulieren
        attachment = MIMEBase("application", "pdf")
        attachment.set_payload(b"%PDF-1.4 fake content")
        encoders.encode_base64(attachment)
        attachment.add_header("Content-Disposition", "attachment", filename="rechnung.pdf")
        msg.attach(attachment)

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, [msg])

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_download_attachments": True,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)
            assert len(files) == 2  # 1 Mail-Text + 1 Anhang
            # Pruefen ob der Anhang existiert
            attachment_files = [f for f in files if "attachments" in f]
            assert len(attachment_files) == 1
            assert "rechnung" in attachment_files[0].lower()

    @patch.object(IMAPScanner, "connect")
    def test_scan_keeps_colliding_folder_names_separate(self, mock_connect):
        """A/B und A_B duerfen weder Pfad noch UID-Registry teilen."""
        mail = self._make_mock_mail("Same Subject", "Same Body")

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, [mail])

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_folders": "A/B,A_B",
                "imap_download_attachments": False,
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            files = scanner.scan(tmpdir)

            assert len(files) == 2
            assert len(set(files)) == 2

            folder_dirs = {os.path.basename(os.path.dirname(path)) for path in files}
            assert folder_dirs == {
                IMAPScanner._folder_key("A/B"),
                IMAPScanner._folder_key("A_B"),
            }

            registry_path = os.path.join(tmpdir, ".uid_registry.json")
            with open(registry_path, encoding="utf-8") as f:
                registry = json.load(f)
            assert set(registry) == folder_dirs

    @patch.object(IMAPScanner, "connect")
    def test_cancelled_scan_writes_no_success_marker_or_registry(self, mock_connect):
        """Cancel darf einen unvollstaendigen IMAP-Scan nicht als frisch markieren."""
        mail = self._make_mock_mail("Cancelled", "Body")

        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        _setup_uid_conn(mock_conn, [mail])

        scanner = IMAPScanner(
            {
                "imap_server": "mail.test.local",
                "imap_username": "user",
                "imap_password": "pass",
                "imap_folders": "INBOX",
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(InterruptedError, match="abgebrochen"):
                scanner.scan(tmpdir, control_callback=lambda: "cancel")

            assert not os.path.exists(os.path.join(tmpdir, ".scan_done"))
            assert not os.path.exists(os.path.join(tmpdir, ".uid_registry.json"))


class TestConnectionFailure:
    """Fehlerhafter Server -> graceful Error."""

    def test_connection_failure(self):
        scanner = IMAPScanner(
            {
                "imap_server": "nonexistent.server.local",
                "imap_port": 993,
                "imap_username": "user",
                "imap_password": "pass",
            }
        )
        result = scanner.test_connection()
        assert result["success"] is False
        assert "message" in result
