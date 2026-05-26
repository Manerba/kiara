"""Roundtrip-Integrationstests fuer Connector-Config (Phase 11.1).

Simuliert den vollen Lifecycle: Erstellen, Speichern, Lesen,
Maskieren, Aktualisieren mit maskiertem/neuem Passwort.
"""

import json
from unittest.mock import patch

import pytest

from kiara.connectors.registry import ensure_loaded
from kiara.connectors.registry import get as get_connector
from kiara.web.api_datasource import (
    _decrypt_config,
    _effective_path,
    _encrypt_config,
    _mask_row,
    _source_display,
)


@pytest.fixture(autouse=True)
def _load_connectors():
    """Stellt sicher, dass alle Konnektoren geladen sind.

    Falls ein vorheriger Test _loaded=True aber _registry leer hinterlassen hat,
    wird _loaded zurueckgesetzt um einen erneuten Load zu erzwingen.
    """
    from kiara.connectors import registry as _reg

    if _reg._loaded and not _reg._registry:
        _reg._loaded = False
    ensure_loaded()


def _create_ds_db():
    """Erstellt eine DB-Verbindung mit datasources-Tabelle."""
    from kiara.shared.db import get_connection

    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS datasources (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'local',
            path TEXT DEFAULT '',
            connector_config TEXT DEFAULT '{}',
            is_active BOOLEAN DEFAULT true
        )
    """)
    # Sicherstellen, dass connector_config existiert (falls Tabelle schon ohne war)
    try:
        conn.execute(
            "ALTER TABLE datasources ADD COLUMN IF NOT EXISTS connector_config TEXT DEFAULT '{}'"
        )
    except Exception:
        pass
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# 8.1 Create → Read → Mask Roundtrip
# ---------------------------------------------------------------------------


class TestCreateAndReadBack:
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", return_value="gAAAAABenc_pw")
    def test_create_imap_ds_and_read_back(self, mock_encrypt, mock_is_enc):
        """Vollstaendiger Lifecycle: Erstellen, Speichern, Lesen, Maskieren."""
        imap_cls = get_connector("imap")

        # 1. Config erstellen und verschluesseln
        config = {
            "server": "mail.firma.de",
            "port": 993,
            "ssl": True,
            "username": "user@firma.de",
            "password": "geheim123",
            "folders": "INBOX",
            "since_days": 30,
            "download_attachments": True,
            "max_attachment_mb": 10,
        }
        encrypted_config = _encrypt_config(config, imap_cls)
        assert encrypted_config["password"] == "gAAAAABenc_pw"
        assert encrypted_config["server"] == "mail.firma.de"

        # 2. In DB speichern
        db = _create_ds_db()
        config_json = json.dumps(encrypted_config, ensure_ascii=False)
        cursor = db.execute(
            "INSERT INTO datasources (name, type, connector_config) VALUES (%s, %s, %s) RETURNING id",
            ("Test-Mail", "imap", config_json),
        )
        ds_id = cursor.fetchone()["id"]
        db.commit()

        # 3. Wieder lesen
        row = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()
        assert row is not None

        # 4. _mask_row: Passwort maskiert
        with patch("kiara.shared.crypto.is_encrypted", return_value=True):
            masked = _mask_row(row)
        assert masked["connector_config"]["password"] == "********"
        assert masked["connector_config"]["server"] == "mail.firma.de"

        # 5. _source_display
        display = _source_display(row)
        assert "user@firma.de" in display
        assert "mail.firma.de" in display

        # 6. _effective_path
        path = _effective_path(row)
        assert f"/var/lib/kiara/mail-cache/ds{ds_id}" == path

        db.close()


# ---------------------------------------------------------------------------
# 8.2 Update — Maskiertes Passwort beibehalten
# ---------------------------------------------------------------------------


class TestUpdateMaskedPassword:
    def test_update_ds_masked_password_preserved(self):
        """Update mit '********': verschluesseltes Passwort bleibt."""
        imap_cls = get_connector("imap")

        existing_config = {
            "server": "mail.de",
            "password": "gAAAAABoriginal_encrypted",
        }

        # Update-Request mit maskiertem Passwort
        update_config = {
            "server": "mail.de",
            "password": "********",
        }

        with patch("kiara.shared.crypto.is_encrypted", return_value=False):
            result = _encrypt_config(update_config, imap_cls, existing_config)

        assert result["password"] == "gAAAAABoriginal_encrypted"

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", return_value="gAAAAABnew_encrypted")
    def test_update_ds_new_password_encrypted(self, mock_encrypt, mock_is_enc):
        """Update mit neuem Passwort: encrypt() wird aufgerufen."""
        imap_cls = get_connector("imap")

        existing_config = {
            "server": "mail.de",
            "password": "gAAAAABold_encrypted",
        }

        update_config = {
            "server": "mail.de",
            "password": "neuesPasswort",
        }

        result = _encrypt_config(update_config, imap_cls, existing_config)
        assert result["password"] == "gAAAAABnew_encrypted"
        mock_encrypt.assert_called_once_with("neuesPasswort")


# ---------------------------------------------------------------------------
# 8.3 Decrypt Roundtrip
# ---------------------------------------------------------------------------


class TestDecryptRoundtrip:
    @patch("kiara.web.api_datasource.is_encrypted", return_value=True)
    @patch("kiara.web.api_datasource.decrypt", return_value="geheim123")
    def test_decrypt_after_encrypt(self, mock_decrypt, mock_is_enc):
        """Verschluesselte Config kann entschluesselt werden."""
        imap_cls = get_connector("imap")

        encrypted_config = {
            "server": "mail.de",
            "password": "gAAAAABenc",
        }

        decrypted = _decrypt_config(encrypted_config, imap_cls)
        assert decrypted["password"] == "geheim123"
        assert decrypted["server"] == "mail.de"
        mock_decrypt.assert_called_once_with("gAAAAABenc")


# ---------------------------------------------------------------------------
# 8.4 SMB Roundtrip
# ---------------------------------------------------------------------------


class TestSMBRoundtrip:
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", return_value="gAAAAABenc_smb")
    def test_create_smb_ds_roundtrip(self, mock_encrypt, mock_is_enc):
        """SMB-Datasource: Erstellen und Auslesen."""
        smb_cls = get_connector("smb")

        config = {
            "server": "192.168.1.10",
            "share": "Documents",
            "username": "admin",
            "password": "secret",
            "domain": "WORKGROUP",
            "sub_path": "Technik",
        }

        encrypted = _encrypt_config(config, smb_cls)
        assert encrypted["password"] == "gAAAAABenc_smb"
        assert encrypted["server"] == "192.168.1.10"

        db = _create_ds_db()
        config_json = json.dumps(encrypted, ensure_ascii=False)
        cursor = db.execute(
            "INSERT INTO datasources (name, type, connector_config) VALUES (%s, %s, %s) RETURNING id",
            ("SMB-Test", "smb", config_json),
        )
        ds_id = cursor.fetchone()["id"]
        db.commit()

        row = db.execute("SELECT * FROM datasources WHERE id = %s", (ds_id,)).fetchone()

        # source_display
        display = _source_display(row)
        assert "192.168.1.10" in display
        assert "Documents" in display

        # effective_path
        path = _effective_path(row)
        assert "/mnt/kiara-" in path
        assert "Technik" in path

        db.close()
