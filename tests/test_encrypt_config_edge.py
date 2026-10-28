"""Edge-Case-Tests fuer _encrypt_config() und _decrypt_config() (Phase 11.1).

Testet Doppelverschluesselung, maskierte Werte, leere Passwoerter,
mehrere Secret-Felder und Immutabilitaet.
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.web.api_datasource import _decrypt_config, _encrypt_config


def _make_connector_cls(secret_fields_list):
    """Erzeugt eine Mock-Connector-Klasse mit gegebenen secret_fields."""
    cls = MagicMock()
    cls.secret_fields.return_value = secret_fields_list
    return cls


# ---------------------------------------------------------------------------
# 3.1 _encrypt_config() — Doppelverschluesselung
# ---------------------------------------------------------------------------


class TestEncryptConfig:
    @patch("kiara.shared.crypto.is_encrypted", return_value=True)
    @patch("kiara.shared.crypto.encrypt")
    def test_encrypt_already_encrypted_value_not_double_encrypted(self, mock_encrypt, mock_is_enc):
        """Bereits verschluesselt: encrypt() darf NICHT aufgerufen werden."""
        config = {"password": "gAAAAABabc123"}
        cls = _make_connector_cls(["password"])
        result = _encrypt_config(config, cls)
        assert result["password"] == "gAAAAABabc123"
        mock_encrypt.assert_not_called()

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt")
    def test_encrypt_masked_value_preserves_existing(self, mock_encrypt, mock_is_enc):
        """Maskiertes Passwort: alter verschluesselter Wert beibehalten."""
        config = {"password": "********"}
        existing_config = {"password": "gAAAAABoriginal"}
        cls = _make_connector_cls(["password"])
        result = _encrypt_config(config, cls, existing_config)
        assert result["password"] == "gAAAAABoriginal"
        mock_encrypt.assert_not_called()

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt")
    def test_encrypt_masked_value_no_existing_config(self, mock_encrypt, mock_is_enc):
        """Maskiertes Passwort ohne existing_config: ValueError statt Silent Data Loss."""
        config = {"password": "********"}
        cls = _make_connector_cls(["password"])
        with pytest.raises(ValueError, match="PASSWORD_MASK"):
            _encrypt_config(config, cls, existing_config=None)
        mock_encrypt.assert_not_called()

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt")
    def test_encrypt_empty_password(self, mock_encrypt, mock_is_enc):
        """Leeres Passwort: nicht verschluesseln."""
        config = {"password": ""}
        cls = _make_connector_cls(["password"])
        result = _encrypt_config(config, cls)
        assert result["password"] == ""
        mock_encrypt.assert_not_called()

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", return_value="gAAAAABencrypted")
    def test_encrypt_new_plaintext(self, mock_encrypt, mock_is_enc):
        """Neuer Klartext: encrypt() wird aufgerufen."""
        config = {"password": "geheim123"}
        cls = _make_connector_cls(["password"])
        result = _encrypt_config(config, cls)
        assert result["password"] == "gAAAAABencrypted"
        mock_encrypt.assert_called_once_with("geheim123")

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt", return_value="gAAAAABenc_dsn")
    def test_encrypt_multiple_secret_fields(self, mock_encrypt, mock_is_enc):
        """SQL-Connector: nur dsn wird verschluesselt, query bleibt."""
        config = {"dsn": "pg://...", "query": "SELECT 1"}
        cls = _make_connector_cls(["dsn"])
        result = _encrypt_config(config, cls)
        assert result["dsn"] == "gAAAAABenc_dsn"
        assert result["query"] == "SELECT 1"
        mock_encrypt.assert_called_once_with("pg://...")

    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.shared.crypto.encrypt")
    def test_encrypt_unknown_secret_field_missing_in_config(self, mock_encrypt, mock_is_enc):
        """Secret-Feld fehlt in Config: leerer String, kein KeyError."""
        config = {"server": "mail.de"}  # kein "password" Key
        cls = _make_connector_cls(["password"])
        result = _encrypt_config(config, cls)
        assert result["password"] == ""
        mock_encrypt.assert_not_called()


# ---------------------------------------------------------------------------
# 3.2 _decrypt_config() — Entschluesselung
# ---------------------------------------------------------------------------


class TestDecryptConfig:
    @patch("kiara.web.api_datasource.is_encrypted", return_value=True)
    @patch("kiara.web.api_datasource.decrypt", return_value="klartext123")
    def test_decrypt_encrypted_value(self, mock_decrypt, mock_is_enc):
        """Verschluesselter Wert: decrypt() wird aufgerufen."""
        config = {"password": "gAAAAABabc123"}
        cls = _make_connector_cls(["password"])
        result = _decrypt_config(config, cls)
        assert result["password"] == "klartext123"
        mock_decrypt.assert_called_once_with("gAAAAABabc123")

    @patch("kiara.web.api_datasource.is_encrypted", return_value=False)
    @patch("kiara.web.api_datasource.decrypt")
    def test_decrypt_plaintext_value_untouched(self, mock_decrypt, mock_is_enc):
        """Klartext: decrypt() wird NICHT aufgerufen."""
        config = {"password": "klartext"}
        cls = _make_connector_cls(["password"])
        result = _decrypt_config(config, cls)
        assert result["password"] == "klartext"
        mock_decrypt.assert_not_called()

    @patch("kiara.web.api_datasource.is_encrypted", return_value=False)
    @patch("kiara.web.api_datasource.decrypt")
    def test_decrypt_empty_value(self, mock_decrypt, mock_is_enc):
        """Leerer Wert: decrypt() wird NICHT aufgerufen."""
        config = {"password": ""}
        cls = _make_connector_cls(["password"])
        result = _decrypt_config(config, cls)
        assert result["password"] == ""
        mock_decrypt.assert_not_called()

    @patch("kiara.web.api_datasource.is_encrypted", return_value=True)
    @patch("kiara.web.api_datasource.decrypt", return_value="decrypted")
    def test_decrypt_returns_copy(self, mock_decrypt, mock_is_enc):
        """Original-Config darf nicht mutiert werden."""
        original = {"password": "gAAAAABx"}
        cls = _make_connector_cls(["password"])
        result = _decrypt_config(original, cls)
        assert result["password"] == "decrypted"
        assert original["password"] == "gAAAAABx"
