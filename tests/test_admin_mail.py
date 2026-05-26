"""Tests fuer Sprint 08: Mail Admin-Funktionen.

Testet mail.config.read, .update, .delete, .test, .send_report — 10 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

from kiara.shared.constants import PASSWORD_MASK


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# mail.config.read
# ---------------------------------------------------------------------------


class TestMailConfigRead:
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_config_read(self, mock_get):
        """Test 39: Config mit maskiertem Passwort."""
        mock_get.return_value = {
            "smtp_server": "smtp.firma.de",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "kiara@firma.de",
            "password": "gAAAAABencrypted...",
            "sender_address": "kiara@firma.de",
            "recipient_addresses": "admin@firma.de",
            "notify_ollama_down": 1,
            "notify_datasource_down": 1,
            "notify_index_error": 0,
            "status_report_enabled": False,
        }

        from kiara.admin.functions.mail import mail_config_read

        result = _run(mail_config_read(MOCK_USER))

        assert result.success is True
        assert result.data["config"]["password"] == PASSWORD_MASK
        assert result.data["config"]["smtp_server"] == "smtp.firma.de"

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_config_read_booleans(self, mock_get):
        """Test 40: DB-int → Python-bool Konvertierung."""
        mock_get.return_value = {
            "smtp_server": "smtp.firma.de",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "",
            "password": "",
            "sender_address": "kiara@firma.de",
            "recipient_addresses": "admin@firma.de",
            "notify_ollama_down": 1,
            "notify_datasource_down": 0,
            "notify_index_error": 1,
            "status_report_enabled": False,
        }

        from kiara.admin.functions.mail import mail_config_read

        result = _run(mail_config_read(MOCK_USER))

        assert result.data["config"]["notify_ollama_down"] is True
        assert result.data["config"]["notify_datasource_down"] is False
        assert result.data["config"]["notify_index_error"] is True
        assert result.data["config"]["status_report_enabled"] is False


# ---------------------------------------------------------------------------
# mail.config.update
# ---------------------------------------------------------------------------


class TestMailConfigUpdate:
    @patch("kiara.shared.crypto.encrypt")
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_config_update(self, mock_get, mock_set, mock_encrypt):
        """Test 41: Speichern mit Confirmation."""
        mock_get.return_value = None
        mock_encrypt.return_value = "gAAAAABnew..."

        from kiara.admin.functions.mail import mail_config_update

        result = _run(
            mail_config_update(
                MOCK_USER,
                smtp_server="smtp.firma.de",
                sender_address="kiara@firma.de",
                recipient_addresses="admin@firma.de",
                password="secret123",
            )
        )

        assert result.success is True
        assert "gespeichert" in result.message
        mock_encrypt.assert_called_once_with("secret123")

    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_config_update_mask(self, mock_get, mock_set):
        """Test 42: PASSWORD_MASK → bestehendes PW beibehalten."""
        mock_get.return_value = {
            "smtp_server": "smtp.firma.de",
            "password": "gAAAAABexisting...",
        }

        from kiara.admin.functions.mail import mail_config_update

        result = _run(
            mail_config_update(
                MOCK_USER,
                smtp_server="smtp.firma.de",
                sender_address="kiara@firma.de",
                recipient_addresses="admin@firma.de",
                password=PASSWORD_MASK,
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        assert call_args[1]["password"] == "gAAAAABexisting..."


# ---------------------------------------------------------------------------
# mail.config.delete
# ---------------------------------------------------------------------------


class TestMailConfigDelete:
    @patch("kiara.repositories.config.ConfigRepository.delete")
    def test_mail_config_delete(self, mock_delete):
        """Test 43: Config geloescht."""
        from kiara.admin.functions.mail import mail_config_delete

        result = _run(mail_config_delete(MOCK_USER))

        assert result.success is True
        assert "gelöscht" in result.message
        mock_delete.assert_called_once_with("mail_config")


# ---------------------------------------------------------------------------
# mail.test
# ---------------------------------------------------------------------------


class TestMailTest:
    @patch("kiara.web.mail_service.send_mail")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_test(self, mock_get, mock_is_enc, mock_send):
        """Test 44: Test-Mail (Mock SMTP)."""
        mock_get.return_value = {
            "smtp_server": "smtp.firma.de",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "kiara",
            "password": "cleartext",
            "sender_address": "kiara@firma.de",
            "recipient_addresses": "admin@firma.de",
        }
        mock_send.return_value = (True, "Mail gesendet")

        from kiara.admin.functions.mail import mail_test

        result = _run(mail_test(MOCK_USER))

        assert result.success is True
        assert "versendet" in result.message
        mock_send.assert_called_once()


# ---------------------------------------------------------------------------
# mail.send_report
# ---------------------------------------------------------------------------


class TestMailSendReport:
    @patch("kiara.web.mail_service.send_mail")
    @patch("kiara.web.mail_service.build_status_report")
    @patch("kiara.shared.db.db_query")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_send_report(
        self, mock_get, mock_config, mock_is_enc, mock_db_query, mock_build, mock_send
    ):
        """Test 45: Status-Report (Mock SMTP)."""
        mock_get.return_value = {
            "smtp_server": "smtp.firma.de",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "",
            "password": "",
            "sender_address": "kiara@firma.de",
            "recipient_addresses": "admin@firma.de",
        }
        mock_config.return_value = {"ollama": {"url": "http://localhost:11434"}}

        # Mock DB query for user count
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_ctx.execute.return_value.fetchone.return_value = {"cnt": 10}
        mock_db_query.return_value = mock_ctx

        mock_build.return_value = "<html>Report</html>"
        mock_send.return_value = (True, "Report gesendet")

        from kiara.admin.functions.mail import mail_send_report

        result = _run(mail_send_report(MOCK_USER))

        assert result.success is True
        assert "versendet" in result.message


# ---------------------------------------------------------------------------
# PASSWORD_MASK Edge Cases (Mail)
# ---------------------------------------------------------------------------


class TestMailPasswordMaskEdgeCases:
    @patch("kiara.repositories.config.ConfigRepository.set")
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_empty_password_new_config(self, mock_get, mock_set):
        """Leerer String bei neuer Config → leeres PW, kein Fehler."""
        mock_get.return_value = None

        from kiara.admin.functions.mail import mail_config_update

        result = _run(
            mail_config_update(
                MOCK_USER,
                smtp_server="smtp.test.de",
                sender_address="a@b.de",
                recipient_addresses="c@d.de",
                password="",
            )
        )

        assert result.success is True
        call_args = mock_set.call_args[0]
        assert call_args[1]["password"] == ""

    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mask_without_existing_fails(self, mock_get):
        """PASSWORD_MASK ohne bestehende Config → Fehler."""
        mock_get.return_value = None

        from kiara.admin.functions.mail import mail_config_update

        result = _run(
            mail_config_update(
                MOCK_USER,
                smtp_server="smtp.test.de",
                sender_address="a@b.de",
                recipient_addresses="c@d.de",
                password=PASSWORD_MASK,
            )
        )

        assert result.success is False
        assert "passwort" in result.message.lower()

    @patch("kiara.web.mail_service.send_mail")
    @patch("kiara.shared.crypto.is_encrypted", return_value=False)
    @patch("kiara.repositories.config.ConfigRepository.get")
    def test_mail_test_send_fails(self, mock_get, mock_is_enc, mock_send):
        """send_mail() Fehler → AdminResult.fail mit Detail."""
        mock_get.return_value = {
            "smtp_server": "smtp.bad.de",
            "smtp_port": 587,
            "encryption": "starttls",
            "username": "",
            "password": "",
            "sender_address": "a@b.de",
            "recipient_addresses": "c@d.de",
        }
        mock_send.return_value = (False, "Connection refused")

        from kiara.admin.functions.mail import mail_test

        result = _run(mail_test(MOCK_USER))

        assert result.success is False
        assert "Connection refused" in result.message
