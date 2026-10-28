"""Security-Tests fuer F-33/F-55: SMB Credentials-File statt Inline-Passwort.

Testet dass mount_smb() und SMBConnector.test_connection() Credentials
in temporaeren Dateien statt auf der Kommandozeile uebergeben und
dass diese Dateien korrekte Berechtigungen haben und aufgeraeumt werden.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tests: mount_smb() — Credentials-File
# ---------------------------------------------------------------------------
from kiara.shared.mount import mount_smb


class TestMountSmbCredentials:
    """F-33: mount_smb() muss Credentials-File statt Inline-Passwort verwenden."""

    @patch("kiara.shared.mount.exec_mount", return_value=(True, "OK"))
    @patch("kiara.shared.mount.os.unlink")
    @patch("kiara.shared.mount.os.chmod")
    @patch("kiara.shared.mount.os.close")
    @patch("kiara.shared.mount.os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-XXXX.cred"))
    def test_mount_smb_uses_credentials_file(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_exec_mount,
    ):
        """Der mount-Befehl muss 'credentials=' enthalten, nicht das Passwort direkt."""
        mount_smb("server", "share", "user", "P@ss,w0rd!", "DOMAIN", "/mnt/test")

        # exec_mount wurde aufgerufen
        mock_exec_mount.assert_called_once()
        cmd = mock_exec_mount.call_args[0][0]

        # Kommandozeile muss credentials=... enthalten
        options_str = " ".join(cmd)
        assert "credentials=" in options_str

        # Passwort darf NICHT in der Kommandozeile stehen
        assert "P@ss,w0rd!" not in options_str

    @patch("kiara.shared.mount.exec_mount", return_value=(True, "OK"))
    @patch("kiara.shared.mount.os.unlink")
    @patch("kiara.shared.mount.os.chmod")
    @patch("kiara.shared.mount.os.close")
    @patch("kiara.shared.mount.os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-XXXX.cred"))
    def test_mount_smb_credentials_file_has_0600(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_exec_mount,
    ):
        """Das temporaere Credentials-File muss Modus 0600 erhalten."""
        mount_smb("server", "share", "user", "pass", "", "/mnt/test")

        # chmod muss mit 0o600 aufgerufen worden sein
        mock_chmod.assert_any_call("/tmp/kiara-smb-XXXX.cred", 0o600)

    @patch("kiara.shared.mount.exec_mount", side_effect=Exception("mount error"))
    @patch("kiara.shared.mount.os.unlink")
    @patch("kiara.shared.mount.os.chmod")
    @patch("kiara.shared.mount.os.close")
    @patch("kiara.shared.mount.os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-XXXX.cred"))
    def test_mount_smb_cleans_up_credentials_on_error(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_exec_mount,
    ):
        """Credentials-File wird auch bei Fehler im finally-Block geloescht."""
        with pytest.raises(Exception, match="mount error"):
            mount_smb("server", "share", "user", "pass", "", "/mnt/test")

        mock_unlink.assert_called_with("/tmp/kiara-smb-XXXX.cred")

    @patch("kiara.shared.mount.exec_mount", return_value=(True, "OK"))
    @patch("kiara.shared.mount.os.unlink")
    @patch("kiara.shared.mount.os.chmod")
    @patch("kiara.shared.mount.os.close")
    @patch("kiara.shared.mount.os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-XXXX.cred"))
    def test_mount_smb_writes_credentials_content(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_exec_mount,
    ):
        """Credentials-File muss username, password und domain enthalten."""
        mount_smb("server", "share", "testuser", "testpass", "TESTDOMAIN", "/mnt/test")

        written = mock_write.call_args[0][1].decode()
        assert "username=testuser" in written
        assert "password=testpass" in written
        assert "domain=TESTDOMAIN" in written


# ---------------------------------------------------------------------------
# Tests: SMBConnector.test_connection() — Auth-File
# ---------------------------------------------------------------------------

from kiara.connectors.smb_connector import SMBConnector


class TestSmbTestConnectionCredentials:
    """F-55: test_connection() muss --authentication-file verwenden."""

    @patch("subprocess.run")
    @patch("os.unlink")
    @patch("os.chmod")
    @patch("os.close")
    @patch("os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-test-XXXX.cred"))
    def test_smb_test_connection_uses_auth_file(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_run,
    ):
        """smbclient-Aufruf muss --authentication-file= enthalten."""
        mock_run.return_value = MagicMock(returncode=0)
        config = {"server": "srv", "share": "sh", "username": "u", "password": "p", "domain": ""}
        row = {"id": 1}

        result = SMBConnector.test_connection(config, row)

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "--authentication-file=" in cmd_str
        # Passwort darf NICHT als eigenes Argument stehen
        assert "-U" not in cmd_str or "u%p" not in cmd_str

    @patch("subprocess.run")
    @patch("os.unlink")
    @patch("os.chmod")
    @patch("os.close")
    @patch("os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-test-XXXX.cred"))
    def test_smb_test_connection_cleans_up_auth_file(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_run,
    ):
        """Credentials-File wird im finally-Block geloescht."""
        mock_run.return_value = MagicMock(returncode=0)
        config = {"server": "srv", "share": "sh", "username": "u", "password": "p", "domain": ""}
        row = {"id": 1}

        SMBConnector.test_connection(config, row)

        mock_unlink.assert_called_with("/tmp/kiara-smb-test-XXXX.cred")

    @patch("subprocess.run", side_effect=Exception("connection failed"))
    @patch("os.unlink")
    @patch("os.chmod")
    @patch("os.close")
    @patch("os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-test-XXXX.cred"))
    def test_smb_test_connection_cleans_up_on_exception(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_run,
    ):
        """Auch bei Exception wird Credentials-File aufgeraeumt."""
        config = {"server": "srv", "share": "sh", "username": "u", "password": "p", "domain": ""}
        row = {"id": 1}

        # test_connection faengt allgemeine Exceptions und gibt dict zurueck
        result = SMBConnector.test_connection(config, row)

        assert result["success"] is False
        mock_unlink.assert_called_with("/tmp/kiara-smb-test-XXXX.cred")

    @patch("subprocess.run")
    @patch("os.unlink")
    @patch("os.chmod")
    @patch("os.close")
    @patch("os.write")
    @patch("tempfile.mkstemp", return_value=(99, "/tmp/kiara-smb-test-XXXX.cred"))
    def test_smb_test_connection_cred_file_has_0600(
        self,
        mock_mkstemp,
        mock_write,
        mock_close,
        mock_chmod,
        mock_unlink,
        mock_run,
    ):
        """Das Auth-File muss Modus 0600 erhalten."""
        mock_run.return_value = MagicMock(returncode=0)
        config = {"server": "srv", "share": "sh", "username": "u", "password": "p", "domain": ""}
        row = {"id": 1}

        SMBConnector.test_connection(config, row)

        mock_chmod.assert_any_call("/tmp/kiara-smb-test-XXXX.cred", 0o600)
