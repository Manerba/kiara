"""Tests fuer Sprint 08: MCP Admin-Funktionen.

Testet mcp.status, .toggle, .keys.list, .keys.create, .keys.delete, .keys.toggle — 8 Unit-Tests.
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# mcp.status
# ---------------------------------------------------------------------------


class TestMCPStatus:
    @patch("asyncio.create_subprocess_exec")
    @patch("kiara.config.load_config")
    def test_mcp_status(self, mock_config, mock_exec):
        """Test 22: Status-Abfrage (enabled/disabled)."""
        mock_config.return_value = {"web": {"mcp_enabled": True, "mcp_port": 8504}}

        # Mock systemctl calls
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"active", b""))
        mock_exec.return_value = mock_proc

        from kiara.admin.functions.mcp import mcp_status

        result = _run(mcp_status(MOCK_USER))

        assert result.success is True
        assert result.data["config_enabled"] is True
        assert result.data["port"] == 8504


# ---------------------------------------------------------------------------
# mcp.toggle
# ---------------------------------------------------------------------------


class TestMCPToggle:
    @patch("asyncio.create_subprocess_exec")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_mcp_toggle_enable(self, mock_config, mock_save, mock_sleep, mock_exec):
        """Test 23: MCP aktivieren mit Confirmation."""
        mock_config.return_value = {"web": {}}

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"active", b""))
        mock_exec.return_value = mock_proc

        from kiara.admin.functions.mcp import mcp_toggle

        result = _run(mcp_toggle(MOCK_USER, enabled=True))

        assert result.success is True
        assert result.data["enabled"] is True
        assert "aktiviert" in result.data["message"]

    @patch("asyncio.create_subprocess_exec")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_mcp_toggle_disable(self, mock_config, mock_save, mock_sleep, mock_exec):
        """Test 24: MCP deaktivieren."""
        mock_config.return_value = {"web": {"mcp_enabled": True}}

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"inactive", b""))
        mock_exec.return_value = mock_proc

        from kiara.admin.functions.mcp import mcp_toggle

        result = _run(mcp_toggle(MOCK_USER, enabled=False))

        assert result.success is True
        assert result.data["enabled"] is False
        assert "deaktiviert" in result.data["message"]

    @patch("asyncio.create_subprocess_exec")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_mcp_toggle_disable_keeps_service_active_when_retrieval_enabled(
        self, mock_config, mock_save, mock_sleep, mock_exec
    ):
        """mcp.toggle(false) stoppt kiara-mcp nicht, wenn REST-Retrieval aktiv ist."""
        mock_config.return_value = {"web": {"mcp_enabled": True, "mcp_retrieval_enabled": True}}

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"active", b""))
        mock_exec.return_value = mock_proc

        from kiara.admin.functions.mcp import mcp_toggle

        result = _run(mcp_toggle(MOCK_USER, enabled=False))

        assert result.success is True
        assert result.data["mcp_endpoint_enabled"] is False
        assert result.data["retrieval_enabled"] is True
        assert result.data["service_should_run"] is True
        assert result.data["service_status"] == "active"
        first_call = mock_exec.call_args_list[0].args
        assert first_call[:4] == ("sudo", "systemctl", "restart", "kiara-mcp.service")

    @patch("asyncio.create_subprocess_exec")
    @patch("asyncio.sleep", new_callable=AsyncMock)
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_mcp_toggle_fails_on_service_status_mismatch(
        self,
        mock_config,
        mock_save,
        mock_sleep,
        mock_exec,
    ):
        """systemctl OK, aber finaler Status falsch -> fail + Config-Rollback."""
        mock_config.return_value = {"web": {"mcp_enabled": False}}

        action_proc = AsyncMock()
        action_proc.returncode = 0
        action_proc.communicate = AsyncMock(return_value=(b"", b""))

        status_proc = AsyncMock()
        status_proc.returncode = 3
        status_proc.communicate = AsyncMock(return_value=(b"failed", b""))

        mock_exec.side_effect = [action_proc] + [status_proc] * 10

        from kiara.admin.functions.mcp import mcp_toggle

        result = _run(mcp_toggle(MOCK_USER, enabled=True))

        assert result.success is False
        assert "inkonsistent" in result.message
        assert result.data["enabled"] is False
        assert result.data["requested_enabled"] is True
        assert result.data["service_status"] == "failed"
        assert mock_config.return_value["web"]["mcp_enabled"] is False
        assert mock_save.call_count == 2


# ---------------------------------------------------------------------------
# mcp.keys.list
# ---------------------------------------------------------------------------


class TestMCPKeysList:
    @patch("kiara.shared.db.db_query")
    def test_mcp_keys_list(self, mock_db_query):
        """Test 25: Key-Liste abrufbar."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_row = {
            "id": 1,
            "name": "test-key",
            "key_prefix": "orag_abc...",
            "is_active": True,
            "expires_at": None,
            "last_used_at": None,
            "created_at": "2026-03-28",
            "user_id": 1,
            "username": "admin",
        }
        mock_ctx.execute.return_value.fetchall.return_value = [mock_row]
        mock_db_query.return_value = mock_ctx

        from kiara.admin.functions.mcp import mcp_keys_list

        result = _run(mcp_keys_list(MOCK_USER))

        assert result.success is True
        assert len(result.data["data"]) == 1
        assert result.data["data"][0]["name"] == "test-key"


# ---------------------------------------------------------------------------
# mcp.keys.create
# ---------------------------------------------------------------------------


class TestMCPKeysCreate:
    @patch("kiara.repositories.api_key.ApiKeyRepository.create")
    @patch("kiara.repositories.user.UserRepository.get_by_id")
    def test_mcp_keys_create(self, mock_user, mock_create):
        """Test 26: Key generiert, full_key zurueckgegeben."""
        mock_user.return_value = {"id": 1, "username": "admin"}

        from kiara.admin.functions.mcp import mcp_keys_create

        result = _run(mcp_keys_create(MOCK_USER, name="Test-Key", user_id=1))

        assert result.success is True
        assert result.data["key"].startswith("orag_")
        assert "prefix" in result.data
        mock_create.assert_called_once()

    @patch("kiara.repositories.user.UserRepository.get_by_id")
    def test_mcp_keys_create_invalid_user(self, mock_user):
        """Test 27: Ungueltige user_id → Fehler."""
        mock_user.return_value = None

        from kiara.admin.functions.mcp import mcp_keys_create

        result = _run(mcp_keys_create(MOCK_USER, name="Test-Key", user_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# mcp.keys.delete
# ---------------------------------------------------------------------------


class TestMCPKeysDelete:
    @patch("kiara.repositories.api_key.ApiKeyRepository.delete")
    @patch("kiara.repositories.api_key.ApiKeyRepository.exists")
    def test_mcp_keys_delete(self, mock_exists, mock_delete):
        """Test 28: Key geloescht (Hard-Delete) mit Confirmation."""
        mock_exists.return_value = True

        from kiara.admin.functions.mcp import mcp_keys_delete

        result = _run(mcp_keys_delete(MOCK_USER, key_id=5))

        assert result.success is True
        assert "gelöscht" in result.message
        mock_delete.assert_called_once_with(5)


# ---------------------------------------------------------------------------
# mcp.keys.toggle
# ---------------------------------------------------------------------------


class TestMCPKeysToggle:
    @patch("kiara.repositories.api_key.ApiKeyRepository.update")
    @patch("kiara.repositories.api_key.ApiKeyRepository.get_by_id")
    def test_mcp_keys_toggle(self, mock_get, mock_update):
        """Test 29: is_active Toggle mit Confirmation."""
        mock_get.return_value = {"id": 3, "is_active": True, "name": "key3"}

        from kiara.admin.functions.mcp import mcp_keys_toggle

        result = _run(mcp_keys_toggle(MOCK_USER, key_id=3))

        assert result.success is True
        assert result.data["is_active"] is False  # Toggled from 1 to 0
        mock_update.assert_called_once_with(3, is_active=False)


# ---------------------------------------------------------------------------
# MCP Toggle — Config-Error
# ---------------------------------------------------------------------------


class TestMCPToggleConfigError:
    @patch("kiara.config.save_config")
    @patch("kiara.config.load_config")
    def test_mcp_toggle_config_save_fails(self, mock_config, mock_save):
        """save_config() PermissionError → AdminResult.fail, kein systemctl."""
        mock_config.return_value = {"web": {}}
        mock_save.side_effect = PermissionError("Permission denied")

        from kiara.admin.functions.mcp import mcp_toggle

        result = _run(mcp_toggle(MOCK_USER, enabled=True))

        assert result.success is False
        assert "Config-Fehler" in result.message


def test_mcp_toggle_sudoers_are_versioned_for_admin_chat():
    root = Path(__file__).resolve().parent.parent
    deploy = (root / "deploy.sh").read_text(encoding="utf-8")
    postinst = (root / "debian" / "postinst").read_text(encoding="utf-8")

    assert "/usr/bin/systemctl restart kiara-mcp.service" in deploy
    assert "/usr/bin/systemctl restart kiara-mcp.service" in postinst


def test_prerm_does_not_disable_mcp_service_on_upgrade():
    root = Path(__file__).resolve().parent.parent
    prerm = (root / "debian" / "prerm").read_text(encoding="utf-8")
    upgrade_block = prerm.split("upgrade)", 1)[1].split(";;", 1)[0]

    assert "systemctl stop kiara-mcp.service" in upgrade_block
    assert "systemctl disable kiara-mcp.service" not in upgrade_block


# ---------------------------------------------------------------------------
# MCP Keys — Not-Found Pfade
# ---------------------------------------------------------------------------


class TestMCPKeysNotFound:
    @patch("kiara.repositories.api_key.ApiKeyRepository.exists")
    def test_mcp_keys_delete_not_found(self, mock_exists):
        """Nicht-existierender Key → AdminResult.fail."""
        mock_exists.return_value = False

        from kiara.admin.functions.mcp import mcp_keys_delete

        result = _run(mcp_keys_delete(MOCK_USER, key_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message

    @patch("kiara.repositories.api_key.ApiKeyRepository.get_by_id")
    def test_mcp_keys_toggle_not_found(self, mock_get):
        """Nicht-existierender Key → AdminResult.fail."""
        mock_get.return_value = None

        from kiara.admin.functions.mcp import mcp_keys_toggle

        result = _run(mcp_keys_toggle(MOCK_USER, key_id=999))

        assert result.success is False
        assert "nicht gefunden" in result.message
