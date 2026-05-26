"""Tests fuer Sprint 08: Auto-Router-Integration.

Testet HTTP-Endpunkte via Auto-Router fuer Sprint-08-Funktionen.
Tests 50-53 aus dem Sprint-08-Testplan.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.datastructures import QueryParams

from kiara.admin.registry import load_admin_functions
from kiara.admin.router import _create_endpoint

load_admin_functions()


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


def _make_request(method="GET", path_params=None, query_params=None, body=None):
    """Erstellt ein Mock-Request-Objekt fuer den Auto-Router."""
    request = MagicMock()
    request.path_params = path_params or {}
    request.query_params = QueryParams(query_params or {})

    if body is not None:

        async def _json():
            return body

        request.json = _json
    else:

        async def _json():
            raise ValueError("No body")

        request.json = _json

    return request


# ---------------------------------------------------------------------------
# Test 50: GET /api/admin/settings via Auto-Router
# ---------------------------------------------------------------------------


class TestSettingsRoute:
    @patch("kiara.config.load_config")
    def test_settings_route(self, mock_config):
        """GET /api/admin/settings → system.settings.read via Auto-Router."""
        mock_config.return_value = {"ollama": {"url": "http://gpu-server.local:11434"}}

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("system.settings.read")
        assert func is not None
        assert func.http_method == "GET" and func.http_path == "/api/admin/settings"

        endpoint = _create_endpoint(func)
        request = _make_request(method="GET")

        with patch("kiara.admin.router.get_admin_user", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MOCK_USER
            response = _run(endpoint(request))

        assert "config" in response
        assert response["config"]["ollama"]["url"] == "http://gpu-server.local:11434"


# ---------------------------------------------------------------------------
# Test 51: GET /api/admin/backup/list via Auto-Router
# ---------------------------------------------------------------------------


class TestBackupListRoute:
    @patch("kiara.admin.services.backup_service.list_backups")
    def test_backup_list_route(self, mock_list):
        """GET /api/admin/backup/list → backup.list via Auto-Router."""
        mock_list.return_value = [{"filename": "test.tar.gz", "size_bytes": 1024}]

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("backup.list")
        assert func is not None
        assert func.http_method == "GET" and func.http_path == "/api/admin/backup/list"

        endpoint = _create_endpoint(func)
        request = _make_request(method="GET")

        with patch("kiara.admin.router.get_admin_user", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MOCK_USER
            response = _run(endpoint(request))

        assert "backups" in response
        assert len(response["backups"]) == 1


# ---------------------------------------------------------------------------
# Test 52: GET /api/admin/elasticsearch/status via Auto-Router
# ---------------------------------------------------------------------------


class TestESStatusRoute:
    @patch("kiara.admin.services.elasticsearch_service.get_es_status")
    def test_es_status_route(self, mock_status):
        """GET /api/admin/elasticsearch/status → elasticsearch.status via Auto-Router."""
        mock_status.return_value = {
            "connected": True,
            "doc_count": 100,
            "qdrant_count": 100,
            "task_running": False,
            "task_action": None,
            "task_progress": None,
        }

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("elasticsearch.status")
        assert func is not None
        assert func.http_method == "GET" and func.http_path == "/api/admin/elasticsearch/status"

        endpoint = _create_endpoint(func)
        request = _make_request(method="GET")

        with patch("kiara.admin.router.get_admin_user", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MOCK_USER
            response = _run(endpoint(request))

        assert response["connected"] is True


# ---------------------------------------------------------------------------
# Test 53: GET /api/admin/mcp/keys via Auto-Router
# ---------------------------------------------------------------------------


class TestMCPKeysRoute:
    @patch("kiara.shared.db.db_query")
    def test_mcp_keys_route(self, mock_db_query):
        """GET /api/admin/mcp/keys → mcp.keys.list via Auto-Router."""
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
        mock_ctx.__exit__ = MagicMock(return_value=False)
        mock_row = {
            "id": 1,
            "name": "key1",
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

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("mcp.keys.list")
        assert func is not None
        assert func.http_method == "GET" and func.http_path == "/api/admin/mcp/keys"

        endpoint = _create_endpoint(func)
        request = _make_request(method="GET")

        with patch("kiara.admin.router.get_admin_user", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MOCK_USER
            response = _run(endpoint(request))

        assert "data" in response
        assert len(response["data"]) == 1


# ---------------------------------------------------------------------------
# Test 54: GET /api/admin/users via Auto-Router (Sprint 1.4 MCP-Pflicht-Load)
# ---------------------------------------------------------------------------


class TestUsersListRoute:
    @patch("kiara.repositories.user.UserRepository")
    def test_users_list_route(self, mock_repo_cls):
        """GET /api/admin/users → users.list via Auto-Router."""
        mock_repo = MagicMock()
        mock_repo.list_safe.return_value = [
            {
                "id": 1,
                "username": "admin",
                "vorname": "Admin",
                "nachname": "User",
                "email": "admin@test.de",
                "is_admin": True,
                "auth_source": "local",
                "ldap_dn": None,
                "is_active": True,
                "created_at": "2026-01-01",
            },
        ]
        mock_repo_cls.return_value = mock_repo

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("users.list")
        assert func is not None
        assert func.http_method == "GET" and func.http_path == "/api/admin/users"

        endpoint = _create_endpoint(func)
        request = _make_request(method="GET")

        with patch("kiara.admin.router.get_admin_user", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MOCK_USER
            response = _run(endpoint(request))

        assert "data" in response
        assert len(response["data"]) == 1
        assert response["data"][0]["username"] == "admin"
