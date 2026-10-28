"""Tests fuer Sprint 05: Auto-Router Write-Integration.

Testet HTTP-Endpunkte via Auto-Router: POST/PUT/DELETE gehen durch
_create_endpoint() und rufen die Admin-Funktionen korrekt auf.
Tests 32-35 aus dem Sprint-05-Testplan.
"""

import asyncio
import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest
from starlette.datastructures import QueryParams

from kiara.admin.function import AdminFunction
from kiara.admin.registry import load_admin_functions
from kiara.admin.result import AdminResult
from kiara.admin.router import _coerce_params, _create_endpoint

load_admin_functions()


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


def _make_request(method="GET", path_params=None, query_params=None, body=None, headers=None):
    """Erstellt ein Mock-Request-Objekt fuer den Auto-Router."""
    request = MagicMock()
    request.path_params = path_params or {}
    request.query_params = QueryParams(query_params or {})
    request.headers = headers or {}

    if body is not None:

        async def _json():
            return body

        request.json = _json
    else:

        async def _json():
            raise ValueError("No body")

        request.json = _json

    return request


def _confirmation_headers(response):
    """Liest den serverseitigen Confirmation-Token aus einer 428-Antwort."""
    assert response.status_code == 428
    data = json.loads(response.body)
    confirmation = data["confirmation"]
    return {
        "x-kiara-confirm-token": confirmation["confirm_token"],
        "x-kiara-confirm-string": confirmation["confirm_string"],
    }


# ---------------------------------------------------------------------------
# Test 32: POST create via Auto-Router
# ---------------------------------------------------------------------------


class TestPostCreateViaAutoRouter:
    @patch("kiara.web.config_changelog.log_config_change")
    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.create")
    @patch("kiara.config.load_config")
    @patch("kiara.connectors.registry.get")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_post_create_via_autorouter(
        self,
        mock_ensure,
        mock_get_connector,
        mock_config,
        mock_create,
        mock_refresh,
        mock_changelog,
    ):
        """POST /api/admin/datasources → datasource.create via Auto-Router.

        Verifiziert: Body-Parameter werden korrekt geparsed,
        raw_response=True gibt data direkt zurueck.
        """
        # Connector-Mock
        mock_cls = MagicMock()
        mock_cls.secret_fields.return_value = []
        mock_get_connector.return_value = mock_cls
        mock_config.return_value = {"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}}
        mock_create.return_value = 42

        # AdminFunction aus Registry holen
        from kiara.admin.registry import admin_registry

        func = admin_registry.get("datasource.create")
        assert func is not None

        endpoint = _create_endpoint(func)
        request = _make_request(
            method="POST",
            body={
                "name": "AutoRouter-Test",
                "type": "local",
                "connector_config": {"path": "/tmp/test"},
            },
        )

        result = _run(endpoint(request, user=MOCK_USER))

        # raw_response=True → data direkt (nicht AdminResult-Wrapper)
        assert isinstance(result, dict)
        assert result["id"] == 42
        assert result["message"] == "Datenquelle erstellt"


# ---------------------------------------------------------------------------
# Test 33: DELETE raw_response Format
# ---------------------------------------------------------------------------


class TestDeleteViaAutoRouterRawResponse:
    @patch("kiara.web.config_changelog.log_config_change")
    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_via_autorouter_raw_response(
        self, mock_config, mock_get, mock_cascade, mock_cache, mock_changelog
    ):
        """DELETE /api/admin/datasources/{ds_id} → raw_response Format.

        Verifiziert: raw_response=True liefert {"message": "..."} — das
        Frontend-Format, das der alte Endpunkt auch lieferte.
        """
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
        }
        mock_get.return_value = {"id": 10, "name": "TestDS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch(
                "kiara.retrieval.colbert.ColBERTStore.list_colbert_collections",
                return_value=[],
            ),
            patch("kiara.ingestion._file_utils.ChunkDatasources"),
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch"),
            patch("kiara.repositories.file_hash.FileHashRepository"),
            patch("kiara.ingestion._file_utils.HashStore"),
            patch("kiara.web.cron_manager.remove_schedule"),
        ):
            mock_vs.list_collections.return_value = []

            from kiara.admin.registry import admin_registry

            func = admin_registry.get("datasource.delete")
            endpoint = _create_endpoint(func)
            request = _make_request(method="DELETE", path_params={"ds_id": "10"})
            first = _run(endpoint(request, user=MOCK_USER))
            headers = _confirmation_headers(first)

            request = _make_request(method="DELETE", path_params={"ds_id": "10"}, headers=headers)
            result = _run(endpoint(request, user=MOCK_USER))

        # raw_response=True → {"message": "..."} direkt
        assert isinstance(result, dict)
        assert result["message"] == "Datenquelle geloescht"


# ---------------------------------------------------------------------------
# Test 34: Changelog source="gui" bei REST-Aufruf
# ---------------------------------------------------------------------------


class TestAutoRouterChangelogSourceGui:
    @patch("kiara.admin.router.log_config_change")
    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.repositories.job.IngestionJobRepository.cleanup_old")
    def test_autorouter_changelog_source_gui(self, mock_cleanup, mock_refresh, mock_changelog):
        """REST-Aufruf setzt Changelog source='gui' (nicht 'chat').

        Verifiziert: Der Auto-Router ruft log_config_change mit source='gui'
        auf — das unterscheidet REST-/GUI-Aktionen von Chat-Aktionen.
        """
        mock_cleanup.return_value = 3

        from kiara.admin.registry import admin_registry

        func = admin_registry.get("jobs.cleanup_history")
        endpoint = _create_endpoint(func)
        request = _make_request(method="DELETE", query_params={"days": "30"})
        first = _run(endpoint(request, user=MOCK_USER))
        headers = _confirmation_headers(first)

        request = _make_request(method="DELETE", query_params={"days": "30"}, headers=headers)
        result = _run(endpoint(request, user=MOCK_USER))

        # Changelog-Aufruf pruefen
        mock_changelog.assert_called_once()
        call_kwargs = mock_changelog.call_args[1]
        assert call_kwargs["source"] == "gui"
        assert call_kwargs["area"] == "jobs"
        assert call_kwargs["action"] == "delete"


# ---------------------------------------------------------------------------
# Test 35: Confirmation-Gate via Auto-Router
# ---------------------------------------------------------------------------


class TestConfirmationGateViaAutoRouter:
    @pytest.mark.parametrize("action", ["start", "stop", "restart"])
    def test_service_action_confirms_before_execution_and_audits(self, action):
        from kiara.admin.registry import admin_registry

        endpoint = _create_endpoint(admin_registry.get(f"services.{action}"))
        params = {"service": "kiara-worker.service"}
        with (
            patch(
                "kiara.admin.functions.services.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ) as run,
            patch("kiara.admin.router.log_config_change") as audit,
        ):
            first = _run(endpoint(_make_request(method="POST", path_params=params), user=MOCK_USER))
            headers = _confirmation_headers(first)
            run.assert_not_called()
            audit.assert_not_called()

            response = _run(
                endpoint(
                    _make_request(method="POST", path_params=params, headers=headers),
                    user=MOCK_USER,
                )
            )
            assert response["success"] is True
            assert response["data"]["action"] == action
            run.assert_called_once()
            audit.assert_called_once()
            assert audit.call_args.kwargs["action"] == action
            assert audit.call_args.kwargs["target_id"] == "kiara-worker.service"

    def test_confirmation_gate_blocks_until_token_roundtrip(self):
        """REST-Aufruf fuehrt requires_confirmation nicht ohne Server-Token aus."""
        from kiara.admin.registry import admin_registry

        # Alle Funktionen mit requires_confirmation
        confirmed_funcs = [
            f for f in admin_registry.list_all() if f.requires_confirmation and f.http_path
        ]

        # Stichprobe: delete, update, assign_groups, cleanup_history
        confirmed_names = {f.name for f in confirmed_funcs}
        assert "datasource.delete" in confirmed_names
        assert "jobs.cleanup_history" in confirmed_names
        assert "jobs.schedule.delete" in confirmed_names
        assert "datasource.update" not in confirmed_names
        assert "users.update" not in confirmed_names
        assert "ki.templates.update" not in confirmed_names
        assert "tools.permissions.update" not in confirmed_names

        calls = []

        async def _mock_delete(user, ds_id):
            calls.append(ds_id)
            return AdminResult.ok("geloescht", data={"message": "geloescht"})

        func = AdminFunction(
            name="test.confirmed",
            category="test",
            description="Test",
            help="Test",
            examples=[],
            parameters={
                "type": "object",
                "properties": {
                    "ds_id": {"type": "integer"},
                },
                "required": ["ds_id"],
            },
            returns="test",
            requires_confirmation=True,
            confirm_template="ds {ds_id} loeschen",
            pk_param="ds_id",
            changelog_area=None,
            changelog_action=None,
            http_method="DELETE",
            http_path="/api/test/{ds_id}",
            fn=_mock_delete,
            raw_response=True,
        )

        endpoint = _create_endpoint(func)
        request = _make_request(method="DELETE", path_params={"ds_id": "5"})

        first = _run(endpoint(request, user=MOCK_USER))
        assert first.status_code == 428
        data = json.loads(first.body)
        assert data["requires_confirmation"] is True
        assert data["confirmation"]["confirm_string"] == "ds 5 loeschen"
        assert calls == []

        headers = _confirmation_headers(first)
        confirmed_request = _make_request(
            method="DELETE", path_params={"ds_id": "5"}, headers=headers
        )
        result = _run(endpoint(confirmed_request, user=MOCK_USER))
        assert result == {"message": "geloescht"}
        assert calls == [5]

    def test_confirmation_token_is_bound_to_params(self):
        """Token fuer andere Parameter wird fail-closed abgelehnt."""
        calls = []

        async def _mock_delete(user, ds_id):
            calls.append(ds_id)
            return AdminResult.ok("geloescht", data={"message": "geloescht"})

        func = AdminFunction(
            name="test.confirmed.bound",
            category="test",
            description="Test",
            help="Test",
            examples=[],
            parameters={
                "type": "object",
                "properties": {
                    "ds_id": {"type": "integer"},
                },
                "required": ["ds_id"],
            },
            returns="test",
            requires_confirmation=True,
            confirm_template="ds {ds_id} loeschen",
            pk_param="ds_id",
            changelog_area=None,
            changelog_action=None,
            http_method="DELETE",
            http_path="/api/test/{ds_id}",
            fn=_mock_delete,
            raw_response=True,
        )

        endpoint = _create_endpoint(func)
        first = _run(
            endpoint(_make_request(method="DELETE", path_params={"ds_id": "5"}), user=MOCK_USER)
        )
        headers = _confirmation_headers(first)
        headers["x-kiara-confirm-string"] = "ds 6 loeschen"

        second = _run(
            endpoint(
                _make_request(method="DELETE", path_params={"ds_id": "6"}, headers=headers),
                user=MOCK_USER,
            )
        )
        assert second.status_code == 428
        assert calls == []


# ---------------------------------------------------------------------------
# Bonus: _coerce_params fuer Write-Endpunkte
# ---------------------------------------------------------------------------


class TestCoerceParamsWrite:
    def test_coerce_ds_id_from_path(self):
        """Path-Parameter ds_id (str) → int."""
        schema = {
            "type": "object",
            "properties": {
                "ds_id": {"type": "integer"},
            },
        }
        result = _coerce_params({"ds_id": "42"}, schema)
        assert result["ds_id"] == 42
        assert isinstance(result["ds_id"], int)

    def test_coerce_days_from_query(self):
        """Query-Parameter days (str) → int."""
        schema = {
            "type": "object",
            "properties": {
                "days": {"type": "integer"},
            },
        }
        result = _coerce_params({"days": "30"}, schema)
        assert result["days"] == 30

    def test_coerce_preserves_dict_from_body(self):
        """Body-Parameter (dict) bleiben unveraendert."""
        schema = {
            "type": "object",
            "properties": {
                "connector_config": {"type": "object"},
                "name": {"type": "string"},
            },
        }
        params = {
            "connector_config": {"path": "/tmp"},
            "name": "Test",
        }
        result = _coerce_params(params, schema)
        assert result["connector_config"] == {"path": "/tmp"}
        assert result["name"] == "Test"
