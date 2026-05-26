"""Tests fuer Sprint 04: Read-Only Registry-Funktionen.

Testet alle 31 neuen Admin-Funktionen, Framework-Erweiterungen
(raw_response, Error-Status-Codes), Auto-Router-Integration und Regressionen.
Reine Unit-Tests mit Mocking — keine DB- oder Netzwerk-Abhaengigkeiten.
"""

import sys
import types

# ldap3 ist eine optionale Abhaengigkeit und steht in der Test-Umgebung nicht
# zur Verfuegung. Stub-Modul vorab registrieren, damit der transitive Import-Pfad
# admin_app -> api_health -> health.checker -> checks_services -> shared.ldap
# nicht abbricht.
if "ldap3" not in sys.modules:
    _ldap3_stub = types.ModuleType("ldap3")
    _ldap3_stub.Server = object
    _ldap3_stub.Connection = object
    _ldap3_stub.NONE = None
    _ldap3_stub.SUBTREE = None
    _ldap3_stub.Tls = object
    sys.modules["ldap3"] = _ldap3_stub

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

from starlette.datastructures import QueryParams

from kiara.admin.function import AdminFunction
from kiara.admin.result import AdminResult
from kiara.admin.router import _coerce_params, _create_endpoint

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


def _make_func(name="test.func", raw_response=False, fn=None):
    """Erstellt eine AdminFunction fuer Tests."""
    if fn is None:

        async def fn(user: dict) -> AdminResult:
            return AdminResult.ok("ok", data={"key": "value"})

    return AdminFunction(
        name=name,
        category="test",
        description="Test-Funktion",
        help="Test-Hilfe",
        examples=["Beispiel"],
        parameters={"type": "object", "properties": {}, "required": []},
        returns="test",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method="GET",
        http_path="/api/test",
        fn=fn,
        raw_response=raw_response,
    )


MOCK_USER = {"id": 1, "username": "admin"}


def _mock_db_context(fetchone_val=None, fetchall_val=None):
    """Erstellt einen Mock fuer db_query() Context-Manager."""
    mock_cursor = MagicMock()
    mock_cursor.execute.return_value.fetchone.return_value = fetchone_val
    mock_cursor.execute.return_value.fetchall.return_value = fetchall_val or []
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


# ---------------------------------------------------------------------------
# 1. Framework-Erweiterung: raw_response + Error-Status-Codes
# ---------------------------------------------------------------------------


class TestRawResponse:
    def test_raw_response_true(self):
        """raw_response=True → Auto-Router gibt .data direkt zurueck."""
        func = _make_func(raw_response=True)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result == {"key": "value"}

    def test_raw_response_false(self):
        """raw_response=False → AdminResult-Wrapper."""
        func = _make_func(raw_response=False)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result["success"] is True
        assert result["data"] == {"key": "value"}

    def test_raw_response_on_error(self):
        """raw_response=True aber Fehler → HTTP 400 mit Error-Body."""

        async def _fail(user: dict) -> AdminResult:
            return AdminResult.fail("Etwas ging schief")

        func = _make_func(raw_response=True, fn=_fail)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result.status_code == 400
        body = json.loads(result.body)
        assert body["success"] is False

    def test_error_status_404(self):
        """'nicht gefunden' in message → HTTP 404."""

        async def _not_found(user: dict) -> AdminResult:
            return AdminResult.fail("Datenquelle nicht gefunden")

        func = _make_func(fn=_not_found)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result.status_code == 404

    def test_error_status_400(self):
        """Anderer Fehler → HTTP 400."""

        async def _bad(user: dict) -> AdminResult:
            return AdminResult.fail("Ungueltige Parameter")

        func = _make_func(fn=_bad)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result.status_code == 400


# ---------------------------------------------------------------------------
# 2. Dashboard
# ---------------------------------------------------------------------------


class TestDashboard:
    @patch("kiara.admin.functions.dashboard.get_dashboard_kpis")
    @patch("kiara.admin.functions.dashboard.load_config")
    def test_dashboard_status(self, mock_config, mock_kpis):
        mock_config.return_value = {}
        mock_kpis.return_value = {
            "backends": {"total": 5, "active": 3},
            "datasources": {"total": 10, "active": 8},
            "vectordb": {"status": "ok"},
            "jobs": {"running": None, "pending": 0},
        }
        from kiara.admin.functions.dashboard import dashboard_status

        result = _run(dashboard_status(user=MOCK_USER))
        assert result.success is True
        assert "backends" in result.data
        assert "datasources" in result.data
        assert "vectordb" in result.data
        assert "jobs" in result.data


# ---------------------------------------------------------------------------
# 3. Datasources
# ---------------------------------------------------------------------------


class TestDatasources:
    @patch("kiara.admin.functions.datasources.cache_repo")
    def test_datasource_list(self, mock_cache):
        mock_cache.get_cached_datasources.return_value = [{"id": 1, "name": "Test"}]
        from kiara.admin.functions.datasources import datasource_list

        result = _run(datasource_list(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data
        assert len(result.data["data"]) == 1

    @patch("kiara.admin.functions.datasources.count_files_limited", return_value=42)
    @patch("kiara.admin.functions.datasources.os.path.isdir", return_value=True)
    @patch("kiara.web.api_datasource._effective_path", return_value="/data/test")
    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_status(self, mock_repo, mock_path, mock_isdir, mock_count):
        mock_repo.get_by_id.return_value = {
            "id": 1,
            "name": "Test",
            "type": "local",
            "connector_config": "{}",
        }
        from kiara.admin.functions.datasources import datasource_status

        result = _run(datasource_status(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert result.data["accessible"] is True
        assert result.data["file_count"] == 42

    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_status_not_found(self, mock_repo):
        mock_repo.get_by_id.return_value = None
        from kiara.admin.functions.datasources import datasource_status

        result = _run(datasource_status(user=MOCK_USER, ds_id=999))
        assert result.success is False
        assert "nicht gefunden" in result.message.lower()

    @patch("kiara.admin.functions.datasources.perm_repo")
    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_groups(self, mock_ds, mock_perm):
        mock_ds.exists.return_value = True
        mock_perm.get_datasource_groups.return_value = [{"id": 1, "name": "Admins"}]
        from kiara.admin.functions.datasources import datasource_groups

        result = _run(datasource_groups(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert "groups" in result.data

    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_groups_not_found(self, mock_ds):
        mock_ds.exists.return_value = False
        from kiara.admin.functions.datasources import datasource_groups

        result = _run(datasource_groups(user=MOCK_USER, ds_id=999))
        assert result.success is False

    @patch("kiara.admin.functions.datasources.perm_repo")
    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_users(self, mock_ds, mock_perm):
        mock_ds.exists.return_value = True
        mock_perm.get_datasource_users.return_value = [{"id": 1, "username": "user1"}]
        from kiara.admin.functions.datasources import datasource_users

        result = _run(datasource_users(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert "users" in result.data

    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_users_not_found(self, mock_ds):
        mock_ds.exists.return_value = False
        from kiara.admin.functions.datasources import datasource_users

        result = _run(datasource_users(user=MOCK_USER, ds_id=999))
        assert result.success is False

    @patch("kiara.connectors.registry.all_connectors")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_datasource_connector_types(self, mock_loaded, mock_connectors):
        mock_cls = MagicMock()
        mock_cls.name = "local"
        mock_cls.label = "Lokales Verzeichnis"
        mock_cls.badge_class = "badge-local"
        mock_cls.supports_browse = False
        mock_cls.config_schema.return_value = {}
        mock_cls.default_config.return_value = {}
        mock_connectors.return_value = {"local": mock_cls}
        from kiara.admin.functions.datasources import datasource_connector_types

        result = _run(datasource_connector_types(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data
        assert result.data["data"][0]["name"] == "local"
        assert result.data["data"][0]["supports_browse"] is False

    @patch("kiara.connectors.registry.all_connectors")
    @patch("kiara.connectors.registry.ensure_loaded")
    def test_datasource_connector_types_supports_browse_fail_closed(
        self, mock_loaded, mock_connectors
    ):
        mock_cls = MagicMock()
        mock_cls.name = "mock"
        mock_cls.label = "Mock"
        mock_cls.badge_class = "badge-mock"
        mock_cls.supports_browse = "yes"
        mock_cls.config_schema.return_value = {}
        mock_cls.default_config.return_value = {}
        mock_connectors.return_value = {"mock": mock_cls}

        from kiara.admin.functions.datasources import datasource_connector_types

        result = _run(datasource_connector_types(user=MOCK_USER))

        assert result.success is True
        assert result.data["data"][0]["supports_browse"] is False

    @patch("kiara.shared.db.db_query")
    def test_datasource_ingestion_status(self, mock_db):
        payload = json.dumps(
            {
                "ingestion_status": "idle",
                "embeddings": [
                    {
                        "embedding_model": "nomic-embed-text",
                        "chunk_count": 100,
                        "file_count": 50,
                        "last_ingested_at": "2026-03-01",
                    },
                ],
            }
        )
        mock_db.return_value = _mock_db_context(fetchone_val={"payload_json": payload})
        from kiara.admin.functions.datasources import datasource_ingestion_status

        result = _run(datasource_ingestion_status(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert result.data["ingestion_status"] == "idle"
        assert result.data["chunk_count"] == 100
        assert result.data["last_ingested_at"] == "2026-03-01"

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.shared.db.db_query")
    def test_datasource_ingestion_status_not_found(self, mock_db, mock_refresh):
        mock_db.return_value = _mock_db_context(fetchone_val=None)
        from kiara.admin.functions.datasources import datasource_ingestion_status

        result = _run(datasource_ingestion_status(user=MOCK_USER, ds_id=999))
        assert result.success is False


# ---------------------------------------------------------------------------
# 4. Jobs
# ---------------------------------------------------------------------------


class TestJobs:
    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list(self, mock_cache):
        mock_cache.get_cached_jobs.return_value = ([{"id": 1}], 1)
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data
        assert "total" in result.data
        assert "limit" in result.data
        assert "offset" in result.data

    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list_filtered(self, mock_cache):
        mock_cache.get_cached_jobs.return_value = ([{"id": 2, "status": "running"}], 1)
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER, ds_id=1, status="running"))
        assert result.success is True
        mock_cache.get_cached_jobs.assert_called_with(
            ds_id=1,
            status="running",
            limit=50,
            offset=0,
        )

    @patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None)
    @patch("kiara.web.api_ingestion_jobs._enrich_resume_chains")
    @patch("kiara.shared.db.db_query")
    def test_jobs_status(self, mock_db, mock_enrich, mock_running):
        job_row = {"id": 1, "status": "completed", "stats_json": "{}", "datasource_name": "Test DS"}
        mock_db.return_value = _mock_db_context(fetchone_val=job_row)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=1))
        assert result.success is True

    @patch("kiara.shared.db.db_query")
    def test_jobs_status_not_found(self, mock_db):
        mock_db.return_value = _mock_db_context(fetchone_val=None)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=999))
        assert result.success is False
        assert "nicht gefunden" in result.message.lower()

    @patch("kiara.admin.functions.jobs._file_log_repo")
    @patch("kiara.admin.functions.jobs._job_repo")
    def test_jobs_logs(self, mock_job, mock_file):
        mock_job.exists.return_value = True
        mock_file.list_by_job.return_value = ([{"file_name": "test.pdf"}], 1)
        mock_file.aggregate_by_status.return_value = {
            "success": {"count": 1, "chunks": 5, "avg_ms": 100},
        }
        from kiara.admin.functions.jobs import jobs_logs

        result = _run(jobs_logs(user=MOCK_USER, job_id=1))
        assert result.success is True
        assert "data" in result.data
        assert "summary" in result.data

    @patch("kiara.admin.functions.jobs._job_repo")
    def test_jobs_logs_not_found(self, mock_job):
        mock_job.exists.return_value = False
        from kiara.admin.functions.jobs import jobs_logs

        result = _run(jobs_logs(user=MOCK_USER, job_id=999))
        assert result.success is False

    @patch("kiara.web.cron_manager.read_schedules")
    def test_jobs_schedule(self, mock_schedules):
        mock_schedules.return_value = {1: {"cron_expr": "0 2 * * *", "name": "Nightly"}}
        from kiara.admin.functions.jobs import jobs_schedule

        result = _run(jobs_schedule(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert result.data["active"] is True
        assert result.data["cron_expr"] == "0 2 * * *"

    @patch("kiara.web.cron_manager.read_schedules")
    def test_jobs_schedule_not_found(self, mock_schedules):
        mock_schedules.return_value = {}
        from kiara.admin.functions.jobs import jobs_schedule

        result = _run(jobs_schedule(user=MOCK_USER, ds_id=999))
        assert result.success is True
        assert result.data["active"] is False


# ---------------------------------------------------------------------------
# 5. System
# ---------------------------------------------------------------------------


class TestSystem:
    @patch("kiara.web.api_health.health_check")
    def test_system_health(self, mock_check):
        mock_resp = MagicMock()
        mock_resp.body = json.dumps({"status": "healthy"}).encode()
        mock_check.return_value = mock_resp
        from kiara.admin.functions.system import system_health

        result = _run(system_health(user=MOCK_USER))
        assert result.success is True
        assert result.data["status"] == "healthy"

    @patch("kiara.web.api_health.readiness_check")
    def test_system_health_ready(self, mock_check):
        mock_resp = MagicMock()
        mock_resp.body = json.dumps({"ready": True}).encode()
        mock_check.return_value = mock_resp
        from kiara.admin.functions.system import system_health_ready

        result = _run(system_health_ready(user=MOCK_USER))
        assert result.success is True

    @patch("kiara.web.api_health.liveness_check")
    def test_system_health_live(self, mock_check):
        mock_resp = MagicMock()
        mock_resp.body = json.dumps({"alive": True}).encode()
        mock_check.return_value = mock_resp
        from kiara.admin.functions.system import system_health_live

        result = _run(system_health_live(user=MOCK_USER))
        assert result.success is True

    @patch("kiara.web.config_changelog.query_changelog")
    def test_system_changelog(self, mock_query):
        mock_query.return_value = {
            "data": [{"id": 1, "area": "datasources"}],
            "pagination": {"page": 1, "page_size": 50, "total_items": 1, "total_pages": 1},
        }
        from kiara.admin.functions.system import system_changelog

        result = _run(system_changelog(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data
        assert "pagination" in result.data

    @patch("kiara.web.config_changelog.query_changelog")
    def test_system_changelog_sort(self, mock_query):
        """sort_field und sort_dir werden korrekt durchgereicht."""
        mock_query.return_value = {"data": [], "pagination": {}}
        from kiara.admin.functions.system import system_changelog

        _run(system_changelog(user=MOCK_USER, sort_field="area", sort_dir="asc"))
        mock_query.assert_called_once_with(
            area=None,
            source=None,
            page=1,
            page_size=50,
            sort_field="area",
            sort_dir="asc",
        )

    @patch("kiara.repositories.changelog.ConfigChangelogRepository")
    def test_system_changelog_areas(self, mock_cls):
        mock_repo = MagicMock()
        mock_repo.list_distinct_areas.return_value = ["datasources", "ki", "users"]
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.system import system_changelog_areas

        result = _run(system_changelog_areas(user=MOCK_USER))
        assert result.success is True
        assert "areas" in result.data
        assert len(result.data["areas"]) == 3


# ---------------------------------------------------------------------------
# 6. KI
# ---------------------------------------------------------------------------


class TestKi:
    @patch("kiara.repositories.backend.KiBackendRepository")
    def test_ki_backends_list(self, mock_cls):
        mock_repo = MagicMock()
        mock_repo.list_ordered.return_value = [
            {
                "id": 1,
                "api_key": None,
                "extra_config": "{}",
                "is_active": True,
                "last_test_ok": 1,
                "trusted": False,
            }
        ]
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_backends_list

        result = _run(ki_backends_list(user=MOCK_USER))
        assert result.success is True
        assert "backends" in result.data
        backend = result.data["backends"][0]
        assert isinstance(backend["trusted"], bool)
        assert backend["trusted"] is False

    @patch("kiara.web.api_ki_backends.get_categories", new_callable=AsyncMock)
    def test_ki_backends_categories(self, mock_cats):
        mock_cats.return_value = {"categories": [{"id": "llm", "label": "LLM"}]}
        from kiara.admin.functions.ki import ki_backends_categories

        result = _run(ki_backends_categories(user=MOCK_USER))
        assert result.success is True
        assert "categories" in result.data

    @patch("kiara.web.api_ki.get_ki_config", new_callable=AsyncMock)
    def test_ki_config(self, mock_config):
        mock_config.return_value = {"config": {"llm_model": "qwen3:8b"}}
        from kiara.admin.functions.ki import ki_config

        result = _run(ki_config(user=MOCK_USER))
        assert result.success is True
        assert "config" in result.data

    @patch("kiara.web.api_ki.list_prompt_templates", new_callable=AsyncMock)
    def test_ki_templates_list(self, mock_templates):
        """Test 20: LIST ohne Filter → alle Templates."""
        mock_templates.return_value = {"data": [{"id": 1, "name": "Standard"}]}
        from kiara.admin.functions.ki import ki_templates_list

        result = _run(ki_templates_list(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data

    @patch("kiara.web.api_ki._template_row_to_dict")
    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_ki_templates_list_filtered(self, mock_cls, mock_to_dict):
        """Test 17/27: LIST mit chain_type=enrichment → gefiltert."""
        mock_repo = MagicMock()
        mock_repo.list_filtered.return_value = (
            [{"id": 10, "name": "enrichment_default", "chain_type": "enrichment"}],
            1,
        )
        mock_cls.return_value = mock_repo
        mock_to_dict.side_effect = lambda r: dict(r)

        from kiara.admin.functions.ki import ki_templates_list

        result = _run(ki_templates_list(user=MOCK_USER, chain_type="enrichment"))
        assert result.success is True
        assert len(result.data["data"]) == 1
        mock_repo.list_filtered.assert_called_once()
        # Verify filter passed
        call_kwargs = mock_repo.list_filtered.call_args[1]
        assert call_kwargs["filters"] == {"chain_type": "enrichment"}

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_ki_templates_history_found(self, mock_cls):
        """Test 23: History-Endpoint mit vorhandenem Template."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {"id": 1, "name": "test"}
        mock_repo.get_history.return_value = [
            {
                "id": 1,
                "template_id": 1,
                "content": "v1",
                "content_hash": "h1",
                "cloud_system_prompt": "",
                "changed_by": "admin",
                "change_reason": "",
                "created_at": "2026-04-03",
            }
        ]
        mock_cls.return_value = mock_repo

        from kiara.admin.functions.ki import ki_templates_history

        result = _run(ki_templates_history(user=MOCK_USER, template_id=1))
        assert result.success is True
        assert len(result.data["data"]) == 1

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_ki_templates_history_not_found(self, mock_cls):
        """Test 24: History-Endpoint mit nicht vorhandenem Template."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = None
        mock_cls.return_value = mock_repo

        from kiara.admin.functions.ki import ki_templates_history

        result = _run(ki_templates_history(user=MOCK_USER, template_id=999))
        assert result.success is False
        assert "nicht gefunden" in result.message

    def test_ki_templates_create_schema_has_enrichment(self):
        """Test 28: Create-Enum enthaelt 'enrichment'."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.templates.create")
        chain_type_enum = func.parameters["properties"]["chain_type"]["enum"]
        assert "enrichment" in chain_type_enum

    def test_ki_templates_create_schema_has_cloud_system_prompt(self):
        """Test 38: cloud_system_prompt im Create-Schema."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.templates.create")
        assert "cloud_system_prompt" in func.parameters["properties"]

    def test_ki_templates_update_schema_has_all_types(self):
        """Test 28b: Update-Enum enthaelt alle Typen inkl. rewriter/admin_agent."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.templates.update")
        chain_type_enum = func.parameters["properties"]["chain_type"]["enum"]
        assert "enrichment" in chain_type_enum
        assert "rewriter" in chain_type_enum
        assert "admin_agent" in chain_type_enum

    def test_ki_templates_list_schema_has_chain_type(self):
        """Test 39: chain_type Query-Parameter im List-Schema."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("ki.templates.list")
        assert "chain_type" in func.parameters["properties"]

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_ki_templates_detail(self, mock_cls):
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {"id": 1, "name": "Standard", "content": "..."}
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_templates_detail

        result = _run(ki_templates_detail(user=MOCK_USER, template_id=1))
        assert result.success is True
        assert result.data["name"] == "Standard"

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_ki_templates_detail_not_found(self, mock_cls):
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = None
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_templates_detail

        result = _run(ki_templates_detail(user=MOCK_USER, template_id=999))
        assert result.success is False
        assert "nicht gefunden" in result.message.lower()

    @patch("kiara.web.api_ki.get_automaton_stats", new_callable=AsyncMock)
    def test_ki_automaton_stats(self, mock_stats):
        mock_stats.return_value = {
            "entry_count": 1000,
            "erp_entries": 800,
            "ldap_entries": 200,
            "last_built": "2026-03-01",
        }
        from kiara.admin.functions.ki import ki_automaton_stats

        result = _run(ki_automaton_stats(user=MOCK_USER))
        assert result.success is True
        assert "entry_count" in result.data

    @patch("kiara.web.api_ki.get_audit_log", new_callable=AsyncMock)
    def test_ki_audit(self, mock_audit):
        mock_audit.return_value = {"data": [], "total": 0, "page": 1}
        from kiara.admin.functions.ki import ki_audit

        result = _run(ki_audit(user=MOCK_USER))
        assert result.success is True
        assert "data" in result.data
        assert "total" in result.data


# ---------------------------------------------------------------------------
# 7. Features
# ---------------------------------------------------------------------------


class TestFeatures:
    @patch("kiara.web.api_datasource_features.ddu_available", new_callable=AsyncMock)
    def test_features_ddu_available(self, mock_ddu):
        mock_ddu.return_value = {"available": True, "mode": "remote"}
        from kiara.admin.functions.features import features_ddu

        result = _run(features_ddu(user=MOCK_USER))
        assert result.success is True
        assert result.data["available"] is True

    @patch("kiara.web.api_datasource_features.vlm_available", new_callable=AsyncMock)
    def test_features_vlm_available(self, mock_vlm):
        mock_vlm.return_value = {"available": False}
        from kiara.admin.functions.features import features_vlm

        result = _run(features_vlm(user=MOCK_USER))
        assert result.success is True

    @patch("kiara.web.api_datasource_features.list_embedding_models", new_callable=AsyncMock)
    def test_features_embedding_models(self, mock_embed):
        mock_embed.return_value = {"models": ["nomic-embed-text"]}
        from kiara.admin.functions.features import features_embedding_models

        result = _run(features_embedding_models(user=MOCK_USER))
        assert result.success is True
        assert "models" in result.data


# ---------------------------------------------------------------------------
# 8. Help
# ---------------------------------------------------------------------------


class TestHelp:
    def _setup_registry(self):
        """Fuellt Registry mit Test-Funktionen."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        return admin_registry

    def test_help_search_all(self):
        """Ohne Parameter → alle Funktionen."""
        self._setup_registry()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER))
        assert result.success is True
        assert len(result.data["functions"]) > 0

    def test_help_search_by_name(self):
        """Exakter Funktionsname → Detail."""
        self._setup_registry()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, function_name="system.ping"))
        assert result.success is True
        assert result.data["name"] == "system.ping"

    def test_help_search_by_category(self):
        """Kategorie → gefilterte Liste (alle aus category='system')."""
        self._setup_registry()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, category="system"))
        assert result.success is True
        assert len(result.data["functions"]) > 0
        # Alle Funktionen muessen in der Registry mit category="system" registriert sein
        from kiara.admin.registry import admin_registry

        system_names = {f.name for f in admin_registry.list_by_category("system")}
        assert all(f["name"] in system_names for f in result.data["functions"])

    def test_help_search_by_query(self):
        """Freitext → Treffer."""
        self._setup_registry()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, query="Datenquelle"))
        assert result.success is True
        assert len(result.data["functions"]) > 0

    def test_help_search_not_found(self):
        """Unbekannter Name → fail."""
        self._setup_registry()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, function_name="nonexistent.func"))
        assert result.success is False


# ---------------------------------------------------------------------------
# 9. Auto-Router-Integration
# ---------------------------------------------------------------------------


class TestAutoRouter:
    def test_autorouter_raw_response_dashboard(self):
        """GET /api/admin/dashboard → Rohdaten (raw_response)."""

        async def _dashboard(user: dict) -> AdminResult:
            return AdminResult.ok("ok", data={"backends": {}, "datasources": {}})

        func = _make_func(name="dashboard.status", raw_response=True, fn=_dashboard)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert "backends" in result
        assert "datasources" in result

    def test_autorouter_error_returns_json_response(self):
        """Fehler → JSONResponse mit status_code."""

        async def _fail(user: dict) -> AdminResult:
            return AdminResult.fail("Nicht gefunden")

        func = _make_func(fn=_fail)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert hasattr(result, "status_code")
        assert result.status_code == 404

    def test_autorouter_no_duplicate_routes(self):
        """Keine doppelten Routen nach Migration."""
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        paths = []
        for func in admin_registry.list_all():
            if func.http_method and func.http_path:
                key = f"{func.http_method} {func.http_path}"
                assert key not in paths, f"Doppelte Route: {key}"
                paths.append(key)


# ---------------------------------------------------------------------------
# 10. Regressionstests
# ---------------------------------------------------------------------------


class TestRegression:
    def test_frontend_dashboard_format(self):
        """Response-Format identisch zu bisherigem Endpoint."""

        async def _dashboard(user: dict) -> AdminResult:
            return AdminResult.ok(
                "ok",
                data={
                    "backends": {"total": 5, "active": 3},
                    "datasources": {"total": 10, "active": 8},
                    "vectordb": {"status": "ok"},
                    "jobs": {"running": None, "pending": 0},
                },
            )

        func = _make_func(name="dashboard.status", raw_response=True, fn=_dashboard)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert isinstance(result, dict)
        assert set(result.keys()) == {"backends", "datasources", "vectordb", "jobs"}

    def test_frontend_datasources_format(self):
        """datasource.list gibt {"data": [...]} zurueck."""

        async def _ds_list(user: dict) -> AdminResult:
            return AdminResult.ok("ok", data={"data": [{"id": 1}]})

        func = _make_func(name="datasource.list", raw_response=True, fn=_ds_list)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert "data" in result
        assert isinstance(result["data"], list)

    def test_frontend_error_status_codes(self):
        """Fehler geben HTTP 4xx, nicht 200."""

        async def _fail(user: dict) -> AdminResult:
            return AdminResult.fail("Datenquelle nicht gefunden")

        func = _make_func(raw_response=True, fn=_fail)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result.status_code == 404

    def test_help_replaces_helptool(self):
        """AdminHelpTool ist nicht mehr in der Tool-Chain."""
        import kiara.admin.help_tool as ht

        assert not hasattr(ht, "AdminHelpTool")

    def test_raw_response_default_false(self):
        """Neue Funktionen ohne raw_response haben Default False."""
        func = _make_func()
        assert func.raw_response is False

    def test_raw_response_field_last(self):
        """raw_response ist das letzte Feld der Dataclass."""
        fields = list(AdminFunction.__dataclass_fields__.keys())
        assert fields[-1] == "raw_response"


# ---------------------------------------------------------------------------
# 11. Prio-1 Edge-Cases — wahrscheinlichste Fehlerpfade
# ---------------------------------------------------------------------------


class TestEdgeCaseDatasources:
    """Edge-Cases fuer Datasource-Funktionen."""

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.admin.functions.datasources.cache_repo")
    def test_datasource_list_cold_start_refresh(self, mock_cache, mock_refresh):
        """Cache leer beim ersten Aufruf → Refresh → zweiter Aufruf liefert Daten."""
        # Erster Aufruf: None/leer, zweiter Aufruf (nach refresh): Daten
        mock_cache.get_cached_datasources.side_effect = [
            None,  # Cold-Start: Cache leer
            [{"id": 1, "name": "After-Refresh"}],  # Nach Refresh
        ]
        from kiara.admin.functions.datasources import datasource_list

        result = _run(datasource_list(user=MOCK_USER))
        assert result.success is True
        assert result.data["data"][0]["name"] == "After-Refresh"
        mock_refresh.assert_called_once()  # refresh_datasource_cache wurde getriggert

    @patch("kiara.admin.functions.datasources.cache_repo")
    def test_datasource_list_cache_stays_empty(self, mock_cache):
        """Cache leer auch nach Refresh (leere Datenbank) → leere Liste, kein Crash."""
        mock_cache.get_cached_datasources.side_effect = [None, None]
        with patch("kiara.web.cache_refresh.refresh_datasource_cache"):
            from kiara.admin.functions.datasources import datasource_list

            result = _run(datasource_list(user=MOCK_USER))
        assert result.success is True
        assert result.data["data"] is None  # Keine Daten, aber kein Fehler

    @patch("kiara.web.api_datasource._effective_path", return_value="")
    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_status_empty_path(self, mock_repo, mock_path):
        """DS ohne konfigurierten Pfad → accessible=False, file_count=0."""
        mock_repo.get_by_id.return_value = {
            "id": 1,
            "name": "NoPath",
            "type": "local",
            "connector_config": "{}",
        }
        from kiara.admin.functions.datasources import datasource_status

        result = _run(datasource_status(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert result.data["path"] == ""
        assert result.data["accessible"] is False
        assert result.data["file_count"] == 0

    @patch("kiara.admin.functions.datasources.count_files_limited", return_value=0)
    @patch("kiara.admin.functions.datasources.os.path.isdir", return_value=False)
    @patch("kiara.web.api_datasource._effective_path", return_value="/mnt/removed-share")
    @patch("kiara.admin.functions.datasources.ds_repo")
    def test_datasource_status_inaccessible(self, mock_repo, mock_path, mock_isdir, mock_count):
        """Pfad existiert in Config, aber Verzeichnis nicht erreichbar."""
        mock_repo.get_by_id.return_value = {
            "id": 1,
            "name": "Offline",
            "type": "smb",
            "connector_config": "{}",
        }
        from kiara.admin.functions.datasources import datasource_status

        result = _run(datasource_status(user=MOCK_USER, ds_id=1))
        assert result.success is True
        assert result.data["path"] == "/mnt/removed-share"
        assert result.data["accessible"] is False
        assert result.data["file_count"] == 0
        mock_count.assert_not_called()  # count_files_limited darf nicht aufgerufen werden


class TestEdgeCaseJobs:
    """Edge-Cases fuer Job-Funktionen."""

    @patch("kiara.web.ingestion_worker.is_job_paused", return_value=False)
    @patch("kiara.web.ingestion_worker.get_running_job_id", return_value=999)
    @patch("kiara.web.api_ingestion_jobs._enrich_resume_chains")
    @patch("kiara.shared.db.db_query")
    def test_jobs_status_ghost_job(self, mock_db, mock_enrich, mock_running, mock_paused):
        """Job hat Status RUNNING, aber anderer Worker laeuft → is_ghost=True."""
        from kiara.shared.enums import IngestionStatus

        job_row = {
            "id": 42,
            "status": IngestionStatus.RUNNING,
            "stats_json": "{}",
            "datasource_name": "Test DS",
        }
        mock_db.return_value = _mock_db_context(fetchone_val=job_row)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=42))
        assert result.success is True
        assert result.data.get("is_ghost") is True
        mock_paused.assert_not_called()  # is_job_paused darf NICHT aufgerufen werden

    @patch("kiara.web.ingestion_worker.is_job_paused", return_value=True)
    @patch("kiara.web.ingestion_worker.get_running_job_id", return_value=42)
    @patch("kiara.web.api_ingestion_jobs._enrich_resume_chains")
    @patch("kiara.shared.db.db_query")
    def test_jobs_status_paused_job(self, mock_db, mock_enrich, mock_running, mock_paused):
        """Job laeuft und ist pausiert → is_paused=True."""
        from kiara.shared.enums import IngestionStatus

        job_row = {
            "id": 42,
            "status": IngestionStatus.RUNNING,
            "stats_json": "{}",
            "datasource_name": "Test DS",
        }
        mock_db.return_value = _mock_db_context(fetchone_val=job_row)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=42))
        assert result.success is True
        assert result.data.get("is_paused") is True
        assert result.data.get("is_ghost") is None

    @patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None)
    @patch("kiara.web.api_ingestion_jobs._enrich_resume_chains")
    @patch("kiara.shared.db.db_query")
    def test_jobs_status_malformed_stats_json(self, mock_db, mock_enrich, mock_running):
        """stats_json ist kein valides JSON → Fallback auf leeres dict."""
        job_row = {
            "id": 1,
            "status": "completed",
            "stats_json": "INVALID{{{JSON",
            "datasource_name": "Test",
        }
        mock_db.return_value = _mock_db_context(fetchone_val=job_row)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=1))
        assert result.success is True
        assert result.data["stats"] == {}  # Fallback auf leeres dict
        assert "stats_json" not in result.data  # Rohfeld entfernt

    @patch("kiara.web.ingestion_worker.get_running_job_id", return_value=None)
    @patch("kiara.web.api_ingestion_jobs._enrich_resume_chains")
    @patch("kiara.shared.db.db_query")
    def test_jobs_status_null_stats_json(self, mock_db, mock_enrich, mock_running):
        """stats_json ist None (DB-NULL) → Fallback auf leeres dict."""
        job_row = {"id": 1, "status": "completed", "stats_json": None, "datasource_name": "Test"}
        mock_db.return_value = _mock_db_context(fetchone_val=job_row)
        from kiara.admin.functions.jobs import jobs_status

        result = _run(jobs_status(user=MOCK_USER, job_id=1))
        assert result.success is True
        assert result.data["stats"] == {}

    @patch("kiara.admin.functions.jobs._file_log_repo")
    @patch("kiara.admin.functions.jobs._job_repo")
    def test_jobs_logs_empty_summary(self, mock_job, mock_file):
        """Job ohne verarbeitete Dateien → Summary bleibt bei Defaults, kein DivisionByZero."""
        mock_job.exists.return_value = True
        mock_file.list_by_job.return_value = ([], 0)
        mock_file.aggregate_by_status.return_value = {}  # Keine Eintraege
        from kiara.admin.functions.jobs import jobs_logs

        result = _run(jobs_logs(user=MOCK_USER, job_id=1))
        assert result.success is True
        summary = result.data["summary"]
        assert summary["success"] == 0
        assert summary["error"] == 0
        assert summary["skipped"] == 0
        assert summary["total_chunks"] == 0
        assert summary["avg_duration_ms"] == 0  # Kein DivisionByZero


class TestEdgeCaseRouter:
    """Edge-Cases fuer den Auto-Router (_create_endpoint)."""

    def test_function_exception_caught(self):
        """Funktion wirft unerwartete Exception → AdminResult.fail + HTTP 500."""

        async def _explode(user: dict) -> AdminResult:
            raise RuntimeError("Unerwarteter Fehler in der Pipeline")

        func = _make_func(fn=_explode)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        # Exception wird gefangen, HTTP 500 zurueckgegeben (Server-Fehler, RFC 7231)
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["success"] is False
        assert "Interner Fehler" in body["message"]
        assert "Unerwarteter Fehler" in body["message"]

    def test_function_exception_raw_response(self):
        """raw_response=True + Exception → trotzdem HTTP 500 mit Error-Body."""

        async def _explode(user: dict) -> AdminResult:
            raise ValueError("Kaputt")

        func = _make_func(raw_response=True, fn=_explode)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result.status_code == 500
        body = json.loads(result.body)
        assert body["success"] is False

    def test_raw_response_data_none(self):
        """raw_response=True + data=None → Frontend bekommt null."""

        async def _no_data(user: dict) -> AdminResult:
            return AdminResult.ok("ok", data=None)

        func = _make_func(raw_response=True, fn=_no_data)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result == {}  # data=None → leeres Dict (valider Response)

    def test_path_params_forwarded(self):
        """Path-Parameter (z.B. ds_id) werden korrekt an die Funktion weitergegeben."""

        async def _with_id(user: dict, ds_id: int) -> AdminResult:
            return AdminResult.ok("ok", data={"ds_id": ds_id})

        func = AdminFunction(
            name="test.path",
            category="test",
            description="Test",
            help="test",
            examples=[],
            returns="test",
            parameters={
                "type": "object",
                "properties": {"ds_id": {"type": "integer", "description": "DS-ID"}},
                "required": ["ds_id"],
            },
            requires_confirmation=False,
            confirm_template=None,
            pk_param=None,
            changelog_area=None,
            changelog_action=None,
            http_method="GET",
            http_path="/api/test/{ds_id}",
            fn=_with_id,
        )
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {"ds_id": "7"}  # Aus URL immer str
        request.query_params = QueryParams()
        result = _run(endpoint(request, user=MOCK_USER))
        assert result["data"]["ds_id"] == 7  # Automatisch int konvertiert


class TestEdgeCaseKi:
    """Edge-Cases fuer KI-Backend-Funktionen."""

    @patch("kiara.repositories.backend.KiBackendRepository")
    def test_ki_backends_api_key_masked(self, mock_cls):
        """Backend mit api_key → wird durch PASSWORD_MASK ersetzt."""
        mock_repo = MagicMock()
        mock_repo.list_ordered.return_value = [
            {
                "id": 1,
                "api_key": "sk-secret-1234",
                "extra_config": "{}",
                "is_active": True,
                "last_test_ok": None,
                "trusted": True,
            }
        ]
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_backends_list

        result = _run(ki_backends_list(user=MOCK_USER))
        assert result.success is True
        backend = result.data["backends"][0]
        assert backend["api_key"] != "sk-secret-1234"  # Nicht das Original
        assert backend["api_key"] == "********"  # PASSWORD_MASK aus constants.py
        assert backend["last_test_ok"] is None  # None bleibt None
        assert backend["trusted"] is True  # int(1) → bool(True)

    @patch("kiara.repositories.backend.KiBackendRepository")
    def test_ki_backends_malformed_extra_config(self, mock_cls):
        """extra_config ist kein valides JSON → Fallback auf leeres dict."""
        mock_repo = MagicMock()
        mock_repo.list_ordered.return_value = [
            {
                "id": 1,
                "api_key": None,
                "extra_config": "NOT_JSON!!!",
                "is_active": False,
                "last_test_ok": 0,
                "trusted": False,
            }
        ]
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_backends_list

        result = _run(ki_backends_list(user=MOCK_USER))
        assert result.success is True
        backend = result.data["backends"][0]
        assert backend["extra_config"] == {}  # Fallback
        assert backend["is_active"] is False  # bool(0)
        assert backend["last_test_ok"] is False  # bool(0)
        assert backend["trusted"] is False  # bool(0)

    @patch("kiara.repositories.backend.KiBackendRepository")
    def test_ki_backends_no_api_key(self, mock_cls):
        """Backend ohne api_key (None) → kein Masking, bleibt None."""
        mock_repo = MagicMock()
        mock_repo.list_ordered.return_value = [
            {
                "id": 1,
                "api_key": None,
                "extra_config": '{"key": "val"}',
                "is_active": True,
                "last_test_ok": 1,
                "trusted": True,
            }
        ]
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_backends_list

        result = _run(ki_backends_list(user=MOCK_USER))
        backend = result.data["backends"][0]
        assert backend["api_key"] is None  # Kein Masking bei None
        assert backend["extra_config"] == {"key": "val"}  # JSON geparst
        assert backend["is_active"] is True
        assert backend["trusted"] is True


# ---------------------------------------------------------------------------
# 12. Prio-2 Edge-Cases — Robustheit und Grenzfaelle
# ---------------------------------------------------------------------------


class TestEdgeCaseIngestionStatusRetry:
    """Cache-Refresh-Retry fuer ingestion_status."""

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.shared.db.db_query")
    def test_ingestion_status_retry_success(self, mock_db, mock_refresh):
        """Erster DB-Zugriff None → Refresh → zweiter Versuch liefert Daten."""
        payload = json.dumps(
            {
                "ingestion_status": "running",
                "embeddings": [
                    {
                        "embedding_model": "nomic-embed-text",
                        "chunk_count": 50,
                        "file_count": 25,
                        "last_ingested_at": "2026-03-01",
                    },
                ],
            }
        )
        # Zwei separate Context-Manager: erster liefert None, zweiter liefert Daten
        cm_empty = _mock_db_context(fetchone_val=None)
        cm_data = _mock_db_context(fetchone_val={"payload_json": payload})
        mock_db.side_effect = [cm_empty, cm_data]
        from kiara.admin.functions.datasources import datasource_ingestion_status

        result = _run(datasource_ingestion_status(user=MOCK_USER, ds_id=5))
        assert result.success is True
        assert result.data["ingestion_status"] == "running"
        mock_refresh.assert_called_once_with(5)  # Refresh mit korrekter ds_id

    @patch("kiara.web.cache_refresh.refresh_datasource_cache")
    @patch("kiara.shared.db.db_query")
    def test_ingestion_status_retry_still_empty(self, mock_db, mock_refresh):
        """Beide DB-Zugriffe None → AdminResult.fail (DS existiert nicht)."""
        cm1 = _mock_db_context(fetchone_val=None)
        cm2 = _mock_db_context(fetchone_val=None)
        mock_db.side_effect = [cm1, cm2]
        from kiara.admin.functions.datasources import datasource_ingestion_status

        result = _run(datasource_ingestion_status(user=MOCK_USER, ds_id=999))
        assert result.success is False
        assert "nicht gefunden" in result.message.lower()


class TestEdgeCaseJobsListRefresh:
    """Cache-Refresh-Bedingung fuer jobs.list."""

    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list_no_refresh_with_filter(self, mock_cache):
        """Leere Liste MIT Status-Filter → KEIN Refresh (Bedingung: not status)."""
        mock_cache.get_cached_jobs.return_value = ([], 0)
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER, status="running"))
        assert result.success is True
        assert result.data["data"] == []
        assert result.data["total"] == 0
        # get_cached_jobs nur einmal aufgerufen (kein Refresh-Retry)
        assert mock_cache.get_cached_jobs.call_count == 1

    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list_no_refresh_with_ds_filter(self, mock_cache):
        """Leere Liste MIT ds_id-Filter → KEIN Refresh (Bedingung: ds_id is None)."""
        mock_cache.get_cached_jobs.return_value = ([], 0)
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER, ds_id=42))
        assert result.success is True
        assert mock_cache.get_cached_jobs.call_count == 1

    @patch("kiara.web.cache_refresh.refresh_jobs_cache")
    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list_cold_start_refresh(self, mock_cache, mock_refresh):
        """Leere Liste OHNE Filter → Refresh wird getriggert (Cold-Start)."""
        mock_cache.get_cached_jobs.side_effect = [
            ([], 0),  # Vor Refresh: leer
            ([{"id": 1}], 1),  # Nach Refresh: Daten
        ]
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER))
        assert result.success is True
        assert result.data["total"] == 1
        mock_refresh.assert_called_once()
        assert mock_cache.get_cached_jobs.call_count == 2

    @patch("kiara.admin.functions.jobs._cache_repo")
    def test_jobs_list_nonzero_total_no_refresh(self, mock_cache):
        """total > 0 aber leere Seite (Offset zu hoch) → KEIN Refresh."""
        mock_cache.get_cached_jobs.return_value = ([], 100)  # total=100, aber Seite leer
        from kiara.admin.functions.jobs import jobs_list

        result = _run(jobs_list(user=MOCK_USER, offset=200))
        assert result.success is True
        assert result.data["total"] == 100
        assert mock_cache.get_cached_jobs.call_count == 1


class TestEdgeCaseHelpPriority:
    """Parameterpriorisierung und Grenzfaelle in help.search."""

    def _setup(self):
        from kiara.admin.registry import load_admin_functions

        load_admin_functions()

    def test_function_name_wins_over_query(self):
        """function_name hat Vorrang vor query — Code prueft if-Reihenfolge."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(
            help_search(
                user=MOCK_USER,
                function_name="system.ping",
                query="Datenquelle",  # Wuerde andere Treffer liefern
            )
        )
        assert result.success is True
        assert result.data["name"] == "system.ping"

    def test_function_name_wins_over_category(self):
        """function_name hat Vorrang vor category."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(
            help_search(
                user=MOCK_USER,
                function_name="system.ping",
                category="datasources",
            )
        )
        assert result.success is True
        assert result.data["name"] == "system.ping"

    def test_category_wins_over_query(self):
        """category hat Vorrang vor query."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(
            help_search(
                user=MOCK_USER,
                category="system",
                query="Datenquelle",
            )
        )
        assert result.success is True
        # Alle Ergebnisse aus Kategorie "system", nicht Freitext "Datenquelle"
        from kiara.admin.registry import admin_registry

        system_names = {f.name for f in admin_registry.list_by_category("system")}
        assert all(f["name"] in system_names for f in result.data["functions"])

    def test_nonexistent_category_fails(self):
        """Nicht existierende Kategorie → AdminResult.fail."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, category="nonexistent"))
        assert result.success is False
        assert "nonexistent" in result.message

    def test_no_results_for_query(self):
        """Freitext ohne Treffer → AdminResult.fail."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER, query="xyzzy_gibberish_42"))
        assert result.success is False
        assert "xyzzy_gibberish_42" in result.message

    def test_all_functions_have_category(self):
        """Ohne Parameter: jede Funktion hat name, description UND category."""
        self._setup()
        from kiara.admin.functions.help import help_search

        result = _run(help_search(user=MOCK_USER))
        assert result.success is True
        for f in result.data["functions"]:
            assert "name" in f
            assert "description" in f
            assert "category" in f  # Nur in der "alle auflisten"-Variante


class TestEdgeCaseRouterQueryParams:
    """Router filtert unbekannte Query-Parameter und coerced korrekt."""

    def test_unknown_query_params_filtered(self):
        """Query-Params die nicht im Schema stehen werden ignoriert."""

        async def _fn(user: dict) -> AdminResult:
            return AdminResult.ok("ok", data={"got": "nothing"})

        func = _make_func(raw_response=True, fn=_fn)
        endpoint = _create_endpoint(func)
        request = MagicMock()
        request.path_params = {}
        request.query_params = QueryParams({"_cache": "123", "random": "noise"})  # Nicht im Schema
        result = _run(endpoint(request, user=MOCK_USER))
        assert result == {"got": "nothing"}  # Funktion sieht keine Extra-Params

    def test_coerce_boolean_false_values(self):
        """Boolean-Coercion: 'false', '0', 'no' → False."""
        schema = {
            "properties": {"flag": {"type": "boolean"}},
        }
        for val in ["false", "0", "no", "FALSE", "No"]:
            result = _coerce_params({"flag": val}, schema)
            assert result["flag"] is False, f"'{val}' sollte False ergeben"

    def test_coerce_boolean_true_values(self):
        """Boolean-Coercion: 'true', '1', 'yes' → True."""
        schema = {
            "properties": {"flag": {"type": "boolean"}},
        }
        for val in ["true", "1", "yes", "TRUE", "Yes"]:
            result = _coerce_params({"flag": val}, schema)
            assert result["flag"] is True, f"'{val}' sollte True ergeben"

    def test_coerce_invalid_integer_stays_string(self):
        """Nicht-parsebarer Integer-Wert bleibt String."""
        schema = {
            "properties": {"count": {"type": "integer"}},
        }
        result = _coerce_params({"count": "abc"}, schema)
        assert result["count"] == "abc"

    def test_coerce_empty_string_integer(self):
        """Leerer String fuer Integer → bleibt String (kein Crash)."""
        schema = {
            "properties": {"id": {"type": "integer"}},
        }
        result = _coerce_params({"id": ""}, schema)
        assert result["id"] == ""


class TestEdgeCaseTemplateDetail:
    """Grenzfaelle fuer ki.templates.detail."""

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_template_without_name_field(self, mock_cls):
        """Template-Row ohne 'name'-Key → Fallback auf template_id in Message."""
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = {"id": 5, "content": "Test-Inhalt"}
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_templates_detail

        result = _run(ki_templates_detail(user=MOCK_USER, template_id=5))
        assert result.success is True
        assert "5" in result.message  # Fallback: template_id statt name

    @patch("kiara.repositories.prompt.PromptTemplateRepository")
    def test_template_data_is_dict(self, mock_cls):
        """Ergebnis ist ein echtes dict, keine DB-Row-Referenz."""
        mock_repo = MagicMock()
        mock_row = MagicMock()
        mock_row.get.return_value = "TestName"
        mock_row.__iter__ = MagicMock(return_value=iter([("id", 1), ("name", "TestName")]))
        mock_row.keys = MagicMock(return_value=["id", "name"])
        mock_repo.get_by_id.return_value = mock_row
        mock_cls.return_value = mock_repo
        from kiara.admin.functions.ki import ki_templates_detail

        result = _run(ki_templates_detail(user=MOCK_USER, template_id=1))
        assert result.success is True
        assert isinstance(result.data, dict)
