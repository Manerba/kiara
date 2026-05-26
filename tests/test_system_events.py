"""Tests fuer das Admin-System-Log-Feature.

Gruppen:
    - _parse_kv_message: 7 Eingaben inkl. Edge-Cases (Entscheidung #4).
    - SystemEventDBHandler: fail-safe, min-level, source-Ableitung, exc_info.
    - _coerce_system_event_filters: Listen/CSV-Strings/repeated Query-Keys.
    - Rate-Limit: 60 req/min pro User, 61. → 429.
    - AdminFunction-Registrierung + Modul-Liste.
    - Config-Helper-Normalisierung.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
import re
from typing import Any
from unittest.mock import patch

import pytest

# ───────────────────────────────────────────────────────────────────
# Parser
# ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def parser():
    from kiara.web.logging_db_handler import _parse_kv_message

    return _parse_kv_message


class TestParseKvMessage:
    def test_login_failed_basic(self, parser):
        r = parser("LOGIN_FAILED user=admin ip=192.0.2.10")
        assert r["event_type"] == "LOGIN_FAILED"
        assert r["kv"] == {"user": "admin", "ip": "192.0.2.10"}
        assert r["raw"] == ""

    def test_login_failed_with_reason_and_detail(self, parser):
        r = parser("LOGIN_FAILED user=admin ip=192.0.2.10 reason=ldap_auth_failed detail=timeout")
        assert r["event_type"] == "LOGIN_FAILED"
        assert r["kv"]["reason"] == "ldap_auth_failed"
        assert r["kv"]["detail"] == "timeout"

    def test_trailing_parens_go_into_raw(self, parser):
        r = parser("LOGIN_FAILED user=admin ip=192.0.2.10 (inactive)")
        assert r["event_type"] == "LOGIN_FAILED"
        assert "(inactive)" in r["raw"]

    def test_rate_limited_structured(self, parser):
        r = parser("RATE_LIMITED key=login:192.168.1.5")
        assert r["event_type"] == "RATE_LIMITED"
        assert r["kv"]["key"] == "login:192.168.1.5"

    def test_unstructured_falls_into_raw(self, parser):
        r = parser("Irgendein unstrukturierter Satz")
        assert r["event_type"] is None
        assert r["raw"] == "Irgendein unstrukturierter Satz"

    def test_quoted_spacey_user(self, parser):
        r = parser('LOGIN_FAILED user="max muster" ip=127.0.0.1')
        assert r["event_type"] == "LOGIN_FAILED"
        assert r["kv"]["user"] == "max muster"
        assert r["kv"]["ip"] == "127.0.0.1"

    def test_empty_message(self, parser):
        r = parser("")
        assert r["event_type"] is None
        assert r["raw"] == ""


# ───────────────────────────────────────────────────────────────────
# SystemEventDBHandler
# ───────────────────────────────────────────────────────────────────


class _FakeConnContext:
    def __init__(self, captured: list):
        self._captured = captured

    def execute(self, sql: str, params: tuple):
        self._captured.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class TestSystemEventDBHandler:
    def test_emit_inserts_with_fields(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.INFO
        captured: list = []
        fake = _FakeConnContext(captured)

        record = logging.LogRecord(
            name="kiara.security",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="LOGIN_FAILED user=admin ip=192.0.2.10 reason=bad_password",
            args=None,
            exc_info=None,
        )

        with patch("kiara.shared.db.db_transaction", return_value=fake):
            handler.emit(record)

        assert len(captured) == 1
        sql, params = captured[0]
        assert "INSERT INTO system_events" in sql
        level, source, event_type, reason, user, ip, message, details_json = params
        assert level == "WARNING"
        assert source == "security"
        assert event_type == "LOGIN_FAILED"
        assert reason == "bad_password"
        assert user == "admin"
        assert ip == "192.0.2.10"
        assert "LOGIN_FAILED" in message

    def test_emit_respects_min_level(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.WARNING
        captured: list = []
        fake = _FakeConnContext(captured)

        record = logging.LogRecord(
            name="kiara.security",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="INFO_EVENT user=x",
            args=None,
            exc_info=None,
        )
        with patch("kiara.shared.db.db_transaction", return_value=fake):
            handler.emit(record)
        assert captured == []

    def test_emit_fail_safe_on_db_exception(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.INFO
        record = logging.LogRecord(
            name="kiara.security",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="LOGIN_SUCCESS user=admin",
            args=None,
            exc_info=None,
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("db down")

        with patch("kiara.shared.db.db_transaction", side_effect=_raise):
            handler.emit(record)  # darf NICHT raisen

    def test_strict_emit_raises_on_db_exception(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler(strict=True)
        handler._min_level_cache = logging.INFO
        record = logging.LogRecord(
            name="kiara.security",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="WORKER_DIED service=kiara.service pid=123 reason=unknown",
            args=None,
            exc_info=None,
        )

        def _raise(*args, **kwargs):
            raise RuntimeError("db down")

        with patch("kiara.shared.db.db_transaction", side_effect=_raise):
            with pytest.raises(RuntimeError, match="db down"):
                handler.emit(record)

    def test_strict_emit_bypasses_min_level_filter(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler(strict=True)
        handler._min_level_cache = logging.CRITICAL
        captured: list = []
        fake = _FakeConnContext(captured)
        record = logging.LogRecord(
            name="kiara.security",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="WORKER_DIED service=kiara.service pid=123 reason=unknown",
            args=None,
            exc_info=None,
        )

        with patch("kiara.shared.db.db_transaction", return_value=fake):
            handler.emit(record)

        assert len(captured) == 1

    def test_strict_emit_raises_when_db_handler_is_disabled(self, monkeypatch):
        monkeypatch.setenv("KIARA_SKIP_DB_LOG_HANDLER", "1")
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler(strict=True)
        record = logging.LogRecord(
            name="kiara.security",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="WORKER_DIED service=kiara.service pid=123 reason=unknown",
            args=None,
            exc_info=None,
        )

        with pytest.raises(RuntimeError, match="KIARA_SKIP_DB_LOG_HANDLER"):
            handler.emit(record)

    def test_emit_source_from_logger_name(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.INFO
        captured: list = []
        fake = _FakeConnContext(captured)

        record = logging.LogRecord(
            name="kiara.ingestion",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="OTHER_EVENT",
            args=None,
            exc_info=None,
        )
        with patch("kiara.shared.db.db_transaction", return_value=fake):
            handler.emit(record)
        assert captured[0][1][1] == "kiara.ingestion"

    def test_exc_info_goes_to_traceback(self, monkeypatch):
        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        from kiara.web.logging_db_handler import SystemEventDBHandler

        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.INFO
        captured: list = []
        fake = _FakeConnContext(captured)

        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="kiara.security",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="LDAP_CONN_FAILED reason=timeout",
            args=None,
            exc_info=exc_info,
        )
        with patch("kiara.shared.db.db_transaction", return_value=fake):
            handler.emit(record)

        details_json = captured[0][1][-1]
        # psycopg2.extras.Json speichert dict in .adapted
        raw = getattr(details_json, "adapted", details_json)
        assert "traceback" in raw
        assert "ValueError" in raw["traceback"]


# ───────────────────────────────────────────────────────────────────
# Logging-Config (propagate=False, Idempotenz)
# ───────────────────────────────────────────────────────────────────


class TestSecurityLoggerSetup:
    def test_propagate_false_set(self):
        import kiara.web.logging_config as lc

        lg = logging.getLogger("kiara.security")
        old_propagate = lg.propagate
        try:
            lg.propagate = True
            lc.setup_security_logger()
            assert lg.propagate is False
        finally:
            lg.propagate = old_propagate

    def test_attach_db_handler_idempotent(self):
        from kiara.web.logging_config import attach_db_handler
        from kiara.web.logging_db_handler import SystemEventDBHandler

        lg = logging.getLogger("kiara.security.test_idempotent")
        lg.handlers.clear()
        attach_db_handler(lg)
        attach_db_handler(lg)
        db_handlers = [h for h in lg.handlers if isinstance(h, SystemEventDBHandler)]
        assert len(db_handlers) == 1

    def test_attach_db_handler_strict_raises_on_attach_failure(self):
        from kiara.web.logging_config import attach_db_handler

        lg = logging.getLogger("kiara.security.test_strict_attach")
        lg.handlers.clear()

        with (
            patch(
                "kiara.web.logging_db_handler.SystemEventDBHandler",
                side_effect=RuntimeError("handler broken"),
            ),
            pytest.raises(RuntimeError, match="angehaengt"),
        ):
            attach_db_handler(lg, strict_db=True)

    def test_attach_db_handler_strict_upgrades_existing_handler(self):
        from kiara.web.logging_config import attach_db_handler
        from kiara.web.logging_db_handler import SystemEventDBHandler

        lg = logging.getLogger("kiara.security.test_strict_upgrade")
        lg.handlers.clear()
        handler = SystemEventDBHandler()
        handler.setLevel(logging.CRITICAL)
        lg.addHandler(handler)

        attach_db_handler(lg, strict_db=True)

        assert handler.strict is True
        assert handler.level == logging.INFO


# ───────────────────────────────────────────────────────────────────
# Filter-Coercion
# ───────────────────────────────────────────────────────────────────


class TestCoerceSystemEventFilters:
    def test_list_input(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"event_types": ["LOGIN_FAILED", "LOGOUT"]})
        assert out["event_types"] == ["LOGIN_FAILED", "LOGOUT"]

    def test_repeated_query_as_list(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"exclude_event_types": ["SYSLOG_VIEW"]})
        assert out["exclude_event_types"] == ["SYSLOG_VIEW"]

    def test_csv_string(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"event_types": "A, B ,C"})
        assert out["event_types"] == ["A", "B", "C"]

    def test_empty_list_ignored(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"event_types": [], "exclude_event_types": ""})
        assert "event_types" not in out
        assert "exclude_event_types" not in out

    def test_unknown_keys_ignored(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"foo": "bar"})
        assert out == {}

    def test_min_level_normalized(self):
        from kiara.repositories.system_events import _coerce_system_event_filters

        out = _coerce_system_event_filters({"min_level": "warning"})
        assert out["min_level"] == "WARNING"
        out2 = _coerce_system_event_filters({"min_level": "banana"})
        assert "min_level" not in out2


# ───────────────────────────────────────────────────────────────────
# Repository-Signatur (syntaktische Prueung, keine DB noetig)
# ───────────────────────────────────────────────────────────────────


class TestSystemEventRepositorySignatures:
    def test_list_accepts_limit_cap_before_kwargs(self):
        """`**filters` muss als letztes Argument stehen (Python-Regel)."""
        import inspect

        from kiara.repositories.system_events import SystemEventRepository

        sig = inspect.signature(SystemEventRepository.list)
        params = list(sig.parameters.values())
        # Letzter Parameter ist VAR_KEYWORD (**filters)
        assert params[-1].kind == inspect.Parameter.VAR_KEYWORD
        # conn=None vorhanden
        assert "conn" in sig.parameters
        assert sig.parameters["conn"].default is None

    def test_exports_in_kiara_db(self):
        import kiara_db

        assert hasattr(kiara_db, "SystemEventRepository")
        assert hasattr(kiara_db, "_coerce_system_event_filters")

    def test_exports_in_kiara_repositories(self):
        import kiara.repositories as kr

        assert hasattr(kr, "SystemEventRepository")
        assert hasattr(kr, "_coerce_system_event_filters")


class TestSystemEventRepositoryWhereNullGuardrail:
    """Sicherheitsnetz gegen 3VL-Filter-Bug bei `exclude_event_types`."""

    def test_exclude_uses_null_safe_clause(self):
        from kiara.repositories.system_events import SystemEventRepository

        repo = SystemEventRepository()
        where, _params = repo._build_where({"exclude_event_types": ["SYSLOG_VIEW"]})
        assert "IS NULL" in where
        assert "<> ALL" in where

    def test_include_sets_standard_equality(self):
        from kiara.repositories.system_events import SystemEventRepository

        repo = SystemEventRepository()
        where, _params = repo._build_where({"event_types": ["LOGIN_FAILED"]})
        assert "event_type = ANY" in where


class TestSystemEventsCsvExport:
    """Backend-Guards fuer den nicht-streamenden CSV-Export."""

    @staticmethod
    def _request(query_string: bytes = b""):
        from starlette.requests import Request

        return Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/admin/system-events/export.csv",
                "query_string": query_string,
                "headers": [],
            }
        )

    def test_export_renders_complete_response_before_return(self, monkeypatch):
        from fastapi.responses import Response, StreamingResponse

        from kiara.web import api_admin_system_events as mod

        captured: dict[str, Any] = {}

        class FakeRepo:
            def list(self, **kwargs):
                captured.update(kwargs)
                return [
                    {
                        "id": 1,
                        "level": "INFO",
                        "event_type": "LOGIN_FAILED",
                        "message": "=SUM(1,1)",
                        "details": {"raw": "ok"},
                    }
                ]

        monkeypatch.setattr(mod, "SystemEventRepository", FakeRepo)
        monkeypatch.setattr(mod, "check_rate_limit", lambda _key: (True, None))

        response = asyncio.run(
            mod.export_system_events_csv(
                self._request(b"event_types=LOGIN_FAILED&event_types=LOGOUT&min_level=warning"),
                user={"username": "admin"},
            )
        )

        assert isinstance(response, Response)
        assert not isinstance(response, StreamingResponse)
        assert response.headers["content-disposition"] == 'attachment; filename="system-events.csv"'
        assert captured["limit"] == 10_000
        assert captured["limit_cap"] == 10_000
        assert captured["event_types"] == ["LOGIN_FAILED", "LOGOUT"]
        assert captured["min_level"] == "WARNING"
        body = response.body.decode("utf-8")
        assert body.startswith(
            "id,ts,level,source,event_type,reason,user_name,ip,message,details\r\n"
        )
        import csv
        import io

        csv_rows = list(csv.reader(io.StringIO(body)))
        assert csv_rows[1][8] == "'=SUM(1,1)"
        assert csv_rows[1][9] == '{"raw": "ok"}'

    def test_export_repo_failure_is_http_error_before_response(self, monkeypatch):
        from fastapi import HTTPException

        from kiara.web import api_admin_system_events as mod

        class FailingRepo:
            def list(self, **_kwargs):
                raise RuntimeError("db disconnect")

        monkeypatch.setattr(mod, "SystemEventRepository", FailingRepo)
        monkeypatch.setattr(mod, "check_rate_limit", lambda _key: (True, None))

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                mod.export_system_events_csv(
                    self._request(),
                    user={"username": "admin"},
                )
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "CSV-Export fehlgeschlagen."

    def test_export_render_failure_is_http_error_before_response(self, monkeypatch):
        from fastapi import HTTPException

        from kiara.web import api_admin_system_events as mod

        class BadValue:
            def __str__(self):
                raise RuntimeError("bad cell")

        class FakeRepo:
            def list(self, **_kwargs):
                return [{"id": 1, "message": BadValue()}]

        monkeypatch.setattr(mod, "SystemEventRepository", FakeRepo)
        monkeypatch.setattr(mod, "check_rate_limit", lambda _key: (True, None))

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(
                mod.export_system_events_csv(
                    self._request(),
                    user={"username": "admin"},
                )
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "CSV-Export fehlgeschlagen."


# ───────────────────────────────────────────────────────────────────
# Rate-Limit
# ───────────────────────────────────────────────────────────────────


class TestRateLimitSyslog:
    def test_allows_up_to_60_then_blocks(self):
        from kiara.web import rate_limit_syslog as rl

        rl.reset_for_tests()
        for _ in range(60):
            allowed, _ = rl.check_rate_limit("user-a")
            assert allowed
        allowed, retry_after = rl.check_rate_limit("user-a")
        assert not allowed
        assert retry_after is not None and retry_after > 0

    def test_separate_users_separate_buckets(self):
        from kiara.web import rate_limit_syslog as rl

        rl.reset_for_tests()
        for _ in range(60):
            rl.check_rate_limit("user-a")
        # user-a blockiert, user-b noch frei
        allowed_a, _ = rl.check_rate_limit("user-a")
        allowed_b, _ = rl.check_rate_limit("user-b")
        assert not allowed_a
        assert allowed_b


# ───────────────────────────────────────────────────────────────────
# AdminFunction-Registrierung + Modul-Liste
# ───────────────────────────────────────────────────────────────────


class TestSystemEventsAdminFunctions:
    def test_module_listed_in_registry_loader(self):
        from kiara.admin import registry as reg

        src = pathlib.Path(reg.__file__).read_text(encoding="utf-8")
        assert "kiara.admin.functions.system_events" in src

    def test_functions_registered_after_load(self):
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        for name in (
            "system.events.list",
            "system.events.get",
            "system.events.distinct_reasons",
            "system.events.distinct_types",
        ):
            assert admin_registry.get(name) is not None, f"AdminFunction '{name}' fehlt"

    def test_list_function_has_pagination_keys_in_schema(self):
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("system.events.list")
        assert func is not None
        props = func.parameters.get("properties", {})
        for key in (
            "page",
            "page_size",
            "sort_field",
            "sort_dir",
            "since",
            "until",
            "min_level",
            "event_types",
            "exclude_event_types",
            "reason",
            "search",
        ):
            assert key in props, f"Schema fehlt '{key}'"

    def test_event_type_filters_use_string_type_for_chat(self):
        """Regression: ToolRegistry._format_params kann keine Type-Listen rendern.

        Auto-Router+Chat brauchen `type: "string"` mit comma-separated Contract.
        """
        from kiara.admin.registry import admin_registry, load_admin_functions

        load_admin_functions()
        func = admin_registry.get("system.events.list")
        props = func.parameters["properties"]
        assert props["event_types"]["type"] == "string"
        assert props["exclude_event_types"]["type"] == "string"


# ───────────────────────────────────────────────────────────────────
# Config-Helper
# ───────────────────────────────────────────────────────────────────


class TestGetSystemEventsConfig:
    def test_defaults_for_missing_block(self):
        from kiara.config import get_system_events_config

        out = get_system_events_config({})
        assert out == {"min_level": "INFO", "retention_days": 90}

    def test_invalid_min_level_fallback(self):
        from kiara.config import get_system_events_config

        out = get_system_events_config({"system_events": {"min_level": "banana"}})
        assert out["min_level"] == "INFO"

    def test_invalid_retention_fallback(self):
        from kiara.config import get_system_events_config

        out = get_system_events_config({"system_events": {"retention_days": -1}})
        assert out["retention_days"] == 90

    def test_accepts_string_retention(self):
        from kiara.config import get_system_events_config

        out = get_system_events_config({"system_events": {"retention_days": "30"}})
        assert out["retention_days"] == 30


# ───────────────────────────────────────────────────────────────────
# Frontend-Strukturchecks
# ───────────────────────────────────────────────────────────────────


_FRONTEND_BASE = pathlib.Path(__file__).resolve().parent.parent
TAB_SYSTEM_LOGS_JS = _FRONTEND_BASE / "kiara" / "web" / "static" / "js" / "tab_system_logs.js"
ADMIN_HTML = _FRONTEND_BASE / "kiara" / "web" / "templates" / "admin.html"


@pytest.fixture(scope="module")
def system_logs_src():
    return TAB_SYSTEM_LOGS_JS.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def admin_html_src():
    return ADMIN_HTML.read_text(encoding="utf-8")


class TestSystemLogsTabStructure:
    def test_file_exists(self, system_logs_src):
        assert len(system_logs_src) > 0

    def test_init_function_top_level(self, system_logs_src):
        assert "async function initSystemLogsTab()" in system_logs_src

    def test_run_tab_cleanups_at_start(self, system_logs_src):
        assert "_runTabCleanups('system-logs')" in system_logs_src

    def test_register_tab_cleanup(self, system_logs_src):
        assert "registerTabCleanup('system-logs'" in system_logs_src

    def test_sl_init_seq_top_level(self, system_logs_src):
        assert re.search(r"let\s+_slInitSeq\s*=\s*0", system_logs_src)

    def test_sl_dom_generation_top_level(self, system_logs_src):
        assert re.search(r"let\s+_slDomGeneration\s*=\s*0", system_logs_src)

    def test_expandable_table_instantiated(self, system_logs_src):
        assert "new SharedUi.table.ExpandableTable(" in system_logs_src

    def test_expandable_table_id(self, system_logs_src):
        assert "id: 'system-logs'" in system_logs_src

    def test_cleanup_deletes_instance(self, system_logs_src):
        assert "SharedUi.table.instances['system-logs']" in system_logs_src

    def test_system_log_filter_bar_class(self, system_logs_src):
        assert "class SystemLogFilterBar" in system_logs_src
        for method in ("render()", "getValues()", "setValues(obj)", "destroy()"):
            assert method in system_logs_src, f"Methode {method} fehlt in SystemLogFilterBar"

    def test_render_json_detail_function(self, system_logs_src):
        assert "function renderJsonDetail(" in system_logs_src
        assert "SharedUi.runtime.escapeHtml" in system_logs_src

    def test_render_json_detail_no_inline_handlers(self, system_logs_src):
        # Body von renderJsonDetail nutzt <details>/<summary>, keine onclick/onchange.
        # Grobes Negativ-Pattern
        match = re.search(
            r"function renderJsonDetail\([^)]*\)\s*\{(.+?)\n\}\s*\n",
            system_logs_src,
            re.DOTALL,
        )
        assert match, "renderJsonDetail-Body nicht auffindbar"
        body = match.group(1)
        assert "onclick=" not in body
        assert "onchange=" not in body

    def test_no_iife_wrapping(self, system_logs_src):
        # Kein "(function(){" am Dateianfang
        head = system_logs_src.strip().splitlines()[0:5]
        joined = "\n".join(head)
        assert "(function" not in joined

    def test_csv_via_window_open(self, system_logs_src):
        assert "window.open(" in system_logs_src
        assert "/api/admin/system-events/export.csv" in system_logs_src

    def test_message_uses_builtin_truncate(self, system_logs_src):
        # truncate: 80 Property in der Column-Konfig
        assert re.search(r"truncate\s*:\s*80", system_logs_src)
        # Kein eigener Helper namens truncateWithEllipsis
        assert "truncateWithEllipsis" not in system_logs_src

    def test_auto_refresh_uses_kiara_polling(self, system_logs_src):
        assert "kiaraPolling.start('system-logs:refresh'" in system_logs_src
        assert "kiaraPolling.stop('system-logs:refresh')" in system_logs_src
        assert "adminCurrentSubTab !== 'system-logs'" in system_logs_src

    def test_manual_filter_resets_page_to_one(self, system_logs_src):
        assert "slTable.pagination.currentPage = 1" in system_logs_src

    def test_filter_bar_body_is_product_neutral(self, system_logs_src):
        # Class-Body soll keine Log/Security-spezifischen Literale enthalten
        match = re.search(
            r"class SystemLogFilterBar\s*\{(.+?)\n\}\s*\n",
            system_logs_src,
            re.DOTALL,
        )
        assert match, "SystemLogFilterBar-Body nicht auffindbar"
        body = match.group(1)
        for bad in ("security", "event_type", "LOGIN", "LDAP", "system_events"):
            assert bad not in body, f"Produktneutral verletzt: '{bad}' in SystemLogFilterBar"

    def test_admin_html_tab_config(self, admin_html_src):
        assert "'system-logs':" in admin_html_src
        assert "initSystemLogsTab" in admin_html_src

    def test_admin_html_tab_info(self, admin_html_src):
        # In TAB_INFO steht ein Eintrag fuer system-logs
        assert "'system-logs':" in admin_html_src

    def test_admin_html_script_tag(self, admin_html_src):
        assert 'src="/static/js/tab_system_logs.js' in admin_html_src
