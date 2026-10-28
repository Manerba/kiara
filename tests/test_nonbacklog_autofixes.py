"""Regressionstests fuer einfache Nicht-Backlog-Bugfixes."""

import asyncio
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException


def test_compliance_audit_rejects_invalid_date_filter():
    from kiara.admin.functions.compliance import compliance_audit

    result = asyncio.run(
        compliance_audit(
            {"id": 1, "username": "admin"},
            date_from="2026-99-99",
        )
    )

    assert result.success is False
    assert result.http_status == 400
    assert "YYYY-MM-DD" in result.message


def test_wopi_lock_ids_are_validated_and_safe_headers_sanitized():
    from kiara.web.api_onlyoffice import (
        _safe_wopi_lock_header,
        _validate_wopi_lock_id,
    )

    assert _validate_wopi_lock_id("lock_ABC-123.4") == "lock_ABC-123.4"
    with pytest.raises(HTTPException):
        _validate_wopi_lock_id("bad\nlock")
    with pytest.raises(HTTPException):
        _validate_wopi_lock_id("x" * 1025)
    assert _safe_wopi_lock_header("bad\rlock") == ""


@pytest.mark.asyncio
async def test_wopi_get_file_enforces_size_during_read(tmp_path, monkeypatch):
    from kiara.web import api_onlyoffice as api

    file_path = tmp_path / "doc.txt"
    file_path.write_bytes(b"abcdef")

    class FakeRequest:
        headers = {}

    async def fake_proof(_request, _access_token):
        return None

    monkeypatch.setattr(api, "MAX_WOPI_FILE_SIZE", 5)
    monkeypatch.setattr(
        api,
        "_validate_wopi_token",
        lambda _token, _file_id: {
            "user_id": 1,
            "username": "alice",
            "file_id": "file1",
            "file_path": str(file_path),
        },
    )
    monkeypatch.setattr(api, "_validate_wopi_proof", fake_proof)
    monkeypatch.setattr(api, "_load_wopi_user_with_acl", lambda _user_id: {"id": 1})
    monkeypatch.setattr(api, "_check_datasource_acl", lambda _path, _user: None)

    with pytest.raises(HTTPException) as exc:
        await api.wopi_get_file("file1", FakeRequest(), access_token="token")

    assert exc.value.status_code == 413


def test_webdav_301_is_not_success(monkeypatch):
    import httpx

    from kiara.connectors.webdav_connector import WebDAVConnector

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def request(self, *args, **kwargs):
            return SimpleNamespace(status_code=301)

    monkeypatch.setattr(httpx, "Client", FakeClient)

    result = WebDAVConnector.test_connection({"url": "https://dav.example/"}, {})

    assert result["success"] is False
    assert result["message"] == "HTTP 301"


def test_local_connector_wraps_config_errors(monkeypatch):
    from kiara.connectors.browse import ConnectorBrowseError
    from kiara.connectors.local_connector import LocalConnector

    def fail_load_config():
        raise KeyError("datasource")

    monkeypatch.setattr("kiara.config.load_config", fail_load_config)

    with pytest.raises(ConnectorBrowseError) as exc:
        LocalConnector.list_directory({})

    assert exc.value.reason == "config_error"


def test_mail_list_folders_classifies_dns_failure_as_timeout():
    from kiara.connectors.base import mail_list_folders

    class FakeConnector:
        @staticmethod
        def _build_scanner_config(config):
            return config

    class FakeScanner:
        def __init__(self, _config):
            pass

        def connect(self):
            raise socket.gaierror("dns failed")

    with pytest.raises(RuntimeError, match="nicht erreichbar"):
        mail_list_folders(FakeConnector, FakeScanner, "IMAP-Server", {})


def test_sql_timeout_setup_failure_is_logged_without_secret(monkeypatch, caplog):
    from kiara.connectors.sql_connector import SQLConnector

    cursor = MagicMock()
    cursor.execute.side_effect = RuntimeError("PWD=supersecret")
    conn = MagicMock()
    conn.autocommit = False
    conn.cursor.return_value = cursor
    psycopg2 = SimpleNamespace(connect=MagicMock(return_value=conn))
    monkeypatch.setitem(sys.modules, "psycopg2", psycopg2)

    connector = SQLConnector(
        "postgresql://user:supersecret@db.example/erp",
        "postgresql",
        statement_timeout=5,
    )

    with caplog.at_level("ERROR"), pytest.raises(RuntimeError):
        connector._connect()

    assert "SQL-Statement-Timeout-Setup fehlgeschlagen" in caplog.text
    assert "supersecret" not in caplog.text


def test_frontend_autofix_markers_present():
    base = Path("kiara/web/static/js")
    ds_src = (base / "tab_datenquellen.js").read_text(encoding="utf-8")
    anon_src = (base / "tab_compliance_anon.js").read_text(encoding="utf-8")
    admin_chat_src = (base / "admin_chat.js").read_text(encoding="utf-8")

    assert "Promise.allSettled" in ds_src
    assert "function _parseDsTimestamp" in ds_src
    assert "SharedUi.runtime.withButtonLock(btn, () => _openBrowsePicker" in ds_src
    assert "const existingPicker = document.getElementById('fpOverlay')" in ds_src
    assert "_dsResolveError(e.body || e" in ds_src
    assert "SharedUi.runtime.withButtonLock(el, saveComplianceAnonConfig)" in anon_src
    assert "if (this._destroyed) return;" in admin_chat_src


def test_degradation_and_collector_autofix_markers_present():
    degradation_src = Path("kiara/resilience/degradation.py").read_text(encoding="utf-8")
    collector_src = Path("kiara/monitoring/collector.py").read_text(encoding="utf-8")

    assert "self._state_lock = threading.RLock()" in degradation_src
    assert "with self._state_lock:" in degradation_src
    assert "last_persist = time.monotonic()" in collector_src
    assert "% METRICS_PERSIST_INTERVAL_SECS" not in collector_src


def test_schema_migration_lock_keys_and_inline_junction_pks():
    core_src = Path("kiara/web/core_schema_migration.py").read_text(encoding="utf-8")
    group_src = Path("kiara/web/group_schema_migration.py").read_text(encoding="utf-8")
    graph_src = Path("kiara/web/graph_schema_migration.py").read_text(encoding="utf-8")

    assert "_MIGRATION_LOCK_KEY = 681031" in core_src
    assert "_MIGRATION_LOCK_KEY = 681032" in group_src
    assert "_MIGRATION_LOCK_KEY = 681033" in graph_src
    assert "PRIMARY KEY (node_id, chunk_id)" in graph_src
    assert "PRIMARY KEY (edge_id, chunk_id)" in graph_src
