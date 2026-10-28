from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from kiara.shared.constants import ADMIN_PORT_DEFAULT
from kiara.web.api_chat import router
from kiara.web.auth import get_chat_user


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_user] = lambda: {
        "id": 7,
        "is_admin": False,
        "datasource_ids": [],
    }
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_status_disabled_returns_false(client):
    with patch(
        "kiara.web.api_chat.load_config", return_value={"tracing": {"chat_debug": False}, "web": {}}
    ):
        resp = client.get("/api/chat-debug/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": False, "admin_port": ADMIN_PORT_DEFAULT}


def test_status_enabled_returns_configured_admin_port(client):
    with patch(
        "kiara.web.api_chat.load_config",
        return_value={
            "tracing": {"chat_debug": True},
            "web": {"admin_port": 9999},
        },
    ):
        resp = client.get("/api/chat-debug/status")
    assert resp.status_code == 200
    assert resp.json() == {"enabled": True, "admin_port": 9999}


def test_trace_endpoint_404_when_debug_disabled(client):
    with patch("kiara.web.api_chat.load_config", return_value={"tracing": {"chat_debug": False}}):
        resp = client.get("/api/chat-debug/trace/abc123")
    assert resp.status_code == 404


def test_trace_endpoint_404_for_foreign_trace_id(client):
    with (
        patch("kiara.web.api_chat.load_config", return_value={"tracing": {"chat_debug": True}}),
        patch("kiara.web.api_chat.db_query") as db,
    ):
        db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = None
        resp = client.get("/api/chat-debug/trace/foreign-trace")
    assert resp.status_code == 404


def test_trace_endpoint_returns_owned_trace(client):
    trace_payload = {"trace_id": "abc123", "root_span": "rag_query", "spans": []}
    with (
        patch("kiara.web.api_chat.load_config", return_value={"tracing": {"chat_debug": True}}),
        patch("kiara.web.api_chat.db_query") as db,
        patch("kiara.web.api_tracing._get_trace_by_id", return_value=trace_payload),
    ):
        db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {
            "?column?": 1
        }
        resp = client.get("/api/chat-debug/trace/abc123")
    assert resp.status_code == 200
    assert resp.json()["trace_id"] == "abc123"


def test_status_requires_auth():
    app = FastAPI()
    app.include_router(router)

    async def deny():
        raise HTTPException(status_code=401, detail="Nicht angemeldet")

    app.dependency_overrides[get_chat_user] = deny
    with TestClient(app) as client:
        resp = client.get("/api/chat-debug/status")
    assert resp.status_code == 401


def test_status_forbidden_for_admin_user():
    app = FastAPI()
    app.include_router(router)

    async def deny_admin():
        raise HTTPException(
            status_code=403,
            detail="Admin-Konten haben keinen Chat-Zugriff. Bitte einen regulaeren Benutzer verwenden.",
        )

    app.dependency_overrides[get_chat_user] = deny_admin
    with TestClient(app) as client:
        resp = client.get("/api/chat-debug/status")
    assert resp.status_code == 403


def test_trace_requires_auth():
    app = FastAPI()
    app.include_router(router)

    async def deny():
        raise HTTPException(status_code=401, detail="Nicht angemeldet")

    app.dependency_overrides[get_chat_user] = deny
    with TestClient(app) as client:
        resp = client.get("/api/chat-debug/trace/abc123")
    assert resp.status_code == 401


def test_trace_forbidden_for_admin_user():
    app = FastAPI()
    app.include_router(router)

    async def deny_admin():
        raise HTTPException(
            status_code=403,
            detail="Admin-Konten haben keinen Chat-Zugriff. Bitte einen regulaeren Benutzer verwenden.",
        )

    app.dependency_overrides[get_chat_user] = deny_admin
    with TestClient(app) as client:
        resp = client.get("/api/chat-debug/trace/abc123")
    assert resp.status_code == 403


def test_trace_endpoint_404_when_spans_deleted_by_retention(client):
    """User besitzt den Trace, aber Spans wurden per Retention geloescht."""
    with (
        patch("kiara.web.api_chat.load_config", return_value={"tracing": {"chat_debug": True}}),
        patch("kiara.web.api_chat.db_query") as db,
        patch("kiara.web.api_tracing._get_trace_by_id", return_value=None),
    ):
        db.return_value.__enter__.return_value.execute.return_value.fetchone.return_value = {
            "?column?": 1
        }
        resp = client.get("/api/chat-debug/trace/expired-trace")
    assert resp.status_code == 404


class TestSafeTraceId:
    """Unit-Tests fuer _safe_trace_id() Grenzfaelle."""

    def test_valid_string(self):
        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id("abc123") == "abc123"

    def test_empty_string(self):
        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id("") is None

    def test_none(self):
        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id(None) is None

    def test_integer(self):
        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id(42) is None

    def test_mock_object(self):
        from unittest.mock import MagicMock

        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id(MagicMock()) is None

    def test_bool_false(self):
        from kiara.web.api_chat import _safe_trace_id

        assert _safe_trace_id(False) is None
