"""Tests fuer den Frontend-Beacon-Endpunkt POST /api/chat-debug/client-error.

Der Endpunkt nimmt einen Bericht ueber Stream-Abrisse oder andere
clientseitige Fehler entgegen, die der Server selbst nicht sieht (z.B.
Sub-Worker-Tod mid-stream, Connection-Reset im Browser). Persistenz erfolgt
ueber `security_log.error` -> `SystemEventDBHandler` -> `system_events`.

Tests verifizieren ohne echte DB:
- 200 + {"ok": True} bei valider Payload, security_log.error wurde aufgerufen.
- Pydantic-Validation kuerzt error_message auf max_length=2000.
- Auth wird durchgereicht (401 wenn get_chat_user wirft).
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from kiara.web.api_chat import router
from kiara.web.auth import get_chat_user


@pytest.fixture
def app_with_user():
    """FastAPI-App mit gemocktem Chat-User. security_log liegt im api_chat-Modul."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_user] = lambda: {
        "id": 7,
        "username": "alice",
        "is_admin": False,
        "datasource_ids": [],
    }
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app_with_user):
    with TestClient(app_with_user) as c:
        yield c


# ---------------------------------------------------------------------------
# Happy Path
# ---------------------------------------------------------------------------


def test_happy_path_returns_ok_and_logs_chat_client_error(client):
    """Valider Payload -> 200 + ok=True + security_log.error.

    Nach Audit-Run-2 Finding #3: Untrusted Beacon-Felder (mode, kind,
    chat_id, error_message) wandern in kiara_details, NICHT ins KV-Format.
    Die Log-Message enthaelt nur noch den authentifizierten Username.
    """
    payload = {
        "chat_id": 42,
        "error_kind": "http",
        "error_message": "Netzwerkfehler bei der Abfrage",
        "mode": "agent",
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)

    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    security_log.error.assert_called_once()
    args, kwargs = security_log.error.call_args
    assert args[0] == "CHAT_CLIENT_ERROR user=%s"
    assert args[1:] == ("alice",)
    extra = kwargs["extra"]["kiara_details"]
    assert extra["mode"] == "agent"
    assert extra["kind"] == "http"
    assert extra["chat_id"] == "42"
    assert extra["error_message"] == "Netzwerkfehler bei der Abfrage"


def test_chat_id_none_becomes_unknown(client):
    """Fehlende chat_id wird auf 'unknown' gemappt (kein Crash)."""
    payload = {"error_kind": "unauthorized", "error_message": "401", "mode": "agent"}
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 200
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    assert extra["chat_id"] == "unknown"


def test_chat_id_string_is_truncated_to_64_chars(client):
    """chat_id als String wird auf 64 Zeichen begrenzt (DB-Vertrag)."""
    payload = {
        "chat_id": "x" * 200,
        "error_kind": "http",
        "error_message": "boom",
        "mode": "agent",
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 200
    chat_id_extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]["chat_id"]
    assert len(chat_id_extra) == 64
    assert chat_id_extra == "x" * 64


# ---------------------------------------------------------------------------
# Pydantic Limits
# ---------------------------------------------------------------------------


def test_error_message_over_2000_chars_is_rejected_by_pydantic(client):
    """error_message hat max_length=2000 -> 422 bei Ueberlaenge."""
    payload = {
        "error_kind": "http",
        "error_message": "x" * 2001,
        "mode": "agent",
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 422
    security_log.error.assert_not_called()


def test_error_message_at_2000_chars_passes(client):
    """Genau 2000 Zeichen ist erlaubt. error_message wird beim Logging
    auf 1000 Zeichen gekappt (kiara_details["error_message"][:1000]).
    """
    payload = {
        "error_kind": "http",
        "error_message": "y" * 2000,
        "mode": "agent",
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 200
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    assert len(extra["error_message"]) == 1000


def test_error_kind_outside_allowlist_is_rejected(client):
    """error_kind ist Literal — unbekannte Werte -> 422 (Audit-Run-2 #3)."""
    payload = {
        "error_kind": "evil_injection",
        "error_message": "x",
        "mode": "agent",
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 422
    security_log.error.assert_not_called()


def test_mode_outside_allowlist_is_rejected(client):
    """mode ist Literal — unbekannte Werte -> 422 (Audit-Run-2 #3)."""
    payload = {
        "error_kind": "http",
        "error_message": "x",
        "mode": "agent reason=oom_killed",  # KV-Injection-Versuch
    }
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json=payload)
    assert resp.status_code == 422
    security_log.error.assert_not_called()


def test_empty_payload_uses_defaults(client):
    """Pydantic liefert Defaults fuer fehlende Felder; logging darf nicht crashen."""
    with patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post("/api/chat-debug/client-error", json={})
    assert resp.status_code == 200
    args, kwargs = security_log.error.call_args
    assert args[0] == "CHAT_CLIENT_ERROR user=%s"
    assert args[1:] == ("alice",)
    extra = kwargs["extra"]["kiara_details"]
    assert extra["mode"] == "unknown"
    assert extra["chat_id"] == "unknown"
    assert extra["kind"] == "unknown"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


def test_unauthenticated_request_returns_401():
    app = FastAPI()
    app.include_router(router)

    async def deny():
        raise HTTPException(status_code=401, detail="Nicht angemeldet")

    app.dependency_overrides[get_chat_user] = deny
    with TestClient(app) as client, patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post(
            "/api/chat-debug/client-error",
            json={
                "error_kind": "http",
                "error_message": "x",
                "mode": "agent",
            },
        )
    assert resp.status_code == 401
    security_log.error.assert_not_called()


def test_admin_user_forbidden_403():
    app = FastAPI()
    app.include_router(router)

    async def deny_admin():
        raise HTTPException(status_code=403, detail="Admin-Konto")

    app.dependency_overrides[get_chat_user] = deny_admin
    with TestClient(app) as client:
        resp = client.post(
            "/api/chat-debug/client-error",
            json={
                "error_kind": "http",
                "error_message": "x",
                "mode": "agent",
            },
        )
    assert resp.status_code == 403


def test_user_without_username_falls_back_to_questionmark():
    """user dict ohne `username`-Key -> user='?' im Log-Format."""
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_chat_user] = lambda: {"id": 1, "is_admin": False}

    with TestClient(app) as client, patch("kiara.web.api_chat.security_log") as security_log:
        resp = client.post(
            "/api/chat-debug/client-error",
            json={
                "error_kind": "http",
                "error_message": "x",
                "mode": "agent",
            },
        )
    assert resp.status_code == 200
    assert security_log.error.call_args.args[1] == "?"
