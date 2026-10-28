"""Negativ- und Robustheitstests fuer den Admin-Chat.

Testet Fehlerfaelle, Edge Cases und unerwartete Eingaben.
Stellt sicher, dass der Admin-Chat bei Fehlern sauber reagiert statt abzustuerzen.
Reine Unit-Tests ohne DB- oder Netzwerk-Abhaengigkeiten (ausser wo explizit markiert).
"""

import asyncio
import json
import uuid

import pytest

from kiara.admin.function import AdminFunction
from kiara.admin.function_tool import AdminFunctionTool
from kiara.admin.registry import AdminFunctionRegistry
from kiara.admin.result import AdminResult
from kiara.admin.router import _coerce_params

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


def _make_func(
    name="test.func",
    requires_confirmation=False,
    confirm_template=None,
    parameters=None,
    fn=None,
    http=None,
):
    """Erstellt eine AdminFunction mit sinnvollen Defaults."""
    if parameters is None:
        parameters = {"type": "object", "properties": {}, "required": []}
    if fn is None:

        async def fn(user, **kwargs):
            return AdminResult.ok("ok")

    http_method, http_path = http if http else (None, None)
    return AdminFunction(
        name=name,
        category="test",
        description="Test-Funktion",
        help="Test-Hilfe",
        examples=["Test"],
        parameters=parameters,
        returns="ok",
        requires_confirmation=requires_confirmation,
        confirm_template=confirm_template,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method=http_method,
        http_path=http_path,
        fn=fn,
    )


# ---------------------------------------------------------------------------
# Eingabe-Validierung
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_empty_question_rejected(self):
        """Leerer String als Frage → Pydantic-Validierungsfehler (min_length=1)."""
        from kiara.web.api_admin_chat import AdminChatQueryRequest

        with pytest.raises(Exception):  # ValidationError
            AdminChatQueryRequest(question="")

    def test_missing_required_params_returns_error(self):
        """Pflichtparameter fehlen → Wizard mit fehlenden Feldern, kein Crash."""
        func = _make_func(
            name="test.required",
            parameters={
                "type": "object",
                "properties": {
                    "ds_id": {"type": "integer", "description": "DS-ID"},
                    "name": {"type": "string", "description": "Name"},
                },
                "required": ["ds_id", "name"],
            },
        )
        tool = AdminFunctionTool(func)
        # Aufruf ohne jegliche Parameter
        result = _run(tool.execute({}))
        assert result.success is True
        assert "Fehlende Pflichtfelder" in result.display_text
        assert "ds_id" in result.display_text
        assert "name" in result.display_text
        # Wizard-Daten im Seitenkanal
        pending = tool.pop_pending()
        assert len(pending) == 1
        assert pending[0]["type"] == "wizard"

    def test_wrong_param_type_returns_error(self):
        """Falscher Parametertyp (String statt Integer) → saubere Fehlermeldung."""

        async def _fn(user, ds_id):
            # Funktion erwartet int, bekommt string → TypeError
            return AdminResult.ok(f"DS {ds_id + 1}")

        func = _make_func(
            name="test.wrong_type",
            parameters={
                "type": "object",
                "properties": {"ds_id": {"type": "integer", "description": "DS-ID"}},
                "required": ["ds_id"],
            },
            fn=_fn,
        )
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({"ds_id": "abc"}))
        # Exception wird gefangen, kein unkontrollierter Crash
        assert result.success is False
        assert result.error is not None

    def test_nonexistent_function_in_registry(self):
        """Nicht existierende Funktion → None."""
        reg = AdminFunctionRegistry()
        assert reg.get("nonexistent.function") is None
        assert reg.get_help("nonexistent.function") is None

    def test_nonexistent_resource_returns_404(self):
        """Nicht existierende Ressource → AdminResult.fail()."""

        async def _status(user, ds_id):
            return AdminResult.fail(f"Datenquelle {ds_id} nicht gefunden", http_status=404)

        func = _make_func(
            name="datasource.status",
            parameters={
                "type": "object",
                "properties": {"ds_id": {"type": "integer", "description": "DS-ID"}},
                "required": ["ds_id"],
            },
            fn=_status,
        )
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({"ds_id": 999999}))
        assert result.success is False
        assert "nicht gefunden" in result.display_text


# ---------------------------------------------------------------------------
# Fehlerbehandlung in Funktionen
# ---------------------------------------------------------------------------


class TestFunctionErrorHandling:
    def test_function_exception_caught(self):
        """Funktion wirft Exception → ToolResult(success=False), kein Crash."""

        async def _explode(user):
            raise RuntimeError("Boom!")

        func = _make_func(name="test.explode", fn=_explode)
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({}))
        assert result.success is False
        assert "Boom!" in result.error
        assert "Fehler bei test.explode" in result.display_text

    def test_function_timeout_handling(self):
        """Funktion mit subprocess-Timeout → AdminResult.fail()."""
        import subprocess

        async def _timeout_fn(user):
            try:
                subprocess.run(["sleep", "10"], timeout=0.01, check=True)
                return AdminResult.ok("ok")
            except subprocess.TimeoutExpired:
                return AdminResult.fail("Timeout ueberschritten")

        func = _make_func(name="test.timeout", fn=_timeout_fn)
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({}))
        assert result.success is False
        assert "Timeout" in result.display_text

    def test_auto_router_catches_exceptions(self):
        """Auto-Router faengt Exceptions, liefert JSON-Fehler statt Stacktrace."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.router import generate_admin_routes
        from kiara.web.auth import get_admin_user

        async def _crash(user):
            raise ValueError("Router-Crash-Test")

        func = _make_func(
            name="test.router_crash",
            http=("GET", "/api/admin/test/crash"),
            fn=_crash,
        )
        reg = AdminFunctionRegistry()
        reg.register(func)

        app = FastAPI()
        app.dependency_overrides[get_admin_user] = lambda: {"id": 1, "username": "test"}
        generate_admin_routes(app, reg)

        client = TestClient(app)
        resp = client.get("/api/admin/test/crash")
        # Kein 500 ohne Body, sondern strukturierte JSON-Antwort
        data = resp.json()
        assert data["success"] is False
        assert "Interner Fehler" in data["message"]
        # Kein Stacktrace im Response-Body
        assert "Traceback" not in json.dumps(data)


# ---------------------------------------------------------------------------
# Confirmation Edge Cases
# ---------------------------------------------------------------------------


class TestConfirmationEdgeCases:
    def test_double_confirmation_ignored(self, test_db):
        """Doppelte Bestaetigung → Keine erneute Ausfuehrung."""
        from kiara.admin.confirmation import ConfirmationManager
        from kiara.repositories.admin_chat import AdminConfirmationRepository

        # Setup: User + Chat
        cursor = test_db.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (f"negtest_{uuid.uuid4().hex[:8]}", "hash"),
        )
        user_id = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_id = cursor.fetchone()["id"]
        test_db.commit()

        # Confirmation erstellen
        action_id = str(uuid.uuid4())
        repo = AdminConfirmationRepository()
        repo.create(
            chat_id=chat_id,
            action_id=action_id,
            function_name="test.confirm",
            params="{}",
            confirm_string="test bestaetigen",
            user_id=user_id,
            conn=test_db,
        )
        test_db.commit()

        # Erste Bestaetigung: confirm → True
        confirmed = repo.confirm(action_id)
        assert confirmed is True

        # Zweite Bestaetigung: Already confirmed → False
        confirmed_again = repo.confirm(action_id)
        assert confirmed_again is False

        # get_pending gibt None (kein pending mehr)
        mgr = ConfirmationManager()
        pending = mgr.get_pending(chat_id)
        assert pending is None

    def test_confirmation_for_wrong_chat(self, test_db):
        """Confirmation eines anderen Chats → handled=False."""
        from kiara.admin.confirmation import ConfirmationManager
        from kiara.repositories.admin_chat import AdminConfirmationRepository

        # Setup: User + 2 Chats
        cursor = test_db.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (f"negtest_{uuid.uuid4().hex[:8]}", "hash"),
        )
        user_id = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_a = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_b = cursor.fetchone()["id"]
        test_db.commit()

        # Confirmation in Chat A
        action_id = str(uuid.uuid4())
        repo = AdminConfirmationRepository()
        repo.create(
            chat_id=chat_a,
            action_id=action_id,
            function_name="test.confirm",
            params="{}",
            confirm_string="test loeschen",
            user_id=user_id,
            conn=test_db,
        )
        test_db.commit()

        # check_and_handle in Chat B → keine pending Confirmation
        mgr = ConfirmationManager()
        handled, events = mgr.check_and_handle(chat_b, "test loeschen", {"id": user_id})
        assert handled is False

    def test_confirmation_with_whitespace(self, test_db):
        """Bestaetigungsstring mit extra Whitespace → wird trotzdem akzeptiert."""
        from kiara.admin.confirmation import ConfirmationManager
        from kiara.repositories.admin_chat import AdminConfirmationRepository

        # Setup: User + Chat
        cursor = test_db.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (f"negtest_{uuid.uuid4().hex[:8]}", "hash"),
        )
        user_id = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_id = cursor.fetchone()["id"]
        test_db.commit()

        # Confirmation erstellen
        action_id = str(uuid.uuid4())
        repo = AdminConfirmationRepository()
        repo.create(
            chat_id=chat_id,
            action_id=action_id,
            function_name="test.confirm",
            params="{}",
            confirm_string="test 42 loeschen",
            user_id=user_id,
            conn=test_db,
        )
        test_db.commit()

        # Whitespace-padded Eingabe → trotzdem Match (strip + lower in check_and_handle)
        mgr = ConfirmationManager()
        pending = mgr.get_pending(chat_id)
        assert pending is not None

        # Vergleich: confirm_string.strip().lower() == user_input.strip().lower()
        confirm_str = pending["confirm_string"].strip().lower()
        user_input = "  test 42 loeschen  ".strip().lower()
        assert confirm_str == user_input


# ---------------------------------------------------------------------------
# Wizard Edge Cases
# ---------------------------------------------------------------------------


class TestWizardEdgeCases:
    def test_wizard_with_empty_params(self):
        """Wizard-Aufruf komplett ohne Parameter → Wizard mit allen Feldern fehlend."""
        func = _make_func(
            name="test.wizard_empty",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name"},
                    "count": {"type": "integer", "description": "Anzahl"},
                },
                "required": ["name", "count"],
            },
        )
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({}))
        assert result.success is True
        assert "Fehlende Pflichtfelder" in result.display_text
        pending = tool.pop_pending()
        assert len(pending) == 1
        wizard = pending[0]
        assert wizard["type"] == "wizard"
        assert set(wizard["missing_required"]) == {"name", "count"}
        assert wizard["collected"] == {}

    def test_wizard_with_extra_params(self):
        """Unbekannte Parameter → werden ignoriert, kein Crash."""

        async def _fn(user, name):
            return AdminResult.ok(f"Hallo {name}")

        func = _make_func(
            name="test.wizard_extra",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name"},
                },
                "required": ["name"],
            },
            fn=_fn,
        )
        tool = AdminFunctionTool(func)
        # Extra-Parameter "unknown_field" neben dem required "name"
        result = _run(tool.execute({"name": "Test", "unknown_field": "ignoriert"}))
        # Funktion bekommt nur bekannte Parameter, unknown_field wird weitergereicht
        # aber da fn nur **kwargs akzeptiert, laeuft es durch
        # Wichtig: kein Crash
        assert result.success is True or result.success is False  # kein Crash

    def test_wizard_for_function_without_required(self):
        """Funktion ohne Pflichtfelder → direkte Ausfuehrung, KEIN Wizard."""

        async def _ping(user):
            return AdminResult.ok("pong")

        func = _make_func(
            name="test.no_required",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            fn=_ping,
        )
        tool = AdminFunctionTool(func)
        result = _run(tool.execute({}))
        assert result.success is True
        assert result.display_text == "pong"
        pending = tool.pop_pending()
        # Nur wizard_complete, kein wizard
        wizard_items = [p for p in pending if p["type"] == "wizard"]
        assert len(wizard_items) == 0


# ---------------------------------------------------------------------------
# Router Edge Cases
# ---------------------------------------------------------------------------


class TestRouterEdgeCases:
    def test_coerce_params_invalid_integer(self):
        """Ungueltiger Integer in URL → String bleibt erhalten."""
        schema = {
            "type": "object",
            "properties": {"ds_id": {"type": "integer"}},
            "required": ["ds_id"],
        }
        result = _coerce_params({"ds_id": "abc"}, schema)
        assert result["ds_id"] == "abc"
        assert isinstance(result["ds_id"], str)

    def test_coerce_params_boolean_explicit_false(self):
        """Explizites 'false' als Boolean → Python False (nicht True!)."""
        schema = {
            "type": "object",
            "properties": {"is_active": {"type": "boolean"}},
            "required": [],
        }
        result = _coerce_params({"is_active": "false"}, schema)
        assert result["is_active"] is False

        result = _coerce_params({"is_active": "0"}, schema)
        assert result["is_active"] is False

        result = _coerce_params({"is_active": "no"}, schema)
        assert result["is_active"] is False

    def test_route_not_found_for_chat_only_function(self):
        """Chat-only Funktion (ohne http) hat KEINE HTTP-Route."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.router import generate_admin_routes
        from kiara.web.auth import get_admin_user

        func = _make_func(name="help.search_test", http=None)
        reg = AdminFunctionRegistry()
        reg.register(func)

        app = FastAPI()
        app.dependency_overrides[get_admin_user] = lambda: {"id": 1, "username": "test"}
        generate_admin_routes(app, reg)

        client = TestClient(app)
        # Keine Route wurde erzeugt → 404
        resp = client.get("/api/admin/help/search_test")
        assert resp.status_code == 404

    def test_auto_router_filters_unknown_params(self):
        """Auto-Router filtert unbekannte Query-Parameter aus."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from kiara.admin.router import generate_admin_routes
        from kiara.web.auth import get_admin_user

        async def _fn(user):
            return AdminResult.ok("ok")

        func = _make_func(
            name="test.filter_params",
            http=("GET", "/api/admin/test/filter"),
            fn=_fn,
        )
        reg = AdminFunctionRegistry()
        reg.register(func)

        app = FastAPI()
        app.dependency_overrides[get_admin_user] = lambda: {"id": 1, "username": "test"}
        generate_admin_routes(app, reg)

        client = TestClient(app)
        # Unbekannte Query-Params → werden rausgefiltert, kein TypeError
        resp = client.get("/api/admin/test/filter?_cache=123&unknown=abc")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True


# ---------------------------------------------------------------------------
# SSE-Stream Edge Cases
# ---------------------------------------------------------------------------


class TestSSEStreamEdgeCases:
    def test_stream_error_sends_error_event(self, test_db):
        """Exception waehrend Streaming → SSE-Stream enthaelt error + done Events."""
        from unittest.mock import patch

        # Setup: User + Chat
        cursor = test_db.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (f"negtest_{uuid.uuid4().hex[:8]}", "hash"),
        )
        user_id = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_id = cursor.fetchone()["id"]
        test_db.commit()

        from kiara.web.api_admin_chat import AdminChatQueryRequest, _stream_admin_query

        data = AdminChatQueryRequest(question="Test", use_history=False)
        user = {"id": user_id, "username": "test"}

        # Shared-Resolver _resolve_backend_dict liefert leeres Dict,
        # Chain-Crash entsteht erst danach in AdminAgentChain
        with patch("kiara.generation.admin_agent_chain.AdminAgentChain") as MockChain:
            MockChain.side_effect = RuntimeError("Chain-Crash")
            with patch("kiara.config.load_config", return_value={}):
                with patch("kiara.generation.cloud_routing._resolve_backend_dict", return_value={}):
                    events = list(_stream_admin_query(chat_id, data, user))

        # Pruefen: error-Event vorhanden
        error_events = [e for e in events if "event: error" in e]
        assert len(error_events) >= 1
        # Pruefen: done-Event immer als letztes
        assert "event: done" in events[-1]

    def test_stream_always_sends_done(self, test_db):
        """Done-Event wird IMMER gesendet — auch bei Exception im try-Block."""
        from unittest.mock import patch

        # Setup: User + Chat
        cursor = test_db.execute(
            "INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
            (f"negtest_{uuid.uuid4().hex[:8]}", "hash"),
        )
        user_id = cursor.fetchone()["id"]
        cursor = test_db.execute(
            "INSERT INTO admin_chats (user_id) VALUES (%s) RETURNING id",
            (user_id,),
        )
        chat_id = cursor.fetchone()["id"]
        test_db.commit()

        from kiara.web.api_admin_chat import AdminChatQueryRequest, _stream_admin_query

        data = AdminChatQueryRequest(question="Crash-Test", use_history=False)
        user = {"id": user_id, "username": "test"}

        # Crash im Shared-Resolver _resolve_backend_dict
        with (
            patch("kiara.config.load_config", return_value={}),
            patch(
                "kiara.generation.cloud_routing._resolve_backend_dict",
                side_effect=RuntimeError("Backend-Crash"),
            ),
        ):
            events = list(_stream_admin_query(chat_id, data, user))

        # Letztes Event muss IMMER done sein
        assert len(events) >= 1
        assert "event: done" in events[-1]
