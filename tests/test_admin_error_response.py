"""Tests: Admin-Router Error-Response-Format.

Stellt sicher, dass Fehler-Antworten des Admin-Routers Felder enthalten,
die das Frontend lesen kann (error, message oder detail).

Verhindert den Feldname-Mismatch (Backend 'error' vs. Frontend 'detail'),
der dazu fuehrte, dass nur 'Speichern fehlgeschlagen' angezeigt wurde
statt der eigentlichen Fehlermeldung (2026-04-05).
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from kiara.admin.function import AdminFunction
from kiara.admin.result import AdminResult
from kiara.admin.router import _create_endpoint


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


def _make_failing_function(raw_response: bool) -> AdminFunction:
    """Erzeugt eine AdminFunction die immer AdminResult.fail() zurueckgibt."""

    async def failing_fn(user):
        return AdminResult.fail("LDAP-Verbindungstest fehlgeschlagen: Connection refused")

    return AdminFunction(
        name="test.fail_stub",
        category="test",
        description="Stub der immer fehlschlaegt",
        help="",
        examples=[],
        parameters={"type": "object", "properties": {}, "required": []},
        returns="",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method="POST",
        http_path="/api/test/fail-stub",
        fn=failing_fn,
        raw_response=raw_response,
    )


def _make_failing_function_with_data() -> AdminFunction:
    """Erzeugt eine raw_response AdminFunction mit Daten im Fehlerfall."""

    async def failing_fn(user):
        return AdminResult.fail(
            "Validierung fehlgeschlagen",
            data={"field": "source", "reason": "invalid"},
        )

    return AdminFunction(
        name="test.fail_with_data_stub",
        category="test",
        description="Stub der mit Daten fehlschlaegt",
        help="",
        examples=[],
        parameters={"type": "object", "properties": {}, "required": []},
        returns="",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method="POST",
        http_path="/api/test/fail-with-data-stub",
        fn=failing_fn,
        raw_response=True,
    )


def _make_mock_request() -> MagicMock:
    """Erzeugt einen Mock-Request fuer den Admin-Router."""
    request = MagicMock()
    request.path_params = {}
    request.headers = {"content-length": "0"}
    request.json = AsyncMock(side_effect=Exception("no body"))
    return request


MOCK_USER = {"id": 1, "username": "admin"}

# Die drei Feldnamen die das Frontend prueft (app_core.js saveConfigToAPI):
FRONTEND_FIELDS = ("detail", "error", "message")


# ---------------------------------------------------------------------------
# AdminResult Vertrag
# ---------------------------------------------------------------------------


class TestAdminResultContract:
    """Prueft, dass AdminResult.fail() die noetige Fehlerinfo traegt."""

    def test_fail_sets_error_and_message(self):
        """AdminResult.fail() muss error und message identisch setzen."""
        result = AdminResult.fail("Testfehler")
        assert result.success is False
        assert result.error == "Testfehler"
        assert result.message == "Testfehler"

    def test_fail_default_http_status_400(self):
        """AdminResult.fail() muss HTTP-Status 400 setzen."""
        result = AdminResult.fail("Testfehler")
        assert result.http_status == 400

    def test_ok_has_no_error(self):
        """AdminResult.ok() darf kein error-Feld setzen."""
        result = AdminResult.ok("Alles gut")
        assert result.success is True
        assert result.error is None


# ---------------------------------------------------------------------------
# Router Error-Response-Format
# ---------------------------------------------------------------------------


class TestRouterErrorResponse:
    """Prueft, dass der Admin-Router Fehler mit frontend-lesbaren Feldern zurueckgibt."""

    def test_raw_response_error_contains_error_field(self):
        """raw_response=True: Fehler-JSON muss 'error'-Feld enthalten."""
        func = _make_failing_function(raw_response=True)
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        # Mindestens eines der Frontend-Felder muss vorhanden sein
        readable_fields = {k: body.get(k) for k in FRONTEND_FIELDS if body.get(k)}
        assert readable_fields, (
            f"Fehler-Response hat keines der Frontend-Felder {FRONTEND_FIELDS}: {body}"
        )
        assert "fehlgeschlagen" in (body.get("error") or "").lower()

    def test_wrapped_response_error_contains_both_fields(self):
        """raw_response=False: Fehler-JSON muss 'error' UND 'message' enthalten."""
        func = _make_failing_function(raw_response=False)
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        assert body.get("error"), f"Feld 'error' fehlt in Response: {body}"
        assert body.get("message"), f"Feld 'message' fehlt in Response: {body}"
        assert body["success"] is False

    def test_error_message_survives_to_response(self):
        """Die konkrete Fehlermeldung muss im Response-Body ankommen."""
        func = _make_failing_function(raw_response=True)
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        # Exakter Text aus AdminResult.fail() muss durchkommen
        all_values = " ".join(str(v) for v in body.values())
        assert "Connection refused" in all_values, (
            f"Konkrete Fehlermeldung geht im Router verloren: {body}"
        )

    def test_raw_response_error_with_data_keeps_error_contract(self):
        """raw_response=True: Fehler mit data behalten error/message-Felder."""
        func = _make_failing_function_with_data()
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        assert body["success"] is False
        assert body["error"] == "Validierung fehlgeschlagen"
        assert body["message"] == "Validierung fehlgeschlagen"
        assert body["field"] == "source"


# ---------------------------------------------------------------------------
# Frontend-Kompatibilitaet
# ---------------------------------------------------------------------------


class TestFrontendFieldChain:
    """Prueft, dass die Frontend-Fallback-Kette fuer beide Response-Typen funktioniert.

    Das Frontend (app_core.js) liest:
        errData.detail || errData.error || errData.message || 'Speichern fehlgeschlagen'
    """

    def test_raw_response_resolves_via_field_chain(self):
        """raw_response=True (kein 'detail'-Feld) muss ueber 'error' aufloesbar sein."""
        func = _make_failing_function(raw_response=True)
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        resolved = body.get("detail") or body.get("error") or body.get("message")
        assert resolved is not None, "Frontend-Fallback-Kette liefert None"
        assert resolved != "Speichern fehlgeschlagen", (
            "Frontend wuerde generische Meldung zeigen statt der echten Fehlermeldung"
        )

    def test_wrapped_response_resolves_via_field_chain(self):
        """raw_response=False muss ueber 'error' oder 'message' aufloesbar sein."""
        func = _make_failing_function(raw_response=False)
        endpoint = _create_endpoint(func)
        request = _make_mock_request()

        response = _run(endpoint(request, user=MOCK_USER))
        body = json.loads(response.body)

        resolved = body.get("detail") or body.get("error") or body.get("message")
        assert resolved is not None, "Frontend-Fallback-Kette liefert None"
        assert "fehlgeschlagen" in resolved.lower()
