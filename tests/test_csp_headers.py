"""Regressionstests fuer staged CSP-Header und Nonce-Transport."""

import re
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.testclient import TestClient

from kiara.shared.constants import ADMIN_CHAT_PORT_DEFAULT
from kiara.web.api_wiki import router as wiki_router
from kiara.web.html_utils import render_html_file
from kiara.web.middleware import add_security_headers_middleware

_BASE = Path(__file__).resolve().parent.parent
_TEMPLATES = _BASE / "kiara" / "web" / "templates"


def _extract_header_nonce(header_value: str) -> str:
    match = re.search(r"'nonce-([^']+)'", header_value)
    assert match, f"Kein CSP-Nonce in Header gefunden: {header_value}"
    return match.group(1)


def _extract_meta_nonce(html: str) -> str:
    match = re.search(r'<meta name="csp-nonce" content="([^"]*)">', html)
    assert match, "Kein csp-nonce Meta-Tag gefunden"
    return match.group(1)


def _template_app(
    template_name: str,
    *,
    replacements: dict[str, object] | None = None,
    strict: bool = True,
    report_only: bool = False,
) -> FastAPI:
    app = FastAPI()
    add_security_headers_middleware(
        app,
        csp_strict=strict,
        csp_report_only=report_only,
        csp_report_uri="/csp-report",
    )

    @app.get("/", response_class=HTMLResponse)
    async def page(request: Request):
        return render_html_file(_TEMPLATES / template_name, request, replacements=replacements)

    @app.get("/json")
    async def json_page():
        return {"ok": True}

    return app


def test_strict_csp_header_matches_login_nonce_and_no_unsafe_inline():
    app = _template_app("login.html", report_only=True)

    with TestClient(app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    csp = resp.headers["Content-Security-Policy"]
    report_only = resp.headers["Content-Security-Policy-Report-Only"]
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-inline'" not in report_only
    assert "report-uri /csp-report" in report_only

    header_nonce = _extract_header_nonce(csp)
    assert header_nonce == _extract_meta_nonce(resp.text)
    assert f'nonce="{header_nonce}"' in resp.text


def test_csp_nonce_changes_between_requests():
    app = _template_app("login.html")

    with TestClient(app) as client:
        first = client.get("/")
        second = client.get("/")

    first_nonce = _extract_header_nonce(first.headers["Content-Security-Policy"])
    second_nonce = _extract_header_nonce(second.headers["Content-Security-Policy"])
    assert first_nonce != second_nonce


def test_non_html_response_has_no_csp_header():
    app = _template_app("login.html")

    with TestClient(app) as client:
        resp = client.get("/json")

    assert resp.status_code == 200
    assert "Content-Security-Policy" not in resp.headers
    assert "Content-Security-Policy-Report-Only" not in resp.headers


def test_admin_shell_injects_chat_port_before_tab_kiara_script():
    app = _template_app("admin.html", replacements={"admin_chat_port": ADMIN_CHAT_PORT_DEFAULT})

    with TestClient(app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    csp = resp.headers["Content-Security-Policy"]
    nonce = _extract_header_nonce(csp)
    assert _extract_meta_nonce(resp.text) == nonce
    assert f"window.ADMIN_CHAT_PORT={ADMIN_CHAT_PORT_DEFAULT};" in resp.text
    assert resp.text.index(f"window.ADMIN_CHAT_PORT={ADMIN_CHAT_PORT_DEFAULT};") < resp.text.index(
        "/static/js/tab_kiara.js?v=1"
    )
    assert (
        f'<script nonce="{nonce}">window.ADMIN_CHAT_PORT={ADMIN_CHAT_PORT_DEFAULT};</script>'
        in resp.text
    )


def test_welle4_report_only_rollout_enforce_stays_legacy():
    """V30 Welle 4 (#1049): Mit csp_strict=False + csp_report_only=True
    liefert die Middleware ZWEI CSP-Header:
    - Content-Security-Policy (Enforce) mit 'unsafe-inline' fuer Legacy-
      Kompatibilitaet und KEINEM report-uri (Enforce ist nicht strict).
    - Content-Security-Policy-Report-Only (strict) OHNE 'unsafe-inline' und
      MIT report-uri — das ist der Browser-Observation-Pfad vor Welle 5.
    """
    app = _template_app("login.html", strict=False, report_only=True)

    with TestClient(app) as client:
        resp = client.get("/")

    assert resp.status_code == 200

    # Enforce-Header bleibt Legacy-kompatibel
    enforce = resp.headers["Content-Security-Policy"]
    assert "'unsafe-inline'" in enforce, (
        "Enforce-Header muss waehrend Welle 4 Legacy bleiben (unsafe-inline erlaubt)"
    )
    assert "report-uri" not in enforce, (
        "Enforce-Header darf kein report-uri haben, nur der Report-Only-Header"
    )

    # Report-Only-Header ist strict und sendet Reports
    report_only = resp.headers["Content-Security-Policy-Report-Only"]
    assert "'unsafe-inline'" not in report_only, (
        "Report-Only-Header muss strict sein (keine unsafe-inline)"
    )
    assert "report-uri /csp-report" in report_only, "Report-Only-Header muss report-uri setzen"

    # Legacy-Enforce hat keinen Nonce (nutzt 'unsafe-inline'); nur Report-Only
    # traegt das request-spezifische Nonce, das mit dem Meta-Tag uebereinstimmt.
    report_nonce = _extract_header_nonce(report_only)
    assert report_nonce == _extract_meta_nonce(resp.text)


def test_welle5_default_flip_strict_enforce_plus_report_only():
    """V30 Welle 5 (#1049): Default-Flip — csp_strict=True UND
    csp_report_only=True. Beide Header sind strict, beide tragen
    report-uri. Der Enforce-Header blockiert jetzt, der Report-Only-
    Header liefert zusaetzlich Observability."""
    app = _template_app("login.html", strict=True, report_only=True)

    with TestClient(app) as client:
        resp = client.get("/")

    assert resp.status_code == 200

    enforce = resp.headers["Content-Security-Policy"]
    # Enforce ist strict — kein unsafe-inline
    assert "'unsafe-inline'" not in enforce, (
        "Welle 5 Enforce-Header muss strict sein (keine unsafe-inline)"
    )
    # Enforce hat report-uri (fuer Browser-Reports von geblockten Violations)
    assert "report-uri /csp-report" in enforce, "Welle 5 Enforce-Header muss report-uri setzen"

    report_only = resp.headers["Content-Security-Policy-Report-Only"]
    assert "'unsafe-inline'" not in report_only
    assert "report-uri /csp-report" in report_only

    # Beide Header teilen sich den gleichen Nonce + Meta-Tag
    enforce_nonce = _extract_header_nonce(enforce)
    report_nonce = _extract_header_nonce(report_only)
    assert enforce_nonce == report_nonce == _extract_meta_nonce(resp.text)


def test_welle4_report_only_off_means_no_report_only_header():
    """Ohne csp_report_only liefert die Middleware nur Enforce-Header."""
    app = _template_app("login.html", strict=False, report_only=False)

    with TestClient(app) as client:
        resp = client.get("/")

    assert resp.status_code == 200
    assert "Content-Security-Policy" in resp.headers
    assert "Content-Security-Policy-Report-Only" not in resp.headers


def test_csp_report_endpoint_accepts_browser_post_and_returns_204():
    """V30 Welle 4: POST /api/admin/csp-report (und /api/csp-report) nimmt
    Browser-CSP-Reports entgegen, antwortet mit 204. Kein Auth, kein CSRF."""
    from kiara.web.api_csp import router as csp_router

    app = FastAPI()
    app.include_router(csp_router)

    violation_report = {
        "csp-report": {
            "document-uri": "https://example.com/",
            "referrer": "",
            "violated-directive": "script-src",
            "blocked-uri": "inline",
        }
    }

    with TestClient(app) as client:
        # Standard-Pfad
        resp1 = client.post(
            "/api/admin/csp-report",
            json=violation_report,
            headers={"Content-Type": "application/csp-report"},
        )
        # Alternative Pfad (User-App)
        resp2 = client.post("/api/csp-report", json=violation_report)
        # Leerer Body (manche Browser senden den)
        resp3 = client.post("/api/admin/csp-report")

    assert resp1.status_code == 204
    assert resp2.status_code == 204
    assert resp3.status_code == 204


def test_wiki_route_gets_request_nonce_in_template_context():
    app = FastAPI()
    add_security_headers_middleware(app, csp_strict=True)
    app.include_router(wiki_router)

    with (
        patch("kiara.web.api_wiki._require_admin", return_value={"id": 1, "is_admin": True}),
        patch("kiara.web.api_wiki.admin_registry") as mock_registry,
    ):
        mock_registry.get_self_check_summary.return_value = {
            "total_functions": 0,
            "total_categories": 0,
            "errors": [],
            "warnings": [],
            "infos": [],
        }
        with TestClient(app) as client:
            resp = client.get("/wiki/")

    assert resp.status_code == 200
    nonce = _extract_header_nonce(resp.headers["Content-Security-Policy"])
    assert _extract_meta_nonce(resp.text) == nonce
    assert f'nonce="{nonce}"' in resp.text
