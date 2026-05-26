"""Tests fuer Sprint 11b — Wiki-Markdown-Rendering und Legacy-Fallback.

7 Tests: MD-Routing, Sidebar, Legacy-Fallback, 404, Auth, guide_pages, read-Fehler.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# DB-Backend initialisieren
import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """TestClient mit gemocktem Auth (Admin-User)."""
    from fastapi import FastAPI

    from kiara.web.api_wiki import router

    app = FastAPI()
    app.include_router(router)

    # Statische Dateien mounten fuer Template-Rendering
    from fastapi.staticfiles import StaticFiles

    static_dir = Path(__file__).parent.parent / "kiara" / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    with patch("kiara.web.api_wiki._require_admin", return_value={"id": 1, "is_admin": True}):
        yield TestClient(app)


@pytest.fixture
def unauth_client():
    """TestClient ohne Auth."""
    from fastapi import FastAPI

    from kiara.web.api_wiki import router

    app = FastAPI()
    app.include_router(router)

    with patch("kiara.web.api_wiki._require_admin", return_value=None):
        yield TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# 1. /wiki/guide/admin-chat laedt Markdown
# ---------------------------------------------------------------------------


def test_admin_chat_guide_loads_markdown(client):
    from kiara.admin.wiki_manifest import CONTENT_ROOT

    md_path = CONTENT_ROOT / "guides/admin-chat.md"
    if md_path.exists():
        resp = client.get("/wiki/guide/admin-chat")
        assert resp.status_code == 200
        assert "wikiMarkdownData" in resp.text
        assert "wikiMarkdownRoot" in resp.text


# ---------------------------------------------------------------------------
# 2. Sidebar wird aus guide_pages gerendert
# ---------------------------------------------------------------------------


def test_sidebar_contains_guide_pages(client):
    resp = client.get("/wiki/guide/admin-chat")
    if resp.status_code == 200:
        assert "/wiki/guide/users" in resp.text
        assert "/wiki/guide/datasources" in resp.text
        assert "/wiki/guide/admin-chat" in resp.text


# ---------------------------------------------------------------------------
# 3. Fehlende MD mit Legacy-Template -> Legacy gerendert
# ---------------------------------------------------------------------------


def test_legacy_fallback_for_missing_md(client):
    """Guide mit Legacy-Template aber ohne MD -> Legacy-HTML."""
    with patch("kiara.web.api_wiki.CONTENT_ROOT", Path("/nonexistent")):
        resp = client.get("/wiki/guide/users")
        assert resp.status_code == 200
        # Kein wikiMarkdownData, da Legacy-HTML
        assert "wikiMarkdownData" not in resp.text


# ---------------------------------------------------------------------------
# 4. Unbekannter Guide-Slug -> 404
# ---------------------------------------------------------------------------


def test_unknown_guide_returns_404(client):
    resp = client.get("/wiki/guide/nonexistent_slug_xyz")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. Auth-Redirect bleibt unveraendert
# ---------------------------------------------------------------------------


def test_auth_redirect(unauth_client):
    resp = unauth_client.get("/wiki/guide/admin-chat")
    assert resp.status_code == 302
    assert "/login" in resp.headers["location"]


# ---------------------------------------------------------------------------
# 6. API-Wiki-Seiten erhalten guide_pages
# ---------------------------------------------------------------------------


def test_wiki_index_has_guide_pages(client):
    with patch("kiara.web.api_wiki.admin_registry") as mock_reg:
        mock_reg.get_self_check_summary.return_value = {
            "total_functions": 0,
            "total_categories": 0,
            "errors": [],
            "warnings": [],
            "infos": [],
        }
        resp = client.get("/wiki/")
        assert resp.status_code == 200
        # Sidebar sollte Guide-Links enthalten
        assert "/wiki/guide/" in resp.text


# ---------------------------------------------------------------------------
# 7. read_text-Fehler liefert 500
# ---------------------------------------------------------------------------


def test_read_error_returns_500(client):
    from kiara.admin.wiki_manifest import CONTENT_ROOT

    md_path = CONTENT_ROOT / "guides/admin-chat.md"
    if md_path.exists():
        with patch.object(Path, "read_text", side_effect=IOError("disk error")):
            resp = client.get("/wiki/guide/admin-chat")
            assert resp.status_code == 500
