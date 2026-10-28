"""Tests fuer IDOR-Schutz im Datasource-Browse-Endpoint (F-31).

Prueft die Berechtigungspruefung in api_datasource_browse.browse_datasource():
- User ohne Berechtigung bekommt 403
- Oeffentliche Datasource (keine Gruppen-/User-Zuordnung) ist erlaubt
- User mit expliziter Berechtigung bekommt Zugriff
"""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection

# ---------------------------------------------------------------------------
# Test-App: Router wird NICHT direkt importiert (Circular-Import vermeiden).
# Stattdessen wird die Endpoint-Funktion ueber den Sub-Router in
# api_datasource eingebunden.
# ---------------------------------------------------------------------------


def _create_test_app():
    """Erstellt eine minimale FastAPI-App mit dem Browse-Endpoint."""
    # Importiere den gesamten Datasource-Router (enthaelt Browse-Sub-Router)
    from kiara.web.api_datasource import router as ds_router

    app = FastAPI()
    app.include_router(ds_router)
    return app


def _create_test_db(ds_dir):
    """Erstellt PostgreSQL-Tabellen mit Datasource- und Berechtigungstabellen."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
    conn.executescript("""
        CREATE TABLE datasources (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL DEFAULT 'local',
            path TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            is_public BOOLEAN DEFAULT true,
            connector_config TEXT DEFAULT '{}',
            restriction_level TEXT DEFAULT 'anonymize',
            ds_contextual_retrieval INTEGER DEFAULT 0,
            ds_enrichment_prompt_id INTEGER DEFAULT NULL,
            ds_enrichment_backend_id INTEGER DEFAULT NULL,
        ds_colbert_enabled BOOLEAN DEFAULT false
        );
        CREATE TABLE datasource_groups (
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            group_id INTEGER NOT NULL,
            UNIQUE(datasource_id, group_id)
        );
        CREATE TABLE datasource_users (
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            UNIQUE(datasource_id, user_id)
        );
    """)
    # Drei Datasources: restricted (DS 1), public (DS 2), user-assigned (DS 3)
    conn.execute(
        "INSERT INTO datasources (name, type, path, connector_config, is_public) VALUES (%s, %s, %s, %s, %s)",
        ("Restricted DS", "local", str(ds_dir), '{"path": "' + str(ds_dir) + '"}', False),
    )
    conn.execute(
        "INSERT INTO datasources (name, type, path, connector_config, is_public) VALUES (%s, %s, %s, %s, %s)",
        ("Public DS", "local", str(ds_dir), '{"path": "' + str(ds_dir) + '"}', True),
    )
    conn.execute(
        "INSERT INTO datasources (name, type, path, connector_config, is_public) VALUES (%s, %s, %s, %s, %s)",
        ("User DS", "local", str(ds_dir), '{"path": "' + str(ds_dir) + '"}', False),
    )
    # DS 1: Einer Gruppe zugeordnet (eingeschraenkt)
    conn.execute(
        "INSERT INTO datasource_groups (datasource_id, group_id) VALUES (%s, %s)",
        (1, 99),
    )
    # DS 2: Keine Zuordnung (oeffentlich)
    # DS 3: Wird dem User ueber datasource_ids zugewiesen
    conn.commit()
    return conn


def _mock_user(datasource_ids=None):
    """Erzeugt ein User-Dict wie von get_current_user zurueckgegeben."""
    return {
        "id": 1,
        "username": "testuser",
        "vorname": "Test",
        "nachname": "User",
        "email": "test@example.com",
        "is_admin": False,
        "must_change_password": False,
        "group_ids": [],
        "datasource_ids": datasource_ids or [],
    }


@pytest.fixture
def browse_env(tmp_path):
    """Baut eine komplette Test-Umgebung fuer Browse-Tests auf."""
    ds_dir = tmp_path / "datasource_files"
    ds_dir.mkdir()
    (ds_dir / "testfile.txt").write_text("hello")
    sub = ds_dir / "subfolder"
    sub.mkdir()

    conn = _create_test_db(ds_dir)

    @contextmanager
    def mock_db_query():
        test_conn = get_connection()
        try:
            yield test_conn
        finally:
            test_conn.close()

    yield {
        "conn": conn,
        "ds_dir": str(ds_dir),
        "mock_db_query": mock_db_query,
    }
    conn.rollback()
    conn.close()


def _build_client(browse_env, user):
    """Erstellt einen TestClient mit allen nötigen Patches."""
    app = _create_test_app()

    from kiara.web.auth import get_admin_user, get_current_user

    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_admin_user] = lambda: user

    return app, TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBrowseAuth:
    def test_browse_unauthorized_datasource_returns_403(self, browse_env):
        """User ohne Berechtigung auf restricted DS bekommt 403."""
        user = _mock_user(datasource_ids=[])  # Keine DS zugewiesen

        app, client = _build_client(browse_env, user)

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path", return_value=browse_env["ds_dir"]
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_env["ds_dir"]}},
            ),
        ):
            # DS 1 ist restricted (hat datasource_groups-Eintrag) und User hat keinen Zugriff
            resp = client.get("/api/datasources/1/browse")
            assert resp.status_code == 403

    def test_browse_public_datasource_allowed(self, browse_env):
        """Oeffentliche DS (keine Gruppen-/User-Zuordnung) ist fuer jeden User erlaubt."""
        user = _mock_user(datasource_ids=[])  # Keine DS zugewiesen

        app, client = _build_client(browse_env, user)

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path", return_value=browse_env["ds_dir"]
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_env["ds_dir"]}},
            ),
        ):
            # DS 2 ist public (keine Zuordnung)
            resp = client.get("/api/datasources/2/browse")
            assert resp.status_code == 200
            data = resp.json()
            assert "entries" in data

    def test_browse_authorized_datasource_allowed(self, browse_env):
        """User mit DS in datasource_ids bekommt Zugriff."""
        user = _mock_user(datasource_ids=[1])  # DS 1 explizit zugewiesen

        app, client = _build_client(browse_env, user)

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path", return_value=browse_env["ds_dir"]
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_env["ds_dir"]}},
            ),
        ):
            # DS 1 ist restricted ABER User hat Zugriff via datasource_ids
            resp = client.get("/api/datasources/1/browse")
            assert resp.status_code == 200
            data = resp.json()
            assert "entries" in data

    def test_browse_inactive_datasource_returns_409(self, browse_env):
        """Inaktive Datasources duerfen nicht browsebar sein."""
        conn = get_connection()
        conn.execute("UPDATE datasources SET is_active = false WHERE id = %s", (2,))
        conn.commit()
        conn.close()

        user = _mock_user(datasource_ids=[])
        app, client = _build_client(browse_env, user)

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path", return_value=browse_env["ds_dir"]
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_env["ds_dir"]}},
            ),
        ):
            resp = client.get("/api/datasources/2/browse")
            assert resp.status_code == 409
