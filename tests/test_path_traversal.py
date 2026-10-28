"""Tests fuer Path-Traversal-Schutz (F-32/F-52).

Prueft die realpath-Checks in:
- api_datasource_browse.browse_datasource() (Zeilen 95-97)
- api_onlyoffice.open_editor() (Zeilen 98-100)
"""

import hashlib
import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


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


def _create_browse_db(tmp_path, ds_dir):
    """Erstellt eine PostgreSQL-DB mit einer oeffentlichen Datasource.

    Returns:
        (conn, ds_id) — Verbindung und die ID der eingefuegten Datasource.
    """
    conn = get_connection()
    # DROP CASCADE + Neuanlage um Test-Schema zu garantieren
    conn.executescript("""
        DROP TABLE IF EXISTS datasource_users CASCADE;
        DROP TABLE IF EXISTS datasource_groups CASCADE;
        DROP TABLE IF EXISTS datasources CASCADE;

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
            group_id INTEGER NOT NULL
        );
        CREATE TABLE datasource_users (
            id SERIAL PRIMARY KEY,
            datasource_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL
        );
    """)
    cur = conn.execute(
        "INSERT INTO datasources (name, type, path, connector_config) VALUES (%s, %s, %s, %s) RETURNING id",
        ("Test DS", "local", str(ds_dir), '{"path": "' + str(ds_dir) + '"}'),
    )
    ds_id = cur.fetchone()["id"]
    conn.commit()
    return conn, ds_id


def _create_browse_app():
    """Erstellt eine FastAPI-App mit dem Browse-Endpoint (via Datasource-Router)."""
    from kiara.web.api_datasource import router as ds_router

    app = FastAPI()
    app.include_router(ds_router)
    return app


# ---------------------------------------------------------------------------
# Browse Path-Traversal Tests
# ---------------------------------------------------------------------------


class TestBrowsePathTraversal:
    @pytest.fixture
    def browse_setup(self, tmp_path):
        """Baut Browse-Test-Umgebung auf."""
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        sub = ds_dir / "subfolder"
        sub.mkdir()
        (sub / "file.txt").write_text("test content")

        conn, ds_id = _create_browse_db(tmp_path, ds_dir)

        from contextlib import contextmanager

        @contextmanager
        def mock_db_query():
            test_conn = get_connection()
            try:
                yield test_conn
            finally:
                test_conn.close()

        yield {
            "ds_dir": str(ds_dir),
            "ds_id": ds_id,
            "mock_db_query": mock_db_query,
            "conn": conn,
        }
        conn.close()

        # Produktions-Schema wiederherstellen (DROP CASCADE hat andere Tabellen zerstoert)
        restore = get_connection()
        try:
            restore.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
            restore.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
            restore.execute("DROP TABLE IF EXISTS datasources CASCADE")
            from kiara.web.schema import create_tables

            create_tables(restore)
            restore.commit()
        except Exception:
            restore.rollback()
        restore.close()

    def test_browse_path_traversal_blocked(self, browse_setup):
        """path='../../etc/passwd' muss 403 zurueckgeben."""
        ds_id = browse_setup["ds_id"]
        user = _mock_user(datasource_ids=[ds_id])
        app = _create_browse_app()

        from kiara.web.auth import get_admin_user, get_current_user

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_admin_user] = lambda: user

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path",
                return_value=browse_setup["ds_dir"],
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_setup["ds_dir"]}},
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(
                f"/api/datasources/{ds_id}/browse", params={"path": "../../etc/passwd"}
            )
            assert resp.status_code == 403

    def test_browse_valid_subpath_allowed(self, browse_setup):
        """Gueltiger Unterpfad 'subfolder' muss erlaubt sein."""
        ds_id = browse_setup["ds_id"]
        user = _mock_user(datasource_ids=[ds_id])
        app = _create_browse_app()

        from kiara.web.auth import get_admin_user, get_current_user

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_admin_user] = lambda: user

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path",
                return_value=browse_setup["ds_dir"],
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_setup["ds_dir"]}},
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/api/datasources/{ds_id}/browse", params={"path": "subfolder"})
            assert resp.status_code == 200
            data = resp.json()
            assert "entries" in data
            assert data["path"] == "subfolder"
            assert data["parent"] == ""

    def test_browse_missing_directory_returns_404(self, browse_setup):
        """Nicht vorhandene Unterverzeichnisse liefern 404."""
        ds_id = browse_setup["ds_id"]
        user = _mock_user(datasource_ids=[ds_id])
        app = _create_browse_app()

        from kiara.web.auth import get_admin_user, get_current_user

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_admin_user] = lambda: user

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path",
                return_value=browse_setup["ds_dir"],
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_setup["ds_dir"]}},
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/api/datasources/{ds_id}/browse", params={"path": "missing"})
            assert resp.status_code == 404

    def test_browse_root_alias_is_normalized(self, browse_setup):
        """Root-Aliase liefern normalisiertes path/parent."""
        ds_id = browse_setup["ds_id"]
        user = _mock_user(datasource_ids=[ds_id])
        app = _create_browse_app()

        from kiara.web.auth import get_admin_user, get_current_user

        app.dependency_overrides[get_current_user] = lambda: user
        app.dependency_overrides[get_admin_user] = lambda: user

        with (
            patch("kiara.web.api_datasource_browse.ensure_loaded"),
            patch("kiara.web.api_datasource_browse.get_connector", return_value=None),
            patch(
                "kiara.web.api_datasource_browse._effective_path",
                return_value=browse_setup["ds_dir"],
            ),
            patch(
                "kiara.web.api_datasource_browse.load_config",
                return_value={"datasource": {"path": browse_setup["ds_dir"]}},
            ),
        ):
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get(f"/api/datasources/{ds_id}/browse", params={"path": "."})
            assert resp.status_code == 200
            data = resp.json()
            assert data["path"] == ""
            assert data["parent"] is None


# ---------------------------------------------------------------------------
# OnlyOffice Path-Traversal Tests
# ---------------------------------------------------------------------------


class TestOnlyOfficePathTraversal:
    def test_onlyoffice_path_traversal_blocked(self, tmp_path):
        """file_path='../../etc/shadow' muss 403 zurueckgeben."""
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()

        user = _mock_user()

        from kiara.web.api_onlyoffice import router as onlyoffice_router

        app = FastAPI()
        app.include_router(onlyoffice_router)

        mock_config = {
            "datasource": {"path": str(ds_dir)},
        }

        with patch("kiara.web.api_onlyoffice.load_config", return_value=mock_config):
            from kiara.web.auth import get_chat_user

            app.dependency_overrides[get_chat_user] = lambda: user

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/onlyoffice/open", params={"file_path": "../../etc/shadow"})
            assert resp.status_code == 403

    def test_onlyoffice_valid_file_path(self, tmp_path):
        """Gueltiger Dateipfad innerhalb des base_path wird akzeptiert."""
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        test_file = ds_dir / "document.docx"
        test_file.write_bytes(b"PK\x03\x04")  # Minimal-ZIP-Header

        user = _mock_user()

        from kiara.web.api_onlyoffice import router as onlyoffice_router

        app = FastAPI()
        app.include_router(onlyoffice_router)

        mock_config = {
            "datasource": {"path": str(ds_dir)},
        }

        mock_oo_row = MagicMock()
        mock_oo_row.__getitem__ = lambda self, key: {
            "document_server_url": "http://localhost:8080",
            "jwt_secret": "testsecret",
        }[key]

        from contextlib import contextmanager

        @contextmanager
        def mock_db_cm():
            mock_conn = MagicMock()
            yield mock_conn

        mock_oo_config = {
            "document_server_url": "http://localhost:8080",
            "jwt_secret": "testsecret",
        }

        with (
            patch("kiara.web.api_onlyoffice.load_config", return_value=mock_config),
            patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
            patch("kiara.web.api_onlyoffice.wopi_repo") as mock_wopi_repo,
            patch("kiara.web.api_onlyoffice._check_datasource_acl"),
        ):
            mock_config_repo.get.return_value = mock_oo_config

            from kiara.web.auth import get_chat_user

            app.dependency_overrides[get_chat_user] = lambda: user

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/onlyoffice/open", params={"file_path": "document.docx"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["file_name"] == "document.docx"
            assert (
                data["file_id"]
                == hashlib.sha256(os.path.realpath(test_file).encode()).hexdigest()[:32]
            )
            mock_wopi_repo.create_token.assert_called_once()
            assert mock_wopi_repo.create_token.call_args.args[1] == data["file_id"]

    def test_onlyoffice_equivalent_paths_share_file_id(self, tmp_path):
        """Verschiedene Schreibweisen derselben Datei muessen denselben Lock-Key nutzen."""
        ds_dir = tmp_path / "datasource"
        ds_dir.mkdir()
        sub_dir = ds_dir / "sub"
        sub_dir.mkdir()
        test_file = ds_dir / "document.docx"
        test_file.write_bytes(b"PK\x03\x04")

        user = _mock_user()

        from kiara.web.api_onlyoffice import router as onlyoffice_router

        app = FastAPI()
        app.include_router(onlyoffice_router)

        mock_config = {
            "datasource": {"path": str(ds_dir)},
        }
        mock_oo_config = {
            "document_server_url": "http://localhost:8080",
            "jwt_secret": "testsecret",
        }

        with (
            patch("kiara.web.api_onlyoffice.load_config", return_value=mock_config),
            patch("kiara.web.api_onlyoffice.config_repo") as mock_config_repo,
            patch("kiara.web.api_onlyoffice.wopi_repo"),
            patch("kiara.web.api_onlyoffice._check_datasource_acl"),
        ):
            mock_config_repo.get.return_value = mock_oo_config

            from kiara.web.auth import get_chat_user

            app.dependency_overrides[get_chat_user] = lambda: user

            client = TestClient(app, raise_server_exceptions=False)
            canonical = client.get("/api/onlyoffice/open", params={"file_path": "document.docx"})
            equivalent = client.get(
                "/api/onlyoffice/open", params={"file_path": "sub/../document.docx"}
            )

            assert canonical.status_code == 200
            assert equivalent.status_code == 200
            assert canonical.json()["file_id"] == equivalent.json()["file_id"]
