"""Tests fuer kiara.web.api_ki_backends — CRUD-API fuer Multi-Backend."""

from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.admin_app import app
from tests.admin_confirm import confirmed_request


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_admin():
    """Mock-Admin-User fuer Dependency-Override."""
    user = {"id": 1, "username": "admin", "is_admin": True}
    from kiara.web.auth import get_admin_user

    app.dependency_overrides[get_admin_user] = lambda: user
    yield user
    app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def mock_db():
    """PostgreSQL-DB mit ki_backends-Tabelle."""
    conn = get_connection()
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS ki_backends (
            id SERIAL PRIMARY KEY,
            category TEXT NOT NULL CHECK (category IN ('embedding', 'llm', 'cloud', 'vlm')),
            provider TEXT NOT NULL CHECK (provider IN ('ollama', 'anthropic', 'openai')),
            name TEXT NOT NULL,
            url TEXT DEFAULT '',
            base_url TEXT DEFAULT '',
            model TEXT NOT NULL,
            api_key TEXT DEFAULT '',
            is_active BOOLEAN DEFAULT true,
            priority INTEGER DEFAULT 10,
            trusted BOOLEAN DEFAULT false,
            extra_config TEXT DEFAULT '{}',
            last_test_at TEXT DEFAULT NULL,
            last_test_ok INTEGER DEFAULT NULL,
            last_test_error TEXT DEFAULT '',
            created_at TEXT DEFAULT (NOW()),
            updated_at TEXT DEFAULT (NOW())
        );
    """)
    conn.commit()

    @contextmanager
    def mock_db_query():
        c = get_connection()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def mock_db_transaction():
        c = get_connection()
        try:
            yield c
            c.commit()
        except Exception:
            c.rollback()
            raise
        finally:
            c.close()

    with (
        patch("kiara_db.base.db_query", mock_db_query),
        patch("kiara_db.base.db_transaction", mock_db_transaction),
        patch("kiara.backend_selector.invalidate_cache"),
    ):
        yield conn

    conn.rollback()
    conn.close()


class TestCategoriesEndpoint:
    def test_get_categories(self, client, mock_admin):
        resp = client.get("/api/admin/ki/backends/categories")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["categories"]) == 5
        ids = [c["id"] for c in data["categories"]]
        assert set(ids) == {"embedding", "llm", "cloud", "vlm", "ddu"}


class TestCRUD:
    def test_create_backend(self, client, mock_admin, mock_db):
        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Test LLM",
                "url": "http://localhost:11434",
                "model": "qwen3:8b",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        assert "id" in resp.json()

    def test_create_invalid_category(self, client, mock_admin, mock_db):
        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "invalid",
                "provider": "ollama",
                "model": "test",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400

    def test_create_invalid_provider(self, client, mock_admin, mock_db):
        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "anthropic",
                "model": "test",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400

    def test_create_no_model(self, client, mock_admin, mock_db):
        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "model": "",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 400

    def test_list_backends(self, client, mock_admin, mock_db):
        # Create one first
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Test",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        resp = client.get("/api/admin/ki/backends/")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["backends"]) == 1
        assert data["backends"][0]["name"] == "Test"

    def test_update_backend(self, client, mock_admin, mock_db):
        # Create
        create_resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Old Name",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        bid = create_resp.json()["id"]

        # Update
        resp = confirmed_request(
            client,
            "PUT",
            f"/api/admin/ki/backends/{bid}",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "New Name",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

    def test_delete_backend(self, client, mock_admin, mock_db):
        create_resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Delete Me",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        bid = create_resp.json()["id"]

        resp = confirmed_request(
            client,
            "DELETE",
            f"/api/admin/ki/backends/{bid}",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

        # Verify deleted
        list_resp = client.get("/api/admin/ki/backends/")
        assert len(list_resp.json()["backends"]) == 0

    def test_toggle_backend(self, client, mock_admin, mock_db):
        create_resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "name": "Toggle",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        bid = create_resp.json()["id"]

        # Deactivate
        resp = client.patch(
            f"/api/admin/ki/backends/{bid}/toggle",
            json={"is_active": False},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
        assert "deaktiviert" in resp.json()["message"]

    def test_delete_nonexistent(self, client, mock_admin, mock_db):
        resp = confirmed_request(
            client,
            "DELETE",
            "/api/admin/ki/backends/9999",
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 404


class TestMultiEmbeddingModels:
    def test_embedding_different_models_allowed(self, client, mock_admin, mock_db):
        """Verschiedene Embedding-Modelle koennen koexistieren (Multi-Modell)."""
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "all-minilm",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

    def test_embedding_same_model_allowed(self, client, mock_admin, mock_db):
        """Zweites Embedding-Backend mit gleichem Modell wird erlaubt (Failover)."""
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200

    def test_embedding_multiple_models_listed(self, client, mock_admin, mock_db):
        """Alle Embedding-Backends mit verschiedenen Modellen werden aufgelistet."""
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "nomic-embed-text",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "embedding",
                "provider": "ollama",
                "model": "all-minilm",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        resp = client.get("/api/admin/ki/backends/?category=embedding")
        assert resp.status_code == 200
        models = {b["model"] for b in resp.json()["backends"]}
        assert models == {"nomic-embed-text", "all-minilm"}

    def test_llm_different_models_allowed(self, client, mock_admin, mock_db):
        """LLM-Backends mit unterschiedlichen Modellen sind erlaubt."""
        client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "model": "qwen3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

        resp = client.post(
            "/api/admin/ki/backends/",
            json={
                "category": "llm",
                "provider": "ollama",
                "model": "llama3:8b",
                "url": "http://localhost:11434",
            },
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
        assert resp.status_code == 200
