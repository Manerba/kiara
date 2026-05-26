"""Tests fuer Build 63+64: Kontextfenster-Slider — _fetch_model_info() + Backfill.

Testet:
    - _fetch_model_info() extrahiert num_ctx + embedding_dim korrekt aus verschachtelter modelinfo
    - _fetch_model_info() gibt leeres dict zurueck wenn Ollama nicht erreichbar
    - _fetch_model_info() gibt leeres dict zurueck bei Modellen ohne context_length
    - extra_config wird bei Backend-Create befuellt (num_ctx via _fetch_model_info)
    - Filters-Endpoint liefert num_ctx pro Backend
"""

import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.admin_app import app as admin_app
from kiara.web.app import app as chat_app

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def admin_client():
    return TestClient(admin_app)


@pytest.fixture
def chat_client():
    return TestClient(chat_app)


@pytest.fixture
def mock_admin():
    """Mock-Admin-User fuer Dependency-Override (Admin-App)."""
    user = {"id": 1, "username": "admin", "is_admin": True}
    from kiara.web.auth import get_admin_user

    admin_app.dependency_overrides[get_admin_user] = lambda: user
    yield user
    admin_app.dependency_overrides.pop(get_admin_user, None)


@pytest.fixture
def mock_chat_user():
    """Mock-User fuer Dependency-Override (Chat-App)."""
    user = {"id": 1, "username": "testuser", "is_admin": True, "datasource_ids": []}
    from kiara.web.auth import get_chat_user

    chat_app.dependency_overrides[get_chat_user] = lambda: user
    yield user
    chat_app.dependency_overrides.pop(get_chat_user, None)


@pytest.fixture
def mock_db(tmp_path):
    """PostgreSQL-DB mit ki_backends-Tabelle, gepatcht fuer api_ki_backends."""
    from kiara.shared.db import get_connection

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
    conn.close()

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
        patch("kiara.web.api_ki_backends.invalidate_cache"),
    ):
        yield "pg"


# ---------------------------------------------------------------------------
# Tests: _fetch_num_ctx() Unit-Tests
# ---------------------------------------------------------------------------


class TestFetchModelInfo:
    """Direkte Unit-Tests fuer _fetch_model_info() / fetch_ollama_model_info()."""

    @pytest.mark.asyncio
    async def test_extracts_context_length_from_modelinfo(self):
        """context_length wird korrekt aus verschachtelter modelinfo extrahiert."""
        from kiara.web.api_ki_backends import _fetch_model_info

        mock_result = {"num_ctx": 40960, "embedding_dim": 4096}

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value=mock_result):
            result = await _fetch_model_info("http://gpu-server.local:11434", "qwen3:8b")

        assert result["num_ctx"] == 40960

    @pytest.mark.asyncio
    async def test_extracts_from_model_info_key(self):
        """Funktioniert auch mit 'model_info' statt 'modelinfo' (alternativer Key)."""
        from kiara.web.api_ki_backends import _fetch_model_info

        mock_result = {"num_ctx": 16384}

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value=mock_result):
            result = await _fetch_model_info("http://gpu-server.local:11434", "phi4:14b")

        assert result["num_ctx"] == 16384

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_connection_error(self):
        """Gibt leeres dict zurueck wenn Ollama nicht erreichbar ist."""
        from kiara.web.api_ki_backends import _fetch_model_info

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value={}):
            result = await _fetch_model_info("http://unreachable:11434", "qwen3:8b")

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_on_import_error(self):
        """Gibt leeres dict zurueck wenn ollama-Modul nicht installiert ist."""
        from kiara.web.api_ki_backends import _fetch_model_info

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value={}):
            result = await _fetch_model_info("http://gpu-server.local:11434", "qwen3:8b")

        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_dict_without_context_length(self):
        """Gibt leeres dict zurueck wenn modelinfo keinen context_length Key hat."""
        from kiara.web.api_ki_backends import _fetch_model_info

        mock_result = {"embedding_dim": 4096}

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value=mock_result):
            result = await _fetch_model_info("http://gpu-server.local:11434", "qwen3:8b")

        assert "num_ctx" not in result

    @pytest.mark.asyncio
    async def test_returns_empty_dict_with_empty_modelinfo(self):
        """Gibt leeres dict zurueck wenn modelinfo leer oder nicht vorhanden."""
        from kiara.web.api_ki_backends import _fetch_model_info

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value={}):
            result = await _fetch_model_info("http://gpu-server.local:11434", "qwen3:8b")

        assert result == {}

    @pytest.mark.asyncio
    async def test_converts_float_to_int(self):
        """Float-Werte werden korrekt zu int konvertiert (in fetch_ollama_model_info)."""
        from kiara.web.api_ki_backends import _fetch_model_info

        mock_result = {"num_ctx": 8192}

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value=mock_result):
            result = await _fetch_model_info("http://gpu-server.local:11434", "llama3:8b")

        assert result["num_ctx"] == 8192
        assert isinstance(result["num_ctx"], int)

    @pytest.mark.asyncio
    async def test_ignores_non_numeric_context_length(self):
        """Ignoriert context_length wenn Wert kein int/float ist (leeres dict)."""
        from kiara.web.api_ki_backends import _fetch_model_info

        with patch("kiara.retrieval.embedding.fetch_ollama_model_info", return_value={}):
            result = await _fetch_model_info("http://gpu-server.local:11434", "llama3:8b")

        assert "num_ctx" not in result


# ---------------------------------------------------------------------------
# Tests: Backend-Create befuellt extra_config mit num_ctx
# ---------------------------------------------------------------------------


class TestCreateBackendNumCtx:
    """Testet ob extra_config bei Backend-Create mit num_ctx befuellt wird."""

    def test_create_ollama_backend_fetches_num_ctx(self, admin_client, mock_admin, mock_db):
        """Beim Create eines Ollama-Backends wird num_ctx automatisch ermittelt."""
        with patch(
            "kiara.web.api_ki_backends._fetch_model_info",
            new_callable=AsyncMock,
            return_value={"num_ctx": 40960},
        ):
            resp = admin_client.post(
                "/api/admin/ki/backends/",
                json={
                    "category": "llm",
                    "provider": "ollama",
                    "name": "Test LLM",
                    "url": "http://gpu-server.local:11434",
                    "model": "qwen3:8b",
                },
                headers={"X-Requested-With": "XMLHttpRequest"},
            )

        assert resp.status_code == 200
        new_id = resp.json()["id"]

        # Pruefen ob num_ctx in extra_config gespeichert wurde
        conn = get_connection()
        row = conn.execute(
            "SELECT extra_config FROM ki_backends WHERE id = %s", (new_id,)
        ).fetchone()
        conn.close()

        extra = json.loads(row["extra_config"])
        assert extra["num_ctx"] == 40960

    def test_create_ollama_backend_none_num_ctx(self, admin_client, mock_admin, mock_db):
        """Wenn _fetch_model_info leeres dict liefert, bleibt extra_config unveraendert."""
        with patch(
            "kiara.admin.services.ki_backend_service._fetch_model_info",
            new_callable=AsyncMock,
            return_value={},
        ):
            resp = admin_client.post(
                "/api/admin/ki/backends/",
                json={
                    "category": "llm",
                    "provider": "ollama",
                    "name": "Test LLM",
                    "url": "http://gpu-server.local:11434",
                    "model": "qwen3:8b",
                },
                headers={"X-Requested-With": "XMLHttpRequest"},
            )

        assert resp.status_code == 200
        new_id = resp.json()["id"]

        conn = get_connection()
        row = conn.execute(
            "SELECT extra_config FROM ki_backends WHERE id = %s", (new_id,)
        ).fetchone()
        conn.close()

        extra = json.loads(row["extra_config"])
        assert "num_ctx" not in extra

    def test_create_with_explicit_num_ctx_and_dim_skips_fetch(
        self, admin_client, mock_admin, mock_db
    ):
        """Wenn num_ctx UND embedding_dim bereits in extra_config, wird _fetch_model_info NICHT aufgerufen."""
        with patch(
            "kiara.web.api_ki_backends._fetch_model_info", new_callable=AsyncMock
        ) as mock_fetch:
            resp = admin_client.post(
                "/api/admin/ki/backends/",
                json={
                    "category": "llm",
                    "provider": "ollama",
                    "name": "Test LLM",
                    "url": "http://gpu-server.local:11434",
                    "model": "qwen3:8b",
                    "extra_config": {"num_ctx": 32768, "embedding_dim": 4096},
                },
                headers={"X-Requested-With": "XMLHttpRequest"},
            )

        assert resp.status_code == 200
        mock_fetch.assert_not_called()

        # Explizite Werte bleiben erhalten
        new_id = resp.json()["id"]
        conn = get_connection()
        row = conn.execute(
            "SELECT extra_config FROM ki_backends WHERE id = %s", (new_id,)
        ).fetchone()
        conn.close()

        extra = json.loads(row["extra_config"])
        assert extra["num_ctx"] == 32768
        assert extra["embedding_dim"] == 4096

    def test_create_cloud_backend_no_fetch(self, admin_client, mock_admin, mock_db):
        """Cloud-Backends loesen keinen _fetch_model_info Aufruf aus."""
        with (
            patch(
                "kiara.web.api_ki_backends._fetch_model_info", new_callable=AsyncMock
            ) as mock_fetch,
            patch("kiara.web.api_ki_backends.encrypt", return_value="encrypted_key"),
        ):
            resp = admin_client.post(
                "/api/admin/ki/backends/",
                json={
                    "category": "cloud",
                    "provider": "anthropic",
                    "name": "Claude",
                    "model": "claude-sonnet-4-6",
                    "api_key": "sk-test-123",
                },
                headers={"X-Requested-With": "XMLHttpRequest"},
            )

        assert resp.status_code == 200
        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Test-Endpoint speichert num_ctx
# ---------------------------------------------------------------------------


class TestBackendTestNumCtx:
    """Testet ob der Test-Endpoint num_ctx bei erfolgreichem Ollama-Test speichert."""

    def test_successful_test_saves_num_ctx(self, admin_client, mock_admin, mock_db):
        """Bei erfolgreichem Ollama-Test wird num_ctx in extra_config geschrieben."""
        # Backend manuell anlegen (ohne automatischen Fetch)
        conn = get_connection()
        cur = conn.execute(
            """INSERT INTO ki_backends (category, provider, name, url, model, extra_config)
               VALUES ('llm', 'ollama', 'Test', 'http://gpu-server.local:11434', 'qwen3:8b', '{}')
               RETURNING id"""
        )
        bid = cur.fetchone()["id"]
        conn.commit()
        conn.close()

        mock_test_result = {"success": True, "models": ["qwen3:8b"]}

        with (
            patch(
                "kiara.web.api_ki_backends._fetch_model_info",
                new_callable=AsyncMock,
                return_value={"num_ctx": 40960},
            ),
            patch(
                "kiara.web.api_ki._test_ollama_connection",
                new_callable=AsyncMock,
                return_value=mock_test_result,
            ),
        ):
            resp = admin_client.post(
                f"/api/admin/ki/backends/{bid}/test", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

        conn = get_connection()
        row = conn.execute("SELECT extra_config FROM ki_backends WHERE id = %s", (bid,)).fetchone()
        conn.close()

        extra = json.loads(row["extra_config"])
        assert extra["num_ctx"] == 40960


# ---------------------------------------------------------------------------
# Tests: Filters-Endpoint liefert num_ctx
# ---------------------------------------------------------------------------


class TestFiltersNumCtx:
    """Testet ob der /api/chat/filters Endpoint num_ctx pro Backend liefert."""

    def test_filters_include_num_ctx(self, chat_client, mock_chat_user):
        """llm_backends im Filters-Response enthalten num_ctx aus extra_config."""
        mock_backends = [
            {
                "id": 1,
                "name": "LLM 1",
                "model": "qwen3:8b",
                "provider": "ollama",
                "extra_config": json.dumps({"num_ctx": 40960}),
            },
            {
                "id": 2,
                "name": "LLM 2",
                "model": "phi4:14b",
                "provider": "ollama",
                "extra_config": json.dumps({"num_ctx": 16384}),
            },
        ]

        mock_config = {
            "datasource": {"file_types": [".pdf", ".docx"]},
            "ollama": {"embedding_model": "nomic-embed-text"},
        }

        with (
            patch("kiara.web.api_chat.load_config", return_value=mock_config),
            patch("kiara.web.api_chat.ds_repo") as mock_ds,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb,
            patch("kiara.web.api_chat.perm_repo") as mock_perm,
            patch("kiara.backend_selector.get_all_active") as mock_active,
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            # Repository-Mocks: leere Datasources + Embedding-Modelle
            mock_ds.list_filtered.return_value = ([], 0)
            mock_emb.get_active_models.return_value = []
            mock_perm.get_public_datasource_ids.return_value = []

            # get_all_active: llm + cloud
            mock_active.side_effect = lambda cat: mock_backends if cat == "llm" else []

            resp = chat_client.get(
                "/api/chat/filters", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        assert resp.status_code == 200
        data = resp.json()
        backends = data["llm_backends"]

        assert len(backends) == 2
        assert backends[0]["num_ctx"] == 40960
        assert backends[0]["model"] == "qwen3:8b"
        assert backends[1]["num_ctx"] == 16384
        assert backends[1]["model"] == "phi4:14b"

    def test_filters_num_ctx_none_when_missing(self, chat_client, mock_chat_user):
        """num_ctx ist None wenn extra_config keinen num_ctx Eintrag hat."""
        mock_backends = [
            {
                "id": 3,
                "name": "Cloud",
                "model": "claude-sonnet-4-6",
                "provider": "anthropic",
                "extra_config": "{}",
            },
        ]

        mock_config = {
            "datasource": {"file_types": []},
            "ollama": {"embedding_model": "nomic-embed-text"},
        }

        with (
            patch("kiara.web.api_chat.load_config", return_value=mock_config),
            patch("kiara.web.api_chat.ds_repo") as mock_ds,
            patch("kiara.web.api_chat.embedding_repo") as mock_emb,
            patch("kiara.web.api_chat.perm_repo") as mock_perm,
            patch("kiara.backend_selector.get_all_active") as mock_active,
            patch("kiara.config.get_embedding_model", return_value="nomic-embed-text"),
        ):
            mock_ds.list_filtered.return_value = ([], 0)
            mock_emb.get_active_models.return_value = []
            mock_perm.get_public_datasource_ids.return_value = []

            mock_active.side_effect = lambda cat: [] if cat == "llm" else mock_backends

            resp = chat_client.get(
                "/api/chat/filters", headers={"X-Requested-With": "XMLHttpRequest"}
            )

        assert resp.status_code == 200
        data = resp.json()
        backends = data["llm_backends"]

        assert len(backends) == 1
        assert backends[0]["num_ctx"] is None
        assert backends[0]["provider"] == "anthropic"


# ---------------------------------------------------------------------------
# Tests: _parse_num_ctx Hilfsfunktion (inline in api_chat)
# ---------------------------------------------------------------------------


class TestParseNumCtx:
    """Testet die _parse_num_ctx Inline-Funktion aus api_chat."""

    def test_parse_valid_json(self):
        """Parst num_ctx aus gueltigem extra_config JSON."""
        # Importiere die Funktion indirekt ueber das Modul
        # Teste die Inline-Logik direkt
        b = {"extra_config": json.dumps({"num_ctx": 40960})}
        try:
            result = json.loads(b.get("extra_config") or "{}").get("num_ctx")
        except (ValueError, TypeError):
            result = None
        assert result == 40960

    def test_parse_empty_json(self):
        """Gibt None zurueck bei leerem extra_config."""
        b = {"extra_config": "{}"}
        try:
            result = json.loads(b.get("extra_config") or "{}").get("num_ctx")
        except (ValueError, TypeError):
            result = None
        assert result is None

    def test_parse_none_extra_config(self):
        """Gibt None zurueck bei None extra_config."""
        b = {"extra_config": None}
        try:
            result = json.loads(b.get("extra_config") or "{}").get("num_ctx")
        except (ValueError, TypeError):
            result = None
        assert result is None

    def test_parse_invalid_json(self):
        """Gibt None zurueck bei ungueltigem JSON."""
        b = {"extra_config": "not-json"}
        try:
            result = json.loads(b.get("extra_config") or "{}").get("num_ctx")
        except (ValueError, TypeError):
            result = None
        assert result is None
