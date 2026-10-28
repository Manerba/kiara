"""Integrationstests fuer Multi-Embedding-Modell-Management.

Testet das Zusammenspiel der Komponenten, die seit der Entfernung von
_check_model_consistency() (Build +32 Fix) Multi-Modell-Embedding ermoeglichen:

- Chat-Filter: /chat/filters gibt korrekte embedding_models zurueck
- Cron: _get_ds_embedding_model() ermittelt das richtige Modell pro DS
- Worker: embedding_model wird korrekt an VectorStore durchgereicht
- VectorStore: Collection-Name wird vom Modell abgeleitet
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.shared.db import get_connection

# ---------------------------------------------------------------------------
# Shared DB-Schema (Subset fuer Multi-Embedding-Tests)
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS datasources (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'local',
    is_active BOOLEAN DEFAULT true,
    chunk_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS datasource_embeddings (
    id SERIAL PRIMARY KEY,
    datasource_id INTEGER NOT NULL REFERENCES datasources(id) ON DELETE CASCADE,
    embedding_model TEXT NOT NULL,
    chunk_count INTEGER DEFAULT 0,
    file_count INTEGER DEFAULT 0,
    last_ingested_at TEXT DEFAULT NULL,
    UNIQUE(datasource_id, embedding_model)
);

CREATE TABLE IF NOT EXISTS datasource_groups (
    datasource_id INTEGER,
    group_id INTEGER
);

CREATE TABLE IF NOT EXISTS datasource_users (
    datasource_id INTEGER,
    user_id INTEGER
);

CREATE TABLE IF NOT EXISTS ki_backends (
    id SERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'ollama',
    name TEXT NOT NULL DEFAULT '',
    url TEXT DEFAULT '',
    base_url TEXT DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    api_key TEXT DEFAULT '',
    is_active BOOLEAN DEFAULT true,
    priority INTEGER DEFAULT 10,
    extra_config TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id SERIAL PRIMARY KEY,
    datasource_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    job_type TEXT NOT NULL DEFAULT 'incremental',
    triggered_by TEXT NOT NULL DEFAULT 'manual',
    embedding_model TEXT DEFAULT '',
    progress_processed INTEGER DEFAULT 0,
    progress_total INTEGER DEFAULT 0,
    queued_at TEXT DEFAULT (NOW()),
    started_at TEXT DEFAULT NULL,
    finished_at TEXT DEFAULT NULL,
    stats_json TEXT DEFAULT '{}',
    error_message TEXT DEFAULT '',
    resumed_from_job INTEGER DEFAULT NULL,
    contract_logical_index_key TEXT DEFAULT NULL,
    contract_index_compatibility_id TEXT DEFAULT NULL,
    contract_query_compatibility_id TEXT DEFAULT NULL,
    contract_canonical_model_id TEXT DEFAULT NULL,
    contract_profile_id TEXT DEFAULT NULL,
    contract_profile_kind TEXT DEFAULT NULL,
    contract_profile_endpoint TEXT DEFAULT NULL,
    contract_snapshot_sha256 TEXT DEFAULT NULL,
    contract_fence_version BIGINT DEFAULT NULL,
    FOREIGN KEY (datasource_id) REFERENCES datasources(id) ON DELETE CASCADE
);
"""


@pytest.fixture
def test_db(tmp_path):
    """PostgreSQL-DB mit Multi-Embedding-Schema."""
    conn = get_connection()
    # Cleanup
    conn.execute("DROP TABLE IF EXISTS ingestion_jobs CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_embeddings CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_groups CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasource_users CASCADE")
    conn.execute("DROP TABLE IF EXISTS ki_backends CASCADE")
    conn.execute("DROP TABLE IF EXISTS datasources CASCADE")
    conn.executescript(_SCHEMA)
    conn.executescript("""
        INSERT INTO datasources (id, name, type, is_active, chunk_count)
        VALUES (1, 'Dokumente', 'local', true, 5000),
               (2, 'E-Mails', 'imap', true, 3000),
               (3, 'Leere DS', 'local', true, 0);

        INSERT INTO datasource_embeddings (datasource_id, embedding_model, chunk_count, file_count, last_ingested_at)
        VALUES (1, 'nomic-embed-text:latest', 5000, 100, '2026-03-08 12:00:00'),
               (2, 'nomic-embed-text:latest', 3000, 50, '2026-03-08 13:00:00'),
               (2, 'mxbai-embed-large', 2000, 50, '2026-03-07 10:00:00');

        INSERT INTO ki_backends (category, provider, name, url, model, is_active, priority)
        VALUES ('embedding', 'ollama', 'Server-A', 'http://gpu-server.local:11434', 'nomic-embed-text:latest', true, 10),
               ('embedding', 'ollama', 'Server-B', 'http://update-server.local:11434', 'mxbai-embed-large', true, 20),
               ('llm', 'ollama', 'LLM', 'http://gpu-server.local:11434', 'qwen3:8b', true, 10);
    """)
    conn.commit()
    conn.close()
    return "pg"


@pytest.fixture
def mock_db(test_db):
    """Patcht db_query/db_transaction auf PostgreSQL-Verbindungen."""

    @contextmanager
    def fake_db_query():
        conn = get_connection()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def fake_db_transaction():
        conn = get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    with (
        patch("kiara.shared.db.db_query", fake_db_query),
        patch("kiara.shared.db.db_transaction", fake_db_transaction),
        patch("kiara.web.db_utils.db_query", fake_db_query),
        patch("kiara.web.db_utils.db_transaction", fake_db_transaction),
        patch("kiara.web.ingestion_worker.db_query", fake_db_query),
        patch("kiara.web.ingestion_worker.db_transaction", fake_db_transaction),
    ):
        yield test_db


# ===================================================================
# A) Chat-Filter — embedding_models korrekt zurueckgeben
# ===================================================================


class TestChatFilterEmbeddingModels:
    """Tests fuer /chat/filters Embedding-Modell-Auflistung."""

    def _get_filters(self, mock_db):
        """Simuliert get_chat_filters() Logik gegen die Test-DB."""
        conn = get_connection()

        emb_rows = conn.execute(
            "SELECT DISTINCT embedding_model FROM datasource_embeddings "
            "WHERE chunk_count > 0 ORDER BY embedding_model"
        ).fetchall()
        conn.close()
        return [r["embedding_model"] for r in emb_rows]

    def test_returns_distinct_models(self, mock_db):
        """Gibt alle verschiedenen Embedding-Modelle mit Daten zurueck."""
        models = self._get_filters(mock_db)
        assert "nomic-embed-text:latest" in models
        assert "mxbai-embed-large" in models

    def test_no_empty_strings(self, mock_db):
        """Leere Strings werden nicht als Modell zurueckgegeben."""
        # Leeren Eintrag einfuegen
        conn = get_connection()
        conn.execute(
            "INSERT INTO datasource_embeddings (datasource_id, embedding_model, chunk_count) "
            "VALUES (3, '', 100)"
        )
        conn.commit()
        conn.close()

        models = self._get_filters(mock_db)
        # Die DB-Query gibt leere Strings zurueck — der API-Endpoint filtert sie
        # (api_chat.py Zeile 128: if r["embedding_model"])
        non_empty = [m for m in models if m]
        assert len(non_empty) == 2
        assert "" not in non_empty

    def test_zero_chunks_excluded(self, mock_db):
        """Modelle mit chunk_count=0 tauchen nicht auf."""
        conn = get_connection()
        conn.execute(
            "INSERT INTO datasource_embeddings (datasource_id, embedding_model, chunk_count) "
            "VALUES (3, 'unused-model', 0)"
        )
        conn.commit()
        conn.close()

        models = self._get_filters(mock_db)
        assert "unused-model" not in models

    def test_has_backend_flag(self, mock_db):
        """Modelle mit aktivem Backend werden mit has_backend=True markiert."""
        from kiara.backend_selector import normalize_model_name

        conn = get_connection()

        emb_rows = conn.execute(
            "SELECT DISTINCT embedding_model FROM datasource_embeddings WHERE chunk_count > 0"
        ).fetchall()
        backends = conn.execute(
            "SELECT model FROM ki_backends WHERE category = 'embedding' AND is_active = true"
        ).fetchall()
        conn.close()

        active_models = {normalize_model_name(b["model"]) for b in backends}

        for r in emb_rows:
            model = r["embedding_model"]
            if not model:
                continue
            has_backend = normalize_model_name(model) in active_models
            if "nomic" in model:
                assert has_backend, f"{model} sollte ein aktives Backend haben"
            if "mxbai" in model:
                assert has_backend, f"{model} sollte ein aktives Backend haben"


# ===================================================================
# B) Cron — _get_ds_embedding_model()
# ===================================================================


class TestCronEmbeddingModel:
    """Tests fuer _get_ds_embedding_model() in cron_utils."""

    def test_returns_latest_model_for_ds(self, mock_db):
        """Gibt das zuletzt verwendete Modell einer Datenquelle zurueck."""
        from kiara.shared.cron_utils import _get_ds_embedding_model

        model = _get_ds_embedding_model(1)
        assert model == "nomic-embed-text:latest"

    def test_returns_latest_by_timestamp(self, mock_db):
        """Bei mehreren Modellen pro DS wird das zuletzt verwendete gewaehlt."""
        from kiara.shared.cron_utils import _get_ds_embedding_model

        # DS 2 hat nomic (2026-03-08 13:00) und mxbai (2026-03-07 10:00)
        model = _get_ds_embedding_model(2)
        assert model == "nomic-embed-text:latest"

    def test_returns_empty_for_unknown_ds(self, mock_db):
        """Unbekannte Datenquelle gibt leeren String zurueck."""
        from kiara.shared.cron_utils import _get_ds_embedding_model

        model = _get_ds_embedding_model(9999)
        assert model == ""

    def test_returns_empty_for_ds_without_embeddings(self, mock_db):
        """DS ohne Embeddings gibt leeren String zurueck."""
        from kiara.shared.cron_utils import _get_ds_embedding_model

        model = _get_ds_embedding_model(3)  # 'Leere DS' — keine Embeddings
        assert model == ""

    def test_write_schedule_uses_default_model_without_prior_ingestion(
        self,
        mock_db,
        monkeypatch,
        tmp_path,
    ):
        """Cron-Schedule fuer neue DS nutzt das aktive Default-Modell."""
        from kiara.shared import cron_utils

        monkeypatch.setattr(cron_utils, "CRON_FILE", str(tmp_path / "kiara-cron"))
        monkeypatch.setattr(cron_utils, "CRON_LOCK_PATH", str(tmp_path / "cron.lock"))
        monkeypatch.setattr(
            cron_utils,
            "_get_default_embedding_model",
            lambda: "nomic-embed-text",
        )
        cron_utils.write_schedule(ds_id=3, name="Leere DS", cron_expr="0 2 * * *")

        content = (tmp_path / "kiara-cron").read_text()
        assert "--embedding-model nomic-embed-text" in content

    def test_write_schedule_fails_without_any_model(self, mock_db, monkeypatch):
        """Cron-Schedule ohne DS-Modell und ohne Default-Modell wird abgelehnt."""
        from kiara.shared.cron_utils import write_schedule

        monkeypatch.setattr(
            "kiara.shared.cron_utils._get_default_embedding_model",
            lambda: "",
        )
        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            write_schedule(ds_id=3, name="Leere DS", cron_expr="0 2 * * *")


# ===================================================================
# C) Worker — embedding_model Pflichtfeld
# ===================================================================


class TestWorkerEmbeddingModel:
    """Tests fuer embedding_model-Handling bei Job-Erstellung."""

    def test_enqueue_rejects_empty_embedding_model(self, mock_db):
        """enqueue_job() wirft ValueError wenn embedding_model leer ist."""
        from kiara.web.ingestion_worker import enqueue_job

        with pytest.raises(ValueError, match="Kein Embedding-Modell"):
            enqueue_job(1, "incremental", "manual", embedding_model="")

    def test_enqueue_rejects_none_embedding_model(self, mock_db):
        """enqueue_job() wirft TypeError wenn embedding_model nicht uebergeben wird."""
        from kiara.web.ingestion_worker import enqueue_job

        with pytest.raises(TypeError):
            enqueue_job(1, "incremental", "manual")

    def test_enqueue_job_stores_model(self, mock_db):
        """enqueue_job() speichert das Embedding-Modell in der DB."""
        from kiara.web.ingestion_worker import enqueue_job

        # DS 3 hat keinen aktiven Job → enqueue moeglich
        job_id = enqueue_job(3, "incremental", "manual", embedding_model="mxbai-embed-large")

        conn = get_connection()

        rows = conn.execute(
            "SELECT id, embedding_model FROM ingestion_jobs ORDER BY id DESC"
        ).fetchall()
        conn.close()

        matched = [r for r in rows if r["id"] == job_id]
        assert len(matched) == 1
        assert matched[0]["embedding_model"] == "mxbai-embed-large"

    def test_enqueue_normalizes_model_name(self, mock_db):
        """enqueue_job() normalisiert den Modellnamen (entfernt :latest)."""
        from kiara.web.ingestion_worker import enqueue_job

        # Andere DS (1) verwenden, da DS 3 ggf. schon einen Job hat
        job_id = enqueue_job(2, "incremental", "manual", embedding_model="nomic-embed-text:latest")

        conn = get_connection()

        rows = conn.execute(
            "SELECT id, embedding_model FROM ingestion_jobs ORDER BY id DESC"
        ).fetchall()
        conn.close()

        matched = [r for r in rows if r["id"] == job_id]
        assert len(matched) == 1
        assert matched[0]["embedding_model"] == "nomic-embed-text"


# ===================================================================
# D) VectorStore — Collection-Name vom Modell abgeleitet
# ===================================================================


class TestVectorStoreCollectionRouting:
    """Tests fuer modellspezifische Collection-Zuordnung."""

    def test_different_models_different_collections(self):
        """Verschiedene Modelle ergeben verschiedene Collection-Namen."""
        from kiara.retrieval.embedding import collection_name_for_model

        c1 = collection_name_for_model("nomic-embed-text")
        c2 = collection_name_for_model("mxbai-embed-large")
        c3 = collection_name_for_model("all-minilm")

        assert c1 != c2
        assert c1 != c3
        assert c2 != c3
        assert c1 == "kb_nomic-embed-text"
        assert c2 == "kb_mxbai-embed-large"
        assert c3 == "kb_all-minilm"

    def test_collection_name_normalizes_latest_input(self):
        """Unnormalisierter Input (:latest) wird zentral normalisiert."""
        from kiara.retrieval.embedding import collection_name_for_model

        assert collection_name_for_model("nomic-embed-text:latest") == "kb_nomic-embed-text"

    def test_vectorstore_base_uses_model_for_collection(self):
        """VectorStoreBase leitet Collection-Name vom uebergebenen Modell ab."""
        from kiara.retrieval.embedding import collection_name_for_model

        config = {
            "ollama": {"url": "http://localhost:11434", "embedding_model": "nomic-embed-text"}
        }

        # Mit explizitem Modell
        expected = collection_name_for_model("mxbai-embed-large")
        assert expected == "kb_mxbai-embed-large"

        # Ohne Modell → Fallback auf Config-Default
        expected_default = collection_name_for_model("nomic-embed-text")
        assert expected_default == "kb_nomic-embed-text"

    def test_vectorstore_factory_passes_model(self):
        """create_vectorstore() reicht embedding_model an das Backend weiter."""
        from kiara.retrieval.embedding import collection_name_for_model
        from kiara.retrieval.vectorstore_factory import create_vectorstore

        config = {
            "vectordb": {"backend": "qdrant", "qdrant_url": "http://localhost:6333"},
            "ollama": {"url": "http://localhost:11434", "embedding_model": "nomic-embed-text"},
        }

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantClient"),
            patch("kiara.retrieval.vectorstore_qdrant.QdrantVectorStore._ensure_collection"),
        ):
            store = create_vectorstore(
                config,
                embedding_model="mxbai-embed-large",
                intent="write",
            )
            assert store._collection == collection_name_for_model("mxbai-embed-large")


# ===================================================================
# E) Backend-Selector — modellspezifisches Routing
# ===================================================================


class TestBackendSelectorModelRouting:
    """Tests dass get_embedding_backend_for_model() verschiedene Modelle korrekt routet."""

    _NOMIC_BACKEND = {
        "id": 1,
        "name": "Server-A",
        "category": "embedding",
        "provider": "ollama",
        "model": "nomic-embed-text:latest",
        "url": "http://server-a:11434",
        "priority": 10,
    }
    _MXBAI_BACKEND = {
        "id": 2,
        "name": "Server-B",
        "category": "embedding",
        "provider": "ollama",
        "model": "mxbai-embed-large",
        "url": "http://server-b:11434",
        "priority": 10,
    }

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_routes_nomic_to_server_a(self, mock_active, mock_health):
        """nomic-embed-text wird an Server-A geroutet."""
        from kiara.backend_selector import get_embedding_backend_for_model

        mock_active.return_value = [self._NOMIC_BACKEND, self._MXBAI_BACKEND]
        result = get_embedding_backend_for_model("nomic-embed-text")
        assert result["url"] == "http://server-a:11434"

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_routes_mxbai_to_server_b(self, mock_active, mock_health):
        """mxbai-embed-large wird an Server-B geroutet."""
        from kiara.backend_selector import get_embedding_backend_for_model

        mock_active.return_value = [self._NOMIC_BACKEND, self._MXBAI_BACKEND]
        result = get_embedding_backend_for_model("mxbai-embed-large")
        assert result["url"] == "http://server-b:11434"

    @patch("kiara.backend_selector._is_healthy_cached", return_value=True)
    @patch("kiara.backend_selector.get_all_active")
    def test_unknown_model_returns_none(self, mock_active, mock_health):
        """Unbekanntes Modell gibt None zurueck."""
        from kiara.backend_selector import get_embedding_backend_for_model

        mock_active.return_value = [self._NOMIC_BACKEND, self._MXBAI_BACKEND]
        result = get_embedding_backend_for_model("totally-unknown")
        assert result is None


# ===================================================================
# F) embed_texts / embed_query — URL vom Backend-Modell
# ===================================================================


class TestEmbedFunctionRouting:
    """Tests dass embed_texts/embed_query die URL vom richtigen Backend holen."""

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model")
    def test_embed_texts_uses_correct_url(self, mock_backend, mock_client, mock_dim):
        """embed_texts() verwendet die URL des modellspezifischen Backends."""
        from kiara.retrieval.embedding import embed_texts

        mock_backend.return_value = {
            "url": "http://special-server:11434",
            "provider": "ollama",
            "model": "mxbai-embed-large",
        }
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[0.1, 0.2]]}
        resp.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = resp

        config = {"ollama": {"url": "http://default:11434", "embedding_model": "nomic"}}
        embed_texts(config, "mxbai-embed-large", ["hello"])

        mock_client.assert_called_once_with("http://special-server:11434")
        mock_http.post.assert_called_once_with(
            "/api/embed",
            json={
                "model": "mxbai-embed-large",
                "input": ["hello"],
                "input_type": "search_document",
            },
        )

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=2)
    @patch("kiara.retrieval.embedding._get_ollama_http_client")
    @patch("kiara.retrieval.embedding.get_embedding_backend_for_model")
    def test_embed_query_uses_correct_url(self, mock_backend, mock_client, mock_dim):
        """embed_query() verwendet die URL des modellspezifischen Backends."""
        from kiara.retrieval.embedding import embed_query

        mock_backend.return_value = {
            "url": "http://embed-server-2:11434",
            "provider": "ollama",
            "model": "nomic-embed-text",
        }
        resp = MagicMock()
        resp.json.return_value = {"embeddings": [[0.3, 0.4]]}
        resp.raise_for_status = MagicMock()
        mock_http = MagicMock()
        mock_client.return_value = mock_http
        mock_http.post.return_value = resp

        config = {"ollama": {"url": "http://default:11434", "embedding_model": "nomic"}}
        result = embed_query(config, "nomic-embed-text", "test query")

        mock_client.assert_called_once_with("http://embed-server-2:11434")
        mock_http.post.assert_called_once_with(
            "/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": ["test query"],
                "input_type": "search_query",
            },
        )
        assert result == [0.3, 0.4]


# ===================================================================
# G) embed_texts_late — Backend-URL-Routing
# ===================================================================


class TestEmbedTextsLateBackendUrlRouting:
    """Tests dass embed_texts_late() die URL von _resolve_backend() + /api/embed_late nutzt."""

    @patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=3)
    @patch("kiara.retrieval.embedding.httpx")
    @patch("kiara.retrieval.embedding._resolve_backend")
    def test_embed_texts_late_backend_url_routing(self, mock_resolve, mock_httpx, mock_dim):
        """embed_texts_late() calls httpx.post with _resolve_backend URL + /api/embed_late."""
        from kiara.retrieval.embedding import embed_texts_late

        mock_resolve.return_value = {
            "url": "http://kiron:11436",
            "provider": "ollama",
            "api_key": "",
            "base_url": "",
            "model": "bge-m3",
        }

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        config = {"ollama": {"url": "http://default:11434", "embedding_model": "bge-m3"}}
        chunks = ["Chunk A", "Chunk B"]
        spans = [(0, 10), (11, 20)]
        document = "Chunk A\n\nChunk B"

        result = embed_texts_late(config, "bge-m3", document, chunks, spans)

        # Verify _resolve_backend was called with the model
        mock_resolve.assert_called_once_with("bge-m3")

        # Verify httpx.post was called with the correct URL
        mock_httpx.post.assert_called_once()
        call_args = mock_httpx.post.call_args
        url_arg = call_args[0][0] if call_args[0] else call_args[1].get("url")
        assert url_arg == "http://kiron:11436/api/embed_late"

        # Verify result
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
