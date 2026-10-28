"""Tests fuer kiara.retrieval.colbert — ColBERT Store/API."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")


class TestCollectionNaming:
    """Collection-Naming und DS-Key-Funktionen."""

    def test_colbert_collection_name_simple(self):
        from kiara.retrieval.colbert import colbert_collection_name

        assert colbert_collection_name("colbert-xm") == "colbert_colbert-xm"

    def test_colbert_collection_name_roundtrip(self):
        from kiara.retrieval.colbert import (
            colbert_collection_name,
            model_name_from_colbert_collection,
        )

        model = "colbert-xm"
        coll = colbert_collection_name(model)
        assert model_name_from_colbert_collection(coll) == model

    def test_colbert_collection_name_special_chars(self):
        from kiara.retrieval.colbert import (
            colbert_collection_name,
            model_name_from_colbert_collection,
        )

        model = "my/model:v2"
        coll = colbert_collection_name(model)
        assert coll.startswith("colbert_")
        assert model_name_from_colbert_collection(coll) == model

    def test_invalid_collection_name(self):
        from kiara.retrieval.colbert import model_name_from_colbert_collection

        with pytest.raises(ValueError):
            model_name_from_colbert_collection("kb_something")

    def test_empty_model_raises(self):
        from kiara.retrieval.colbert import colbert_collection_name

        with pytest.raises(ValueError):
            colbert_collection_name("")


class TestDsModelKey:
    """colbert_ds_model_key / model_name_from_colbert_ds_key."""

    def test_roundtrip(self):
        from kiara.retrieval.colbert import colbert_ds_model_key, model_name_from_colbert_ds_key

        key = colbert_ds_model_key("colbert-xm")
        assert key == "colbert::colbert-xm"
        assert model_name_from_colbert_ds_key(key) == "colbert-xm"

    def test_invalid_key(self):
        from kiara.retrieval.colbert import model_name_from_colbert_ds_key

        with pytest.raises(ValueError):
            model_name_from_colbert_ds_key("nomic-embed-text")


class TestResolveColbertUrl:
    """_resolve_colbert_url Strategien."""

    def test_explicit_url(self):
        from kiara.retrieval.colbert import _resolve_colbert_url

        config = {"retrieval": {"colbert": {"url": "http://myhost:1234"}}}
        assert _resolve_colbert_url(config) == "http://myhost:1234"

    @patch("kiara.retrieval.colbert.get_embedding_model", return_value="nomic-embed-text")
    @patch("kiara.retrieval.colbert.get_embedding_backend_for_model")
    def test_primary_backend_fallback(self, mock_backend, mock_model):
        from kiara.retrieval.colbert import _resolve_colbert_url

        mock_backend.return_value = {"url": "http://gpu-server.local:11434"}
        config = {"retrieval": {"colbert": {"url": ""}}}
        assert _resolve_colbert_url(config) == "http://gpu-server.local:11434"

    @patch("kiara.retrieval.colbert.get_embedding_model", return_value="")
    @patch("kiara.retrieval.colbert.get_all_active", return_value=[])
    def test_no_backend_raises(self, mock_active, mock_model):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import _resolve_colbert_url

        config = {"retrieval": {"colbert": {"url": ""}}}
        with pytest.raises(BackendUnavailableError):
            _resolve_colbert_url(config)

    @patch(
        "kiara.retrieval.colbert.get_embedding_model", side_effect=AssertionError("programming bug")
    )
    def test_unexpected_primary_resolution_error_propagates(self, mock_model):
        """Unerwartete Fehler werden nicht durch Fallback-Logik verschluckt."""
        from kiara.retrieval.colbert import _resolve_colbert_url

        config = {"retrieval": {"colbert": {"url": ""}}}
        with pytest.raises(AssertionError):
            _resolve_colbert_url(config)


class TestEmbedColbert:
    """embed_colbert HTTP-Client."""

    def test_empty_texts(self):
        from kiara.retrieval.colbert import embed_colbert

        result = embed_colbert({}, "colbert-xm", [], input_type="search_document")
        assert result == []

    @patch("kiara.retrieval.colbert._resolve_colbert_url", return_value="http://localhost:11434")
    @patch("kiara.retrieval.colbert.httpx")
    def test_successful_response(self, mock_httpx, mock_url):
        from kiara.retrieval.colbert import embed_colbert

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embeddings": [[[0.1, 0.2], [0.3, 0.4]]]}
        mock_resp.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_resp
        result = embed_colbert({}, "colbert-xm", ["test"], input_type="search_document")
        assert len(result) == 1
        assert len(result[0]) == 2

    @patch("kiara.retrieval.colbert._resolve_colbert_url", return_value="http://localhost:11434")
    @patch("kiara.retrieval.colbert.httpx")
    def test_http_error_raises(self, mock_httpx, mock_url):
        import httpx

        from kiara.backend_selector import BackendUnavailableError
        from kiara.retrieval.colbert import embed_colbert

        mock_httpx.post.side_effect = httpx.HTTPError("connection refused")
        mock_httpx.HTTPError = httpx.HTTPError
        with pytest.raises(BackendUnavailableError):
            embed_colbert({}, "colbert-xm", ["test"], input_type="search_document")


class TestQdrantFilter:
    """build_qdrant_filter Konvertierung (shared mit QdrantVectorStore)."""

    def test_none_filter(self):
        from kiara.retrieval.vectorstore_qdrant import build_qdrant_filter

        assert build_qdrant_filter(None) is None

    def test_ds_ids_filter(self):
        from kiara.retrieval.vectorstore_qdrant import build_qdrant_filter

        f = build_qdrant_filter({"ds_ids": {"$in": [1, 2]}})
        assert f is not None
        assert len(f.must) == 1

    def test_combined_filter(self):
        from kiara.retrieval.vectorstore_qdrant import build_qdrant_filter

        f = build_qdrant_filter(
            {
                "$and": [
                    {"ds_ids": {"$in": [1]}},
                    {"file_type": ".pdf"},
                    {"source": {"$contains": "test"}},
                ]
            }
        )
        assert f is not None
        assert len(f.must) == 3


class TestColBERTStoreInit:
    """ColBERTStore Konstruktion."""

    @patch("kiara.retrieval.colbert.embed_colbert", return_value=[[[0.1] * 128]])
    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_collection_name(self, mock_client, _mock_embed):
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore({"vectordb": {}}, "colbert-xm", intent="write")
        assert store.get_collection_name() == "colbert_colbert-xm"

    def test_source_payload_index_is_text(self):
        """source braucht TEXT, weil filter_source MatchText/$contains nutzt."""
        from qdrant_client.models import PayloadSchemaType

        from kiara.retrieval.colbert import ColBERTStore

        assert ColBERTStore._PAYLOAD_INDEXES["source"] == PayloadSchemaType.TEXT

    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_maintenance_intent_skips_semantic_setup_and_blocks_upsert(self, mock_client):
        from kiara.retrieval.colbert import ColBERTStore

        client = MagicMock()
        mock_client.return_value = client
        with patch.object(ColBERTStore, "ensure_collection") as ensure_collection:
            store = ColBERTStore(
                {"vectordb": {}},
                "colbert-xm",
                intent="maintenance",
            )

        ensure_collection.assert_not_called()
        with pytest.raises(
            RuntimeError,
            match="maintenance_colbert_semantic_operation_forbidden",
        ):
            store.add(
                texts=["text"],
                metadatas=[{"source": "a.txt"}],
                ids=["id-01"],
                token_embeddings=[[[0.1] * 4]],
            )
        client.upsert.assert_not_called()

    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_payload_index_type_mismatch_is_recreated(self, mock_client):
        """Falscher bestehender source-Index wird auf TEXT umgebaut."""
        from qdrant_client.models import PayloadSchemaType

        from kiara.retrieval.colbert import ColBERTStore

        client = MagicMock()
        collections = MagicMock()
        col = MagicMock()
        col.name = "colbert_colbert-xm"
        collections.collections = [col]
        client.get_collections.return_value = collections

        source_schema = MagicMock()
        source_schema.data_type = PayloadSchemaType.KEYWORD
        coll_info = MagicMock()
        coll_info.payload_schema = {"source": source_schema}
        client.get_collection.return_value = coll_info
        mock_client.return_value = client

        ColBERTStore({"vectordb": {}}, "colbert-xm", intent="write")

        client.delete_payload_index.assert_called_once_with(
            collection_name="colbert_colbert-xm",
            field_name="source",
        )
        assert any(
            call.kwargs.get("field_name") == "source"
            and call.kwargs.get("field_schema") == PayloadSchemaType.TEXT
            for call in client.create_payload_index.call_args_list
        )


class TestColBERTVectorDBConnectionError:
    """V35 #734 — Qdrant-Fehler in ColBERT-Methoden propagieren als
    VectorDBConnectionError. Length-Guards bleiben ValueError."""

    def _mock_client_for_existing_collection(self):
        client = MagicMock()
        coll_info = MagicMock()
        coll_info.payload_schema = {}
        collections = MagicMock()
        col = MagicMock()
        col.name = "colbert_colbert-xm"
        collections.collections = [col]
        client.get_collections.return_value = collections
        client.get_collection.return_value = coll_info
        return client

    def test_ensure_collection_get_collections_failure_typed(self):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.retrieval.colbert import ColBERTStore

        with patch("kiara.retrieval.colbert.QdrantClient") as MockClient:
            client = MagicMock()
            client.get_collections.side_effect = RuntimeError("qdrant unreachable")
            MockClient.return_value = client
            with pytest.raises(VectorDBConnectionError):
                ColBERTStore({"vectordb": {}}, "colbert-xm")

    def test_add_upsert_failure_typed(self):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.retrieval.colbert import ColBERTStore

        with patch("kiara.retrieval.colbert.QdrantClient") as MockClient:
            client = self._mock_client_for_existing_collection()
            MockClient.return_value = client
            store = ColBERTStore({"vectordb": {}}, "colbert-xm")
            client.upsert.side_effect = RuntimeError("qdrant upsert failed")
            with pytest.raises(VectorDBConnectionError):
                store.add(
                    texts=["t"],
                    metadatas=[{"source": "a.txt"}],
                    ids=["id-01"],
                    token_embeddings=[[[0.1] * 4]],
                )

    def test_add_length_guard_remains_value_error(self):
        from kiara.retrieval.colbert import ColBERTStore

        with patch("kiara.retrieval.colbert.QdrantClient") as MockClient:
            client = self._mock_client_for_existing_collection()
            MockClient.return_value = client
            store = ColBERTStore({"vectordb": {}}, "colbert-xm")
            with pytest.raises(ValueError):
                store.add(
                    texts=["t"],
                    metadatas=[{"source": "a.txt"}],
                    ids=["id-01", "id-02"],
                    token_embeddings=[[[0.1] * 4]],
                )
            client.upsert.assert_not_called()

    def test_update_ds_ids_set_payload_failure_typed(self):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.retrieval.colbert import ColBERTStore

        with patch("kiara.retrieval.colbert.QdrantClient") as MockClient:
            client = self._mock_client_for_existing_collection()
            MockClient.return_value = client
            store = ColBERTStore({"vectordb": {}}, "colbert-xm")
            client.set_payload.side_effect = RuntimeError("payload timeout")
            with pytest.raises(VectorDBConnectionError):
                store.update_ds_ids({"id-01": [1]}, wait=True)

    def test_search_query_points_failure_typed(self):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.retrieval.colbert import ColBERTStore

        with patch("kiara.retrieval.colbert.QdrantClient") as MockClient:
            client = self._mock_client_for_existing_collection()
            MockClient.return_value = client
            store = ColBERTStore({"vectordb": {}}, "colbert-xm")
            client.query_points.side_effect = RuntimeError("qdrant query timeout")
            with pytest.raises(VectorDBConnectionError, match="query_points"):
                store.search([[0.1] * 4], k=3)
