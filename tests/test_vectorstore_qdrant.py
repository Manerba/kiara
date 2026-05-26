"""Tests for QdrantVectorStore (fully mocked, no real Qdrant server needed)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointIdsList,
    VectorParams,
)

# Mock get_embedding_model/get_embedding_dimension BEFORE QdrantVectorStore import
_embed_model_patch = patch(
    "kiara.retrieval.vectorstore_base.get_embedding_model", return_value="nomic-embed-text"
)
_embed_dim_patch = patch("kiara.retrieval.embedding.get_embedding_dimension", return_value=768)
_embed_model_patch.start()
_embed_dim_patch.start()

CONFIG = {
    "ollama": {
        "url": "http://localhost:11434",
        "embedding_model": "nomic-embed-text",
        "embedding_url": "",
    },
    "vectordb": {
        "backend": "qdrant",
        "path": "/tmp/test",
        "collection": "test",
        "vector_size": 768,
        "qdrant_url": "http://localhost:6333",
        "qdrant_api_key": "",
    },
}

FAKE_EMBEDDING = [0.1, 0.2, 0.3] * 256  # 768-dim


def _make_mock_client(source_key_ready=False):
    """Create a mock QdrantClient with empty collections.

    Args:
        source_key_ready: If True, get_collection returns a schema with
            source_key index fully populated. If False (default), no
            source_key index — _detect_source_key_ready() returns False.
    """
    client = MagicMock()
    collections_response = MagicMock()
    collections_response.collections = []
    client.get_collections.return_value = collections_response
    client.create_collection.return_value = None
    client.upsert.return_value = None

    # Explizites CollectionInfo fuer _detect_source_key_ready()
    collection_info = MagicMock()
    collection_info.points_count = 100
    # Schema-Validierung in _ensure_payload_indexes erwartet eine echte
    # VectorParams-Instanz (Cosinus, 768 dim) — sonst greift fail-closed.
    collection_info.config.params.vectors = VectorParams(
        size=768,
        distance=Distance.COSINE,
    )
    schema = {}
    if source_key_ready:
        sk_info = MagicMock()
        sk_info.points = 100
        schema["source_key"] = sk_info
    collection_info.payload_schema = schema
    client.get_collection.return_value = collection_info

    return client


@pytest.fixture
def qdrant_store():
    """Create a QdrantVectorStore with fully mocked dependencies."""
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts") as mock_embed_texts,
        patch("kiara.retrieval.vectorstore_qdrant.embed_query") as mock_embed_query,
    ):
        mock_client = _make_mock_client()
        MockClient.return_value = mock_client

        mock_embed_texts.return_value = [[0.1, 0.2, 0.3] * 256]
        mock_embed_query.return_value = [0.1, 0.2, 0.3] * 256

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(CONFIG)

        store._mock_client = mock_client
        store._mock_embed_texts = mock_embed_texts
        store._mock_embed_query = mock_embed_query

        yield store


# ── 1. test_add_creates_points ───────────────────────────────────────────────


def test_add_creates_points(qdrant_store):
    """Upsert is called with correct PointStructs."""
    from qdrant_client.models import PointStruct

    texts = ["Hello World"]
    metadatas = [{"source": "test.txt", "directory": "/docs"}]
    ids = ["id-001"]

    qdrant_store.add(texts, metadatas, ids)

    qdrant_store._mock_embed_texts.assert_called_once()

    call_args = qdrant_store._mock_client.upsert.call_args
    assert call_args is not None, "upsert() was not called"

    _, kwargs = call_args
    points = kwargs["points"]
    assert len(points) == 1

    point = points[0]
    assert isinstance(point, PointStruct)
    assert point.id == "id-001"
    assert point.vector == FAKE_EMBEDDING
    assert point.payload["_text"] == "Hello World"
    assert point.payload["source"] == "test.txt"
    assert point.payload["directory"] == "/docs"


# ── 2. test_query_returns_scored_results ─────────────────────────────────────


def test_query_returns_scored_results(qdrant_store):
    """Return format is correct: list of {text, metadata, score}."""
    mock_point = MagicMock()
    mock_point.payload = {"_text": "Result text", "source": "doc.pdf", "file_type": "pdf"}
    mock_point.score = 0.95

    query_response = MagicMock()
    query_response.points = [mock_point]
    qdrant_store._mock_client.query_points.return_value = query_response

    results = qdrant_store.query("test query", k=3)

    qdrant_store._mock_embed_query.assert_called_once()
    qdrant_store._mock_client.query_points.assert_called_once()

    assert isinstance(results, list)
    assert len(results) == 1

    result = results[0]
    assert result["text"] == "Result text"
    assert result["score"] == 0.95
    assert result["metadata"]["source"] == "doc.pdf"
    assert result["metadata"]["file_type"] == "pdf"
    # _text must be popped from metadata
    assert "_text" not in result["metadata"]


# ── 3. test_filter_conversion ────────────────────────────────────────────────


class TestFilterConversion:
    """where-Filter to Qdrant-Filter conversion tests."""

    def test_simple_match_value(self, qdrant_store):
        """{"key": value} -> FieldCondition(key, MatchValue(value))."""
        where = {"source": "readme.md"}
        result = qdrant_store._build_filter(where)

        assert isinstance(result, Filter)
        assert len(result.must) == 1

        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "source"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "readme.md"

    def test_in_operator(self, qdrant_store):
        """{"key": {"$in": [...]}} -> FieldCondition(key, MatchAny(any=[...]))."""
        where = {"file_type": {"$in": ["pdf", "docx", "txt"]}}
        result = qdrant_store._build_filter(where)

        assert isinstance(result, Filter)
        assert len(result.must) == 1

        cond = result.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "file_type"
        assert isinstance(cond.match, MatchAny)
        assert cond.match.any == ["pdf", "docx", "txt"]

    def test_and_operator(self, qdrant_store):
        """{"$and": [...]} -> Filter(must=[...])."""
        where = {
            "$and": [
                {"source": "readme.md"},
                {"file_type": {"$in": ["md", "txt"]}},
            ]
        }
        result = qdrant_store._build_filter(where)

        assert isinstance(result, Filter)
        assert len(result.must) == 2

        # First condition: simple match
        cond0 = result.must[0]
        assert isinstance(cond0, FieldCondition)
        assert cond0.key == "source"
        assert isinstance(cond0.match, MatchValue)
        assert cond0.match.value == "readme.md"

        # Second condition: $in match
        cond1 = result.must[1]
        assert isinstance(cond1, FieldCondition)
        assert cond1.key == "file_type"
        assert isinstance(cond1.match, MatchAny)
        assert cond1.match.any == ["md", "txt"]


# ── 4. test_ensure_collection_creates_if_missing ────────────────────────────


def test_ensure_collection_creates_if_missing(qdrant_store):
    """Collection is created when not existing."""
    # _ensure_collection was already called during __init__ (collections list was empty).
    qdrant_store._mock_client.get_collections.assert_called_once()
    qdrant_store._mock_client.create_collection.assert_called_once()

    call_kwargs = qdrant_store._mock_client.create_collection.call_args[1]
    assert call_kwargs["collection_name"] == qdrant_store._collection
    vectors_config = call_kwargs["vectors_config"]
    assert vectors_config.size == 768


def test_ensure_collection_skips_if_exists():
    """Collection is NOT created when it already exists."""
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = _make_mock_client()
        existing_collection = MagicMock()
        existing_collection.name = "kb_nomic-embed-text"
        collections_response = MagicMock()
        collections_response.collections = [existing_collection]
        mock_client.get_collections.return_value = collections_response
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(CONFIG)

        mock_client.get_collections.assert_called_once()
        mock_client.create_collection.assert_not_called()


def _make_dimension_info(points_count: int, size: int, distance=Distance.COSINE):
    info = MagicMock()
    info.points_count = points_count
    info.config.params.vectors = VectorParams(size=size, distance=distance)
    info.payload_schema = {}
    return info


def _make_named_vector_info(points_count: int, size: int = 768):
    """CollectionInfo mit dict-based Named-Vector-Layout (statt unbenannter VectorParams)."""
    info = MagicMock()
    info.points_count = points_count
    info.config.params.vectors = {
        "default": VectorParams(size=size, distance=Distance.COSINE),
    }
    info.payload_schema = {}
    return info


def _patch_existing_collection(coll_name, collection_info):
    """Helper: Patcht QdrantClient so, dass die Collection bereits existiert."""
    mock_client = _make_mock_client()
    collections_response = MagicMock()
    collections_response.collections = [SimpleNamespace(name=coll_name)]
    mock_client.get_collections.return_value = collections_response
    mock_client.get_collection.return_value = collection_info
    return mock_client


def test_dimension_mismatch_non_empty_collection_fails_closed():
    """Nicht-leere Collection mit falscher Vektordimension darf nicht weiterlaufen."""
    from kiara.exceptions import VectorDBConnectionError
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        MockClient.return_value = _patch_existing_collection(
            coll_name,
            _make_dimension_info(points_count=12, size=384),
        )

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        with pytest.raises(VectorDBConnectionError):
            QdrantVectorStore(CONFIG)


def test_dimension_mismatch_empty_collection_is_recreated():
    """Leere Collection mit falscher Dimension wird ohne Datenverlust neu erstellt."""
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = _patch_existing_collection(
            coll_name,
            _make_dimension_info(points_count=0, size=384),
        )
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_called_once_with(collection_name=coll_name)
        assert mock_client.create_collection.call_args.kwargs["collection_name"] == coll_name


def test_distance_mismatch_non_empty_collection_fails_closed():
    """Nicht-leere Collection mit abweichender Distance-Metrik muss fail-closed enden.

    DOT/EUCLID auf Cosinus-trainierten Embeddings liefert falsch geordnete
    Treffer — Retrieval-Integritaet ist verletzt.
    """
    from kiara.exceptions import VectorDBConnectionError
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        MockClient.return_value = _patch_existing_collection(
            coll_name,
            _make_dimension_info(points_count=42, size=768, distance=Distance.DOT),
        )

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        with pytest.raises(VectorDBConnectionError) as exc_info:
            QdrantVectorStore(CONFIG)
        assert "Distance" in str(exc_info.value)


def test_distance_mismatch_empty_collection_is_recreated():
    """Leere Collection mit abweichender Distance wird neu erstellt (Cosinus erzwungen)."""
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = _patch_existing_collection(
            coll_name,
            _make_dimension_info(points_count=0, size=768, distance=Distance.EUCLID),
        )
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_called_once_with(collection_name=coll_name)
        recreate_kwargs = mock_client.create_collection.call_args.kwargs
        assert recreate_kwargs["collection_name"] == coll_name
        assert recreate_kwargs["vectors_config"].distance == Distance.COSINE


def test_named_vector_layout_non_empty_collection_fails_closed():
    """Bestehende dict-basierte Named-Vector-Configs muessen fail-closed enden.

    Der else-Branch hatte zuvor jegliche Validierung uebersprungen, sodass
    Upserts spaeter zur Laufzeit gegen das KIara-API-Format kollidiert waeren.
    """
    from kiara.exceptions import VectorDBConnectionError
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        MockClient.return_value = _patch_existing_collection(
            coll_name,
            _make_named_vector_info(points_count=5),
        )

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        with pytest.raises(VectorDBConnectionError) as exc_info:
            QdrantVectorStore(CONFIG)
        assert "Named-Vector" in str(exc_info.value)


def test_named_vector_layout_empty_collection_is_recreated():
    """Leeres Named-Vector-Layout wird mit unbenannter VectorParams neu erstellt."""
    from kiara.retrieval.embedding import collection_name_for_model

    coll_name = collection_name_for_model("nomic-embed-text")
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = _patch_existing_collection(
            coll_name,
            _make_named_vector_info(points_count=0),
        )
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_called_once_with(collection_name=coll_name)
        recreate_kwargs = mock_client.create_collection.call_args.kwargs
        assert recreate_kwargs["collection_name"] == coll_name
        assert isinstance(recreate_kwargs["vectors_config"], VectorParams)
        assert recreate_kwargs["vectors_config"].distance == Distance.COSINE


# ══════════════════════════════════════════════════════════════════════════════
# source_key Tests
# ══════════════════════════════════════════════════════════════════════════════


def _make_collection_info(points_count, source_key_points=None):
    """Helper: Erzeugt ein Mock-CollectionInfo mit optionalem source_key-Index."""
    info = MagicMock()
    info.points_count = points_count
    info.config.params.vectors = VectorParams(size=768, distance=Distance.COSINE)
    schema = {}
    # source-Index ist immer da (TEXT)
    src_info = MagicMock()
    src_info.data_type = "text"
    src_info.points = points_count
    schema["source"] = src_info
    if source_key_points is not None:
        sk_info = MagicMock()
        sk_info.data_type = "keyword"
        sk_info.points = source_key_points
        schema["source_key"] = sk_info
    info.payload_schema = schema
    return info


def _make_store_with_collection_info(collection_info):
    """Helper: Erzeugt QdrantVectorStore mit vorkonfiguriertem get_collection-Mock."""
    with (
        patch("kiara.retrieval.vectorstore_qdrant.QdrantClient") as MockClient,
        patch("kiara.retrieval.vectorstore_qdrant.embed_texts"),
        patch("kiara.retrieval.vectorstore_qdrant.embed_query"),
    ):
        mock_client = _make_mock_client()
        mock_client.get_collection.return_value = collection_info
        MockClient.return_value = mock_client

        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(CONFIG)
        store._mock_client = mock_client
        return store


# ── 5. _detect_source_key_ready ──────────────────────────────────────────────


class TestDetectSourceKeyReady:
    """Startup-Check: Entscheidung ob source_key (KEYWORD) genutzt wird."""

    def test_ready_when_all_points_indexed(self):
        """Migration abgeschlossen: indexed >= total -> True."""
        info = _make_collection_info(points_count=3800000, source_key_points=3800000)
        store = _make_store_with_collection_info(info)
        assert store._use_source_key is True

    def test_false_when_no_source_key_index(self):
        """Kein source_key-Index vorhanden -> False."""
        info = _make_collection_info(points_count=3800000, source_key_points=None)
        store = _make_store_with_collection_info(info)
        assert store._use_source_key is False

    def test_false_when_migration_incomplete(self):
        """Migration unvollstaendig: indexed < total -> False."""
        info = _make_collection_info(points_count=3800000, source_key_points=1000000)
        store = _make_store_with_collection_info(info)
        assert store._use_source_key is False

    def test_ready_when_collection_empty(self):
        """Leere Collection -> True (nichts zu migrieren)."""
        info = _make_collection_info(points_count=0, source_key_points=None)
        # Leere Collection: kein Index, aber points_count=0
        info.payload_schema = {}
        store = _make_store_with_collection_info(info)
        # points_count=0 UND kein source_key-Index -> sk_info is None -> False
        # Aber: wenn source_key Index existiert und Collection leer:
        info2 = _make_collection_info(points_count=0, source_key_points=0)
        store2 = _make_store_with_collection_info(info2)
        assert store2._use_source_key is True


# ── 6. _source_match_filter ──────────────────────────────────────────────────


class TestSourceMatchFilter:
    """Filter-Routing: source_key (KEYWORD) vs source (TEXT) je nach Migration."""

    def test_uses_source_key_when_ready(self):
        """Nach Migration: Filter auf source_key (KEYWORD-Index)."""
        info = _make_collection_info(points_count=100, source_key_points=100)
        store = _make_store_with_collection_info(info)
        assert store._use_source_key is True

        f = store._source_match_filter("/mnt/data/test.pdf")

        assert isinstance(f, Filter)
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "source_key"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "/mnt/data/test.pdf"

    def test_falls_back_to_source_before_migration(self):
        """Vor Migration: Filter auf source (TEXT-Index, Full Scan)."""
        info = _make_collection_info(points_count=100, source_key_points=None)
        store = _make_store_with_collection_info(info)
        assert store._use_source_key is False

        f = store._source_match_filter("/mnt/data/test.pdf")

        assert isinstance(f, Filter)
        assert len(f.must) == 1
        cond = f.must[0]
        assert cond.key == "source"
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "/mnt/data/test.pdf"


# ── 7. Payload-Writes ────────────────────────────────────────────────────────


def test_add_includes_source_key(qdrant_store):
    """add() schreibt source_key mit identischem Wert wie source in Payload."""
    texts = ["content"]
    metadatas = [{"source": "/data/report.pdf", "directory": "/data"}]
    ids = ["chunk-1"]

    qdrant_store.add(texts, metadatas, ids)

    call_args = qdrant_store._mock_client.upsert.call_args
    point = call_args[1]["points"][0]
    assert point.payload["source_key"] == "/data/report.pdf"
    assert point.payload["source"] == point.payload["source_key"]


def test_add_with_embeddings_includes_source_key(qdrant_store):
    """add_with_embeddings() schreibt source_key in Payload."""
    texts = ["content"]
    metadatas = [{"source": "/data/invoice.pdf", "directory": "/data"}]
    ids = ["chunk-2"]
    embeddings = [FAKE_EMBEDDING]

    qdrant_store.add_with_embeddings(texts, metadatas, ids, embeddings)

    call_args = qdrant_store._mock_client.upsert.call_args
    point = call_args[1]["points"][0]
    assert point.payload["source_key"] == "/data/invoice.pdf"
    assert point.payload["source"] == point.payload["source_key"]


def test_delete_by_ids_uses_point_ids_list(qdrant_store):
    """delete_by_ids nutzt den Qdrant-Vertrag fuer explizite Point-IDs."""
    qdrant_store.delete_by_ids(["id-1", "id-2"])

    kwargs = qdrant_store._mock_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == qdrant_store._collection
    assert isinstance(kwargs["points_selector"], PointIdsList)
    assert kwargs["points_selector"].points == ["id-1", "id-2"]


# ── 8. Migration ─────────────────────────────────────────────────────────────


def test_migrate_source_key_skips_existing(qdrant_store):
    """Bereits migrierte Punkte (source_key == source) werden uebersprungen."""
    # 2 Punkte: einer bereits migriert, einer ohne source_key
    p_migrated = MagicMock()
    p_migrated.id = "id-1"
    p_migrated.payload = {"source": "/data/a.pdf", "source_key": "/data/a.pdf"}

    p_pending = MagicMock()
    p_pending.id = "id-2"
    p_pending.payload = {"source": "/data/b.pdf"}

    # scroll gibt einen Batch zurueck, dann None-Offset (Ende)
    qdrant_store._mock_client.scroll.return_value = ([p_migrated, p_pending], None)

    stats = qdrant_store.migrate_source_key()

    assert stats["total"] == 2
    assert stats["skipped"] == 1
    assert stats["migrated"] == 1
    assert stats["errors"] == 0

    # set_payload nur fuer den nicht-migrierten Punkt
    call_args = qdrant_store._mock_client.set_payload.call_args
    assert call_args[1]["payload"] == {"source_key": "/data/b.pdf"}
    assert call_args[1]["points"] == ["id-2"]


def test_migrate_source_key_handles_errors(qdrant_store):
    """set_payload-Fehler werden gezaehlt, Migration laeuft weiter."""
    p1 = MagicMock()
    p1.id = "id-1"
    p1.payload = {"source": "/data/a.pdf"}

    qdrant_store._mock_client.scroll.return_value = ([p1], None)
    qdrant_store._mock_client.set_payload.side_effect = RuntimeError("connection lost")

    stats = qdrant_store.migrate_source_key()

    assert stats["total"] == 1
    assert stats["migrated"] == 0
    assert stats["errors"] == 1


# ── 9. add_with_embeddings ──────────────────────────────────────────────────


def test_add_with_embeddings_basic(qdrant_store):
    """add_with_embeddings() stores pre-computed vectors via upsert."""
    from qdrant_client.models import PointStruct

    texts = ["Chunk A", "Chunk B"]
    metadatas = [
        {"source": "/data/a.pdf", "directory": "/data"},
        {"source": "/data/b.pdf", "directory": "/data"},
    ]
    ids = ["id-a", "id-b"]
    embeddings = [FAKE_EMBEDDING, FAKE_EMBEDDING]

    qdrant_store.add_with_embeddings(texts, metadatas, ids, embeddings)

    call_args = qdrant_store._mock_client.upsert.call_args
    assert call_args is not None, "upsert() was not called"

    _, kwargs = call_args
    points = kwargs["points"]
    assert len(points) == 2

    for i, point in enumerate(points):
        assert isinstance(point, PointStruct)
        assert point.id == ids[i]
        assert point.vector == FAKE_EMBEDDING
        assert point.payload["_text"] == texts[i]
        assert point.payload["source"] == metadatas[i]["source"]
        assert point.payload["directory"] == metadatas[i]["directory"]


def test_add_with_embeddings_multi_batch(qdrant_store):
    """add_with_embeddings() splits into multiple upsert calls when exceeding batch_size."""
    from kiara.shared.constants import QDRANT_UPSERT_BATCH_SIZE

    count = QDRANT_UPSERT_BATCH_SIZE + 5
    texts = [f"Text {i}" for i in range(count)]
    metadatas = [{"source": f"/data/file{i}.txt", "directory": "/data"} for i in range(count)]
    ids = [f"id-{i}" for i in range(count)]
    embeddings = [FAKE_EMBEDDING for _ in range(count)]

    qdrant_store._mock_client.upsert.reset_mock()

    qdrant_store.add_with_embeddings(texts, metadatas, ids, embeddings)

    upsert_calls = qdrant_store._mock_client.upsert.call_args_list
    assert len(upsert_calls) == 2, f"Expected 2 upsert batches, got {len(upsert_calls)}"

    # First batch: QDRANT_UPSERT_BATCH_SIZE points
    first_batch_points = upsert_calls[0][1]["points"]
    assert len(first_batch_points) == QDRANT_UPSERT_BATCH_SIZE

    # Second batch: remaining 5 points
    second_batch_points = upsert_calls[1][1]["points"]
    assert len(second_batch_points) == 5

    # Verify correct embedding slicing: last point in second batch
    last_point = second_batch_points[-1]
    assert last_point.id == f"id-{count - 1}"
    assert last_point.vector == FAKE_EMBEDDING
    assert last_point.payload["_text"] == f"Text {count - 1}"


# ── 10. bge-m3 dimension ────────────────────────────────────────────────────


def test_bge_m3_dimension_1024():
    """get_embedding_dimension() returns 1024 for bge-m3 via KNOWN_EMBEDDING_DIMENSIONS."""
    from kiara.shared.constants import KNOWN_EMBEDDING_DIMENSIONS

    assert "bge-m3" in KNOWN_EMBEDDING_DIMENSIONS
    assert KNOWN_EMBEDDING_DIMENSIONS["bge-m3"] == 1024

    # Stop the module-level patch to test the real get_embedding_dimension
    _embed_dim_patch.stop()
    try:
        from importlib import reload

        import kiara.retrieval.embedding as emb_module

        reload(emb_module)

        with (
            patch("kiara.retrieval.embedding.get_all_active", return_value=[]),
            patch("kiara.retrieval.embedding.get_embedding_backend_for_model", return_value=None),
        ):
            dim = emb_module.get_embedding_dimension("bge-m3", {})
            assert dim == 1024
    finally:
        _embed_dim_patch.start()


class TestListCollections:
    """list_collections unterscheidet Datenfehler von echten Qdrant-Fehlern."""

    def test_get_collection_failure_is_not_hidden_as_empty_list(self):
        from kiara.retrieval.embedding import collection_name_for_model

        client = MagicMock()
        client.get_collections.return_value.collections = [
            SimpleNamespace(name=collection_name_for_model("nomic-embed-text")),
        ]
        client.get_collection.side_effect = RuntimeError("qdrant api changed")

        with patch("kiara.retrieval.vectorstore_qdrant.QdrantClient", return_value=client):
            from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

            with pytest.raises(RuntimeError):
                QdrantVectorStore.list_collections(CONFIG)

    def test_invalid_collection_name_is_skipped_locally(self):
        client = MagicMock()
        client.get_collections.return_value.collections = [
            SimpleNamespace(name="kb_bad"),
            SimpleNamespace(name="kb_good"),
        ]
        client.get_collection.return_value.points_count = 3

        def _model_name(name):
            if name == "kb_bad":
                raise ValueError("bad name")
            return "good"

        with (
            patch("kiara.retrieval.vectorstore_qdrant.QdrantClient", return_value=client),
            patch(
                "kiara.retrieval.vectorstore_qdrant.model_name_from_collection",
                side_effect=_model_name,
            ),
        ):
            from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

            result = QdrantVectorStore.list_collections(CONFIG)

        assert result == [{"name": "kb_good", "model": "good", "chunks": 3}]
        client.get_collection.assert_called_once_with("kb_good")


# ── 10. VectorDBConnectionError-Klassifikation (V35 #734) ────────────────────


class TestVectorDBConnectionErrorClassification:
    """Isolierte Qdrant-Upsert-/set_payload-Fehler werden als
    VectorDBConnectionError typisiert. ValueError-Laengenguards und
    BackendUnavailableError aus embed_texts bleiben unveraendert."""

    def test_add_upsert_failure_typed_as_vectordb_connection_error(self, qdrant_store):
        from kiara.exceptions import VectorDBConnectionError

        qdrant_store._mock_client.upsert.side_effect = RuntimeError("qdrant down")

        with pytest.raises(VectorDBConnectionError) as exc_info:
            qdrant_store.add(
                ["txt"],
                [{"source": "a.txt"}],
                ["id-01"],
            )
        assert "qdrant down" in str(exc_info.value)

    def test_add_length_mismatch_remains_value_error(self, qdrant_store):
        # Embedding-Liefermenge abweichend → ValueError, nicht VectorDBConnectionError
        qdrant_store._mock_embed_texts.return_value = [[0.1] * 768, [0.2] * 768]
        with pytest.raises(ValueError):
            qdrant_store.add(["a"], [{"source": "a.txt"}], ["id-01"])
        qdrant_store._mock_client.upsert.assert_not_called()

    def test_add_embedding_backend_error_not_reclassified(self, qdrant_store):
        from kiara.backend_selector import BackendUnavailableError

        qdrant_store._mock_embed_texts.side_effect = BackendUnavailableError("ollama down")
        with pytest.raises(BackendUnavailableError):
            qdrant_store.add(["a"], [{"source": "a.txt"}], ["id-01"])
        # Embedding-Pfad darf NICHT als VectorDBConnectionError laufen
        qdrant_store._mock_client.upsert.assert_not_called()

    def test_add_with_embeddings_upsert_failure_typed(self, qdrant_store):
        from kiara.exceptions import VectorDBConnectionError

        qdrant_store._mock_client.upsert.side_effect = RuntimeError("upsert blew up")

        texts = ["t"]
        metas = [{"source": "a.txt"}]
        ids = ["id-01"]
        embs = [[0.1] * 768]
        with pytest.raises(VectorDBConnectionError):
            qdrant_store.add_with_embeddings(texts, metas, ids, embs)

    def test_add_with_embeddings_length_mismatch_remains_value_error(self, qdrant_store):
        texts = ["t"]
        metas = [{"source": "a.txt"}]
        ids = ["id-01", "id-02"]
        embs = [[0.1] * 768]
        with pytest.raises(ValueError):
            qdrant_store.add_with_embeddings(texts, metas, ids, embs)
        qdrant_store._mock_client.upsert.assert_not_called()

    def test_update_ds_ids_final_set_payload_error_typed(self, qdrant_store):
        from kiara.exceptions import VectorDBConnectionError

        qdrant_store._mock_client.set_payload.side_effect = RuntimeError("payload timeout")
        ds_ids_map = {"id-01": [1], "id-02": [1, 2]}

        with patch("time.sleep"), pytest.raises(VectorDBConnectionError):
            qdrant_store.update_ds_ids(ds_ids_map)
