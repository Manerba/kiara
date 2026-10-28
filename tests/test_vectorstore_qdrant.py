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

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

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

        store = QdrantVectorStore(CONFIG, intent="write")

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

    qdrant_store._mock_embed_texts.assert_called_once_with(
        CONFIG,
        "nomic-embed-text",
        texts,
    )

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

    qdrant_store._mock_embed_query.assert_called_once_with(
        CONFIG,
        "nomic-embed-text",
        "test query",
    )
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


def test_search_routes_raw_semantic_query_to_operations_based_embedder(qdrant_store):
    """Pagination-Suche nutzt denselben search_query-Requestpfad."""
    response = MagicMock()
    response.points = []
    qdrant_store._mock_client.query_points.return_value = response

    assert qdrant_store.search("expanded semantic query", k=4, offset=2) == []

    qdrant_store._mock_embed_query.assert_called_once_with(
        CONFIG,
        "nomic-embed-text",
        "expanded semantic query",
    )


def test_query_by_ids_routes_raw_hybrid_query_to_operations_based_embedder(qdrant_store):
    """ID-vorgefilterte Hybrid-Suche nutzt ebenfalls search_query."""
    response = MagicMock()
    response.points = []
    qdrant_store._mock_client.query_points.return_value = response

    assert qdrant_store.query_by_ids("hybrid query", ["id-1"], k=3) == []

    qdrant_store._mock_embed_query.assert_called_once_with(
        CONFIG,
        "nomic-embed-text",
        "hybrid query",
    )


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


def test_dimension_mismatch_empty_collection_fails_without_recreation():
    """Auch eine leere Collection mit falscher Dimension bleibt unangetastet."""
    from kiara.exceptions import VectorDBConnectionError
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

        with pytest.raises(VectorDBConnectionError):
            QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()


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


def test_distance_mismatch_empty_collection_fails_without_recreation():
    """Eine leere Collection mit abweichender Distance wird nicht ersetzt."""
    from kiara.exceptions import VectorDBConnectionError
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

        with pytest.raises(VectorDBConnectionError):
            QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()


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


def test_named_vector_layout_empty_collection_fails_without_recreation():
    """Auch ein leeres Named-Vector-Layout wird niemals automatisch ersetzt."""
    from kiara.exceptions import VectorDBConnectionError
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

        with pytest.raises(VectorDBConnectionError):
            QdrantVectorStore(CONFIG)

        mock_client.delete_collection.assert_not_called()
        mock_client.create_collection.assert_not_called()


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
        collections = MagicMock()
        collections.collections = [SimpleNamespace(name="kb_nomic-embed-text")]
        mock_client.get_collections.return_value = collections
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


# ── 7. Paginierter Source-Read ───────────────────────────────────────────────


class TestListSourcesForDatasource:
    def test_explicit_sources_use_one_indexed_match_any_filter(self):
        """Eine Dateiseite liest nur deren source_key-Werte aus Qdrant."""
        info = _make_collection_info(points_count=100, source_key_points=100)
        store = _make_store_with_collection_info(info)
        point_a = MagicMock()
        point_a.payload = {
            "source": "/data/a.pdf",
            "file_type": ".pdf",
            "modified": "2026-01-01T00:00:00Z",
        }
        point_b = MagicMock()
        point_b.payload = dict(point_a.payload)
        store._mock_client.scroll.return_value = ([point_a, point_b], None)

        result = store.list_sources_for_datasource(
            12,
            sources=["/data/a.pdf", "/data/b.pdf"],
        )

        assert result == [
            {
                "source": "/data/a.pdf",
                "file_type": ".pdf",
                "chunk_count": 2,
                "modified": "2026-01-01T00:00:00Z",
            }
        ]
        scroll_filter = store._mock_client.scroll.call_args.kwargs["scroll_filter"]
        assert [condition.key for condition in scroll_filter.must] == ["ds_ids", "source_key"]
        assert isinstance(scroll_filter.must[1].match, MatchAny)
        assert scroll_filter.must[1].match.any == ["/data/a.pdf", "/data/b.pdf"]

    def test_empty_explicit_source_page_skips_qdrant(self):
        info = _make_collection_info(points_count=100, source_key_points=100)
        store = _make_store_with_collection_info(info)
        store._mock_client.scroll.reset_mock()

        assert store.list_sources_for_datasource(12, sources=[]) == []
        store._mock_client.scroll.assert_not_called()


# ── 8. Payload-Writes ────────────────────────────────────────────────────────


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

    def test_payload_update_uses_stored_binding_guard(self, qdrant_store):
        from contextlib import nullcontext

        with patch.object(
            qdrant_store,
            "_index_contract_guard",
            return_value=nullcontext(),
        ) as guard:
            qdrant_store.update_ds_ids({"id-01": [1]}, wait=True)

        guard.assert_called_once_with(
            operation="vector_payload_update",
            write=True,
            validation_scope="stored_binding",
        )

    def test_exact_delete_uses_stored_binding_and_waits_for_commit(self, qdrant_store):
        from contextlib import nullcontext

        with patch.object(
            qdrant_store,
            "_index_contract_guard",
            return_value=nullcontext(),
        ) as guard:
            qdrant_store.delete_by_ids(["id-01"])

        guard.assert_called_once_with(
            operation="vector_delete",
            write=True,
            validation_scope="stored_binding",
        )
        qdrant_store._mock_client.delete.assert_called_once()
        assert qdrant_store._mock_client.delete.call_args.kwargs["wait"] is True


def test_maintenance_intent_never_probes_embedding_or_allows_upsert():
    mock_client = _make_mock_client(source_key_ready=True)
    with (
        patch(
            "kiara.retrieval.vectorstore_qdrant.QdrantClient",
            return_value=mock_client,
        ),
        patch("kiara.retrieval.embedding.get_embedding_dimension") as dimension,
    ):
        from kiara.retrieval.vectorstore_qdrant import QdrantVectorStore

        store = QdrantVectorStore(
            CONFIG,
            embedding_model="nomic-embed-text",
            intent="maintenance",
        )

    dimension.assert_not_called()
    with pytest.raises(RuntimeError, match="maintenance_vectorstore_semantic_operation_forbidden"):
        store.add(["text"], [{"source": "a.txt"}], ["id-01"])
    mock_client.upsert.assert_not_called()


def test_document_range_is_bounded_filtered_and_needs_no_embedding(qdrant_store):
    from contextlib import nullcontext

    client = qdrant_store._mock_client
    client.scroll.return_value = (
        [
            SimpleNamespace(
                id="c1",
                payload={"source": "doc.txt", "chunk_index": 4, "ds_ids": [7], "_text": "Original"},
            )
        ],
        None,
    )
    with patch.object(qdrant_store, "_index_contract_guard", return_value=nullcontext()) as guard:
        result = qdrant_store.get_document_chunks("doc.txt", 4, 7, [7])
    assert result["documents"] == ["Original"]
    call = client.scroll.call_args.kwargs
    assert call["limit"] == 4 and call["with_vectors"] is False
    conditions = {condition.key: condition for condition in call["scroll_filter"].must}
    assert conditions["source"].match.value == "doc.txt"
    assert conditions["chunk_index"].range.gte == 4
    assert conditions["chunk_index"].range.lt == 7
    assert conditions["ds_ids"].match.any == [7]
    assert conditions["parser"].match.value == "standard"
    guard.assert_called_once_with(
        operation="vector_payload_read", write=False, validation_scope="stored_binding"
    )
    qdrant_store._mock_embed_query.assert_not_called()
    client.query_points.assert_not_called()


def test_document_range_does_not_ignore_ambiguous_extra_chunks(qdrant_store):
    from kiara.exceptions import VectorDBConnectionError

    qdrant_store._mock_client.scroll.return_value = ([], "more-points")
    with pytest.raises(VectorDBConnectionError):
        qdrant_store.get_document_chunks("doc.txt", 0, 1, [7])
    qdrant_store._mock_client.scroll.reset_mock()
    assert qdrant_store.get_document_chunks("doc.txt", 0, 1, []) == {
        "ids": [],
        "documents": [],
        "metadatas": [],
    }
    qdrant_store._mock_client.scroll.assert_not_called()


def test_document_read_filters_actual_qdrant_points_by_parser(qdrant_store):
    import asyncio
    from contextlib import nullcontext

    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    from kiara.retrieval.retriever import Retriever
    from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool

    memory = QdrantClient(location=":memory:")
    collection = qdrant_store.get_collection_name()
    memory.create_collection(
        collection, vectors_config=VectorParams(size=1, distance=Distance.COSINE)
    )
    meta = {
        "source": "drawing.png",
        "ds_ids": [7],
        "modified": "2026-10-01T00:00:00+00:00",
        "size_bytes": 1234,
    }
    points = [
        PointStruct(
            id=1,
            vector=[1.0],
            payload={
                **meta,
                "parser": "docling",
                "chunk_index": 0,
                "chunk_total": 2,
                "_text": "OCR Anfang",
            },
        ),
        PointStruct(
            id=2,
            vector=[1.0],
            payload={
                **meta,
                "parser": "docling",
                "chunk_index": 1,
                "chunk_total": 2,
                "_text": "OCR Ende",
            },
        ),
        PointStruct(
            id=3,
            vector=[1.0],
            payload={
                **meta,
                "parser": "vlm",
                "chunk_index": 0,
                "chunk_total": 1,
                "_text": "VLM Bildbeschreibung",
            },
        ),
    ]
    memory.upsert(collection, points)
    qdrant_store._mock_client.scroll.side_effect = memory.scroll
    retriever = Retriever.__new__(Retriever)
    retriever._store = qdrant_store
    references = DocumentReferences(collection)
    try:
        with (
            patch.object(qdrant_store, "_index_contract_guard", return_value=nullcontext()),
            patch(
                "kiara.ingestion._file_utils.ChunkDatasources.get_datasource_ids_for_chunks",
                return_value={"1": [7], "2": [7], "3": [7]},
            ),
        ):
            for point, expected in [
                (points[0], ["OCR Anfang", "OCR Ende"]),
                (points[2], ["VLM Bildbeschreibung"]),
            ]:
                payload = dict(point.payload)
                chunk = {"text": payload.pop("_text"), "metadata": payload}
                reference = references.annotate([chunk])[0]["metadata"]["document_ref"]
                result = asyncio.run(
                    ReadDocumentTool(retriever, {}, references, [7]).execute(
                        {"document_id": reference}
                    )
                )
                assert result.success
                assert [chunk["text"] for chunk in result.data["results"]] == expected
    finally:
        memory.close()


def test_document_position_index_is_idempotent_and_preserves_payload(qdrant_store):
    from contextlib import nullcontext

    from qdrant_client.models import PayloadSchemaType

    client = qdrant_store._mock_client
    schema = {}
    client.get_collection.return_value.payload_schema = schema
    client.create_payload_index.reset_mock()

    def create_index(**kwargs):
        schema[kwargs["field_name"]] = SimpleNamespace(data_type=kwargs["field_schema"])

    client.create_payload_index.side_effect = create_index
    with patch.object(qdrant_store, "_index_contract_guard", return_value=nullcontext()) as guard:
        assert qdrant_store.ensure_document_read_index() is True
        assert qdrant_store.ensure_document_read_index() is False
    client.create_payload_index.assert_called_once_with(
        collection_name=qdrant_store._collection,
        field_name="chunk_index",
        field_schema=PayloadSchemaType.INTEGER,
        wait=True,
    )
    assert all(
        call.kwargs
        == {
            "operation": "vector_payload_migration",
            "write": True,
            "validation_scope": "stored_binding",
        }
        for call in guard.call_args_list
    )
    client.upsert.assert_not_called()
    client.set_payload.assert_not_called()
    client.delete_payload_index.assert_not_called()


def test_document_position_index_rejects_wrong_type(qdrant_store):
    from qdrant_client.models import PayloadSchemaType

    from kiara.exceptions import VectorDBConnectionError

    client = qdrant_store._mock_client
    client.get_collection.return_value.payload_schema = {
        "chunk_index": SimpleNamespace(data_type=PayloadSchemaType.KEYWORD)
    }
    client.create_payload_index.reset_mock()
    with pytest.raises(VectorDBConnectionError, match="Integer-Index"):
        qdrant_store.ensure_document_read_index()
    client.create_payload_index.assert_not_called()
    client.delete_payload_index.assert_not_called()


def test_document_position_index_requires_verified_creation(qdrant_store):
    from kiara.exceptions import VectorDBConnectionError

    with pytest.raises(VectorDBConnectionError, match="nicht bestaetigt"):
        qdrant_store.ensure_document_read_index()
