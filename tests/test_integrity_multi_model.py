"""Tests fuer Multi-Modell Integritaets-Checks und Repair.

Prueft, dass check_orphan_chunks, check_ds_coverage und _run_repair_orphans
alle nicht-leeren Qdrant-Collections iterieren (nicht nur die aktuell konfigurierte).
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_task():
    """Setzt den globalen _task-State vor jedem Test zurueck."""
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        svc._task.update(
            {
                "running": True,
                "action": "repair_orphans",
                "progress": None,
                "result": None,
            }
        )
    yield
    with svc._lock:
        svc._task.update(
            {
                "running": False,
                "action": None,
                "progress": None,
                "result": None,
            }
        )


def _get_result():
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        return dict(svc._task.get("result") or {})


def _is_running():
    from kiara.admin.services import integrity_service as svc

    with svc._lock:
        return svc._task["running"]


def _make_point(chunk_id, source="test.txt"):
    """Erzeugt ein Mock-Point-Objekt fuer Qdrant scroll."""
    p = MagicMock()
    p.id = chunk_id
    p.payload = {"source": source}
    return p


def _make_store_mock(points_by_collection):
    """Erzeugt einen VectorStore-Mock, der pro Collection verschiedene Punkte liefert.

    points_by_collection: dict[model_name] -> list[MockPoint]
    """
    stores = {}

    def _vs_factory(config, embedding_model=None, intent=None):
        model = embedding_model or "default"
        if model not in stores:
            store = MagicMock()
            store._collection = f"kb_{model}"
            store._client = MagicMock()
            pts = points_by_collection.get(model, [])
            store._client.scroll.return_value = (pts, None)

            def _scroll_all(batch_size=1000):
                pts = points_by_collection.get(model, [])
                if pts:
                    yield [(str(p.id), p.payload) for p in pts]

            store.scroll_all = _scroll_all
            store.delete_by_ids = MagicMock()
            stores[model] = store
        return stores[model]

    return _vs_factory, stores


# Patch-Pfade
_P_VS = "kiara.retrieval.vectorstore.VectorStore"
_P_VS_FACTORY_LIST = "kiara.retrieval.vectorstore_factory.list_all_retrieval_collections"
_P_VS_FACTORY_CREATE = "kiara.retrieval.vectorstore_factory.create_retrieval_store"
_P_DB = "kiara.shared.db.db_query"
_P_CONFIG = "kiara.config.load_config"


# ---------------------------------------------------------------------------
# Test 1: check_orphan_chunks — Multi-Collection
# ---------------------------------------------------------------------------


def test_orphan_chunks_multi_collection():
    """2 Collections: Orphans in beiden werden gefunden."""
    coll_a_points = [_make_point("aaa", "/data/a.txt"), _make_point("bbb", "/data/b.txt")]
    coll_b_points = [_make_point("ccc", "/data/c.txt")]

    vs_factory, stores = _make_store_mock(
        {
            "model-a": coll_a_points,
            "model-b": coll_b_points,
        }
    )

    collections = [
        {"name": "kb_model-a", "chunks": 200, "model": "model-a", "kind": "kb"},
        {"name": "kb_model-b", "chunks": 100, "model": "model-b", "kind": "kb"},
    ]

    # "bbb" und "ccc" sind nicht in chunk_datasources → Orphans
    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            if params:
                known = {"aaa"}  # nur "aaa" ist bekannt
                rows = [{"chunk_id": p} for p in params if p in known]
            else:
                rows = []
            result.fetchall.return_value = rows
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_VS_FACTORY_LIST, return_value=collections),
        patch(
            _P_VS_FACTORY_CREATE,
            side_effect=lambda config, kind, model: vs_factory(config, embedding_model=model),
        ),
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
    ):
        from kiara.integrity.checks import check_orphan_chunks

        result = check_orphan_chunks({"vectordb": {}}, sample_size=10000)

    assert result["sampled"] == 3
    assert result["orphans_found"] == 2  # bbb + ccc
    assert result["status"] == "warning"  # >5%
    orphan_ids = {o["chunk_id"] for o in result["sample_orphans"]}
    assert "bbb" in orphan_ids
    assert "ccc" in orphan_ids


# ---------------------------------------------------------------------------
# Test 2: check_orphan_chunks — leere Collections uebersprungen
# ---------------------------------------------------------------------------


def test_orphan_chunks_empty_collections_skipped():
    """Leere Collections werden uebersprungen (kein Scroll-Aufruf).

    check_orphan_chunks nutzt heute vectorstore_factory.list_all_retrieval_collections
    und vectorstore_factory.create_retrieval_store, nicht mehr direkt
    VectorStore.list_collections. Der Test muss die Factory patchen, sonst
    faellt die Live-Qdrant-Instanz durch und scrollt echte Daten.
    """
    collections = [
        {"kind": "kb", "name": "kb_model-a", "chunks": 0, "model": "model-a"},
        {"kind": "kb", "name": "kb_model-b", "chunks": 0, "model": "model-b"},
    ]

    with (
        patch(_P_VS_FACTORY_LIST, return_value=collections),
        patch(_P_VS_FACTORY_CREATE) as mock_create,
    ):
        from kiara.integrity.checks import check_orphan_chunks

        result = check_orphan_chunks({"vectordb": {}})

    assert result["sampled"] == 0
    assert result["orphans_found"] == 0
    assert result["status"] == "ok"
    # Kein create_retrieval_store-Aufruf, wenn alle Collections leer sind
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: check_orphan_chunks — proportionales Sampling
# ---------------------------------------------------------------------------


def test_orphan_chunks_proportional_sampling():
    """Bei 90%/10% Chunk-Verteilung wird Budget proportional verteilt."""
    # 9000 + 1000 Chunks → Budget 9000 und max(100, 1000)
    large_points = [_make_point(f"l{i}") for i in range(500)]
    small_points = [_make_point(f"s{i}") for i in range(50)]

    vs_factory, stores = _make_store_mock(
        {
            "model-large": large_points,
            "model-small": small_points,
        }
    )

    collections = [
        {"name": "kb_model-large", "chunks": 9000, "model": "model-large", "kind": "kb"},
        {"name": "kb_model-small", "chunks": 1000, "model": "model-small", "kind": "kb"},
    ]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()
        result = MagicMock()

        # Alle IDs als "bekannt" zurueckgeben (keine Orphans)
        def _execute(sql, params=None):
            result = MagicMock()
            rows = [{"chunk_id": p} for p in params] if params else []
            result.fetchall.return_value = rows
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_VS_FACTORY_LIST, return_value=collections),
        patch(
            _P_VS_FACTORY_CREATE,
            side_effect=lambda config, kind, model: vs_factory(config, embedding_model=model),
        ),
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
    ):
        from kiara.integrity.checks import check_orphan_chunks

        result = check_orphan_chunks({"vectordb": {}}, sample_size=10000)

    # Beide Collections wurden gescrollt
    assert "model-large" in stores
    assert "model-small" in stores
    # Budget fuer model-large: int(10000 * 9000/10000) = 9000
    # Budget fuer model-small: max(100, int(10000 * 1000/10000)) = 1000
    # Tatsaechlich gesammelt: 500 + 50 = 550 (da nur so viele Punkte existieren)
    assert result["sampled"] == 550
    assert result["orphans_found"] == 0


# ---------------------------------------------------------------------------
# Test 4: check_ds_coverage — pro Modell
# ---------------------------------------------------------------------------


def test_ds_coverage_per_model():
    """DS mit 2 Modellen: Inkonsistenz nur bei einem Modell → Warning pro Modell-Zeile."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                # datasources
                result.fetchall.return_value = [{"id": 1, "name": "TestDS"}]
            elif n == 2:
                # chunk_datasources: Modell A hat 100, Modell B hat 50
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "model-a", "cnt": 100},
                    {"datasource_id": 1, "embedding_model": "model-b", "cnt": 50},
                ]
            elif n == 3:
                # datasource_embeddings: Modell A erwartet 100, Modell B erwartet 80
                result.fetchall.return_value = [
                    {
                        "datasource_id": 1,
                        "embedding_model": "model-a",
                        "chunk_count": 100,
                        "file_count": 10,
                    },
                    {
                        "datasource_id": 1,
                        "embedding_model": "model-b",
                        "chunk_count": 80,
                        "file_count": 10,
                    },
                ]
            elif n == 4:
                # file_hashes
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "model-a", "cnt": 10},
                    {"datasource_id": 1, "embedding_model": "model-b", "cnt": 10},
                ]
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
        patch("kiara.integrity.checks.is_configured", return_value=False, create=True),
    ):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage({"vectordb": {}})

    ds_list = result["datasources"]
    assert len(ds_list) == 2

    # Modell A: 100/100 → OK
    ds_a = next(d for d in ds_list if d["embedding_model"] == "model-a")
    assert ds_a["status"] == "ok"
    assert ds_a["chunk_ds_count"] == 100

    # Modell B: 50/80 → Warning (Differenz -30)
    ds_b = next(d for d in ds_list if d["embedding_model"] == "model-b")
    assert ds_b["status"] == "warning"
    assert ds_b["chunk_ds_count"] == 50
    assert ds_b["de_chunk_count"] == 80
    assert len(ds_b["issues"]) > 0

    assert result["ds_warnings"] == 1
    assert result["status"] == "warning"


# ---------------------------------------------------------------------------
# Test 5: check_ds_coverage — leere Kombinationen uebersprungen
# ---------------------------------------------------------------------------


def test_ds_coverage_skip_empty_combinations():
    """DS+Modell-Kombination mit 0/0/0 wird nicht aufgefuehrt."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                # 2 Datasources
                result.fetchall.return_value = [
                    {"id": 1, "name": "DS-A"},
                    {"id": 2, "name": "DS-B"},
                ]
            elif n == 2:
                # chunk_datasources: nur DS 1 + model-x
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "model-x", "cnt": 50},
                ]
            elif n == 3:
                # datasource_embeddings: nur DS 1 + model-x
                result.fetchall.return_value = [
                    {
                        "datasource_id": 1,
                        "embedding_model": "model-x",
                        "chunk_count": 50,
                        "file_count": 5,
                    },
                ]
            elif n == 4:
                # file_hashes: nur DS 1 + model-x
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "model-x", "cnt": 5},
                ]
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch(_P_DB, side_effect=lambda: _db_query_ctx()):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage({"vectordb": {}})

    # DS-B hat keine Daten in keiner Schicht → nicht aufgefuehrt
    assert len(result["datasources"]) == 1
    assert result["datasources"][0]["id"] == 1
    assert result["datasources"][0]["embedding_model"] == "model-x"


# ---------------------------------------------------------------------------
# Test 6: _run_repair_orphans — Multi-Collection
# ---------------------------------------------------------------------------


def test_repair_orphans_multi_collection():
    """Repair loescht Orphans in beiden Collections, orphan_ids pro Collection zurueckgesetzt."""
    # Collection A: chunk "a1" bekannt, "a2" orphan
    # Collection B: chunk "b1" orphan
    points_a = [("a1", {"source": "a1.txt"}), ("a2", {"source": "a2.txt"})]
    points_b = [("b1", {"source": "b1.txt"})]

    known_chunks = {"a1"}  # nur a1 ist in chunk_datasources

    def _make_store(model_name, points):
        store = MagicMock()
        store._collection = f"kb_{model_name}"

        def _scroll_all(batch_size=1000):
            if points:
                yield points

        store.scroll_all = _scroll_all
        store.delete_by_ids = MagicMock()
        return store

    store_a = _make_store("model-a", points_a)
    store_b = _make_store("model-b", points_b)

    store_map = {"model-a": store_a, "model-b": store_b}

    def _vs_factory(config, embedding_model=None, intent=None):
        return store_map[embedding_model]

    collections = [
        {"name": "kb_model-a", "chunks": 2, "model": "model-a"},
        {"name": "kb_model-b", "chunks": 1, "model": "model-b"},
    ]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            result = MagicMock()
            rows = [{"chunk_id": p} for p in params if p in known_chunks] if params else []
            result.fetchall.return_value = rows
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_CONFIG, return_value={"vectordb": {}}),
        patch(_P_VS) as mock_vs,
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
    ):
        mock_vs.list_collections.return_value = collections
        mock_vs.side_effect = _vs_factory

        from kiara.admin.services.integrity_service import _run_repair_orphans

        _run_repair_orphans()

    result = _get_result()
    assert result["status"] == "success"
    assert result["scanned"] == 3
    assert result["deleted"] == 2  # a2 + b1

    # Collection A: nur "a2" geloescht (nicht "b1"!)
    store_a.delete_by_ids.assert_called_once_with(["a2"])
    # Collection B: nur "b1" geloescht (nicht "a2"!)
    store_b.delete_by_ids.assert_called_once_with(["b1"])

    assert not _is_running()


# ---------------------------------------------------------------------------
# Test 7: check_layer_counts — Retrieval-Union (kb + colbert)
# ---------------------------------------------------------------------------

_P_LIST_ALL = "kiara.retrieval.vectorstore_factory.list_all_retrieval_collections"


def test_layer_counts_retrieval_union():
    """check_layer_counts zaehlt kb_* + colbert_* Collections und trennt dense/colbert totals."""
    collections = [
        {"kind": "kb", "name": "kb_nomic", "model": "nomic", "chunks": 100},
        {"kind": "colbert", "name": "colbert_colbert-xm", "model": "colbert-xm", "chunks": 30},
    ]

    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                # chunk_datasources (mit Dense/ColBERT-Split + Keyspace-Distinct)
                result.fetchone.return_value = {
                    "distinct_chunks": 120,
                    "chunk_ds_keyspace_distinct_chunks": 130,
                    "total_rows": 140,
                    "dense_chunk_ds_distinct_chunks": 100,
                    "dense_chunk_ds_total_rows": 110,
                    "colbert_chunk_ds_distinct_chunks": 30,
                    "colbert_chunk_ds_total_rows": 30,
                }
            elif n == 2:
                # file_hashes
                result.fetchone.return_value = {"cnt": 10}
            elif n == 3:
                # datasource_embeddings (mit Dense/ColBERT-Split)
                result.fetchone.return_value = {
                    "total": 130,
                    "dense_total": 100,
                    "colbert_total": 30,
                }
            else:
                result.fetchone.return_value = {}
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_LIST_ALL, return_value=collections),
        patch("qdrant_client.QdrantClient") as mock_qc,
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
    ):
        mock_qc.return_value.get_collections.return_value = MagicMock()
        from kiara.integrity.checks import check_layer_counts

        result = check_layer_counts({"vectordb": {}})

    assert result["qdrant_total"] == 130
    assert result["qdrant_dense_total"] == 100
    assert result["qdrant_colbert_total"] == 30
    assert len(result["collections"]) == 2
    # Neue Felder muessen im Ergebnis vorhanden sein
    assert result["chunk_ds_keyspace_distinct_chunks"] == 130
    assert result["dense_chunk_ds_total_rows"] == 110
    assert result["colbert_chunk_ds_total_rows"] == 30
    assert result["dense_ds_embeddings_total"] == 100
    assert result["colbert_ds_embeddings_total"] == 30


def test_layer_counts_es_compares_physical_pg_copies():
    """Regular multiple ownership is not an ES/Qdrant discrepancy."""
    collections = [
        {"kind": "kb", "name": "kb_nomic", "model": "nomic", "chunks": 100},
        {"kind": "colbert", "name": "colbert_xm", "model": "colbert-xm", "chunks": 50},
    ]

    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                result.fetchone.return_value = {
                    "distinct_chunks": 150,
                    "chunk_ds_keyspace_distinct_chunks": 150,
                    "total_rows": 150,
                    "dense_chunk_ds_distinct_chunks": 100,
                    "dense_chunk_ds_total_rows": 100,
                    "colbert_chunk_ds_distinct_chunks": 50,
                    "colbert_chunk_ds_total_rows": 50,
                }
            elif n == 2:
                result.fetchone.return_value = {"cnt": 10}
            elif n == 3:
                result.fetchone.return_value = {
                    "total": 100,
                    "dense_total": 100,
                    "colbert_total": 0,
                }
            else:
                result.fetchone.return_value = {}
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_LIST_ALL, return_value=collections),
        patch("qdrant_client.QdrantClient") as mock_qc,
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
        patch("kiara.integrity.checks.is_configured", return_value=True, create=True),
        patch(
            "kiara.retrieval.elasticsearch_service.is_configured", return_value=True, create=True
        ),
        patch("kiara.retrieval.elasticsearch_service._get_client") as mock_es_client,
        patch(
            "kiara.retrieval.elasticsearch_service.get_status",
            return_value={"doc_count": 140, "counts_match": True, "error": None},
        ),
    ):
        mock_qc.return_value.get_collections.return_value = MagicMock()
        mock_es = MagicMock()
        mock_es.indices.exists.return_value = True
        mock_es.indices.stats.return_value = {"_all": {"primaries": {"docs": {"count": 100}}}}
        mock_es_client.return_value = mock_es
        from kiara.integrity.checks import check_layer_counts

        result = check_layer_counts({"vectordb": {}})

    # 140 physical ES copies may represent 100 unique dense chunks.
    assert "ES" not in result["details"]
    assert result["qdrant_dense_total"] == 100


# ---------------------------------------------------------------------------
# Test 7b: Keyspace-Distinct Regression — kein Warning wenn Keyspace stimmt
# ---------------------------------------------------------------------------


def test_layer_counts_keyspace_no_warning_when_total_rows_differ():
    """qdrant_total == chunk_ds_keyspace_distinct_chunks → kein Warning,
    auch wenn qdrant_total != chunk_ds_total_rows (Zuordnungen)."""
    collections = [
        {"kind": "kb", "name": "kb_nomic", "model": "nomic", "chunks": 100},
        {"kind": "colbert", "name": "colbert_xm", "model": "colbert-xm", "chunks": 50},
    ]
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                # Keyspace = 150 (== qdrant_total), aber total_rows = 200
                result.fetchone.return_value = {
                    "distinct_chunks": 120,
                    "chunk_ds_keyspace_distinct_chunks": 150,
                    "total_rows": 200,
                    "dense_chunk_ds_distinct_chunks": 100,
                    "dense_chunk_ds_total_rows": 150,
                    "colbert_chunk_ds_distinct_chunks": 50,
                    "colbert_chunk_ds_total_rows": 50,
                }
            elif n == 2:
                result.fetchone.return_value = {"cnt": 80}
            elif n == 3:
                result.fetchone.return_value = {
                    "total": 150,
                    "dense_total": 100,
                    "colbert_total": 50,
                }
            else:
                result.fetchone.return_value = {}
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with (
        patch(_P_LIST_ALL, return_value=collections),
        patch("qdrant_client.QdrantClient") as mock_qc,
        patch(_P_DB, side_effect=lambda: _db_query_ctx()),
    ):
        mock_qc.return_value.get_collections.return_value = MagicMock()
        from kiara.integrity.checks import check_layer_counts

        result = check_layer_counts({"vectordb": {}})

    # Kein Warning — Keyspace passt
    assert result["status"] == "ok"
    assert result["chunk_ds_keyspace_distinct_chunks"] == 150
    assert result["chunk_ds_total_rows"] == 200  # Unterschiedlich, aber kein Problem


# ---------------------------------------------------------------------------
# Test 8: check_ds_coverage — ColBERT kind-Rows
# ---------------------------------------------------------------------------


def test_ds_coverage_colbert_kind_rows():
    """ColBERT-Zeilen haben kind='colbert' und de_chunk_count/fh_count = None."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                result.fetchall.return_value = [{"id": 1, "name": "TestDS"}]
            elif n == 2:
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "nomic", "cnt": 100},
                    {"datasource_id": 1, "embedding_model": "colbert::colbert-xm", "cnt": 20},
                ]
            elif n == 3:
                result.fetchall.return_value = [
                    {
                        "datasource_id": 1,
                        "embedding_model": "nomic",
                        "chunk_count": 100,
                        "file_count": 10,
                    },
                ]
            elif n == 4:
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "nomic", "cnt": 10},
                ]
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    with patch(_P_DB, side_effect=lambda: _db_query_ctx()):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage({"vectordb": {}})

    ds_list = result["datasources"]
    assert len(ds_list) == 2

    kb_row = next(d for d in ds_list if d["kind"] == "kb")
    assert kb_row["de_chunk_count"] == 100
    assert kb_row["fh_count"] == 10

    colbert_row = next(d for d in ds_list if d["kind"] == "colbert")
    assert colbert_row["de_chunk_count"] is None
    assert colbert_row["fh_count"] is None
    assert colbert_row["chunk_ds_count"] == 20
    assert colbert_row["status"] == "ok"


# ---------------------------------------------------------------------------
# Test 9: check_ds_coverage — colbert_warnings
# ---------------------------------------------------------------------------


def test_ds_coverage_colbert_warnings():
    """DS mit colbert_enabled=1 + Dense-Daten aber ohne colbert::* → colbert_warning."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                # datasources (active)
                result.fetchall.return_value = [{"id": 1, "name": "TestDS"}]
            elif n == 2:
                # chunk_datasources: nur Dense
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "nomic", "cnt": 50},
                ]
            elif n == 3:
                # datasource_embeddings
                result.fetchall.return_value = [
                    {
                        "datasource_id": 1,
                        "embedding_model": "nomic",
                        "chunk_count": 50,
                        "file_count": 5,
                    },
                ]
            elif n == 4:
                # file_hashes
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "nomic", "cnt": 5},
                ]
            elif n == 5:
                # colbert DS query (ds_colbert_enabled=True)
                result.fetchall.return_value = [{"id": 1, "name": "TestDS"}]
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    config = {"vectordb": {}, "retrieval": {"colbert": {"enabled": True}}}

    with patch(_P_DB, side_effect=lambda: _db_query_ctx()):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage(config)

    assert result["colbert_warning_count"] == 1
    assert result["colbert_warnings"][0]["id"] == 1
    assert result["ds_warnings"] == 0
    assert result["status"] == "warning"


# ---------------------------------------------------------------------------
# Test 10: check_ds_coverage — colbert_warnings not triggered for empty DS
# ---------------------------------------------------------------------------


def test_ds_coverage_colbert_no_warning_for_empty_ds():
    """Leere / noch nie ingestete DS erzeugen kein ColBERT-Backfill-Warning."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                result.fetchall.return_value = [{"id": 1, "name": "EmptyDS"}]
            elif n == 2:
                # Keine chunk_datasources
                result.fetchall.return_value = []
            elif n == 3 or n == 4:
                result.fetchall.return_value = []
            elif n == 5:
                # DS hat colbert_enabled=1
                result.fetchall.return_value = [{"id": 1, "name": "EmptyDS"}]
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    config = {"vectordb": {}, "retrieval": {"colbert": {"enabled": True}}}

    with patch(_P_DB, side_effect=lambda: _db_query_ctx()):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage(config)

    # Kein Warning weil DS keine Dense-Daten hat
    assert result["colbert_warning_count"] == 0
    assert result["status"] == "ok"


def test_ds_coverage_warns_for_chunk_datasources_without_embedding_stats():
    """DS mit chunk_datasources, aber fehlender datasource_embeddings-Zeile ist inkonsistent."""
    call_count = [0]

    def _db_query_ctx():
        ctx = MagicMock()
        conn = MagicMock()

        def _execute(sql, params=None):
            call_count[0] += 1
            result = MagicMock()
            n = call_count[0]
            if n == 1:
                result.fetchall.return_value = [{"id": 1, "name": "BrokenDS"}]
            elif n == 2:
                result.fetchall.return_value = [
                    {"datasource_id": 1, "embedding_model": "nomic", "cnt": 100},
                ]
            elif n in (3, 4):
                result.fetchall.return_value = []
            else:
                result.fetchall.return_value = []
            return result

        conn.execute = _execute
        ctx.__enter__ = lambda s: conn
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    config = {"vectordb": {}, "retrieval": {"colbert": {"enabled": False}}}

    with patch(_P_DB, side_effect=lambda: _db_query_ctx()):
        from kiara.integrity.checks import check_ds_coverage

        result = check_ds_coverage(config)

    assert result["status"] == "warning"
    assert result["ds_warnings"] == 1
    row = result["datasources"][0]
    assert row["status"] == "warning"
    assert row["chunk_ds_count"] == 100
    assert row["de_chunk_count"] == 0
    assert "datasource_embeddings (0)" in row["issues"][0]
