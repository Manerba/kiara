"""Tests fuer Graph-Cleanup bei DS-Loeschung/Neuindexierung.

Testet:
- GraphStore.delete_by_chunk_ids (Batch-Methode)
- Graph-Cleanup in _cleanup_stale_chunks (Pipeline)
- Graph-Cleanup in delete_datasource_full (DS-Loeschung)
"""

import asyncio
from unittest.mock import ANY, MagicMock, patch

import pytest

pytestmark = pytest.mark.usefixtures("s4_guard_passthrough")

from kiara.graph.store import GraphStore

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test 1 + 2: GraphStore.delete_by_chunk_ids (Batch-Methode)
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_store():
    """Erstellt einen GraphStore (PostgreSQL, via conftest._init_pg_backend)."""
    store = GraphStore()
    store.clear()
    yield store
    store.clear()


def test_delete_by_chunk_ids_batch(graph_store):
    """Batch-Delete: 2 von 3 Chunks loeschen, Orphan-Nodes korrekt behandeln."""
    # Setup: 3 Chunks mit Nodes und Edges
    graph_store.add_node("topic:a", "topic", "Thema A", source="extraction")
    graph_store.add_node("topic:b", "topic", "Thema B", source="extraction")
    graph_store.add_node("topic:shared", "topic", "Gemeinsam", source="extraction")
    graph_store.add_node("customer:erp1", "customer", "ERP Kunde", source="erp_import")

    # chunk_1: topic:a -> topic:shared
    graph_store.add_edge("topic:a", "topic:shared", "referenziert", chunk_id="chunk_1")
    # chunk_2: topic:b -> topic:shared
    graph_store.add_edge("topic:b", "topic:shared", "referenziert", chunk_id="chunk_2")
    # chunk_3: topic:shared -> customer:erp1 (bleibt erhalten)
    graph_store.add_edge("topic:shared", "customer:erp1", "betrifft", chunk_id="chunk_3")

    stats_before = graph_store.get_stats()
    assert stats_before["total_nodes"] == 4
    assert stats_before["total_edges"] == 3

    # Act: chunk_1 und chunk_2 loeschen
    graph_store.delete_by_chunk_ids(["chunk_1", "chunk_2"])

    stats = graph_store.get_stats()
    # Edges: chunk_3 bleibt (1 Edge)
    assert stats["total_edges"] == 1
    # topic:a verwaist (extraction) -> geloescht
    assert len(graph_store.find_nodes("Thema A")) == 0
    # topic:b verwaist (extraction) -> geloescht
    assert len(graph_store.find_nodes("Thema B")) == 0
    # topic:shared hat noch Edge (chunk_3) -> bleibt
    assert len(graph_store.find_nodes("Gemeinsam")) == 1
    # customer:erp1 (erp_import) -> bleibt immer (auch wenn verwaist)
    assert len(graph_store.find_nodes("ERP Kunde")) == 1


def test_delete_by_chunk_ids_empty(graph_store):
    """Leere Liste: keine Exception, keine Aenderung."""
    graph_store.add_node("topic:x", "topic", "X", source="extraction")
    graph_store.add_edge("topic:x", "topic:x", "self", chunk_id="c1")

    graph_store.delete_by_chunk_ids([])

    assert graph_store.get_stats()["total_edges"] == 1
    assert graph_store.get_stats()["total_nodes"] == 1


# ---------------------------------------------------------------------------
# Test 3-5: Graph-Cleanup in _cleanup_stale_chunks (Pipeline)
# ---------------------------------------------------------------------------


def _make_pipeline_mocks(graph_enabled=True):
    """Erstellt Config + Mocks fuer _cleanup_stale_chunks."""
    config = {
        "graph": {"enabled": graph_enabled},
        "embedding_model": "nomic-embed-text",
    }
    store = MagicMock()
    store.get_ids_by_sources.return_value = ["id_1", "id_2"]
    store.delete_by_sources = MagicMock()

    hash_store = MagicMock()
    chunk_ds = MagicMock()
    chunk_ds.get_other_datasources.return_value = []
    chunk_ds.filter_unreferenced_chunk_ids.return_value = ["id_1", "id_2"]

    return config, store, hash_store, chunk_ds


def test_cleanup_stale_chunks_with_graph():
    """Graph-Cleanup wird aufgerufen wenn graph.enabled=True.

    Seit #686 laeuft der Cleanup in derselben Transaktion wie chunk_datasources,
    daher wird ``conn=<db_transaction context>`` als Kwarg uebergeben.
    """
    config, store, hash_store, chunk_ds = _make_pipeline_mocks(graph_enabled=True)

    with (
        patch("kiara.ingestion.pipeline.db_transaction") as mock_tx,
        patch("kiara.ingestion.pipeline.delete_from_elasticsearch", create=True),
        patch("kiara.graph.store.GraphStore.delete_by_chunk_ids") as mock_graph,
    ):
        mock_tx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import _cleanup_stale_chunks

        _cleanup_stale_chunks(
            config,
            store,
            hash_store,
            chunk_ds,
            datasource_id=1,
            embedding_model="nomic-embed-text",
            files_to_delete=["/tmp/test.txt"],
        )

    chunk_ds.filter_unreferenced_chunk_ids.assert_called_once_with(["id_1", "id_2"], conn=ANY)
    mock_graph.assert_called_once_with(["id_1", "id_2"], conn=ANY)


def test_cleanup_stale_chunks_graph_cleanup_ignores_disabled_flag():
    """Persistenter Graph-Cleanup laeuft auch wenn graph.enabled=False."""
    config, store, hash_store, chunk_ds = _make_pipeline_mocks(graph_enabled=False)

    with (
        patch("kiara.ingestion.pipeline.db_transaction") as mock_tx,
        patch("kiara.ingestion.pipeline.delete_from_elasticsearch", create=True),
        patch("kiara.graph.store.GraphStore.delete_by_chunk_ids") as mock_graph,
    ):
        mock_tx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import _cleanup_stale_chunks

        _cleanup_stale_chunks(
            config,
            store,
            hash_store,
            chunk_ds,
            datasource_id=1,
            embedding_model="nomic-embed-text",
            files_to_delete=["/tmp/test.txt"],
        )

    chunk_ds.filter_unreferenced_chunk_ids.assert_called_once_with(["id_1", "id_2"], conn=ANY)
    mock_graph.assert_called_once_with(["id_1", "id_2"], conn=ANY)


def test_cleanup_stale_chunks_graph_failure_propagates():
    """Graph-Cleanup-Fehler ist seit #686 FATAL: Exception propagiert.

    Graph-Cleanup laeuft in derselben Transaktion wie chunk_datasources-Cleanup.
    Ein Fehler rollt die gesamte Transaktion zurueck — keine stale graph_edges
    auf Chunks, die zwar aus chunk_datasources entfernt aber im Graph noch
    referenziert sind.
    """
    config, store, hash_store, chunk_ds = _make_pipeline_mocks(graph_enabled=True)

    with (
        patch("kiara.ingestion.pipeline.db_transaction") as mock_tx,
        patch("kiara.ingestion.pipeline.delete_from_elasticsearch", create=True),
        patch(
            "kiara.graph.store.GraphStore.delete_by_chunk_ids",
            side_effect=RuntimeError("Graph kaputt"),
        ),
    ):
        mock_tx.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import _cleanup_stale_chunks

        with pytest.raises(RuntimeError, match="Graph kaputt"):
            _cleanup_stale_chunks(
                config,
                store,
                hash_store,
                chunk_ds,
                datasource_id=1,
                embedding_model="nomic-embed-text",
                files_to_delete=["/tmp/test.txt"],
            )


def test_stale_colbert_db_cleanup_ignores_disabled_flag():
    """Stale-ColBERT-DB-Cleanup filtert Graph-Chunks all-keyspace."""
    conn = MagicMock()
    delete_cursor = MagicMock(rowcount=1)

    def execute(sql, _params=None):
        if sql.startswith("SELECT chunk_id FROM chunk_datasources"):
            return MagicMock(fetchall=MagicMock(return_value=[{"chunk_id": "c1"}]))
        if sql.startswith("DELETE FROM chunk_datasources"):
            return delete_cursor
        return MagicMock(fetchall=MagicMock(return_value=[]))

    conn.execute.side_effect = execute
    chunk_ds = MagicMock()
    chunk_ds.filter_unreferenced_chunk_ids.return_value = ["c1"]

    with (
        patch("kiara.ingestion.pipeline.db_transaction") as mock_tx,
        patch("kiara.graph.store.GraphStore.delete_by_chunk_ids") as mock_graph,
    ):
        mock_tx.return_value.__enter__ = MagicMock(return_value=conn)
        mock_tx.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import (
            _delete_colbert_datasource_entries_with_graph_cleanup,
        )

        deleted = _delete_colbert_datasource_entries_with_graph_cleanup(
            {"graph": {"enabled": False}},
            chunk_ds,
            1,
            "colbert::old",
        )

    assert deleted == 1
    chunk_ds.acquire_graph_cleanup_lock.assert_called_once_with(conn)
    chunk_ds.delete_datasource_entries.assert_not_called()
    chunk_ds.filter_unreferenced_chunk_ids.assert_called_once_with(["c1"], conn=conn)
    mock_graph.assert_called_once_with(["c1"], conn=conn)


# ---------------------------------------------------------------------------
# Test 6-7: Graph-Cleanup in delete_datasource_full (DS-Loeschung)
# ---------------------------------------------------------------------------


class TestDeleteDatasourceGraphCleanup:
    """Tests fuer Graph-Cleanup in delete_datasource_full()."""

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_datasource_full_graph_cleanup(
        self,
        mock_config,
        mock_get,
        mock_cascade,
        mock_cache,
    ):
        """Graph-Cleanup wird mit gesammelten Orphan-IDs unabh. vom Flag aufgerufen."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
            "graph": {"enabled": False},
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources") as mock_cds,
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch("kiara.graph.store.GraphStore.delete_by_chunk_ids") as mock_graph,
            patch("kiara.retrieval.colbert.ColBERTStore.list_colbert_collections", return_value=[]),
        ):
            # Eine Collection mit 2 Orphan-Chunks
            mock_vs.list_collections.return_value = [
                {"model": "nomic-embed-text", "count": 100},
            ]
            store_instance = MagicMock()
            mock_vs.return_value = store_instance
            store_instance.get_collection_name.return_value = "kb_nomic-embed-text"

            chunk_ds_instance = mock_cds.return_value
            chunk_ds_instance.preview_datasource_removal.return_value = (
                ["orphan_1", "orphan_2"],
                {},
            )
            chunk_ds_instance.preview_graph_orphan_chunk_ids_after_datasource_removal.return_value = [
                "orphan_1",
                "orphan_2",
            ]

            from kiara.admin.services.datasource_service import delete_datasource_full

            result = _run(delete_datasource_full(10))

        assert result["message"] == "Datenquelle geloescht"
        mock_graph.assert_called_once_with(["orphan_1", "orphan_2"], conn=ANY)

    @patch("kiara.web.cache_refresh.delete_datasource_cache")
    @patch("kiara.repositories.datasource.DatasourceRepository.delete_cascade")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    @patch("kiara.admin.services.datasource_service.load_config")
    def test_delete_datasource_full_graph_failure_is_fatal(
        self,
        mock_config,
        mock_get,
        mock_cascade,
        mock_cache,
    ):
        """Graph-Cleanup-Fehler rollt den finalen Cascade-Block zurueck."""
        mock_config.return_value = {
            "vectordb": {"path": "/tmp/vdb"},
            "embedding_model": "nomic-embed-text",
            "graph": {"enabled": True},
        }
        mock_get.return_value = {"id": 10, "name": "DS", "type": "local"}

        with (
            patch("kiara.retrieval.vectorstore.VectorStore") as mock_vs,
            patch("kiara.ingestion._file_utils.ChunkDatasources") as mock_cds,
            patch("kiara.ingestion.vectordb_ops.invalidate_bm25"),
            patch(
                "kiara.graph.store.GraphStore.delete_by_chunk_ids",
                side_effect=RuntimeError("Graph kaputt"),
            ),
            patch("kiara.retrieval.colbert.ColBERTStore.list_colbert_collections", return_value=[]),
        ):
            mock_vs.list_collections.return_value = [
                {"model": "nomic-embed-text", "count": 100},
            ]
            store_instance = MagicMock()
            mock_vs.return_value = store_instance
            store_instance.get_collection_name.return_value = "kb_nomic-embed-text"

            chunk_ds_instance = mock_cds.return_value
            chunk_ds_instance.preview_datasource_removal.return_value = (["orphan_1"], {})
            chunk_ds_instance.preview_graph_orphan_chunk_ids_after_datasource_removal.return_value = [
                "orphan_1",
            ]

            from kiara.admin.services.datasource_service import delete_datasource_full

            with pytest.raises(RuntimeError, match="Graph kaputt"):
                _run(delete_datasource_full(10))

        mock_cascade.assert_not_called()
