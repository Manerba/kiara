"""Tests fuer Sprint 08: VectorDB Admin-Funktionen.

Testet vectordb.files, .delete_files — 4 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# vectordb.files
# ---------------------------------------------------------------------------


class TestVectorDBFiles:
    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_vectordb_files_list(self, mock_get, mock_config, mock_vs_cls):
        """Test 46: Dateiliste mit Chunk-Counts."""
        mock_get.return_value = {"id": 3, "name": "TestDS"}
        mock_config.return_value = {}
        mock_vs = MagicMock()
        mock_vs.list_sources_for_datasource.return_value = [
            {"source": "/tmp/file1.pdf", "chunk_count": 10},
            {"source": "/tmp/file2.docx", "chunk_count": 5},
        ]
        mock_vs_cls.return_value = mock_vs

        from kiara.admin.functions.vectordb import vectordb_files

        result = _run(vectordb_files(MOCK_USER, ds_id=3, model="nomic-embed-text"))

        assert result.success is True
        assert len(result.data["files"]) == 2
        assert result.data["total_chunks"] == 15
        assert result.data["ds_id"] == 3

    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_vectordb_files_unknown_ds(self, mock_get):
        """Test 47: Unbekannte DS → Fehler."""
        mock_get.return_value = None

        from kiara.admin.functions.vectordb import vectordb_files

        result = _run(vectordb_files(MOCK_USER, ds_id=999, model="nomic-embed-text"))

        assert result.success is False
        assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# vectordb.delete_files
# ---------------------------------------------------------------------------


class TestVectorDBDeleteFiles:
    @patch("kiara.ingestion._file_utils.HashStore")
    @patch("kiara.ingestion._file_utils.ChunkDatasources")
    @patch("kiara.ingestion.vectordb_ops.delete_files")
    @patch("kiara.config.load_config")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_by_id")
    def test_vectordb_delete_files(
        self, mock_get, mock_config, mock_delete, mock_chunk_ds, mock_hash_store
    ):
        """Test 48: Loeschen mit Confirmation."""
        mock_get.return_value = {"id": 3, "name": "TestDS"}
        mock_config.return_value = {"vectordb": {"path": "/tmp/vdb"}}
        mock_delete.return_value = {"deleted_chunks": 15}

        from kiara.admin.functions.vectordb import vectordb_delete_files

        result = _run(
            vectordb_delete_files(
                MOCK_USER,
                ds_id=3,
                embedding_model="nomic-embed-text",
                file_paths=["/tmp/file1.pdf"],
            )
        )

        assert result.success is True
        assert "gelöscht" in result.message
        mock_delete.assert_called_once()

    def test_vectordb_delete_empty_list(self):
        """Test 49: Leere file_paths → Fehler."""
        from kiara.admin.functions.vectordb import vectordb_delete_files

        result = _run(
            vectordb_delete_files(
                MOCK_USER, ds_id=3, embedding_model="nomic-embed-text", file_paths=[]
            )
        )

        assert result.success is False
        assert "Keine Dateien" in result.message


# ---------------------------------------------------------------------------
# vectordb.delete_collection — Graph-Cleanup-Pfad (Issue #1046)
# ---------------------------------------------------------------------------


class TestVectorDBDeleteCollectionGraphCleanup:
    """Tests fuer vectordb_delete_collection Graph-Cleanup:
    Kandidaten-Snapshot vor DB-Delete + filter_unreferenced + GraphStore.delete
    in derselben DB-Transaktion + Rollback bei Graph-Cleanup-Fehler.
    """

    def _build_db_mock(self, graph_candidates, db_remains=3, active_job=None, affected_ds=1):
        """DB-Mock mit sequentiellen execute()-Antworten fuer den Delete-Pfad."""
        db = MagicMock()

        call_log = []

        def _execute(sql, params=None):
            call_log.append((sql, params))
            cursor = MagicMock()
            s = " ".join(sql.split()).lower()  # normalize
            if "db_remains" in s or "(select count(*) from datasource_embeddings" in s.replace(
                "\n", " "
            ):
                cursor.fetchone.return_value = {"cnt": db_remains}
            elif "from ingestion_jobs" in s:
                cursor.fetchone.return_value = active_job
            elif "count(distinct datasource_id)" in s:
                cursor.fetchone.return_value = {"cnt": affected_ds}
            elif "select distinct chunk_id from chunk_datasources" in s:
                cursor.fetchall.return_value = [{"chunk_id": cid} for cid in graph_candidates]
            elif "delete from" in s:
                cursor.rowcount = len(graph_candidates) if "chunk_datasources" in s else 1
            else:
                cursor.fetchone.return_value = {"cnt": 0}
            return cursor

        db.execute = _execute
        return db, call_log

    def _patch_stack(
        self,
        *,
        db,
        config,
        qdrant_collections,
        graph_enabled=True,
        filter_unref_return=None,
        graph_delete_side_effect=None,
    ):
        """Gemeinsamer Patch-Stack fuer delete_collection-Tests."""
        from contextlib import ExitStack, contextmanager

        @contextmanager
        def _cm():
            yield db

        stack = ExitStack()
        stack.enter_context(patch("kiara.config.load_config", return_value=config))
        stack.enter_context(
            patch(
                "kiara.shared.db.db_query",
                _cm,
            )
        )
        stack.enter_context(
            patch(
                "kiara.shared.db.db_transaction",
                _cm,
            )
        )
        stack.enter_context(patch("kiara.ingestion.vectordb_ops.invalidate_bm25"))
        stack.enter_context(
            patch(
                "kiara.web.cache_refresh.refresh_datasource_cache",
                create=True,
            )
        )
        stack.enter_context(
            patch(
                "kiara.web.cache_refresh.refresh_vectordb_cache",
                create=True,
            )
        )
        qc = stack.enter_context(patch("qdrant_client.QdrantClient"))
        client = qc.return_value
        collections_mock = MagicMock()
        collections_mock.collections = [MagicMock(name=c) for c in qdrant_collections]
        # Set the .name attribute explicitly (MagicMock(name=...) uses `name` as mock-name)
        for m, n in zip(collections_mock.collections, qdrant_collections, strict=False):
            m.name = n
        client.get_collections.return_value = collections_mock
        info = MagicMock()
        info.points_count = 5
        client.get_collection.return_value = info
        client.delete_collection.return_value = True

        # ChunkDatasources(): acquire_graph_cleanup_lock + filter_unreferenced_chunk_ids
        cd_cls = stack.enter_context(
            patch(
                "kiara.ingestion._file_utils.ChunkDatasources",
            )
        )
        cd = cd_cls.return_value
        cd.acquire_graph_cleanup_lock = MagicMock()
        cd.filter_unreferenced_chunk_ids = MagicMock(
            return_value=filter_unref_return if filter_unref_return is not None else [],
        )

        # GraphStore: delete_by_chunk_ids
        gs_cls = stack.enter_context(patch("kiara.graph.store.GraphStore"))
        gs = gs_cls.return_value
        if graph_delete_side_effect is not None:
            gs.delete_by_chunk_ids = MagicMock(side_effect=graph_delete_side_effect)
        else:
            gs.delete_by_chunk_ids = MagicMock()

        return stack, cd, gs, client

    def test_delete_collection_graph_cleanup_order_and_conn(self):
        """#1046: Kandidaten-Snapshot VOR DELETE, filter_unref + GraphStore unter db.
        Beide Graph-Helper bekommen conn=db (selbe Transaktion).
        """
        config = {"vectordb": {"qdrant_url": "http://x"}, "graph": {"enabled": True}}
        graph_candidates = ["c1", "c2", "c3"]
        db, call_log = self._build_db_mock(graph_candidates=graph_candidates)

        stack, cd, gs, client = self._patch_stack(
            db=db,
            config=config,
            qdrant_collections=["kb_test"],
            filter_unref_return=["c1", "c2"],  # c3 referenziert noch anderswo
        )
        with stack:
            # collection_name_for_model -> erwartet "kb_test" fuer Modell "test"
            with patch(
                "kiara.retrieval.embedding.collection_name_for_model", return_value="kb_test"
            ):
                from kiara.admin.functions.vectordb import vectordb_delete_collection

                result = _run(vectordb_delete_collection(MOCK_USER, embedding_model="test"))

        assert result.success, result.message
        # Snapshot wurde gemacht
        cd.acquire_graph_cleanup_lock.assert_called_once_with(db)
        cd.filter_unreferenced_chunk_ids.assert_called_once()
        passed_ids, passed_kwargs = cd.filter_unreferenced_chunk_ids.call_args
        assert set(passed_ids[0]) == set(graph_candidates)
        assert passed_kwargs.get("conn") is db, "filter muss conn=db bekommen"
        # GraphStore.delete mit gefilterten IDs, conn=db
        gs.delete_by_chunk_ids.assert_called_once_with(["c1", "c2"], conn=db)

    def test_delete_collection_graph_cleanup_ignores_disabled_flag(self):
        """Persistenter Graph-Cleanup laeuft auch wenn graph.enabled=False."""
        config = {"vectordb": {"qdrant_url": "http://x"}, "graph": {"enabled": False}}
        db, _ = self._build_db_mock(graph_candidates=["c1"])

        stack, cd, gs, client = self._patch_stack(
            db=db,
            config=config,
            qdrant_collections=["kb_test"],
            graph_enabled=False,
            filter_unref_return=["c1"],
        )
        with (
            stack,
            patch("kiara.retrieval.embedding.collection_name_for_model", return_value="kb_test"),
        ):
            from kiara.admin.functions.vectordb import vectordb_delete_collection

            result = _run(vectordb_delete_collection(MOCK_USER, embedding_model="test"))

        assert result.success
        cd.filter_unreferenced_chunk_ids.assert_called_once()
        gs.delete_by_chunk_ids.assert_called_once()

    def test_delete_collection_empty_snapshot_no_graph_call(self):
        """#1046: Leere Snapshot -> kein filter + kein GraphStore.delete."""
        config = {"vectordb": {"qdrant_url": "http://x"}, "graph": {"enabled": True}}
        db, _ = self._build_db_mock(graph_candidates=[])

        stack, cd, gs, client = self._patch_stack(
            db=db,
            config=config,
            qdrant_collections=["kb_test"],
        )
        with (
            stack,
            patch("kiara.retrieval.embedding.collection_name_for_model", return_value="kb_test"),
        ):
            from kiara.admin.functions.vectordb import vectordb_delete_collection

            result = _run(vectordb_delete_collection(MOCK_USER, embedding_model="test"))

        assert result.success
        gs.delete_by_chunk_ids.assert_not_called()

    def test_delete_collection_graph_error_propagates(self):
        """#1046: GraphStore.delete wirft -> Exception propagiert (db_transaction rollt zurueck)."""
        config = {"vectordb": {"qdrant_url": "http://x"}, "graph": {"enabled": True}}
        db, _ = self._build_db_mock(graph_candidates=["c1"])

        stack, cd, gs, client = self._patch_stack(
            db=db,
            config=config,
            qdrant_collections=["kb_test"],
            filter_unref_return=["c1"],
            graph_delete_side_effect=RuntimeError("graph kaputt"),
        )
        with (
            stack,
            patch("kiara.retrieval.embedding.collection_name_for_model", return_value="kb_test"),
        ):
            from kiara.admin.functions.vectordb import vectordb_delete_collection

            with pytest.raises(RuntimeError, match="graph kaputt"):
                _run(vectordb_delete_collection(MOCK_USER, embedding_model="test"))

        # GraphStore.delete wurde genau einmal versucht
        gs.delete_by_chunk_ids.assert_called_once()
