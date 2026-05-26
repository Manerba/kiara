"""Tests fuer Full-Reingest ES-Cleanup in ingestion_worker.py.

Prueft, dass _prepare_full_reingest() die Preview-First-/Dense-last-
Reihenfolge einhaelt:
- preview_datasource_removal() fuer Dense und DS-relevante ColBERT-Keys
- Qdrant-Cleanup (Orphans + Non-Orphan-ds_ids) vor DB-/Hash-Mutation
- VectorDBConnectionError aus update_ds_ids() bricht VOR DB-/Hash-Loeschung ab
- ES-Loeschung fuer alte Dateipfade
- BM25-Invalidation NACH Dense-Qdrant-Cleanup
- EIN gemeinsamer DB-/Hash-Commit-Punkt fuer chunk_datasources,
  file_hashes, hash_mode_meta und datasource_embeddings
"""

from unittest.mock import MagicMock, patch

import pytest


class TestFullReingestEsCleanup:
    """_prepare_full_reingest() ruft delete_from_elasticsearch() fuer alte Dateipfade."""

    @patch("kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds", return_value=[])
    @patch("kiara.ingestion.vectordb_ops.invalidate_bm25")
    @patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch")
    @patch("kiara.ingestion._file_utils.ChunkDatasources")
    @patch("kiara.ingestion._file_utils.HashStore")
    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.web.ingestion_worker.IngestionWorker._apply_full_reingest_db_cleanup")
    def test_es_delete_called_with_old_paths(
        self,
        mock_db_cleanup,
        mock_vs_cls,
        mock_hs_cls,
        mock_cds_cls,
        mock_es_delete,
        mock_bm25,
        mock_resolve_colbert,
    ):
        """ES-Delete wird mit alten Dateipfaden und datasource_id aufgerufen."""
        from kiara.web.ingestion_worker import IngestionWorker

        config = {"retrieval": {"hybrid_search": "elasticsearch"}}
        ds_id = 42
        embedding_model = "model1"

        hash_store = MagicMock()
        hash_store.load.return_value = {
            "/data/file1.txt": "hash1",
            "/data/file2.txt": "hash2",
        }
        mock_hs_cls.return_value = hash_store

        chunk_ds = MagicMock()
        chunk_ds.preview_datasource_removal.return_value = (
            ["c1"],  # orphan_ids
            {"c2": [1], "c3": [1, 2]},  # non_orphan_ds_ids_map
        )
        mock_cds_cls.return_value = chunk_ds

        store = MagicMock()
        store.get_collection_name.return_value = "kb_test"
        mock_vs_cls.return_value = store

        worker = IngestionWorker.__new__(IngestionWorker)
        worker._prepare_full_reingest("job-1", ds_id, config, embedding_model)

        # Preview wurde aufgerufen
        chunk_ds.preview_datasource_removal.assert_called_with(ds_id, embedding_model)

        # Orphans in Qdrant geloescht
        store.delete_by_ids.assert_called_once_with(["c1"])

        # Non-orphans: ds_ids aktualisiert
        store.update_ds_ids.assert_called_once_with({"c2": [1], "c3": [1, 2]})

        # ES-Delete mit alten Pfaden + ds_id
        mock_es_delete.assert_called_once()
        args, kwargs = mock_es_delete.call_args
        assert args[0] == config
        assert set(args[1]) == {"/data/file1.txt", "/data/file2.txt"}
        assert kwargs.get("datasource_id") == ds_id

        # BM25 nach Dense-Cleanup invalidiert
        mock_bm25.assert_called_once_with("kb_test")

        # Gemeinsamer DB-/Hash-Commit-Punkt: all-keyspace Graph-Candidates
        # (Orphans + Non-Orphans aus Dense+ColBERT). Non-Orphans muessen mit,
        # damit filter_unreferenced_chunk_ids spaeter korrekt filtern kann —
        # Chunks die in Dense+ColBERT fuer dieselbe DS non-orphan sind wuerden
        # sonst den Graph-Cleanup verpassen (ImplScan V28, Finding 5).
        mock_db_cleanup.assert_called_once_with(
            ds_id,
            embedding_model,
            [],
            ["c1", "c2", "c3"],
        )

    @patch("kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds", return_value=[])
    @patch("kiara.ingestion.vectordb_ops.invalidate_bm25")
    @patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch")
    @patch("kiara.ingestion._file_utils.ChunkDatasources")
    @patch("kiara.ingestion._file_utils.HashStore")
    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.web.ingestion_worker.IngestionWorker._apply_full_reingest_db_cleanup")
    def test_no_es_delete_when_no_old_paths(
        self,
        mock_db_cleanup,
        mock_vs_cls,
        mock_hs_cls,
        mock_cds_cls,
        mock_es_delete,
        mock_bm25,
        mock_resolve_colbert,
    ):
        """Keine alten Dateipfade -> kein ES-Delete-Call."""
        from kiara.web.ingestion_worker import IngestionWorker

        hash_store = MagicMock()
        hash_store.load.return_value = {}  # Keine alten Hashes
        mock_hs_cls.return_value = hash_store

        chunk_ds = MagicMock()
        chunk_ds.preview_datasource_removal.return_value = ([], {})
        mock_cds_cls.return_value = chunk_ds

        store = MagicMock()
        store.get_collection_name.return_value = "kb_empty"
        mock_vs_cls.return_value = store

        worker = IngestionWorker.__new__(IngestionWorker)
        worker._prepare_full_reingest("job-1", 1, {}, "model1")

        mock_es_delete.assert_not_called()
        mock_db_cleanup.assert_called_once_with(1, "model1", [], [])


class TestFullReingestDenseLastRetryAnchor:
    """V35 #734 — VectorDBConnectionError bricht VOR DB-/Hash-Loeschung ab."""

    @patch("kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds", return_value=[])
    @patch("kiara.ingestion.vectordb_ops.delete_from_elasticsearch")
    @patch("kiara.ingestion._file_utils.ChunkDatasources")
    @patch("kiara.ingestion._file_utils.HashStore")
    @patch("kiara.retrieval.vectorstore.VectorStore")
    @patch("kiara.web.ingestion_worker.IngestionWorker._apply_full_reingest_db_cleanup")
    def test_vectordb_error_preserves_db_and_hash(
        self,
        mock_db_cleanup,
        mock_vs_cls,
        mock_hs_cls,
        mock_cds_cls,
        mock_es_delete,
        mock_resolve_colbert,
    ):
        from kiara.exceptions import VectorDBConnectionError
        from kiara.web.ingestion_worker import IngestionWorker

        hash_store = MagicMock()
        hash_store.load.return_value = {"/a.txt": "h1"}
        mock_hs_cls.return_value = hash_store

        chunk_ds = MagicMock()
        chunk_ds.preview_datasource_removal.return_value = (
            ["c1"],
            {"c2": [2]},
        )
        mock_cds_cls.return_value = chunk_ds

        store = MagicMock()
        store.update_ds_ids.side_effect = VectorDBConnectionError("qdrant down")
        mock_vs_cls.return_value = store

        worker = IngestionWorker.__new__(IngestionWorker)
        with pytest.raises(VectorDBConnectionError):
            worker._prepare_full_reingest("job-1", 7, {}, "model1")

        # ES, BM25 und DB-/Hash-Commit DUERFEN NICHT laufen
        mock_es_delete.assert_not_called()
        mock_db_cleanup.assert_not_called()
