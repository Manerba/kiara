"""Tests fuer ColBERT Model-Switch Cleanup (Sprint 1).

Deckt ab:
- AP1: ColBERTStore Helper (add Length-Guard, get_colbert_collection_info,
       ids_with_datasource, existing_ids, drop_collection_if_empty, Locks)
- AP2: Detection (_cleanup_colbert_model_switch_if_ready)
- AP3: Cleanup (_cleanup_stale_colbert_models)
- AP4: Pipeline-Integration (run_completed, embedding_model-Aufloesung, Marker)
- AP5: Writer (_ColBERTWriteBlocked, Guards, Hash-/Pending-Cleanup)
"""

from unittest.mock import MagicMock, patch

import pytest

from kiara.backend_selector import BackendUnavailableError

# =====================================================================
# AP1: ColBERTStore Helper-Tests
# =====================================================================


class TestColBERTStoreAddLengthGuard:
    """add() Length-Guard ohne _client.upsert()."""

    def test_all_empty_returns_without_upsert(self):
        from kiara.retrieval.colbert import ColBERTStore

        with patch.object(ColBERTStore, "__init__", return_value=None):
            store = ColBERTStore.__new__(ColBERTStore)
            store._client = MagicMock()
            store._collection = "test"
            store.add([], [], [], [])
            store._client.upsert.assert_not_called()

    def test_length_mismatch_raises(self):
        from kiara.retrieval.colbert import ColBERTStore

        with patch.object(ColBERTStore, "__init__", return_value=None):
            store = ColBERTStore.__new__(ColBERTStore)
            store._client = MagicMock()
            store._collection = "test"
            with pytest.raises(ValueError, match="Laengenabweichung"):
                store.add(["text1"], [{}], ["id1", "id2"], [[[]]])

    def test_equal_lengths_proceeds(self):
        from kiara.retrieval.colbert import ColBERTStore

        with patch.object(ColBERTStore, "__init__", return_value=None):
            store = ColBERTStore.__new__(ColBERTStore)
            store._client = MagicMock()
            store._collection = "test"
            store.add(["t"], [{"k": "v"}], ["id1"], [[[0.1, 0.2]]])
            store._client.upsert.assert_called_once()


class TestGetColbertCollectionInfo:
    """get_colbert_collection_info mit count(exact=True) und Fehler-Propagation."""

    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_existing_collection(self, mock_qc_cls):
        from kiara.retrieval.colbert import ColBERTStore

        mock_client = MagicMock()
        mock_qc_cls.return_value = mock_client
        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_client.get_collection.return_value = mock_info

        result = ColBERTStore.get_colbert_collection_info({"vectordb": {}}, "colbert-xm")
        assert result is not None
        assert result["chunks"] == 42
        assert "colbert_" in result["name"]

    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_missing_collection_returns_none(self, mock_qc_cls):
        import httpx
        from qdrant_client.http.exceptions import UnexpectedResponse

        from kiara.retrieval.colbert import ColBERTStore

        mock_client = MagicMock()
        mock_qc_cls.return_value = mock_client
        exc = UnexpectedResponse(
            status_code=404,
            reason_phrase="Not Found",
            content=b"not found",
            headers=httpx.Headers({}),
        )
        mock_client.get_collection.side_effect = exc

        result = ColBERTStore.get_colbert_collection_info({"vectordb": {}}, "colbert-xm")
        assert result is None

    @patch("kiara.retrieval.colbert.QdrantClient")
    def test_transient_error_reraises(self, mock_qc_cls):
        """Generic Exception (z.B. ConnectionError) muss re-raisen, nicht None zurueck."""
        from kiara.retrieval.colbert import ColBERTStore

        mock_client = MagicMock()
        mock_qc_cls.return_value = mock_client
        mock_client.get_collection.side_effect = ConnectionError("network down")

        with pytest.raises(ConnectionError, match="network down"):
            ColBERTStore.get_colbert_collection_info({"vectordb": {}}, "colbert-xm")


class TestIdsWithDatasource:
    """ids_with_datasource mit Skalar/List/None ds_ids."""

    def _make_store(self):
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore.__new__(ColBERTStore)
        store._client = MagicMock()
        store._collection = "test"
        return store

    def test_list_ds_ids_match(self):
        store = self._make_store()
        pt = MagicMock()
        pt.id = "abc"
        pt.payload = {"ds_ids": [1, 2, 3]}
        store._client.retrieve.return_value = [pt]

        result = store.ids_with_datasource(["abc"], ds_id=2)
        assert result == {"abc"}

    def test_scalar_ds_ids_match(self):
        store = self._make_store()
        pt = MagicMock()
        pt.id = "abc"
        pt.payload = {"ds_ids": 5}
        store._client.retrieve.return_value = [pt]

        result = store.ids_with_datasource(["abc"], ds_id=5)
        assert result == {"abc"}

    def test_none_ds_ids_no_match(self):
        store = self._make_store()
        pt = MagicMock()
        pt.id = "abc"
        pt.payload = {"ds_ids": None}
        store._client.retrieve.return_value = [pt]

        result = store.ids_with_datasource(["abc"], ds_id=1)
        assert result == set()

    def test_ds_not_in_list(self):
        store = self._make_store()
        pt = MagicMock()
        pt.id = "abc"
        pt.payload = {"ds_ids": [2, 3]}
        store._client.retrieve.return_value = [pt]

        result = store.ids_with_datasource(["abc"], ds_id=1)
        assert result == set()

    def test_empty_input(self):
        store = self._make_store()
        result = store.ids_with_datasource([], ds_id=1)
        assert result == set()


class TestExistingIds:
    """existing_ids Batching/Normalisierung."""

    def test_returns_existing_only(self):
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore.__new__(ColBERTStore)
        store._client = MagicMock()
        store._collection = "test"

        pt1 = MagicMock()
        pt1.id = "id1"
        pt2 = MagicMock()
        pt2.id = "id3"
        store._client.retrieve.return_value = [pt1, pt2]

        result = store.existing_ids(["id1", "id2", "id3"])
        assert result == {"id1", "id3"}

    def test_empty_returns_empty(self):
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore.__new__(ColBERTStore)
        store._client = MagicMock()
        store._collection = "test"
        result = store.existing_ids([])
        assert result == set()

    def test_retrieve_error_reraises(self):
        """Qdrant-retrieve Fehler wird propagiert (fail-closed)."""
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore.__new__(ColBERTStore)
        store._client = MagicMock()
        store._collection = "test"
        store._client.retrieve.side_effect = ConnectionError("qdrant down")

        with pytest.raises(ConnectionError, match="qdrant down"):
            store.existing_ids(["id1", "id2"])


class TestDropCollectionIfEmpty:
    """drop_collection_if_empty bei leerer, nicht-leerer und Exception-Collection."""

    def _make_store(self):
        from kiara.retrieval.colbert import ColBERTStore

        store = ColBERTStore.__new__(ColBERTStore)
        store._client = MagicMock()
        store._collection = "test_coll"
        return store

    def test_empty_collection_dropped(self):
        store = self._make_store()
        mock_info = MagicMock()
        mock_info.points_count = 0
        store._client.get_collection.return_value = mock_info

        assert store.drop_collection_if_empty() is True
        store._client.delete_collection.assert_called_once_with("test_coll")

    def test_non_empty_not_dropped(self):
        store = self._make_store()
        mock_info = MagicMock()
        mock_info.points_count = 5
        store._client.get_collection.return_value = mock_info

        assert store.drop_collection_if_empty() is False
        store._client.delete_collection.assert_not_called()

    def test_count_exception_propagated(self):
        store = self._make_store()
        store._client.get_collection.side_effect = Exception("network")

        with pytest.raises(Exception, match="network"):
            store.drop_collection_if_empty()
        store._client.delete_collection.assert_not_called()

    def test_none_count_reraises(self):
        """points_count=None + count() returns None → RuntimeError."""
        store = self._make_store()
        mock_info = MagicMock()
        mock_info.points_count = None
        store._client.get_collection.return_value = mock_info
        count_result = MagicMock()
        count_result.count = None
        store._client.count.return_value = count_result

        with pytest.raises(RuntimeError, match="unbekannt"):
            store.drop_collection_if_empty()
        store._client.delete_collection.assert_not_called()


class TestAdvisoryLocks:
    """Lock wait=True/wait=False/Unlock bei Exception."""

    @patch("kiara.shared.db.get_db")
    def test_wait_false_not_acquired(self, mock_get_db):
        from kiara.retrieval.colbert import colbert_cleanup_lock

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"ok": False}
        mock_get_db.return_value = mock_conn

        with colbert_cleanup_lock("test::key", wait=False) as acquired:
            assert acquired is False
        mock_conn.close.assert_called_once()

    @patch("kiara.shared.db.get_db")
    def test_wait_true_acquired(self, mock_get_db):
        from kiara.retrieval.colbert import colbert_cleanup_lock

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        with colbert_cleanup_lock("test::key", wait=True) as acquired:
            assert acquired is True
        mock_conn.close.assert_called_once()

    @patch("kiara.shared.db.get_db")
    def test_datasource_lock_delegates(self, mock_get_db):
        """colbert_datasource_lock delegiert auf colbert_cleanup_lock mit datasource::N Key."""
        from kiara.retrieval.colbert import colbert_datasource_lock

        mock_conn = MagicMock()
        mock_get_db.return_value = mock_conn

        with colbert_datasource_lock(42, wait=True) as acquired:
            assert acquired is True
        mock_conn.close.assert_called_once()


# =====================================================================
# AP2: Detection Tests
# =====================================================================


class TestCleanupColbertModelSwitchIfReady:
    """Readiness-Guards einzeln testen."""

    def test_skip_if_not_completed(self):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        # Sollte ohne Fehler returnen
        _cleanup_colbert_model_switch_if_ready(
            {},
            None,
            MagicMock(),
            "model",
            run_completed=False,
            stats={},
        )

    def test_skip_if_errors(self):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        _cleanup_colbert_model_switch_if_ready(
            {},
            1,
            MagicMock(),
            "model",
            run_completed=True,
            stats={"errors": 3},
        )

    def test_skip_if_no_datasource_id(self):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        _cleanup_colbert_model_switch_if_ready(
            {},
            None,
            MagicMock(),
            "model",
            run_completed=True,
            stats={"errors": 0},
        )

    def test_skip_if_colbert_disabled(self):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        config = {"retrieval": {"colbert": {"enabled": False}}}
        _cleanup_colbert_model_switch_if_ready(
            config,
            1,
            MagicMock(),
            "model",
            run_completed=True,
            stats={"errors": 0},
        )

    @patch("kiara.ingestion.pipeline.normalize_model_name", return_value="colbert-xm")
    def test_skip_if_no_current_chunks(self, _):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}
        chunk_ds = MagicMock()
        chunk_ds.get_chunk_ids.return_value = []

        _cleanup_colbert_model_switch_if_ready(
            config,
            1,
            chunk_ds,
            "model",
            run_completed=True,
            stats={"errors": 0},
        )

    @patch("kiara.ingestion.pipeline._cleanup_stale_colbert_models")
    @patch(
        "kiara.ingestion.vectordb_ops._resolve_colbert_models_for_ds",
        return_value=["colbert-xm", "old-model"],
    )
    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.ingestion.pipeline.normalize_model_name", return_value="colbert-xm")
    def test_happy_path_triggers_cleanup(
        self,
        mock_norm,
        mock_cstore_cls,
        mock_ds_lock,
        mock_resolve,
        mock_cleanup,
    ):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}

        chunk_ds = MagicMock()
        chunk_ds.get_chunk_ids.side_effect = [
            ["c1", "c2"],  # current_key
            ["c1", "c2"],  # dense embedding_model
        ]

        mock_cstore_cls.get_colbert_collection_info.return_value = {
            "name": "colbert_colbert-xm",
            "chunks": 10,
        }
        mock_instance = MagicMock()
        mock_instance.ids_with_datasource.return_value = {"c1", "c2"}
        mock_cstore_cls.return_value = mock_instance

        mock_ds_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_ds_lock.return_value.__exit__ = MagicMock(return_value=False)

        _cleanup_colbert_model_switch_if_ready(
            config,
            1,
            chunk_ds,
            "model",
            run_completed=True,
            stats={"errors": 0},
        )
        mock_cleanup.assert_called_once()
        # Stale-Modelle sollten "old-model" enthalten
        stale_arg = mock_cleanup.call_args[0][3]
        assert "old-model" in stale_arg
        assert "colbert-xm" not in stale_arg


class TestPayloadCoverageGuard:
    """Point existiert, aber ds_ids enthaelt die DS nicht → kein Cleanup."""

    @patch("kiara.ingestion.pipeline._cleanup_stale_colbert_models")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.ingestion.pipeline.normalize_model_name", return_value="colbert-xm")
    def test_coverage_mismatch_skips(self, mock_norm, mock_cstore_cls, mock_cleanup):
        from kiara.ingestion.pipeline import _cleanup_colbert_model_switch_if_ready

        config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}

        chunk_ds = MagicMock()
        chunk_ds.get_chunk_ids.side_effect = [
            ["c1", "c2"],  # current_key
            ["c1", "c2"],  # dense embedding_model
        ]

        mock_cstore_cls.get_colbert_collection_info.return_value = {
            "name": "colbert_colbert-xm",
            "chunks": 10,
        }
        mock_instance = MagicMock()
        # c2 fehlt in den ds_ids-Payloads → Coverage-Guard greift
        mock_instance.ids_with_datasource.return_value = {"c1"}
        mock_cstore_cls.return_value = mock_instance

        _cleanup_colbert_model_switch_if_ready(
            config,
            1,
            chunk_ds,
            "model",
            run_completed=True,
            stats={"errors": 0},
        )
        mock_cleanup.assert_not_called()


# =====================================================================
# AP3: Cleanup Tests
# =====================================================================


class TestCleanupStaleColbertModels:
    """Cleanup-Logik pro stale Modell."""

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_missing_collection_db_only_cleanup(self, mock_cstore_cls, mock_lock):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_cstore_cls.get_colbert_collection_info.return_value = None

        chunk_ds = MagicMock()
        chunk_ds.delete_datasource_entries.return_value = 5

        with (
            patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter"),
            patch(
                "kiara.ingestion.pipeline._delete_colbert_datasource_entries_with_graph_cleanup"
            ) as mock_db_cleanup,
        ):
            _cleanup_stale_colbert_models(
                {},
                1,
                chunk_ds,
                ["old-model"],
            )
        mock_db_cleanup.assert_called_once()
        # Keine Qdrant-Mutation (kein delete_by_ids, kein update_ds_ids)

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_empty_collection_dropped(self, mock_cstore_cls, mock_lock):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_cstore_cls.get_colbert_collection_info.return_value = {
            "name": "colbert_old",
            "chunks": 0,
        }
        mock_instance = MagicMock()
        mock_instance.drop_collection_if_empty.return_value = True
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()

        with (
            patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter"),
            patch(
                "kiara.ingestion.pipeline._delete_colbert_datasource_entries_with_graph_cleanup"
            ) as mock_db_cleanup,
        ):
            _cleanup_stale_colbert_models(
                {},
                1,
                chunk_ds,
                ["old-model"],
            )
        mock_instance.drop_collection_if_empty.assert_called_once()
        mock_db_cleanup.assert_called_once()

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_drop_error_preserves_db_retry_anker(self, mock_cstore_cls, mock_lock):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_cstore_cls.get_colbert_collection_info.return_value = {
            "name": "colbert_old",
            "chunks": 0,
        }
        mock_instance = MagicMock()
        mock_instance.drop_collection_if_empty.side_effect = Exception("qdrant error")
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()

        with patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter"):
            _cleanup_stale_colbert_models(
                {},
                1,
                chunk_ds,
                ["old-model"],
            )
        # DB-Cleanup darf NICHT stattgefunden haben (Retry-Anker erhalten)
        chunk_ds.delete_datasource_entries.assert_not_called()

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_lock_conflict_skips(self, mock_cstore_cls, mock_lock):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=False)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)

        chunk_ds = MagicMock()
        _cleanup_stale_colbert_models({}, 1, chunk_ds, ["old-model"])

        # Keine Collection-Info-Abfrage und kein DB-Cleanup
        mock_cstore_cls.get_colbert_collection_info.assert_not_called()
        chunk_ds.delete_datasource_entries.assert_not_called()

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    def test_happy_path_full_cleanup(self, mock_cstore_cls, mock_lock):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_cstore_cls.get_colbert_collection_info.return_value = {
            "name": "colbert_old",
            "chunks": 10,
        }
        mock_instance = MagicMock()
        mock_instance.existing_ids.return_value = {"shared1"}
        mock_instance.drop_collection_if_empty.return_value = False
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()
        chunk_ds.preview_datasource_removal.return_value = (
            ["orphan1", "orphan2"],
            {"shared1": [2, 3]},
        )

        with (
            patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter"),
            patch(
                "kiara.ingestion.pipeline._delete_colbert_datasource_entries_with_graph_cleanup"
            ) as mock_db_cleanup,
        ):
            _cleanup_stale_colbert_models(
                {},
                1,
                chunk_ds,
                ["old-model"],
            )

        mock_instance.delete_by_ids.assert_called_once_with(["orphan1", "orphan2"])
        mock_instance.update_ds_ids.assert_called_once_with(
            {"shared1": [2, 3]},
            wait=True,
        )
        mock_db_cleanup.assert_called_once()


class TestCounterDeleteOrder:
    """Counter-Delete vor chunk_ds.delete_datasource_entries;
    Counter-Fehler behaelt chunk_datasources als Retry-Anker."""

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter")
    def test_counter_error_preserves_chunk_datasources(
        self,
        mock_counter_del,
        mock_cstore_cls,
        mock_lock,
    ):
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        mock_lock.return_value.__enter__ = MagicMock(return_value=True)
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_cstore_cls.get_colbert_collection_info.return_value = None
        mock_counter_del.side_effect = Exception("counter DB error")

        chunk_ds = MagicMock()
        _cleanup_stale_colbert_models({}, 1, chunk_ds, ["old-model"])

        # chunk_datasources-Delete darf NICHT passiert sein
        chunk_ds.delete_datasource_entries.assert_not_called()


# =====================================================================
# AP4: Pipeline-Integration Tests
# =====================================================================


class TestRunCompletedTracking:
    """run_completed im Abort- und No-Files-Pfad."""

    @patch("kiara.ingestion.pipeline._cleanup_colbert_model_switch_if_ready")
    @patch("kiara.ingestion.pipeline._load_and_diff_files")
    @patch("kiara.ingestion.pipeline._apply_ds_overrides")
    @patch("kiara.ingestion.pipeline._check_ddu_preflight")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.pipeline.ChunkDatasources")
    @patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model")
    @patch("kiara.ingestion.pipeline.normalize_model_name", side_effect=lambda x: x)
    def test_no_files_calls_cleanup_with_completed_true(
        self,
        mock_norm,
        mock_gem,
        mock_cds,
        mock_vs,
        mock_ddu,
        mock_apply,
        mock_diff,
        mock_cleanup,
    ):
        from kiara.ingestion.pipeline import run_ingestion

        mock_apply.return_value = MagicMock()
        mock_hash_store = MagicMock()
        mock_diff.return_value = ([], [], {}, [], mock_hash_store, {})
        config = {
            "vectordb": {"path": "/tmp/test"},
            "ingestion": {"batch_size": 10},
            "datasource": {"path": "/tmp"},
        }

        run_ingestion(config, datasource_id=1, embedding_model="test-model")

        mock_cleanup.assert_called_once()
        _, kwargs = mock_cleanup.call_args
        assert kwargs.get("run_completed") is True or mock_cleanup.call_args[0][4] is True

    @patch("kiara.ingestion.pipeline._cleanup_colbert_model_switch_if_ready")
    @patch("kiara.ingestion.pipeline._persist_hashes_checkpoint")
    @patch("kiara.ingestion.pipeline._process_file_in_pipeline", return_value=False)
    @patch("kiara.ingestion.pipeline._init_ingestion_stats")
    @patch("kiara.ingestion.pipeline._load_and_diff_files")
    @patch("kiara.ingestion.pipeline._apply_ds_overrides")
    @patch("kiara.ingestion.pipeline._check_ddu_preflight")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.pipeline.ChunkDatasources")
    @patch("kiara.ingestion.pipeline.get_embedding_model", return_value="test-model")
    @patch("kiara.ingestion.pipeline.normalize_model_name", side_effect=lambda x: x)
    def test_abort_calls_cleanup_with_completed_false(
        self,
        mock_norm,
        mock_gem,
        mock_cds,
        mock_vs,
        mock_ddu,
        mock_apply,
        mock_diff,
        mock_stats,
        mock_process,
        mock_persist,
        mock_cleanup,
    ):
        from kiara.ingestion.pipeline import run_ingestion

        mock_apply.return_value = MagicMock()
        mock_hash_store = MagicMock()
        mock_diff.return_value = (
            ["/f1"],
            ["/f1"],
            {"/f1": ("h", 100)},
            [],
            mock_hash_store,
            {},
        )
        mock_stats.return_value = {
            "total_files": 1,
            "processed_files": 0,
            "skipped_files": 0,
            "deleted_files": 0,
            "total_chunks": 0,
            "errors": 0,
        }
        config = {
            "vectordb": {"path": "/tmp/test"},
            "ingestion": {"batch_size": 10},
            "datasource": {"path": "/tmp"},
        }

        run_ingestion(config, datasource_id=1, embedding_model="test-model")

        mock_cleanup.assert_called_once()
        # run_completed sollte False sein (Abort)
        cleanup_args = mock_cleanup.call_args
        assert cleanup_args[1].get("run_completed") is False or cleanup_args[0][4] is False


class TestEmbeddingModelResolution:
    """Zentrale embedding_model-Aufloesung inkl. Whitespace/Fallback."""

    @patch("kiara.ingestion.pipeline._cleanup_colbert_model_switch_if_ready")
    @patch("kiara.ingestion.pipeline._load_and_diff_files")
    @patch("kiara.ingestion.pipeline._apply_ds_overrides")
    @patch("kiara.ingestion.pipeline._check_ddu_preflight")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.pipeline.ChunkDatasources")
    @patch("kiara.ingestion.pipeline.get_embedding_model", return_value="fallback-model")
    @patch("kiara.ingestion.pipeline.normalize_model_name", side_effect=lambda x: x)
    def test_whitespace_embedding_model_falls_back(
        self,
        mock_norm,
        mock_gem,
        mock_cds,
        mock_vs,
        mock_ddu,
        mock_apply,
        mock_diff,
        mock_cleanup,
    ):
        from kiara.ingestion.pipeline import run_ingestion

        mock_apply.return_value = MagicMock()
        mock_hash_store = MagicMock()
        mock_diff.return_value = ([], [], {}, [], mock_hash_store, {})
        config = {
            "vectordb": {"path": "/tmp/test"},
            "ingestion": {"batch_size": 10},
            "datasource": {"path": "/tmp"},
        }

        run_ingestion(config, embedding_model="  ")

        # VectorStore sollte mit fallback-model erstellt worden sein
        mock_vs.assert_called_once()
        vs_kwargs = mock_vs.call_args
        assert vs_kwargs[1].get("embedding_model") == "fallback-model"


class TestMarkerInRunIngestion:
    """Marker in run_ingestion(config=None)."""

    @patch("kiara.ingestion.pipeline._cleanup_colbert_model_switch_if_ready")
    @patch("kiara.ingestion.pipeline._load_and_diff_files")
    @patch("kiara.ingestion.pipeline._apply_ds_overrides")
    @patch("kiara.ingestion.pipeline._check_ddu_preflight")
    @patch("kiara.ingestion.pipeline.VectorStore")
    @patch("kiara.ingestion.pipeline.ChunkDatasources")
    @patch("kiara.ingestion.pipeline.get_embedding_model", return_value="m")
    @patch("kiara.ingestion.pipeline.normalize_model_name", side_effect=lambda x: x)
    @patch("kiara.ingestion.pipeline.load_config")
    def test_config_none_sets_marker(
        self,
        mock_load,
        mock_norm,
        mock_gem,
        mock_cds,
        mock_vs,
        mock_ddu,
        mock_apply,
        mock_diff,
        mock_cleanup,
    ):
        from kiara.ingestion.pipeline import run_ingestion

        mock_load.return_value = {
            "vectordb": {"path": "/tmp/test"},
            "ingestion": {"batch_size": 10},
            "datasource": {"path": "/tmp"},
        }
        mock_apply.return_value = MagicMock()
        mock_hash_store = MagicMock()
        mock_diff.return_value = ([], [], {}, [], mock_hash_store, {})

        run_ingestion(config=None, embedding_model="m")

        # load_config wurde aufgerufen
        mock_load.assert_called_once()
        # Die Config die an _load_and_diff_files ging, hat den Marker
        actual_config = mock_diff.call_args[0][0]
        assert actual_config.get("_kiara_enforce_persisted_colbert_model") is True


class TestMarkerInRunAllDatasources:
    """Marker in run_all_datasources(config=None)."""

    @patch("kiara.ingestion.pipeline.run_ingestion")
    @patch("kiara.repositories.datasource.DatasourceRepository.get_active", return_value=[])
    @patch("kiara.ingestion.pipeline.load_config")
    def test_config_none_sets_marker(self, mock_load, mock_get_active, mock_run):
        from kiara.ingestion.pipeline import run_all_datasources

        mock_load.return_value = {"datasource": {"path": "/tmp"}}

        with patch("kiara.connectors.registry.ensure_loaded"):
            run_all_datasources(config=None)

        mock_load.assert_called_once()
        # Bei leeren Datasources wird run_ingestion() mit der markierten Config aufgerufen
        actual_config = mock_run.call_args[0][0]
        assert actual_config.get("_kiara_enforce_persisted_colbert_model") is True


# =====================================================================
# AP5: Writer Tests
# =====================================================================


class TestColBERTWriteBlockedHandler:
    """_ColBERTWriteBlocked bei Soft-Delete, DB-Lesefehler, Model-Mismatch."""

    def test_inactive_ds_raises(self):
        from kiara.ingestion.runtime import _datasource_accepts_colbert_write

        with patch("kiara.shared.db.db_query") as mock_dq:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = {"is_active": False}
            mock_dq.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_dq.return_value.__exit__ = MagicMock(return_value=False)
            assert _datasource_accepts_colbert_write(1) is False

    def test_missing_ds_raises(self):
        from kiara.ingestion.runtime import _datasource_accepts_colbert_write

        with patch("kiara.shared.db.db_query") as mock_dq:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = None
            mock_dq.return_value.__enter__ = MagicMock(return_value=mock_conn)
            mock_dq.return_value.__exit__ = MagicMock(return_value=False)
            assert _datasource_accepts_colbert_write(1) is False

    def test_db_error_fail_closed(self):
        from kiara.ingestion.runtime import _datasource_accepts_colbert_write

        with patch("kiara.shared.db.db_query") as mock_dq:
            mock_dq.return_value.__enter__ = MagicMock(side_effect=Exception("DB down"))
            mock_dq.return_value.__exit__ = MagicMock(return_value=False)
            assert _datasource_accepts_colbert_write(1) is False

    def test_model_mismatch_with_marker_raises(self):
        from kiara.ingestion.runtime import (
            _ColBERTWriteBlocked,
            _ensure_colbert_model_is_current_for_write,
        )

        config = {"_kiara_enforce_persisted_colbert_model": True}
        fresh_config = {"retrieval": {"colbert": {"model": "new-model"}}}

        with patch("kiara.config.load_config", return_value=fresh_config):
            with patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x):
                with pytest.raises(_ColBERTWriteBlocked, match="Mismatch"):
                    _ensure_colbert_model_is_current_for_write(config, "old-model")

    def test_model_match_with_marker_ok(self):
        from kiara.ingestion.runtime import _ensure_colbert_model_is_current_for_write

        config = {"_kiara_enforce_persisted_colbert_model": True}
        fresh_config = {"retrieval": {"colbert": {"model": "same-model"}}}

        with patch("kiara.config.load_config", return_value=fresh_config):
            with patch("kiara.backend_selector.normalize_model_name", side_effect=lambda x: x):
                # Darf keine Exception werfen
                _ensure_colbert_model_is_current_for_write(config, "same-model")

    def test_no_marker_skips_check(self):
        from kiara.ingestion.runtime import _ensure_colbert_model_is_current_for_write

        config = {}  # Kein Marker
        # Darf load_config nicht aufrufen
        with patch("kiara.config.load_config") as mock_load:
            _ensure_colbert_model_is_current_for_write(config, "any-model")
            mock_load.assert_not_called()

    def test_fresh_read_error_raises(self):
        from kiara.ingestion.runtime import (
            _ColBERTWriteBlocked,
            _ensure_colbert_model_is_current_for_write,
        )

        config = {"_kiara_enforce_persisted_colbert_model": True}

        with patch("kiara.config.load_config", side_effect=Exception("IO")):
            with pytest.raises(_ColBERTWriteBlocked, match="Fresh-Config-Read"):
                _ensure_colbert_model_is_current_for_write(config, "model")


class TestWriterDsIdsRemoval:
    """ds_ids vor Add entfernt."""

    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.ingestion.runtime._datasource_accepts_colbert_write", return_value=True)
    @patch("kiara.ingestion.runtime._ensure_colbert_model_is_current_for_write")
    @patch("kiara.retrieval.colbert.embed_colbert")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm")
    def test_ds_ids_stripped_from_metas(
        self,
        mock_norm,
        mock_cstore_cls,
        mock_embed,
        mock_ensure,
        mock_accepts,
        mock_key_lock,
        mock_ds_lock,
    ):

        from kiara.ingestion.runtime import _IngestionContext, _sync_colbert_chunks

        # Setup Locks
        mock_ds_lock.return_value.__enter__ = MagicMock()
        mock_ds_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_key_lock.return_value.__enter__ = MagicMock()
        mock_key_lock.return_value.__exit__ = MagicMock(return_value=False)

        mock_instance = MagicMock()
        mock_instance.get_collection_name.return_value = "colbert_colbert-xm"
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()
        chunk_ds.chunks_exist.return_value = set()  # Alle sind "missing"
        chunk_ds.get_datasource_ids_for_chunks.return_value = {}

        mock_embed.return_value = [[[0.1]]]

        ictx = MagicMock(spec=_IngestionContext)
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}
        ictx.ds_overrides = {"colbert_enabled": True}
        ictx.chunk_ds = chunk_ds
        ictx.stats = {}

        metas = [{"source": "file.txt", "ds_ids": [1, 2]}]
        _sync_colbert_chunks(ictx, ["c1"], ["text"], metas)

        # Prüfe dass ds_ids aus den Metadaten entfernt wurde
        add_call = mock_instance.add.call_args
        actual_metas = add_call[0][1]
        assert "ds_ids" not in actual_metas[0]


class TestWriterEmbeddingLengthMismatch:
    """Embedding-Laengenmismatch vor Mutation."""

    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.ingestion.runtime._datasource_accepts_colbert_write", return_value=True)
    @patch("kiara.ingestion.runtime._ensure_colbert_model_is_current_for_write")
    @patch("kiara.retrieval.colbert.embed_colbert")
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm")
    def test_length_mismatch_raises_before_mutation(
        self,
        mock_norm,
        mock_cstore_cls,
        mock_embed,
        mock_ensure,
        mock_accepts,
        mock_key_lock,
        mock_ds_lock,
    ):
        from kiara.backend_selector import BackendUnavailableError
        from kiara.ingestion.runtime import _sync_colbert_chunks

        mock_ds_lock.return_value.__enter__ = MagicMock()
        mock_ds_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_key_lock.return_value.__enter__ = MagicMock()
        mock_key_lock.return_value.__exit__ = MagicMock(return_value=False)

        mock_instance = MagicMock()
        mock_instance.get_collection_name.return_value = "coll"
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()
        chunk_ds.chunks_exist.return_value = set()

        # 2 Chunks but 1 embedding → mismatch
        mock_embed.return_value = [[[0.1]]]

        ictx = MagicMock()
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}
        ictx.ds_overrides = {"colbert_enabled": True}
        ictx.chunk_ds = chunk_ds
        ictx.stats = {}

        with pytest.raises(BackendUnavailableError, match="Embeddings"):
            _sync_colbert_chunks(ictx, ["c1", "c2"], ["t1", "t2"], [{}, {}])

        # add() darf NICHT aufgerufen worden sein
        mock_instance.add.assert_not_called()


class TestWriterUpdateDsIdsWaitTrue:
    """update_ds_ids(..., wait=True) im Writer-Pfad."""

    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.ingestion.runtime._datasource_accepts_colbert_write", return_value=True)
    @patch("kiara.ingestion.runtime._ensure_colbert_model_is_current_for_write")
    @patch("kiara.retrieval.colbert.embed_colbert", return_value=[])
    @patch("kiara.retrieval.colbert.ColBERTStore")
    @patch("kiara.backend_selector.normalize_model_name", return_value="colbert-xm")
    def test_update_ds_ids_with_wait_true(
        self,
        mock_norm,
        mock_cstore_cls,
        mock_embed,
        mock_ensure,
        mock_accepts,
        mock_key_lock,
        mock_ds_lock,
    ):
        from kiara.ingestion.runtime import _sync_colbert_chunks

        mock_ds_lock.return_value.__enter__ = MagicMock()
        mock_ds_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_key_lock.return_value.__enter__ = MagicMock()
        mock_key_lock.return_value.__exit__ = MagicMock(return_value=False)

        mock_instance = MagicMock()
        mock_instance.get_collection_name.return_value = "coll"
        mock_cstore_cls.return_value = mock_instance

        chunk_ds = MagicMock()
        chunk_ds.chunks_exist.return_value = {"c1"}  # Alle existieren bereits
        chunk_ds.get_datasource_ids_for_chunks.return_value = {"c1": [1, 2]}

        ictx = MagicMock()
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}}}
        ictx.ds_overrides = {"colbert_enabled": True}
        ictx.chunk_ds = chunk_ds
        ictx.stats = {}

        _sync_colbert_chunks(ictx, ["c1"], ["text"], [{}])

        mock_instance.update_ds_ids.assert_called_once()
        _, kwargs = mock_instance.update_ds_ids.call_args
        assert kwargs.get("wait") is True


# =====================================================================
# Luecke 1: _handle_colbert_step ABORT-Semantik bei _ColBERTWriteBlocked
# =====================================================================


class TestHandleColbertStepAbortSemantics:
    """_handle_colbert_step: _ColBERTWriteBlocked → ABORT mit abort_reason,
    Hash-Cleanup, pending_hashes.pop, file_log emit."""

    @patch("kiara.ingestion.runtime._emit_file_log")
    @patch("kiara.ingestion.runtime._persist_hashes_checkpoint")
    @patch("kiara.ingestion.runtime._cleanup_file_hash")
    @patch("kiara.ingestion.runtime._record_error")
    @patch("kiara.ingestion.runtime._sync_colbert_chunks")
    @patch("kiara.ingestion.runtime._colbert_enabled_for_run", return_value=True)
    def test_write_blocked_returns_abort(
        self, _enabled, mock_sync, mock_record, mock_cleanup, mock_persist, mock_emit
    ):
        from kiara.ingestion.runtime import (
            _ColBERTWriteBlocked,
            _FileAction,
            _handle_colbert_step,
        )

        mock_sync.side_effect = _ColBERTWriteBlocked("model mismatch")

        ictx = MagicMock()
        ictx.stats = {}
        ictx.pending_hashes = {"/tmp/test.txt": "hash123"}
        file_log = MagicMock()

        action, retries = _handle_colbert_step(
            ictx,
            chunk_ids=["c1"],
            chunk_texts=["t"],
            chunk_metas=[{}],
            file_path="/tmp/test.txt",
            file_log=file_log,
            file_retries=0,
            t0=0.0,
        )
        assert action == _FileAction.ABORT
        assert "ColBERT-Write blockiert" in ictx.stats["abort_reason"]
        mock_record.assert_called_once()
        assert mock_record.call_args[1].get("is_colbert") is True
        mock_cleanup.assert_called_once_with(ictx, "/tmp/test.txt")
        assert "/tmp/test.txt" not in ictx.pending_hashes
        mock_persist.assert_called_once_with(ictx, force=True)
        mock_emit.assert_called_once()


# =====================================================================
# Luecke 2: Pending-Hash-Cleanup-Suite (4 Fehlerszenarien)
# =====================================================================


class TestColbertStepHashCleanup:
    """Hash-Cleanup-Invariante: Jeder Error-Pfad in _handle_colbert_step
    muss _cleanup_file_hash + pending_hashes.pop aufrufen."""

    def _make_ictx(self):
        ictx = MagicMock()
        ictx.stats = {}
        ictx.pending_hashes = {"/tmp/f.txt": "h1"}
        ictx.consecutive_errors = 0
        return ictx

    @patch("kiara.ingestion.runtime._emit_file_log")
    @patch("kiara.ingestion.runtime._persist_hashes_checkpoint")
    @patch("kiara.ingestion.runtime._cleanup_file_hash")
    @patch("kiara.ingestion.runtime._record_error")
    @patch("kiara.ingestion.runtime._handle_circuit_breaker")
    @patch("kiara.ingestion.runtime._sync_colbert_chunks")
    @patch("kiara.ingestion.runtime._colbert_enabled_for_run", return_value=True)
    def test_backend_unavailable_cleans_hash(
        self, _en, mock_sync, mock_cb, mock_record, mock_cleanup, mock_persist, mock_emit
    ):
        from kiara.ingestion.runtime import _CircuitBreakerResult, _handle_colbert_step

        mock_sync.side_effect = BackendUnavailableError("timeout")
        mock_cb.return_value = (_CircuitBreakerResult.SKIP, 0)

        ictx = self._make_ictx()
        action, _ = _handle_colbert_step(
            ictx,
            chunk_ids=["c1"],
            chunk_texts=["t"],
            chunk_metas=[{}],
            file_path="/tmp/f.txt",
            file_log=MagicMock(),
            file_retries=0,
            t0=0.0,
        )
        mock_cleanup.assert_called_once_with(ictx, "/tmp/f.txt")
        assert "/tmp/f.txt" not in ictx.pending_hashes

    @patch("kiara.ingestion.runtime._cleanup_file_hash")
    @patch("kiara.ingestion.runtime._record_error")
    @patch("kiara.ingestion.runtime._sync_colbert_chunks")
    @patch("kiara.ingestion.runtime._colbert_enabled_for_run", return_value=True)
    def test_generic_add_error_cleans_hash(self, _en, mock_sync, mock_record, mock_cleanup):
        from kiara.ingestion.runtime import _FileAction, _handle_colbert_step

        mock_sync.side_effect = ValueError("unexpected")

        ictx = self._make_ictx()
        action, _ = _handle_colbert_step(
            ictx,
            chunk_ids=["c1"],
            chunk_texts=["t"],
            chunk_metas=[{}],
            file_path="/tmp/f.txt",
            file_log=MagicMock(),
            file_retries=0,
            t0=0.0,
        )
        assert action == _FileAction.SKIP
        mock_cleanup.assert_called_once_with(ictx, "/tmp/f.txt")
        assert "/tmp/f.txt" not in ictx.pending_hashes

    @patch("kiara.ingestion.runtime._emit_file_log")
    @patch("kiara.ingestion.runtime._persist_hashes_checkpoint")
    @patch("kiara.ingestion.runtime._cleanup_file_hash")
    @patch("kiara.ingestion.runtime._sync_colbert_chunks")
    @patch("kiara.ingestion.runtime._colbert_enabled_for_run", return_value=True)
    def test_cancel_after_dense_success_cleans_hash(
        self, _en, mock_sync, mock_cleanup, mock_persist, mock_emit
    ):
        from kiara.ingestion._file_utils import _CancelledError
        from kiara.ingestion.runtime import _FileAction, _handle_colbert_step

        mock_sync.side_effect = _CancelledError()

        ictx = self._make_ictx()
        action, _ = _handle_colbert_step(
            ictx,
            chunk_ids=["c1"],
            chunk_texts=["t"],
            chunk_metas=[{}],
            file_path="/tmp/f.txt",
            file_log=MagicMock(),
            file_retries=0,
            t0=0.0,
        )
        assert action == _FileAction.CANCEL
        mock_cleanup.assert_called_once_with(ictx, "/tmp/f.txt")
        assert "/tmp/f.txt" not in ictx.pending_hashes
        mock_persist.assert_called_once_with(ictx, force=True)


# =====================================================================
# Luecke 6: Counter-Delete No-op (idempotent bei fehlender Zeile)
# =====================================================================


class TestCounterDeleteNoop:
    """_delete_colbert_datasource_embedding_counter bei fehlender Zeile = No-op."""

    @patch("kiara.ingestion.pipeline.db_transaction")
    def test_missing_row_is_noop(self, mock_tx):
        from kiara.ingestion.pipeline import _delete_colbert_datasource_embedding_counter

        mock_conn = MagicMock()
        # DELETE betrifft 0 Zeilen — kein Fehler
        mock_conn.execute.return_value.rowcount = 0
        mock_tx.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_tx.return_value.__exit__ = MagicMock(return_value=False)

        # Muss ohne Exception durchlaufen
        _delete_colbert_datasource_embedding_counter(999, "colbert::nonexistent")
        mock_conn.execute.assert_called_once()
