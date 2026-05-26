"""Tests fuer ColBERT Cleanup Repair (Sprint 2 AP3).

Testet:
- cleanup_stale_colbert_models() Public Wrapper
- cleanup_stale_colbert_collections_without_db() Collection-only Helper
- _cleanup_stale_colbert_models() defensive Validierung
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

_CFG = {
    "retrieval": {"colbert": {"enabled": True, "model": "colbert-xm"}},
    "vectordb": {"qdrant_url": "http://localhost:6333"},
}

_CFG_DISABLED = {
    "retrieval": {"colbert": {"enabled": False, "model": "colbert-xm"}},
    "vectordb": {"qdrant_url": "http://localhost:6333"},
}


@contextmanager
def _lock_true(*args, **kwargs):
    yield True


@contextmanager
def _lock_false(*args, **kwargs):
    yield False


def _make_chunk_ds(**overrides):
    chunk_ds = MagicMock()
    chunk_ds.distinct_models.return_value = overrides.get("distinct_models", [])
    chunk_ds.get_chunk_ids.return_value = overrides.get("get_chunk_ids", [])
    chunk_ds.count_by_datasource.return_value = overrides.get("count_by_datasource", 0)
    return chunk_ds


# -------------------------------------------------------------------------
# cleanup_stale_colbert_models() Public Wrapper
# -------------------------------------------------------------------------


class TestCleanupStaleColbertModelsWrapper:
    @patch("kiara.retrieval.colbert.colbert_datasource_lock", side_effect=_lock_false)
    def test_lock_conflict_returns_stats(self, _lock):
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        stats = cleanup_stale_colbert_models(_CFG, 1, chunk_ds, ["old-model"])
        assert stats["lock_conflict"] is True
        assert stats["cleanup_invoked"] is False
        assert stats["ds_lock_acquired"] is False

    @patch("kiara.ingestion.pipeline._cleanup_stale_colbert_models")
    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    def test_normal_stale_invokes_private_cleanup(self, mock_lock, mock_private):
        mock_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        stats = cleanup_stale_colbert_models(_CFG, 1, chunk_ds, ["old-model"])
        assert stats["cleanup_invoked"] is True
        assert stats["ds_lock_acquired"] is True
        mock_private.assert_called_once()

    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    def test_current_model_skipped_when_enabled(self, mock_lock):
        mock_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        stats = cleanup_stale_colbert_models(_CFG, 1, chunk_ds, ["colbert-xm"])
        assert stats["skipped_current_models"] == ["colbert-xm"]
        assert stats["cleanup_invoked"] is False

    @patch("kiara.ingestion.pipeline._cleanup_stale_colbert_models")
    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    def test_current_model_not_skipped_when_disabled(self, mock_lock, mock_private):
        mock_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        stats = cleanup_stale_colbert_models(_CFG_DISABLED, 1, chunk_ds, ["colbert-xm"])
        assert stats["skipped_current_models"] == []
        assert stats["cleanup_invoked"] is True

    def test_invalid_models_skipped(self):
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        # colbert:: prefix und leerer String
        stats = cleanup_stale_colbert_models(
            _CFG,
            1,
            chunk_ds,
            ["", "colbert::bad", None],
        )
        assert len(stats["skipped_invalid_models"]) == 3
        assert stats["attempted_models"] == []

    @patch("kiara.ingestion.pipeline._delete_colbert_datasource_embedding_counter")
    @patch("kiara.retrieval.colbert.ColBERTStore.get_colbert_collection_info", return_value=None)
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    def test_db_only_current_model_cleanup(
        self, mock_ds_lock, mock_key_lock, mock_info, mock_counter_del
    ):
        """allow_current_models: Collection fehlt → DB-only Cleanup."""
        mock_ds_lock.side_effect = _lock_true
        mock_key_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds(get_chunk_ids=["c1", "c2"])
        with patch(
            "kiara.ingestion.pipeline._delete_colbert_datasource_entries_with_graph_cleanup"
        ) as mock_db_cleanup:
            stats = cleanup_stale_colbert_models(
                _CFG,
                1,
                chunk_ds,
                ["colbert-xm"],
                allow_current_models={"colbert-xm"},
            )
        assert stats["db_only_current_models"] == ["colbert-xm"]
        assert stats["cleanup_invoked"] is True
        mock_counter_del.assert_called_once()
        mock_db_cleanup.assert_called_once()

    @patch("kiara.retrieval.colbert.ColBERTStore.existing_ids", return_value={"c1"})
    @patch(
        "kiara.retrieval.colbert.ColBERTStore.get_colbert_collection_info",
        return_value={"name": "colbert_abc", "chunks": 10},
    )
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.retrieval.colbert.colbert_datasource_lock")
    def test_recreated_current_model_skipped(
        self, mock_ds_lock, mock_key_lock, mock_info, mock_existing
    ):
        """TOCTOU: Collection neu angelegt mit betroffenen Qdrant-Punkten → skip."""
        mock_ds_lock.side_effect = _lock_true
        mock_key_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds(get_chunk_ids=["c1", "c2"])
        stats = cleanup_stale_colbert_models(
            _CFG,
            1,
            chunk_ds,
            ["colbert-xm"],
            allow_current_models={"colbert-xm"},
        )
        assert len(stats["skipped_recreated_current_models"]) == 1
        assert stats["skipped_recreated_current_models"][0]["model"] == "colbert-xm"
        assert stats["cleanup_invoked"] is False


# -------------------------------------------------------------------------
# cleanup_stale_colbert_collections_without_db()
# -------------------------------------------------------------------------


class TestCleanupStaleCollectionsWithoutDb:
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.shared.db.db_query")
    def test_empty_datasources_deleted(self, mock_db_query, mock_lock):
        mock_lock.side_effect = _lock_true
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"cnt": 0}
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import cleanup_stale_colbert_collections_without_db

        with patch("qdrant_client.QdrantClient") as MockQC:
            client = MockQC.return_value
            stats = cleanup_stale_colbert_collections_without_db(
                _CFG,
                [
                    {
                        "model": "old-model",
                        "name": "colbert_old-model",
                        "chunks": 42,
                        "datasources": [],
                    }
                ],
            )
        assert "colbert_old-model" in stats["deleted"]
        client.delete_collection.assert_called_once()

    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    @patch("kiara.shared.db.db_query")
    def test_new_db_anchors_skip_deletion(self, mock_db_query, mock_lock):
        """Unter Lock: chunk_datasources hat neue Anker → skip."""
        mock_lock.side_effect = _lock_true
        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = {"cnt": 5}
        mock_db_query.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db_query.return_value.__exit__ = MagicMock(return_value=False)

        from kiara.ingestion.pipeline import cleanup_stale_colbert_collections_without_db

        with patch("qdrant_client.QdrantClient"):
            stats = cleanup_stale_colbert_collections_without_db(
                _CFG,
                [
                    {
                        "model": "old-model",
                        "name": "colbert_old-model",
                        "chunks": 42,
                        "datasources": [],
                    }
                ],
            )
        assert len(stats["skipped_with_datasources"]) == 1
        assert stats["deleted"] == []

    def test_name_model_mismatch_error(self):
        """name != colbert_collection_name(model) → error, kein delete."""
        from kiara.ingestion.pipeline import cleanup_stale_colbert_collections_without_db

        with patch("qdrant_client.QdrantClient") as MockQC:
            client = MockQC.return_value
            stats = cleanup_stale_colbert_collections_without_db(
                _CFG,
                [{"model": "old-model", "name": "colbert_WRONG", "chunks": 42, "datasources": []}],
            )
        assert len(stats["errors"]) == 1
        assert "mismatch" in stats["errors"][0]["error"]
        client.delete_collection.assert_not_called()

    def test_items_with_datasources_skipped(self):
        """Items mit nicht-leerer datasources-Liste werden uebersprungen."""
        from kiara.ingestion.pipeline import cleanup_stale_colbert_collections_without_db

        with patch("qdrant_client.QdrantClient"):
            stats = cleanup_stale_colbert_collections_without_db(
                _CFG,
                [
                    {
                        "model": "old",
                        "name": "colbert_old",
                        "chunks": 10,
                        "datasources": [{"id": 1, "name": "DS1"}],
                    }
                ],
            )
        assert len(stats["skipped_with_datasources"]) == 1


# -------------------------------------------------------------------------
# _cleanup_stale_colbert_models() defensive Validierung
# -------------------------------------------------------------------------


class TestPrivateCleanupDefensiveValidation:
    @patch("kiara.retrieval.colbert.colbert_cleanup_lock")
    def test_empty_model_skipped(self, mock_lock):
        """Leerer Modellname darf nicht crashen."""
        mock_lock.side_effect = _lock_true
        from kiara.ingestion.pipeline import _cleanup_stale_colbert_models

        chunk_ds = _make_chunk_ds()
        # Sollte nicht werfen
        _cleanup_stale_colbert_models(_CFG, 1, chunk_ds, ["", None])
        # Lock sollte nicht versucht worden sein (leere Modelle werden uebersprungen)


# -------------------------------------------------------------------------
# Repair-Dispatch (integrity_service.py)
# -------------------------------------------------------------------------


class TestRepairCleanupColbertDispatch:
    def test_valid_repairs_includes_cleanup_colbert(self):
        from kiara.admin.services.integrity_service import _VALID_REPAIRS

        assert "repair_cleanup_colbert" in _VALID_REPAIRS

    def test_dispatch_dict_includes_cleanup_colbert(self):
        """start_repair("repair_cleanup_colbert") findet die Funktion."""
        from kiara.admin.services import integrity_service as svc

        # Pruefen dass start_repair den Key akzeptiert (ValueError = unbekannt)
        with patch.object(
            svc, "_task", {"running": False, "action": None, "progress": None, "result": None}
        ):
            with patch("threading.Thread") as mock_thread:
                svc.start_repair("repair_cleanup_colbert")
                mock_thread.assert_called_once()
                target = mock_thread.call_args[1]["target"]
                assert target.__name__ == "_run_repair_cleanup_colbert"


class TestRepairCleanupColbertAdminFunction:
    def test_admin_function_calls_start_repair(self):
        import asyncio

        from kiara.admin.functions.integrity import integrity_cleanup_colbert

        with (
            patch("kiara.admin.services.integrity_service.is_running", return_value=False),
            patch("kiara.admin.services.integrity_service.start_repair") as mock_start,
        ):
            result = asyncio.run(integrity_cleanup_colbert({"id": 1, "username": "admin"}))
            mock_start.assert_called_once_with("repair_cleanup_colbert")
            assert result.success

    def test_admin_function_fails_when_running(self):
        import asyncio

        from kiara.admin.functions.integrity import integrity_cleanup_colbert

        with patch("kiara.admin.services.integrity_service.is_running", return_value=True):
            result = asyncio.run(integrity_cleanup_colbert({"id": 1, "username": "admin"}))
            assert not result.success
            assert "laeuft bereits" in result.message
