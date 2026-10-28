"""Tests fuer die extrahierten Hilfsfunktionen aus run_ingestion().

Testet die 9 neuen Funktionen aus dem Decomposition-Refactoring (V8):
- _init_ingestion_stats
- _persist_hashes_checkpoint
- _handle_circuit_breaker (inkl. Bug-Fix is_infra_error)
- _handle_loading_result
- _handle_chunk_dedup
- _handle_embedding_phase
- _finalize_file
"""

import time
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
from kiara_ingestion_core.models import FileLogEntry, ProcessingContext

from kiara.backend_selector import BackendUnavailableError
from kiara.ingestion._file_utils import _CancelledError
from kiara.ingestion.runtime import (
    _chunk_and_collect,
    _CircuitBreakerResult,
    _DedupResult,
    _FileAction,
    _finalize_file,
    _handle_chunk_dedup,
    _handle_circuit_breaker,
    _handle_embedding_phase,
    _handle_loading_result,
    _IngestionContext,
    _init_ingestion_stats,
    _late_chunking_file_eligible,
    _persist_hashes_checkpoint,
    _process_file_in_pipeline,
    _refresh_late_chunking_snapshots,
    _rollback_embedding_persistence,
    _RollbackFailedError,
    _should_use_late_chunking,
    _write_embeddings,
    _write_to_elasticsearch,
)
from kiara.shared.constants import (
    HASH_CHECKPOINT_INTERVAL,
    INGESTION_MAX_CONSECUTIVE_ERRORS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ictx(**overrides):
    """Erzeugt einen minimalen _IngestionContext mit Mock-Objekten."""
    store = MagicMock()
    store.finalize_document_versions.side_effect = lambda ids, *, publish_search, on_complete: (
        on_complete({})
    )
    defaults = dict(
        config={
            "vectordb": {"path": "/tmp"},
            "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
        },
        ctx=ProcessingContext(),
        store=store,
        chunk_ds=MagicMock(),
        hash_store=MagicMock(),
        batch_size=10,
        datasource_id=1,
        embedding_model="test-model",
        hash_mode="mtime",
        ds_overrides=None,
        file_hashes={},
        stats={
            "total_files": 10,
            "processed_files": 0,
            "skipped_files": 0,
            "errors": 0,
            "total_chunks": 0,
        },
        pending_hashes={},
        file_completion_callback=MagicMock(),
        consecutive_errors=0,
        skipped=0,
        total=10,
    )
    defaults.update(overrides)
    return _IngestionContext(**defaults)


def _make_file_log(file_path="/test/file.txt", **overrides):
    """Erzeugt einen minimalen FileLogEntry."""
    p = Path(file_path)
    defaults = dict(
        file_path=file_path,
        file_name=p.name,
        file_size=1024,
        file_ext=p.suffix.lower(),
    )
    defaults.update(overrides)
    return FileLogEntry(**defaults)


def _mock_rollback_documents(ictx, ids):
    """Model actual vector removal so the rollback can verify its result."""
    remaining = set(ids)

    def read(requested, **kwargs):
        found = [cid for cid in requested if cid in remaining]
        return {"ids": found, "metadatas": [{"ds_ids": [ictx.datasource_id]} for _ in found]}

    ictx.store.get_documents_by_ids.side_effect = read
    ictx.store.delete_by_ids.side_effect = remaining.difference_update


# ===================================================================
# _init_ingestion_stats
# ===================================================================


class TestInitIngestionStats:
    def test_basic_calculation(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world!!")

        stats = _init_ingestion_stats(
            [str(f1), str(f2)],
            [str(f1)],
            ["/old/deleted.txt"],
            {"resumed.txt": ("abc", 50)},
        )
        assert stats["total_files"] == 2
        assert stats["processed_files"] == 0
        assert stats["skipped_files"] == 1
        assert stats["resumed_skipped"] == 1
        assert stats["deleted_files"] == 1
        assert stats["total_chunks"] == 0
        assert stats["errors"] == 0

    def test_oserror_fallback(self, tmp_path):
        f1 = tmp_path / "exists.txt"
        f1.write_text("data")
        stats = _init_ingestion_stats(
            [str(f1), "/nonexistent/ghost.txt"],
            [str(f1)],
            [],
            {},
        )
        assert stats["total_files"] == 2

    def test_empty_lists(self):
        stats = _init_ingestion_stats([], [], [], {})
        assert stats["total_files"] == 0
        assert stats["skipped_files"] == 0


# ===================================================================
# _persist_hashes_checkpoint
# ===================================================================


class TestPersistHashesCheckpoint:
    def test_empty_pending_noop(self):
        ictx = _make_ictx(pending_hashes={})
        _persist_hashes_checkpoint(ictx)
        ictx.hash_store.upsert_many.assert_not_called()

    def test_force_flushes_immediately(self):
        pending = {"a.txt": ("h1", 100), "b.txt": ("h2", 200)}
        ictx = _make_ictx(pending_hashes=pending)
        _persist_hashes_checkpoint(ictx, force=True)
        ictx.hash_store.upsert_many.assert_called_once_with(1, "test-model", pending)
        assert ictx.pending_hashes == {}

    def test_interval_hit_flushes(self):
        pending = {"a.txt": ("h1", 100)}
        ictx = _make_ictx(
            pending_hashes=pending,
            stats={"processed_files": HASH_CHECKPOINT_INTERVAL, "errors": 0},
        )
        _persist_hashes_checkpoint(ictx)
        ictx.hash_store.upsert_many.assert_called_once()
        assert ictx.pending_hashes == {}

    def test_interval_not_hit_no_flush(self):
        pending = {"a.txt": ("h1", 100)}
        ictx = _make_ictx(
            pending_hashes=pending,
            stats={"processed_files": 1, "errors": 0},
        )
        _persist_hashes_checkpoint(ictx)
        ictx.hash_store.upsert_many.assert_not_called()
        assert ictx.pending_hashes == {"a.txt": ("h1", 100)}


# ===================================================================
# _handle_circuit_breaker
# ===================================================================


class TestHandleCircuitBreaker:
    def test_content_error_continue_no_increment(self):
        """is_infra=False → CONTINUE, consecutive_errors unverändert."""
        ictx = _make_ictx(consecutive_errors=3)
        result, retries = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="loading",
            error_description="korrupt",
            is_infra=False,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.CONTINUE
        assert ictx.consecutive_errors == 3

    def test_infra_under_threshold_continue(self):
        """Infra-Fehler unter Schwelle → CONTINUE + Inkrement."""
        ictx = _make_ictx(consecutive_errors=0)
        result, retries = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="loading",
            error_description="timeout",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.CONTINUE
        assert ictx.consecutive_errors == 1

    def test_threshold_max_retries_skip(self):
        """Schwelle erreicht + >2 Retries → SKIP + Reset."""
        ictx = _make_ictx(consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1)
        result, retries = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="loading",
            error_description="err",
            is_infra=True,
            file_retries=2,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.SKIP
        assert retries == 3
        assert ictx.consecutive_errors == 0

    def test_cli_mode_abort(self):
        """Kein control_callback → ABORT mit abort_reason."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=None,
            file_log_callback=MagicMock(),
        )
        file_log = _make_file_log()
        result, _ = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            file_log,
            error_phase="loading",
            error_description="DDU down",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.ABORT
        assert "abort_reason" in ictx.stats
        assert "Backend-Fehler" in ictx.stats["abort_reason"]

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_worker_retry_with_recovery(self, mock_wait):
        """Worker-Modus: Backend recovered → RETRY."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
        )
        file_log = _make_file_log(
            status="error",
            error_message="DDU timeout",
            error_phase="loading",
            is_infra_error=True,
        )
        result, retries = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            file_log,
            error_phase="loading",
            error_description="timeout",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.RETRY
        # consecutive_errors wird NICHT bei RETRY zurueckgesetzt — erst nach
        # erfolgreichem Retry durch die Success-Pfade (Livelock-Vermeidung)
        assert ictx.consecutive_errors == INGESTION_MAX_CONSECUTIVE_ERRORS
        assert file_log.status == "pending"
        assert file_log.error_message == ""

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=False)
    def test_worker_cancel_during_wait(self, mock_wait):
        """Cancel/Shutdown während Wait → CANCEL."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
        )
        result, _ = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="embedding",
            error_description="err",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.CANCEL

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_stats_correction_loading_ddu_timeout(self, mock_wait):
        """Loading-RETRY korrigiert ddu_timeout Stats."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            stats={"processed_files": 0, "errors": 3, "ddu_timeout": 2},
        )
        file_log = _make_file_log(
            error_message="DDU timeout",
            error_phase="loading",
            is_infra_error=True,
        )
        _handle_circuit_breaker(
            ictx,
            "/test.txt",
            file_log,
            error_phase="loading",
            error_description="timeout",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert ictx.stats["errors"] == 2
        assert ictx.stats["ddu_timeout"] == 1

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_stats_correction_embedding_error_details_pop(self, mock_wait):
        """Embedding-RETRY entfernt letzten error_details Eintrag."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            stats={
                "processed_files": 0,
                "errors": 3,
                "error_details": [{"file": "a"}, {"file": "b"}],
            },
        )
        _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="embedding",
            error_description="err",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert ictx.stats["errors"] == 2
        assert len(ictx.stats["error_details"]) == 1

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_bugfix_is_infra_error_reset_on_retry(self, mock_wait):
        """Bug-Fix: is_infra_error wird bei RETRY immer zurueckgesetzt."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
        )
        file_log = _make_file_log(
            is_infra_error=True,
            status="error",
            error_message="Backend down",
            error_phase="embedding",
        )
        result, _ = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            file_log,
            error_phase="embedding",
            error_description="err",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.RETRY
        assert file_log.is_infra_error is False
        assert file_log.status == "pending"
        assert file_log.error_message == ""
        assert file_log.error_phase == ""

    def test_abort_calls_file_log_callback(self):
        """ABORT ruft file_log_callback auf wenn vorhanden."""
        log_cb = MagicMock()
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=None,
            file_log_callback=log_cb,
        )
        file_log = _make_file_log()
        _handle_circuit_breaker(
            ictx,
            "/test.txt",
            file_log,
            error_phase="loading",
            error_description="err",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic() - 0.01,
        )
        log_cb.assert_called_once()
        assert file_log.duration_ms >= 10


# ===================================================================
# _handle_loading_result
# ===================================================================


class TestHandleLoadingResult:
    def test_infra_error_under_threshold_skip(self):
        """Infra-Fehler unter CB-Schwelle → SKIP + file_log_callback."""
        log_cb = MagicMock()
        ictx = _make_ictx(
            stats={"errors": 1, "processed_files": 0, "total_files": 10},
            file_log_callback=log_cb,
        )
        file_log = _make_file_log(
            is_infra_error=True,
            error_message="DDU down",
            error_phase="loading",
        )
        action, _ = _handle_loading_result(
            ictx,
            "/test.txt",
            file_log,
            errors_before=0,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert action == _FileAction.SKIP
        ictx.hash_store.remove_files.assert_called_once()
        log_cb.assert_called_once()

    def test_content_error_no_consecutive_increment(self):
        """Content-Fehler → SKIP, consecutive_errors NICHT inkrementiert."""
        ictx = _make_ictx(
            stats={"errors": 1, "processed_files": 0, "total_files": 10},
            consecutive_errors=5,
        )
        file_log = _make_file_log(
            is_infra_error=False,
            error_message="Korrupte Datei",
            error_phase="loading",
        )
        action, _ = _handle_loading_result(
            ictx,
            "/test.txt",
            file_log,
            errors_before=0,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert action == _FileAction.SKIP
        assert ictx.consecutive_errors == 5  # unchanged

    def test_no_error_no_content_skip_with_progress(self):
        """Kein Fehler + kein Inhalt → SKIP + progress + file_log.

        Skip-Faelle erhoehen ``skipped``, NICHT ``processed_files`` —
        eine ueberlesene Datei ist keine erfolgreich verarbeitete Datei.
        """
        progress_cb = MagicMock()
        log_cb = MagicMock()
        ictx = _make_ictx(
            stats={"errors": 0, "processed_files": 0, "total_files": 10},
            file_hashes={"/test.txt": ("hash123", 1024)},
            progress_callback=progress_cb,
            file_log_callback=log_cb,
            skipped=3,
            total=10,
        )
        file_log = _make_file_log(status="success")

        action, _ = _handle_loading_result(
            ictx,
            "/test.txt",
            file_log,
            errors_before=0,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert action == _FileAction.SKIP
        assert ictx.skipped == 4
        assert ictx.stats["processed_files"] == 0
        assert ictx.consecutive_errors == 0
        progress_cb.assert_called_once_with(4, 10)
        log_cb.assert_called_once()
        assert file_log.status == "skipped"

    def test_error_updates_progress(self):
        """Error-SKIP aktualisiert progress_callback (Fortschrittsanzeige laeuft weiter)."""
        progress_cb = MagicMock()
        log_cb = MagicMock()
        ictx = _make_ictx(
            stats={"errors": 1, "processed_files": 0, "total_files": 10},
            progress_callback=progress_cb,
            file_log_callback=log_cb,
        )
        file_log = _make_file_log(
            is_infra_error=False,
            error_message="parse error",
            error_phase="loading",
        )
        _handle_loading_result(
            ictx,
            "/test.txt",
            file_log,
            errors_before=0,
            file_retries=0,
            t0=time.monotonic(),
        )
        progress_cb.assert_called_once()
        log_cb.assert_called_once()

    def test_no_file_log_no_callback(self):
        """Ohne file_log_callback: kein Crash."""
        ictx = _make_ictx(
            stats={"errors": 0, "processed_files": 0, "total_files": 10},
            file_hashes={"/test.txt": ("h1", 512)},
            file_log_callback=None,
            progress_callback=None,
        )
        action, _ = _handle_loading_result(
            ictx,
            "/test.txt",
            None,
            errors_before=0,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert action == _FileAction.SKIP


# ===================================================================
# _handle_chunk_dedup
# ===================================================================


class TestHandleChunkDedup:
    def test_no_datasource_id_no_dedup(self):
        ictx = _make_ictx(datasource_id=None)
        result, texts, metas, ids, spans, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1"],
            [{"m": 1}],
            ["id1"],
            [(0, 2)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP
        assert texts == ["t1"]

    def test_empty_file_ids_no_dedup(self):
        ictx = _make_ictx()
        result, _, _, _, _, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            [],
            [],
            [],
            [],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP

    def test_no_existing_chunks(self):
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = set()
        result, texts, _, _, _, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP
        assert len(texts) == 2

    def test_full_dedup(self):
        progress_cb = MagicMock()
        log_cb = MagicMock()
        ictx = _make_ictx(
            file_hashes={"/f.txt": ("h1", 123)},
            progress_callback=progress_cb,
            file_log_callback=log_cb,
        )
        ictx.chunk_ds.chunks_exist.return_value = {"id1", "id2"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1", "id2"]}
        file_log = _make_file_log()

        result, texts, _, _, spans, eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            file_log,
            time.monotonic(),
        )
        assert result == _DedupResult.FULL
        assert texts == []
        assert spans == []
        # Finalisierung (processed_files, callbacks) jetzt beim Caller
        assert ictx.stats["dedup_chunks"] == 2
        assert eb is not None
        assert eb.ids == ["id1", "id2"]

    def test_partial_dedup(self):
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}

        result, texts, metas, ids, spans, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.PARTIAL
        assert texts == ["t2"]
        assert metas == [{"m": 2}]
        assert ids == ["id2"]
        assert spans == [(3, 5)]
        assert ictx.stats["dedup_chunks"] == 1

    def test_chunks_exist_exception_graceful(self):
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.side_effect = Exception("DB error")

        result, texts, _, _, _, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1"],
            [{"m": 1}],
            ["id1"],
            [(0, 2)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP
        assert texts == ["t1"]

    def test_orphan_pg_entry_without_qdrant_point_is_re_embedded(self):
        """Bug H-020: chunk_datasources hat Eintrag, Qdrant-Vektor fehlt."""
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = {"id1", "id2"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}

        result, texts, metas, ids, spans, eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.PARTIAL
        assert texts == ["t2"]
        assert ids == ["id2"]
        assert spans == [(3, 5)]
        assert ictx.stats["dedup_chunks"] == 1
        assert eb is not None and eb.ids == ["id1"]

    def test_all_orphans_re_embed_no_dedup(self):
        """Alle PG-Eintraege sind Orphans -> komplettes Re-Embedding."""
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = {"id1", "id2"}
        ictx.store.get_documents_by_ids.return_value = {"ids": []}

        result, texts, _, ids, _, eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP
        assert ids == ["id1", "id2"]
        assert texts == ["t1", "t2"]
        assert eb is None
        assert ictx.stats.get("dedup_chunks", 0) == 0

    def test_qdrant_verification_exception_falls_back_to_re_embed(self):
        """Generischer Qdrant-Lookup-Fehler -> Re-Embedding (konservativ)."""
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.side_effect = RuntimeError("qdrant glitch")

        result, _, _, ids, _, eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1"],
            [{"m": 1}],
            ["id1"],
            [(0, 2)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.NO_DEDUP
        assert ids == ["id1"]
        assert eb is None

    def test_qdrant_verification_vectordb_connection_error_propagates(self):
        """VectorDBConnectionError waehrend Verifikation -> Caller-Pfad."""
        from kiara.exceptions import VectorDBConnectionError

        ictx = _make_ictx(datasource_id=1)
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.side_effect = VectorDBConnectionError("qdrant unreachable")

        with pytest.raises(VectorDBConnectionError):
            _handle_chunk_dedup(
                ictx,
                "/f.txt",
                ["t1"],
                [{"m": 1}],
                ["id1"],
                [(0, 2)],
                _make_file_log(),
                time.monotonic(),
            )

    def test_full_dedup_returns_existing_batch(self):
        ictx = _make_ictx(
            consecutive_errors=5,
            file_hashes={"/f.txt": ("h1", 123)},
        )
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}

        result, _, _, _, _, eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1"],
            [{"m": 1}],
            ["id1"],
            [(0, 2)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.FULL
        # consecutive_errors reset jetzt beim Caller
        assert eb is not None
        assert eb.ids == ["id1"]
        assert eb.texts == ["t1"]

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids")
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_full_dedup_syncs_es(self, mock_es_configured, mock_es_update):
        """Spec #10: Dedup-Block ruft es_svc.update_ds_ids() auf."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "retrieval": {"hybrid_search": "elasticsearch"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
            },
            file_hashes={"/f.txt": ("h1", 123)},
        )
        ictx.chunk_ds.chunks_exist.return_value = {"id1", "id2"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1", "id2"]}
        ds_map = {"id1": [1, 2], "id2": [1]}
        ictx.chunk_ds.get_datasource_ids_for_chunks.return_value = ds_map

        result, _, _, _, _, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            [(0, 2), (3, 5)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.FULL
        # Qdrant wurde aktualisiert
        ictx.store.update_ds_ids.assert_called_once_with(ds_map, wait=True)
        # ES wurde ebenfalls aktualisiert
        mock_es_update.assert_called_once_with(
            ictx.config, ds_map, refresh=False, skip_unchanged=True
        )

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids", side_effect=Exception("ES down"))
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_dedup_es_failure_cannot_finish_or_reembed_shared_chunks(
        self, mock_es_configured, mock_es_update
    ):
        """Ein ES-Fehler bleibt ein Backendfehler; kein destruktiver Dedup-Fallback."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "retrieval": {"hybrid_search": "elasticsearch"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
            },
            file_hashes={"/f.txt": ("h1", 123)},
        )
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}
        ictx.chunk_ds.get_datasource_ids_for_chunks.return_value = {"id1": [1]}

        with pytest.raises(BackendUnavailableError, match="DS-Zuordnungsabgleich"):
            result, _, _, _, _, _eb = _handle_chunk_dedup(
                ictx,
                "/f.txt",
                ["t1"],
                [{"m": 1}],
                ["id1"],
                [(0, 2)],
                None,
                time.monotonic(),
            )


# ===================================================================
# _handle_embedding_phase
# ===================================================================


class TestHandleEmbeddingPhase:
    @patch(
        "kiara.retrieval.elasticsearch_service.bulk_index_chunks",
        side_effect=RuntimeError("es down"),
    )
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_write_to_elasticsearch_counts_and_reraises(
        self,
        mock_configured,
        mock_bulk,
    ):
        stats = {}

        with pytest.raises(RuntimeError, match="es down"):
            _write_to_elasticsearch(
                {"retrieval": {"hybrid_search": "elasticsearch"}},
                ["text"],
                [{"source": "/f.txt"}],
                ["id1"],
                stats,
            )

        assert stats["es_errors"] == 1
        mock_bulk.assert_called_once()

    @patch(
        "kiara.retrieval.elasticsearch_service.bulk_index_chunks",
        return_value=1,
    )
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    def test_write_to_elasticsearch_partial_bulk_is_fatal(
        self,
        mock_configured,
        mock_bulk,
    ):
        """Partial-Indexing darf nicht als konsistenter Erfolg gelten."""
        stats = {}

        with pytest.raises(RuntimeError, match="unvollstaendig"):
            _write_to_elasticsearch(
                {"retrieval": {"hybrid_search": "elasticsearch"}},
                ["text1", "text2"],
                [{"source": "/f.txt"}, {"source": "/f.txt"}],
                ["id1", "id2"],
                stats,
            )

        assert "es_indexed" not in stats
        assert stats["es_errors"] == 1
        mock_bulk.assert_called_once()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_success(self, mock_write, mock_es, mock_graph):
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)})
        action, retries = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.SUCCESS
        assert ictx.stats["total_chunks"] == 1
        assert ictx.stats["processed_files"] == 0
        ictx.file_completion_callback.assert_not_called()
        assert ictx.consecutive_errors == 0
        mock_write.assert_called_once()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_cancelled_error(self, mock_write, mock_es, mock_graph):
        mock_write.side_effect = _CancelledError()
        log_cb = MagicMock()
        ictx = _make_ictx(file_log_callback=log_cb)
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.CANCEL
        log_cb.assert_called_once()

    def test_index_compatibility_error_aborts_after_safe_partial_rollback(self):
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexVectorKind,
            make_index_target,
        )

        target = make_index_target(
            {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
            collection_name="kb_guarded",
            vector_kind=IndexVectorKind.DENSE,
            requested_model="guarded-model",
        )
        error = IndexCompatibilityError(
            "index_contract_unverified",
            "document_embedding",
            target,
            "stored contract is unverified",
            stored_state="unverified",
            current_state="valid",
        )
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)})
        file_log = _make_file_log("/f.txt")

        with (
            patch(
                "kiara.ingestion.runtime._write_embeddings",
                return_value=["id1"],
            ) as write,
            patch(
                "kiara.ingestion.runtime._write_to_elasticsearch",
                side_effect=error,
            ),
            patch(
                "kiara.ingestion.runtime._rollback_elasticsearch_write",
                return_value=0,
            ),
            patch("kiara.ingestion.runtime._persist_graph_and_associations") as graph,
            patch("kiara.ingestion.runtime._persist_hashes_checkpoint") as checkpoint,
        ):
            action, _ = _handle_embedding_phase(
                ictx,
                "/f.txt",
                ["text"],
                [{"source": "/f.txt"}],
                ["id1"],
                file_log,
                0,
                time.monotonic(),
            )

        assert action == _FileAction.ABORT
        assert "index_contract_unverified" in ictx.stats["abort_reason"]
        assert ictx.stats["index_compatibility_error"] == error.to_dict()
        assert ictx.stats["error_details"][-1]["reason_code"] == error.reason_code
        assert ictx.stats["error_details"][-1]["diagnostic"] == error.to_dict()
        assert file_log.error_phase == "index_compatibility"
        assert ictx.stats["processed_files"] == 0
        assert ictx.stats["total_chunks"] == 0
        write.assert_called_once()
        graph.assert_not_called()
        ictx.store.delete_by_ids.assert_called_once_with(["id1"])
        ictx.hash_store.remove_files.assert_called_once_with(
            ictx.datasource_id,
            ictx.embedding_model,
            ["/f.txt"],
        )
        checkpoint.assert_called_once_with(ictx, force=True)

    @pytest.mark.parametrize(
        "reason_code",
        [
            "index_contract_missing",
            "index_contract_unverified",
            "index_compatibility_id_mismatch",
            "index_contract_evidence_failed",
        ],
    )
    def test_late_contract_error_never_falls_back_to_dense_or_success(self, reason_code):
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexVectorKind,
            make_index_target,
        )

        target = make_index_target(
            {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
            collection_name="kb_guarded_late",
            vector_kind=IndexVectorKind.LATE_CHUNKING,
            requested_model="guarded-model",
        )
        error = IndexCompatibilityError(
            reason_code,
            "late_document_embedding",
            target,
            "injected late-contract failure",
            current_state="unverified",
        )
        ictx = _make_ictx(file_hashes={"/late.txt": ("h1", 123)})

        with (
            patch("kiara.ingestion.runtime._should_use_late_chunking", return_value=True),
            patch("kiara.ingestion.runtime._late_chunking_file_eligible", return_value=True),
            patch("kiara.ingestion.runtime.embed_texts_late", side_effect=error) as late,
            patch("kiara.ingestion.runtime._write_embeddings") as dense,
            patch("kiara.ingestion.runtime._write_to_elasticsearch") as elasticsearch,
            patch("kiara.ingestion.runtime._persist_graph_and_associations") as graph,
        ):
            action, _ = _handle_embedding_phase(
                ictx,
                "/late.txt",
                ["text"],
                [{"source": "/late.txt"}],
                ["id1"],
                _make_file_log("/late.txt"),
                0,
                time.monotonic(),
            )

        assert action == _FileAction.ABORT
        assert reason_code in ictx.stats["abort_reason"]
        assert ictx.stats["index_compatibility_error"]["reason_code"] == reason_code
        assert ictx.stats["processed_files"] == 0
        assert ictx.stats["total_chunks"] == 0
        late.assert_called_once()
        dense.assert_not_called()
        elasticsearch.assert_not_called()
        graph.assert_not_called()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_categorical_error_under_threshold(self, mock_write, mock_es, mock_graph):
        mock_write.side_effect = BackendUnavailableError("no backend")
        ictx = _make_ictx(consecutive_errors=0)
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.SKIP
        assert ictx.stats["errors"] == 1
        assert file_log.status == "error"
        assert file_log.error_phase == "embedding"
        ictx.hash_store.remove_files.assert_called_once()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_generic_exception_skip_resets_consecutive(self, mock_write, mock_es, mock_graph):
        mock_write.side_effect = ValueError("bad data")
        ictx = _make_ictx(consecutive_errors=5)
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.SKIP
        assert ictx.stats["errors"] == 1
        assert ictx.consecutive_errors == 0
        assert file_log.status == "error"
        assert file_log.error_phase == "embedding"
        ictx.hash_store.remove_files.assert_called_once()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_elasticsearch_write_error_does_not_persist_hash(
        self,
        mock_write,
        mock_es,
        mock_graph,
    ):
        mock_write.return_value = ["id1"]
        mock_es.side_effect = RuntimeError("es down")
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)})
        _mock_rollback_documents(ictx, ["id1"])
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )

        assert action == _FileAction.SKIP
        assert ictx.stats["processed_files"] == 0
        assert ictx.pending_hashes == {}
        assert ictx.stats["errors"] == 1
        assert file_log.status == "error"
        assert file_log.error_phase == "embedding"
        ictx.hash_store.remove_files.assert_called_once_with(
            ictx.datasource_id,
            ictx.embedding_model,
            ["/f.txt"],
        )
        ictx.store.delete_by_ids.assert_called_once_with(["id1"])
        mock_graph.assert_not_called()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_elasticsearch_write_error_rollback_failure_aborts(
        self,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """Wenn Qdrant-Rollback fehlschlaegt, wird fail-closed abgebrochen."""
        from kiara.exceptions import VectorDBConnectionError

        mock_write.return_value = ["id1"]
        mock_es.side_effect = RuntimeError("es down")
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)})
        ictx.store.delete_by_ids.side_effect = VectorDBConnectionError("qdrant down")
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )

        assert action == _FileAction.ABORT
        assert "Rollback" in ictx.stats["abort_reason"]
        assert ictx.stats["rollback_errors"] == 1
        assert ictx.stats["processed_files"] == 0
        ictx.hash_store.remove_files.assert_called_once_with(
            ictx.datasource_id,
            ictx.embedding_model,
            ["/f.txt"],
        )
        mock_graph.assert_not_called()

    @patch("kiara.ingestion.runtime._rollback_db_associations", return_value=1)
    @patch("kiara.ingestion.runtime._rollback_elasticsearch_write", return_value=0)
    @patch("kiara.ingestion.runtime._persist_graph_and_associations")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch", return_value=1)
    @patch("kiara.ingestion.runtime._write_embeddings", return_value=["id1"])
    def test_late_persistence_error_rolls_back_db_es_and_vectors(
        self,
        mock_write,
        mock_es,
        mock_persist,
        mock_es_rollback,
        mock_db_rollback,
    ):
        """Fehler nach VectorDB+ES loescht alle bereits geschriebenen Ebenen."""
        mock_persist.side_effect = RuntimeError("db association failed")
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)}, datasource_id=42)
        _mock_rollback_documents(ictx, ["id1"])
        file_log = _make_file_log()

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )

        assert action == _FileAction.SKIP
        mock_db_rollback.assert_called_once()
        mock_es_rollback.assert_called_once()
        ictx.store.delete_by_ids.assert_called_once_with(["id1"])
        assert ictx.stats["db_rollback_chunks"] == 1
        assert ictx.stats["vectordb_rollback_chunks"] == 1
        assert ictx.stats["processed_files"] == 0
        assert ictx.pending_hashes == {}

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_success_with_ds_association(self, mock_write, mock_es, mock_graph):
        """Erfolg mit datasource_id → add_associations aufgerufen."""
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)}, datasource_id=42)
        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.SUCCESS
        ictx.chunk_ds.add_associations.assert_called_once_with(
            ["id1"],
            42,
            "test-model",
            conn=ANY,
            lock_already_held=True,
        )
        ictx.chunk_ds.acquire_graph_cleanup_lock.assert_called_once_with(ANY)

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_success_without_ds_no_association(self, mock_write, mock_es, mock_graph):
        """Erfolg ohne datasource_id → kein add_associations."""
        ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 123)}, datasource_id=None)
        _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        ictx.chunk_ds.add_associations.assert_not_called()

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_cancelled_flushes_hashes(self, mock_write, mock_es, mock_graph):
        """CancelledError flusht pending_hashes."""
        mock_write.side_effect = _CancelledError()
        pending = {"prev.txt": ("old_hash", 100)}
        ictx = _make_ictx(pending_hashes=pending)

        _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        ictx.hash_store.upsert_many.assert_called_once()

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids")
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_embed_syncs_es(
        self, mock_write, mock_es_write, mock_graph, mock_es_configured, mock_es_update
    ):
        """Spec #11: Embed-Block ruft es_svc.update_ds_ids() auf."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "retrieval": {"hybrid_search": "elasticsearch"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
            },
            file_hashes={"/f.txt": ("h1", 123)},
            datasource_id=42,
        )
        ds_map = {"id1": [42]}
        ictx.chunk_ds.get_datasource_ids_for_chunks.return_value = ds_map

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        assert action == _FileAction.SUCCESS
        # Qdrant
        ictx.store.update_ds_ids.assert_called_once_with(ds_map, wait=True)
        # ES
        mock_es_update.assert_called_once_with(
            ictx.config, ds_map, refresh=False, skip_unchanged=True
        )

    @patch("kiara.retrieval.elasticsearch_service.update_ds_ids", side_effect=Exception("ES down"))
    @patch("kiara.retrieval.elasticsearch_service.is_configured", return_value=True)
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    def test_embed_es_failure_cannot_persist_success_hash(
        self, mock_write, mock_es_write, mock_graph, mock_es_configured, mock_es_update
    ):
        """ES-Fehler muessen Datei-Erfolg und Hash-Checkpoint verhindern."""
        ictx = _make_ictx(
            config={
                "vectordb": {"path": "/tmp"},
                "retrieval": {"hybrid_search": "elasticsearch"},
                "ingestion": {"batch_size": 10, "chunk_size": 500, "chunk_overlap": 50},
            },
            file_hashes={"/f.txt": ("h1", 123)},
            datasource_id=42,
        )
        ictx.chunk_ds.get_datasource_ids_for_chunks.return_value = {"id1": [42]}

        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"m": 1}],
            ["id1"],
            None,
            0,
            time.monotonic(),
        )
        assert action != _FileAction.SUCCESS
        assert ictx.stats["total_chunks"] == 0
        assert "/f.txt" not in ictx.pending_hashes


# ===================================================================
# _finalize_file
# ===================================================================


class TestFinalizeFile:
    def test_both_callbacks_called(self):
        log_cb = MagicMock()
        progress_cb = MagicMock()
        ictx = _make_ictx(
            file_log_callback=log_cb,
            progress_callback=progress_cb,
            stats={"processed_files": 5},
            skipped=3,
            total=20,
        )
        file_log = _make_file_log()

        _finalize_file(ictx, file_log, time.monotonic() - 0.05)

        log_cb.assert_called_once()
        progress_cb.assert_called_once_with(8, 20)
        assert file_log.duration_ms > 0

    def test_no_file_log_only_progress(self):
        progress_cb = MagicMock()
        log_cb = MagicMock()
        ictx = _make_ictx(
            progress_callback=progress_cb,
            file_log_callback=log_cb,
            stats={"processed_files": 0},
            skipped=0,
            total=5,
        )
        _finalize_file(ictx, None, time.monotonic())
        log_cb.assert_not_called()
        progress_cb.assert_called_once()

    def test_no_callbacks_no_crash(self):
        ictx = _make_ictx(file_log_callback=None, progress_callback=None)
        _finalize_file(ictx, None, time.monotonic())


# ===================================================================
# _process_file_in_pipeline (Integration)
# ===================================================================


class TestProcessFileInPipeline:
    def test_cancel_at_start(self):
        """control_callback gibt 'cancel' → sofortiger return False."""
        ictx = _make_ictx(control_callback=MagicMock(return_value="cancel"))
        assert _process_file_in_pipeline(ictx, "/f.txt") is False

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_success_e2e(self, mock_psf, mock_write, mock_es, mock_graph):
        """Loading + Embedding Erfolg → return True + Stats korrekt."""
        progress_cb = MagicMock()
        log_cb = MagicMock()
        ictx = _make_ictx(
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=log_cb,
            progress_callback=progress_cb,
            datasource_id=None,
        )
        mock_psf.return_value = (["text"], [{"source": "/f.txt"}], ["id1"], "full text", [(0, 4)])

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        assert ictx.stats["processed_files"] == 1
        assert ictx.stats["total_chunks"] == 1
        mock_write.assert_called_once()
        progress_cb.assert_called_once()
        log_cb.assert_called_once()

    def test_run_ingestion_contract_abort_stops_before_next_file_and_writes(self, tmp_path):
        from kiara.ingestion.pipeline import run_ingestion
        from kiara.retrieval.index_compatibility import (
            IndexCompatibilityError,
            IndexVectorKind,
            make_index_target,
        )

        files = [str(tmp_path / name) for name in ("first.txt", "second.txt")]
        for path in files:
            Path(path).write_text("chunk")
        config = {
            "datasource": {
                "path": str(tmp_path),
                "file_types": [".txt"],
                "exclude_patterns": [],
            },
            "vectordb": {"path": str(tmp_path / "vectors")},
            "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
            "graph": {"enabled": False},
        }
        hash_store = MagicMock()
        store = MagicMock()
        chunk_ds = MagicMock()
        chunk_ds.chunks_exist.return_value = set()
        target = make_index_target(
            {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
            collection_name="kb_worker_abort",
            vector_kind=IndexVectorKind.DENSE,
            requested_model="guarded-model",
        )
        error = IndexCompatibilityError(
            "index_contract_missing",
            "document_embedding",
            target,
            "no persisted contract",
            current_state="missing",
        )

        with (
            patch(
                "kiara.ingestion.pipeline._load_and_diff_files",
                return_value=(
                    files,
                    files,
                    {path: (f"hash-{index}", 100) for index, path in enumerate(files)},
                    [],
                    hash_store,
                    {},
                ),
            ),
            patch("kiara.ingestion.pipeline.VectorStore", return_value=store),
            patch("kiara.ingestion.pipeline.ChunkDatasources", return_value=chunk_ds),
            patch("kiara.ingestion.pipeline.get_embedding_model", return_value="guarded-model"),
            patch(
                "kiara.ingestion.runtime._process_single_file",
                return_value=(
                    ["chunk"],
                    [{"source": files[0]}],
                    ["id1"],
                    "full text",
                    [(0, 5)],
                ),
            ) as process_file,
            patch(
                "kiara.ingestion.runtime._write_embeddings",
                side_effect=error,
            ) as guarded_write,
            patch("kiara.ingestion.runtime._write_to_elasticsearch") as elasticsearch,
            patch("kiara.ingestion.runtime._persist_graph_and_associations") as graph,
        ):
            stats = run_ingestion(config, embedding_model="guarded-model")

        assert "index_contract_missing" in stats["abort_reason"]
        assert stats["index_compatibility_error"] == error.to_dict()
        assert stats["processed_files"] == 0
        assert process_file.call_count == 1
        guarded_write.assert_called_once()
        store.add.assert_not_called()
        elasticsearch.assert_not_called()
        graph.assert_not_called()

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_loading_retry_e2e(self, mock_psf, mock_write, mock_es, mock_graph, mock_wait):
        """Loading-Error → CB RETRY → Recovery → Re-Load → Erfolg."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            datasource_id=None,
        )

        call_count = [0]

        def psf_side_effect(path, config, overrides, ds_id, ctx, stats, file_log=None):
            call_count[0] += 1
            if call_count[0] == 1:
                stats["errors"] += 1
                if file_log:
                    file_log.status = "error"
                    file_log.error_message = "DDU timeout"
                    file_log.error_phase = "loading"
                    file_log.is_infra_error = True
                return [], [], [], "", []
            return ["text"], [{"source": path}], ["id1"], "full text", [(0, 4)]

        mock_psf.side_effect = psf_side_effect

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        assert mock_psf.call_count == 2
        assert ictx.stats["processed_files"] == 1
        assert ictx.file_log_callback.call_args.args[0].status == "success"
        mock_wait.assert_called_once()

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_embedding_retry_skip_loading(
        self, mock_psf, mock_write, mock_es, mock_graph, mock_wait
    ):
        """Embedding-Error → CB RETRY → skip_loading=True → nur Embedding erneut."""
        ictx = _make_ictx(
            consecutive_errors=INGESTION_MAX_CONSECUTIVE_ERRORS - 1,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            datasource_id=None,
        )
        mock_psf.return_value = (["text"], [{"source": "/f.txt"}], ["id1"], "full text", [(0, 4)])
        mock_write.side_effect = [BackendUnavailableError("no backend"), None]

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        # _process_single_file nur EINMAL aufgerufen (skip_loading beim Retry)
        assert mock_psf.call_count == 1
        # _write_embeddings zweimal: erst Fehler, dann Erfolg
        assert mock_write.call_count == 2
        mock_wait.assert_called_once()

    @patch("kiara.ingestion.runtime.INGESTION_MAX_CONSECUTIVE_ERRORS", 1)
    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_double_retry_then_skip(self, mock_psf, mock_write, mock_es, mock_graph, mock_wait):
        """3x Loading-Error → 2 Retries → Max Retries → SKIP → return True."""
        ictx = _make_ictx(
            consecutive_errors=0,
            control_callback=MagicMock(return_value=None),
            status_callback=MagicMock(),
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            datasource_id=None,
        )

        def psf_always_fail(path, config, overrides, ds_id, ctx, stats, file_log=None):
            stats["errors"] += 1
            if file_log:
                file_log.status = "error"
                file_log.error_message = "DDU down"
                file_log.error_phase = "loading"
                file_log.is_infra_error = True
            return [], [], [], "", []

        mock_psf.side_effect = psf_always_fail

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True  # SKIP, nicht Abort
        assert mock_psf.call_count == 3  # 1 Original + 2 Retries
        assert mock_wait.call_count == 2  # 2x Recovery-Wait

    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_partial_dedup_then_cancelled(self, mock_psf, mock_write, mock_es, mock_graph):
        """Partial-Dedup → Embedding CancelledError → return False + Hashes geflusht."""
        ictx = _make_ictx(
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            datasource_id=1,
            pending_hashes={"prev.txt": ("old_hash", 100)},  # vorherige Datei
        )
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}
        mock_psf.return_value = (
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            "full text",
            [(0, 2), (3, 5)],
        )
        mock_write.side_effect = _CancelledError()

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is False
        assert ictx.stats.get("dedup_chunks") == 1  # id1 deduped
        ictx.hash_store.upsert_many.assert_called()  # pending Hashes geflusht


# ===================================================================
# _should_use_late_chunking
# ===================================================================


class TestShouldUseLateChuking:
    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m)
    def test_should_use_late_chunking_enabled(self, mock_normalize):
        """enabled=True und gueltige Identitaet aktiviert die Vertragspruefung."""
        model = "contract-model"
        config = {"enrichment": {"late_chunking": True}}
        assert _should_use_late_chunking(config, model) is True

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m)
    def test_should_use_late_chunking_disabled(self, mock_normalize):
        """enabled=False -> False."""
        model = "contract-model"
        config = {"enrichment": {"late_chunking": False}}
        assert _should_use_late_chunking(config, model) is False

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m)
    def test_should_use_late_chunking_has_no_model_allowlist(self, mock_normalize):
        """Modellnamen entscheiden nicht mehr ueber die Late-Faehigkeit."""
        config = {"enrichment": {"late_chunking": True}}
        assert _should_use_late_chunking(config, "previously-unknown-model") is True

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m)
    def test_should_use_late_chunking_defers_provider_to_contract_guard(self, mock_normalize):
        """Providerheuristiken werden nicht vor die Profilpruefung gezogen."""
        model = "contract-model"
        config = {"enrichment": {"late_chunking": True}}
        assert _should_use_late_chunking(config, model) is True

    def test_should_use_late_chunking_no_model(self):
        """model=None -> False."""
        config = {"enrichment": {"late_chunking": True}}
        assert _should_use_late_chunking(config, None) is False

    @patch("kiara.backend_selector.normalize_model_name", side_effect=lambda m: m)
    def test_should_use_late_chunking_does_not_preselect_backend(self, mock_normalize):
        """Die Backend-/Profilpruefung bleibt am atomaren Embedding-Guard."""
        model = "contract-model"
        config = {"enrichment": {"late_chunking": True}}
        assert _should_use_late_chunking(config, model) is True

    def test_should_use_late_chunking_enrichment_none(self):
        """config={'enrichment': None} -> False."""
        config = {"enrichment": None}
        assert _should_use_late_chunking(config, "nomic-embed-text") is False


# ===================================================================
# _late_chunking_file_eligible
# ===================================================================


class TestLateChunkingFileEligible:
    def test_late_chunking_file_eligible_basic(self):
        """Valid full_text, chunks, spans, no VLM -> True."""
        assert (
            _late_chunking_file_eligible(
                "hello world",
                ["hello", "world"],
                [(0, 5), (6, 11)],
                [{"parser": "docling-remote"}, {"parser": "docling-remote"}],
            )
            is True
        )

    def test_late_chunking_file_eligible_vlm(self):
        """parser=vlm -> False."""
        assert (
            _late_chunking_file_eligible(
                "hello world",
                ["hello", "world"],
                [(0, 5), (6, 11)],
                [{"parser": "vlm"}, {"parser": "docling-remote"}],
            )
            is False
        )

    def test_late_chunking_file_eligible_empty(self):
        """empty full_text -> False."""
        assert (
            _late_chunking_file_eligible(
                "",
                ["hello"],
                [(0, 5)],
                [{"parser": "docling-remote"}],
            )
            is False
        )

    def test_late_chunking_file_eligible_mismatch(self):
        """len(chunks) != len(spans) -> False."""
        assert (
            _late_chunking_file_eligible(
                "hello world",
                ["hello", "world"],
                [(0, 5)],
                [{"parser": "docling-remote"}, {"parser": "docling-remote"}],
            )
            is False
        )


# ===================================================================
# _refresh_late_chunking_snapshots
# ===================================================================


class TestRefreshLateChunkingSnapshots:
    def test_refresh_late_chunking_snapshots_first_pass(self):
        """First pass builds ID maps from pre-enrichment chunks."""
        ictx = _make_ictx()
        file_ids = ["id1", "id2"]
        file_texts = ["chunk1", "chunk2"]
        file_spans = [(0, 6), (7, 13)]

        _refresh_late_chunking_snapshots(
            ictx,
            file_ids,
            file_texts,
            file_spans,
            enrichment_applied=False,
        )

        assert ictx.last_original_chunks_by_id == {"id1": "chunk1", "id2": "chunk2"}
        assert ictx.last_original_spans_by_id == {"id1": (0, 6), "id2": (7, 13)}
        assert ictx.last_original_chunks == ["chunk1", "chunk2"]
        assert ictx.last_original_spans == [(0, 6), (7, 13)]

    def test_refresh_late_chunking_snapshots_retry(self):
        """Retry reconstructs from ID maps after partial dedup."""
        ictx = _make_ictx()
        # Simulate first pass already done
        ictx.last_original_chunks_by_id = {"id1": "chunk1", "id2": "chunk2", "id3": "chunk3"}
        ictx.last_original_spans_by_id = {"id1": (0, 6), "id2": (7, 13), "id3": (14, 20)}

        # After partial dedup, only id2 and id3 remain
        _refresh_late_chunking_snapshots(
            ictx,
            ["id2", "id3"],
            ["enriched2", "enriched3"],
            [(7, 13), (14, 20)],
            enrichment_applied=True,
        )

        assert ictx.last_original_chunks == ["chunk2", "chunk3"]
        assert ictx.last_original_spans == [(7, 13), (14, 20)]

    def test_refresh_late_chunking_snapshots_incomplete(self):
        """Missing key clears all snapshots."""
        ictx = _make_ictx()
        ictx.last_original_chunks_by_id = {"id1": "chunk1"}
        ictx.last_original_spans_by_id = {"id1": (0, 6)}

        # id_missing is not in the maps
        _refresh_late_chunking_snapshots(
            ictx,
            ["id_missing"],
            ["text"],
            [(0, 4)],
            enrichment_applied=True,
        )

        assert ictx.last_original_chunks is None
        assert ictx.last_original_spans is None
        assert ictx.last_original_chunks_by_id is None
        assert ictx.last_original_spans_by_id is None


# ===================================================================
# _write_embeddings
# ===================================================================


class TestWriteEmbeddings:
    def test_write_embeddings_length_guard(self):
        """embeddings too short/long -> ValueError."""
        store = MagicMock()
        with pytest.raises(ValueError, match="length mismatch"):
            _write_embeddings(
                ["t1", "t2"],
                [{"m": 1}, {"m": 2}],
                ["id1", "id2"],
                store,
                batch_size=10,
                embeddings=[[0.1, 0.2]],
            )

    def test_write_embeddings_with_embeddings(self):
        """embeddings provided -> store.add_with_embeddings() called."""
        store = MagicMock()
        embs = [[0.1, 0.2], [0.3, 0.4]]
        written = _write_embeddings(
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            store,
            batch_size=10,
            embeddings=embs,
        )
        assert written == ["id1", "id2"]
        store.add_with_embeddings.assert_called_once_with(
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            embs,
        )
        store.add.assert_not_called()

    def test_write_embeddings_without_embeddings(self):
        """embeddings=None -> store.add() called."""
        store = MagicMock()
        written = _write_embeddings(
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
            store,
            batch_size=10,
            embeddings=None,
        )
        assert written == ["id1", "id2"]
        store.add.assert_called_once_with(
            ["t1", "t2"],
            [{"m": 1}, {"m": 2}],
            ["id1", "id2"],
        )
        store.add_with_embeddings.assert_not_called()

    def test_write_embeddings_partial_batch_failure_rolls_back_written_ids(self):
        """Fehler in Batch N rollt erfolgreich geschriebene Vorgaenger zurueck."""
        from kiara.exceptions import VectorDBConnectionError

        store = MagicMock()
        store.add.side_effect = [None, VectorDBConnectionError("qdrant down")]

        with pytest.raises(VectorDBConnectionError, match="qdrant down"):
            _write_embeddings(
                ["t1", "t2"],
                [{"m": 1}, {"m": 2}],
                ["id1", "id2"],
                store,
                batch_size=1,
                embeddings=None,
            )

        store.delete_by_ids.assert_called_once_with(["id1"])

    def test_write_embeddings_partial_batch_rollback_failure_raises_abort_error(self):
        """Rollback-Fehler wird als eigener harter Fehler signalisiert."""
        from kiara.exceptions import VectorDBConnectionError

        store = MagicMock()
        store.add.side_effect = [None, VectorDBConnectionError("qdrant down")]
        store.delete_by_ids.side_effect = RuntimeError("rollback down")

        with pytest.raises(_RollbackFailedError, match="Partial-Write"):
            _write_embeddings(
                ["t1", "t2"],
                [{"m": 1}, {"m": 2}],
                ["id1", "id2"],
                store,
                batch_size=1,
                embeddings=None,
            )


# ===================================================================
# _handle_chunk_dedup with spans (Late Chunking)
# ===================================================================


class TestHandleChunkDedupWithSpans:
    def test_handle_chunk_dedup_with_spans(self):
        """Partial dedup filters spans synchronously."""
        ictx = _make_ictx()
        ictx.chunk_ds.chunks_exist.return_value = {"id1", "id3"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1", "id3"]}

        result, texts, metas, ids, spans, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            ["t1", "t2", "t3", "t4"],
            [{"m": 1}, {"m": 2}, {"m": 3}, {"m": 4}],
            ["id1", "id2", "id3", "id4"],
            [(0, 2), (3, 5), (6, 8), (9, 11)],
            None,
            time.monotonic(),
        )
        assert result == _DedupResult.PARTIAL
        assert texts == ["t2", "t4"]
        assert metas == [{"m": 2}, {"m": 4}]
        assert ids == ["id2", "id4"]
        assert spans == [(3, 5), (9, 11)]
        assert ictx.stats["dedup_chunks"] == 2


# ===================================================================
# Spec #15: Partial Dedup + Late Chunking 5->3
# ===================================================================


class TestPartialDedupLateCunking:
    def test_partial_dedup_late_chunking_5_to_3(self):
        """5 chunks with spans, 2 deduped, 3 remaining — snapshots match."""
        ictx = _make_ictx(
            datasource_id=1,
            embedding_model="nomic-embed-text",
        )
        all_ids = ["id1", "id2", "id3", "id4", "id5"]
        all_texts = ["t1", "t2", "t3", "t4", "t5"]
        all_metas = [{"m": i} for i in range(5)]
        all_spans = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14)]

        # Mock: id2 and id4 already exist -> dedup those
        ictx.chunk_ds.chunks_exist.return_value = {"id2", "id4"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id2", "id4"]}
        ictx.chunk_ds.add_associations.return_value = None
        ictx.chunk_ds.get_datasource_ids_for_chunks.return_value = {
            "id2": [1],
            "id4": [1],
        }

        result, texts, metas, ids, spans, _eb = _handle_chunk_dedup(
            ictx,
            "/f.txt",
            all_texts,
            all_metas,
            all_ids,
            all_spans,
            None,
            time.monotonic(),
        )

        assert result == _DedupResult.PARTIAL
        assert len(texts) == 3
        assert len(metas) == 3
        assert len(ids) == 3
        assert len(spans) == 3

        # Remaining should be the non-deduped items
        assert texts == ["t1", "t3", "t5"]
        assert ids == ["id1", "id3", "id5"]
        assert spans == [(0, 2), (6, 8), (12, 14)]

        # Now call _refresh_late_chunking_snapshots with the 3 remaining
        _refresh_late_chunking_snapshots(
            ictx,
            ids,
            texts,
            spans,
            enrichment_applied=False,
        )

        assert ictx.last_original_chunks is not None
        assert len(ictx.last_original_chunks) == 3
        assert ictx.last_original_chunks == ["t1", "t3", "t5"]

        assert ictx.last_original_spans is not None
        assert len(ictx.last_original_spans) == 3
        assert ictx.last_original_spans == [(0, 2), (6, 8), (12, 14)]


# ===================================================================
# Spec #23: Legacy mocks without char_start/char_end
# ===================================================================


class TestChunkAndCollectLegacyMocks:
    @patch("kiara.ingestion.runtime._determine_chunks")
    def test_chunk_and_collect_legacy_mocks_no_spans(self, mock_determine):
        """Chunks in old format (no char_start/char_end) produce None spans."""
        mock_determine.return_value = [
            {
                "text": "hello",
                "metadata": {
                    "source": "f.txt",
                    "parser": "standard",
                    "chunk_index": 0,
                },
            },
        ]
        docs = [{"text": "hello world", "metadata": {"source": "f.txt"}}]
        file_texts, file_metas, file_ids, file_spans = _chunk_and_collect(
            docs,
            None,
            1000,
            200,
        )

        assert len(file_texts) == 1
        assert file_texts[0] == "hello"
        assert len(file_spans) == 1
        assert file_spans[0] is None  # No char_start/char_end -> None


# ===================================================================
# Spec #10: ictx.last_full_text correctly set
# ===================================================================


class TestProcessFileInPipelineSetsLastFullText:
    @patch("kiara.ingestion.runtime._extract_graph_entities")
    @patch("kiara.ingestion.runtime._write_to_elasticsearch")
    @patch("kiara.ingestion.runtime._write_embeddings")
    @patch("kiara.ingestion.runtime._process_single_file")
    def test_process_file_in_pipeline_sets_last_full_text(
        self,
        mock_psf,
        mock_write,
        mock_es,
        mock_graph,
    ):
        """_process_file_in_pipeline stores full_text in ictx.last_full_text."""
        full_doc_text = "This is the full document"
        ictx = _make_ictx(
            file_hashes={"/f.txt": ("h1", 123)},
            file_log_callback=MagicMock(),
            progress_callback=MagicMock(),
            datasource_id=None,
        )
        mock_psf.return_value = (
            ["chunk1"],
            [{"source": "/f.txt"}],
            ["id1"],
            full_doc_text,
            [(0, 6)],
        )

        result = _process_file_in_pipeline(ictx, "/f.txt")

        assert result is True
        assert ictx.last_full_text == full_doc_text


# ===================================================================
# VectorDBConnectionError Circuit-Breaker (V35 #734)
# ===================================================================


class TestVectorDBCircuitBreaker:
    """Runtime-Branching fuer VectorDBConnectionError + Stats-Korrektur."""

    def test_check_vectordb_health_ok(self):
        from kiara.ingestion.runtime import _check_vectordb_health

        with patch("qdrant_client.QdrantClient") as MockClient:
            client = MagicMock()
            client.get_collections.return_value = MagicMock()
            MockClient.return_value = client
            cfg = {"vectordb": {"qdrant_url": "http://q:6333", "qdrant_api_key": "x"}}
            assert _check_vectordb_health(cfg) is True
            MockClient.assert_called_once_with(
                url="http://q:6333",
                api_key="x",
                timeout=2,
            )

    def test_check_vectordb_health_none_vectordb_block(self):
        from kiara.ingestion.runtime import _check_vectordb_health
        from kiara.shared.constants import DEFAULT_QDRANT_URL

        with patch("qdrant_client.QdrantClient") as MockClient:
            client = MagicMock()
            client.get_collections.return_value = MagicMock()
            MockClient.return_value = client
            assert _check_vectordb_health({"vectordb": None}) is True
            kwargs = MockClient.call_args.kwargs
            assert kwargs["url"] == DEFAULT_QDRANT_URL
            assert kwargs["api_key"] is None

    def test_check_vectordb_health_exception_returns_false(self):
        from kiara.ingestion.runtime import _check_vectordb_health

        with patch("qdrant_client.QdrantClient") as MockClient:
            client = MagicMock()
            client.get_collections.side_effect = RuntimeError("timeout")
            MockClient.return_value = client
            assert _check_vectordb_health({"vectordb": {}}) is False

    def test_wait_for_backend_vectordb_path(self):
        from kiara.ingestion.runtime import _wait_for_backend

        with (
            patch(
                "kiara.ingestion.runtime._check_vectordb_health", return_value=True
            ) as mock_health,
            patch("kiara.ingestion.runtime.BACKEND_WAIT_INTERVAL_SECS", 0),
            patch("time.sleep"),
        ):
            ok = _wait_for_backend(
                config={"vectordb": {}},
                ctx=MagicMock(),
                control_callback=None,
                status_callback=None,
                error_type="vectordb",
                embedding_model=None,
            )
        assert ok is True
        mock_health.assert_called_once()

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_circuit_breaker_vectordb_waits_immediately(self, mock_wait):
        """error_phase=vectordb wartet schon beim ersten Fehler, nicht erst ab Schwelle."""
        ictx = _make_ictx(
            consecutive_errors=0,
            stats={
                "errors": 1,
                "vectordb_errors": 1,
                "error_details": [{"file": "x", "phase": "vectordb"}],
            },
            control_callback=lambda: None,
        )
        result, retries = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="vectordb",
            error_description="VectorDB-Fehler: qdrant down",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.RETRY
        assert retries == 1
        # _wait_for_backend wurde mit error_type=vectordb gerufen
        assert mock_wait.call_args.kwargs["error_type"] == "vectordb"
        # Stats-Korrektur
        assert ictx.stats["errors"] == 0
        assert ictx.stats["vectordb_errors"] == 0
        assert ictx.stats["error_details"] == []
        # vectordb_recoveries hochgezählt
        assert ictx.stats.get("vectordb_recoveries") == 1

    @patch("kiara.ingestion.runtime._wait_for_backend", return_value=True)
    def test_circuit_breaker_vectordb_colbert_also_corrects_colbert_errors(self, mock_wait):
        ictx = _make_ictx(
            consecutive_errors=0,
            stats={
                "errors": 1,
                "vectordb_errors": 1,
                "colbert_errors": 1,
                "error_details": [{"file": "x", "phase": "vectordb"}],
            },
            control_callback=lambda: None,
        )
        result, _ = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="vectordb",
            error_description="ColBERT-VectorDB-Fehler",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
            is_colbert=True,
        )
        assert result == _CircuitBreakerResult.RETRY
        assert ictx.stats["colbert_errors"] == 0
        assert ictx.stats["vectordb_errors"] == 0

    def test_circuit_breaker_vectordb_cli_mode_aborts(self):
        """CLI-Modus (kein control_callback) -> ABORT beim ersten vectordb-Fehler."""
        ictx = _make_ictx(
            consecutive_errors=0,
            stats={"errors": 1, "vectordb_errors": 1},
            control_callback=None,
        )
        result, _ = _handle_circuit_breaker(
            ictx,
            "/test.txt",
            None,
            error_phase="vectordb",
            error_description="VectorDB-Fehler",
            is_infra=True,
            file_retries=0,
            t0=time.monotonic(),
        )
        assert result == _CircuitBreakerResult.ABORT
        assert "abort_reason" in ictx.stats

    def test_embedding_phase_classifies_vectordb_error(self):
        """_handle_embedding_phase: VectorDBConnectionError -> error_phase=vectordb."""
        from kiara.exceptions import VectorDBConnectionError

        ictx = _make_ictx(
            file_hashes={"/f.txt": ("h1", 1)},
            control_callback=None,
        )
        file_log = _make_file_log("/f.txt")
        with patch(
            "kiara.ingestion.runtime._write_embeddings",
            side_effect=VectorDBConnectionError("qdrant down"),
        ):
            action, _ = _handle_embedding_phase(
                ictx,
                "/f.txt",
                ["chunk"],
                [{"source": "/f.txt"}],
                ["id1"],
                file_log,
                0,
                time.monotonic(),
            )
        # CLI-Mode ohne control_callback => ABORT beim ersten vectordb-Fehler
        assert action == _FileAction.ABORT
        assert file_log.error_phase == "vectordb"

    def test_dedup_vectordb_error_raises(self):
        """_handle_chunk_dedup re-raised VectorDBConnectionError aus _sync_ds_ids."""
        from kiara.exceptions import VectorDBConnectionError

        ictx = _make_ictx(datasource_id=1)
        ictx.chunk_ds.chunks_exist.return_value = {"id1"}
        ictx.store.get_documents_by_ids.return_value = {"ids": ["id1"]}
        with patch(
            "kiara.ingestion.runtime._sync_ds_ids",
            side_effect=VectorDBConnectionError("qdrant down"),
        ):
            with pytest.raises(VectorDBConnectionError):
                _handle_chunk_dedup(
                    ictx,
                    "/f.txt",
                    ["text"],
                    [{"source": "/f.txt"}],
                    ["id1"],
                    [None],
                    _make_file_log(),
                    time.monotonic(),
                )


# ===================================================================
# S4-Rework: Guardfehler duerfen an keiner Ingestion-Grenze degradieren
# ===================================================================


def _s4_contract_error(reason_code="index_contract_unverified", operation="vector_read"):
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        IndexVectorKind,
        make_index_target,
    )

    target = make_index_target(
        {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        collection_name="kb_guard_rework",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="test-model",
    )
    return IndexCompatibilityError(
        reason_code,
        operation,
        target,
        "injected S4 guard failure",
        current_state="unverified",
    )


@pytest.mark.parametrize(
    ("existing_ids", "dedup_kind"),
    [
        ({"id1", "id2"}, "full"),
        ({"id1"}, "partial"),
    ],
)
@pytest.mark.parametrize("failure_boundary", ["payload_read", "ds_ids_sync"])
def test_dedup_contract_error_aborts_without_reembedding(
    existing_ids,
    dedup_kind,
    failure_boundary,
):
    """FULL/PARTIAL und beide guarded Dedup-Consumer enden jobweit in ABORT."""
    import contextlib

    error = _s4_contract_error(
        operation=("vector_payload_read" if failure_boundary == "payload_read" else "vector_write")
    )
    ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 1)})
    ictx.chunk_ds.chunks_exist.return_value = existing_ids
    if failure_boundary == "payload_read":
        ictx.store.get_documents_by_ids.side_effect = error
    else:
        ictx.store.get_documents_by_ids.return_value = {"ids": list(existing_ids)}

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch(
                "kiara.ingestion.runtime._process_single_file",
                return_value=(
                    ["text1", "text2"],
                    [{"source": "/f.txt"}, {"source": "/f.txt"}],
                    ["id1", "id2"],
                    "full text",
                    [(0, 5), (6, 11)],
                ),
            )
        )
        embed = stack.enter_context(patch("kiara.ingestion.runtime._handle_embedding_phase"))
        if failure_boundary == "ds_ids_sync":
            stack.enter_context(patch("kiara.ingestion.runtime._sync_ds_ids", side_effect=error))

        result = _process_file_in_pipeline(ictx, "/f.txt")

    assert result is False, dedup_kind
    assert ictx.stats["abort_reason"].startswith("Index-Kompatibilitaetsguard:")
    assert ictx.stats["index_compatibility_error"] == error.to_dict()
    assert ictx.stats["error_details"][-1]["consumer"] == "dedup"
    assert ictx.stats["processed_files"] == 0
    embed.assert_not_called()
    ictx.store.add.assert_not_called()
    ictx.store.add_with_embeddings.assert_not_called()


@pytest.mark.parametrize(
    ("reason_code", "operation"),
    [
        ("index_contract_missing", "colbert_embedding"),
        ("index_contract_unverified", "colbert_upsert"),
        ("index_compatibility_id_mismatch", "colbert_payload_update"),
    ],
)
def test_colbert_contract_error_is_structured_abort(reason_code, operation):
    """ColBERT-Guardfehler erreichen weder SKIP noch den Circuit-Breaker."""
    from kiara.ingestion.runtime import _handle_colbert_step

    error = _s4_contract_error(reason_code=reason_code, operation=operation)
    ictx = _make_ictx(
        config={"retrieval": {"colbert": {"enabled": True}}},
        ds_overrides={"colbert_enabled": True},
        file_hashes={"/f.txt": ("h1", 1)},
        pending_hashes={"/f.txt": ("h1", 1)},
    )
    file_log = _make_file_log("/f.txt")

    with (
        patch("kiara.ingestion.runtime._sync_colbert_chunks", side_effect=error),
        patch("kiara.ingestion.runtime._handle_circuit_breaker") as circuit_breaker,
    ):
        action, retries = _handle_colbert_step(
            ictx,
            chunk_ids=["id1"],
            chunk_texts=["text"],
            chunk_metas=[{}],
            file_path="/f.txt",
            file_log=file_log,
            file_retries=0,
            t0=time.monotonic(),
        )

    assert action == _FileAction.ABORT
    assert retries == 0
    assert file_log.error_phase == "index_compatibility"
    assert ictx.stats["index_compatibility_error"] == error.to_dict()
    assert ictx.stats["error_details"][-1]["consumer"] == "colbert"
    assert ictx.stats["colbert_errors"] == 1
    assert "/f.txt" not in ictx.pending_hashes
    circuit_breaker.assert_not_called()


@pytest.mark.parametrize("failure_boundary", ["embed", "add", "payload_update"])
def test_colbert_sync_reraises_contract_error_and_rolls_back_partial_effects(
    failure_boundary,
):
    """Embed/Add/Update verlieren niemals den originalen Guardfehler."""
    import contextlib

    from kiara.ingestion.runtime import _sync_colbert_chunks

    error = _s4_contract_error(operation=f"colbert_{failure_boundary}")
    ictx = _make_ictx(
        config={
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-test"}},
            "vectordb": {"qdrant_url": "http://qdrant.test:6333"},
        },
        ds_overrides={"colbert_enabled": True},
    )
    ictx.chunk_ds.chunks_exist.return_value = set()
    ictx.chunk_ds.get_datasource_ids_for_chunks.side_effect = [
        {},
        {"id1": [ictx.datasource_id]},
    ]
    cstore = MagicMock()
    cstore.get_collection_name.return_value = "colbert_test"
    if failure_boundary == "add":
        cstore.add.side_effect = error
    elif failure_boundary == "payload_update":
        cstore.update_ds_ids.side_effect = error

    embed_side_effect = error if failure_boundary == "embed" else None
    embed_result = None if failure_boundary == "embed" else [[[0.1, 0.2]]]

    with (
        patch(
            "kiara.retrieval.colbert.colbert_datasource_lock",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(True),
        ),
        patch(
            "kiara.retrieval.colbert.colbert_cleanup_lock",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(True),
        ),
        patch("kiara.ingestion.runtime._datasource_accepts_colbert_write", return_value=True),
        patch("kiara.ingestion.runtime._ensure_colbert_model_is_current_for_write"),
        patch(
            "kiara.retrieval.index_compatibility.guard_index",
            side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ),
        patch("kiara.retrieval.index_compatibility.raw_qdrant_client", return_value=MagicMock()),
        patch("kiara.retrieval.colbert.ColBERTStore", return_value=cstore),
        patch(
            "kiara.retrieval.colbert.embed_colbert",
            return_value=embed_result,
            side_effect=embed_side_effect,
        ) as embed,
        patch("kiara.ingestion.runtime._rollback_colbert_associations", return_value=1) as rollback,
    ):
        with pytest.raises(type(error)) as caught:
            _sync_colbert_chunks(ictx, ["id1"], ["text"], [{"ds_ids": [1]}])

    assert caught.value is error
    embed.assert_called_once_with(
        ictx.config,
        "colbert-test",
        ["text"],
        input_type="search_document",
    )
    if failure_boundary == "embed":
        cstore.add.assert_not_called()
    if failure_boundary == "payload_update":
        rollback.assert_called_once()
    else:
        rollback.assert_not_called()
    if failure_boundary in {"add", "payload_update"}:
        # add() kann vor dem Guardfehler bereits einzelne Batches geschrieben
        # haben. Auch diese neuen, zuvor unbesessenen Punkte zuruecknehmen.
        cstore.delete_by_ids.assert_called_once_with(["id1"])
    else:
        cstore.delete_by_ids.assert_not_called()


def test_dense_mid_batch_contract_and_blocked_rollback_preserve_primary_reason():
    """Der Rollback-Guard darf den urspruenglichen Batch-Reason nicht verdecken."""
    import contextlib

    primary = _s4_contract_error(
        reason_code="index_compatibility_id_mismatch",
        operation="vector_upsert",
    )
    rollback_error = _s4_contract_error(
        reason_code="index_contract_unverified",
        operation="vector_delete",
    )
    ictx = _make_ictx(
        batch_size=1,
        file_hashes={"/f.txt": ("h1", 1)},
    )
    ictx.store._index_contract_guard.return_value = contextlib.nullcontext()
    ictx.store.add.side_effect = [None, primary]
    ictx.store.delete_by_ids.side_effect = rollback_error

    with (
        patch("kiara.ingestion.runtime._write_to_elasticsearch") as elasticsearch,
        patch("kiara.ingestion.runtime._persist_graph_and_associations") as graph,
    ):
        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text1", "text2"],
            [{"source": "/f.txt"}, {"source": "/f.txt"}],
            ["id1", "id2"],
            _make_file_log("/f.txt"),
            0,
            time.monotonic(),
        )

    assert action == _FileAction.ABORT
    assert ictx.stats["index_compatibility_error"]["reason_code"] == primary.reason_code
    assert ictx.stats["index_compatibility_error"]["operation"] == primary.operation
    rollback_diagnostics = ictx.stats["index_compatibility_rollback_errors"]
    assert rollback_diagnostics[0]["diagnostic"]["reason_code"] == rollback_error.reason_code
    assert "Partial-Rollback fehlgeschlagen" in ictx.stats["abort_reason"]
    assert ictx.store.add.call_count == 2
    ictx.store.delete_by_ids.assert_called_once_with(["id1"])
    elasticsearch.assert_not_called()
    graph.assert_not_called()


def test_contract_error_first_seen_during_rollback_is_never_degraded():
    """Auch ein Guardfehler im Rollback eines Nicht-Contractfehlers bleibt strukturiert."""
    rollback_error = _s4_contract_error(
        reason_code="index_contract_evidence_failed",
        operation="vector_delete",
    )
    ictx = _make_ictx(file_hashes={"/f.txt": ("h1", 1)})
    ictx.store.delete_by_ids.side_effect = rollback_error

    with (
        patch("kiara.ingestion.runtime._write_embeddings", return_value=["id1"]),
        patch(
            "kiara.ingestion.runtime._write_to_elasticsearch",
            side_effect=ValueError("ordinary ES write error"),
        ),
        patch("kiara.ingestion.runtime._persist_graph_and_associations") as graph,
        patch("kiara.ingestion.runtime._handle_circuit_breaker") as circuit_breaker,
    ):
        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"source": "/f.txt"}],
            ["id1"],
            _make_file_log("/f.txt"),
            0,
            time.monotonic(),
        )

    assert action == _FileAction.ABORT
    assert ictx.stats["index_compatibility_error"] == rollback_error.to_dict()
    assert ictx.stats["error_details"][-1]["consumer"] == "embedding_rollback"
    assert "index_contract_evidence_failed" in ictx.stats["abort_reason"]
    graph.assert_not_called()
    circuit_breaker.assert_not_called()


def test_transient_contract_outage_waits_and_retries_same_file_without_rollback():
    from kiara.retrieval.index_compatibility import (
        IndexContractUnavailableError,
        IndexVectorKind,
        make_index_target,
    )

    target = make_index_target(
        {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        collection_name="kb_contract_recovery",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="test-model",
    )
    unavailable = IndexContractUnavailableError(
        "index_current_profile_not_verified",
        "vector_batch_write",
        target,
        "contract_discovery_request_failed",
        stored_state="verified",
        current_state="discovery_error",
        issues=(
            {
                "reason_code": "contract_discovery_request_failed",
                "field_path": "/api/tags",
                "detail": "ReadTimeout",
            },
        ),
        discovery_attempts=3,
    )
    ictx = _make_ictx(
        control_callback=MagicMock(return_value=None),
        file_hashes={"/f.txt": ("h1", 1)},
    )
    file_log = _make_file_log("/f.txt")

    with (
        patch("kiara.ingestion.runtime._write_embeddings", side_effect=unavailable),
        patch(
            "kiara.ingestion.runtime._wait_for_index_contract_recovery",
            return_value=True,
        ) as wait,
        patch("kiara.ingestion.runtime._rollback_embedding_persistence") as rollback,
        patch("kiara.ingestion.runtime._abort_index_compatibility") as abort,
    ):
        action, _ = _handle_embedding_phase(
            ictx,
            "/f.txt",
            ["text"],
            [{"source": "/f.txt"}],
            ["id1"],
            file_log,
            0,
            time.monotonic(),
        )

    assert action == _FileAction.RETRY
    wait.assert_called_once_with(ictx, unavailable)
    rollback.assert_not_called()
    abort.assert_not_called()
    assert file_log.status == "pending"


def test_transient_contract_outage_during_store_startup_waits_before_hash_diff(tmp_path):
    from kiara.ingestion.pipeline import run_ingestion
    from kiara.retrieval.index_compatibility import (
        IndexContractUnavailableError,
        IndexVectorKind,
        make_index_target,
    )

    config = {
        "datasource": {
            "path": str(tmp_path),
            "file_types": [".txt"],
            "exclude_patterns": [],
        },
        "vectordb": {"path": str(tmp_path / "vectors")},
        "ingestion": {"chunk_size": 500, "chunk_overlap": 50, "batch_size": 10},
        "graph": {"enabled": False},
    }
    target = make_index_target(
        config,
        collection_name="kb_test-model",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="test-model",
    )
    unavailable = IndexContractUnavailableError(
        "index_current_profile_not_verified",
        "first_write",
        target,
        "contract_discovery_request_failed",
    )
    store = MagicMock()
    hash_store = MagicMock()
    control = MagicMock(return_value=None)
    status = MagicMock()

    def recover(_config, error, **kwargs):
        assert error is unavailable
        kwargs["stats"].update(
            {
                "contract_discovery_waits": 1,
                "contract_discovery_recoveries": 1,
                "contract_discovery_wait_seconds": 2,
            }
        )
        return True

    with (
        patch(
            "kiara.ingestion.pipeline.VectorStore",
            side_effect=[unavailable, store],
        ) as vectorstore,
        patch(
            "kiara.ingestion.pipeline._wait_for_index_contract_target_recovery",
            side_effect=recover,
        ) as wait,
        patch(
            "kiara.ingestion.pipeline._load_and_diff_files",
            return_value=([], [], {}, [], hash_store, {}),
        ) as load_diff,
        patch("kiara.ingestion.pipeline.ChunkDatasources", return_value=MagicMock()),
        patch("kiara.ingestion.pipeline._run_colbert_model_switch_cleanup", return_value=True),
    ):
        stats = run_ingestion(
            config,
            datasource_id=None,
            embedding_model="test-model",
            control_callback=control,
            status_callback=status,
        )

    assert vectorstore.call_count == 2
    wait.assert_called_once()
    load_diff.assert_called_once()
    assert stats["contract_discovery_waits"] == 1
    assert stats["contract_discovery_recoveries"] == 1
    assert stats["contract_discovery_wait_seconds"] == 2


def test_embedding_rollback_is_qdrant_then_elasticsearch_then_database():
    events: list[str] = []
    ictx = _make_ictx()
    _mock_rollback_documents(ictx, ["id1"])
    delete = ictx.store.delete_by_ids.side_effect

    def delete_and_record(ids):
        delete(ids)
        events.append("qdrant")

    ictx.store.delete_by_ids.side_effect = delete_and_record

    with (
        patch(
            "kiara.ingestion.runtime._rollback_elasticsearch_write",
            side_effect=lambda *_args, **_kwargs: events.append("elasticsearch") or 1,
        ),
        patch(
            "kiara.ingestion.runtime._rollback_db_associations",
            side_effect=lambda *_args, **_kwargs: events.append("database") or 1,
        ),
    ):
        result = _rollback_embedding_persistence(
            ictx,
            "/f.txt",
            ["id1"],
            rollback_elasticsearch=True,
            rollback_db_associations=True,
            context="order_test",
        )

    assert result is True
    assert events == ["qdrant", "elasticsearch", "database"]


def test_failed_qdrant_rollback_preserves_es_and_database_retry_anchors():
    from kiara.exceptions import VectorDBConnectionError

    ictx = _make_ictx()
    ictx.store.delete_by_ids.side_effect = VectorDBConnectionError("qdrant down")

    with (
        patch("kiara.ingestion.runtime._rollback_elasticsearch_write") as elasticsearch,
        patch("kiara.ingestion.runtime._rollback_db_associations") as database,
    ):
        result = _rollback_embedding_persistence(
            ictx,
            "/f.txt",
            ["id1"],
            rollback_elasticsearch=True,
            rollback_db_associations=True,
            context="anchor_test",
        )

    assert result is False
    elasticsearch.assert_not_called()
    database.assert_not_called()
    assert ictx.stats["rollback_errors"] == 1
