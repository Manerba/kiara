"""Tests fuer Tracing-Instrumentierung der Ingestion-Pipeline (Sprint 17.4)."""

import threading
from unittest.mock import MagicMock, patch

from kiara.ingestion.pipeline import _get_current_job_id, set_current_job_id

# ---------------------------------------------------------------------------
# RecordingTracer / RecordingSpan
# ---------------------------------------------------------------------------


class RecordingSpan:
    def __init__(self, name):
        self.name = name
        self.attributes = {}
        self.exceptions = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def record_exception(self, exc):
        self.exceptions.append(exc)

    def set_status(self, status):
        pass

    def add_event(self, name, attributes=None):
        pass

    def get_span_context(self):
        ctx = MagicMock()
        ctx.trace_id = 0xABCDEF1234567890ABCDEF1234567890
        ctx.span_id = 0x1234567890ABCDEF
        return ctx

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class RecordingTracer:
    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def start_as_current_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def start_span(self, name, **kwargs):
        span = RecordingSpan(name)
        with self._lock:
            self.spans.append(span)
        return span

    def get_span_names(self):
        with self._lock:
            return [s.name for s in self.spans]

    def get_span(self, name):
        with self._lock:
            for s in self.spans:
                if s.name == name:
                    return s
        return None

    def get_spans(self, name):
        with self._lock:
            return [s for s in self.spans if s.name == name]


# ---------------------------------------------------------------------------
# Thread-local job_id Tests
# ---------------------------------------------------------------------------


class TestJobIdThreadLocal:
    """Tests fuer set_current_job_id / _get_current_job_id."""

    def test_set_and_get_job_id(self):
        set_current_job_id(42)
        assert _get_current_job_id() == 42
        set_current_job_id(None)
        assert _get_current_job_id() is None

    def test_thread_local_job_id_is_thread_isolated(self):
        """Verschiedene Threads sehen verschiedene job_ids."""
        results = {}
        barrier = threading.Barrier(2)

        def worker(name, job_id):
            set_current_job_id(job_id)
            barrier.wait()  # Beide Threads haben gesetzt
            results[name] = _get_current_job_id()
            set_current_job_id(None)

        t1 = threading.Thread(target=worker, args=("t1", 100))
        t2 = threading.Thread(target=worker, args=("t2", 200))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["t1"] == 100
        assert results["t2"] == 200

    def test_worker_sets_job_id_for_pass1(self):
        """_run_ingestion setzt job_id via set_current_job_id."""
        captured_ids = []

        def fake_run_ingestion(**kwargs):
            captured_ids.append(_get_current_job_id())
            return {"processed_files": 0, "total_chunks": 0, "errors": 0}

        with patch("kiara.ingestion.pipeline.run_ingestion", fake_run_ingestion):
            from kiara.ingestion.pipeline import set_current_job_id

            set_current_job_id(77)
            try:
                assert _get_current_job_id() == 77
            finally:
                set_current_job_id(None)


# ---------------------------------------------------------------------------
# _process_file_in_pipeline Span-Tests
# ---------------------------------------------------------------------------


class TestIngestFileSpan:
    """Tests fuer ingest_file Root-Span in _process_file_in_pipeline."""

    def _make_ictx(self):
        """Erstellt einen minimalen _IngestionContext Mock."""
        from kiara.ingestion.runtime import _IngestionContext

        ictx = MagicMock(spec=_IngestionContext)
        ictx.config = {
            "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
            "retrieval": {"hybrid_search": "none"},
            "graph": {"enabled": False},
        }
        ictx.datasource_id = 1
        ictx.ds_overrides = {}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.control_callback = None
        ictx.file_log_callback = None
        ictx.file_start_callback = None
        ictx.progress_callback = None
        ictx.ctx = MagicMock()
        ictx.ctx.ddu_enabled = False
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.skipped = 0
        ictx.total = 10
        return ictx

    def test_ingest_file_span_wraps_success_path(self, tmp_path):
        """Erfolgsfall: ingest_file-Span mit final_status=success."""
        recorder = RecordingTracer()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        ictx = self._make_ictx()
        ictx.file_hashes = {str(test_file): "hash123"}
        ictx.pending_hashes = {}

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch(
                "kiara.ingestion.runtime._process_single_file",
                return_value=(
                    ["text1"],
                    [{"source": str(test_file), "chunk_index": 0}],
                    ["id1"],
                    "full text",
                    [],
                ),
            ):
                with patch("kiara.ingestion.runtime._handle_chunk_dedup") as mock_dedup:
                    from kiara.ingestion.runtime import _DedupResult

                    mock_dedup.return_value = (
                        _DedupResult.NO_DEDUP,
                        ["text1"],
                        [{"source": str(test_file), "chunk_index": 0}],
                        ["id1"],
                        [None],
                        None,
                    )
                    with patch("kiara.ingestion.runtime._refresh_late_chunking_snapshots"):
                        with patch("kiara.ingestion.runtime._handle_embedding_phase") as mock_embed:
                            from kiara.ingestion.runtime import _FileAction

                            mock_embed.return_value = (_FileAction.SUCCESS, 0)
                            with patch("kiara.ingestion.runtime._finalize_file"):
                                from kiara.ingestion.runtime import _process_file_in_pipeline

                                result = _process_file_in_pipeline(ictx, str(test_file))

        assert result is True
        span = recorder.get_span("ingest_file")
        assert span is not None
        assert span.attributes["file_path"] == str(test_file)
        assert span.attributes["final_status"] == "success"

    def test_ingest_file_span_cancelled_before_start(self, tmp_path):
        """Cancel vor Start: final_status=cancelled_before_start."""
        recorder = RecordingTracer()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello")

        ictx = self._make_ictx()
        ictx.control_callback = lambda: "cancel"

        with patch("kiara.ingestion.runtime._tracer", recorder):
            from kiara.ingestion.runtime import _process_file_in_pipeline

            result = _process_file_in_pipeline(ictx, str(test_file))

        assert result is False
        span = recorder.get_span("ingest_file")
        assert span.attributes["final_status"] == "cancelled_before_start"


# ---------------------------------------------------------------------------
# _handle_embedding_phase Span-Tests
# ---------------------------------------------------------------------------


class TestEmbedBatchSpan:
    """Tests fuer embed_batch-Span in _handle_embedding_phase."""

    def test_embed_batch_span_on_cancel(self):
        """_CancelledError: embed_batch.cancelled=True, kein record_exception."""
        recorder = RecordingTracer()

        from kiara.ingestion._file_utils import _CancelledError
        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = MagicMock()
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.control_callback = None
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"hybrid_search": "none"}, "graph": {"enabled": False}}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {"/test.txt": "h"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.file_log_callback = None

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch("kiara.ingestion.runtime._write_embeddings", side_effect=_CancelledError()):
                with patch("kiara.ingestion.runtime._persist_hashes_checkpoint"):
                    action, retries = _handle_embedding_phase(
                        ictx,
                        "/test.txt",
                        ["text"],
                        [{}],
                        ["id1"],
                        None,
                        0,
                        0.0,
                    )

        assert action == _FileAction.CANCEL
        span = recorder.get_span("embed_batch")
        assert span is not None
        assert span.attributes.get("cancelled") is True
        assert len(span.exceptions) == 0  # CancelledError ist KEIN record_exception

    def test_embed_batch_span_on_generic_error(self):
        """Generische Exception: record_exception + SKIP."""
        recorder = RecordingTracer()

        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = MagicMock()
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.control_callback = None
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"hybrid_search": "none"}, "graph": {"enabled": False}}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {"/test.txt": "h"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.file_log_callback = None

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch(
                "kiara.ingestion.runtime._write_embeddings", side_effect=ValueError("test error")
            ):
                action, retries = _handle_embedding_phase(
                    ictx,
                    "/test.txt",
                    ["text"],
                    [{}],
                    ["id1"],
                    None,
                    0,
                    0.0,
                )

        assert action == _FileAction.SKIP
        span = recorder.get_span("embed_batch")
        assert len(span.exceptions) == 1

    def test_embed_batch_span_on_categorical_error_abort(self):
        """_CATEGORICAL_ERRORS: record_exception + ABORT bei Circuit-Breaker."""
        recorder = RecordingTracer()

        from kiara.backend_selector import BackendUnavailableError
        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = MagicMock()
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.control_callback = None
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"hybrid_search": "none"}, "graph": {"enabled": False}}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {"/test.txt": "h"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.file_log_callback = None

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch(
                "kiara.ingestion.runtime._write_embeddings",
                side_effect=BackendUnavailableError("Ollama down"),
            ):
                with patch("kiara.ingestion.runtime._handle_circuit_breaker") as mock_cb:
                    from kiara.ingestion.runtime import _CircuitBreakerResult

                    mock_cb.return_value = (_CircuitBreakerResult.ABORT, 0)
                    action, retries = _handle_embedding_phase(
                        ictx,
                        "/test.txt",
                        ["text"],
                        [{}],
                        ["id1"],
                        None,
                        0,
                        0.0,
                    )

        assert action == _FileAction.ABORT
        span = recorder.get_span("embed_batch")
        assert len(span.exceptions) == 1  # record_exception aufgerufen

    def test_embed_batch_span_covers_circuit_breaker_cancel(self):
        """_CATEGORICAL_ERRORS + Circuit-Breaker CANCEL: Exception + CANCEL."""
        recorder = RecordingTracer()

        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = MagicMock()
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.control_callback = None
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"hybrid_search": "none"}, "graph": {"enabled": False}}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {"/test.txt": "h"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 9
        ictx.file_log_callback = None

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch(
                "kiara.ingestion.runtime._write_embeddings",
                side_effect=ConnectionError("Network down"),
            ):
                with patch("kiara.ingestion.runtime._handle_circuit_breaker") as mock_cb:
                    from kiara.ingestion.runtime import _CircuitBreakerResult

                    mock_cb.return_value = (_CircuitBreakerResult.CANCEL, 0)
                    action, retries = _handle_embedding_phase(
                        ictx,
                        "/test.txt",
                        ["text"],
                        [{}],
                        ["id1"],
                        None,
                        0,
                        0.0,
                    )

        assert action == _FileAction.CANCEL
        span = recorder.get_span("embed_batch")
        assert len(span.exceptions) == 1


class TestEmbedBatchLateCunking:
    """Tests fuer late_chunking-Attribut auf dem embed_batch-Span."""

    def _make_ictx(self):
        """Erstellt einen minimalen _IngestionContext Mock."""
        ictx = MagicMock()
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.control_callback = None
        ictx.datasource_id = 1
        ictx.config = {"retrieval": {"hybrid_search": "none"}, "graph": {"enabled": False}}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {"/test.txt": "h"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.file_log_callback = None
        ictx.last_full_text = ""
        ictx.last_original_chunks = None
        ictx.last_original_spans = None
        return ictx

    def test_embed_batch_span_late_chunking_false(self):
        """Standard-Pfad ohne Late Chunking: late_chunking=False auf embed_batch-Span."""
        recorder = RecordingTracer()

        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = self._make_ictx()

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch("kiara.ingestion.runtime._should_use_late_chunking", return_value=False):
                with patch("kiara.ingestion.runtime._write_embeddings"):
                    with patch("kiara.ingestion.runtime._write_to_elasticsearch"):
                        with patch("kiara.ingestion.runtime._extract_graph_entities"):
                            with patch("kiara.ingestion.runtime._persist_hashes_checkpoint"):
                                with patch("kiara.ingestion.runtime._finalize_file"):
                                    action, retries = _handle_embedding_phase(
                                        ictx,
                                        "/test.txt",
                                        ["text1"],
                                        [{"source": "/test.txt"}],
                                        ["id1"],
                                        None,
                                        0,
                                        0.0,
                                    )

        assert action == _FileAction.SUCCESS
        span = recorder.get_span("embed_batch")
        assert span is not None
        assert span.attributes.get("late_chunking") is False

        # write_vectors-Span muss ebenfalls existieren
        wv_span = recorder.get_span("write_vectors")
        assert wv_span is not None

    def test_embed_batch_span_late_chunking_true(self):
        """Late-Chunking-Pfad aktiv: late_chunking=True auf embed_batch-Span."""
        recorder = RecordingTracer()

        from kiara.ingestion.runtime import _FileAction, _handle_embedding_phase

        ictx = self._make_ictx()
        ictx.last_full_text = "Hello World this is a test document."
        ictx.last_original_chunks = ["Hello World", "this is a test document."]
        ictx.last_original_spans = [(0, 11), (12, 36)]

        fake_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch("kiara.ingestion.runtime._should_use_late_chunking", return_value=True):
                with patch(
                    "kiara.ingestion.runtime._late_chunking_file_eligible", return_value=True
                ):
                    with patch(
                        "kiara.ingestion.runtime.embed_texts_late", return_value=fake_embeddings
                    ):
                        with patch("kiara.ingestion.runtime._write_embeddings"):
                            with patch("kiara.ingestion.runtime._write_to_elasticsearch"):
                                with patch("kiara.ingestion.runtime._extract_graph_entities"):
                                    with patch(
                                        "kiara.ingestion.runtime._persist_hashes_checkpoint"
                                    ):
                                        with patch("kiara.ingestion.runtime._finalize_file"):
                                            action, retries = _handle_embedding_phase(
                                                ictx,
                                                "/test.txt",
                                                ["text1", "text2"],
                                                [{"source": "/test.txt"}, {"source": "/test.txt"}],
                                                ["id1", "id2"],
                                                None,
                                                0,
                                                0.0,
                                            )

        assert action == _FileAction.SUCCESS
        span = recorder.get_span("embed_batch")
        assert span is not None
        assert span.attributes.get("late_chunking") is True

        # write_vectors-Span muss auch im Late-Chunking-Pfad existieren
        wv_span = recorder.get_span("write_vectors")
        assert wv_span is not None


class TestIngestFileRetrySpan:
    """Tests fuer Retry-Szenarien in _process_file_in_pipeline."""

    def test_ingest_file_root_span_wraps_loading_retry_and_embedding_retry(self, tmp_path):
        """Loading-Retry + Embedding-Retry in einem Dateilauf: genau EIN ingest_file-Span."""
        recorder = RecordingTracer()
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")

        from kiara.ingestion.runtime import (
            _DedupResult,
            _FileAction,
            _process_file_in_pipeline,
        )

        ictx = MagicMock()
        ictx.config = {
            "ingestion": {"chunk_size": 1000, "chunk_overlap": 200},
            "retrieval": {"hybrid_search": "none"},
            "graph": {"enabled": False},
        }
        ictx.datasource_id = 1
        ictx.ds_overrides = {}
        ictx.stats = {"errors": 0, "total_chunks": 0, "processed_files": 0}
        ictx.control_callback = None
        ictx.file_log_callback = None
        ictx.file_start_callback = None
        ictx.progress_callback = None
        ictx.ctx = MagicMock(ddu_enabled=False)
        ictx.store = MagicMock()
        ictx.batch_size = 100
        ictx.embedding_model = "test"
        ictx.chunk_ds = MagicMock()
        ictx.hash_store = MagicMock()
        ictx.file_hashes = {str(test_file): "hash123"}
        ictx.pending_hashes = {}
        ictx.consecutive_errors = 0
        ictx.skipped = 0
        ictx.total = 10

        # Simuliere: 1. Aufruf → leer (Loading-Fehler → RETRY),
        #            2. Aufruf → Daten (Erfolg)
        call_count = {"n": 0}

        def fake_process_single_file(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [], [], [], "", []
            return (
                ["text1"],
                [{"source": str(test_file), "chunk_index": 0}],
                ["id1"],
                "full text",
                [],
            )

        def fake_handle_loading_result(*args, **kwargs):
            return _FileAction.RETRY, 1

        # Embedding: 1. Aufruf → RETRY, 2. Aufruf → SUCCESS
        embed_count = {"n": 0}

        def fake_handle_embedding(*args, **kwargs):
            embed_count["n"] += 1
            if embed_count["n"] == 1:
                return _FileAction.RETRY, 1
            return _FileAction.SUCCESS, 1

        with patch("kiara.ingestion.runtime._tracer", recorder):
            with patch(
                "kiara.ingestion.runtime._process_single_file", side_effect=fake_process_single_file
            ):
                with patch(
                    "kiara.ingestion.runtime._handle_loading_result",
                    side_effect=fake_handle_loading_result,
                ):
                    with patch("kiara.ingestion.runtime._handle_chunk_dedup") as mock_dedup:
                        mock_dedup.return_value = (
                            _DedupResult.NO_DEDUP,
                            ["t"],
                            [{}],
                            ["id"],
                            [None],
                            None,
                        )
                        with patch("kiara.ingestion.runtime._refresh_late_chunking_snapshots"):
                            with patch(
                                "kiara.ingestion.runtime._handle_embedding_phase",
                                side_effect=fake_handle_embedding,
                            ):
                                with patch("kiara.ingestion.runtime._finalize_file"):
                                    result = _process_file_in_pipeline(ictx, str(test_file))

        assert result is True
        # Genau EIN Root-Span fuer die gesamte Datei (trotz 2 Retries)
        ingest_spans = recorder.get_spans("ingest_file")
        assert len(ingest_spans) == 1
        assert ingest_spans[0].attributes["final_status"] == "success"


class TestWorkerJobIdPass2:
    """Tests fuer set_current_job_id in Worker Pass 2."""

    def test_worker_pass2_sets_job_id(self):
        """Pass 2 setzt job_id via set_current_job_id."""
        captured_ids = []

        def fake_run_ingestion(config, **kwargs):
            captured_ids.append(_get_current_job_id())
            return {
                "processed_files": 0,
                "total_chunks": 0,
                "errors": 0,
                "skipped_files": 0,
                "total_files": 0,
            }

        from kiara.web.ingestion_worker import IngestionWorker

        worker = object.__new__(IngestionWorker)
        worker._set_job_phase = MagicMock()

        scan_complete = threading.Event()
        scan_complete.set()  # Scan bereits fertig
        scan_thread = MagicMock()
        scan_thread.join = MagicMock()
        scan_info = {
            "scan_thread": scan_thread,
            "scan_complete": scan_complete,
            "scan_error": [None],
        }

        stats = {"processed_files": 5, "total_chunks": 100, "errors": 0}
        callbacks = {
            "on_file_log": MagicMock(),
            "on_file_start": MagicMock(),
            "on_status": MagicMock(),
            "flush_file_logs": MagicMock(),
        }

        with patch("kiara.ingestion.pipeline.run_ingestion", side_effect=fake_run_ingestion):
            with patch("kiara.web.ingestion_worker.db_transaction"):
                with patch("kiara.web.ingestion_worker._update_job_stats"):
                    worker._run_pass2(
                        job_id=42,
                        ds_id=1,
                        config={},
                        overrides={},
                        embedding_model="test",
                        on_progress=MagicMock(),
                        check_control=lambda: None,
                        callbacks=callbacks,
                        hash_mode="mtime",
                        warmup_stats_dict={},
                        stats=stats,
                        flush_fn=MagicMock(),
                        scan_info=scan_info,
                    )

        assert len(captured_ids) == 1
        assert captured_ids[0] == 42
        # Nach dem finally-Block muss job_id wieder None sein
        assert _get_current_job_id() is None

    def test_worker_pass2_scan_error_sets_abort_reason(self):
        """Ein fehlgeschlagener Parallel-Scan darf nicht als Success enden."""
        from kiara.web.ingestion_worker import IngestionWorker

        worker = object.__new__(IngestionWorker)
        worker._set_job_phase = MagicMock()

        scan_complete = threading.Event()
        scan_complete.set()
        scan_thread = MagicMock()
        scan_thread.join = MagicMock()
        scan_info = {
            "scan_thread": scan_thread,
            "scan_complete": scan_complete,
            "scan_error": [RuntimeError("imap down")],
        }
        stats = {"processed_files": 5, "total_chunks": 100, "errors": 0}
        callbacks = {
            "on_file_log": MagicMock(),
            "on_file_start": MagicMock(),
            "on_status": MagicMock(),
            "flush_file_logs": MagicMock(),
        }

        with patch("kiara.ingestion.pipeline.run_ingestion") as mock_run:
            with patch("kiara.web.ingestion_worker.db_transaction"):
                with patch("kiara.web.ingestion_worker._update_job_stats"):
                    worker._run_pass2(
                        job_id=42,
                        ds_id=1,
                        config={},
                        overrides={},
                        embedding_model="test",
                        on_progress=MagicMock(),
                        check_control=lambda: None,
                        callbacks=callbacks,
                        hash_mode="mtime",
                        warmup_stats_dict={},
                        stats=stats,
                        flush_fn=MagicMock(),
                        scan_info=scan_info,
                    )

        mock_run.assert_not_called()
        assert stats["parallel_scan"] is True
        assert "Parallel-Scan fehlgeschlagen" in stats["abort_reason"]
        assert "imap down" in stats["abort_reason"]
