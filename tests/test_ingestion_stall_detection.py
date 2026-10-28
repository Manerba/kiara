"""Tests fuer CIFS-D-State-Erkennung und direkte Ingestion-Aktivitaet."""

from __future__ import annotations

import copy
import threading
from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest

from kiara.config import DEFAULT_CONFIG, _validate, get_ingestion_stall_detection_config
from kiara.ingestion.runtime import _write_embeddings
from kiara.web.ingestion_stall_detection import (
    CIFSDStateDetector,
    IngestionActivitySnapshot,
    IngestionActivityTracker,
    ThreadProcSnapshot,
    read_python_thread_snapshot,
    read_thread_proc_snapshot,
)


def _activity(
    *,
    sequence: int = 7,
    connector_type: str = "smb",
    phase: str = "loading",
    chunks_created: int = 12,
    chunks_written: int = 3,
) -> IngestionActivitySnapshot:
    return IngestionActivitySnapshot(
        job_id=402,
        datasource_id=33,
        connector_type=connector_type,
        file_name="grosses dokument.md",
        file_path="/mnt/kiara-ds-33/grosses dokument.md",
        file_started_at="2026-09-03T21:42:55+00:00",
        phase=phase,
        phase_started_at="2026-09-03T21:42:55+00:00",
        last_event="file_started",
        last_activity_at="2026-09-03T21:42:55+00:00",
        last_activity_monotonic=10.0,
        activity_sequence=sequence,
        chunks_created_current_file=chunks_created,
        chunks_written_current_file=chunks_written,
    )


def _thread(
    *,
    state: str = "D",
    wchan: str = "folio_wait_bit_common",
    kernel_stack: str = "cifs_strict_readv\nfilemap_read",
    user_ticks: int = 100,
    system_ticks: int = 20,
    voluntary: int = 50,
    nonvoluntary: int = 4,
) -> ThreadProcSnapshot:
    return ThreadProcSnapshot(
        pid=1234,
        tid=1235,
        state=state,
        state_label=f"{state} (disk sleep)",
        wchan=wchan,
        kernel_stack=kernel_stack,
        user_ticks=user_ticks,
        system_ticks=system_ticks,
        voluntary_context_switches=voluntary,
        nonvoluntary_context_switches=nonvoluntary,
    )


class TestCIFSDStateDetector:
    def test_detect_confirm_clear_transitions_are_deduplicated(self):
        detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        activity = _activity()
        blocked = _thread()

        assert detector.observe(activity, blocked, now_monotonic=0) == []
        assert detector.observe(activity, blocked, now_monotonic=29) == []

        detected = detector.observe(activity, blocked, now_monotonic=30)
        assert [event.event_type for event in detected] == ["INGESTION_DSTATE_DETECTED"]
        assert detected[0].details["classification"] == "cifs_kernel_wait"
        assert detected[0].details["automatic_recovery"] is False

        assert detector.observe(activity, blocked, now_monotonic=90) == []
        confirmed = detector.observe(activity, blocked, now_monotonic=120)
        assert [event.event_type for event in confirmed] == ["INGESTION_DSTATE_STALL_CONFIRMED"]
        assert confirmed[0].level == "ERROR"
        assert confirmed[0].details["activity"]["chunks_written_current_file"] == 3

        assert detector.observe(activity, blocked, now_monotonic=180) == []
        cleared = detector.observe(activity, _thread(state="S"), now_monotonic=181)
        assert [event.event_type for event in cleared] == ["INGESTION_DSTATE_CLEARED"]
        assert cleared[0].reason == "thread_left_d_state"
        assert detector.observe(activity, _thread(state="S"), now_monotonic=182) == []

    def test_pipeline_activity_restarts_timeout(self):
        detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        blocked = _thread()

        detector.observe(_activity(sequence=1), blocked, now_monotonic=0)
        detected = detector.observe(_activity(sequence=1), blocked, now_monotonic=30)
        assert detected[0].event_type == "INGESTION_DSTATE_DETECTED"

        resumed = detector.observe(_activity(sequence=2), blocked, now_monotonic=100)
        assert [event.event_type for event in resumed] == ["INGESTION_DSTATE_CLEARED"]
        assert resumed[0].reason == "pipeline_activity_resumed"

        events = detector.observe(_activity(sequence=2), blocked, now_monotonic=219)
        assert [event.event_type for event in events] == ["INGESTION_DSTATE_DETECTED"]
        confirmed = detector.observe(_activity(sequence=2), blocked, now_monotonic=220)
        assert [event.event_type for event in confirmed] == ["INGESTION_DSTATE_STALL_CONFIRMED"]

    def test_cpu_active_large_document_never_becomes_cifs_stall(self):
        detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        running = _thread(state="R", wchan="0", kernel_stack="")

        assert detector.observe(_activity(), running, now_monotonic=0) == []
        assert detector.observe(_activity(), running, now_monotonic=600) == []

    def test_non_cifs_d_state_is_visible_but_not_confirmed_as_cifs_stall(self):
        detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        unrelated = _thread(wchan="do_wait", kernel_stack="schedule")

        detector.observe(_activity(), unrelated, now_monotonic=0)
        detected = detector.observe(_activity(), unrelated, now_monotonic=30)
        assert detected[0].event_type == "INGESTION_DSTATE_DETECTED"
        assert detected[0].details["classification"] == "unclassified"
        assert detector.observe(_activity(), unrelated, now_monotonic=120) == []

    def test_generic_folio_wait_only_classifies_during_smb_loading(self):
        loading = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        generic_wait = _thread(kernel_stack="")
        loading.observe(_activity(phase="loading"), generic_wait, now_monotonic=0)
        confirmed = loading.observe(
            _activity(phase="loading"),
            generic_wait,
            now_monotonic=120,
        )
        assert [event.event_type for event in confirmed] == [
            "INGESTION_DSTATE_DETECTED",
            "INGESTION_DSTATE_STALL_CONFIRMED",
        ]
        assert confirmed[-1].reason == "smb_filemap_wait"

        embedding = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        embedding.observe(_activity(phase="embedding"), generic_wait, now_monotonic=0)
        events = embedding.observe(
            _activity(phase="embedding"),
            generic_wait,
            now_monotonic=120,
        )
        assert [event.event_type for event in events] == ["INGESTION_DSTATE_DETECTED"]

    def test_scheduler_activity_restarts_candidate(self):
        detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
        activity = _activity()

        detector.observe(activity, _thread(), now_monotonic=0)
        detector.observe(activity, _thread(), now_monotonic=30)
        events = detector.observe(activity, _thread(voluntary=51), now_monotonic=119)
        assert [event.event_type for event in events] == ["INGESTION_DSTATE_CLEARED"]
        assert events[0].reason == "scheduler_activity_resumed"
        assert detector.observe(activity, _thread(voluntary=51), now_monotonic=120) == []

    def test_non_smb_job_is_out_of_scope(self):
        detector = CIFSDStateDetector(warning_after_seconds=1, timeout_seconds=2)
        assert detector.observe(_activity(connector_type="local"), _thread(), now_monotonic=0) == []
        assert (
            detector.observe(_activity(connector_type="local"), _thread(), now_monotonic=10) == []
        )


def test_activity_tracker_counts_chunks_per_current_file():
    tracker = IngestionActivityTracker()
    tracker.begin_job(402, 33, "smb")
    tracker.record(
        402,
        "file_started",
        phase="loading",
        file_name="gross.md",
        file_path="/mnt/share/gross.md",
    )
    tracker.record(402, "content_chunked", phase="chunking_complete", chunks_created=23)
    tracker.record(402, "vector_batch_written", phase="embedding", chunks_written_delta=10)
    tracker.record(402, "vector_batch_written", phase="embedding", chunks_written_delta=10)
    tracker.record(402, "vector_batch_written", phase="embedding", chunks_written_delta=3)

    snapshot = tracker.snapshot()
    assert snapshot.connector_type == "smb"
    assert snapshot.file_path == "/mnt/share/gross.md"
    assert snapshot.chunks_created_current_file == 23
    assert snapshot.chunks_written_current_file == 23
    assert snapshot.last_event == "vector_batch_written"
    assert "last_activity_monotonic" not in snapshot.for_persistence()

    tracker.finish_job(402)
    assert tracker.snapshot().job_id is None


def test_read_thread_proc_snapshot_parses_native_thread_files(tmp_path):
    task_dir = tmp_path / "1234" / "task" / "1235"
    task_dir.mkdir(parents=True)
    (task_dir / "status").write_text(
        "Name:\tingestion worker\n"
        "State:\tD (disk sleep)\n"
        "voluntary_ctxt_switches:\t50\n"
        "nonvoluntary_ctxt_switches:\t4\n",
        encoding="utf-8",
    )
    stat_fields = ["D", *("0" for _ in range(10)), "100", "20"]
    (task_dir / "stat").write_text(
        f"1235 (ingestion worker) {' '.join(stat_fields)}\n",
        encoding="utf-8",
    )
    (task_dir / "wchan").write_text("folio_wait_bit_common\n", encoding="utf-8")
    (task_dir / "stack").write_text("cifs_strict_readv\nfilemap_read\n", encoding="utf-8")
    (task_dir / "syscall").write_text("0 0x7 0xabc 0x72d 0 0 0 0 0\n", encoding="utf-8")
    (task_dir / "sched").write_text("nr_switches : 54\n", encoding="utf-8")
    (task_dir / "io").write_text("rchar: 1234\n", encoding="utf-8")

    snapshot = read_thread_proc_snapshot(1234, 1235, proc_root=tmp_path)

    assert snapshot is not None
    assert snapshot.state == "D"
    assert snapshot.user_ticks == 100
    assert snapshot.system_ticks == 20
    assert snapshot.voluntary_context_switches == 50
    assert snapshot.nonvoluntary_context_switches == 4
    assert snapshot.wchan == "folio_wait_bit_common"
    assert "cifs_strict_readv" in snapshot.kernel_stack
    assert snapshot.syscall.startswith("0 0x7")
    assert "nr_switches" in snapshot.sched
    assert snapshot.io == "rchar: 1234"


def test_python_thread_snapshot_captures_current_python_stack():
    snapshot = read_python_thread_snapshot(threading.get_native_id())

    assert snapshot["available"] is True
    assert snapshot["python_ident"] == threading.get_ident()
    assert "test_python_thread_snapshot_captures_current_python_stack" in snapshot["stack"]


def test_vector_batches_emit_direct_chunk_write_activity():
    store = MagicMock()
    store._index_contract_guard.return_value = nullcontext()
    activity = MagicMock()
    texts = [f"chunk-{index}" for index in range(5)]
    metas = [{} for _ in texts]
    ids = [f"id-{index}" for index in range(5)]

    written = _write_embeddings(
        texts,
        metas,
        ids,
        store,
        2,
        activity_callback=activity,
    )

    assert written == ids
    assert [call.kwargs["chunks_written_delta"] for call in activity.call_args_list] == [2, 2, 1]
    assert all(call.args == ("vector_batch_written",) for call in activity.call_args_list)


class TestStallDetectionConfig:
    def test_defaults_are_enabled_with_120_second_timeout(self):
        config = get_ingestion_stall_detection_config({})
        assert config == {
            "enabled": True,
            "poll_interval_seconds": 5.0,
            "warning_after_seconds": 30.0,
            "timeout_seconds": 120.0,
        }

    @pytest.mark.parametrize(
        "field,value",
        [
            ("poll_interval_seconds", 0),
            ("warning_after_seconds", -1),
            ("timeout_seconds", "120"),
        ],
    )
    def test_invalid_numeric_values_fail_closed(self, field, value):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["ingestion"]["stall_detection"][field] = value
        with pytest.raises(ValueError):
            _validate(config)

    def test_warning_must_not_exceed_timeout(self):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["ingestion"]["stall_detection"]["warning_after_seconds"] = 121
        config["ingestion"]["stall_detection"]["timeout_seconds"] = 120
        with pytest.raises(ValueError, match="warning_after_seconds"):
            _validate(config)


def test_worker_stall_event_uses_structured_system_logger():
    from kiara.web.ingestion_worker import IngestionWorker

    detector = CIFSDStateDetector(warning_after_seconds=30, timeout_seconds=120)
    detector.observe(_activity(), _thread(), now_monotonic=0)
    event = detector.observe(_activity(), _thread(), now_monotonic=30)[0]
    worker = IngestionWorker()

    with (
        patch("kiara.web.ingestion_worker.stall_security_logger") as security_logger,
        patch(
            "kiara.web.ingestion_worker.read_cifs_proc_diagnostics",
            return_value={"DebugData": {"available": True, "raw": "MID: 8", "error": ""}},
        ),
        patch(
            "kiara.web.ingestion_worker.read_python_thread_snapshot",
            return_value={"available": True, "stack": "loader.py: read"},
        ),
        patch(
            "kiara.web.ingestion_worker.request_privileged_ingestion_forensics",
            return_value={"available": True, "pending_mids": [{"mid": 8}]},
        ),
        patch.object(worker, "_persist_stall_event") as persist,
    ):
        worker._handle_stall_event(event)

    security_logger.log.assert_called_once()
    level, message = security_logger.log.call_args.args
    assert level > 0
    assert message.startswith("INGESTION_DSTATE_DETECTED ")
    details = security_logger.log.call_args.kwargs["extra"]["kiara_details"]
    assert details["activity"]["file_path"].endswith("grosses dokument.md")
    assert details["cifs_proc"]["DebugData"]["raw"] == "MID: 8"
    assert details["python_thread"]["stack"] == "loader.py: read"
    assert details["privileged_forensics"]["pending_mids"][0]["mid"] == 8
    persist.assert_called_once_with(event, details)
