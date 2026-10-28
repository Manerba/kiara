"""S5 F1/F2: durable release evidence and BEIR status contract."""

import json
import os
import sys
from pathlib import Path

import pytest
from test_elasticsearch_s4 import backend, corpus, run  # noqa: F401

from kiara.benchmarks.beir import _elasticsearch_snapshot
from kiara.retrieval import elasticsearch_maintenance as maintenance
from kiara.retrieval import elasticsearch_service as es


@pytest.fixture
def release(monkeypatch, tmp_path):
    marker = tmp_path / "state" / "marker.json"
    report = tmp_path / "evidence" / "release.json"
    report.parent.mkdir()
    config = {"retrieval": {"elasticsearch_maintenance_file": str(marker)}}
    monkeypatch.setattr("kiara.config.load_config", lambda _: config)
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda _: "test-model")
    monkeypatch.setattr("kiara.shared.db.init_db_backend", lambda _: None)
    monkeypatch.setattr(maintenance, "assert_stopped", lambda: None)
    maintenance.begin(config)

    def verify(config, *, full):
        maintenance.require_session(config)
        assert full is True
        return {"status": "success", "release_ready": True, "verification": "full"}

    monkeypatch.setattr(es, "verify_integrity", verify)
    monkeypatch.setattr(
        sys,
        "argv",
        ["maintenance", "release", "--config", "isolated.yaml", "--report", str(report)],
    )
    return marker, report


def test_release_syncs_report_file_and_directory_before_marker_removal(monkeypatch, release):
    marker, report = release
    events = []
    fsync, unlink = os.fsync, Path.unlink

    def sync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        if path == report:
            # The complete JSON must already have been flushed.
            assert json.loads(report.read_text())["verification"] == "full"
        events.append(("fsync", path))
        fsync(fd)

    def remove(path, *args, **kwargs):
        events.append(("unlink", path))
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "fsync", sync)
    monkeypatch.setattr(Path, "unlink", remove)
    maintenance.main()
    assert events == [
        ("fsync", report),
        ("fsync", report.parent),
        ("unlink", marker),
        ("fsync", marker.parent),
    ]
    assert not marker.exists()


@pytest.mark.parametrize("failed_sync", ["file", "directory"])
def test_release_report_sync_failure_preserves_marker(monkeypatch, release, failed_sync):
    marker, report = release
    original_marker = marker.read_bytes()
    target = report if failed_sync == "file" else report.parent
    fsync = os.fsync
    synced = []

    def sync(fd):
        path = Path(os.readlink(f"/proc/self/fd/{fd}"))
        synced.append(path)
        if path == target:
            raise OSError(f"injected {failed_sync} fsync failure")
        fsync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(OSError, match=f"injected {failed_sync} fsync failure"):
        maintenance.main()
    assert synced == ([report] if failed_sync == "file" else [report, report.parent])
    assert marker.read_bytes() == original_marker


@pytest.mark.parametrize("missing_copy", [False, True])
def test_beir_consumes_actual_source_inventory(corpus, backend, missing_copy):
    assert run(corpus)["release_ready"]
    if missing_copy:
        backend.data["s4_test_ds_0"].pop(next(iter(backend.data["s4_test_ds_0"])))
    with maintenance.offline_session(corpus.config):
        snapshot = _elasticsearch_snapshot(corpus.config)
        status = es.get_status(corpus.config)
    # Real producer/inventory, test PG and QdrantLocal; controlled ES transport.
    assert snapshot == status
    assert "index_name" not in snapshot
    assert snapshot["index_prefix"] == "s4_test"
    assert snapshot["unit"] == "physical_primary_documents"
    assert snapshot["embedding_model"] == corpus.model
    assert snapshot["doc_count"] == (15 if missing_copy else 16)
    assert snapshot["expected_physical_documents"] == 16
    assert snapshot["counts_match"] is (not missing_copy)
    assert snapshot["health"] == ("red" if missing_copy else "green")
    assert bool(snapshot["error"]) is missing_copy
    rows = {row["datasource_id"]: row for row in snapshot["datasources"]}
    assert set(rows) == set(corpus.ids)
    assert rows[0]["index_name"] == "s4_test_ds_0"
    assert rows[0]["expected"] == 12
    assert rows[corpus.ids[1]]["observed"] == 4
    assert rows[corpus.ids[2]]["observed"] == 0


def test_beir_unavailable_status_preserves_unknown_counts(monkeypatch, s4_guard_passthrough):
    def unavailable(config):
        raise RuntimeError("injected unavailable ES")

    monkeypatch.setattr(es, "_get_client", unavailable)
    snapshot = _elasticsearch_snapshot({"retrieval": {"hybrid_search": "elasticsearch"}})
    assert snapshot["available"] is False
    assert snapshot["health"] == "unknown"
    assert snapshot["error"] == "injected unavailable ES"
    for key in (
        "doc_count",
        "expected_physical_documents",
        "counts_match",
        "datasources",
        "unit",
        "embedding_model",
    ):
        assert snapshot[key] is None
    assert "index_name" not in snapshot


def test_beir_status_exception_remains_error(monkeypatch):
    def failed(config):
        raise RuntimeError("injected status failure")

    monkeypatch.setattr(es, "get_status", failed)
    assert _elasticsearch_snapshot({"retrieval": {"hybrid_search": "elasticsearch"}}) == {
        "available": False,
        "error": "injected status failure",
    }


@pytest.mark.parametrize("config", [{}, {"retrieval": {"hybrid_search": "bm25"}}])
def test_beir_disabled_does_not_request_status(monkeypatch, config):
    def unexpected(config):
        pytest.fail("Disabled Elasticsearch must not request status")

    monkeypatch.setattr(es, "get_status", unexpected)
    assert _elasticsearch_snapshot(config) is None
