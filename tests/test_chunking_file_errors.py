"""Inhaltliche Tabellenfehler bleiben dateilokal und im Jobprotokoll sichtbar."""

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import psycopg2
import pytest
from kiara_ingestion_core.models import ProcessingContext

from kiara.backend_selector import BackendUnavailableError
from kiara.exceptions import ChunkingContentError, ConfigurationError
from kiara.ingestion import runtime
from kiara.ingestion._file_utils import _CancelledError
from kiara.ingestion.markdown_chunks import prepare_markdown_chunks
from kiara.retrieval.index_compatibility import IndexCompatibilityError
from kiara.shared.constants import INGESTION_MAX_CONSECUTIVE_ERRORS

_BAD_TABLES = [
    ("| " + "Kopf " * 250 + " |\n| --- |", "Tabellenkopf überschreitet"),
    ("| " + "Feld " * 250 + " |\n| --- |\n| Wert |", "Feldnamen"),
    ("| Artikel | Preis |\n| --- | --- |\n| A | 17 EUR | extra |", "Spaltenzahl"),
]


@pytest.fixture
def file_run(monkeypatch, tmp_path):
    logs, progress, persisted = [], [], {}
    store = Mock()
    store.finalize_document_versions.side_effect = lambda ids, *, publish_search, on_complete: (
        on_complete({})
    )
    hashes = Mock()
    hashes.remove_files.side_effect = lambda ds, model, paths: [
        persisted.pop(path, None) for path in paths
    ]
    complete = Mock(side_effect=lambda path: persisted.update({path: "completed"}))
    prepare = Mock(return_value=False)
    ictx = runtime._IngestionContext(
        config={"ingestion": {"chunk_size": 1000, "chunk_overlap": 200}},
        ctx=ProcessingContext(),
        store=store,
        chunk_ds=Mock(),
        hash_store=hashes,
        batch_size=10,
        datasource_id=12,
        embedding_model="test-model",
        hash_mode="mtime",
        ds_overrides={"chunk_strategy": "auto"},
        file_hashes={},
        stats={"processed_files": 0, "total_chunks": 0, "errors": 0},
        pending_hashes={},
        file_completion_callback=complete,
        file_preparation_callback=prepare,
        file_log_callback=logs.append,
        progress_callback=lambda *args: progress.append(args),
    )

    def load(path, **kwargs):
        return {
            "text": Path(path).read_text(),
            "metadata": {
                "source": path,
                "file_type": ".txt",
                "content_format": "markdown",
                "source_mime_type": "text/html",
                "parser": "html-to-markdown",
            },
        }

    monkeypatch.setattr(runtime, "load_file", load)
    dedup = Mock(return_value=(runtime._DedupResult.FULL, [], [], [], [], None))
    monkeypatch.setattr(runtime, "_handle_chunk_dedup", dedup)
    return SimpleNamespace(
        ictx=ictx, logs=logs, progress=progress, persisted=persisted, dedup=dedup
    )


@pytest.mark.parametrize("with_file_logs", [True, False])
def test_chunking_errors_are_counted_and_later_file_completes(file_run, tmp_path, with_file_logs):
    run = file_run
    if not with_file_logs:
        run.ictx.file_log_callback = None
    count = max(INGESTION_MAX_CONSECUTIVE_ERRORS + 1, len(_BAD_TABLES))
    bad_paths = []
    for number in range(count):
        path = tmp_path / f"bad-{number}.txt"
        path.write_text(_BAD_TABLES[number % len(_BAD_TABLES)][0])
        bad_paths.append(str(path))
        run.persisted[str(path)] = "old-hash"
    good = tmp_path / "valid.txt"
    good.write_text("Diese Mail kann vollstaendig verarbeitet werden.")
    run.ictx.total = count + 1

    for path in [*bad_paths, str(good)]:
        assert runtime._process_file_with_publication(run.ictx, path, Mock()) is True

    assert run.ictx.stats["errors"] == count
    assert run.ictx.stats["processed_files"] == 1
    assert "abort_reason" not in run.ictx.stats
    assert run.ictx.consecutive_errors == 0
    assert run.ictx.skipped == count
    assert run.progress == [(number, count + 1) for number in range(1, count + 2)]
    assert run.persisted == {str(good): "completed"}
    assert not run.ictx.pending_hashes
    assert run.ictx.hash_store.remove_files.call_args_list == [
        call(12, "test-model", [path]) for path in bad_paths
    ]
    run.ictx.file_completion_callback.assert_called_once_with(str(good))
    run.ictx.file_preparation_callback.assert_called_once()
    assert run.ictx.file_preparation_callback.call_args.args[0] == str(good)
    run.dedup.assert_called_once()
    assert len(run.ictx.stats["error_details"]) == count
    for number, detail in enumerate(run.ictx.stats["error_details"]):
        assert detail["file"] == Path(bad_paths[number]).name
        assert detail["phase"] == "chunking"
        assert _BAD_TABLES[number % len(_BAD_TABLES)][1] in detail["error"]
    if with_file_logs:
        assert [entry.status for entry in run.logs] == ["error"] * count + ["success"]
        assert all(entry.error_phase == "chunking" for entry in run.logs[:-1])
        assert all(not entry.is_infra_error for entry in run.logs)
        assert all(entry.error_message for entry in run.logs[:-1])
    else:
        assert not run.logs


@pytest.mark.parametrize("text,message", _BAD_TABLES)
def test_table_content_errors_have_an_explicit_type(text, message):
    with pytest.raises(ChunkingContentError, match=message):
        prepare_markdown_chunks(text, 640, 100)


@pytest.mark.parametrize("size,overlap", [(0, 0), (640, 640), (640, -1)])
def test_invalid_chunk_configuration_still_stops_the_run(file_run, tmp_path, size, overlap):
    path = tmp_path / "valid.txt"
    path.write_text("Gueltiger Inhalt")
    file_run.ictx.config["ingestion"].update(chunk_size=size, chunk_overlap=overlap)
    file_run.ictx.ds_overrides = {"chunk_size": size}
    with pytest.raises(ValueError, match="Ungültiges Chunk-Budget"):
        runtime._process_file_with_publication(file_run.ictx, str(path), Mock())
    assert file_run.ictx.stats["errors"] == 0
    assert not file_run.logs
    file_run.dedup.assert_not_called()


def _failing_load(bad_names, error_factory):
    """Loader-Mock: wirft fuer die genannten Dateien eine unerwartete Ausnahme."""

    def load(path, **kwargs):
        if Path(path).name in bad_names:
            raise error_factory()
        return {
            "text": Path(path).read_text(),
            "metadata": {"source": path, "file_type": ".txt", "parser": "text"},
        }

    return load


def _write_files(tmp_path, names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text(f"Inhalt von {name}")
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("with_file_logs", [True, False])
def test_unexpected_loader_error_stays_file_local(file_run, monkeypatch, tmp_path, with_file_logs):
    """Gitea #8: Programmfehler einer Datei beendet nicht den ganzen Job."""
    run = file_run
    if not with_file_logs:
        run.ictx.file_log_callback = None
    monkeypatch.setattr(
        runtime,
        "load_file",
        _failing_load({"b.odt"}, lambda: UnboundLocalError("ddu_error nicht belegt")),
    )
    paths = _write_files(tmp_path, ["a.txt", "b.odt", "c.txt"])
    run.persisted[paths[1]] = "old-hash"
    run.ictx.total = 3

    for path in paths:
        assert runtime._process_file_with_publication(run.ictx, path, Mock()) is True

    assert run.ictx.stats["errors"] == 1
    assert run.ictx.stats["unexpected_file_errors"] == 1
    assert run.ictx.stats["processed_files"] == 2
    assert "abort_reason" not in run.ictx.stats
    assert run.persisted == {paths[0]: "completed", paths[2]: "completed"}
    run.ictx.hash_store.remove_files.assert_called_once_with(12, "test-model", [paths[1]])
    detail = run.ictx.stats["error_details"][0]
    assert detail["file"] == "b.odt"
    assert detail["phase"] == "loading"
    assert detail["error"].startswith("UnboundLocalError: ddu_error nicht belegt (")
    assert run.ictx.unexpected_error_streak == 0
    if with_file_logs:
        assert [entry.status for entry in run.logs] == ["success", "error", "success"]
        assert run.logs[1].error_phase == "loading"
        assert not run.logs[1].is_infra_error


def test_unexpected_chunking_error_is_reported_as_chunking(file_run, monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_chunk_and_collect", Mock(side_effect=KeyError("spans")))
    (path,) = _write_files(tmp_path, ["a.txt"])

    assert runtime._process_file_with_publication(file_run.ictx, path, Mock()) is True

    assert file_run.ictx.stats["errors"] == 1
    assert file_run.ictx.stats["error_details"][0]["phase"] == "chunking"
    assert [entry.error_phase for entry in file_run.logs] == ["chunking"]
    file_run.ictx.file_completion_callback.assert_not_called()


def _bare(exc_type):
    return exc_type.__new__(exc_type)


@pytest.mark.parametrize(
    "error_factory",
    [
        lambda: psycopg2.OperationalError("canceling statement due to statement timeout"),
        lambda: _bare(IndexCompatibilityError),
        lambda: BackendUnavailableError("Docling nicht erreichbar"),
        lambda: ConfigurationError("kaputte Konfiguration"),
        lambda: _CancelledError(),
        lambda: MemoryError(),
    ],
)
def test_job_stopping_errors_are_not_swallowed(file_run, monkeypatch, tmp_path, error_factory):
    error = error_factory()
    monkeypatch.setattr(runtime, "_process_single_file", Mock(side_effect=error))
    with pytest.raises(type(error)):
        runtime._process_file_with_publication(file_run.ictx, str(tmp_path / "mail.txt"), Mock())
    assert file_run.ictx.stats["errors"] == 0
    assert not file_run.logs
    file_run.ictx.file_completion_callback.assert_not_called()


def test_unexpected_error_avalanche_aborts_job(file_run, monkeypatch, tmp_path):
    run = file_run
    run.ictx.config["ingestion"]["max_consecutive_unexpected_errors"] = 3
    names = ["bad-1.txt", "bad-2.txt", "good.txt", "bad-3.txt", "bad-4.txt", "bad-5.txt"]
    monkeypatch.setattr(
        runtime,
        "load_file",
        _failing_load({n for n in names if n.startswith("bad")}, lambda: RuntimeError("Bug")),
    )
    paths = _write_files(tmp_path, names)

    results = [runtime._process_file_with_publication(run.ictx, path, Mock()) for path in paths]

    # Die erfolgreiche Datei unterbricht die Serie; erst drei gleiche
    # Fehler direkt hintereinander brechen ab.
    assert results == [True, True, True, True, True, False]
    assert run.ictx.stats["errors"] == 5
    assert run.ictx.stats["processed_files"] == 1
    assert run.ictx.stats["abort_reason"].startswith(
        "Abbruch: 3 Dateien hintereinander mit demselben unerwarteten Fehler (RuntimeError in "
    )
    assert [entry.status for entry in run.logs] == ["error"] * 2 + ["success"] + ["error"] * 3
    assert run.ictx.hash_store.remove_files.call_count == 5


def test_different_unexpected_errors_do_not_add_up(file_run, monkeypatch, tmp_path):
    run = file_run
    run.ictx.config["ingestion"]["max_consecutive_unexpected_errors"] = 2
    errors = iter([RuntimeError("eins"), KeyError("zwei"), RuntimeError("drei")])

    def load(path, **kwargs):
        raise next(errors)

    monkeypatch.setattr(runtime, "load_file", load)
    paths = _write_files(tmp_path, ["a.txt", "b.txt", "c.txt"])

    assert [runtime._process_file_with_publication(run.ictx, p, Mock()) for p in paths] == [
        True,
        True,
        True,
    ]
    assert "abort_reason" not in run.ictx.stats


def test_worker_persists_file_errors_and_finishes_job(file_run, test_db, tmp_path, monkeypatch):
    from kiara.web.ingestion_worker import IngestionWorker

    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type) VALUES ('Chunking-Test', 'local') RETURNING id"
    ).fetchone()["id"]
    job_id = test_db.execute(
        "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, 'running') RETURNING id",
        (ds_id,),
    ).fetchone()["id"]
    test_db.commit()
    worker = IngestionWorker()
    callbacks = worker._build_callbacks(job_id, ds_id)
    ictx = file_run.ictx
    ictx.datasource_id = ds_id
    ictx.total = 2
    ictx.stats["total_files"] = 2
    ictx.file_log_callback = callbacks["on_file_log"]
    ictx.progress_callback = callbacks["on_progress"]
    for name, text in (("bad.txt", _BAD_TABLES[0][0]), ("good.txt", "Gueltiger Inhalt")):
        path = tmp_path / name
        path.write_text(text)
        assert runtime._process_file_with_publication(ictx, str(path), Mock()) is True
    callbacks["flush_file_logs"]()

    monkeypatch.setattr(worker, "_aggregate_final_stats", Mock(return_value=(1, datetime.now())))
    monkeypatch.setattr(worker, "_log_job_completion", Mock())
    worker._resolve_job_outcome(
        job_id, ds_id, ictx.config, ictx.embedding_model, ictx.stats, 1, lambda: None
    )
    job = test_db.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    assert job["status"] == "success"
    assert job["progress_processed"] == job["progress_total"] == 2
    stats = job["stats_json"]
    assert stats["errors"] == stats["processed_files"] == 1
    logs = test_db.execute(
        "SELECT file_name, status, error_phase, error_message FROM ingestion_file_logs "
        "WHERE job_id = %s ORDER BY id",
        (job_id,),
    ).fetchall()
    assert [(log["file_name"], log["status"]) for log in logs] == [
        ("bad.txt", "error"),
        ("good.txt", "success"),
    ]
    assert logs[0]["error_phase"] == "chunking"
    assert "Tabellenkopf überschreitet" in logs[0]["error_message"]


def test_worker_finishes_job_after_unexpected_file_error(file_run, test_db, tmp_path, monkeypatch):
    from kiara.web.ingestion_worker import IngestionWorker

    ds_id = test_db.execute(
        "INSERT INTO datasources (name, type) VALUES ('Issue-8-Test', 'local') RETURNING id"
    ).fetchone()["id"]
    job_id = test_db.execute(
        "INSERT INTO ingestion_jobs (datasource_id, status) VALUES (%s, 'running') RETURNING id",
        (ds_id,),
    ).fetchone()["id"]
    test_db.commit()
    worker = IngestionWorker()
    callbacks = worker._build_callbacks(job_id, ds_id)
    ictx = file_run.ictx
    ictx.datasource_id = ds_id
    ictx.total = 3
    ictx.stats["total_files"] = 3
    ictx.file_log_callback = callbacks["on_file_log"]
    ictx.progress_callback = callbacks["on_progress"]
    monkeypatch.setattr(
        runtime, "load_file", _failing_load({"b.odt"}, lambda: RuntimeError("Loader-Bug"))
    )
    for path in _write_files(tmp_path, ["a.txt", "b.odt", "c.txt"]):
        assert runtime._process_file_with_publication(ictx, path, Mock()) is True
    callbacks["flush_file_logs"]()

    monkeypatch.setattr(worker, "_aggregate_final_stats", Mock(return_value=(2, datetime.now())))
    monkeypatch.setattr(worker, "_log_job_completion", Mock())
    worker._resolve_job_outcome(
        job_id, ds_id, ictx.config, ictx.embedding_model, ictx.stats, 1, lambda: None
    )
    job = test_db.execute("SELECT * FROM ingestion_jobs WHERE id = %s", (job_id,)).fetchone()
    assert job["status"] == "success"
    stats = job["stats_json"]
    if isinstance(stats, str):
        stats = json.loads(stats)
    assert stats["errors"] == 1
    assert stats["unexpected_file_errors"] == 1
    assert stats["processed_files"] == 2
    logs = test_db.execute(
        "SELECT file_name, status, error_phase, error_message FROM ingestion_file_logs "
        "WHERE job_id = %s ORDER BY id",
        (job_id,),
    ).fetchall()
    assert [(log["file_name"], log["status"]) for log in logs] == [
        ("a.txt", "success"),
        ("b.odt", "error"),
        ("c.txt", "success"),
    ]
    assert logs[1]["error_phase"] == "loading"
    assert logs[1]["error_message"].startswith("RuntimeError: Loader-Bug (")
