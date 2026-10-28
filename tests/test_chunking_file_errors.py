"""Inhaltliche Tabellenfehler bleiben dateilokal und im Jobprotokoll sichtbar."""

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest
from kiara_ingestion_core.models import ProcessingContext

from kiara.exceptions import ChunkingContentError
from kiara.ingestion import runtime
from kiara.ingestion.markdown_chunks import prepare_markdown_chunks
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


@pytest.mark.parametrize("error", [ValueError("interner Fehler"), RuntimeError("DB-Fehler")])
def test_unclassified_errors_are_not_swallowed(file_run, monkeypatch, tmp_path, error):
    monkeypatch.setattr(runtime, "_process_single_file", Mock(side_effect=error))
    with pytest.raises(type(error), match=str(error)):
        runtime._process_file_with_publication(file_run.ictx, str(tmp_path / "mail.txt"), Mock())
    assert file_run.ictx.stats["errors"] == 0
    file_run.ictx.file_completion_callback.assert_not_called()


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
