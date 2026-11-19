from contextlib import nullcontext
from unittest.mock import Mock

import httpx
import pytest
from reportlab.pdfgen.canvas import Canvas
from test_ingestion_replay_p64 import diff_run  # noqa: F401
from test_loader_content_format import BACKEND, _process

from kiara.ingestion import pipeline

FAILURE = {
    "status": "failure",
    "task_id": "task-test",
    "errors": [{"code": "conversion_failed", "message": "page dimensions"}],
}


def _two_pages(path, first="Nativer Text auf Seite eins"):
    pdf = Canvas(str(path))
    if first:
        pdf.drawString(80, 700, first)
    pdf.showPage()
    pdf.showPage()
    pdf.save()


def _server(monkeypatch, primary, pages, *, code=200):
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs["data"]))
        is_ocr = url.endswith("/pdf-pages/ocr")
        return httpx.Response(
            code if is_ocr else 200,
            json=({"pages": pages} if is_ocr else primary),
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", post)
    return calls


def test_recover_only_the_missing_page_and_log_the_task(tmp_path, monkeypatch):
    path = tmp_path / "plan.pdf"
    _two_pages(path)
    calls = _server(monkeypatch, FAILURE, [{"page": 2, "status": "text", "text": "Maßstab 1:50"}])
    result, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert len(calls) == 2 and calls[1][1]["pages"] == ["2"]
    assert log.parser == "pdf-page-ocr" and log.parser_fallback
    assert log.status == "success" and stats["errors"] == 0
    assert "task-test" in log.error_message and "OCR: 2" in log.error_message
    assert "Seite eins" in " ".join(result[0]) and "Maßstab" in " ".join(result[0])


@pytest.mark.parametrize("status", ["success", "partial_success"])
def test_keep_primary_markdown_and_add_ocr(tmp_path, monkeypatch, status):
    path = tmp_path / "plan.pdf"
    _two_pages(path)
    primary = {
        "status": status,
        "task_id": "task-partial" if status == "partial_success" else None,
        "document": {
            "md_content": "# Formatierte Tabelle",
            "source_page_count": 2,
            "missing_text_pages": [2],
        },
    }
    calls = _server(monkeypatch, primary, [{"page": 2, "status": "text", "text": "Blatt 2"}])
    result, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert calls[1][1]["pages"] == ["2"]
    assert log.status == "success" and "Formatierte Tabelle" in " ".join(result[0])
    assert "OCR: 2" in log.error_message
    assert bool(stats.get("ddu_document_errors")) == (status == "partial_success")
    if status == "partial_success":
        assert "task-partial" in log.error_message
    assert "Blatt 2" in " ".join(result[0])


def test_only_primary_text_keeps_docling_as_parser(tmp_path, monkeypatch):
    path = tmp_path / "plan.pdf"
    _two_pages(path)
    primary = {
        "status": "success",
        "document": {
            "md_content": "Vorhandene Beschriftung",
            "source_page_count": 2,
            "missing_text_pages": [2],
        },
    }
    _server(monkeypatch, primary, [{"page": 2, "status": "no_text", "text": ""}])
    result, log, stats = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert result[0] and log.status == "success" and log.parser == "docling-remote"
    assert "ohne Text: 2" in log.error_message and stats["ddu_remote_files"] == 1


@pytest.mark.parametrize("ocr", [False, True])
def test_placeholders_are_not_an_empty_completion_or_content(tmp_path, monkeypatch, ocr):
    path = tmp_path / "plan.pdf"
    _two_pages(path, first="")
    primary = {
        "status": "success",
        "document": {
            "md_content": "<!-- image -->",
            "source_page_count": 2,
            "missing_text_pages": [1, 2],
        },
    }
    calls = _server(
        monkeypatch, primary, [{"page": i, "status": "no_text", "text": ""} for i in [1, 2]]
    )
    result, log, _ = _process(path, ddu_enabled=True, ddu_backends=[{**BACKEND, "do_ocr": ocr}])
    assert len(calls) == (2 if ocr else 1)
    assert log.status == "error" and not log.empty_content_confirmed and not result[0]


@pytest.mark.parametrize(
    ("response_code", "rows", "infra"),
    [
        (503, [], True),
        (422, [], False),
        (200, [], True),
        (200, [{"page": 1, "status": "text", "text": "wrong page"}], True),
    ],
)
def test_failed_ocr_does_not_publish_incomplete_text(
    tmp_path, monkeypatch, response_code, rows, infra
):
    path = tmp_path / "plan.pdf"
    _two_pages(path)
    calls = _server(monkeypatch, FAILURE, rows, code=response_code)
    result, log, _ = _process(path, ddu_enabled=True, ddu_backends=[BACKEND])
    assert len(calls) == 2 and not result[0]
    assert log.status == "error" and log.is_infra_error is infra
    assert "task-test" in log.error_message and "Seiten 2" in log.error_message


def test_retry_never_cleans_existing_files_before_extraction(diff_run, tmp_path):
    config, path, hashes, cleanup = diff_run
    config["_retry_file_paths"] = [path]
    hashes.load.return_value = {path: "old", str(tmp_path / "unrelated.pdf"): "old"}
    result = pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    assert result[0] == result[1] == [path] and result[3] == []
    assert cleanup.call_args.args[-1] == []
    hashes.load.assert_not_called()


def test_reject_retry_outside_source_before_cleanup(diff_run):
    config, _, _, cleanup = diff_run
    config["_retry_file_paths"] = ["/outside.pdf"]
    with pytest.raises(ValueError, match="ausserhalb"):
        pipeline._load_and_diff_files(config, Mock(), True, 1, "nomic-embed-text")
    cleanup.assert_not_called()


def test_subset_progress_updates_only_the_job(monkeypatch):
    from kiara.web import ingestion_worker

    db = Mock()
    monkeypatch.setattr(ingestion_worker, "db_transaction", lambda: nullcontext(db))
    callback = ingestion_worker.IngestionWorker()._build_callbacks(
        70, 1, update_source_progress=False
    )["on_progress"]
    callback(12, 125)
    db.execute.assert_called_once()
    statement, values = db.execute.call_args.args
    assert "UPDATE ingestion_jobs" in statement and values == (12, 125, 70)


def test_subset_keeps_full_source_file_count_and_updates_chunk_count(monkeypatch):
    from kiara.web import ingestion_worker

    db = Mock()
    db.execute.return_value.fetchone.return_value = {
        "stats_json": {"retry_file_log_ids": [13, 14], "retry_source_file_count": 400}
    }
    for repo in ("_emb_repo", "_ds_repo", "_job_repo"):
        monkeypatch.setattr(ingestion_worker, repo, Mock())
    ingestion_worker.IngestionWorker()._update_datasource_metadata(
        db, 70, 1, {"total_files": 2, "processed_files": 2}, {}, "bge-m3", 880, None
    )
    counts = ingestion_worker._emb_repo.upsert_stats.call_args.kwargs
    assert (counts["chunk_count"], counts["file_count"]) == (880, 400)
    assert "processed_files" not in ingestion_worker._ds_repo.update.call_args.kwargs


def test_interrupted_subset_restores_model_inventory():
    from kiara.web import ingestion_worker

    db = Mock()
    db.execute.return_value.fetchone.return_value = {
        "stats_json": {"retry_file_log_ids": [13], "retry_source_file_count": 400}
    }
    stats = {}
    ingestion_worker._preserve_interrupted_selection(db, 70, stats)
    assert stats["retry_file_log_ids"] == [13]
    statement, values = db.execute.call_args.args
    assert "UPDATE datasource_embeddings" in statement and values == (400, 70)
