"""P80: Leere Ergebnisse mit echten PG-Abschluessen sicher wiederverwenden."""

import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from psycopg2.errors import CheckViolation
from test_document_publication_memberships import _second_datasource, _use_es_sdk
from test_document_publication_pipeline import whole_pipeline  # noqa: F401
from test_document_version_completion import document_run  # noqa: F401

from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline, processing_revision, runtime
from kiara.ingestion import vectordb_ops as ops
from kiara.ingestion.ddu_client import DDUResult
from kiara.shared.db import db_query, db_transaction
from kiara.web.core_schema_migration import migrate_core_schema


def _run(case, *, incremental=True, datasource_id=None, overrides=None, **options):
    return pipeline.run_ingestion(
        case.store._config,
        incremental=incremental,
        datasource_id=case.ds if datasource_id is None else datasource_id,
        embedding_model="test-model",
        ds_overrides={"chunk_size": 100, "chunk_overlap": 0, **(overrides or {})},
        **options,
    )


def _publication(case):
    with db_query() as conn:
        return conn.execute(
            "SELECT * FROM document_publications WHERE collection_name='kb_test' AND source=%s",
            (case.path,),
        ).fetchone()


def _assert_empty_completed(case):
    row = _publication(case)
    assert row["completed_at"] is not None
    assert row["empty_file_hash"] and row["empty_processing_signature"]
    assert row["versions"] == row["cleanup_chunks"] == {}
    assert case.hashes.load(case.ds, "test-model") == {}
    with db_query() as conn:
        assert (
            conn.execute(
                "SELECT * FROM datasource_embeddings WHERE datasource_id=%s",
                (case.ds,),
            ).fetchall()
            == []
        )


def _assert_not_completed(case):
    row = _publication(case)
    assert row["completed_at"] is None
    assert row["empty_file_hash"] is row["empty_processing_signature"] is None
    assert case.hashes.load(case.ds, "test-model") == {}


@pytest.fixture
def empty_text(whole_pipeline):
    Path(whole_pipeline.path).write_text("")
    return whole_pipeline


@pytest.fixture
def empty_image(whole_pipeline, monkeypatch):
    case = whole_pipeline
    Path(case.path).unlink()
    case.path = str(Path(case.path).with_suffix(".png"))
    # Der Loader liest dieses Bild; nur der entfernte Konverter ist ersetzt.
    from PIL import Image

    Image.new("RGB", (16, 16)).save(case.path)
    case.store._config["datasource"]["file_types"] = [".png"]
    case.backend = {"url": "http://ddu.test:5001", "model": "docling", "do_ocr": True}
    monkeypatch.setattr("kiara.config.get_ddu_backends", lambda _: [case.backend])
    monkeypatch.setattr(
        "kiara.config.get_vlm_backend",
        lambda _: {"url": "http://vlm.test:11434", "model": "vision-test"},
    )
    monkeypatch.setattr(pipeline, "_check_ddu_preflight", lambda *args: None)
    case.convert = Mock(return_value=DDUResult(content="", error=""))
    monkeypatch.setattr("kiara.ingestion.ddu_client.convert_document", case.convert)
    return case


def test_empty_file_completed_once_and_resume_uses_no_loader(empty_text, monkeypatch, test_db):
    case = empty_text
    logs = []
    load = Mock(wraps=runtime.load_file)
    monkeypatch.setattr(runtime, "load_file", load)
    first = _run(case, file_log_callback=logs.append)
    assert first["empty_files"] == first["skipped_files"] == 1
    assert first["processed_files"] == first["total_chunks"] == first["errors"] == 0
    assert [log.status for log in logs] == ["skipped"]
    _assert_empty_completed(case)
    first_completed_at = _publication(case)["completed_at"]
    for options in ({}, {"resume_skip_files": {case.path}}):
        result = _run(case, **options)
        assert result["skipped_files"] == 1 and result["processed_files"] == 0
    assert load.call_count == 1
    assert _publication(case)["completed_at"] == first_completed_at
    # Gleiche Quelle und Aufbereitung: auch eine andere DS braucht keinen Parser.
    other = _second_datasource(case, test_db)
    assert _run(case, datasource_id=other)["skipped_files"] == 1
    assert load.call_count == 1


def test_png_ddu_result_is_reused_without_remote_call(empty_image):
    case = empty_image
    assert _run(case, overrides={"ddu_enabled": True})["empty_files"] == 1
    _assert_empty_completed(case)
    assert _run(case, overrides={"ddu_enabled": True})["skipped_files"] == 1
    case.convert.assert_called_once()


@pytest.mark.parametrize("previously_indexed", [False, True])
def test_empty_completion_and_skip_keep_elasticsearch_empty(
    whole_pipeline, monkeypatch, request, previously_indexed
):
    case = whole_pipeline
    _use_es_sdk(case, case.ds, monkeypatch, request)
    from kiara.retrieval.elasticsearch_service import _get_client

    monkeypatch.setattr(
        _get_client(case.store._config),
        "count",
        lambda **kw: {"count": len(case.es), "_shards": {"failed": 0, "successful": 1, "total": 1}},
    )
    if previously_indexed:
        assert _run(case)["processed_files"] == 1
        assert case.es
    Path(case.path).write_text("")
    assert _run(case)["empty_files"] == 1
    _assert_empty_completed(case)
    assert case.es == {} and case.snapshot() == []
    result = _run(case)
    assert result["skipped_files"] == 1 and result["errors"] == 0
    assert all(not docs for docs in case.es_backend.visible.values())


def test_empty_cli_source_does_not_require_chunk_memberships(empty_text):
    case = empty_text
    assert case.run_whole(None)["empty_files"] == 1
    assert case.run_whole(None)["skipped_files"] == 1
    assert case.hashes.load(None, "test-model") == {}


@pytest.mark.parametrize("change", ["ocr", "languages", "model", "url", "revision", "full"])
def test_changed_processing_or_full_run_rechecks_empty_image(empty_image, monkeypatch, change):
    case = empty_image
    _run(case, overrides={"ddu_enabled": True})
    before = _publication(case)["empty_processing_signature"]
    if change == "ocr":
        case.backend["do_ocr"] = False
    elif change == "languages":
        case.backend["ocr_lang"] = ["en"]
    elif change == "model":
        case.backend["model"] = "docling-revision-2"
    elif change == "url":
        case.backend["url"] = "http://other-ddu.test:5001"
    elif change == "revision":
        monkeypatch.setattr(processing_revision, "EMPTY_EXTRACTION_REVISION", "new-revision")
    assert (
        _run(case, incremental=change != "full", overrides={"ddu_enabled": True})["empty_files"]
        == 1
    )
    assert case.convert.call_count == 2
    assert (_publication(case)["empty_processing_signature"] != before) == (change != "full")


def test_changed_source_is_indexed_after_empty_completion(empty_text):
    case = empty_text
    _run(case)
    Path(case.path).write_text("Jetzt ist ein Inhalt vorhanden.")
    result = _run(case)
    assert result["processed_files"] == 1 and result["skipped_files"] == 0
    row = _publication(case)
    assert row["completed_at"] and row["versions"]
    assert row["empty_file_hash"] is row["empty_processing_signature"] is None
    assert case.path in case.hashes.load(case.ds, "test-model")


def test_content_hash_rechecks_changed_bytes_with_same_size_and_mtime(empty_image):
    case = empty_image
    _run(case, overrides={"ddu_enabled": True}, hash_mode="content")
    path = Path(case.path)
    before = path.stat()
    content = path.read_bytes()
    path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert _run(case, overrides={"ddu_enabled": True}, hash_mode="content")["empty_files"] == 1
    assert case.convert.call_count == 2


@pytest.mark.parametrize(
    "error",
    [
        "timeout",
        "connection",
        "http_error",
        "missing_url",
        "unexpected",
        "document_error",
        "request_error",
        "invalid_image",
        "image_too_large",
    ],
)
def test_ddu_failure_never_completes_an_empty_file(empty_image, error):
    case = empty_image
    if error == "unexpected":
        case.convert.side_effect = RuntimeError("unexpected parser failure")
    else:
        case.convert.return_value = DDUResult(content=None, error=error)
    for _ in range(2):
        result = _run(case, overrides={"ddu_enabled": True})
        assert result["errors"] == 1 and not result.get("empty_files")
        _assert_not_completed(case)
    assert case.convert.call_count == 2
    case.convert.side_effect = None
    case.convert.return_value = DDUResult(content="", error="")
    assert _run(case, overrides={"ddu_enabled": True})["empty_files"] == 1
    _assert_empty_completed(case)


def test_tiny_image_policy_skip_is_not_cached_as_extracted_empty(empty_image):
    case = empty_image
    case.convert.return_value = DDUResult(None, "image_too_small", "Bild zu klein: 1x1")
    logs = []
    for _ in range(2):
        result = _run(case, overrides={"ddu_enabled": True}, file_log_callback=logs.append)
        assert result["errors"] == 0 and result["ddu_images_skipped"] == 1
        assert result["skipped_files"] == 1 and not result.get("empty_files")
        _assert_not_completed(case)
    assert case.convert.call_count == 2
    assert all(log.status == "skipped" and "1x1" in log.error_message for log in logs)


@pytest.mark.parametrize("ddu_content", ["", "Ein teilweise erkanntes Dokument."])
def test_vlm_failure_stays_retryable_after_prior_empty_completion(
    empty_image, monkeypatch, ddu_content
):
    case = empty_image
    _run(case, overrides={"ddu_enabled": True})
    case.convert.return_value = DDUResult(content=ddu_content, error="")
    describe = Mock(return_value=None)
    monkeypatch.setattr(runtime, "describe_image", describe)
    for _ in range(2):
        result = _run(case, overrides={"ddu_enabled": True, "vlm_enabled": True})
        assert result["errors"] == result["vlm_errors"] == 1
        assert not result.get("empty_files")
        _assert_not_completed(case)
        assert case.snapshot() == []
    assert describe.call_count == 2 and case.convert.call_count == 3


@pytest.mark.parametrize("feature", ["ddu_enabled", "vlm_enabled"])
def test_missing_requested_backend_does_not_cache_empty(empty_text, monkeypatch, feature):
    case = empty_text
    monkeypatch.setattr("kiara.config.get_ddu_backends", lambda _: [])
    monkeypatch.setattr("kiara.config.get_vlm_backend", lambda _: None)
    result = _run(case, overrides={feature: True})
    assert not result.get("empty_files")
    _assert_not_completed(case)


class _HardStop(BaseException):
    pass


@pytest.mark.parametrize("failure", [RuntimeError, _HardStop])
def test_recount_failure_rolls_back_empty_completion_before_log(empty_text, monkeypatch, failure):
    case = empty_text
    recount = ops._recount_datasource_embeddings
    logs = []

    def fail_after_completion(conn, *args):
        row = conn.execute(
            "SELECT empty_file_hash FROM document_publications WHERE source=%s", (case.path,)
        ).fetchone()
        if row and row["empty_file_hash"]:
            raise failure("after empty write")
        return recount(conn, *args)

    with monkeypatch.context() as patch:
        patch.setattr(ops, "_recount_datasource_embeddings", fail_after_completion)
        with pytest.raises(failure, match="after empty write"):
            _run(case, file_log_callback=logs.append)
    assert not logs
    _assert_not_completed(case)
    assert _run(case)["empty_files"] == 1
    _assert_empty_completed(case)


def test_source_changed_after_cleanup_cannot_complete_empty(empty_text, monkeypatch):
    case = empty_text
    prepare = pipeline._prepare_file_chunks

    def change_after_cleanup(*args, **kwargs):
        result = prepare(*args, **kwargs)
        Path(case.path).write_text("Aenderung waehrend der Verarbeitung")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(pipeline, "_prepare_file_chunks", change_after_cleanup)
        with pytest.raises(OSError, match="waehrend der Verarbeitung"):
            _run(case)
    _assert_not_completed(case)
    assert _run(case)["processed_files"] == 1


def test_failed_cleanup_preserves_retry_then_completes_empty(whole_pipeline, monkeypatch):
    case = whole_pipeline
    _run(case)
    Path(case.path).write_text("")
    with monkeypatch.context() as patch:
        patch.setattr(
            case.store, "delete_by_sources", Mock(side_effect=RuntimeError("cleanup failed"))
        )
        with pytest.raises(RuntimeError, match="cleanup failed"):
            _run(case)
    _assert_not_completed(case)
    assert _run(case)["empty_files"] == 1
    assert case.snapshot() == []
    _assert_empty_completed(case)


def test_open_publication_and_explicit_deletion_invalidate_empty_cache(empty_text):
    case = empty_text
    _run(case)
    publication.invalidate_documents(
        "kb_test", [case.path], "test-model", hash_datasource_id=case.ds
    )
    _assert_not_completed(case)
    assert _run(case)["empty_files"] == 1
    ops.delete_files(
        case.store._config,
        case.ds,
        "test-model",
        [case.path],
        chunk_ds=runtime.ChunkDatasources(),
        hash_store=case.hashes,
    )
    _assert_not_completed(case)
    assert _run(case)["empty_files"] == 1


def test_database_rejects_empty_completion_with_pending_cleanup(empty_text):
    case = empty_text
    _run(case)
    with pytest.raises(CheckViolation), db_transaction() as conn:
        conn.execute(
            "UPDATE document_publications SET cleanup_chunks = %s::jsonb WHERE source=%s",
            ('{"test-model": []}', case.path),
        )
    _assert_empty_completed(case)


def test_schema_upgrade_retains_existing_publications_and_is_idempotent(empty_text, test_db):
    case = empty_text
    _run(case)
    before = _publication(case)
    test_db.execute(
        "ALTER TABLE document_publications DROP COLUMN empty_file_hash CASCADE, "
        "DROP COLUMN empty_processing_signature CASCADE"
    )
    migrate_core_schema(test_db)
    migrate_core_schema(test_db)
    test_db.commit()
    after = _publication(case)
    assert after["generation"] == before["generation"]
    assert after["completed_at"] == before["completed_at"]
    assert after["empty_file_hash"] is after["empty_processing_signature"] is None
    assert _run(case)["empty_files"] == 1
