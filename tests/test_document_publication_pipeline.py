"""Vollstaendige Pipeline: Planung, Eingangsvergleich, Wiederanlauf und CLI."""

import os
import threading
from pathlib import Path

import pytest
from test_document_publication_memberships import _second_datasource, _use_es_sdk, _use_real_guards
from test_document_version_completion import document_run, read  # noqa: F401
from test_index_compatibility_s4 import _resolved

from kiara.exceptions import RetrievalError
from kiara.ingestion import document_publication as publication
from kiara.ingestion import pipeline, runtime
from kiara.ingestion import vectordb_ops as ops
from kiara.ingestion._file_utils import ChunkDatasources
from kiara.retrieval import index_compatibility as ic


@pytest.fixture
def whole_pipeline(document_run, test_db, monkeypatch):
    case = document_run
    _use_real_guards(case, test_db, monkeypatch)
    config = case.store._config
    config["datasource"] = {
        "path": str(Path(case.path).parent),
        "file_types": [".txt"],
        "exclude_patterns": [],
    }
    config["vectordb"]["path"] = str(Path(case.path).parent / "vector-cache")
    config["ingestion"]["batch_size"] = 2
    monkeypatch.setattr(
        ic, "resolve_current_index_contract", lambda *a, **kw: _resolved(case.store._index_target)
    )
    monkeypatch.setattr(pipeline, "VectorStore", lambda *a, **kw: case.store)
    add = case.store.add

    def embed(texts, metas, ids):
        case.embeddings += len(ids)
        return add(texts, metas, ids)

    monkeypatch.setattr(case.store, "add", embed)

    def run(ds, *, incremental=True, chunk_size=100, **options):
        return pipeline.run_ingestion(
            config,
            incremental=incremental,
            datasource_id=ds,
            embedding_model="test-model",
            ds_overrides={
                "chunk_size": chunk_size,
                "chunk_overlap": 0,
                "contextual_retrieval": True,
            },
            **options,
        )

    case.run_whole = run
    return case


def _assert_readable(case):
    tool, ref = case.reader()
    assert read(tool, ref, max_chunks=3).success


def _assert_shared(case, db, owners):
    chunks = case.snapshot()
    pg = ChunkDatasources().get_datasource_ids_for_chunks(
        [c["_doc_id"] for c in chunks], "test-model"
    )
    assert all(sorted(ds) == owners for ds in pg.values())
    assert all(c["metadata"]["ds_ids"] == owners for c in chunks)
    for ds in owners:
        assert case.path in case.hashes.load(ds, "test-model")
        row = db.execute(
            "SELECT chunk_count, file_count FROM datasource_embeddings "
            "WHERE datasource_id=%s AND embedding_model='test-model'",
            (ds,),
        ).fetchone()
        assert (row["chunk_count"], row["file_count"]) == (len(chunks), 1)


def test_incremental_shared_source_is_adopted_without_embedding(whole_pipeline, test_db):
    case = whole_pipeline
    ds2 = _second_datasource(case, test_db)
    case.prefix = "Erhaltener Kontext: "
    assert case.run_whole(case.ds)["processed_files"] == 1
    before = case.snapshot()
    case.enrich.reset_mock()
    adopted = case.run_whole(ds2)
    assert adopted["processed_files"] == 1 and adopted["dedup_chunks"] == 3
    skipped = case.run_whole(case.ds)
    assert skipped["processed_files"] == 0 and skipped["skipped_files"] == 1
    assert case.embeddings == 3
    case.enrich.assert_not_called()
    assert [c["text"] for c in case.snapshot()] == [c["text"] for c in before]
    _assert_shared(case, test_db, [case.ds, ds2])
    _assert_readable(case)


@pytest.mark.parametrize("remove_failed_ds", [False, True])
def test_matching_hash_retries_another_datasources_failed_publication(
    whole_pipeline, test_db, monkeypatch, remove_failed_ds
):
    case = whole_pipeline
    ds2 = _second_datasource(case, test_db)
    second_ready, first_done = threading.Event(), threading.Event()
    active = threading.local()
    results, errors = {}, {}
    process, set_payload = runtime._process_file_in_pipeline, case.client.set_payload

    def rendezvous(ctx, path):
        active.ds = ctx.datasource_id
        if ctx.datasource_id == case.ds:
            assert second_ready.wait(5)
        else:
            second_ready.set()
            assert first_done.wait(5)
        return process(ctx, path)

    def fail_second(*args, **kwargs):
        if getattr(active, "ds", None) == ds2 and "processing_version" in kwargs.get("payload", {}):
            raise RuntimeError("Injected publication failure")
        return set_payload(*args, **kwargs)

    def run(ds):
        try:
            results[ds] = case.run_whole(ds)
        except BaseException as exc:
            errors[ds] = exc
        finally:
            if ds == case.ds:
                first_done.set()

    with monkeypatch.context() as race:
        race.setattr(pipeline, "_process_file_in_pipeline", rendezvous)
        race.setattr(case.client, "set_payload", fail_second)
        workers = [threading.Thread(target=run, args=(ds,)) for ds in (case.ds, ds2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=15)
        assert all(not worker.is_alive() for worker in workers)
    assert case.ds not in errors and results[case.ds]["processed_files"] == 1
    assert isinstance(errors.get(ds2), RuntimeError)
    assert "Injected publication failure" in str(errors[ds2])
    assert case.path in case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader()
    if remove_failed_ds:
        ops.delete_files(
            case.store._config,
            ds2,
            "test-model",
            [case.path],
            chunk_ds=ChunkDatasources(),
            hash_store=case.hashes,
        )
    case.enrich.reset_mock()
    retried = case.run_whole(case.ds)
    assert retried["processed_files"] == 1 and retried["skipped_files"] == 0
    assert retried["dedup_chunks"] == 3 and case.embeddings == 3
    case.enrich.assert_not_called()
    _assert_readable(case)
    again = case.run_whole(case.ds)
    assert again["skipped_files"] == 1 and again["processed_files"] == 0


@pytest.mark.parametrize("repair_prefix", ["Alt: ", "Neu: "])
def test_full_pipeline_repair_preserves_healthy_enrichment(whole_pipeline, repair_prefix):
    case = whole_pipeline
    case.prefix = "Alt: "
    assert case.run_whole(case.ds)["processed_files"] == 1
    before = case.snapshot()
    case.store.delete_by_ids([before[1]["_doc_id"]])
    publication.invalidate_documents(
        "kb_test", [case.path], "test-model", hash_datasource_id=case.ds
    )
    case.prefix = repair_prefix
    case.enrich.reset_mock()
    stats = case.run_whole(case.ds)
    after = case.snapshot()
    assert stats["dedup_chunks"] == 2 and case.embeddings == 4
    assert case.enrich.call_args.kwargs["chunk_texts"] == ["B" * 100]
    assert after[0]["text"] == before[0]["text"] and after[2]["text"] == before[2]["text"]
    assert after[1]["text"] == repair_prefix + "B" * 100
    _assert_readable(case)


def test_content_change_with_same_positions_and_stat_is_replaced(whole_pipeline, test_db):
    case = whole_pipeline
    ds2 = _second_datasource(case, test_db)
    case.prefix = "Kontext: "
    case.run_whole(case.ds, hash_mode="content")
    case.run_whole(ds2, hash_mode="content")
    before = case.snapshot()
    old_reader, old_ref = case.reader(before)
    assert read(old_reader, old_ref, max_chunks=1).success
    stat = Path(case.path).stat()
    Path(case.path).write_text("X" * 100 + "B" * 100 + "C" * 100)
    os.utime(case.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    stats = case.run_whole(case.ds, hash_mode="content")
    assert stats["total_chunks"] == 3 and case.embeddings == 6
    after = case.snapshot()
    assert [c["_doc_id"] for c in before] == [c["_doc_id"] for c in after]
    assert after[0]["text"] == case.prefix + "X" * 100
    assert (
        before[0]["metadata"]["processing_input_version"]
        != after[0]["metadata"]["processing_input_version"]
    )
    with pytest.raises(RetrievalError, match="veraendert"):
        read(old_reader, old_ref, start_chunk=1)
    # Die ersetzte Darstellung entwertet alle alten Hashes; DS2 uebernimmt neu.
    assert case.path not in case.hashes.load(ds2, "test-model")
    case.enrich.reset_mock()
    assert case.run_whole(ds2, hash_mode="content")["dedup_chunks"] == 3
    assert case.embeddings == 6
    case.enrich.assert_not_called()
    _assert_shared(case, test_db, [case.ds, ds2])
    _assert_readable(case)


def test_new_datasource_with_other_chunking_replaces_old_boundaries(whole_pipeline, test_db):
    case = whole_pipeline
    ds2 = _second_datasource(case, test_db)
    case.run_whole(case.ds)
    before = case.snapshot()
    stats = case.run_whole(ds2, chunk_size=110)
    after = case.snapshot()
    assert stats["total_chunks"] == 3 and case.embeddings == 6
    assert before[0]["metadata"]["chunk_total"] == after[0]["metadata"]["chunk_total"] == 3
    assert after[0]["metadata"]["chunk_size_used"] == 110
    assert all(c["metadata"]["ds_ids"] == [ds2] for c in after)


def test_shortened_shared_document_removes_stale_es_chunks(
    whole_pipeline, test_db, monkeypatch, request
):
    case = whole_pipeline
    _use_es_sdk(case, case.ds, monkeypatch, request)
    from elastic_transport import ApiResponseMeta, NodeConfig
    from elastic_transport._response import ObjectApiResponse

    from kiara.retrieval.elasticsearch_service import _get_client

    es = _get_client(case.store._config)
    transport = es.perform_request

    def request_with_delete(*args, **kwargs):
        if kwargs.get("endpoint_id") != "delete_by_query":
            return transport(*args, **kwargs)
        paths = kwargs["body"]["query"]["terms"]["metadata.source.raw"]
        removed = [cid for cid, (_, meta) in case.es.items() if meta["source"] in paths]
        for cid in removed:
            del case.es[cid]
        return ObjectApiResponse(
            {"deleted": len(removed), "failures": []},
            ApiResponseMeta(200, "1.1", {}, 0.0, NodeConfig("http", "127.0.0.1", 1)),
        )

    monkeypatch.setattr(es, "perform_request", request_with_delete)
    monkeypatch.setattr(
        es,
        "count",
        lambda **kw: {
            "count": len(case.es),
            "_shards": {"failed": 0, "successful": 1, "total": 1},
        },
    )
    ds2 = _second_datasource(case, test_db)
    case.run_whole(case.ds)
    case.run_whole(ds2)
    previous_ids = {c["_doc_id"] for c in case.snapshot()}
    Path(case.path).write_text("X" * 100 + "B" * 100)
    assert case.run_whole(case.ds)["total_chunks"] == 2
    chunks = case.snapshot()
    current_ids = {c["_doc_id"] for c in chunks}
    assert len(current_ids) == 2
    assert set(case.es) == current_ids
    assert not ChunkDatasources().get_datasource_ids_for_chunks(
        list(previous_ids - current_ids), "test-model"
    )
    assert all(meta["ds_ids"] == [case.ds] for _, meta in case.es.values())
    tool, ref = case.reader(
        [{"_doc_id": cid, "text": text, "metadata": meta} for cid, (text, meta) in case.es.items()]
    )
    assert read(tool, ref, max_chunks=3).success


def test_newly_empty_document_removes_old_search_candidates(whole_pipeline, test_db):
    case = whole_pipeline
    case.run_whole(case.ds)
    ids = [c["_doc_id"] for c in case.snapshot()]
    Path(case.path).write_text("")
    result = case.run_whole(case.ds)
    assert result["errors"] == 0 and result["processed_files"] == 0
    assert case.snapshot() == []
    assert not ChunkDatasources().get_datasource_ids_for_chunks(ids, "test-model")
    assert case.path not in case.hashes.load(case.ds, "test-model")


@pytest.mark.parametrize("failure", ["loading", "preparation"])
def test_failed_preparation_cannot_leave_old_source_published(whole_pipeline, monkeypatch, failure):
    case = whole_pipeline
    case.run_whole(case.ds)
    before = case.snapshot()
    Path(case.path).write_text("X" * 300)
    if failure == "loading":

        def fail_load(*args, **kwargs):
            raise RuntimeError("load failed")

        monkeypatch.setattr(runtime, "_process_single_file", fail_load)
    else:

        def fail_read(*args, **kwargs):
            raise RuntimeError("preparation failed")

        monkeypatch.setattr(case.store, "get_ids_by_sources", fail_read)
    with pytest.raises(RuntimeError, match="failed"):
        case.run_whole(case.ds)
    assert case.path not in case.hashes.load(case.ds, "test-model")
    with pytest.raises(RetrievalError, match="nicht abgeschlossen"):
        case.reader(before)


def test_publication_lookup_failure_does_not_report_successful_skip(whole_pipeline, monkeypatch):
    case = whole_pipeline
    case.run_whole(case.ds)
    before = case.snapshot()

    def unavailable():
        raise RuntimeError("publication database unavailable")

    monkeypatch.setattr(publication, "db_query", unavailable)
    with pytest.raises(RuntimeError, match="publication database unavailable"):
        case.run_whole(case.ds)
    assert case.snapshot() == before and case.embeddings == 3


def test_cli_without_datasource_supports_incremental_retry_and_delete(whole_pipeline, test_db):
    case = whole_pipeline
    assert test_db.execute("SELECT id FROM datasources WHERE id=0").fetchone() is None
    other = Path(case.path).with_name("other.txt")
    other.write_text("Weitere Quelle")
    assert case.run_whole(None)["processed_files"] == 2
    assert len(case.hashes.load(None, "test-model")) == 2
    assert case.run_whole(None)["skipped_files"] == 2
    publication.invalidate_documents("kb_test", [case.path], "test-model", hash_datasource_id=0)
    assert case.run_whole(None)["processed_files"] == 1
    Path(case.path).unlink()
    stats = case.run_whole(None)
    assert stats["processed_files"] == 0 and stats["skipped_files"] == 1
    assert set(case.hashes.load(None, "test-model")) == {str(other)}
    owner = test_db.execute("SELECT * FROM datasources WHERE id=0").fetchone()
    assert owner["name"] == "CLI (Konfiguration)" and owner["is_active"] is False
    counts = test_db.execute(
        "SELECT chunk_count,file_count FROM datasource_embeddings WHERE datasource_id=0"
    ).fetchone()
    assert (counts["chunk_count"], counts["file_count"]) == (1, 1)
    retained_ids = case.store.get_ids_by_sources([str(other)])
    assert ChunkDatasources().get_datasource_ids_for_chunks(retained_ids, "test-model") == {
        cid: [0] for cid in retained_ids
    }
    assert case.snapshot() == []


def test_cli_adoption_and_repair_preserve_real_datasource_memberships(whole_pipeline, test_db):
    case = whole_pipeline
    case.prefix = "Erhaltener Kontext: "
    case.run_whole(case.ds)
    original = case.snapshot()
    assert case.run_whole(None)["dedup_chunks"] == 3
    expected = [{**c, "metadata": {**c["metadata"], "ds_ids": [0, case.ds]}} for c in original]
    assert case.snapshot() == expected and case.embeddings == 3
    case.store.delete_by_ids([original[1]["_doc_id"]])
    publication.invalidate_documents("kb_test", [case.path], "test-model", hash_datasource_id=0)
    retry = case.run_whole(None)
    assert retry["processed_files"] == 1 and retry["dedup_chunks"] == 2
    assert case.embeddings == 4
    _assert_shared(case, test_db, [0, case.ds])
    _assert_readable(case)
