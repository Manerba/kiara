"""Opt-in isolated S4 rehearsal with real ES, test PG and QdrantLocal.

KIARA_S4_ES_URL=http://localhost:9200 enables the test. No production inputs.
"""

import json
import os
import time
import tracemalloc
import uuid
from pathlib import Path

import pytest
from elasticsearch import Elasticsearch
from test_elasticsearch_s4 import corpus  # noqa: F401

from kiara.retrieval import elasticsearch_maintenance as maintenance
from kiara.retrieval import elasticsearch_service as es
from kiara.retrieval.elasticsearch_indices import datasource_index_name


@pytest.mark.parametrize("corpus", [1200], indirect=True)
def test_isolated_switch_failure_restart_and_retained_old_index(corpus, monkeypatch):
    url = os.environ.get("KIARA_S4_ES_URL")
    if not url:
        pytest.skip("KIARA_S4_ES_URL enables isolated maintenance rehearsal")
    prefix = "kiara_p69_s4_test_" + uuid.uuid4().hex
    corpus.config["retrieval"].update(elasticsearch_url=url, elasticsearch_index=prefix)
    client = Elasticsearch(url)
    names = [datasource_index_name(corpus.config, ds) for ds in corpus.ids]
    names += [prefix, prefix + "_unrelated"]
    report = {"prefix": prefix, "version": client.info()["version"]["number"], "names": names}
    monkeypatch.setattr(es, "_get_client", lambda config: client)
    from contextlib import contextmanager

    from kiara.retrieval import elasticsearch_inventory as inventory

    original_query = inventory.db_query
    batches = []

    @contextmanager
    def counted_query():
        with original_query() as connection:

            class Counted:
                def execute(self, sql, params=None):
                    if "chunk_id = ANY" in sql:
                        batches.append(len(params[1]))
                    return connection.execute(sql, params)

            yield Counted()

    monkeypatch.setattr(inventory, "db_query", counted_query)
    try:
        for name in names[-2:]:
            client.indices.create(index=name)
            client.index(index=name, id="sentinel", document={"text": name}, refresh=True)
        uuids = {
            name: client.indices.get_settings(index=name)[name]["settings"]["index"]["uuid"]
            for name in names[-2:]
        }
        preview = es.rebuild_from_vectorstore(corpus.config, dry_run=True, batch_size=128)
        assert preview["expected_physical_documents"] == 1204
        assert (
            dict(
                client.indices.get_settings(
                    index=names[:3], ignore_unavailable=True, allow_no_indices=True
                )
            )
            == {}
        )
        report["preview"] = preview
        maintenance.begin(corpus.config)
        refresh = es.refresh_index

        def fail_refresh(*args, **kwargs):
            raise RuntimeError("isolated injected refresh failure")

        monkeypatch.setattr(es, "refresh_index", fail_refresh)
        with maintenance.offline_session(corpus.config):
            failed = es.rebuild_from_vectorstore(corpus.config, batch_size=128)
        assert failed["status"] == "error" and not failed["release_ready"]
        assert maintenance.marker_path(corpus.config).exists()
        with pytest.raises(RuntimeError, match="gesperrt"):
            maintenance.assert_access(corpus.config)
        report["failure"] = failed
        monkeypatch.setattr(es, "refresh_index", refresh)
        tracemalloc.start()
        started = time.perf_counter()
        with maintenance.offline_session(corpus.config):
            result = es.rebuild_from_vectorstore(corpus.config, batch_size=128)
            assert result["release_ready"], result
            status = es.get_status(corpus.config)
            storage = es.storage_stats(corpus.config)
            assert es.check_missing_ds_ids(corpus.config)["safe_to_proceed"]
        report.update(
            result=result,
            elapsed_seconds=time.perf_counter() - started,
            python_peak_bytes=tracemalloc.get_traced_memory()[1],
            status=status,
            storage=storage,
        )
        tracemalloc.stop()
        for name in names[-2:]:
            assert client.get(index=name, id="sentinel")["_source"] == {"text": name}
            assert (
                client.indices.get_settings(index=name)[name]["settings"]["index"]["uuid"]
                == uuids[name]
            )
        # No automatic release, even with complete replacement.
        assert maintenance.marker_path(corpus.config).exists()
        assert status["doc_count"] == 1204
        assert set(storage["indices"]) == set(names[:3])
        assert not storage["retained_common_index_included"]
        # A lost expected index is unknown/unhealthy, never a successful zero.
        client.indices.delete(index=names[0])
        with maintenance.offline_session(corpus.config):
            assert es.get_status(corpus.config)["error"]
            assert es.rebuild_from_vectorstore(corpus.config, batch_size=128)["release_ready"]
        report["lost_index_retry"] = "passed"
        report["membership_batches"] = {
            "calls_all_phases": len(batches),
            "max_chunk_ids": max(batches),
        }
        assert max(batches) <= 128
        assert len(batches) < len(corpus.points)
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        for name in names:
            assert name.startswith(prefix)
            client.indices.delete(index=name, ignore_unavailable=True)
        report["remaining_indices"] = dict(
            client.indices.get_settings(index=names, ignore_unavailable=True, allow_no_indices=True)
        )
        client.close()
        path = os.environ.get("KIARA_S4_REPORT")
        if path:
            Path(path).write_text(json.dumps(report, indent=2))
    assert report["remaining_indices"] == {}
