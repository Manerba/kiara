"""S3: both consumers with real resolution, inventory and response validation.

PG transport/model-fence doubles isolate the corpus contract. Publication
checks execute the existing validator; live ES evidence is in the opt-in suite.
"""

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from kiara.retrieval import elasticsearch_service as svc
from kiara.retrieval.elasticsearch_indices import datasource_index_name
from kiara.retrieval.retriever import HybridRetriever, _FilterParams
from kiara.retrieval.source_policy import validate_source_backend_budget
from kiara.web.api_search import SearchRequest, _search_text


@pytest.fixture
def env(monkeypatch):
    config = {"retrieval": {"hybrid_search": "elasticsearch", "elasticsearch_index": "s3_test"}}
    state = SimpleNamespace(
        config=config,
        ids=[0, 1, 2, 3],
        populated={0, 1, 2},
        present={0, 1, 2},
        counts={0: 1, 1: 4, 2: 7},
        windows={0: 10000, 1: 10000, 2: 600},
        pg_calls=[],
        guarded=False,
        publications=[],
        owners={"c1": [1]},
    )
    client = MagicMock()
    state.client = client

    def execute(sql, args=()):
        state.pg_calls.append((sql, args))
        if sql.startswith("SELECT id FROM datasources"):
            rows = [{"id": i} for i in state.ids if not args or i in args[0]]
        elif "AS populated" in sql:
            assert state.guarded
            rows = [{"id": i, "populated": i in state.populated} for i in args[1]]
        elif "FROM document_publications" in sql:
            rows = state.publications
        else:
            raise AssertionError(sql)
        return SimpleNamespace(fetchall=lambda: rows)

    @contextmanager
    def db_query():
        yield SimpleNamespace(execute=execute)

    @contextmanager
    def guard(*args, **kwargs):
        assert kwargs["require_configured_default"]
        assert kwargs["validation_scope"] == "stored_binding"
        state.guarded = True
        try:
            yield None
        finally:
            state.guarded = False

    def settings(**kwargs):
        assert state.guarded
        names = kwargs["index"]
        assert isinstance(names, list) and names
        if kwargs["name"] == "index.uuid":
            return {
                datasource_index_name(config, i): {"settings": {"index": {"uuid": f"u{i}"}}}
                for i in state.present
                if datasource_index_name(config, i) in names
            }
        return {
            datasource_index_name(config, i): {
                "settings": {"index.max_result_window": str(state.windows[i])}
            }
            for i in state.present
            if datasource_index_name(config, i) in names
        }

    def stats(**kwargs):
        names = kwargs["index"]
        return {
            "_shards": {"total": len(names), "successful": len(names), "failed": 0},
            "indices": {
                datasource_index_name(config, i): {
                    "uuid": f"u{i}",
                    "primaries": {"docs": {"count": state.counts[i]}},
                }
                for i in state.present
                if datasource_index_name(config, i) in names
            },
        }

    client.indices.get_settings.side_effect = settings
    client.indices.stats.side_effect = stats
    client.search.return_value = response([])
    getter = MagicMock(return_value=client)
    state.getter = getter
    monkeypatch.setattr(svc, "_get_client", getter)
    monkeypatch.setattr("kiara.shared.db.db_query", db_query)
    monkeypatch.setattr("kiara.ingestion.document_publication.db_query", db_query)
    monkeypatch.setattr("kiara.config.get_embedding_model", lambda c: "test-model")
    monkeypatch.setattr(
        "kiara.retrieval.index_compatibility.stored_primary_index_target",
        lambda *a: SimpleNamespace(collection_name="kb_test_model"),
    )
    monkeypatch.setattr("kiara.retrieval.index_compatibility.guard_index", guard)
    monkeypatch.setattr(
        "kiara.ingestion._file_utils.ChunkDatasources.get_datasource_ids_for_chunks",
        lambda self, ids, model: state.owners,
    )
    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever._config = config
    retriever._embed_model_name = "test-model"
    retriever._es_fuzziness = "AUTO"
    retriever._es_client = client
    retriever._retrieval_diagnostics_local = threading.local()
    retriever._chunk_ds = MagicMock()
    retriever._chunk_ds.filter_allowed.return_value = {"c1"}
    state.retriever = retriever
    return state


def response(hits):
    return {
        "timed_out": False,
        "_shards": {"total": 1, "successful": 1, "failed": 0},
        "hits": {"total": {"value": len(hits)}, "hits": hits},
    }


def hit(**metadata):
    return {
        "_index": "s3_test_ds_1",
        "_id": "c1",
        "_score": 1.2,
        "_source": {
            "text": "Freigegeben",
            "metadata": {"source": "/data/a.pdf", "ds_ids": [1], **metadata},
        },
    }


def search(env, path, ids, **kwargs):
    if path == "hybrid":
        return env.retriever._es_search(
            "Drehmoment", k=10, filters=_FilterParams(datasource_ids=ids, **kwargs)
        )
    return _search_text(env.config, SearchRequest(query="Drehmoment", **kwargs), ids)


@pytest.mark.parametrize("path", ["hybrid", "admin"])
@pytest.mark.parametrize(
    "ids,expected",
    [
        ([1], [1]),
        ([2, 1, 2], [1, 2]),
        (None, [0, 1, 2]),
        ([0], [0]),
        ([99], []),
        ([], []),
        ([3], []),
    ],
)
def test_selection_and_known_empty(env, path, ids, expected):
    search(env, path, ids)
    if expected:
        call = env.client.search.call_args.kwargs
        assert call["index"] == [datasource_index_name(env.config, i) for i in expected]
        assert call["search_type"] == "dfs_query_then_fetch"
        assert call["expand_wildcards"] == "none"
        assert call["ignore_unavailable"] is False
        assert call["allow_no_indices"] is False
        assert call["allow_partial_search_results"] is False
    else:
        env.client.search.assert_not_called()
        if ids != [3]:
            env.getter.assert_not_called()
    assert all("COUNT(" not in sql.upper() for sql, _ in env.pg_calls)
    if path == "hybrid":
        assert env.retriever._retrieval_diagnostics_local.es_indices == [
            datasource_index_name(env.config, i) for i in expected
        ]


@pytest.mark.parametrize("path", ["hybrid", "admin"])
@pytest.mark.parametrize("ids", [[True], [1.0], ["1"], [-1], [1, "*"]])
def test_invalid_selection_cannot_broaden(env, path, ids):
    with pytest.raises((ValueError, HTTPException)):
        search(env, path, ids)
    env.getter.assert_not_called()


@pytest.mark.parametrize("path", ["hybrid", "admin"])
@pytest.mark.parametrize("loss", ["absent", "empty"])
def test_expected_loss_is_error(env, path, loss):
    if loss == "absent":
        env.present.remove(1)
    else:
        env.counts[1] = 0
    with pytest.raises((svc.ElasticsearchPartialFailureError, HTTPException)):
        search(env, path, [1])
    env.client.search.assert_not_called()


@pytest.mark.parametrize("path", ["hybrid", "admin"])
@pytest.mark.parametrize(
    "change",
    [
        {"timed_out": True},
        {"timed_out": None},
        {"terminated_early": True},
        {"_shards": {"total": 2, "successful": 1, "failed": 1}},
        {"_shards": {"total": 2, "successful": 1, "failed": 0}},
        {"_shards": {"total": 0, "successful": 0, "failed": 0}},
        {"_shards": {"total": True, "successful": 1, "failed": 0}},
    ],
)
def test_partial_answers_fail_closed(env, path, change):
    env.client.search.return_value.update(change)
    with pytest.raises((svc.ElasticsearchPartialFailureError, HTTPException)):
        search(env, path, [1])


@pytest.mark.parametrize("path", ["hybrid", "admin"])
def test_foreign_hit_is_not_a_valid_search(env, path):
    h = hit()
    h["_index"] = "s3_test_ds_2"
    env.client.search.return_value = response([h])
    with pytest.raises((svc.ElasticsearchPartialFailureError, HTTPException)):
        search(env, path, [1])


@pytest.mark.parametrize("path", ["hybrid", "admin"])
def test_deletion_between_inspection_and_search_is_error(env, path):
    env.client.search.side_effect = RuntimeError("index_not_found_exception")
    with pytest.raises((RuntimeError, HTTPException)):
        search(env, path, [1])


def test_model_mismatch_precedes_es(env):
    env.retriever._embed_model_name = "other-model"
    with pytest.raises(ValueError, match="Default-Dense"):
        search(env, "hybrid", [1])
    env.getter.assert_not_called()


def test_model_guard_failure_precedes_es(env, monkeypatch):
    monkeypatch.setattr(
        "kiara.retrieval.index_compatibility.guard_index",
        MagicMock(side_effect=RuntimeError("blocked binding")),
    )
    with pytest.raises(RuntimeError, match="binding"):
        search(env, "hybrid", [1])
    env.getter.assert_not_called()


def test_strictest_window_and_pagination(env):
    env.windows[2] = 30
    with pytest.raises(ValueError, match="40.*30"):
        search(env, "hybrid", [1, 2])
    env.client.search.assert_not_called()
    with pytest.raises(HTTPException) as error:
        search(env, "admin", [1, 2], page=4, page_size=10)
    assert error.value.status_code == 400
    env.client.search.assert_not_called()
    search(env, "admin", [1, 2], page=3, page_size=10)
    assert env.client.search.call_args.kwargs["body"]["from"] == 20


@pytest.mark.parametrize(
    "settings", [{}, {"foreign": {"settings": {"index.max_result_window": "100"}}}]
)
def test_window_requires_exact_targets(env, settings):
    original = env.client.indices.get_settings.side_effect
    env.client.indices.get_settings.side_effect = lambda **kw: (
        settings if kw["name"] == "index.max_result_window" else original(**kw)
    )
    with pytest.raises(ValueError, match="Ziele"):
        search(env, "hybrid", [1])
    env.client.search.assert_not_called()


def test_save_budget_concrete_targets_and_empty(env):
    env.config["retrieval"]["reranking"] = True
    with pytest.raises(ValueError, match="1600.*600"):
        validate_source_backend_budget(env.config)
    env.ids = []
    env.getter.reset_mock()
    validate_source_backend_budget(env.config)
    env.getter.assert_not_called()


def test_metadata_filters_and_hybrid_pg_recheck(env):
    env.client.search.return_value = response([hit()])
    env.retriever._chunk_ds.filter_allowed.return_value = set()
    assert search(env, "hybrid", [1], directory="/data", file_type=".pdf", source="a.pdf") == []
    body = env.client.search.call_args.kwargs["body"]
    assert body["query"]["bool"]["must"][0]["multi_match"]["fields"] == [
        "text",
        "metadata.source^2",
    ]
    assert len(body["query"]["bool"]["filter"]) == 4
    env.retriever._chunk_ds.filter_allowed.assert_called_once_with(["c1"], [1], "test-model")


@pytest.mark.parametrize(
    "state", ["pending", "wrong_version", "no_owner", "foreign_owner", "payload_pending"]
)
def test_admin_publication_and_pg_boundaries(env, state):
    h = hit(processing_version="v1")
    env.publications = [
        {"source": "/data/a.pdf", "completed_at": "now", "versions": {"standard": "v1"}}
    ]
    if state == "pending":
        env.publications[0]["completed_at"] = None
    if state == "wrong_version":
        env.publications[0]["versions"]["standard"] = "v2"
    if state == "no_owner":
        env.owners = {}
    if state == "foreign_owner":
        env.owners = {"c1": [2]}
    if state == "payload_pending":
        h["_source"]["metadata"]["processing_pending"] = True
    env.client.search.return_value = response([h])
    with pytest.raises(HTTPException) as error:
        search(env, "admin", [1])
    assert error.value.status_code == 503


def test_admin_published_page_and_metadata_filters(env):
    env.client.search.return_value = response([hit(processing_version="v1")])
    env.publications = [
        {"source": "/data/a.pdf", "completed_at": "now", "versions": {"standard": "v1"}}
    ]
    result, total = search(
        env,
        "admin",
        [1],
        filter_type=".pdf",
        filter_dir="/data",
        filter_date_from="2026-01-01",
        filter_date_to="2026-10-05",
    )
    assert total == 1 and result[0]["id"] == "c1"
    filters = env.client.search.call_args.kwargs["body"]["query"]["bool"]["filter"]
    assert {"term": {"metadata.file_type": ".pdf"}} in filters
    assert any("range" in f for f in filters)


@pytest.mark.parametrize("selected,expected", [(None, [1]), ([1, 2], [1]), ([2], []), ([], [])])
def test_route_permissions_precede_target_resolution(env, monkeypatch, selected, expected):
    import asyncio

    from kiara.web import api_search

    monkeypatch.setattr(api_search, "load_config", lambda: env.config)
    monkeypatch.setattr("kiara.web.api_chat.perm_repo.get_public_datasource_ids", lambda: [])
    result = asyncio.run(
        api_search.search(
            SearchRequest(query="Test", filter_datasource_ids=selected),
            user={"is_admin": False, "datasource_ids": [1]},
        )
    )
    assert result["total"] == 0
    if expected:
        assert env.client.search.call_args.kwargs["index"] == ["s3_test_ds_1"]
    else:
        env.getter.assert_not_called()


@pytest.mark.parametrize("ids", [[True], [1.0], ["1"], ["*"]])
def test_http_schema_does_not_coerce_invalid_ids(ids):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        SearchRequest(query="Test", filter_datasource_ids=ids)


def test_pg_inventory_keeps_inactive_cli_owner_and_exact_selection(test_db):
    from kiara.ingestion._file_utils import ChunkDatasources

    test_db.execute(
        "INSERT INTO datasources (id, name, type, is_active) VALUES (0, 'CLI', 'local', false) ON CONFLICT (id) DO UPDATE SET is_active=false"
    )
    other = test_db.execute(
        "INSERT INTO datasources (name, type) VALUES ('S3', 'local') RETURNING id"
    ).fetchone()["id"]
    test_db.commit()
    ChunkDatasources().add_associations(["s3-chunk"], 0, "model-a")
    assert 0 in svc.search_datasource_ids(None)
    assert svc.search_datasource_ids([0, other, 999999999]) == [0, other]
    targets = svc.targets_for_datasources({}, [0, other], "model-a")
    assert [(t.datasource_id, t.expected_document_count) for t in targets] == [(0, 1), (other, 0)]
    assert svc.targets_for_datasources({}, [0], "model-b")[0].expected_document_count == 0
