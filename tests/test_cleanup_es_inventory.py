"""Vollstaendige und fehlgeschlagene ES-Scrolls fuer das Quellen-Löschinventar."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from kiara.retrieval import elasticsearch_service as service


def _page(ids, total=3, **extra):
    return {
        "_scroll_id": "cursor",
        "_shards": {"failed": 0},
        "timed_out": False,
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": [{"_id": cid} for cid in ids],
        },
        **extra,
    }


@pytest.fixture
def es_inventory(monkeypatch):
    client = Mock()
    client.options.return_value = client
    client.indices.exists.return_value = True
    client.indices.refresh.return_value = {"_shards": {"failed": 0}}
    client.search.return_value = _page(["id1", "id2"])
    client.scroll.return_value = _page(["id3"])
    monkeypatch.setattr(service, "_get_client", lambda config: client)
    config = {"retrieval": {"hybrid_search": "elasticsearch", "elasticsearch_index": "test-index"}}
    return config, client


def test_inventory_reads_exact_source_and_all_pages_without_document_text(es_inventory):
    config, client = es_inventory
    source = "/a/Ä\u00ad report.txt"
    assert service.get_ids_by_source(config, source) == ["id1", "id2", "id3"]
    query = client.search.call_args.kwargs
    assert query["query"] == {"term": {"metadata.source.raw": source}}
    assert query["track_total_hits"] is True and query["allow_partial_search_results"] is False
    assert query["source"] is False
    client.indices.refresh.assert_called_once_with(index="test-index")
    client.scroll.assert_called_once_with(scroll_id="cursor", scroll="2m")
    client.clear_scroll.assert_called_once_with(scroll_id="cursor")


@pytest.mark.parametrize(
    "failure",
    [
        "refresh",
        "timeout",
        "shards",
        "short",
        "duplicate",
        "invalid_id",
        "missing_cursor",
        "inexact_total",
        "scroll_error",
        "empty_failed_page",
    ],
)
def test_partial_inventory_is_never_accepted(es_inventory, failure):
    config, client = es_inventory
    if failure == "refresh":
        client.indices.refresh.return_value = {"_shards": {"failed": 1}}
    elif failure == "timeout":
        client.search.return_value["timed_out"] = True
    elif failure == "shards":
        client.scroll.return_value["_shards"]["failed"] = 1
    elif failure == "short":
        client.scroll.return_value = _page([])
    elif failure == "duplicate":
        client.scroll.return_value = _page(["id2"])
    elif failure == "invalid_id":
        client.scroll.return_value = _page([None])
    elif failure == "missing_cursor":
        del client.search.return_value["_scroll_id"]
    elif failure == "inexact_total":
        client.search.return_value["hits"]["total"]["relation"] = "gte"
    elif failure == "scroll_error":
        client.scroll.side_effect = ConnectionError("ES unavailable")
    else:
        client.search.return_value = _page([], total=0, _shards={"failed": 1})
    with pytest.raises((service.ElasticsearchPartialFailureError, ConnectionError)):
        service.get_ids_by_source(config, "/source")
    client.delete_by_query.assert_not_called()
    if failure == "refresh":
        client.search.assert_not_called()
    elif failure != "missing_cursor":
        client.clear_scroll.assert_called_once()


def test_empty_or_unconfigured_index_is_an_empty_inventory(es_inventory):
    config, client = es_inventory
    client.search.return_value = _page([], total=0)
    assert service.get_ids_by_source(config, "/source") == []
    client.scroll.assert_not_called()
    client.indices.exists.return_value = False
    client.search.reset_mock()
    assert service.get_ids_by_source(config, "/source") == []
    client.search.assert_not_called()
    disabled = deepcopy(config)
    disabled["retrieval"]["hybrid_search"] = "none"
    client.indices.exists.reset_mock()
    assert service.get_ids_by_source(disabled, "/source") == []
    client.indices.exists.assert_not_called()
