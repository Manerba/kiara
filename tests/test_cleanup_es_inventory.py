"""Vollstaendige und fehlgeschlagene ES-Scrolls fuer das Quellen-Löschinventar.

Seit den getrennten Datenquellen-Indizes (``<praefix>_ds_<id>``) liest das
Inventar nur explizite, vom Aufrufer autorisierte Ziele und liefert vollstaendige
Dokumentadressen (Datenquelle, Index, Chunk-ID) statt nackter Chunk-IDs.
"""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from kiara.retrieval import elasticsearch_service as service
from kiara.retrieval.elasticsearch_indices import (
    DatasourceIndexTarget,
    document_address,
    resolve_datasource_targets,
)

PREFIX = "test-index"
INDEX_1 = f"{PREFIX}_ds_1"
INDEX_2 = f"{PREFIX}_ds_2"


def _page(hits, total=3, **extra):
    return {
        "_scroll_id": "cursor",
        "_shards": {"failed": 0},
        "timed_out": False,
        "hits": {
            "total": {"value": total, "relation": "eq"},
            "hits": [{"_index": index, "_id": cid} for index, cid in hits],
        },
        **extra,
    }


def _settings(*names):
    return {name: {"settings": {"index": {"uuid": f"uuid-{name}"}}} for name in names}


def _stats(*names):
    return {
        "_shards": {"total": len(names), "successful": len(names), "failed": 0},
        "indices": {
            name: {"uuid": f"uuid-{name}", "primaries": {"docs": {"count": 3}}} for name in names
        },
    }


def _refresh_ok(count=2):
    return {"_shards": {"total": count, "successful": count, "failed": 0}}


@pytest.fixture
def es_inventory(monkeypatch):
    client = Mock()
    client.options.return_value = client
    client.indices.get_settings.return_value = _settings(INDEX_1, INDEX_2)
    client.indices.stats.return_value = _stats(INDEX_1, INDEX_2)
    client.indices.refresh.return_value = _refresh_ok()
    client.search.return_value = _page([(INDEX_1, "id1"), (INDEX_2, "id1")])
    client.scroll.return_value = _page([(INDEX_1, "id3")])
    monkeypatch.setattr(service, "_get_client", lambda config: client)
    config = {"retrieval": {"hybrid_search": "elasticsearch", "elasticsearch_index": PREFIX}}
    return config, client


def _targets(config, counts=None):
    counts = {1: 1, 2: 1} if counts is None else counts
    return resolve_datasource_targets(
        config, None, allowed_ds_ids=list(counts), expected_document_counts=counts
    )


def test_inventory_reads_exact_source_and_all_pages_without_document_text(es_inventory):
    config, client = es_inventory
    source = "/a/Ä­ report.txt"
    result = service.get_ids_by_source(config, source, targets=_targets(config))
    # Gleiche Chunk-ID in zwei Datenquellen bleibt als zwei Adressen erhalten.
    assert result == [
        document_address(config, 1, "id1"),
        document_address(config, 1, "id3"),
        document_address(config, 2, "id1"),
    ]
    query = client.search.call_args.kwargs
    assert query["index"] == [INDEX_1, INDEX_2]
    assert query["query"] == {"term": {"metadata.source.raw": source}}
    assert query["track_total_hits"] is True and query["allow_partial_search_results"] is False
    assert query["source"] is False
    # Kein Wildcard-/Teilkorpus: fehlende Indizes nach der Inventur sind fatal.
    assert query["ignore_unavailable"] is False and query["allow_no_indices"] is False
    assert query["expand_wildcards"] == "none"
    client.indices.refresh.assert_called_once()
    assert client.indices.refresh.call_args.kwargs["index"] == [INDEX_1, INDEX_2]
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
        "foreign_index",
        "missing_cursor",
        "inexact_total",
        "scroll_error",
        "empty_failed_page",
    ],
)
def test_partial_inventory_is_never_accepted(es_inventory, failure):
    config, client = es_inventory
    if failure == "refresh":
        client.indices.refresh.return_value = {
            "_shards": {"total": 2, "successful": 1, "failed": 1}
        }
    elif failure == "timeout":
        client.search.return_value["timed_out"] = True
    elif failure == "shards":
        client.scroll.return_value["_shards"]["failed"] = 1
    elif failure == "short":
        client.scroll.return_value = _page([])
    elif failure == "duplicate":
        client.scroll.return_value = _page([(INDEX_2, "id1")])
    elif failure == "invalid_id":
        client.scroll.return_value = _page([(INDEX_1, None)])
    elif failure == "foreign_index":
        client.scroll.return_value = _page([(f"{PREFIX}_ds_3", "id3")])
    elif failure == "missing_cursor":
        del client.search.return_value["_scroll_id"]
    elif failure == "inexact_total":
        client.search.return_value["hits"]["total"]["relation"] = "gte"
    elif failure == "scroll_error":
        client.scroll.side_effect = ConnectionError("ES unavailable")
    else:
        client.search.return_value = _page([], total=0, _shards={"failed": 1})
    # Ungueltige Chunk-IDs scheitern bereits an der Adressvalidierung (ValueError).
    with pytest.raises((service.ElasticsearchPartialFailureError, ConnectionError, ValueError)):
        service.get_ids_by_source(config, "/source", targets=_targets(config))
    client.delete_by_query.assert_not_called()
    if failure == "refresh":
        client.search.assert_not_called()
    elif failure != "missing_cursor":
        client.clear_scroll.assert_called_once()


def test_empty_or_unconfigured_index_is_an_empty_inventory(es_inventory):
    config, client = es_inventory
    client.search.return_value = _page([], total=0)
    assert service.get_ids_by_source(config, "/source", targets=_targets(config)) == []
    client.scroll.assert_not_called()

    # Fehlender Index ist nur fuer PG-bestaetigt leere Datenquellen ein leeres Inventar.
    client.indices.get_settings.return_value = {}
    client.search.reset_mock()
    client.indices.refresh.reset_mock()
    empty = _targets(config, {1: 0, 2: 0})
    assert service.get_ids_by_source(config, "/source", targets=empty) == []
    client.search.assert_not_called()
    client.indices.refresh.assert_not_called()
    with pytest.raises(service.ElasticsearchPartialFailureError):
        service.get_ids_by_source(config, "/source", targets=_targets(config))
    client.search.assert_not_called()

    # Keine Ziele: kein ES-Zugriff.
    client.indices.get_settings.reset_mock()
    assert service.get_ids_by_source(config, "/source", targets=()) == []
    client.indices.get_settings.assert_not_called()

    disabled = deepcopy(config)
    disabled["retrieval"]["hybrid_search"] = "none"
    assert service.get_ids_by_source(disabled, "/source", targets=_targets(config)) == []
    client.indices.get_settings.assert_not_called()


def test_inventory_rejects_targets_outside_configured_prefix(es_inventory):
    config, client = es_inventory
    foreign = (DatasourceIndexTarget(1, "other_ds_1", 1),)
    with pytest.raises(ValueError):
        service.get_ids_by_source(config, "/source", targets=foreign)
    client.indices.get_settings.assert_not_called()
    client.search.assert_not_called()
