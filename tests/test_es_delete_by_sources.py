"""DS-scoped physical deletion; no global path query or shared-index updates."""

from copy import deepcopy

import pytest
from test_elasticsearch_datasource_lifecycle import CONFIG, backend, targets, write  # noqa: F401

from kiara.retrieval import elasticsearch_service as es


@pytest.mark.parametrize("owners", [[1], [1, 2]])
def test_only_selected_index_is_deleted(backend, owners):
    write(1, owners=owners)
    write(2, owners=[1, 2])
    foreign = deepcopy(backend.docs["s2_ds_2"])
    assert es.delete_by_sources(CONFIG, ["/file"], targets=targets(CONFIG, 1)) == 2
    assert backend.docs["s2_ds_1"] == {}
    assert backend.docs["s2_ds_2"] == foreign


def test_empty_target_never_sends_unbounded_request(backend):
    write(1)
    assert es.delete_by_sources(CONFIG, ["/file"], targets=()) == 0
    assert backend.refreshes == []


def test_missing_empty_source_is_distinct_from_lost_inventory(backend):
    assert es.delete_by_sources(CONFIG, ["/file"], targets=targets(CONFIG, 1)) == 0
    with pytest.raises(es.ElasticsearchPartialFailureError):
        es.delete_by_sources(CONFIG, ["/file"], targets=targets(CONFIG, 1, expected=1))


@pytest.mark.parametrize(
    "response",
    [
        {"deleted": 1, "failures": ["error"]},
        {"deleted": 1, "timed_out": True},
        {"deleted": 1, "version_conflicts": 1},
        {},
    ],
)
def test_partial_response_is_fatal(backend, response):
    write(1)
    backend.delete_by_query = lambda **kw: response
    with pytest.raises(es.ElasticsearchPartialFailureError):
        es.delete_by_sources(CONFIG, ["/file"], targets=targets(CONFIG, 1))


def test_paths_are_batched(backend, monkeypatch):
    write(1)
    monkeypatch.setattr(es, "ES_TERMS_CHUNK_SIZE", 2)
    calls = []
    original = backend.delete_by_query

    def delete(**kw):
        calls.append(kw)
        return original(**kw)

    backend.delete_by_query = delete
    assert (
        es.delete_by_sources(CONFIG, ["/other", "/file", "/third"], targets=targets(CONFIG, 1)) == 2
    )
    assert len(calls) == 2
    assert all(c["index"] == ["s2_ds_1"] and c["conflicts"] == "abort" for c in calls)
