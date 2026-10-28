"""S1 contract tests; no live ES/PG or productive index mutations."""

from unittest.mock import Mock, create_autospec

import pytest
from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig, ObjectApiResponse
from elasticsearch import Elasticsearch

from kiara.retrieval import elasticsearch_service as service
from kiara.retrieval.elasticsearch_indices import (
    DatasourceIndexTarget,
    ElasticsearchDocumentAddress,
    datasource_index_name,
    document_address,
    resolve_datasource_targets,
)


@pytest.mark.parametrize("ds_id", [None, True, False, -1, 1.0, "1", "1,2", "*", []])
def test_invalid_ids_fail_before_addressing(ds_id):
    with pytest.raises(ValueError):
        datasource_index_name({}, ds_id)


@pytest.mark.parametrize(
    "prefix",
    [None, "", "Kiara", " foo", "foo ", "*", "a,b", "a:b", "a/b", "a?", ".a", "_a", "-a", "ümlaut"],
)
def test_prefix_cannot_be_an_index_expression(prefix):
    with pytest.raises(ValueError):
        datasource_index_name({"retrieval": {"elasticsearch_index": prefix}}, 12)


def test_configured_prefix_and_actual_name_length():
    assert datasource_index_name({}, 12) == "kiara_ds_12"
    assert (
        datasource_index_name({"retrieval": {"elasticsearch_index": "company.test-1"}}, 9)
        == "company.test-1_ds_9"
    )
    config = {"retrieval": {"elasticsearch_index": "a" * 250}}
    assert len(datasource_index_name(config, 1)) == 255
    with pytest.raises(ValueError, match="255"):
        datasource_index_name(config, 10)


def test_existing_pg_backed_cli_owner_zero_is_explicit_and_permission_bound():
    assert document_address({}, 0, "cli-chunk").index_name == "kiara_ds_0"
    targets = resolve_datasource_targets(
        {}, [0], allowed_ds_ids=[0], expected_document_counts={0: 4}
    )
    assert targets == (DatasourceIndexTarget(0, "kiara_ds_0", 4),)
    assert (
        resolve_datasource_targets({}, [0], allowed_ds_ids=[1], expected_document_counts={1: 0})
        == ()
    )
    with pytest.raises(ValueError):
        resolve_datasource_targets({}, [0], allowed_ds_ids=[0], expected_document_counts={})


@pytest.mark.parametrize(
    ("selected", "allowed", "expected"),
    [
        ([2], [1, 2, 3], [2]),
        ([3, 1, 3], [1, 2, 3], [1, 3]),
        (None, [3, 1], [1, 3]),
        ([], [1, 2, 3], []),
        (None, [], []),
        ([1, 2, 99], [2, 3], [2]),
        ([1], [], []),
        ([99], [1], []),
    ],
)
def test_selection_never_expands_permission_set(selected, allowed, expected):
    targets = resolve_datasource_targets(
        {}, selected, allowed_ds_ids=allowed, expected_document_counts={1: 0, 2: 3, 3: 7}
    )
    assert [target.datasource_id for target in targets] == expected
    assert [target.index_name for target in targets] == [f"kiara_ds_{ds}" for ds in expected]


@pytest.mark.parametrize("allowed", [None, "1", {1: 0}, [True], ["1"]])
def test_permissions_must_be_explicit_strict_ids(allowed):
    with pytest.raises(ValueError):
        resolve_datasource_targets(
            {}, None, allowed_ds_ids=allowed, expected_document_counts={1: 0}
        )


def test_invalid_selection_is_not_silently_discarded():
    with pytest.raises(ValueError):
        resolve_datasource_targets({}, [999, "bad"], allowed_ds_ids=[], expected_document_counts={})


@pytest.mark.parametrize(
    "counts", [{}, {1: None}, {1: -1}, {1: True}, {1: 1.0}, {True: 0}, {1.0: 0}, {"1": 0}, None]
)
def test_unknown_or_invalid_expectation_is_not_zero(counts):
    with pytest.raises(ValueError):
        resolve_datasource_targets({}, [1], allowed_ds_ids=[1], expected_document_counts=counts)


def test_address_keeps_datasource_identity_for_same_chunk():
    a = document_address({}, 1, "shared-chunk")
    b = document_address({}, 2, "shared-chunk")
    assert len({a, b}) == 2
    assert a.chunk_id == b.chunk_id
    assert a.index_name == "kiara_ds_1"
    with pytest.raises(ValueError):
        ElasticsearchDocumentAddress(2, "kiara_ds_1", "chunk")
    with pytest.raises(ValueError):
        ElasticsearchDocumentAddress(1, "*_ds_1", "chunk")
    with pytest.raises(ValueError):
        document_address({}, 1, "")


@pytest.fixture
def client(monkeypatch):
    client = Mock()
    # Check the real installed ES 8.x keyword signatures as well as our mocks.
    with Elasticsearch("http://unused.invalid:9200") as signature_client:
        client.indices = create_autospec(type(signature_client.indices), instance=True)
    getter = Mock(return_value=client)
    monkeypatch.setattr(service, "_get_client", getter)
    client.getter = getter
    return client


def _targets(counts):
    return resolve_datasource_targets(
        {}, None, allowed_ds_ids=list(counts), expected_document_counts=counts
    )


def _backend(client, counts):
    client.indices.get_settings.return_value = {
        f"kiara_ds_{ds}": {"settings": {"index": {"uuid": f"original-{ds}"}}} for ds in counts
    }
    client.indices.stats.return_value = {
        "_shards": {"total": len(counts), "successful": len(counts), "failed": 0},
        "indices": {
            f"kiara_ds_{ds}": {
                "uuid": f"original-{ds}",
                "primaries": {"docs": {"count": count}},
                "total": {"docs": {"count": 1000}},
            }
            for ds, count in counts.items()
        },
    }


def _response(body):
    return ObjectApiResponse(
        body=body,
        meta=ApiResponseMeta(
            200, "1.1", HttpHeaders(), 0.0, NodeConfig("http", "unused.invalid", 9200)
        ),
    )


@pytest.mark.parametrize("count", [1, 4])
def test_real_response_wrappers_with_stable_identity(client, count):
    _backend(client, {1: count})
    client.indices.get_settings.return_value = _response(client.indices.get_settings.return_value)
    client.indices.stats.return_value = _response(client.indices.stats.return_value)
    assert service.inspect_datasource_indices({}, _targets({1: 4})).document_counts == {1: count}


def test_real_response_wrapper_with_valid_empty_object(client):
    client.indices.get_settings.return_value = _response({})
    result = service.inspect_datasource_indices({}, _targets({1: 0}))
    assert result.absent_empty == _targets({1: 0})
    client.indices.stats.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("count", [1, 4])
def test_index_replacement_is_fatal_even_with_positive_or_equal_count(client, wrapped, count):
    _backend(client, {1: count})
    client.indices.stats.return_value["indices"]["kiara_ds_1"]["uuid"] = "replacement"
    if wrapped:
        client.indices.get_settings.return_value = _response(
            client.indices.get_settings.return_value
        )
        client.indices.stats.return_value = _response(client.indices.stats.return_value)
    with pytest.raises(service.ElasticsearchPartialFailureError, match="ersetzt"):
        service.inspect_datasource_indices({}, _targets({1: 4}))


@pytest.mark.parametrize("phase", ["get_settings", "stats"])
@pytest.mark.parametrize("uuid", [None, "", " ", "with space", "_na_", 1])
def test_missing_or_invalid_identity_evidence_is_fatal(client, phase, uuid):
    _backend(client, {1: 1})
    if phase == "get_settings":
        identity = client.indices.get_settings.return_value["kiara_ds_1"]["settings"]["index"]
    else:
        identity = client.indices.stats.return_value["indices"]["kiara_ds_1"]
    if uuid is None:
        del identity["uuid"]
    else:
        identity["uuid"] = uuid
    with pytest.raises(service.ElasticsearchPartialFailureError, match="UUID"):
        service.inspect_datasource_indices({}, _targets({1: 1}))
    if phase == "get_settings":
        client.indices.stats.assert_not_called()


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("body", [[], "", None, 0])
def test_non_object_settings_are_not_confirmed_empty(client, wrapped, body):
    client.indices.get_settings.return_value = _response(body) if wrapped else body
    with pytest.raises(service.ElasticsearchPartialFailureError, match="Objekt"):
        service.inspect_datasource_indices({}, _targets({1: 0}))
    client.indices.stats.assert_not_called()


@pytest.mark.parametrize("entry", [[], {}, {"settings": []}, {"settings": {"index": []}}])
def test_invalid_nested_settings_are_fatal(client, entry):
    client.indices.get_settings.return_value = _response({"kiara_ds_1": entry})
    with pytest.raises(service.ElasticsearchPartialFailureError, match="Objekt"):
        service.inspect_datasource_indices({}, _targets({1: 0}))
    client.indices.stats.assert_not_called()


def test_empty_selection_never_builds_client_or_queries_cluster(client):
    result = service.inspect_datasource_indices({}, ())
    assert result.present == result.absent_empty == ()
    assert result.physical_documents == 0
    client.getter.assert_not_called()


def test_empty_sources_are_explicit_and_counts_are_primary_entries(client):
    _backend(client, {2: 0, 3: 4, 4: 4})
    result = service.inspect_datasource_indices({}, _targets({1: 0, 2: 0, 3: 4, 4: 4}))
    assert [target.datasource_id for target in result.present] == [2, 3, 4]
    assert [target.datasource_id for target in result.absent_empty] == [1]
    assert result.document_counts == {1: 0, 2: 0, 3: 4, 4: 4}
    assert result.physical_documents == 8  # even if these are the same four logical chunks
    client.indices.get_settings.assert_called_once_with(
        index=["kiara_ds_1", "kiara_ds_2", "kiara_ds_3", "kiara_ds_4"],
        name="index.uuid",
        expand_wildcards="none",
        ignore_unavailable=True,
        allow_no_indices=True,
    )
    client.indices.stats.assert_called_once_with(
        index=["kiara_ds_2", "kiara_ds_3", "kiara_ds_4"],
        metric="docs",
        level="indices",
        expand_wildcards="none",
    )


def test_only_known_empty_missing_sources_skip_stats(client):
    _backend(client, {})
    result = service.inspect_datasource_indices({}, _targets({1: 0}))
    assert result.present == ()
    assert result.absent_empty == _targets({1: 0})
    client.indices.stats.assert_not_called()


@pytest.mark.parametrize("observed", [{}, {1: 0}])
def test_lost_or_empty_replacement_of_expected_index_is_fatal(client, observed):
    _backend(client, observed)
    with pytest.raises(service.ElasticsearchPartialFailureError, match="Bestandsindex"):
        service.inspect_datasource_indices({}, _targets({1: 4}))


def test_availability_is_not_full_integrity_or_publication_proof(client):
    _backend(client, {1: 2})
    result = service.inspect_datasource_indices({}, _targets({1: 4}))
    assert result.document_counts == {1: 2}  # exact membership verification remains S4


@pytest.mark.parametrize("phase", ["get_settings", "stats"])
def test_backend_failure_is_never_reported_as_empty(client, phase):
    _backend(client, {1: 0})
    getattr(client.indices, phase).side_effect = ConnectionError("ES unavailable")
    with pytest.raises(ConnectionError, match="ES unavailable"):
        service.inspect_datasource_indices({}, _targets({1: 0}))


@pytest.mark.parametrize(
    "shards",
    [
        {},
        {"total": 1, "successful": 0, "failed": 1},
        {"total": 1, "successful": 0, "failed": 0},
        {"total": 0, "successful": 0, "failed": 0},
        {"total": 1, "successful": True, "failed": 0},
    ],
)
def test_partial_or_invalid_shard_evidence_is_fatal(client, shards):
    _backend(client, {1: 2})
    client.indices.stats.return_value["_shards"] = shards
    with pytest.raises(service.ElasticsearchPartialFailureError):
        service.inspect_datasource_indices({}, _targets({1: 2}))


@pytest.mark.parametrize("count", [None, True, -1, "2"])
def test_missing_or_invalid_backend_counts_are_fatal(client, count):
    _backend(client, {1: count})
    with pytest.raises(service.ElasticsearchPartialFailureError):
        service.inspect_datasource_indices({}, _targets({1: 0}))


@pytest.mark.parametrize("phase", ["get_settings", "stats"])
def test_alias_expansion_to_unrequested_index_is_rejected(client, phase):
    _backend(client, {1: 1})
    if phase == "get_settings":
        client.indices.get_settings.return_value["other_ds_1"] = {}
    else:
        client.indices.stats.return_value["indices"]["other_ds_1"] = {}
    with pytest.raises(service.ElasticsearchPartialFailureError, match="unerwartet"):
        service.inspect_datasource_indices({}, _targets({1: 1}))


def test_index_disappearing_between_inspection_requests_is_fatal(client):
    _backend(client, {1: 0})
    client.indices.stats.return_value["indices"] = {}
    with pytest.raises(service.ElasticsearchPartialFailureError):
        service.inspect_datasource_indices({}, _targets({1: 0}))


def test_timeout_is_fatal_even_with_complete_counts(client):
    _backend(client, {1: 1})
    client.indices.stats.return_value["timed_out"] = True
    with pytest.raises(service.ElasticsearchPartialFailureError):
        service.inspect_datasource_indices({}, _targets({1: 1}))


def test_foreign_prefix_or_duplicate_targets_fail_before_io(client):
    with pytest.raises(ValueError, match="Praefix"):
        service.inspect_datasource_indices({}, (DatasourceIndexTarget(1, "foreign_ds_1", 0),))
    with pytest.raises(ValueError, match="Doppeltes"):
        service.inspect_datasource_indices({}, _targets({1: 0}) * 2)
    client.getter.assert_not_called()
