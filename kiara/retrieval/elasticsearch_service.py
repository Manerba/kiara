"""Elasticsearch-Integration fuer KIara Hybrid-Suche.

Verwaltet den ES-Index (Erstellen, Befuellen, Loeschen, Integritaetspruefung)
und stellt die Bulk-Write-Funktion fuer die Ingestion-Pipeline bereit.
"""

import logging
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

from kiara.retrieval.elasticsearch_indices import (
    DatasourceIndexAvailability,
    DatasourceIndexTarget,
    ElasticsearchDocumentAddress,
    datasource_index_name,
    document_address,
    resolve_datasource_targets,
)
from kiara.shared.constants import (
    DEFAULT_ELASTICSEARCH_URL,
    ES_BULK_CHUNK_SIZE,
    ES_TERMS_CHUNK_SIZE,
    ES_VERIFY_SAMPLE_SIZE,
)

logger = logging.getLogger(__name__)
_visibility_state: ContextVar[dict | None] = ContextVar("es_ingestion_visibility", default=None)


def _mark_search_dirty(targets):
    state = _visibility_state.get()
    if state is not None:
        state["dirty"] = True
        state["targets"].update({t.datasource_id: t for t in targets})


def mark_search_data_adopted(config, datasource_ids):
    """Require visibility after verified chunk adoption, even if the run aborts.

    Call only after live membership reconciliation succeeds. A lookup or a
    rejected guard alone does not authorize a completion refresh.
    """
    targets = _operation_targets(config, datasource_ids, expected=1)
    state = _visibility_state.get()
    if state is not None:
        state["adopted"] = True
        state["targets"].update({t.datasource_id: t for t in targets})


@contextmanager
def ingestion_visibility(config, *, refresh_on_success=True):
    """One completion barrier, also for a successful retry without new writes.

    Pipeline preflight opts out until it has accepted persisted files. Failed
    runs refresh after a write attempt or verified chunk adoption; secondary
    errors preserve the outgoing exception or structured failure stats.
    """
    state = {
        "dirty": False,
        "adopted": False,
        "completed": refresh_on_success,
        "stats": None,
        "targets": {},
    }
    token = _visibility_state.set(state)
    primary = None
    try:
        yield state
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _visibility_state.reset(token)
        if state["dirty"] or state["adopted"] or (primary is None and state["completed"]):
            stats = state["stats"]
            adopted = state["adopted"] or bool(
                stats and (stats.get("skipped_files") or stats.get("dedup_chunks"))
            )
            try:
                refresh_index(
                    config,
                    targets=tuple(state["targets"].values()),
                    require_index=state["dirty"] or adopted,
                    require_documents=False,
                )
            except Exception as refresh_error:
                if primary is None and not (
                    stats and (stats.get("errors") or stats.get("abort_reason"))
                ):
                    raise
                diagnostic = {
                    "phase": "elasticsearch_refresh",
                    "error_type": type(refresh_error).__name__,
                    "error": str(refresh_error),
                }
                if primary is not None:
                    primary.add_note(
                        f"Elasticsearch-Abschluss-Refresh fehlgeschlagen: {refresh_error}"
                    )
                    primary.ingestion_secondary_errors = [
                        *getattr(primary, "ingestion_secondary_errors", []),
                        diagnostic,
                    ]
                else:
                    stats.setdefault("error_details", []).append(diagnostic)
                logger.error(
                    "Zusaetzlicher Elasticsearch-Abschlussfehler: %s", refresh_error, exc_info=True
                )


class ElasticsearchPartialFailureError(RuntimeError):
    """Elasticsearch hat einen Bulk-/Delete-Teil der Operation nicht ausgefuehrt."""


def _inventory_object(value) -> Mapping:
    if not isinstance(value, Mapping):
        raise ElasticsearchPartialFailureError("Elasticsearch-Indexinventar braucht ein Objekt")
    return value


def _inventory_uuid(value) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value == "_na_"
        or any(character.isspace() for character in value)
    ):
        raise ElasticsearchPartialFailureError("Elasticsearch-Indexinventar ohne gueltige UUID")
    return value


def inspect_datasource_indices(
    config: dict, targets: tuple[DatasourceIndexTarget, ...]
) -> DatasourceIndexAvailability:
    """Check explicit S1 targets in two batched requests, without wildcards.

    Missing indices are allowed only for PG-confirmed empty/new sources.
    A positive expectation also rejects an empty replacement index. Transport,
    permission and backend errors propagate; partial/malformed replies fail.
    Counts are observed primary-index entries, not distinct Qdrant chunks.
    Call after the write/refresh barrier when checking newly ingested data.
    Consumers must skip ES searches for an empty ``present`` tuple and keep
    ignore_unavailable=False / allow_no_indices=False for subsequent requests
    so deletion between inspection and use cannot silently narrow the corpus.
    UUID equality detects replacement within these two reads, not after their
    return. This inspection is not an atomic search/publication lock; callers
    still need the coordinated maintenance and publication barriers.
    """
    targets = tuple(targets)
    names = []
    seen = set()
    for target in targets:
        if target.index_name != datasource_index_name(config, target.datasource_id):
            raise ValueError("Elasticsearch-Ziel passt nicht zum konfigurierten Praefix")
        if target.index_name in seen:
            raise ValueError("Doppeltes Elasticsearch-Ziel")
        names.append(target.index_name)
        seen.add(target.index_name)
    if not names:
        return DatasourceIndexAvailability((), (), {})

    client = _get_client(config)
    settings = client.indices.get_settings(
        index=names,
        name="index.uuid",
        expand_wildcards="none",
        ignore_unavailable=True,
        allow_no_indices=True,
    )
    settings = _inventory_object(getattr(settings, "body", settings))
    if set(settings) - set(names):
        raise ElasticsearchPartialFailureError("Elasticsearch-Indexinventar unerwartet")
    uuids = {}
    for name, entry in settings.items():
        index_settings = _inventory_object(_inventory_object(entry).get("settings"))
        index = _inventory_object(index_settings.get("index"))
        uuids[name] = _inventory_uuid(index.get("uuid"))
    present, absent_empty, document_counts = [], [], {}
    for target in targets:
        if target.index_name in settings:
            present.append(target)
        elif target.expected_document_count:
            raise ElasticsearchPartialFailureError(
                f"Elasticsearch-Bestandsindex '{target.index_name}' fehlt"
            )
        else:
            absent_empty.append(target)
            document_counts[target.datasource_id] = 0
    if not present:
        return DatasourceIndexAvailability((), tuple(absent_empty), document_counts)

    # stats has no ignore_unavailable option in elasticsearch-py 8.x. Resolve
    # missing empty indices first, then require every observed index to remain.
    present_names = [target.index_name for target in present]
    response = client.indices.stats(
        index=present_names, metric="docs", level="indices", expand_wildcards="none"
    )
    response = _inventory_object(getattr(response, "body", response))
    shards = _inventory_object(response.get("_shards"))
    counts = [shards.get(key) for key in ("total", "successful", "failed")]
    if (
        any(type(value) is not int or value < 0 for value in counts)
        or counts[2] != 0
        or counts[0] != counts[1]
        or response.get("timed_out")
    ):
        raise ElasticsearchPartialFailureError("Elasticsearch-Indexinventar unvollstaendig")
    indices = response.get("indices")
    if not isinstance(indices, Mapping) or set(indices) != set(present_names):
        raise ElasticsearchPartialFailureError("Elasticsearch-Indexinventar unerwartet")
    if indices and not counts[1]:
        raise ElasticsearchPartialFailureError(
            "Elasticsearch-Indexinventar ohne erfolgreiche Shards"
        )

    for target in present:
        index_stats = _inventory_object(indices[target.index_name])
        if _inventory_uuid(index_stats.get("uuid")) != uuids[target.index_name]:
            raise ElasticsearchPartialFailureError(
                f"Elasticsearch-Index '{target.index_name}' wurde waehrend der Inventarpruefung ersetzt"
            )
        primaries = _inventory_object(index_stats.get("primaries"))
        count = _inventory_object(primaries.get("docs")).get("count")
        if type(count) is not int or count < 0:
            raise ElasticsearchPartialFailureError(
                "Elasticsearch-Dokumentzahl fehlt oder ist ungueltig"
            )
        if target.expected_document_count and count == 0:
            raise ElasticsearchPartialFailureError(
                f"Elasticsearch-Bestandsindex '{target.index_name}' ist unerwartet leer"
            )
        document_counts[target.datasource_id] = count
    return DatasourceIndexAvailability(tuple(present), tuple(absent_empty), document_counts)


def _guard_default_dense_consumer(operation: str):
    """Fence Qdrant-derived ES operations before their first ES side effect."""

    def decorate(function):
        @wraps(function)
        def wrapped(config: dict, *args, **kwargs):
            from kiara.config import get_embedding_model
            from kiara.retrieval.index_compatibility import (
                guard_index,
                stored_primary_index_target,
            )

            target = stored_primary_index_target(config, get_embedding_model(config))
            with guard_index(
                config,
                target,
                operation=operation,
                write=False,
                require_configured_default=True,
                validation_scope="stored_binding",
            ):
                return function(config, *args, **kwargs)

        return wrapped

    return decorate


# Index-Mapping mit deutschem Analyzer fuer BM25-artige Keyword-Suche
_INDEX_SETTINGS = {
    "settings": {
        "number_of_replicas": 0,
        "refresh_interval": "5s",
        "analysis": {
            "filter": {
                "german_stop": {"type": "stop", "stopwords": "_german_"},
                "german_stemmer": {"type": "stemmer", "language": "light_german"},
            },
            "analyzer": {
                "german_custom": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "german_stop", "german_stemmer"],
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "german_custom"},
            "metadata": {
                "properties": {
                    "source": {
                        "type": "text",
                        "analyzer": "simple",
                        "fields": {"raw": {"type": "keyword"}},
                    },
                    "directory": {"type": "keyword"},
                    "file_type": {"type": "keyword"},
                    "ds_ids": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "modified": {"type": "date"},
                },
            },
        },
    },
}


import atexit
import threading

_es_client_cache: dict = {}  # (url, timeout) -> Elasticsearch-Client
_es_client_lock = threading.Lock()


def _close_all_clients() -> None:
    """Schliesst alle gecachten ES-Clients beim Interpreter-Shutdown."""
    with _es_client_lock:
        for client in _es_client_cache.values():
            try:
                client.close()
            except Exception:
                # Shutdown-Cleanup; geschlossene Clients sind harmlos.
                logger.debug("ES-Client Close-on-Shutdown ignoriert", exc_info=True)
        _es_client_cache.clear()


atexit.register(_close_all_clients)


def _get_client(config: dict):
    """Erstellt oder cached einen Elasticsearch-Client aus der Config."""
    from kiara.retrieval.elasticsearch_maintenance import assert_access

    assert_access(config)
    from elasticsearch import Elasticsearch

    retrieval_cfg = config.get("retrieval", {})
    url = retrieval_cfg.get("elasticsearch_url", DEFAULT_ELASTICSEARCH_URL)
    timeout = retrieval_cfg.get("elasticsearch_timeout", 30)

    kwargs: dict = {"request_timeout": timeout}
    if "elasticsearch_verify_certs" in retrieval_cfg:
        kwargs["verify_certs"] = bool(retrieval_cfg["elasticsearch_verify_certs"])
    ca_certs = retrieval_cfg.get("elasticsearch_ca_certs")
    if ca_certs:
        kwargs["ca_certs"] = ca_certs
    api_key = retrieval_cfg.get("elasticsearch_api_key")
    if api_key:
        kwargs["api_key"] = api_key
    basic_auth_user = retrieval_cfg.get("elasticsearch_basic_auth_user")
    basic_auth_pass = retrieval_cfg.get("elasticsearch_basic_auth_password")
    if basic_auth_user and basic_auth_pass:
        kwargs["basic_auth"] = (basic_auth_user, basic_auth_pass)

    cache_key = (
        url,
        timeout,
        kwargs.get("verify_certs"),
        kwargs.get("ca_certs"),
        kwargs.get("api_key"),
        kwargs.get("basic_auth"),
    )
    with _es_client_lock:
        if cache_key not in _es_client_cache:
            _es_client_cache[cache_key] = Elasticsearch(url, **kwargs)
        return _es_client_cache[cache_key]


def is_configured(config: dict) -> bool:
    """Prueft ob Elasticsearch als Hybrid-Suche konfiguriert ist."""
    val = config.get("retrieval", {}).get("hybrid_search", "none")
    return val == "elasticsearch"


@_guard_default_dense_consumer("elasticsearch_status")
def get_status(config: dict) -> dict:
    """Exact PG memberships versus physical ES primary documents per source."""
    from kiara.retrieval.elasticsearch_inventory import inventory

    result = {
        "available": False,
        "index_exists": False,
        "doc_count": None,
        "health": "unknown",
        "error": None,
        "index_prefix": config.get("retrieval", {}).get("elasticsearch_index", "kiara"),
    }
    try:
        _get_client(config).info()
        result["available"] = True
        counts, available = inventory(config)
        result.update(
            counts,
            index_exists=bool(available.present),
            health="green" if counts["counts_match"] else "red",
        )
        if not counts["counts_match"]:
            result["error"] = "PG-/ES-Mengenabweichung pro Datenquelle"
    except Exception as exc:
        result.update(error=str(exc), health="unknown")
    return result


def _operation_targets(config, datasource_ids, *, expected=0):
    ids = list(datasource_ids)
    # Validate before constructing mappings (bool and int must not alias).
    for ds in ids:
        datasource_index_name(config, ds)
    return resolve_datasource_targets(
        config,
        ids,
        allowed_ds_ids=ids,
        expected_document_counts={ds: expected for ds in ids},
    )


def targets_for_datasources(
    config: dict, datasource_ids: Iterable[int], embedding_model: str
) -> tuple[DatasourceIndexTarget, ...]:
    """Bounded PG existence evidence for concrete sources, not corpus counts.

    expected_document_count is a lower bound (0/1) here; inventory is only an
    availability check. No million-row COUNT per file and no aggregate-cache
    zero assumption. Call under the existing model/publication guard.
    """
    targets = _operation_targets(config, datasource_ids)
    if not targets:
        return targets
    from kiara.shared.db import db_query

    with db_query() as conn:
        rows = conn.execute(
            "SELECT ds.id, EXISTS (SELECT 1 FROM chunk_datasources c "
            "WHERE c.datasource_id = ds.id AND c.embedding_model = %s) AS populated "
            "FROM unnest(%s::bigint[]) AS ds(id)",
            (embedding_model, [t.datasource_id for t in targets]),
        ).fetchall()
    counts = {row["id"]: int(row["populated"]) for row in rows}
    return (
        resolve_datasource_targets(
            config,
            counts.keys(),
            allowed_ds_ids=[t.datasource_id for t in targets],
            expected_document_counts=counts,
        )
        if len(counts) == len(targets)
        else _incomplete("PG-Zielinventar unvollstaendig")
    )


def search_datasource_ids(allowed_ds_ids: Iterable[int] | None) -> list[int]:
    """Materialize caller permissions against PG owners, including CLI DS 0.

    None is the existing unrestricted admin/CLI context. is_active controls
    ingestion, not this search permission contract. No aggregate count filter.
    """
    from kiara.retrieval.elasticsearch_indices import _datasource_ids
    from kiara.shared.db import db_query

    selected = None if allowed_ds_ids is None else _datasource_ids(allowed_ds_ids)
    if selected == set():
        return []
    with db_query() as conn:
        rows = conn.execute(
            "SELECT id FROM datasources" + (" WHERE id = ANY(%s)" if selected is not None else ""),
            (sorted(selected),) if selected is not None else (),
        ).fetchall()
    return sorted(row["id"] for row in rows)


@contextmanager
def search_scope(config: dict, allowed_ds_ids: Iterable[int] | None, embedding_model=None):
    """Hold the bound default-model fence through inventory, search and checks.

    Empty permissions/PG intersections cause no ES access or model discovery.
    The ingestion replay exception deliberately does not apply to searches.
    """
    ids = search_datasource_ids(allowed_ds_ids)
    if not ids:
        yield ()
        return
    from kiara.backend_selector import normalize_model_name
    from kiara.config import get_embedding_model
    from kiara.retrieval.index_compatibility import guard_index, stored_primary_index_target

    model = normalize_model_name(get_embedding_model(config))
    if embedding_model is not None and normalize_model_name(
        embedding_model
    ) != normalize_model_name(model):
        raise ValueError("Elasticsearch ist an das konfigurierte Default-Dense-Modell gebunden")
    target = stored_primary_index_target(config, model)
    with guard_index(
        config,
        target,
        operation="elasticsearch_search",
        write=False,
        require_configured_default=True,
        validation_scope="stored_binding",
    ):
        targets = targets_for_datasources(config, ids, model)
        yield inspect_datasource_indices(config, targets).present


def search_explicit_targets(client, targets, body):
    """Search only inspected canonical targets; partial replies are fatal."""
    from kiara.retrieval.source_policy import elasticsearch_result_window

    if not targets:
        raise ValueError("Elasticsearch-Suche braucht konkrete Ziele")
    names = [target.index_name for target in targets]
    limit = elasticsearch_result_window(client, targets)
    requested = body.get("from", 0) + body["size"]
    if requested > limit:
        raise ValueError(
            f"Suchanfrage fordert {requested} Kandidaten an; "
            f"Elasticsearch erlaubt {limit}. Quellenzahl oder Vorabruf reduzieren."
        )
    response = client.search(
        index=names,
        body=body,
        search_type="dfs_query_then_fetch",
        expand_wildcards="none",
        ignore_unavailable=False,
        allow_no_indices=False,
        allow_partial_search_results=False,
    )
    response = _inventory_object(getattr(response, "body", response))
    shards = _inventory_object(response.get("_shards"))
    counts = [shards.get(key) for key in ("total", "successful", "failed")]
    if (
        response.get("timed_out") is not False
        or response.get("terminated_early", False) is not False
        or any(type(value) is not int or value < 0 for value in counts)
        or counts[0] == 0
        or counts[0] != counts[1]
        or counts[2] != 0
    ):
        raise ElasticsearchPartialFailureError("Elasticsearch-Suche unvollstaendig")
    hits = _inventory_object(response.get("hits")).get("hits")
    if not isinstance(hits, list) or any(
        not isinstance(hit, Mapping) or hit.get("_index") not in names for hit in hits
    ):
        raise ElasticsearchPartialFailureError("Elasticsearch-Suche liefert unerwartete Ziele")
    return response


def model_uses_elasticsearch(config: dict, embedding_model: str) -> bool:
    from kiara.backend_selector import normalize_model_name
    from kiara.config import get_embedding_model

    return is_configured(config) and normalize_model_name(embedding_model) == normalize_model_name(
        get_embedding_model(config)
    )


def _incomplete(message):
    raise ElasticsearchPartialFailureError(message)


def _validate_targets(config, targets):
    for target in targets:
        if not isinstance(
            target, DatasourceIndexTarget
        ) or target.index_name != datasource_index_name(config, target.datasource_id):
            raise ValueError("Ungueltiges Elasticsearch-Ziel")
    return tuple(dict.fromkeys(targets))


def _validate_addresses(config, addresses):
    result = []
    for address in addresses:
        if not isinstance(address, ElasticsearchDocumentAddress) or address != document_address(
            config, address.datasource_id, address.chunk_id
        ):
            raise ValueError("Vollstaendige Elasticsearch-Dokumentadresse erforderlich")
        result.append(address)
    return list(dict.fromkeys(result))


def create_index(config: dict, *, datasource_id: int) -> bool:
    """Create exactly one canonical datasource index; failures are fatal."""
    index_name = datasource_index_name(config, datasource_id)
    es = _get_client(config)
    if not es.indices.exists(index=index_name):
        try:
            result = es.indices.create(index=index_name, body=_INDEX_SETTINGS)
            if not result.get("acknowledged"):
                _incomplete("Elasticsearch Indexanlage nicht bestaetigt")
        except Exception as exc:
            # Concurrent creators are harmless; all other errors remain fatal.
            if getattr(exc, "error", None) != "resource_already_exists_exception":
                raise
    return True


def delete_datasource_index(config, *, datasource_id):
    """Lifecycle deletion of one explicit DS, including entries without hashes."""
    name = datasource_index_name(config, datasource_id)
    client = _get_client(config)
    if client.indices.exists(index=name):
        result = client.indices.delete(index=name)
        if not result.get("acknowledged"):
            _incomplete("Elasticsearch DS-Loeschung nicht bestaetigt")


@_guard_default_dense_consumer("elasticsearch_delete_indices")
def delete_index(config: dict) -> bool:
    """Delete only current PG-owned source indices, under offline maintenance."""
    from kiara.retrieval.elasticsearch_inventory import exact_targets
    from kiara.retrieval.elasticsearch_maintenance import require_session

    require_session(config)
    for target in exact_targets(config):
        delete_datasource_index(config, datasource_id=target.datasource_id)
    return True


def check_missing_ds_ids(config: dict) -> dict:
    if not is_configured(config):
        return {"total_docs": 0, "missing_ds_ids": 0, "safe_to_proceed": True, "error": None}
    return _check_missing_ds_ids(config)


@_guard_default_dense_consumer("elasticsearch_missing_memberships")
def _check_missing_ds_ids(config: dict) -> dict:
    from kiara.retrieval.elasticsearch_inventory import inventory

    result = {"total_docs": None, "missing_ds_ids": None, "safe_to_proceed": False, "error": None}
    if not is_configured(config):
        return {**result, "total_docs": 0, "missing_ds_ids": 0, "safe_to_proceed": True}
    try:
        counts, available = inventory(config)
        missing = 0
        for target in available.present:
            response = search_explicit_targets(
                _get_client(config),
                (target,),
                {
                    "size": 0,
                    "track_total_hits": True,
                    "query": {
                        "bool": {"must_not": [{"term": {"metadata.ds_ids": target.datasource_id}}]}
                    },
                },
            )
            total = response["hits"]["total"]
            if total["relation"] != "eq" or type(total["value"]) is not int:
                _incomplete("Unvollstaendige Zuordnungszaehlung")
            missing += total["value"]
        result.update(
            total_docs=counts["doc_count"],
            missing_ds_ids=missing,
            safe_to_proceed=missing == 0 and counts["counts_match"],
            datasources=counts["datasources"],
        )
        if not counts["counts_match"]:
            result["error"] = "PG-/ES-Mengenabweichung"
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _mget(config, addresses):
    addresses = _validate_addresses(config, addresses)
    if not addresses:
        return {}
    client = _get_client(config)
    result = {}
    for start in range(0, len(addresses), ES_BULK_CHUNK_SIZE):
        batch = addresses[start : start + ES_BULK_CHUNK_SIZE]
        by_key = {(a.index_name, a.chunk_id): a for a in batch}
        docs = client.mget(
            docs=[{"_index": a.index_name, "_id": a.chunk_id} for a in batch], realtime=True
        )["docs"]
        if len(docs) != len(batch) or {(d.get("_index"), d.get("_id")) for d in docs} != set(
            by_key
        ):
            _incomplete("Elasticsearch mget unvollstaendig")
        for doc in docs:
            if doc.get("error") or type(doc.get("found")) is not bool:
                _incomplete("Elasticsearch-Zuordnungsabfrage fehlgeschlagen")
            result[by_key[(doc["_index"], doc["_id"])]] = doc
    return result


def _bulk_actions(config, actions):
    if not actions:
        return 0
    from elasticsearch import helpers

    success, errors = helpers.bulk(
        _get_client(config), actions, chunk_size=ES_BULK_CHUNK_SIZE, raise_on_error=False
    )
    if errors or success != len(actions):
        _incomplete(f"Elasticsearch Bulk unvollstaendig: {success}/{len(actions)} Aktionen")
    return success


def update_ds_ids(
    config: dict,
    ds_ids_map: Mapping[ElasticsearchDocumentAddress, list[int]],
    *,
    refresh: bool = True,
    skip_unchanged: bool = False,
) -> int:
    """CAS metadata updates to complete addresses; never creates documents."""
    if not ds_ids_map or not is_configured(config):
        return 0
    from kiara.retrieval.datasource_metadata import same_ds_ids

    addresses = _validate_addresses(config, ds_ids_map)
    for address, owners in ds_ids_map.items():
        _operation_targets(config, owners)
        if address.datasource_id not in owners:
            raise ValueError("DS-Ziel fehlt in Metadaten; expliziten Delete verwenden")
    docs = _mget(config, addresses)
    actions = []
    for address, doc in docs.items():
        if not doc["found"]:
            _incomplete("Elasticsearch-Zuordnungsdokument fehlt")
        owners = ds_ids_map[address]
        if skip_unchanged and same_ds_ids(
            doc.get("_source", {}).get("metadata", {}).get("ds_ids"), owners
        ):
            continue
        actions.append(
            {
                "_op_type": "update",
                "_index": address.index_name,
                "_id": address.chunk_id,
                "if_seq_no": doc["_seq_no"],
                "if_primary_term": doc["_primary_term"],
                "doc": {"metadata": {"ds_ids": owners}},
            }
        )
    targets = _operation_targets(config, [a.datasource_id for a in addresses], expected=1)
    if actions:
        _mark_search_dirty(targets)
    count = _bulk_actions(config, actions)
    if count and refresh:
        refresh_index(config, targets=targets, require_index=True)
    return count


def reconcile_memberships(config, memberships, *, adopting_datasource_id, replayable=()):
    """Reconcile existing DS copies; new owner gets canonical text at publication.

    memberships is a batched PG-confirmed snapshot under the caller's existing
    model/document guards. Missing foreign copies require explicit replayable
    addresses, backed by a pending publication without an owner file completion.
    Replay only permits reaching canonical publication; it is not a write proof.
    Use realtime reads before refresh, without an additional corpus-count query.
    """
    datasource_index_name(config, adopting_datasource_id)
    mapping = {
        document_address(config, ds, cid): owners
        for cid, owners in memberships.items()
        for ds in owners
    }
    if not mapping:
        return {}
    replayable = set(replayable)
    if not replayable.issubset(mapping):
        raise ValueError("Replay ohne bestaetigte PG-Zuordnung")
    targets = _operation_targets(config, {a.datasource_id for a in mapping})
    result = _get_client(config).indices.get_settings(
        index=[t.index_name for t in targets],
        name="index.uuid",
        expand_wildcards="none",
        ignore_unavailable=True,
        allow_no_indices=True,
    )
    present = _inventory_object(getattr(result, "body", result))
    if set(present) - {t.index_name for t in targets} or any(
        a.datasource_id != adopting_datasource_id
        and a not in replayable
        and a.index_name not in present
        for a in mapping
    ):
        _incomplete("ES-Zuordnungsindex fehlt")
    addresses = [a for a in mapping if a.index_name in present]
    actual = get_ds_ids(config, addresses)
    if any(
        a not in actual and a.datasource_id != adopting_datasource_id and a not in replayable
        for a in addresses
    ):
        _incomplete("ES-Zuordnungsdokument fehlt")
    existing = {a: mapping[a] for a in actual}
    update_ds_ids(config, existing, refresh=False, skip_unchanged=True)
    return existing


def get_ds_ids(
    config: dict, addresses: Sequence[ElasticsearchDocumentAddress]
) -> dict[ElasticsearchDocumentAddress, list[int] | None]:
    if not addresses or not is_configured(config):
        return {}
    return {
        a: d.get("_source", {}).get("metadata", {}).get("ds_ids")
        for a, d in _mget(config, addresses).items()
        if d["found"]
    }


def refresh_index(
    config: dict,
    *,
    targets: tuple[DatasourceIndexTarget, ...],
    require_index: bool = False,
    require_documents: bool = False,
) -> None:
    targets = _validate_targets(config, targets)
    if not targets or not is_configured(config):
        return
    es = _get_client(config)
    # Refresh BEFORE stats: new writes may not be visible yet.
    settings = es.indices.get_settings(
        index=[t.index_name for t in targets],
        name="index.uuid",
        expand_wildcards="none",
        ignore_unavailable=True,
        allow_no_indices=True,
    )
    present = _inventory_object(getattr(settings, "body", settings))
    if set(present) - {t.index_name for t in targets}:
        _incomplete("Elasticsearch Refresh: unerwartete Indizes")
    if any(
        t.index_name not in present
        and (require_index or require_documents or t.expected_document_count)
        for t in targets
    ):
        _incomplete("Elasticsearch-Abschluss: Index fehlt trotz erwarteter Suchdaten")
    if present:
        result = es.indices.refresh(
            index=list(present),
            expand_wildcards="none",
            ignore_unavailable=False,
            allow_no_indices=False,
        )
        shards = result.get("_shards", {})
        if (
            not isinstance(shards, dict)
            or any(type(shards.get(key)) is not int for key in ("total", "successful", "failed"))
            or shards["total"] <= 0
            or shards["successful"] != shards["total"]
            or shards["failed"] != 0
        ):
            _incomplete("Elasticsearch-Abschluss-Refresh unvollstaendig")
        # Reuse the strict S1 inventory including UUID and shard validation.
        expected = tuple(
            DatasourceIndexTarget(
                t.datasource_id,
                t.index_name,
                max(t.expected_document_count, int(require_documents)),
            )
            for t in targets
        )
        inspect_datasource_indices(config, expected)


def bulk_index_chunks(
    config: dict, texts: list[str], metadatas: list[dict], ids: list[str], *, datasource_id: int
) -> int:
    index_name = datasource_index_name(config, datasource_id)
    if not (len(texts) == len(metadatas) == len(ids)):
        raise ValueError("bulk_index_chunks Laengenabweichung")
    if len(set(ids)) != len(ids):
        raise ValueError("Doppelte Chunk-IDs im DS-Batch")
    for cid, meta in zip(ids, metadatas, strict=True):
        document_address(config, datasource_id, cid)
        owners = meta.get("ds_ids")
        if not isinstance(owners, list):
            raise ValueError("ES-Write ohne bestaetigte DS-Zuordnung")
        _operation_targets(config, owners)
        if datasource_id not in owners:
            raise ValueError("ES-Write ohne Ziel-DS in Metadaten")
    if not ids:
        return 0
    targets = _operation_targets(config, [datasource_id])
    _mark_search_dirty(targets)
    create_index(config, datasource_id=datasource_id)
    count = _bulk_actions(
        config,
        [
            {"_index": index_name, "_id": cid, "_source": {"text": text, "metadata": meta}}
            for text, meta, cid in zip(texts, metadatas, ids, strict=True)
        ],
    )

    mark_search_data_adopted(config, [datasource_id])
    return count


def delete_by_ids(config: dict, addresses: Sequence[ElasticsearchDocumentAddress]) -> int:
    """Realtime, CAS-protected physical deletes, confined to addressed indices."""
    if not addresses or not is_configured(config):
        return 0
    addresses = _validate_addresses(config, addresses)
    targets = _operation_targets(config, [a.datasource_id for a in addresses])
    changed = 0
    for start in range(0, len(addresses), ES_BULK_CHUNK_SIZE):
        docs = _mget(config, addresses[start : start + ES_BULK_CHUNK_SIZE])
        actions = [
            {
                "_op_type": "delete",
                "_index": a.index_name,
                "_id": a.chunk_id,
                "if_seq_no": d["_seq_no"],
                "if_primary_term": d["_primary_term"],
            }
            for a, d in docs.items()
            if d["found"]
        ]
        if actions:
            _mark_search_dirty(targets)
            changed += _bulk_actions(config, actions)
    if changed and _visibility_state.get() is None:
        refresh_index(config, targets=targets, require_index=True)
    return changed


def get_ids_by_source(
    config: dict, source: str, *, targets: tuple[DatasourceIndexTarget, ...]
) -> list[ElasticsearchDocumentAddress]:
    """Vollstaendige Quellen-IDs fuer Cleanup sichern; Teilantworten sind fatal."""
    if not is_configured(config):
        return []
    targets = _validate_targets(config, targets)
    if not targets:
        return []
    refresh_index(config, targets=targets)
    available = inspect_datasource_indices(config, targets)
    if not available.present:
        return []
    es = _get_client(config)
    index_name = [t.index_name for t in available.present]
    by_index = {t.index_name: t.datasource_id for t in available.present}

    response = es.search(
        index=index_name,
        query={"term": {"metadata.source.raw": source}},
        source=False,
        sort=["_doc"],
        scroll="2m",
        size=ES_BULK_CHUNK_SIZE,
        track_total_hits=True,
        allow_partial_search_results=False,
        ignore_unavailable=False,
        allow_no_indices=False,
        expand_wildcards="none",
    )
    scroll_id = response.get("_scroll_id")
    ids = set()
    try:
        total = response.get("hits", {}).get("total", {})
        if total.get("relation") != "eq" or not isinstance(total.get("value"), int):
            raise ElasticsearchPartialFailureError("ES-Cleanup-Inventar: Gesamtzahl fehlt")
        expected = total["value"]
        while True:
            if response.get("timed_out") or response.get("_shards", {}).get("failed", 0):
                raise ElasticsearchPartialFailureError("ES-Cleanup-Inventar: Suche unvollstaendig")
            hits = response["hits"]["hits"]
            for hit in hits:
                if hit.get("_index") not in by_index:
                    _incomplete("ES-Cleanup-Inventar: fremder Index")
                cid = document_address(config, by_index[hit["_index"]], hit.get("_id"))
                if cid in ids:
                    raise ElasticsearchPartialFailureError(
                        "ES-Cleanup-Inventar: ungueltige Chunk-ID"
                    )
                ids.add(cid)
            if not hits or len(ids) >= expected:
                break
            if not scroll_id:
                raise ElasticsearchPartialFailureError("ES-Cleanup-Inventar: Fortsetzung fehlt")
            response = es.scroll(scroll_id=scroll_id, scroll="2m")
            scroll_id = response.get("_scroll_id")
        if len(ids) != expected:
            raise ElasticsearchPartialFailureError("ES-Cleanup-Inventar: Chunk-IDs fehlen")
        return sorted(ids, key=lambda a: (a.index_name, a.chunk_id))
    finally:
        if scroll_id:
            es.options(ignore_status=404).clear_scroll(scroll_id=scroll_id)


def delete_by_sources(
    config: dict, file_paths: list[str], *, targets: tuple[DatasourceIndexTarget, ...]
) -> int:
    """Delete physical entries in explicit DS targets using bounded search batches.

    Query deletes retain ES version-conflict protection; never update/delete a
    foreign datasource index. Missing expected indices and partial results fail.
    """
    targets = _validate_targets(config, targets)
    if not file_paths or not targets or not is_configured(config):
        return 0
    refresh_index(config, targets=targets)
    available = inspect_datasource_indices(config, targets)
    if not available.present:
        return 0
    client = _get_client(config)
    deleted = 0
    _mark_search_dirty(available.present)
    for start in range(0, len(file_paths), ES_TERMS_CHUNK_SIZE):
        response = client.delete_by_query(
            index=[t.index_name for t in available.present],
            query={
                "terms": {"metadata.source.raw": file_paths[start : start + ES_TERMS_CHUNK_SIZE]}
            },
            refresh=True,
            conflicts="abort",
            ignore_unavailable=False,
            allow_no_indices=False,
        )
        if (
            response.get("failures")
            or response.get("timed_out")
            or response.get("version_conflicts")
            or type(response.get("deleted")) is not int
        ):
            _incomplete("Elasticsearch delete_by_query unvollstaendig")
        deleted += response["deleted"]
    return deleted


@_guard_default_dense_consumer("elasticsearch_rebuild_from_vectors")
def rebuild_from_vectorstore(
    config: dict, progress_callback=None, *, dry_run=False, batch_size=500, cancel=None
) -> dict:
    from kiara.retrieval.elasticsearch_inventory import rebuild

    return rebuild(config, progress_callback, dry_run=dry_run, batch_size=batch_size, cancel=cancel)


@_guard_default_dense_consumer("elasticsearch_integrity_read")
def verify_integrity(
    config: dict, sample_size: int = ES_VERIFY_SAMPLE_SIZE, *, full=False, batch_size=500
) -> dict:
    from kiara.retrieval.elasticsearch_inventory import verify

    return verify(config, sample_size, full=full, batch_size=batch_size)


@_guard_default_dense_consumer("elasticsearch_repair_memberships")
def repair_memberships(config, memberships, embedding_model):
    """Repair existing physical copies only; missing copies require offline rebuild."""
    if not model_uses_elasticsearch(config, embedding_model):
        return 0
    from kiara.ingestion.document_publication import assert_documents_published
    from kiara.retrieval.elasticsearch_inventory import _store
    from kiara.retrieval.embedding import collection_name_for_model
    from kiara.shared.db import db_query

    with db_query() as conn:
        rows = conn.execute(
            "SELECT chunk_id, datasource_id FROM chunk_datasources "
            "WHERE embedding_model = %s AND chunk_id = ANY(%s)",
            (embedding_model, list(memberships)),
        ).fetchall()
    confirmed = {cid: [] for cid in memberships}
    for row in rows:
        confirmed[row["chunk_id"]].append(row["datasource_id"])
    if any(
        not owners or sorted(owners) != sorted(confirmed[cid])
        for cid, owners in memberships.items()
    ):
        _incomplete("Reparatur ohne bestaetigte PG-Zuordnungen")
    docs = _store(config).get_documents_by_ids(list(memberships), include=["metadatas"])
    if set(docs["ids"]) != set(memberships):
        _incomplete("Kanonische Reparaturdokumente fehlen")
    assert_documents_published(
        collection_name_for_model(embedding_model), [{"metadata": m} for m in docs["metadatas"]]
    )
    updates = {
        document_address(config, ds, cid): owners
        for cid, owners in memberships.items()
        for ds in owners
    }
    return update_ds_ids(config, updates)


@_guard_default_dense_consumer("elasticsearch_storage")
def storage_stats(config):
    from kiara.retrieval.elasticsearch_inventory import inventory

    counts, available = inventory(config)
    if not counts["counts_match"]:
        _incomplete("PG-/ES-Mengenabweichung bei Speichermessung")
    names = [t.index_name for t in available.present]
    result = {
        "status": "ok",
        "size_bytes": 0,
        "primary_store_bytes": 0,
        "total_store_bytes": 0,
        "primary_translog_bytes": 0,
        "total_translog_bytes": 0,
        "indices": names,
        "unit": "primary_and_replica_store_plus_translog_bytes",
        "retained_common_index_included": False,
    }
    if not names:
        return result
    response = (
        _get_client(config)
        .options(request_timeout=10, max_retries=0)
        .indices.stats(
            index=names, metric="store,translog", level="indices", expand_wildcards="none"
        )
    )
    response = _inventory_object(getattr(response, "body", response))
    shards = response.get("_shards", {})
    values = [shards.get(k) for k in ("total", "successful", "failed")]
    if (
        any(type(v) is not int or v < 0 for v in values)
        or not values[0]
        or values[0] != values[1]
        or values[2]
        or response.get("timed_out")
        or set(response.get("indices", {})) != set(names)
    ):
        _incomplete("Unvollstaendige ES-Speichermessung")
    for entry in response["indices"].values():
        for section, label in (("primaries", "primary"), ("total", "total")):
            for metric in ("store", "translog"):
                value = entry[section][metric]["size_in_bytes"]
                if type(value) is not int or value < 0:
                    _incomplete("Ungueltige ES-Speichermessung")
                result[f"{label}_{metric}_bytes"] += value
    result["size_bytes"] = result["total_store_bytes"] + result["total_translog_bytes"]
    return result
