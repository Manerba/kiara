"""Elasticsearch-Integration fuer KIara Hybrid-Suche.

Verwaltet den ES-Index (Erstellen, Befuellen, Loeschen, Integritaetspruefung)
und stellt die Bulk-Write-Funktion fuer die Ingestion-Pipeline bereit.
"""

import hashlib
import logging
from typing import Iterator

from kiara.shared.constants import (
    DEFAULT_ELASTICSEARCH_URL,
    ES_BULK_CHUNK_SIZE,
    ES_PROGRESS_CALLBACK_INTERVAL,
    ES_TERMS_CHUNK_SIZE,
    ES_VERIFY_SAMPLE_SIZE,
)

logger = logging.getLogger(__name__)


class ElasticsearchPartialFailureError(RuntimeError):
    """Elasticsearch hat einen Bulk-/Delete-Teil der Operation nicht ausgefuehrt."""


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


def _get_index_name(config: dict) -> str:
    return config.get("retrieval", {}).get("elasticsearch_index", "kiara")


def is_configured(config: dict) -> bool:
    """Prueft ob Elasticsearch als Hybrid-Suche konfiguriert ist."""
    val = config.get("retrieval", {}).get("hybrid_search", "none")
    return val == "elasticsearch"


def get_status(config: dict) -> dict:
    """Gibt den aktuellen ES-Index-Status zurueck.

    Returns:
        Dict mit keys: available, index_exists, doc_count, health, error
    """
    index_name = _get_index_name(config)
    result = {
        "available": False,
        "index_exists": False,
        "doc_count": 0,
        "health": "unknown",
        "index_name": index_name,
        "error": None,
    }

    try:
        es = _get_client(config)
        es.info()
        result["available"] = True
    except Exception as exc:
        logger.warning("Elasticsearch nicht erreichbar fuer get_status: %s", exc, exc_info=True)
        result["error"] = f"Elasticsearch nicht erreichbar: {exc}"
        return result

    try:
        if es.indices.exists(index=index_name):
            result["index_exists"] = True
            stats = es.indices.stats(index=index_name)
            result["doc_count"] = stats["_all"]["primaries"]["docs"]["count"]
            health = es.cluster.health(index=index_name)
            result["health"] = health["status"]
    except Exception as exc:
        logger.warning(
            "ES Index-Status fuer '%s' nicht lesbar: %s",
            index_name,
            exc,
            exc_info=True,
        )
        result["error"] = f"Index-Status konnte nicht gelesen werden: {exc}"

    return result


def create_index(config: dict) -> bool:
    """Erstellt den ES-Index mit deutschem Analyzer (idempotent)."""
    try:
        es = _get_client(config)
        index_name = _get_index_name(config)
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=_INDEX_SETTINGS)
            logger.info("Elasticsearch-Index '%s' erstellt", index_name)
        return True
    except Exception as exc:
        logger.error("Elasticsearch create_index fehlgeschlagen: %s", exc, exc_info=True)
        return False


def delete_index(config: dict) -> bool:
    """Loescht den ES-Index."""
    try:
        es = _get_client(config)
        index_name = _get_index_name(config)
        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)
            logger.info("Elasticsearch-Index '%s' geloescht", index_name)
        return True
    except Exception as exc:
        logger.error("Elasticsearch delete_index fehlgeschlagen: %s", exc, exc_info=True)
        return False


def check_missing_ds_ids(config: dict) -> dict:
    """Prueft ob ES-Dokumente ohne metadata.ds_ids existieren.

    Returns:
        {
            "total_docs": int,
            "missing_ds_ids": int,
            "safe_to_proceed": bool,
            "error": str | None,
        }
    """
    result = {
        "total_docs": 0,
        "missing_ds_ids": 0,
        "safe_to_proceed": True,
        "error": None,
    }

    if not is_configured(config):
        return result

    try:
        es = _get_client(config)
        index_name = _get_index_name(config)

        if not es.indices.exists(index=index_name):
            return result

        total = es.count(index=index_name)["count"]
        missing = es.count(
            index=index_name,
            body={"query": {"bool": {"must_not": [{"exists": {"field": "metadata.ds_ids"}}]}}},
        )["count"]

        result.update(
            {
                "total_docs": total,
                "missing_ds_ids": missing,
                "safe_to_proceed": missing == 0,
            }
        )
    except Exception as exc:
        logger.warning("ES Pre-Flight fehlgeschlagen: %s", exc, exc_info=True)
        result["safe_to_proceed"] = False
        result["error"] = str(exc)

    return result


def update_ds_ids(config: dict, ds_ids_map: dict[str, list[int]]) -> int:
    """Aktualisiert metadata.ds_ids fuer die gegebenen Chunk-IDs per Bulk-Update."""
    if not ds_ids_map or not is_configured(config):
        return 0

    es = _get_client(config)
    index_name = _get_index_name(config)
    if not es.indices.exists(index=index_name):
        return 0

    actions = [
        {
            "_op_type": "update",
            "_index": index_name,
            "_id": chunk_id,
            "doc": {"metadata": {"ds_ids": ds_ids}},
        }
        for chunk_id, ds_ids in ds_ids_map.items()
    ]

    from elasticsearch import helpers

    success, errors = helpers.bulk(
        es,
        actions,
        chunk_size=ES_BULK_CHUNK_SIZE,
        raise_on_error=False,
    )
    if errors:
        raise ElasticsearchPartialFailureError(
            f"Elasticsearch ds_ids-Update unvollstaendig: "
            f"{len(errors)} Fehler bei {len(actions)} Aktionen"
        )
    if success:
        es.indices.refresh(index=index_name)
    return success


def bulk_index_chunks(config: dict, texts: list[str], metadatas: list[dict], ids: list[str]) -> int:
    """Schreibt Chunks per Bulk-API in den ES-Index.

    Erstellt den Index automatisch falls noetig.

    Returns:
        Anzahl erfolgreich indexierter Dokumente.
    """
    if not texts:
        return 0

    if not (len(texts) == len(metadatas) == len(ids)):
        raise ValueError(
            f"bulk_index_chunks Laengenabweichung: "
            f"texts={len(texts)}, metadatas={len(metadatas)}, ids={len(ids)}"
        )

    es = _get_client(config)
    index_name = _get_index_name(config)

    if not create_index(config):
        raise ElasticsearchPartialFailureError(
            f"Elasticsearch-Index '{index_name}' konnte nicht erstellt/verifiziert werden — "
            f"Bulk-Indexierung abgebrochen (fail-closed)"
        )

    actions = []
    for text, meta, doc_id in zip(texts, metadatas, ids, strict=False):
        actions.append(
            {
                "_index": index_name,
                "_id": doc_id,
                "_source": {
                    "text": text,
                    "metadata": meta,
                },
            }
        )

    from elasticsearch import helpers

    success, errors = helpers.bulk(es, actions, chunk_size=ES_BULK_CHUNK_SIZE, raise_on_error=False)
    if errors:
        raise ElasticsearchPartialFailureError(
            f"Elasticsearch Bulk-Indexierung unvollstaendig: "
            f"{len(errors)} Fehler bei {len(actions)} Aktionen"
        )

    return success


def delete_by_sources(config: dict, file_paths: list[str], datasource_id: int | None = None) -> int:
    """Loescht alle Chunks fuer die angegebenen Quelldateien aus dem ES-Index.

    Wenn datasource_id gesetzt ist, wird DS-spezifisch geloescht:
    - Dokumente mit nur dieser DS-ID werden komplett geloescht.
    - Dokumente mit weiteren DS-IDs behalten das Dokument, nur die ID wird entfernt.
    - Altdaten ohne ds_ids-Feld bleiben unangetastet.

    Ohne datasource_id: pfadbasiertes Loeschen.

    Returns:
        Anzahl geloeschter Dokumente.
    """
    es = _get_client(config)
    index_name = _get_index_name(config)

    if not es.indices.exists(index=index_name):
        return 0

    # file_paths in Chunks splitten: ES `terms`-Queries expandieren in
    # bool/should-Klauseln und brechen bei sehr grossen Listen mit
    # too_many_clauses (Default-Limit 1024-4096 je nach ES-Version).
    def _chunked(items: list[str], size: int) -> Iterator[list[str]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    deleted = 0

    if datasource_id is not None:
        from elasticsearch import helpers

        for batch in _chunked(file_paths, ES_TERMS_CHUNK_SIZE):
            # DS-spezifisches Loeschen: Betroffene Dokumente per Scroll finden
            query = {
                "bool": {
                    "must": [
                        {"terms": {"metadata.source.raw": batch}},
                        {"term": {"metadata.ds_ids": datasource_id}},
                    ]
                }
            }

            def _generate_actions(query=query):
                for hit in helpers.scan(
                    es,
                    index=index_name,
                    query={"query": query},
                    _source=["metadata.ds_ids"],
                    scroll="2m",
                    size=1000,
                ):
                    ds_ids = hit["_source"].get("metadata", {}).get("ds_ids", [])
                    new_ids = [x for x in ds_ids if x != datasource_id]
                    if not new_ids:
                        yield {"_op_type": "delete", "_index": index_name, "_id": hit["_id"]}
                    else:
                        yield {
                            "_op_type": "update",
                            "_index": index_name,
                            "_id": hit["_id"],
                            "doc": {"metadata": {"ds_ids": new_ids}},
                        }

            # Actions materialisieren: helpers.bulk's success-Counter mischt
            # delete und update. Counter im Generator zu mutieren waere bei
            # Bulk-Abbruch unzuverlaessig (Items werden vor ES-Ack geyieldet).
            actions = list(_generate_actions())
            delete_actions = sum(1 for a in actions if a["_op_type"] == "delete")

            success, errors = helpers.bulk(
                es, actions, chunk_size=ES_BULK_CHUNK_SIZE, raise_on_error=False
            )
            if errors:
                raise ElasticsearchPartialFailureError(
                    f"Elasticsearch DS-Loeschung unvollstaendig: {len(errors)} Fehler"
                )
            deleted += delete_actions
            if success or errors:
                es.indices.refresh(index=index_name)
    else:
        # Pfadbasierte Loeschung ohne DS-spezifische ds_ids-Aktualisierung.
        for batch in _chunked(file_paths, ES_TERMS_CHUNK_SIZE):
            resp = es.delete_by_query(
                index=index_name,
                body={"query": {"terms": {"metadata.source.raw": batch}}},
                refresh=True,
            )
            failures = resp.get("failures") or []
            if failures:
                raise ElasticsearchPartialFailureError(
                    f"Elasticsearch delete_by_query unvollstaendig: {len(failures)} Fehler"
                )
            deleted += resp.get("deleted", 0)

    if deleted:
        logger.info("ES: %d Chunks fuer %d Dateien geloescht", deleted, len(file_paths))
    return deleted


def rebuild_from_vectorstore(config: dict, progress_callback=None) -> dict:
    """Kopiert alle Chunks aus dem VectorStore nach Elasticsearch.

    Loescht den bestehenden Index und baut ihn komplett neu auf.

    Args:
        config: KIara-Konfiguration.
        progress_callback: Optional callback(indexed, total).

    Returns:
        Dict mit keys: total, indexed, errors.
    """
    from elasticsearch import helpers

    from kiara.retrieval.vectorstore import VectorStore

    es = _get_client(config)
    index_name = _get_index_name(config)

    # Index loeschen und neu erstellen
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    # Fuer Bulk-Import: refresh deaktivieren
    settings = _INDEX_SETTINGS.copy()
    settings["settings"] = {**settings["settings"], "refresh_interval": "-1"}
    es.indices.create(index=index_name, body=settings)

    total = 0
    indexed = 0
    errors = 0

    try:
        store = VectorStore(config)
        total = store.get_stats().get("total_chunks", 0)

        if total == 0:
            return {"total": 0, "indexed": 0, "errors": 0}

        def _generate_actions() -> Iterator[dict]:
            """Scrollt Qdrant batch-weise und yieldet ES-Bulk-Actions (streaming, kein OOM)."""
            from kiara.shared.text_utils import sanitize_utf8

            for batch in store.scroll_all_with_text(batch_size=1000):
                for chunk_id, text, m in batch:
                    text = sanitize_utf8(text)
                    if "ds_ids" not in m and "datasource_id" in m:
                        ds_id = m.pop("datasource_id", None)
                        if ds_id is not None:
                            m["ds_ids"] = [ds_id] if not isinstance(ds_id, list) else ds_id
                    yield {
                        "_index": index_name,
                        "_id": chunk_id,
                        "_source": {
                            "text": text,
                            "metadata": m,
                        },
                    }

        for ok, _item in helpers.streaming_bulk(
            es, _generate_actions(), chunk_size=ES_BULK_CHUNK_SIZE, raise_on_error=False
        ):
            if ok:
                indexed += 1
            else:
                errors += 1

            if progress_callback and (indexed + errors) % ES_PROGRESS_CALLBACK_INTERVAL == 0:
                progress_callback(indexed + errors, total)
    finally:
        # Refresh wieder aktivieren und Index refreshen
        es.indices.put_settings(index=index_name, body={"refresh_interval": "5s"})
        es.indices.refresh(index=index_name)

    logger.info("ES-Rebuild abgeschlossen: %d/%d indexiert, %d Fehler", indexed, total, errors)

    return {"total": total, "indexed": indexed, "errors": errors}


def verify_integrity(config: dict, sample_size: int = ES_VERIFY_SAMPLE_SIZE) -> dict:
    """Prueft die Integritaet des ES-Index gegen den VectorStore.

    Vergleicht eine Stichprobe von Chunks (Text-Hash-Vergleich).

    Args:
        config: KIara-Konfiguration.
        sample_size: Anzahl der Stichproben-Chunks.

    Returns:
        Dict mit keys: qdrant_count, es_count, sampled, matching, mismatches,
                       missing_in_es, errors, integrity_pct.
    """
    from elasticsearch import NotFoundError

    from kiara.retrieval.vectorstore import VectorStore

    store = VectorStore(config)
    es = _get_client(config)
    index_name = _get_index_name(config)

    qdrant_count = store.get_stats().get("total_chunks", 0)

    es_count = 0
    if es.indices.exists(index=index_name):
        stats = es.indices.stats(index=index_name)
        es_count = stats["_all"]["primaries"]["docs"]["count"]

    result = {
        "qdrant_count": qdrant_count,
        "es_count": es_count,
        "sampled": 0,
        "matching": 0,
        "mismatches": [],
        "missing_in_es": 0,
        "errors": 0,
        "integrity_pct": 0.0,
    }

    if qdrant_count == 0 or es_count == 0:
        return result

    # Nur IDs laden (kein Text -> kein OOM bei grossen Collections)
    all_ids = store.get_all_documents(include=[])["ids"]

    import random

    sample_ids = random.sample(all_ids, min(sample_size, len(all_ids)))

    # Stichproben-Chunks gezielt laden
    sample_docs = store.get_documents_by_ids(sample_ids, include=["documents", "metadatas"])
    sample_texts = sample_docs.get("documents", [])
    sample_metas = sample_docs.get("metadatas", [])
    sample_resolved_ids = sample_docs["ids"]

    sampled = 0
    matching = 0
    missing = 0
    errors = 0
    mismatches = []

    for i, chunk_id in enumerate(sample_resolved_ids):
        qdrant_text = (sample_texts[i] if i < len(sample_texts) else "") or ""
        qdrant_hash = hashlib.md5(qdrant_text.encode()).hexdigest()

        try:
            es_doc = es.get(index=index_name, id=chunk_id, _source=["text"])
            es_text = es_doc["_source"].get("text", "")
            es_hash = hashlib.md5(es_text.encode()).hexdigest()
            sampled += 1

            if qdrant_hash == es_hash:
                matching += 1
            else:
                source = (sample_metas[i] if i < len(sample_metas) else {}) or {}
                mismatches.append(
                    {
                        "chunk_id": chunk_id,
                        "source": source.get("source", "unbekannt"),
                    }
                )
        except NotFoundError:
            # Echte Datenintegritaets-Luecke: Chunk in Qdrant, fehlt in ES
            missing += 1
            sampled += 1
        except Exception as exc:
            # Auth/Timeout/Connection-Fehler etc. — keine Aussage zur Integritaet,
            # daher separat als errors zaehlen und NICHT in sampled aufnehmen
            logger.warning(
                "ES verify_integrity Sample-Fetch fehlgeschlagen fuer %s: %s",
                chunk_id,
                exc,
                exc_info=True,
            )
            errors += 1

    result["sampled"] = sampled
    result["matching"] = matching
    result["missing_in_es"] = missing
    result["errors"] = errors
    result["mismatches"] = mismatches[:20]
    result["integrity_pct"] = round(matching / sampled * 100, 1) if sampled > 0 else 0.0

    return result
