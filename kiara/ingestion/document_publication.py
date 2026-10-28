"""Durable Freigabe einer ganzen Quelle, gemeinsam mit ihrem Dateiabschluss."""

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from uuid import uuid4

from kiara.exceptions import RetrievalError
from kiara.shared.db import db_query, db_transaction

_active_publication = ContextVar("document_publication", default=None)


def pending_source_cleanups(
    collection: str, sources: list[str] | None = None
) -> dict[str, dict[str, list[str]]]:
    """Dauerhaftes ID-Inventar, auch nachdem alle externen Punkte fehlen."""
    with db_query() as conn:
        rows = conn.execute(
            "SELECT source, cleanup_chunks FROM document_publications "
            "WHERE collection_name = %s AND cleanup_chunks <> '{}'::jsonb"
            + (" AND source = ANY(%s)" if sources is not None else ""),
            (collection, sources) if sources is not None else (collection,),
        ).fetchall()
    return {row["source"]: row["cleanup_chunks"] for row in rows}


def pending_cleanup_sources(collection: str) -> list[str]:
    """Job discovery needs paths only, never the complete chunk inventories."""
    with db_query() as conn:
        return [
            row["source"]
            for row in conn.execute(
                "SELECT source FROM document_publications WHERE collection_name = %s "
                "AND cleanup_chunks <> '{}'::jsonb",
                (collection,),
            ).fetchall()
        ]


def pending_cleanup_models(collection: str, sources: list[str]) -> set[str]:
    """Read model keys for locking without materializing any saved chunk IDs."""
    with db_query() as conn:
        return {
            row["model"]
            for row in conn.execute(
                "SELECT DISTINCT jsonb_object_keys(cleanup_chunks) AS model "
                "FROM document_publications WHERE collection_name = %s AND source = ANY(%s)",
                (collection, sources),
            ).fetchall()
        }


def can_replay_unpublished_datasource(store, chunk_ds, datasource_id, model, sources):
    """Allow an empty target only if *all* its memberships belong to pending replay.

    Completed file hashes disqualify the target. Count its PG memberships once
    per run and compare source-local Qdrant/PG evidence; never load a DS corpus.
    This is permission to reach canonical publication, not a completion proof.
    """
    pending = pending_document_sources(str(store.get_collection_name()), sources)
    if not pending:
        return False
    with db_query() as conn:
        if conn.execute(
            "SELECT 1 FROM file_hashes WHERE datasource_id = %s AND embedding_model = %s LIMIT 1",
            (datasource_id, model),
        ).fetchone():
            return False
    expected = chunk_ds.count_by_datasource(datasource_id, model)
    if not expected:
        return False
    confirmed = 0
    for source in pending:
        ids = store.get_ids_by_sources([source])
        owners = chunk_ds.get_datasource_ids_for_chunks(ids, model)
        confirmed += sum(datasource_id in ds_ids for ds_ids in owners.values())
    return confirmed == expected


def replayable_memberships(memberships):
    """Only the locked document's pre-existing pending publication permits replay."""
    publication = _active_publication.get()
    if publication is None or not publication.was_pending:
        return set()
    return {
        (ds, cid)
        for cid, owners in memberships.items()
        for ds in owners
        if ds not in publication.completed_owners
    }


def remember_source_cleanup(collection: str, source: str, chunks: dict[str, list[str]]) -> None:
    """Vor externen Deletes sichern; ein neuer Versuch darf alte IDs nicht verlieren."""
    with db_transaction() as conn:
        row = conn.execute(
            "SELECT cleanup_chunks FROM document_publications "
            "WHERE collection_name = %s AND source = %s FOR UPDATE",
            (collection, source),
        ).fetchone()
        if row is None:
            raise RuntimeError("Quellen-Cleanup ohne Veroeffentlichungssperre")
        saved = row["cleanup_chunks"]
        for model, ids in chunks.items():
            saved[model] = sorted(set(saved.get(model, [])) | set(ids))
        conn.execute(
            "UPDATE document_publications SET cleanup_chunks = %s::jsonb "
            "WHERE collection_name = %s AND source = %s",
            (json.dumps(saved), collection, source),
        )


def finish_source_cleanup(collection: str, sources: list[str]) -> None:
    """Erst nach geprueften Deletes unter den bestehenden Indexsperren aufrufen."""
    with db_transaction() as conn:
        conn.execute(
            "UPDATE document_publications SET cleanup_chunks = '{}'::jsonb "
            "WHERE collection_name = %s AND source = ANY(%s)",
            (collection, sources),
        )


def finish_colbert_model_cleanup(conn, model: str) -> None:
    """Nach bestaetigtem Collection-Delete im selben Index-Guard quittieren.

    Der Aufrufer entfernt auch den Indexvertrag in dieser Transaktion.
    Primaere IDs bleiben als Wiederanlaufanker ihrer Quellen erhalten.
    """
    if conn.execute(
        "SELECT 1 FROM chunk_datasources WHERE embedding_model = %s LIMIT 1", (model,)
    ).fetchone():
        raise RuntimeError("ColBERT-Modell-Cleanup hat noch PG-Zuordnungen")
    conn.execute(
        "UPDATE document_publications SET cleanup_chunks = cleanup_chunks - %s "
        "WHERE cleanup_chunks ? %s",
        (model, model),
    )


def pending_document_sources(collection: str, sources: list[str]) -> set[str]:
    """Offene Quellen muessen auch bei erhaltenem Datei-Hash erneut verarbeitet werden."""
    if not sources:
        return set()
    with db_query() as conn:
        rows = conn.execute(
            "SELECT source FROM document_publications WHERE collection_name = %s "
            "AND source = ANY(%s) AND completed_at IS NULL",
            (collection, sources),
        ).fetchall()
    return {row["source"] for row in rows}


def invalidate_documents(
    collection: str,
    sources: list[str],
    model: str | None,
    *,
    hash_datasource_id: int | None = None,
    cleanup: bool = False,
    cleanup_es: bool = False,
) -> str:
    """Quelle dauerhaft sperren und betroffene Dateiabschluesse entwerten.

    Ein Verarbeitungsversuch entwertet nur seinen eigenen DS-Abschluss.
    Quellenweites Cleanup laesst den Filter weg und entwertet alle Zuordnungen.
    Die Lesesperre gilt in beiden Faellen fuer die ganze Quelle.
    """
    if cleanup and not model:
        raise ValueError("Quellen-Cleanup braucht ein Primaermodell")
    generation = uuid4().hex
    with db_transaction() as conn:
        for source in sorted(set(sources)):
            conn.execute(
                "INSERT INTO document_publications "
                "(collection_name, source, generation) VALUES (%s, %s, %s) "
                "ON CONFLICT (collection_name, source) DO UPDATE SET "
                "generation = EXCLUDED.generation, versions = '{}'::jsonb, "
                "started_at = now(), completed_at = NULL",
                (collection, source, generation),
            )
        if cleanup:
            # Den Quellenauftrag zusammen mit der Hashentwertung sichern,
            # bevor schon die erste Inventarabfrage fehlschlagen kann. Eine
            # leere ID-Liste bleibt offen; bereits gesicherte IDs haben Vorrang.
            conn.execute(
                "UPDATE document_publications SET cleanup_chunks = %s::jsonb || cleanup_chunks "
                "WHERE collection_name = %s AND source = ANY(%s)",
                (json.dumps({model: []}), collection, sources),
            )
        if cleanup_es:
            # Preserve source owners atomically with hash invalidation. Needed
            # even if both vector stores are lost before the first inventory read.
            key = f"es::{model}::owners"
            for source in sources:
                owners = conn.execute(
                    "SELECT datasource_id FROM file_hashes WHERE embedding_model = %s AND file_path = %s",
                    (model, source),
                ).fetchall()
                saved = conn.execute(
                    "SELECT cleanup_chunks FROM document_publications "
                    "WHERE collection_name = %s AND source = %s FOR UPDATE",
                    (collection, source),
                ).fetchone()["cleanup_chunks"]
                saved[key] = sorted(
                    set(saved.get(key, [])) | {str(row["datasource_id"]) for row in owners}
                )
                conn.execute(
                    "UPDATE document_publications SET cleanup_chunks = %s::jsonb "
                    "WHERE collection_name = %s AND source = %s",
                    (json.dumps(saved), collection, source),
                )
        delete_hashes = "DELETE FROM file_hashes WHERE embedding_model = %s AND file_path = ANY(%s)"
        params = [model, sources]
        if hash_datasource_id is not None:
            delete_hashes += " AND datasource_id = %s"
            params.append(hash_datasource_id)
        removed = conn.execute(delete_hashes + " RETURNING datasource_id", params).fetchall()
        if removed:
            from kiara.ingestion.vectordb_ops import _recount_datasource_embeddings

            for ds_id in sorted({row["datasource_id"] for row in removed}):
                _recount_datasource_embeddings(conn, ds_id, model)
    return generation


@dataclass
class _Publication:
    collection: str
    source: str
    model: str | None
    datasource_id: int
    was_pending: bool = False
    completed_owners: set[int] = field(default_factory=set)
    generation: str | None = None
    versions: dict = field(default_factory=dict)

    def begin(self) -> None:
        if self.generation is None:
            self.generation = invalidate_documents(
                self.collection,
                [self.source],
                self.model,
                hash_datasource_id=self.datasource_id,
            )


@contextmanager
def document_publication(
    collection: str, source: str, model: str | None, *, datasource_id: int | None = None
):
    """Serialisiert Writer derselben Quelle; andere Dokumente bleiben unabhaengig.

    Der ContextVar verbindet den Runtime-Abschluss mit _complete_file, dessen
    Transaktion auch File-Hash und Zaehler schreibt. Die Sperrtransaktion selbst
    enthaelt keine Freigabe: der offene Zustand wird separat dauerhaft committet.
    """
    publication = _Publication(
        collection, source, model, 0 if datasource_id is None else int(datasource_id)
    )
    with db_transaction() as lock_conn:
        lock_conn.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%s, 681032))",
            (json.dumps([collection, source]),),
        )
        # Capture evidence before begin() invalidates this attempt's hash.
        publication.was_pending = bool(pending_document_sources(collection, [source]))
        with db_query() as conn:
            publication.completed_owners = {
                row["datasource_id"]
                for row in conn.execute(
                    "SELECT datasource_id FROM file_hashes WHERE embedding_model = %s AND file_path = %s",
                    (model, source),
                ).fetchall()
            }
        token = _active_publication.set(publication)
        try:
            yield publication
        finally:
            _active_publication.reset(token)


def stage_document_versions(versions: dict) -> None:
    publication = _active_publication.get()
    if publication is not None:
        publication.versions = versions


def complete_document_publication(
    conn, source: str, model: str | None, *, datasource_id: int | None
) -> None:
    """Nur die gemeinsame PostgreSQL-Transaktion darf die Quelle freigeben."""
    publication = _active_publication.get()
    if publication is None:
        return
    if (
        source != publication.source
        or model != publication.model
        or (0 if datasource_id is None else int(datasource_id)) != publication.datasource_id
        or publication.generation is None
        or set(publication.versions) != {source}
        or not publication.versions[source]
    ):
        raise RuntimeError("Dokumentabschluss ohne bestaetigte Gesamtversion")
    changed = conn.execute(
        "UPDATE document_publications SET versions = %s::jsonb, completed_at = now() "
        "WHERE collection_name = %s AND source = %s AND generation = %s "
        "AND completed_at IS NULL AND cleanup_chunks = '{}'::jsonb RETURNING source",
        (
            json.dumps(publication.versions[source]),
            publication.collection,
            source,
            publication.generation,
        ),
    ).fetchone()
    if changed is None:
        raise RuntimeError("Dokumentabschluss wurde durch einen anderen Schreibvorgang entwertet")


def assert_documents_published(collection: str, chunks: list[dict]) -> None:
    """Prueft den aktuellen dokumentweiten Zustand, auch fuer alte Suchindex-Kopien."""
    if any((chunk.get("metadata") or {}).get("processing_pending") for chunk in chunks):
        raise RetrievalError("Dokumentverarbeitung ist noch nicht abgeschlossen.")
    sources = list({(c.get("metadata") or {}).get("source") for c in chunks} - {None, ""})
    if not collection or not sources:
        return
    with db_query() as conn:
        rows = conn.execute(
            "SELECT source, completed_at, versions FROM document_publications "
            "WHERE collection_name = %s AND source = ANY(%s)",
            (collection, sources),
        ).fetchall()
    states = {row["source"]: row for row in rows}
    for chunk in chunks:
        meta = chunk.get("metadata") or {}
        state = states.get(meta.get("source"))
        if state is None:
            # Fuer diese Quelle wurde kein Verarbeitungsvorgang registriert.
            continue
        if state["completed_at"] is None:
            raise RetrievalError("Dokumentverarbeitung ist noch nicht abgeschlossen.")
        version = state["versions"].get(meta.get("parser") or "standard")
        if not version or version != meta.get("processing_version"):
            raise RetrievalError(
                "Dokumentversion wurde veraendert. Bitte eine neue Anfrage starten."
            )
