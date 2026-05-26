"""Qdrant-Backend fuer VectorStore."""

import logging
from collections.abc import Iterator

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchText,
    MatchValue,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from kiara.exceptions import VectorDBConnectionError
from kiara.retrieval.embedding import (
    collection_name_for_model,
    embed_query,
    embed_texts,
    model_name_from_collection,
)
from kiara.retrieval.vectorstore_base import VectorStoreBase
from kiara.shared.constants import (
    DEFAULT_QDRANT_URL,
    QDRANT_DELETE_BATCH_SIZE,
    QDRANT_SCROLL_BATCH_SIZE,
    QDRANT_UPSERT_BATCH_SIZE,
)


def _normalize_point_id(point_id) -> str:
    """Qdrant-Punkt-ID in das gespeicherte Hex-Format zurueckverwandeln.

    Qdrant erkennt 32-Zeichen-Hex-Strings als UUIDs und gibt sie mit
    Dashes zurueck (z.B. '7b935a45-3fdf-4e3c-fa96-eac14e55c277').
    Die chunk_id in PostgreSQL (chunk_datasources etc.) speichert aber
    das urspruengliche dashless Hex-Format ('7b935a453fdf4e3cfa96eac14e55c277').
    """
    return str(point_id).replace("-", "")


from kiara.shared.text_utils import sanitize_utf8

logger = logging.getLogger(__name__)


def _parse_points_to_results(points) -> list[dict]:
    """Qdrant ScoredPoints in einheitliches Ergebnis-Format wandeln."""
    results = []
    for point in points:
        payload = dict(point.payload or {})
        text = sanitize_utf8(payload.pop("_text", ""))
        results.append(
            {
                "text": text,
                "metadata": payload,
                "score": point.score,
                "_doc_id": _normalize_point_id(point.id),
            }
        )
    return results


def build_qdrant_filter(where: dict | None) -> Filter | None:
    """Baut Qdrant-Filter aus abstraktem where-Dict.

    Unterstuetzt: $eq, $in, $contains, $gte, $lte, $and.
    Wird von QdrantVectorStore und ColBERTStore gemeinsam genutzt.
    """
    if where is None:
        return None

    if "$and" in where:
        conditions = [build_qdrant_filter(w) for w in where["$and"]]
        must = []
        for c in conditions:
            if isinstance(c, Filter):
                must.extend(c.must or [])
            elif c is not None:
                must.append(c)
        return Filter(must=must)

    conditions = []
    for key, value in where.items():
        if isinstance(value, dict):
            if "$eq" in value:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value["$eq"])))
            elif "$in" in value:
                conditions.append(FieldCondition(key=key, match=MatchAny(any=value["$in"])))
            elif "$contains" in value:
                conditions.append(FieldCondition(key=key, match=MatchText(text=value["$contains"])))
            elif "$gte" in value or "$lte" in value:
                conditions.append(
                    FieldCondition(
                        key=key,
                        range=DatetimeRange(
                            gte=value.get("$gte"),
                            lte=value.get("$lte"),
                        ),
                    )
                )
            else:
                raise ValueError(
                    f"Unbekannter Filter-Operator fuer Feld '{key}': {sorted(value.keys())!r}. "
                    "Unterstuetzt: $eq, $in, $contains, $gte, $lte."
                )
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
    return Filter(must=conditions)


class QdrantVectorStore(VectorStoreBase):
    """Qdrant-Backend (Remote-Server via HTTP/gRPC)."""

    def __init__(self, config: dict, embedding_model: str | None = None) -> None:
        super().__init__(config, embedding_model)
        qdrant_cfg = config.get("vectordb", {})
        self._client: QdrantClient = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=qdrant_cfg.get("timeout", 60),
            prefer_grpc=bool(qdrant_cfg.get("qdrant_grpc")),
        )
        self._collection: str = collection_name_for_model(self._embed_model)
        from kiara.retrieval.embedding import get_embedding_dimension

        self._vector_size: int = get_embedding_dimension(self._embed_model, config)
        self._ensure_collection()
        self._use_source_key = self._detect_source_key_ready()

    # Payload-Felder, fuer die Qdrant einen Index anlegen soll.
    # Ohne Index muss jeder Filter-Query alle Punkte scannen.
    _PAYLOAD_INDEXES = {
        "source": PayloadSchemaType.TEXT,  # Substring-Suche (MatchText benoetigt TEXT-Index)
        "source_key": PayloadSchemaType.KEYWORD,  # Exakter Match (MatchValue, O(1))
        "directory": PayloadSchemaType.TEXT,
        "file_type": PayloadSchemaType.KEYWORD,
        "ds_ids": PayloadSchemaType.INTEGER,
        "modified": PayloadSchemaType.DATETIME,
    }

    def _ensure_collection(self) -> None:
        """Collection und Payload-Indexes erstellen falls noetig."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection not in collections:
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
            except Exception as exc:
                # TOCTOU: paralleler Service-Start (kiara/kiara-admin/kiara-worker)
                # kann die Collection zwischen get_collections() und create_collection()
                # angelegt haben. Re-Check, fail-closed nur wenn wirklich fehlt.
                if not self._client.collection_exists(self._collection):
                    raise
                logger.info(
                    "Collection '%s' wurde parallel angelegt — Race ignoriert (%s)",
                    self._collection,
                    exc.__class__.__name__,
                )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Legt Payload-Indexes an, aktualisiert Index-Typen, validiert Vektor-Schema."""
        try:
            info = self._client.get_collection(self._collection)

            # Vektor-Schema-Validierung: falsche Dimension, abweichende Distance-
            # Metrik oder Named-Vector-Layouts brechen Retrieval-Integritaet
            # (DOT/EUCLID liefert falsche Scores fuer Cosinus-trainierte
            # Embeddings, Named-Vectors verursachen Upsert/Query-Fehler).
            # Nicht-leere Collections fail-closed; leere Collections werden
            # mit dem korrekten Schema neu erzeugt.
            vectors_cfg = info.config.params.vectors
            mismatch_reason: str | None = None
            if not isinstance(vectors_cfg, VectorParams):
                mismatch_reason = (
                    f"Named-Vector-Layout (Typ {type(vectors_cfg).__name__}) — "
                    "KIara erzeugt ausschliesslich unbenannte VectorParams"
                )
            elif vectors_cfg.size != self._vector_size:
                mismatch_reason = (
                    f"Dimension {vectors_cfg.size} (gespeichert) != "
                    f"{self._vector_size} (Modell '{self._embed_model}')"
                )
            elif vectors_cfg.distance != Distance.COSINE:
                mismatch_reason = (
                    f"Distance {vectors_cfg.distance} (gespeichert) != "
                    f"{Distance.COSINE} (KIara erzeugt Collections immer mit Cosinus-Distanz)"
                )

            if mismatch_reason is not None:
                message = (
                    f"Collection '{self._collection}' hat inkompatibles Vektor-Schema: "
                    f"{mismatch_reason}. Collection loeschen und neu indexieren um das zu beheben."
                )
                if (info.points_count or 0) > 0:
                    raise VectorDBConnectionError(message)
                logger.warning("%s Leere Collection wird neu erstellt.", message)
                self._client.delete_collection(collection_name=self._collection)
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                schema: dict = {}
            else:
                schema = info.payload_schema or {}

            for field, schema_type in self._PAYLOAD_INDEXES.items():
                if field not in schema:
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=schema_type,
                    )
                else:
                    # Typ pruefen — bei Mismatch Index neu erstellen
                    existing_type = getattr(schema[field], "data_type", None)
                    expected_name = (
                        schema_type.value if hasattr(schema_type, "value") else str(schema_type)
                    )
                    current_name = (
                        existing_type.value
                        if hasattr(existing_type, "value")
                        else str(existing_type)
                    )
                    if current_name != expected_name:
                        self._client.delete_payload_index(
                            collection_name=self._collection,
                            field_name=field,
                        )
                        self._client.create_payload_index(
                            collection_name=self._collection,
                            field_name=field,
                            field_schema=schema_type,
                        )
        except VectorDBConnectionError:
            raise
        except Exception as exc:
            logger.warning(
                "Payload-Index-Erstellung fehlgeschlagen (Filter-Queries fallen auf Full-Scan zurueck): %s",
                exc,
                exc_info=True,
            )

    def add(self, texts: list[str], metadatas: list[dict], ids: list[str]) -> None:
        """Dokumente mit Embeddings zur Collection hinzufuegen (Upsert).

        Berechnet Embeddings via Ollama und speichert den Text als
        _text-Feld im Qdrant-Payload. Upsert in Batches a 256 Punkte.
        Retry bei transienten Fehlern (Timeout, Connection-Reset).
        """
        embeddings = embed_texts(self._config, self._embed_model, texts)
        lengths = [len(texts), len(metadatas), len(embeddings), len(ids)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"QdrantVectorStore.add() Laengenabweichung: "
                f"texts={lengths[0]}, metadatas={lengths[1]}, "
                f"embeddings={lengths[2]}, ids={lengths[3]}"
            )
        for idx, meta in enumerate(metadatas):
            if "source" not in meta:
                raise ValueError(
                    f"QdrantVectorStore.add() metadata[{idx}] ohne Pflichtfeld 'source' "
                    f"(doc_id={ids[idx]}); Loader/Chunker muss 'source' garantieren."
                )
        import time

        points: list[PointStruct] = []
        for text, meta, emb, doc_id in zip(texts, metadatas, embeddings, ids, strict=False):
            payload = {**meta, "_text": text, "source_key": meta["source"]}
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=emb,
                    payload=payload,
                )
            )
        batch_size = QDRANT_UPSERT_BATCH_SIZE
        max_retries = 3
        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self._client.upsert(
                        collection_name=self._collection,
                        points=batch_points,
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise VectorDBConnectionError(
                            f"Qdrant-Upsert nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                        ) from exc
                    time.sleep(2**attempt)

    def query(self, query_text: str, k: int = 5, where: dict | None = None) -> list[dict]:
        """Vektorsuche. Gibt Liste von {text, metadata, score} zurueck.

        Args:
            query_text: Suchanfrage als Freitext.
            k: Anzahl der gewuenschten Ergebnisse.
            where: Optionaler Metadaten-Filter (where-Filter-Format).

        Retry bei transienten Fehlern (Timeout, Connection-Reset) analog
        zu add()/delete_by_sources(), damit der Read-Hauptpfad nicht
        fragiler ist als die Schreibpfade.
        """
        import time

        query_vector = embed_query(self._config, self._embed_model, query_text)
        qdrant_filter = build_qdrant_filter(where)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = self._client.query_points(
                    collection_name=self._collection,
                    query=query_vector,
                    limit=k,
                    query_filter=qdrant_filter,
                    with_payload=True,
                )
                return _parse_points_to_results(results.points)
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise VectorDBConnectionError(
                        f"Qdrant-Query nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                    ) from exc
                time.sleep(2**attempt)

    @staticmethod
    def _build_filter(where: dict) -> Filter:
        """Kompatibilitaets-Alias fuer build_qdrant_filter()."""
        result = build_qdrant_filter(where)
        return result if result is not None else Filter(must=[])

    def _detect_source_key_ready(self) -> bool:
        """Prueft ob source_key-Migration abgeschlossen ist.

        Vergleicht PayloadIndexInfo.points fuer source_key mit der
        Gesamt-Punktzahl der Collection. Nur wenn ALLE Punkte source_key
        haben, wird der schnelle KEYWORD-Index genutzt.
        """
        import logging

        log = logging.getLogger("kiara.vectorstore")
        try:
            info = self._client.get_collection(self._collection)
            schema = info.payload_schema or {}
            sk_info = schema.get("source_key")
            if sk_info is None:
                log.info("source_key-Index nicht vorhanden — nutze source (Full Scan)")
                return False
            total = info.points_count or 0
            if total == 0:
                log.info("Collection leer — source_key aktiv")
                return True
            indexed = getattr(sk_info, "points", 0)
            ready = indexed >= total
            if ready:
                log.info(
                    "source_key-Migration abgeschlossen (%d/%d) — KEYWORD-Index aktiv",
                    indexed,
                    total,
                )
            else:
                log.info(
                    "source_key-Migration unvollstaendig (%d/%d) — nutze source (Full Scan). "
                    "Migration starten: kiara migrate-source-key",
                    indexed,
                    total,
                )
            return ready
        except Exception as exc:
            log.warning(
                "source_key-Detection fehlgeschlagen, nutze Fallback: %s",
                exc,
                exc_info=True,
            )
            return False

    def _source_match_filter(self, source: str) -> Filter:
        """FieldCondition fuer exakten Source-Match.

        Nutzt source_key (KEYWORD, O(1)) wenn Migration abgeschlossen,
        sonst source (TEXT, Full Scan aber korrekt).
        """
        key = "source_key" if self._use_source_key else "source"
        return Filter(must=[FieldCondition(key=key, match=MatchValue(value=source))])

    def search(
        self, query_text: str, k: int = 10, offset: int = 0, where: dict | None = None
    ) -> list[dict]:
        """Vektorsuche mit offset-Support fuer Pagination."""
        query_vector = embed_query(self._config, self._embed_model, query_text)
        qdrant_filter = build_qdrant_filter(where)
        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=k,
            offset=offset,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return _parse_points_to_results(results.points)

    def count_filtered(self, where: dict | None = None) -> int:
        """Anzahl Chunks mit optionalem Filter."""
        qdrant_filter = build_qdrant_filter(where)
        result = self._client.count(
            collection_name=self._collection,
            count_filter=qdrant_filter,
            exact=True,
        )
        return result.count

    def update_ds_ids(self, ds_ids_map: dict[str, list[int]], *, wait: bool = False) -> None:
        """Aktualisiert ds_ids-Array fuer die gegebenen Chunks (batched).

        Gruppiert Chunks mit identischen ds_ids-Listen fuer effiziente
        Batch-Updates via set_payload (ein API-Call pro Gruppe+Batch).
        Retry bei transienten Fehlern (Timeout, Connection-Reset).
        """
        if not ds_ids_map:
            return
        import time
        from collections import defaultdict

        groups: dict[tuple, list[str]] = defaultdict(list)
        for chunk_id, ds_ids in ds_ids_map.items():
            groups[tuple(sorted(ds_ids))].append(chunk_id)
        batch_size = QDRANT_DELETE_BATCH_SIZE
        max_retries = 3
        for ds_ids_tuple, chunk_ids in groups.items():
            for i in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[i : i + batch_size]
                for attempt in range(max_retries):
                    try:
                        self._client.set_payload(
                            collection_name=self._collection,
                            payload={"ds_ids": list(ds_ids_tuple)},
                            points=batch,
                            wait=wait,
                        )
                        break
                    except Exception as exc:
                        if attempt == max_retries - 1:
                            raise VectorDBConnectionError(
                                f"Qdrant set_payload nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                            ) from exc
                        time.sleep(2**attempt)

    def delete_by_sources(self, sources: list[str]) -> None:
        """Alle Chunks der angegebenen Quelldateien loeschen.

        Bei aktivem source_key-KEYWORD-Index alle Sources in einem
        MatchAny-Filter buendeln (1 Roundtrip statt N). Sonst pro Source
        iterieren. Retry bei transienten Fehlern; einzelne Source-Fehler
        brechen die Schleife nicht ab — am Ende sammelt eine
        VectorDBConnectionError alle fehlgeschlagenen Sources.
        """
        if not sources:
            return
        import time

        max_retries = 3
        if self._use_source_key:
            match_filter = Filter(
                must=[FieldCondition(key="source_key", match=MatchAny(any=list(sources)))]
            )
            for attempt in range(max_retries):
                try:
                    self._client.delete(
                        collection_name=self._collection,
                        points_selector=match_filter,
                    )
                    return
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise VectorDBConnectionError(
                            f"Qdrant delete_by_sources (MatchAny) nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                        ) from exc
                    time.sleep(2**attempt)
            return
        failed: list[tuple[str, Exception]] = []
        for source in sources:
            for attempt in range(max_retries):
                try:
                    self._client.delete(
                        collection_name=self._collection,
                        points_selector=self._source_match_filter(source),
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        failed.append((source, exc))
                        break
                    time.sleep(2**attempt)
        if failed:
            sources_failed = ", ".join(s for s, _ in failed)
            raise VectorDBConnectionError(
                f"Qdrant delete_by_sources fuer {len(failed)} Quellen fehlgeschlagen: {sources_failed}"
            )

    def delete_by_ids(self, ids: list[str]) -> None:
        """Chunks anhand ihrer IDs loeschen.

        Retry mit Backoff bei transienten Qdrant-Fehlern, analog zu
        delete_by_sources. Verhindert Drift zwischen Vektor- und
        PostgreSQL-Schichten im 4-Schichten-Datenmodell.
        """
        if not ids:
            return
        import time

        max_retries = 3
        batch_size = QDRANT_DELETE_BATCH_SIZE
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self._client.delete(
                        collection_name=self._collection,
                        points_selector=PointIdsList(points=batch),
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise VectorDBConnectionError(
                            f"Qdrant delete_by_ids (Batch {i}..{i + len(batch)}) nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                        ) from exc
                    time.sleep(2**attempt)

    def _scroll_points(
        self,
        scroll_filter=None,
        with_payload=True,
        with_vectors=False,
        batch_size=QDRANT_SCROLL_BATCH_SIZE,
    ):
        """Generator der alle Punkte per Scroll-API liefert."""
        offset = None
        first = True
        prev_offset = object()
        while first or offset is not None:
            first = False
            points, offset = self._client.scroll(
                collection_name=self._collection,
                scroll_filter=scroll_filter,
                with_payload=with_payload,
                with_vectors=with_vectors,
                limit=batch_size,
                offset=offset,
            )
            if not points:
                return
            if offset is not None and offset == prev_offset:
                logger.error(
                    "Qdrant-Scroll lieferte unveraenderten Cursor (%r); breche ab um Endlosschleife zu vermeiden",
                    offset,
                )
                yield from points
                return
            prev_offset = offset
            yield from points

    def get_ids_by_sources(self, sources: list[str], *, ds_id: int | None = None) -> list[str]:
        """Chunk-IDs fuer gegebene Quelldateien zurueckgeben."""
        result_ids = []
        for source in sources:
            src_filter = self._source_match_filter(source)
            if ds_id is not None:
                filt = Filter(
                    must=[
                        src_filter.must[0],
                        FieldCondition(key="ds_ids", match=MatchAny(any=[ds_id])),
                    ]
                )
            else:
                filt = src_filter
            result_ids.extend(
                _normalize_point_id(p.id) for p in self._scroll_points(filt, with_payload=False)
            )
        return result_ids

    def list_sources_for_datasource(
        self,
        ds_id: int,
        batch_size: int = QDRANT_SCROLL_BATCH_SIZE,
    ) -> list[dict]:
        """Scrollt alle Chunks einer DS und gruppiert nach source."""
        scroll_filter = Filter(must=[FieldCondition(key="ds_ids", match=MatchAny(any=[ds_id]))])
        by_source: dict[str, dict] = {}
        for p in self._scroll_points(
            scroll_filter, with_payload=["source", "file_type", "modified"], batch_size=batch_size
        ):
            payload = p.payload or {}
            src = payload.get("source", "")
            if src not in by_source:
                by_source[src] = {
                    "source": src,
                    "file_type": payload.get("file_type", ""),
                    "chunk_count": 0,
                    "modified": payload.get("modified", ""),
                }
            by_source[src]["chunk_count"] += 1
        return list(by_source.values())

    def count_total(self) -> int:
        """Gesamtanzahl aller Chunks in der Collection."""
        info = self._client.get_collection(self._collection)
        return info.points_count or 0

    def query_by_ids(
        self, query_text: str, ids: list[str], k: int = 5, where: dict | None = None
    ) -> list[dict]:
        """ID-basierte ANN-Suche mit HasIdCondition-Filter.

        Beschraenkt die Vektorsuche auf eine vordefinierte Menge von
        Chunk-IDs (z.B. aus Source-Filter oder Vorauswahl). Kombiniert
        HasIdCondition mit optionalen where-Filtern.

        Args:
            query_text: Text-Query zum Embedden.
            ids: Whitelist von Chunk-IDs fuer die Filterung.
            k: Maximale Ergebnisse.
            where: Zusaetzliche Metadaten-Filter.

        Returns:
            Liste von Dicts mit 'text', 'metadata', 'score'.
        """
        import time

        from qdrant_client.models import HasIdCondition

        query_vector = embed_query(self._config, self._embed_model, query_text)

        conditions = [HasIdCondition(has_id=ids)]
        if where:
            qdrant_where = build_qdrant_filter(where)
            if qdrant_where and qdrant_where.must:
                conditions.extend(qdrant_where.must)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                results = self._client.query_points(
                    collection_name=self._collection,
                    query=query_vector,
                    limit=k,
                    query_filter=Filter(must=conditions),
                    with_payload=True,
                )
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise VectorDBConnectionError(
                        f"Qdrant query_by_ids nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                    ) from exc
                time.sleep(2**attempt)
        return _parse_points_to_results(results.points)

    def scroll_all(
        self, batch_size: int = QDRANT_SCROLL_BATCH_SIZE
    ) -> Iterator[list[tuple[str, dict]]]:
        """Iteriert ueber alle Chunks. Gibt (chunk_id, metadata) Tupel in Batches zurueck."""
        for batch in self.scroll_all_with_text(batch_size):
            yield [(cid, meta) for cid, _text, meta in batch]

    def scroll_all_with_text(
        self, batch_size: int = QDRANT_SCROLL_BATCH_SIZE
    ) -> Iterator[list[tuple[str, str, dict]]]:
        """Iteriert ueber alle Chunks inkl. Text. Gibt (chunk_id, text, metadata) Tupel in Batches zurueck."""
        offset = None
        first = True
        prev_offset = object()
        while first or offset is not None:
            first = False
            points, offset = self._client.scroll(
                self._collection,
                limit=batch_size,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            if not points:
                break
            batch = []
            for p in points:
                payload = dict(p.payload or {})
                text = payload.pop("_text", "")
                batch.append((_normalize_point_id(p.id), text, payload))
            yield batch
            if offset is not None and offset == prev_offset:
                logger.error(
                    "Qdrant-Scroll lieferte unveraenderten Cursor (%r); breche ab um Endlosschleife zu vermeiden",
                    offset,
                )
                return
            prev_offset = offset

    def remove_payload_field(self, field_name: str) -> None:
        """Entfernt ein Feld aus allen Chunk-Payloads + zugehoerigen Index."""
        batch: list = []
        for p in self._scroll_points(with_payload=[field_name]):
            if (p.payload or {}).get(field_name) is not None:
                batch.append(p.id)
            if len(batch) >= QDRANT_SCROLL_BATCH_SIZE:
                self._client.delete_payload(
                    collection_name=self._collection,
                    keys=[field_name],
                    points=batch,
                )
                batch = []
        if batch:
            self._client.delete_payload(
                collection_name=self._collection,
                keys=[field_name],
                points=batch,
            )

        # Index loeschen (best-effort)
        try:
            self._client.delete_payload_index(
                collection_name=self._collection,
                field_name=field_name,
            )
        except Exception as exc:
            logger.debug(
                "Payload-Index '%s' konnte nicht entfernt werden (best-effort): %s", field_name, exc
            )

    def add_with_embeddings(
        self, texts: list[str], metadatas: list[dict], ids: list[str], embeddings: list[list[float]]
    ) -> None:
        """Chunks MIT vorberechneten Embeddings hinzufuegen (fuer Migration).

        Gleiche Payload-Struktur wie add() — _text wird im Payload gespeichert.
        Retry bei transienten Fehlern (Timeout, Connection-Reset).
        """
        lengths = [len(texts), len(metadatas), len(ids), len(embeddings)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"QdrantVectorStore.add_with_embeddings() Laengenabweichung: "
                f"texts={lengths[0]}, metadatas={lengths[1]}, "
                f"ids={lengths[2]}, embeddings={lengths[3]}"
            )
        for idx, meta in enumerate(metadatas):
            if "source" not in meta:
                raise ValueError(
                    f"QdrantVectorStore.add_with_embeddings() metadata[{idx}] ohne Pflichtfeld 'source' "
                    f"(doc_id={ids[idx]}); Loader/Chunker muss 'source' garantieren."
                )
        import time

        points: list[PointStruct] = []
        for text, meta, emb, doc_id in zip(texts, metadatas, embeddings, ids, strict=False):
            payload = {**meta, "_text": text, "source_key": meta["source"]}
            points.append(PointStruct(id=doc_id, vector=emb, payload=payload))
        batch_size = QDRANT_UPSERT_BATCH_SIZE
        max_retries = 3
        for i in range(0, len(points), batch_size):
            batch_points = points[i : i + batch_size]
            for attempt in range(max_retries):
                try:
                    self._client.upsert(
                        collection_name=self._collection,
                        points=batch_points,
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise VectorDBConnectionError(
                            f"Qdrant-Upsert (add_with_embeddings) nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                        ) from exc
                    time.sleep(2**attempt)

    def get_documents_by_source(self, source: str) -> dict:
        """Alle Chunks einer Quelldatei aus Qdrant laden."""
        ids, documents, metadatas = [], [], []
        scroll_filter = self._source_match_filter(source)
        for p in self._scroll_points(scroll_filter):
            payload = dict(p.payload or {})
            ids.append(_normalize_point_id(p.id))
            documents.append(sanitize_utf8(payload.pop("_text", "")))
            payload.pop("_text", None)
            metadatas.append(payload)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def migrate_source_key(self, batch_size: int = QDRANT_SCROLL_BATCH_SIZE) -> dict:
        """Bestehende Punkte um source_key-Feld ergaenzen.

        Scrollt ueber alle Punkte, liest source aus Payload und schreibt
        source_key zurueck. Idempotent — kann bei Abbruch neu gestartet werden.

        Returns:
            {"total": int, "migrated": int, "skipped": int, "errors": int}
        """
        from collections import defaultdict

        log = logging.getLogger("kiara.migration")
        stats = {"total": 0, "migrated": 0, "skipped": 0, "errors": 0}
        offset = None
        first = True

        while first or offset is not None:
            first = False
            results, offset = self._client.scroll(
                collection_name=self._collection,
                limit=batch_size,
                offset=offset,
                with_payload=["source", "source_key"],
                with_vectors=False,
            )
            if not results:
                break

            # Punkte sammeln die source_key noch nicht haben oder Wert abweicht
            by_source: dict[str, list] = defaultdict(list)
            for point in results:
                stats["total"] += 1
                payload = point.payload or {}
                source_val = payload.get("source")
                existing_key = payload.get("source_key")
                if source_val is None:
                    stats["skipped"] += 1
                    continue
                if existing_key == source_val:
                    stats["skipped"] += 1
                    continue
                by_source[source_val].append(point.id)

            # Gruppiert nach Source-Wert schreiben (set_payload setzt gleichen
            # Wert fuer alle Punkte im Batch → muss pro Source gruppiert sein)
            for src_val, point_ids in by_source.items():
                try:
                    self._client.set_payload(
                        collection_name=self._collection,
                        payload={"source_key": src_val},
                        points=point_ids,
                        wait=False,
                    )
                    stats["migrated"] += len(point_ids)
                except Exception as exc:
                    log.warning(
                        "set_payload fehlgeschlagen fuer %d Punkte: %s",
                        len(point_ids),
                        exc,
                        exc_info=True,
                    )
                    stats["errors"] += len(point_ids)

            if stats["total"] % 100_000 < batch_size:
                log.info(
                    "Migration Fortschritt: %d verarbeitet, %d migriert, "
                    "%d uebersprungen, %d Fehler",
                    stats["total"],
                    stats["migrated"],
                    stats["skipped"],
                    stats["errors"],
                )

        log.info("Migration abgeschlossen: %s", stats)
        return stats

    def get_collection_name(self) -> str:
        """Name der aktuellen Qdrant-Collection."""
        return self._collection

    def get_all_documents(
        self,
        include: list[str] | None = None,
    ) -> dict:
        """Alle Dokumente aus der Qdrant-Collection via Scroll laden."""
        if include is None:
            include = ["documents", "metadatas"]
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        for p in self._scroll_points():
            payload = dict(p.payload or {})
            ids.append(_normalize_point_id(p.id))
            if "documents" in include:
                documents.append(sanitize_utf8(payload.pop("_text", "")))
            if "metadatas" in include:
                payload.pop("_text", None)
                metadatas.append(payload)
        result: dict = {"ids": ids}
        if "documents" in include:
            result["documents"] = documents
        if "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def get_all_documents_batched(
        self,
        include: list[str] | None = None,
        limit: int = 10_000,
        offset: int = 0,
    ) -> dict:
        """Dokumente seitenweise via Qdrant-Scroll laden.

        Nutzt Cursor-basiertes Scrolling statt der Basisklassen-Implementierung,
        die bei jedem Batch alle Dokumente neu laedt.
        """
        if include is None:
            include = ["documents", "metadatas"]
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        # Offset ueberspringen via Scroll
        cursor = None
        skipped = 0
        if offset > 0:
            while skipped < offset:
                batch_limit = min(QDRANT_SCROLL_BATCH_SIZE, offset - skipped)
                points, cursor = self._client.scroll(
                    self._collection,
                    limit=batch_limit,
                    with_payload=False,
                    offset=cursor,
                )
                if not points:
                    logger.debug(
                        "get_all_documents_batched: offset %d uebersteigt "
                        "Collection-Groesse (skipped=%d, keine weiteren Points)",
                        offset,
                        skipped,
                    )
                    return {"ids": [], "documents": [], "metadatas": []}
                skipped += len(points)
                if cursor is None and skipped < offset:
                    logger.debug(
                        "get_all_documents_batched: offset %d uebersteigt "
                        "Collection-Groesse (skipped=%d, Cursor erschoepft)",
                        offset,
                        skipped,
                    )
                    return {"ids": [], "documents": [], "metadatas": []}

        # Eigentliche Daten laden
        collected = 0
        first = True
        while collected < limit and (first or cursor is not None):
            first = False
            batch_limit = min(QDRANT_SCROLL_BATCH_SIZE, limit - collected)
            points, cursor = self._client.scroll(
                self._collection,
                limit=batch_limit,
                with_payload=True,
                offset=cursor,
            )
            if not points:
                break
            for p in points:
                payload = dict(p.payload or {})
                ids.append(_normalize_point_id(p.id))
                if "documents" in include:
                    documents.append(sanitize_utf8(payload.pop("_text", "")))
                if "metadatas" in include:
                    payload.pop("_text", None)
                    metadatas.append(payload)
            collected += len(points)

        result: dict = {"ids": ids}
        if "documents" in include:
            result["documents"] = documents
        if "metadatas" in include:
            result["metadatas"] = metadatas
        return result

    def get_documents_by_ids(
        self,
        ids: list[str],
        include: list[str] | None = None,
    ) -> dict:
        """Dokumente anhand ihrer IDs aus Qdrant laden.

        Retry bei transienten Fehlern (Timeout, Connection-Reset).
        """
        if include is None:
            include = ["documents", "metadatas"]
        with_vectors = "embeddings" in include
        out_ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []
        embeddings: list = []
        import time

        max_retries = 3
        for start in range(0, len(ids), QDRANT_DELETE_BATCH_SIZE):
            batch = ids[start : start + QDRANT_DELETE_BATCH_SIZE]
            points = None
            for attempt in range(max_retries):
                try:
                    points = self._client.retrieve(
                        collection_name=self._collection,
                        ids=batch,
                        with_payload=True,
                        with_vectors=with_vectors,
                    )
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        raise VectorDBConnectionError(
                            f"Qdrant retrieve (get_documents_by_ids) nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                        ) from exc
                    time.sleep(2**attempt)
            for p in points:
                payload = dict(p.payload or {})
                out_ids.append(_normalize_point_id(p.id))
                if "documents" in include:
                    documents.append(sanitize_utf8(payload.pop("_text", "")))
                if "metadatas" in include:
                    payload.pop("_text", None)
                    metadatas.append(payload)
                if with_vectors:
                    embeddings.append(p.vector)
        result: dict = {"ids": out_ids}
        if "documents" in include:
            result["documents"] = documents
        if "metadatas" in include:
            result["metadatas"] = metadatas
        if with_vectors:
            result["embeddings"] = embeddings
        return result

    def get_stats(self) -> dict:
        """Collection-Statistiken zurueckgeben (Chunks, Backend, Status)."""
        info = self._client.get_collection(self._collection)
        return {
            "collection": self._collection,
            "total_chunks": info.points_count,
            "backend": "qdrant",
            "qdrant_url": self._config.get("vectordb", {}).get("qdrant_url", ""),
            "status": info.status.value,
        }

    def get_directories(self) -> list[str]:
        """Eindeutige Verzeichnisse in der Collection via Scroll."""
        dirs: set[str] = set()
        for p in self._scroll_points(with_payload=["directory"]):
            d = (p.payload or {}).get("directory")
            if d:
                dirs.add(d)
        return sorted(dirs)

    def get_file_types(self) -> list[str]:
        """Eindeutige Dateitypen in der Collection via Scroll."""
        types: set[str] = set()
        for p in self._scroll_points(with_payload=["file_type"]):
            t = (p.payload or {}).get("file_type")
            if t:
                types.add(t)
        return sorted(types)

    @classmethod
    def list_collections(cls, config: dict) -> list[dict]:
        """Alle kb_*-Collections mit Chunk-Anzahl auflisten."""
        qdrant_cfg = config.get("vectordb", {})
        try:
            client = QdrantClient(
                url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
                api_key=qdrant_cfg.get("qdrant_api_key") or None,
                timeout=qdrant_cfg.get("timeout", 60),
                prefer_grpc=bool(qdrant_cfg.get("qdrant_grpc")),
            )
            result: list[dict] = []
            for coll in client.get_collections().collections:
                if coll.name.startswith("kb_"):
                    try:
                        model = model_name_from_collection(coll.name)
                    except Exception as exc:
                        logger.warning(
                            "Collection '%s' uebersprungen: %s",
                            coll.name,
                            exc,
                            exc_info=True,
                        )
                        continue
                    info = client.get_collection(coll.name)
                    result.append(
                        {
                            "name": coll.name,
                            "model": model,
                            "chunks": info.points_count or 0,
                        }
                    )
            return result
        except httpx.TransportError as exc:
            logger.error("Qdrant nicht erreichbar: %s", exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error("Qdrant list_collections fehlgeschlagen: %s", exc, exc_info=True)
            raise
