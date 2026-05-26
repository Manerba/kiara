"""ColBERT Multi-Vector Retrieval via Qdrant.

ColBERT erzeugt Token-Level-Vektoren pro Chunk. Beim Retrieval wird per
MaxSim-Scoring (Maximum Similarity) ueber alle Token-Paare gematcht.
Die Vektoren liegen in separaten Qdrant-Collections mit Multi-Vector-
Konfiguration (``colbert_<modellname>``).

Beziehungen:
    - kiara.retrieval.embedding liefert das Hex-Encoding fuer Collection-Namen
    - kiara.backend_selector liefert BackendUnavailableError und URL-Aufloesung
    - kiara.retrieval.vectorstore_qdrant liefert _normalize_point_id als Vorlage
    - kiara.ingestion.pipeline nutzt ColBERTStore.add() fuer die Indexierung
    - kiara.retrieval.retriever nutzt ColBERTStore.search() fuer das Retrieval
"""

import hashlib
import logging
from contextlib import contextmanager

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    MultiVectorComparator,
    MultiVectorConfig,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from kiara.backend_selector import (
    BackendUnavailableError,
    get_all_active,
    get_embedding_backend_for_model,
    normalize_model_name,
)
from kiara.config import get_embedding_model
from kiara.exceptions import RetrievalError, VectorDBConnectionError
from kiara.retrieval.embedding import _hex_decode, _hex_encode
from kiara.retrieval.vectorstore_qdrant import _normalize_point_id, build_qdrant_filter
from kiara.shared.constants import DEFAULT_QDRANT_URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collection-Naming
# ---------------------------------------------------------------------------


def colbert_collection_name(model: str) -> str:
    """Erzeugt den Qdrant-Collection-Namen fuer ColBERT.

    Format: ``colbert_<hex-encoded_modellname>``
    Verwendet dasselbe _hh-Hex-Encoding wie collection_name_for_model()
    in embedding.py. Nur der Prefix ist anders: ``colbert_`` statt ``kb_``.

    Args:
        model: Normalisierter ColBERT-Modellname (z.B. "colbert-xm").
    """
    normalized = normalize_model_name(model)
    if not normalized:
        raise ValueError("colbert_collection_name erwartet einen nicht-leeren Modellnamen")
    return "colbert_" + _hex_encode(normalized)


def model_name_from_colbert_collection(collection_name: str) -> str:
    """Rekonstruiert den Modellnamen aus einem ColBERT-Collection-Namen.

    Inverse von ``colbert_collection_name()``.
    """
    if not collection_name.startswith("colbert_"):
        raise ValueError(f"Kein ColBERT-Collection-Name: {collection_name!r}")
    return _hex_decode(collection_name[8:], collection_name)


# ---------------------------------------------------------------------------
# DS-Model-Keys fuer chunk_datasources
# ---------------------------------------------------------------------------


def colbert_ds_model_key(model: str) -> str:
    """Eindeutiger ``chunk_datasources.embedding_model``-Key fuer ColBERT.

    Format: ``colbert::<normalized_model>``.
    Dense bleibt bei reinen Modellnamen, ColBERT wird per Prefix unterschieden.
    """
    normalized = normalize_model_name(model)
    if not normalized:
        raise ValueError("colbert_ds_model_key erwartet einen nicht-leeren Modellnamen")
    return f"colbert::{normalized}"


def model_name_from_colbert_ds_key(ds_key: str) -> str:
    """Umkehrfunktion fuer ``colbert_ds_model_key()``.

    Erwartet ``colbert::<model>`` und liefert ``<model>``.
    """
    if not ds_key.startswith("colbert::"):
        raise ValueError(f"Kein ColBERT-DS-Key: {ds_key!r}")
    return normalize_model_name(ds_key[9:])


# ---------------------------------------------------------------------------
# Advisory-Lock-Infrastruktur
# ---------------------------------------------------------------------------


def _stable_lock_id(key: str) -> int:
    """Erzeugt einen stabilen signed 64-bit Integer aus einem String-Key.

    PostgreSQL Advisory-Locks erwarten bigint. Blake2b mit 8 Bytes liefert
    einen 64-bit Hash; Konvertierung zu signed int via struct-artiger Logik.
    """
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    val = int.from_bytes(h, "big")
    # Konvertierung unsigned -> signed 64-bit
    if val >= (1 << 63):
        val -= 1 << 64
    return val


@contextmanager
def colbert_cleanup_lock(ds_model_key: str, *, wait: bool = False):
    """Advisory-Lock pro uebergebenem Key (``colbert::*`` oder ``datasource::*``).

    Yields ``bool`` (lock_acquired). Bei ``wait=False`` + Lock-Konflikt
    wird ``False`` geliefert, der Body laeuft trotzdem (Caller muss pruefen).
    Bei ``wait=True`` blockiert der Aufruf bis der Lock frei ist.

    Haelt dieselbe DB-Connection ueber den gesamten yield-Body, damit
    ``pg_advisory_unlock()`` auf derselben Session laeuft wie der Lock-Erwerb.
    Session-level Advisory-Locks halten ueber Transaction-Commits hinweg.
    """
    from kiara.shared.db import get_db

    lock_id = _stable_lock_id(ds_model_key)
    conn = get_db()
    acquired = False
    try:
        if wait:
            conn.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
            acquired = True
        else:
            row = conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (lock_id,)).fetchone()
            acquired = bool(row and row["ok"])
        conn.commit()
        yield acquired
    finally:
        try:
            if acquired:
                try:
                    conn.rollback()
                except Exception:
                    # Pre-Unlock-Rollback ist defensiv; bereits saubere
                    # Transaktionen sind harmlos.
                    logger.debug(
                        "Pre-Unlock-Rollback fuer %s ohne Effekt",
                        ds_model_key,
                        exc_info=True,
                    )
                conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))
                conn.commit()
        except Exception as exc:
            logger.warning(
                "Advisory-Lock-Unlock fehlgeschlagen fuer %s: %s",
                ds_model_key,
                exc,
                exc_info=True,
            )
        finally:
            conn.close()


@contextmanager
def colbert_datasource_lock(datasource_id: int, *, wait: bool = True):
    """DS-weiter Advisory-Lock, delegiert auf colbert_cleanup_lock.

    Default ``wait=True`` — Writer warten auf laufende Cleanups.
    """
    with colbert_cleanup_lock(f"datasource::{datasource_id}", wait=wait) as acquired:
        yield acquired


# ---------------------------------------------------------------------------
# URL-Resolution
# ---------------------------------------------------------------------------


def _resolve_colbert_url(config: dict) -> str:
    """Ermittelt die KIron-URL fuer ColBERT-Requests.

    Strategie:
    1. ``retrieval.colbert.url`` aus der Config, falls explizit gesetzt.
    2. URL des aktiven Primaer-Embedding-Backends.
    3. Fallback: URL des ERSTEN aktiven Embedding-Backends.

    Raises:
        BackendUnavailableError: Kein Embedding-Backend konfiguriert.
    """
    explicit_url = config.get("retrieval", {}).get("colbert", {}).get("url", "")
    if explicit_url:
        return explicit_url.rstrip("/")

    # Primaer-Backend fuer das aktive Dense-Modell
    try:
        primary_model = get_embedding_model(config)
        if primary_model:
            backend = get_embedding_backend_for_model(primary_model)
            if backend and backend.get("url", "").strip():
                return backend["url"].rstrip("/")
    except (
        BackendUnavailableError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        logger.debug("Primaer-Embedding-Backend fuer ColBERT nicht verwendbar: %s", exc)

    # Fallback: erstes aktives Embedding-Backend mit nicht-leerer URL
    for backend in get_all_active("embedding"):
        if backend.get("url", "").strip():
            return backend["url"].rstrip("/")

    raise BackendUnavailableError(
        "Kein Embedding-Backend fuer ColBERT verfuegbar. "
        "Bitte retrieval.colbert.url konfigurieren oder ein Embedding-Backend aktivieren."
    )


# ---------------------------------------------------------------------------
# Embedding-Funktionen
# ---------------------------------------------------------------------------


def embed_colbert(
    config: dict,
    model: str,
    texts: list[str],
) -> list[list[list[float]]]:
    """Token-Level ColBERT-Embeddings via KIron /api/embed_colbert.

    Nutzt NICHT das ollama SDK, sondern einen manuellen httpx.post() Request.

    Args:
        config: Globale Konfiguration (fuer Backend-URL Fallback).
        model:  ColBERT-Modellname (normalisiert).
        texts:  Liste der Chunk-Texte.

    Returns:
        Pro Text eine Liste von Token-Vektoren.
        [[token1_vec, token2_vec, ...], [token1_vec, ...], ...]

    Raises:
        BackendUnavailableError: KIron nicht erreichbar oder ungueltiges Payload.
    """
    if not texts:
        return []

    try:
        url = _resolve_colbert_url(config)
        resp = httpx.post(
            f"{url}/api/embed_colbert",
            json={"model": model, "input": texts},
            timeout=120.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        embeddings = payload["embeddings"]
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        raise BackendUnavailableError(
            f"ColBERT-Backend fuer Modell '{model}' nicht erreichbar "
            f"oder lieferte eine ungueltige Antwort"
        ) from exc

    if not isinstance(embeddings, list):
        raise BackendUnavailableError(
            f"ColBERT-Backend lieferte ungueltiges embeddings-Payload fuer Modell '{model}'"
        )
    if len(embeddings) != len(texts):
        raise BackendUnavailableError(
            f"ColBERT-Backend lieferte {len(embeddings)} Embeddings fuer "
            f"{len(texts)} Texte (Modell '{model}')"
        )
    return embeddings


def embed_colbert_query(
    config: dict,
    model: str,
    query: str,
) -> list[list[float]]:
    """Token-Level ColBERT-Embedding fuer eine Query.

    Args:
        config: Globale Konfiguration.
        model:  ColBERT-Modellname.
        query:  Suchanfrage.

    Returns:
        Liste von Token-Vektoren fuer die Query.
    """
    result = embed_colbert(config, model, [query])
    if not result:
        return []
    return result[0]


# ---------------------------------------------------------------------------
# ColBERTStore
# ---------------------------------------------------------------------------


class ColBERTStore:
    """Wrapper um Qdrant-Collection mit Multi-Vector-Konfiguration.

    Erstellt die Collection automatisch bei Bedarf mit der
    korrekten multivector_config (MaxSim comparator).
    """

    _PAYLOAD_INDEXES = {
        "source": PayloadSchemaType.TEXT,
        "source_key": PayloadSchemaType.KEYWORD,
        "directory": PayloadSchemaType.TEXT,
        "file_type": PayloadSchemaType.KEYWORD,
        "ds_ids": PayloadSchemaType.INTEGER,
        "modified": PayloadSchemaType.DATETIME,
    }

    def __init__(self, config: dict, model: str):
        self._config = config
        self._model = normalize_model_name(model) or model
        self._collection = colbert_collection_name(self._model)

        qdrant_cfg = config.get("vectordb", {})
        self._client = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=qdrant_cfg.get("timeout", 60),
        )
        self.ensure_collection()

    def ensure_collection(self) -> None:
        """Erstellt die Collection falls nicht vorhanden.

        Collection-Config:
            vectors: size=<probed vom Backend>, distance=Cosine,
                     multivector_config.comparator=max_sim
            payload_indexes: ds_ids (integer), source (keyword)
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        try:
            collections = [c.name for c in self._client.get_collections().collections]
        except Exception as exc:
            raise VectorDBConnectionError(f"ColBERT get_collections fehlgeschlagen: {exc}") from exc
        if self._collection not in collections:
            dim = self._probe_dimension()
            try:
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                        multivector_config=MultiVectorConfig(
                            comparator=MultiVectorComparator.MAX_SIM,
                        ),
                    ),
                )
                logger.info("ColBERT-Collection %s angelegt (dim=%d)", self._collection, dim)
            except UnexpectedResponse as exc:
                # Race-Condition: paralleler Worker hat die Collection in der
                # Zwischenzeit angelegt (TOCTOU zwischen get_collections und
                # create_collection). 409 ist hier kein Fehler.
                if exc.status_code == 409:
                    logger.info(
                        "ColBERT-Collection %s wurde von parallelem Worker angelegt — nutze existierende",
                        self._collection,
                    )
                else:
                    raise VectorDBConnectionError(
                        f"ColBERT create_collection {self._collection!r} fehlgeschlagen: {exc}"
                    ) from exc
            except ValueError as exc:
                # qdrant-client validiert teils intern und mappt AlreadyExists
                # nicht immer auf UnexpectedResponse — String-Match als Fallback.
                if "already exists" in str(exc).lower():
                    logger.info(
                        "ColBERT-Collection %s wurde von parallelem Worker angelegt — nutze existierende",
                        self._collection,
                    )
                else:
                    raise VectorDBConnectionError(
                        f"ColBERT create_collection {self._collection!r} fehlgeschlagen: {exc}"
                    ) from exc
            except Exception as exc:
                raise VectorDBConnectionError(
                    f"ColBERT create_collection {self._collection!r} fehlgeschlagen: {exc}"
                ) from exc
        self._ensure_payload_indexes()

    def _probe_dimension(self) -> int:
        """Ermittelt die Token-Dimension des Modells per Test-Embedding.

        Raises:
            RetrievalError: Wenn das Backend nicht erreichbar ist oder kein
                gueltiges Embedding liefert. Verhindert, dass eine Collection
                mit falscher Dimension angelegt wird (Qdrant-Collections sind
                dimensionsfix; ein falscher Wert fuehrt zu Shape-Fehlern in
                MaxSim und einer dauerhaft korrupten Collection).
        """
        try:
            vectors = embed_colbert(self._config, self._model, ["probe"])
        except BackendUnavailableError as exc:
            raise RetrievalError(
                f"ColBERT-Dimension-Probing fuer Modell '{self._model}' "
                f"fehlgeschlagen: Backend nicht erreichbar ({exc})"
            ) from exc
        if not (vectors and vectors[0] and vectors[0][0]):
            raise RetrievalError(
                f"ColBERT-Dimension-Probing fuer Modell '{self._model}' "
                f"lieferte kein gueltiges Embedding (leeres Payload)"
            )
        return len(vectors[0][0])

    def _ensure_payload_indexes(self) -> None:
        """Legt Payload-Indexes an falls noetig."""
        try:
            info = self._client.get_collection(self._collection)
            schema = info.payload_schema or {}
            for field, schema_type in self._PAYLOAD_INDEXES.items():
                if field not in schema:
                    self._client.create_payload_index(
                        collection_name=self._collection,
                        field_name=field,
                        field_schema=schema_type,
                    )
                else:
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
        except Exception as exc:
            logger.debug("ColBERT Payload-Index-Erstellung fehlgeschlagen: %s", exc)

    @classmethod
    def get_colbert_collection_info(cls, config: dict, model: str) -> dict | None:
        """Prueft Collection-Existenz vor Store-Konstruktion.

        Vermeidet die automatische Collection-Erstellung durch ``__init__``.

        Returns:
            ``{"name": str, "chunks": int}`` wenn die Collection existiert,
            ``None`` nur bei echtem 404/NotFound.
        Raises:
            Transiente/Auth-/Timeout-Fehler werden re-raised (fail-closed).
        """
        from qdrant_client.http.exceptions import UnexpectedResponse

        normalized = normalize_model_name(model) or model
        coll_name = colbert_collection_name(normalized)

        qdrant_cfg = config.get("vectordb", {})
        client = QdrantClient(
            url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
            api_key=qdrant_cfg.get("qdrant_api_key") or None,
            timeout=qdrant_cfg.get("timeout", 60),
        )
        try:
            info = client.get_collection(coll_name)
        except UnexpectedResponse as exc:
            if exc.status_code == 404:
                return None
            logger.warning(
                "ColBERT-Collection-Info fuer %s fehlgeschlagen: %s",
                coll_name,
                exc,
                exc_info=True,
            )
            raise
        except Exception as exc:
            logger.warning(
                "ColBERT-Collection-Info fuer %s fehlgeschlagen: %s",
                coll_name,
                exc,
                exc_info=True,
            )
            raise
        points_count = info.points_count
        if points_count is None:
            count_result = client.count(collection_name=coll_name, exact=True)
            points_count = count_result.count
        if points_count is None:
            raise RuntimeError(f"Qdrant-Count fuer {coll_name} ist unbekannt")
        return {"name": coll_name, "chunks": int(points_count)}

    def ids_with_datasource(self, ids: list[str], ds_id: int) -> set[str]:
        """Gibt nur Point-IDs zurueck, deren Qdrant-Payload ``ds_ids`` diese DS enthaelt.

        Prueft ``with_payload=["ds_ids"]`` und normalisiert Skalar/List/None.
        """
        if not ids:
            return set()

        result = set()
        from kiara.shared.constants import QDRANT_SCROLL_BATCH_SIZE

        batch_size = QDRANT_SCROLL_BATCH_SIZE

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            try:
                points = self._client.retrieve(
                    collection_name=self._collection,
                    ids=batch,
                    with_payload=["ds_ids"],
                    with_vectors=False,
                )
            except Exception as exc:
                logger.warning(
                    "ids_with_datasource retrieve fehlgeschlagen: %s",
                    exc,
                    exc_info=True,
                )
                continue

            for pt in points:
                pt_id = _normalize_point_id(pt.id)
                payload = pt.payload or {}
                raw_ds_ids = payload.get("ds_ids")
                if raw_ds_ids is None:
                    continue
                if isinstance(raw_ds_ids, (int, float)):
                    ds_list = [int(raw_ds_ids)]
                elif isinstance(raw_ds_ids, list):
                    ds_list = [int(d) for d in raw_ds_ids]
                else:
                    continue
                if ds_id in ds_list:
                    result.add(pt_id)
        return result

    def existing_ids(self, ids: list[str]) -> set[str]:
        """Gibt nur existierende Point-IDs zurueck (fuer Shared-Point-Update-Filter).

        Raises bei Qdrant-Fehlern (fail-closed), damit der Caller keine
        DB-Eintraege auf Basis unvollstaendiger Daten loescht.
        """
        if not ids:
            return set()

        result = set()
        from kiara.shared.constants import QDRANT_SCROLL_BATCH_SIZE

        batch_size = QDRANT_SCROLL_BATCH_SIZE

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            points = self._client.retrieve(
                collection_name=self._collection,
                ids=batch,
                with_payload=False,
                with_vectors=False,
            )
            for pt in points:
                result.add(_normalize_point_id(pt.id))
        return result

    def drop_collection_if_empty(self) -> bool:
        """Loescht die Collection nur wenn sie exakt 0 Punkte enthaelt.

        Count-/Drop-Fehler werden NICHT als "leer" gewertet und als Exception
        propagiert, damit der Caller Retry-Logik anwenden kann.
        ``points_count is None`` wird per ``count(exact=True)`` aufgeloest;
        bleibt der Count unbekannt, wird ein RuntimeError propagiert.

        Returns:
            True wenn gedroppt, False wenn nicht leer.
        """
        info = self._client.get_collection(self._collection)
        count = info.points_count
        if count is None:
            count_result = self._client.count(collection_name=self._collection, exact=True)
            count = count_result.count
        if count is None:
            raise RuntimeError(f"Qdrant-Count fuer {self._collection} ist unbekannt")
        if count > 0:
            return False
        self._client.delete_collection(self._collection)
        logger.info("Leere ColBERT-Collection %s gedroppt", self._collection)
        return True

    def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        token_embeddings: list[list[list[float]]],
    ) -> None:
        """Speichert Chunks mit Token-Level-Vektoren.

        Args:
            texts:            Chunk-Texte (fuer Payload).
            metadatas:        Chunk-Metadaten (fuer Payload).
            ids:              Chunk-IDs (identisch zu Standard-Collection).
            token_embeddings: Pro Chunk eine Liste von Token-Vektoren.
        """
        # Exact-Length-Guard: alle vier Listen muessen gleich lang sein.
        # Leere Listen sind OK (fruehes Return), aber Laengenabweichungen
        # wuerden durch zip() zu stillen Teilwrites fuehren.
        lengths = [len(texts), len(metadatas), len(ids), len(token_embeddings)]
        if all(l == 0 for l in lengths):
            return
        if len(set(lengths)) != 1:
            raise ValueError(
                f"ColBERTStore.add() Laengenabweichung: "
                f"texts={lengths[0]}, metadatas={lengths[1]}, "
                f"ids={lengths[2]}, token_embeddings={lengths[3]}"
            )

        from kiara.shared.constants import QDRANT_UPSERT_BATCH_SIZE

        batch_size = QDRANT_UPSERT_BATCH_SIZE

        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            batch_ids = ids[start:end]
            batch_texts = texts[start:end]
            batch_metas = metadatas[start:end]
            batch_embeddings = token_embeddings[start:end]

            points = []
            for cid, text, meta, tok_embs in zip(
                batch_ids, batch_texts, batch_metas, batch_embeddings, strict=False
            ):
                payload = dict(meta) if meta else {}
                payload["text"] = text
                points.append(
                    PointStruct(
                        id=cid,
                        vector=tok_embs,
                        payload=payload,
                    )
                )

            try:
                self._client.upsert(
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )
            except Exception as exc:
                raise VectorDBConnectionError(f"ColBERT-Upsert fehlgeschlagen: {exc}") from exc

    def update_ds_ids(
        self,
        ds_ids_map: dict[str, list[int]],
        *,
        wait: bool = False,
    ) -> None:
        """Aktualisiert ds_ids-Payloads fuer Shared-Chunks (batched).

        Gruppiert Chunks mit identischen ds_ids-Listen fuer effiziente
        Batch-Updates via set_payload (ein API-Call pro Gruppe+Batch).
        Retry bei transienten Fehlern (Timeout, Connection-Reset).

        Args:
            ds_ids_map: Dict chunk_id -> [ds_id, ...].
            wait: True fuer synchrones Warten (Loesch-/Repair-Pfade).
        """
        if not ds_ids_map:
            return

        import time
        from collections import defaultdict

        from kiara.shared.constants import QDRANT_DELETE_BATCH_SIZE

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
                            logger.error(
                                "ColBERT update_ds_ids nach %d Versuchen fehlgeschlagen (batch_size=%d): %s",
                                max_retries,
                                len(batch),
                                exc,
                                exc_info=True,
                            )
                            raise VectorDBConnectionError(
                                f"ColBERT set_payload nach {max_retries} Versuchen fehlgeschlagen: {exc}"
                            ) from exc
                        time.sleep(2**attempt)

    def search(
        self,
        query_tokens: list[list[float]],
        k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """MaxSim-Suche ueber Token-Level-Vektoren.

        Args:
            query_tokens: Token-Vektoren der Query.
            k:            Anzahl Ergebnisse.
            where:        Backend-agnostisches Where-Format von _build_filter().

        Returns:
            Liste von Ergebnis-Dicts im normalen Retrieval-Shape:
            ``{"text": str, "metadata": dict, "score": float, "_doc_id": str}``.
        """
        if not query_tokens:
            return []

        qdrant_filter = build_qdrant_filter(where)

        try:
            hits = self._client.query_points(
                collection_name=self._collection,
                query=query_tokens,
                limit=k,
                query_filter=qdrant_filter,
                with_payload=True,
            ).points
        except Exception as exc:
            logger.warning("ColBERT-Suche fehlgeschlagen: %s", exc, exc_info=True)
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.pop("text", "")
            results.append(
                {
                    "text": text,
                    "metadata": payload,
                    "score": hit.score,
                    "_doc_id": _normalize_point_id(hit.id),
                }
            )
        return results

    def get_collection_name(self) -> str:
        """Gibt den Collection-Namen zurueck."""
        return self._collection

    def delete_by_ids(self, ids: list[str]) -> None:
        """Loescht Chunks nach IDs."""
        if not ids:
            return

        from kiara.shared.constants import QDRANT_DELETE_BATCH_SIZE

        batch_size = QDRANT_DELETE_BATCH_SIZE

        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            try:
                self._client.delete(
                    collection_name=self._collection,
                    points_selector=batch,
                    wait=True,
                )
            except Exception as exc:
                raise VectorDBConnectionError(
                    f"ColBERT delete_by_ids fehlgeschlagen: {exc}"
                ) from exc

    def get_point_count(self) -> int:
        """Gibt die Anzahl der Punkte in der Collection zurueck."""
        try:
            info = self._client.get_collection(self._collection)
            return info.points_count or 0
        except Exception:
            logger.debug(
                "ColBERT get_point_count fuer %s fehlgeschlagen, liefere 0",
                self._collection,
                exc_info=True,
            )
            return 0

    @classmethod
    def list_colbert_collections(cls, config: dict) -> list[dict]:
        """Listet alle ColBERT-Collections in Qdrant auf.

        Returns:
            [{"kind": "colbert", "name": "colbert_colbert-xm",
              "model": "colbert-xm", "chunks": int}, ...]
        """
        qdrant_cfg = config.get("vectordb", {})
        try:
            client = QdrantClient(
                url=qdrant_cfg.get("qdrant_url", DEFAULT_QDRANT_URL),
                api_key=qdrant_cfg.get("qdrant_api_key") or None,
                timeout=qdrant_cfg.get("timeout", 60),
            )
            all_collections = client.get_collections().collections
        except httpx.TransportError as exc:
            logger.error("Qdrant nicht erreichbar: %s", exc, exc_info=True)
            raise
        except Exception as exc:
            logger.error(
                "Qdrant list_colbert_collections fehlgeschlagen: %s",
                exc,
                exc_info=True,
            )
            raise

        result = []
        for coll in all_collections:
            if not coll.name.startswith("colbert_"):
                continue
            try:
                model = model_name_from_colbert_collection(coll.name)
            except ValueError:
                continue
            try:
                info = client.get_collection(coll.name)
                chunks = info.points_count or 0
            except Exception as exc:
                logger.warning(
                    "ColBERT-Collection %s nicht abrufbar: %s",
                    coll.name,
                    exc,
                    exc_info=True,
                )
                chunks = -1
            result.append(
                {
                    "kind": "colbert",
                    "name": coll.name,
                    "model": model,
                    "chunks": chunks,
                }
            )
        return result
