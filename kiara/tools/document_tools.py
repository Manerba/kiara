"""Gezieltes Lesen bereits gefundener Dokumente innerhalb einer Agent-Anfrage."""

import hashlib
import json
import threading

from kiara.exceptions import RetrievalError
from kiara.generation.context_selection import document_key
from kiara.shared.constants import AGENT_DOCUMENT_READ_MAX_CHUNKS
from kiara.tools.base import BaseTool, ToolDefinition, ToolResult


def _document_version(metadata: dict) -> tuple | None:
    """Datei und gespeicherte Verarbeitung muessen bei jeder Leseseite gleich bleiben."""
    if metadata.get("processing_pending"):
        return None
    modified, size = metadata.get("modified"), metadata.get("size_bytes")
    file_hash = metadata.get("file_hash")
    if not (isinstance(modified, str) and modified.strip() and type(size) is int and size >= 0):
        return None
    if file_hash is not None and not isinstance(file_hash, str):
        return None
    processing = tuple(
        metadata.get(field)
        for field in (
            "chunk_strategy",
            "chunk_size_used",
            "chunk_overlap_used",
            "html_chunking_revision",
            "vlm_model",
            "processing_version",
        )
    )
    return modified, size, file_hash, metadata.get("chunk_total"), processing


class DocumentReferences:
    """Verknuepft Suchtreffer mit unverwechselbaren Referenzen fuer diese Anfrage."""

    def __init__(self, collection: str = "") -> None:
        self._collection = collection
        self._references: dict[str, dict] = {}
        self._versions: dict[tuple, tuple] = {}
        self._lock = threading.Lock()

    def _reference_id(self, key: tuple, version: tuple) -> str:
        digest = hashlib.sha256(json.dumps((self._collection, key, version)).encode()).hexdigest()
        return "doc_" + digest[:24]

    def annotate(self, chunks: list[dict]) -> list[dict]:
        from kiara.ingestion.document_publication import assert_documents_published

        assert_documents_published(self._collection, chunks)
        annotated = []
        with self._lock:
            # Erst den gesamten Suchbatch pruefen, bevor Referenzen oder
            # Deduplizierungszustand neue Treffer als uebernommen markieren.
            versions = dict(self._versions)
            for chunk in chunks:
                key = document_key(chunk)
                meta = chunk.get("metadata") or {}
                if meta.get("processing_pending"):
                    raise RetrievalError("Dokumentverarbeitung ist noch nicht abgeschlossen.")
                version = _document_version(meta)
                if key in versions and version != versions[key]:
                    raise RetrievalError(
                        "Dokumentversion wurde veraendert. Bitte eine neue Anfrage starten."
                    )
                if version is not None:
                    versions[key] = version
                    saved = self._references.get(self._reference_id(key, version))
                    known_hash = (
                        saved["chunk_hashes"].get(meta.get("chunk_index")) if saved else None
                    )
                    if known_hash is not None and (
                        not isinstance(chunk.get("text"), str)
                        or hashlib.sha256(chunk["text"].encode()).hexdigest() != known_hash
                    ):
                        raise RetrievalError(
                            "Dokumentversion wurde veraendert. Bitte eine neue Anfrage starten."
                        )
            for chunk in chunks:
                meta = dict(chunk.get("metadata") or {})
                meta.pop("document_ref", None)
                index, total = meta.get("chunk_index"), meta.get("chunk_total")
                version = _document_version(meta)
                if (
                    meta.get("source")
                    and version is not None
                    and type(index) is int
                    and type(total) is int
                    and 0 <= index < total
                ):
                    key = document_key(chunk)
                    reference = self._reference_id(key, version)
                    self._references.setdefault(
                        reference,
                        {"key": key, "metadata": meta, "version": version, "chunk_hashes": {}},
                    )
                    meta["document_ref"] = reference
                annotated.append({**chunk, "metadata": meta})
            self._versions = versions
        return annotated

    def resolve(self, reference: str) -> dict | None:
        with self._lock:
            return self._references.get(reference)

    def validate_read(self, reference: str, chunks: list[dict]) -> None:
        """Verhindert Versionswechsel und verschobene Offsets bereits gelesener Chunks."""
        from kiara.ingestion.document_publication import assert_documents_published

        assert_documents_published(self._collection, chunks)
        with self._lock:
            saved = self._references[reference]
            version = saved["version"]
            hashes = {
                chunk["metadata"]["chunk_index"]: hashlib.sha256(chunk["text"].encode()).hexdigest()
                for chunk in chunks
            }
            if (
                self._versions.get(saved["key"], version) != version
                or any(_document_version(chunk["metadata"]) != version for chunk in chunks)
                or any(
                    index in saved["chunk_hashes"] and saved["chunk_hashes"][index] != digest
                    for index, digest in hashes.items()
                )
            ):
                raise RetrievalError(
                    "Dokumentversion wurde veraendert. Bitte eine neue Anfrage starten."
                )
            saved["chunk_hashes"].update(hashes)
            self._versions[saved["key"]] = version


class DocumentReadChunks(list):
    """Lesebereich mit internem, separat validiertem Anonymisierungskontext.

    Die Liste serialisiert ausschliesslich die angeforderten Chunks. Der
    Kontext wird erst von einem anonymisierenden Adapter benoetigt.
    """

    def __init__(self, chunks, load_context):
        super().__init__(chunks)
        self.load_context = load_context


class ReadDocumentTool(BaseTool):
    """Liest einen zusammenhaengenden Bereich einer gefundenen Quelle."""

    def __init__(self, retriever, config, references, allowed_datasource_ids=None):
        self._retriever = retriever
        self._references = references
        self._allowed_datasource_ids = allowed_datasource_ids
        self._max_chunks = (
            config.get("chat", {})
            .get("agent", {})
            .get("document_read_max_chunks", AGENT_DOCUMENT_READ_MAX_CHUNKS)
        )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="read_document",
            description=(
                "Liest Originaltext eines mit search_documents gefundenen Dokuments. "
                "Wenn eine Fundstelle Angaben, Bedingungen oder Belege offenlaesst, "
                "hier gezielt nachlesen statt erneut aehnlich zu suchen. Kleine Dokumente "
                "ab Chunk 0 vollstaendig lesen; bei grossen Dokumenten die Fundstelle "
                "und ihre Nachbarn lesen. document_id exakt aus dem Suchergebnis "
                "uebernehmen. Das Ergebnis nennt den gelesenen Bereich und die "
                "Parameter zum Weiterlesen. Nur tatsaechlich gelesene Inhalte belegen."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "document_id": {"type": "string", "minLength": 1},
                    "start_chunk": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Erster Chunk (0-basiert), Standard: Dokumentanfang.",
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": self._max_chunks,
                        "default": self._max_chunks,
                    },
                    "start_offset": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "Zeichenposition im ersten Chunk; nur zum Weiterlesen.",
                    },
                },
                "required": ["document_id"],
                "additionalProperties": False,
            },
            category="builtin",
        )

    async def execute(self, params: dict, user_id: int | None = None) -> ToolResult:
        if error := self._validate_or_error(params):
            return error
        reference = self._references.resolve(params["document_id"])
        if reference is None:
            return ToolResult(
                False, None, "Unbekannte Dokument-ID. Zuerst search_documents nutzen.", ""
            )
        meta = reference["metadata"]
        start = params.get("start_chunk", 0)
        offset = params.get("start_offset", 0)
        total = meta["chunk_total"]
        if start >= total:
            return ToolResult(False, None, f"start_chunk muss kleiner als {total} sein.", "")
        end = min(total, start + params.get("max_chunks", self._max_chunks))
        from kiara.retrieval.retriever import _normalize_datasource_ids

        reference_ds = _normalize_datasource_ids(meta.get("ds_ids"))
        allowed = self._allowed_datasource_ids
        if allowed is not None:
            reference_ds.intersection_update(allowed)
            if not reference_ds:
                return ToolResult(False, None, "Dokument ist nicht freigegeben.", "")
        filter_ds = sorted(reference_ds) if reference_ds else None

        def load_range(low, high):
            try:
                chunks = self._retriever.load_document_range(
                    meta["source"],
                    low,
                    high,
                    filter_datasource_ids=filter_ds,
                    parser=meta.get("parser") or "standard",
                )
                chunks.sort(key=lambda c: c["metadata"]["chunk_index"])
                if len(chunks) != high - low or any(
                    document_key(chunk) != reference["key"]
                    or type(chunk["metadata"].get("chunk_index")) is not int
                    or chunk["metadata"]["chunk_index"] != index
                    or type(chunk["metadata"].get("chunk_total")) is not int
                    or chunk["metadata"].get("chunk_total") != total
                    or not isinstance(chunk.get("text"), str)
                    or not chunk["text"]
                    for index, chunk in enumerate(chunks, low)
                ):
                    raise RetrievalError(
                        "Dokumentbereich fehlt, ist nicht eindeutig oder wurde veraendert. "
                        "Dokument erneut suchen."
                    )
                self._references.validate_read(params["document_id"], chunks)
                return chunks
            except Exception as exc:
                from kiara.retrieval.index_compatibility import IndexCompatibilityError

                if isinstance(exc, (IndexCompatibilityError, RetrievalError)):
                    raise
                raise RetrievalError("Dokumentbereich konnte nicht gelesen werden") from exc

        chunks = load_range(start, end)
        if offset >= len(chunks[0]["text"]) and offset != 0:
            return ToolResult(False, None, "start_offset liegt ausserhalb des ersten Chunks.", "")
        chunks = [
            {**chunk, "metadata": {**chunk["metadata"], "document_ref": params["document_id"]}}
            for chunk in chunks
        ]
        request = {
            "document_id": params["document_id"],
            "start_chunk": start,
            "start_offset": offset,
            "total_chunks": total,
        }
        return ToolResult(
            True,
            {
                "results": DocumentReadChunks(
                    chunks,
                    lambda: load_range(0, total),
                ),
                "document_read": request,
                "sources": [
                    {
                        "source": meta["source"],
                        "score": 0.0,
                        "excerpt": "",
                        "document_id": params["document_id"],
                    }
                ],
            },
            None,
            f"Dokumentbereich geladen: Chunks {start} bis {end - 1} von {total}.",
        )
