"""Version identities for prepared files and their stored document representations."""

import hashlib
import json
from pathlib import Path


def validate_document_representation(texts: list[str], metadatas: list[dict]) -> None:
    """Nur vollstaendige, einheitlich verarbeitete Darstellungen freigeben."""
    groups = {}
    fields = (
        "modified",
        "size_bytes",
        "chunk_total",
        "chunk_strategy",
        "chunk_size_used",
        "chunk_overlap_used",
        "html_chunking_revision",
        "vlm_model",
    )
    for text, meta in zip(texts, metadatas, strict=True):
        index, total = meta.get("chunk_index"), meta.get("chunk_total")
        if (
            not isinstance(text, str)
            or not text
            or not meta.get("source")
            or type(index) is not int
            or type(total) is not int
            or not 0 <= index < total
        ):
            raise ValueError("Ungueltiger Chunk beim Dokumentabschluss")
        key = (meta["source"], meta.get("parser", "standard"))
        signature = tuple(meta.get(field) for field in fields)
        if key not in groups:
            groups[key] = (signature, set())
        expected, indices = groups[key]
        if signature != expected or index in indices:
            raise ValueError("Uneinheitliche Dokumentdarstellung beim Abschluss")
        indices.add(index)
    for signature, indices in groups.values():
        if len(indices) != signature[2]:
            raise ValueError("Unvollstaendige Dokumentdarstellung beim Abschluss")


def stamp_document_versions(texts: list[str], metadatas: list[dict]) -> None:
    """Bind every chunk to the complete text and boundaries of its representation.

    Hash incrementally so large documents need no second serialized full-text copy.
    Parser representations have independent versions. The ingestion completion
    step stamps the full persisted representation, including enriched prefixes.
    """
    fields = (
        "chunk_index",
        "chunk_total",
        "chunk_strategy",
        "chunk_size_used",
        "chunk_overlap_used",
        "html_chunking_revision",
        "vlm_model",
        "source_char_start",
        "source_char_end",
        "source_gap_before",
        "source_overlap_before",
        "source_body_start",
    )
    digests = {}
    # Qdrant liefert IDs nicht zwingend in Dokumentreihenfolge zurueck.
    ordered = sorted(
        zip(texts, metadatas, strict=True), key=lambda row: row[1].get("chunk_index", 0)
    )
    for text, meta in ordered:
        key = (meta.get("source"), meta.get("parser", "standard"))
        if key not in digests:
            digests[key] = hashlib.sha256(b"kiara-document-processing-v1\0")
        digest = digests[key]
        for value in (
            json.dumps([meta.get(field) for field in fields], ensure_ascii=False).encode(),
            text.encode(),
        ):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    versions = {key: digest.hexdigest() for key, digest in digests.items()}
    for meta in metadatas:
        meta["processing_version"] = versions[(meta.get("source"), meta.get("parser", "standard"))]


def processing_file_hash(config: dict, path: str, source_hash: str) -> str:
    """Unchanged bytes still need ingestion after a relevant preparation change."""
    suffix = Path(path).suffix.lower()
    if suffix not in {".html", ".htm", ".txt"}:
        return source_hash

    from kiara_document_loaders.html import conversion_revision

    from kiara.config import get_mail_archive_config
    from kiara.ingestion.mail_archive import read_mail_metadata, read_mail_original
    from kiara.ingestion.markdown_chunks import HTML_CHUNKING_REVISION

    provenance = None
    if suffix == ".txt":
        archive = get_mail_archive_config(config)
        original = read_mail_original(path, archive)
        if original is not None:
            provenance = read_mail_metadata(path, Path(path).read_text(encoding="utf-8"), archive)
    if suffix not in {".html", ".htm"} and provenance is None:
        return source_hash

    ingestion = config.get("ingestion", {})
    identity = {
        "source_hash": source_hash,
        "html": conversion_revision(),
        "chunking": HTML_CHUNKING_REVISION,
        "metadata": "html-mail-provenance-v1",
        "chunk_size": ingestion.get("chunk_size"),
        "chunk_overlap": ingestion.get("chunk_overlap"),
        "provenance": provenance,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
