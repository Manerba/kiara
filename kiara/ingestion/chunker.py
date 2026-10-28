"""Chunking strategies for text and code documents."""

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

from kiara.shared.constants import CODE_EXTENSIONS

_LANG_MAP = {
    ".cs": Language.CSHARP,
    ".ps1": Language.POWERSHELL,
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".html": Language.HTML,
    ".sh": Language.MARKDOWN,  # Reasonable fallback for shell scripts
    ".vb": Language.VISUALBASIC6,
    ".java": Language.JAVA,
    ".rs": Language.RUST,
    ".go": Language.GO,
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cc": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
}


def get_splitter(
    file_type: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    *,
    add_start_index: bool = False,
) -> RecursiveCharacterTextSplitter:
    """Return an appropriate text splitter for the given file type.

    Args:
        file_type: File extension (e.g. '.py', '.docx').
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between chunks.
        add_start_index: If True, each chunk's metadata will contain
            ``start_index`` (character offset in the original text).

    Returns:
        A LangChain text splitter instance.
    """
    if file_type in _LANG_MAP:
        return RecursiveCharacterTextSplitter.from_language(
            language=_LANG_MAP[file_type],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )

    if file_type in CODE_EXTENSIONS:
        # Generic code splitter with code-aware separators.
        # Empty-string separator forces char-level fallback for minified or
        # otherwise whitespace-free code so chunk_size cannot be exceeded.
        return RecursiveCharacterTextSplitter(
            separators=["\nclass ", "\ndef ", "\nfunction ", "\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=add_start_index,
        )

    # Default: text splitter
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )


def chunk_document(
    text: str, metadata: dict, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[dict]:
    """Split a document into chunks, preserving metadata on each chunk.

    Each chunk dict contains ``char_start`` / ``char_end`` (character offsets
    in the original *text*).  These are used by Late Chunking (Phase 18a) to
    map chunk boundaries onto the document-level token sequence.

    Args:
        text: The document text.
        metadata: Metadata dict to attach to each chunk.
        chunk_size: Maximum chunk size.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of dicts with 'text', 'metadata', 'char_start', 'char_end' keys.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not (0 <= chunk_overlap < chunk_size):
        raise ValueError(
            f"chunk_overlap must be >= 0 and < chunk_size ({chunk_size}), got {chunk_overlap}"
        )

    file_type = metadata.get("file_type", "")
    if metadata.get("content_format") == "markdown":
        file_type = ".md"
    splitter = get_splitter(file_type, chunk_size, chunk_overlap, add_start_index=True)

    docs = splitter.create_documents([text], metadatas=[metadata])
    docs = [doc for doc in docs if doc.page_content.strip()]

    result = []
    for i, doc in enumerate(docs):
        chunk_meta = metadata.copy()
        chunk_meta["chunk_index"] = i
        chunk_meta["chunk_total"] = len(docs)
        start = int(doc.metadata["start_index"])
        end = start + len(doc.page_content)
        result.append(
            {
                "text": doc.page_content,
                "metadata": chunk_meta,
                "char_start": start,
                "char_end": end,
            }
        )

    return result


def chunk_document_with_full_text(
    text: str, metadata: dict, chunk_size: int = 1000, chunk_overlap: int = 200
) -> tuple[str, list[dict]]:
    """HTML-Markdown vorbereiten und Spans relativ zum zurückgegebenen Text bilden.

    Aufrufer müssen diesen Volltext auch für Late Chunking verwenden. Der
    bestehende ``chunk_document``-Vertrag für unveränderte Eingaben bleibt erhalten.
    """
    if not (
        metadata.get("content_format") == "markdown"
        and metadata.get("source_mime_type") == "text/html"
    ):
        return text, chunk_document(text, metadata, chunk_size, chunk_overlap)

    from kiara.ingestion.markdown_chunks import HTML_CHUNKING_REVISION, prepare_markdown_chunks

    full_text, spans = prepare_markdown_chunks(text, chunk_size, chunk_overlap)
    chunks = []
    for i, (start, end) in enumerate(spans):
        chunk_meta = {
            **metadata,
            "html_chunking_revision": HTML_CHUNKING_REVISION,
            "chunk_index": i,
            "chunk_total": len(spans),
        }
        chunks.append(
            {
                "text": full_text[start:end],
                "metadata": chunk_meta,
                "char_start": start,
                "char_end": end,
            }
        )
    return full_text, chunks
