"""Markdown-Tabellen aus HTML vor dem Bestimmen ihrer Chunk-Spans vorbereiten."""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from kiara.exceptions import ChunkingContentError

HTML_CHUNKING_REVISION = "html-markdown-tables-v2"
_SEPARATOR = re.compile(r":?-{3,}:?\Z")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _cells(line: str) -> list[str] | None:
    """Unescaped Pipes teilen Zellen; escaped Pipes bleiben Bestandteil des Texts."""
    parts, current = [], []
    escaped = False
    for char in line.strip():
        if char == "|" and not escaped:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
        escaped = char == "\\" and not escaped
    if not parts:
        return None
    parts.append("".join(current).strip())
    if not parts[0]:
        parts.pop(0)
    if parts and not parts[-1]:
        parts.pop()
    return parts


def _long_row(headers: list[str], cells: list[str], size: int, table: int, row: int):
    """Jeden Werteabschnitt mit Feldname und gemeinsamer Zeilenidentität markieren."""
    labels = [name or f"Spalte {i + 1}" for i, name in enumerate(headers)]
    prefix = f"Tabelle {table}, Zeile {row} (Fortsetzung)\n"
    first = f"{labels[0]}: {cells[0]}"
    start = 0
    # Ein kurzer erster Schlüssel bleibt in jeder Fortsetzung lesbar. Lange
    # Schlüssel werden selbst als Feld aufgeteilt; die Zeilenidentität bleibt.
    if len(cells) > 1 and cells[0] and len(first) <= size // 3:
        prefix += first + "\n"
        start = 1
    sections = []
    for label, value in zip(labels[start:], cells[start:], strict=True):
        heading = f"{prefix}{label}: "
        available = size - len(heading)
        if available < 1:
            raise ChunkingContentError("Chunk-Budget reicht nicht für Tabellenzeile und Feldnamen")
        if not value:
            sections.append(heading.rstrip())
        position = 0
        while position < len(value):
            cut = min(position + available, len(value))
            if cut < len(value):
                word_end = value.rfind(" ", position, cut)
                if word_end >= position + (cut - position) // 2:
                    cut = word_end + 1
            sections.append(heading + value[position:cut])
            position = cut
    return sections


def _table_sections(header_lines, rows, headers, size, table):
    header = "\n".join(header_lines)
    if len(header) > size and not rows:
        raise ChunkingContentError("Tabellenkopf überschreitet das Chunk-Budget")
    sections, current = [], header
    has_rows = False
    for row_number, line in enumerate(rows, 1):
        cells = _cells(line)
        if cells is None or len(cells) != len(headers):
            raise ChunkingContentError(
                "Tabellenzeile hat eine andere Spaltenzahl als der Tabellenkopf"
            )
        if len(header) + 1 + len(line) > size:
            if has_rows:
                sections.append(current)
            sections.extend(_long_row(headers, cells, size, table, row_number))
            current, has_rows = header, False
        elif len(current) + 1 + len(line) > size:
            sections.append(current)
            current, has_rows = header + "\n" + line, True
        else:
            current += "\n" + line
            has_rows = True
    if has_rows or not rows:
        sections.append(current)
    return sections


def prepare_markdown_chunks(
    text: str, chunk_size: int, chunk_overlap: int
) -> tuple[str, list[tuple[int, int]]]:
    """Liefert tatsächlichen Volltext und dessen begrenzte Chunk-Spans.

    Tabellenköpfe werden im Volltext wiederholt, bevor Spans entstehen.
    Ganze Tabellenzeilen überlappen nicht. Prosa behält den normalen
    Absatz-/Zeilensplitter einschließlich des konfigurierten Overlaps.
    """
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Ungültiges Chunk-Budget oder Chunk-Overlap")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    pieces, spans, prose = [], [], []
    offset = 0

    def append_prose():
        nonlocal offset
        segment = "".join(prose)
        if not segment:
            return
        for doc in splitter.create_documents([segment]):
            if doc.page_content.strip():
                start = int(doc.metadata["start_index"])
                end = start + len(doc.page_content)
                if start < 0 or segment[start:end] != doc.page_content:
                    raise ValueError("Ungültiger Markdown-Chunk-Span")
                spans.append((offset + start, offset + end))
        pieces.append(segment)
        offset += len(segment)
        prose.clear()

    lines = text.splitlines(keepends=True)
    i, table = 0, 0
    fence = None
    while i < len(lines):
        match = _FENCE.match(lines[i].rstrip("\r\n"))
        if match:
            marker, suffix = match.groups()
            if fence is None:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence) and not suffix.strip():
                fence = None
            prose.append(lines[i])
            i += 1
            continue
        headers = _cells(lines[i]) if fence is None else None
        separators = _cells(lines[i + 1]) if i + 1 < len(lines) else None
        if not (
            headers
            and separators
            and len(headers) == len(separators)
            and all(_SEPARATOR.fullmatch(cell) for cell in separators)
        ):
            prose.append(lines[i])
            i += 1
            continue
        append_prose()
        table += 1
        header_lines = [line.rstrip("\r\n") for line in lines[i : i + 2]]
        end = i + 2
        while end < len(lines) and lines[end].strip() and _cells(lines[end]) is not None:
            next_header = _cells(lines[end])
            next_separator = _cells(lines[end + 1]) if end + 1 < len(lines) else None
            if (
                next_separator
                and len(next_header) == len(next_separator)
                and all(_SEPARATOR.fullmatch(cell) for cell in next_separator)
            ):
                break
            end += 1
        rows = [line.rstrip("\r\n") for line in lines[i + 2 : end]]
        sections = _table_sections(header_lines, rows, headers, chunk_size, table)
        for number, section in enumerate(sections):
            if number:
                pieces.append("\n\n")
                offset += 2
            pieces.append(section)
            spans.append((offset, offset + len(section)))
            offset += len(section)
        if lines[end - 1].endswith("\n"):
            pieces.append("\n")
            offset += 1
        i = end
    append_prose()
    return "".join(pieces), spans
