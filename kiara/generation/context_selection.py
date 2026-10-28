"""Strukturerhaltende Auswahl von Originaltext fuer Agent-Tool-Ergebnisse.

Die Auswahl erfolgt einmal vor den Modelladaptern. Einheiten sind ganze
Absaetze einschliesslich ihrer Nachbarn, Tabellen und Codebloecke. Es gibt
keine fachgebietsspezifische Gewichtung und keine erzeugten Zusammenfassungen.
"""

import math
import re
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import pairwise

from kiara.exceptions import RetrievalError
from kiara.shared.constants import DEFAULT_MAX_CONTEXT_CHUNKS
from kiara.shared.prompt_defense import (
    TAG_CONTEXT,
    SanitizeReport,
    sanitize_with_report,
    wrap_external_data,
)
from kiara.shared.prompt_defense_audit import log_defense_event


@dataclass(frozen=True)
class ModelCallBudget:
    """Preflight fuer einen vollstaendigen, frischen Modellkontext.

    UTF-8-Bytes sind eine bewusst konservative Schaetzung, kein Tokenizer.
    Die Reserve deckt Nachrichtenhuelle und Antwort/Thinking separat ab.
    Tool-Schemata und Verlauf muessen Teil der geprueften Nutzlast sein.
    """

    window: int
    output_tokens: int
    safety_tokens: int = 1024
    bytes_per_token: float = 2.0
    window_source: str = "configured_unverified"

    def inspect(self, system: str, user: str, tools: str = "") -> dict:
        estimate = (
            math.ceil(len((system + user + tools).encode("utf-8")) / self.bytes_per_token) + 128
        )
        required = estimate + self.output_tokens + self.safety_tokens
        return {
            "input_tokens": estimate,
            "counting": "utf8_estimate",
            "bytes_per_token": self.bytes_per_token,
            "window": self.window,
            "window_source": self.window_source,
            "output_reserve": self.output_tokens,
            "safety_reserve": self.safety_tokens,
            "required_tokens": required,
            "fits": required <= self.window,
            "exact_token_check": False,
        }


_GAP = "[Inhalt ausgelassen; dieser Auszug ist unvollstaendig.]"
_UNLOADED = "[Nicht geladener Dokumentbereich zwischen diesen Chunks.]"
_BUDGET_NOTICE = "[Kontextbudget: Weitere Dokumentinhalte wurden ausgelassen.]"
_EMPTY_NOTICE = (
    "[Kein vollstaendiger Abschnitt passt ins Kontextbudget. Dokument gezielt nachlesen.]"
)
_PARTIAL_PARAGRAPH = (
    "[Teilabschnitt eines langen Absatzes; Grenzen entsprechen gespeicherten Chunks.]"
)


@dataclass
class PreparedContext:
    text: str
    chunks: list[dict]
    input_chars: int
    truncated: bool
    references: list[dict] = field(default_factory=list)


def _integer(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def document_key(chunk: dict) -> tuple:
    """Trennt Datenquellen, Modelle und Parserdarstellungen desselben Pfads."""
    meta = chunk.get("metadata") or {}
    ds_ids = meta.get("ds_ids") or []
    if not isinstance(ds_ids, (list, tuple, set)):
        ds_ids = [ds_ids]
    return (
        str(meta.get("source") or ""),
        tuple(sorted({str(value) for value in ds_ids})),
        str(meta.get("embedding_model") or ""),
        str(meta.get("document_id") or ""),
        str(meta.get("parser") or "standard"),
    )


def _context_source_key(metadata: dict) -> tuple:
    reference = metadata.get("document_ref")
    return ("reference", reference) if reference else document_key({"metadata": metadata})


def _chunk_join(chunk: dict, report=None, *, raw: bool = False) -> dict:
    """Nur vom Erzeuger belegter Ueberlapp darf Belegtext entfernen."""
    meta = chunk.get("metadata") or {}
    if "context_join" in chunk:
        return chunk["context_join"]
    overlap = meta.get("source_overlap_before")
    gap = meta.get("source_gap_before")
    if (
        type(overlap) is not int
        or overlap < 0
        or not isinstance(gap, str)
        or meta.get("source_body_start") != 0
    ):
        return {"overlap": 0, "separator": "\n"}
    if report is None:
        if raw:
            return {"overlap": overlap, "separator": gap}
        report = sanitize_with_report(chunk["text"], track_offsets=True)
    clean_overlap = bisect_left(report.source_offsets, overlap)
    return {
        "overlap": clean_overlap,
        "separator": sanitize_with_report(gap).clean_text,
    }


def _join_adjacent(left: str, right: str, join: dict) -> tuple[int, str]:
    overlap = join.get("overlap", 0)
    if overlap and overlap <= min(len(left), len(right)) and left[-overlap:] == right[:overlap]:
        return overlap, ""
    # Ohne passende Herkunft wird kein Text gestrichen. Unbekannte Grenzen
    # behalten einen sichtbaren Abstand, belegte Nullabstaende bleiben leer.
    return 0, join.get("separator", "\n") if not overlap else "\n"


def _block_ranges(text: str) -> list[tuple[int, int]]:
    """Absatzgrenzen; Tabellen und fenced Code bleiben atomar.

    Einzelne Zeilen werden bewusst nicht bewertet oder getrennt. Das erhaelt
    unter anderem Beschriftung/Wert-Paare und mehrzeilige Tabellenzellen.
    """
    ranges: list[tuple[int, int]] = []
    start = 0
    offset = 0
    fence = ""
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        match = re.match(r"^(`{3,}|~{3,})", stripped)
        if match:
            marker = match.group(1)
            if not fence:
                fence = marker
            elif marker[0] == fence[0] and len(marker) >= len(fence):
                fence = ""
        if not stripped and not fence:
            if text[start:offset].strip():
                ranges.append((start, offset))
            start = offset + len(line)
        offset += len(line)
    if text[start:].strip():
        ranges.append((start, len(text)))
    return ranges


@dataclass
class _Document:
    metadata: dict
    text: str = ""
    # Positionen im zusammengesetzten Text -> Original-Chunk und dessen Offset.
    origins: list[tuple[int, int, int | None]] = field(default_factory=list)
    blocks: list[tuple[int, int]] = field(default_factory=list)
    selected: set[int] = field(default_factory=set)
    split_paragraphs: list[set[int]] = field(default_factory=list)
    score: float = 0.0

    def split_long_paragraphs(self, target: int, prepare_text: Callable[[str], str]) -> None:
        """Nutze Original-Chunk-Grenzen, wenn Absatzstruktur im Index fehlt.

        Tabellen, Listen und Code werden nicht auf diese Weise zerlegt.
        Aus solchen Teilabschnitten werden weiterhin zusammenhaengende Fenster
        einschliesslich Nachbarabschnitten gewaehlt und als unvollstaendig markiert.
        """
        blocks = []
        for start, end in self.blocks:
            text = self.text[start:end]
            structured = re.search(r"(?m)^\s*(?:[|`~]|#{1,6}\s|[-*+]\s|\d+[.)]\s)|\t", text)
            boundaries = sorted({low for low, _, _ in self.origins if start < low < end})
            if boundaries and not structured and len(prepare_text(text)) > target:
                offsets = [start, *boundaries, end]
                indices = set(range(len(blocks), len(blocks) + len(offsets) - 1))
                self.split_paragraphs.append(indices)
                blocks.extend(pairwise(offsets))
            else:
                blocks.append((start, end))
        self.blocks = blocks

    def excerpt(self) -> dict:
        if len(self.selected) == len(self.blocks):
            metadata = dict(self.metadata)
            metadata["context_spans"] = [
                {"chunk_index": index, "start": 0, "end": end - start}
                for start, end, index in self.origins
            ]
            metadata["context_truncated"] = False
            return {"text": self.text, "metadata": metadata, "score": self.score}
        ranges = [self.blocks[i] for i in sorted(self.selected)]
        parts: list[str] = []
        spans: list[dict] = []
        previous = 0
        for start, end in ranges:
            if self.text[previous:start].strip():
                parts.append(_GAP)
            # Keep original whitespace between consecutive selected blocks.
            elif parts:
                parts[-1] += self.text[previous:start]
            parts.append(self.text[start:end])
            for origin_start, origin_end, chunk_index in self.origins:
                low, high = max(start, origin_start), min(end, origin_end)
                if low < high:
                    spans.append(
                        {
                            "chunk_index": chunk_index,
                            "start": low - origin_start,
                            "end": high - origin_start,
                        }
                    )
            previous = end
        if self.text[previous:].strip():
            parts.append(_GAP)
        partial_paragraph = any(
            self.selected.intersection(indices) and not indices.issubset(self.selected)
            for indices in self.split_paragraphs
        )
        if partial_paragraph:
            parts.insert(0, _PARTIAL_PARAGRAPH + "\n")
        metadata = dict(self.metadata)
        metadata["context_spans"] = spans
        metadata["context_truncated"] = len(self.selected) < len(self.blocks)
        metadata["context_partial_paragraph"] = partial_paragraph
        return {
            "text": "".join(
                ("\n\n" if part == _GAP or (i and parts[i - 1] == _GAP) else "") + part
                for i, part in enumerate(parts)
            ).strip(),
            "metadata": metadata,
            "score": self.score,
        }


def _documents(
    chunks: list[dict], *, preserve_whitespace: bool = False, sanitized: bool = False
) -> list[_Document]:
    groups: dict[tuple, list[dict]] = {}
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            continue
        key = document_key(chunk)
        if not key[0] and not key[3]:
            key = (*key, position)
        groups.setdefault(key, []).append(chunk)

    documents = []
    for group in groups.values():
        ordered = sorted(
            group, key=lambda c: _integer((c.get("metadata") or {}).get("chunk_index")) or 0
        )
        meta = dict(ordered[0].get("metadata") or {})
        source_report = sanitize_with_report(str(meta.get("source") or "unbekannt"))
        meta["source"] = source_report.clean_text
        if source_report.severity != "none":
            log_defense_event("retrieval_context", source_report)
        document = _Document(metadata=meta, score=ordered[0].get("score", 0.0))
        seen: set[tuple] = set()
        previous_index = None
        previous_text = ""
        for chunk in ordered:
            index = _integer((chunk.get("metadata") or {}).get("chunk_index"))
            text = chunk["text"]
            if not sanitized:
                report = sanitize_with_report(text, track_offsets=True)
                if report.severity != "none":
                    log_defense_event("retrieval_context", report, document_source=meta["source"])
                text = report.clean_text
            join = _chunk_join(chunk, report if not sanitized else None)
            if not text and sanitized:
                # Beim Lesen wurde dieser Chunk vollstaendig verarbeitet,
                # auch wenn die Bereinigung keinen Belegtext uebrig laesst.
                if index is not None and (previous_index is None or index == previous_index + 1):
                    previous_index, previous_text = index, ""
                continue
            if (
                not text
                or (not preserve_whitespace and not text.strip())
                or (index is not None and (index, text) in seen)
            ):
                continue
            seen.add((index, text))
            overlap = 0
            if document.text:
                if index is not None and previous_index is not None and index == previous_index + 1:
                    overlap, separator = _join_adjacent(previous_text, text, join)
                    document.text += separator
                else:
                    document.text += f"\n\n{_UNLOADED}\n"
            start = len(document.text) - overlap
            document.text += text[overlap:]
            document.origins.append((start, start + len(text), index))
            previous_index, previous_text = index, text
        document.blocks = _block_ranges(document.text)
        if document.blocks or (preserve_whitespace and document.text):
            documents.append(document)
    return documents


def _render(
    chunks: list[dict],
    omitted: bool,
    references: list[dict] | None = None,
    *,
    body_reports: list[SanitizeReport] | None = None,
) -> str:
    parts = []
    for position, chunk in enumerate(chunks):
        meta = chunk["metadata"]
        # Belegte Nullabstaende koennen aus einzeln bereinigten Fragmenten
        # wieder Steuer-Tags bilden. Erst den zusammengesetzten Quellenkoerper
        # bereinigen, dann unsere eigene Quellenhuelle hinzufuegen.
        report = (
            body_reports[position]
            if body_reports is not None
            else sanitize_with_report(chunk["text"])
        )
        body = report.clean_text
        if report.severity != "none":
            log_defense_event("retrieval_context", report, document_source=meta["source"])
        number = meta["context_source_id"]
        indices = sorted(
            {
                span["chunk_index"]
                for span in meta["context_spans"]
                if span["chunk_index"] is not None
            }
        )
        location = ", ".join(map(str, indices)) or "unbekannt"
        reference = ""
        if meta.get("document_ref"):
            reference = (
                f"; document_id: {meta['document_ref']}; "
                f"Chunks gesamt: {meta.get('chunk_total', 'unbekannt')}"
            )
        reading = meta.get("document_read")
        read_notice = ""
        if reading:
            read_notice = (
                f"Gelesen: Chunk {reading['start_chunk']} ab Zeichen {reading['start_offset']} "
                f"bis Chunk {reading['end_chunk']} vor Zeichen {reading['end_offset']}. "
            )
            if not body:
                read_notice += "Dieser Bereich enthaelt nach Bereinigung keinen lesbaren Text. "
            if reading["complete_document"]:
                read_notice += "Dokument vollstaendig gelesen."
            else:
                if reading["start_chunk"] or reading["start_offset"]:
                    read_notice += "Davor liegen weitere Dokumentinhalte. "
                if reading["has_more"]:
                    read_notice += (
                        f'Weiterlesen: read_document(document_id="{meta["document_ref"]}", '
                        f"start_chunk={reading['next_chunk']}, "
                        f"start_offset={reading['next_offset']})."
                    )
                else:
                    read_notice += "Dokumentende erreicht."
            read_notice += "\n"
        parts.append(
            f"[Quelle {number}: {meta['source']}; Chunks: {location}{reference}]\n"
            f"{read_notice}{body}"
        )
    for meta in references or []:
        parts.append(
            f"[Nicht gelesene Quelle {meta['context_source_id']}: {meta['source']}; "
            f"document_id: {meta['document_ref']}; Chunk: {meta.get('chunk_index', 0)}; "
            f"Chunks gesamt: {meta.get('chunk_total', 'unbekannt')}. "
            "Inhalt ausgelassen; gezielt mit read_document nachlesen.]"
        )
    if omitted:
        parts.append(_BUDGET_NOTICE)
    return wrap_external_data("\n\n---\n\n".join(parts), TAG_CONTEXT)


def render_context(chunks: list[dict], references: list[dict] | None = None) -> str:
    """Rendert bereits ausgewaehlte Passagen, ohne sie erneut zu kuerzen."""
    omitted = bool(references) or any(c["metadata"].get("context_truncated") for c in chunks)
    return _render(chunks, omitted, references)


def _document_entity_ranges(chunks: list[dict], prepare_text: Callable) -> dict:
    """Erkennt Entitaeten in der belegten Erzeugerrepräsentation vor dem Seitenschnitt.

    Die Rueckgabe liegt in Originaloffsets der gespeicherten Chunks. Auch
    Fragmente einer ueber mehrere Chunks reichenden Entitaet werden maskiert.
    """
    if len(chunks) < 2:
        return {}
    source = ""
    bodies = []
    for chunk in chunks:
        meta = chunk["metadata"]
        start, end, body = (
            meta.get(k) for k in ("source_char_start", "source_char_end", "source_body_start")
        )
        gap = meta.get("source_gap_before")
        overlap = meta.get("source_overlap_before")
        if not (
            all(type(v) is int for v in (start, end, body, overlap))
            and 0 <= start < end
            and end >= len(source)
            and 0 <= body <= len(chunk["text"])
            and len(chunk["text"]) - body == end - start
            and isinstance(gap, str)
            and len(gap) == max(0, start - len(source))
            and overlap == max(0, len(source) - start)
        ):
            raise RetrievalError(
                "Originalnachbarschaft fuer Anonymisierung fehlt oder ist widerspruechlich. "
                "Dokument neu indizieren."
            )
        text = chunk["text"][body:]
        if overlap and source[start:] != text[:overlap]:
            raise RetrievalError("Widerspruechlicher Originaltext an einer Chunkgrenze.")
        source += gap + text[overlap:]
        bodies.append((start, end, body, meta["chunk_index"]))
    report = sanitize_with_report(source, track_offsets=True)
    prepared = prepare_text(report.clean_text)
    replacements = getattr(prepared, "replacements", ())
    if getattr(prepared, "protected_spans", ()) and not replacements:
        raise RetrievalError("Anonymisierung liefert keine pruefbaren Entitaetspositionen.")
    ranges = {}
    ends = [end for _, end, _, _ in bodies]
    for low, high, replacement in replacements:
        raw_low = report.source_offsets[low]
        raw_high = report.source_offsets[high] if high < len(report.source_offsets) else len(source)
        for start, end, body, index in bodies[bisect_left(ends, raw_low + 1) :]:
            if start >= raw_high:
                break
            ranges.setdefault(index, []).append(
                (
                    body + max(raw_low, start) - start,
                    body + min(raw_high, end) - start,
                    replacement,
                    (raw_low, raw_high),
                )
            )
    return ranges


def prepare_document_read(
    chunks: list[dict],
    *,
    request: dict,
    budget: int,
    source_numbers: dict[tuple, int],
    prepare_text: Callable[[str], str] = str,
    context_chunks: list[dict] | None = None,
) -> PreparedContext:
    """Erhaelt einen zusammenhaengenden Originalbereich mit exaktem Lesecursor.

    Bei Platzmangel endet die Seite innerhalb eines Chunks. Die Fortsetzung
    beginnt an derselben Zeichenposition; es gibt keine Relevanzauswahl.
    """
    selected: list[dict] = []
    input_chars = sum(len(c["text"]) for c in chunks)
    best = PreparedContext("", [], input_chars, True)
    # Einmal den ganzen Chunk bereinigen. Praefixe desselben Rohtexts koennen
    # unterschiedlich bereinigt werden (z.B. ein halbiertes Steuer-Tag).
    # Die Suche arbeitet deshalb auf sichtbaren Zeichen, Cursor bleiben roh.
    cleaned = []
    positions = {}
    protected = {}
    entity_ranges = _document_entity_ranges(context_chunks, prepare_text) if context_chunks else {}
    replacements = {}
    for chunk in chunks:
        report = sanitize_with_report(chunk["text"], track_offsets=True)
        if report.severity != "none":
            log_defense_event(
                "retrieval_context", report, document_source=chunk["metadata"].get("source")
            )
        index = chunk["metadata"]["chunk_index"]
        offset = request["start_offset"] if index == request["start_chunk"] else 0
        offsets = report.source_offsets
        start = bisect_left(offsets, offset)
        # Vor dem Schnitt den ganzen bereinigten Chunk erkennen lassen.
        # Auch eine Fortsetzung braucht den links vom Cursor liegenden Text.
        prepared = prepare_text(report.clean_text)
        spans = getattr(prepared, "protected_spans", ())
        replacements[index] = [
            (
                bisect_left(offsets, low) - start,
                bisect_left(offsets, high) - start,
                replacement,
                entity,
            )
            for low, high, replacement, entity in entity_ranges.get(index, ())
            if high > offset
        ]
        spans = sorted(
            set(spans) | {(low + start, high + start) for low, high, _, _ in replacements[index]}
        )
        merged_spans = []
        for low, high in spans:
            if merged_spans and low < merged_spans[-1][1]:
                merged_spans[-1] = merged_spans[-1][0], max(merged_spans[-1][1], high)
            else:
                merged_spans.append((low, high))
        spans = merged_spans
        if any(low < start < high for low, high in spans):
            raise RetrievalError(
                "Lesecursor liegt innerhalb einer anonymisierten Entitaet. "
                "Dokument ab dem Chunk-Anfang erneut lesen."
            )
        protected[index] = [(low - start, high - start) for low, high in spans if high > start]
        positions[index] = (offsets, start, len(chunk["text"]))
        cleaned.append(
            {**chunk, "text": report.clean_text[start:], "context_join": _chunk_join(chunk, report)}
        )

    def safe_boundary(index: int, length: int) -> int:
        for low, high in protected[index]:
            if low < length < high:
                return low
            if low >= length:
                break
        return length

    def raw_position(index: int, length: int) -> int:
        offsets, start, raw_length = positions[index]
        # Terminalwert nimmt auch entfernte Zeichen am Chunk-Ende mit.
        return offsets[start + length] if start + length < len(offsets) else raw_length

    def page(candidate: list[dict], end_offset: int) -> PreparedContext:
        last = chunks[len(candidate) - 1]
        end_chunk = last["metadata"]["chunk_index"]
        whole_chunk = end_offset == len(last["text"])
        next_chunk = end_chunk + 1 if whole_chunk else end_chunk
        next_offset = 0 if whole_chunk else end_offset
        has_more = next_chunk < request["total_chunks"]
        complete = request["start_chunk"] == 0 and request["start_offset"] == 0 and not has_more
        reading = {
            **request,
            "end_chunk": end_chunk,
            "end_offset": end_offset,
            "has_more": has_more,
            "next_chunk": next_chunk if has_more else None,
            "next_offset": next_offset if has_more else None,
            "complete_document": complete,
        }
        documents = _documents(candidate, preserve_whitespace=True, sanitized=True)
        if documents:
            doc = documents[0]
        else:
            # Ein vollstaendig entfernter Bereich braucht ebenfalls einen
            # Fortsetzungscursor. Er liefert keinen erfundenen Belegtext.
            meta = dict(candidate[0]["metadata"])
            meta["source"] = sanitize_with_report(meta.get("source", "unbekannt")).clean_text
            doc = _Document(metadata=meta)
        key = _context_source_key(doc.metadata)
        doc.metadata["context_source_id"] = source_numbers.setdefault(key, len(source_numbers) + 1)
        doc.metadata["document_read"] = reading
        doc.selected = set(range(len(doc.blocks)))
        excerpt = doc.excerpt()
        # Positionen beziehen sich auf den vollstaendig bereinigten Chunk,
        # nicht auf Originaloffsets oder die spaeter anonymisierte Darstellung.
        # Ueberlappende Seiten lassen sich so ohne Textheuristik vereinigen.
        origins = {index: (start, end) for start, end, index in doc.origins}
        read_spans = []
        for part in candidate:
            index = part["metadata"]["chunk_index"]
            offsets, start, _ = positions[index]
            low, high = origins[index] if part["text"] else (0, 0)
            read_spans.append(
                {
                    "chunk_index": index,
                    "clean_start": start,
                    "clean_end": start + len(part["text"]),
                    "clean_length": len(offsets),
                    "text_start": low,
                    "text_end": high,
                    "join_before": part["context_join"],
                }
            )
        excerpt["metadata"]["document_read_spans"] = read_spans
        for span in excerpt["metadata"]["context_spans"]:
            span["start"] = raw_position(span["chunk_index"], span["start"])
            span["end"] = raw_position(span["chunk_index"], span["end"])
        # Belegbereiche bleiben in den stabilen Koordinaten der einzeln
        # bereinigten Chunks. Die zusaetzliche Bereinigung nach dem Verbinden
        # wird fuer die Modellpositionen darauf abgebildet; Lesecursor und
        # ueberlappende Benchmark-Seiten behalten ihren Originalbezug.
        body_report = sanitize_with_report(excerpt["text"], track_offsets=True)
        rendered = _render([excerpt], False, body_reports=[body_report])
        prepared = prepare_text(rendered)
        # Auch ein Schnitt direkt vor einer Entitaet kann Erkennungskontext
        # entfernen (z.B. einen Regex-Lookbehind). Die fertige Darstellung muss
        # alle im ganzen Chunk erkannten, hier enthaltenen Bereiche abdecken.
        # Das letzte Vorkommen mit Folge-Newline ist der Text vor dem End-Tag.
        body_start = rendered.rfind(body_report.clean_text + "\n")

        def rendered_position(index: int, offset: int) -> int:
            return body_start + bisect_left(body_report.source_offsets, origins[index][0] + offset)

        rendered_spans = getattr(prepared, "protected_spans", ())
        masks = []
        for part in candidate:
            index = part["metadata"]["chunk_index"]
            for low, high, replacement, entity in replacements[index]:
                if low < len(part["text"]) and high > 0:
                    start = rendered_position(index, max(0, low))
                    end = rendered_position(index, min(high, len(part["text"])))
                    if start < end:
                        masks.append((start, end, replacement, entity))
        for part in candidate:
            index = part["metadata"]["chunk_index"]
            for low, high in protected[index]:
                if low >= len(part["text"]):
                    continue
                start = rendered_position(index, low)
                end = rendered_position(index, high)
                if start == end:
                    continue
                if body_start < 0 or not any(
                    a <= start and end <= b
                    for a, b in [*rendered_spans, *((a, b) for a, b, _, _ in masks)]
                ):
                    raise RetrievalError(
                        "Anonymisierung des Lesebereichs ist nach dem Seitenschnitt "
                        "nicht vollstaendig. Dokument mit groesserem Seitenbudget erneut lesen."
                    )
        if masks:
            merged = []
            for low, high, replacement, entity in sorted(set(masks)):
                if merged and (
                    low < merged[-1][1]
                    or (entity == merged[-1][3] and not rendered[merged[-1][1] : low].strip())
                ):
                    old_low, old_high, old_replacement, old_entity = merged[-1]
                    if replacement != old_replacement:
                        raise RetrievalError("Widerspruechliche Entitaeten an einer Chunkgrenze.")
                    merged[-1] = old_low, max(old_high, high), replacement, old_entity
                else:
                    merged.append((low, high, replacement, entity))
            masked = rendered
            for low, high, replacement, _ in reversed(merged):
                masked = masked[:low] + replacement + masked[high:]
            prepared = prepare_text(masked)
        return PreparedContext(prepared, [excerpt], input_chars, not complete)

    for chunk in cleaned:
        index = chunk["metadata"]["chunk_index"]
        text = chunk["text"]
        candidate = [*selected, {**chunk, "text": text}]
        full = page(candidate, raw_position(index, len(text)))
        if full.chunks and len(full.text) <= budget:
            selected = candidate
            best = full
            continue
        low, high = 0, len(text)
        fitting = best
        while low < high:
            probe = (low + high + 1) // 2
            length = safe_boundary(index, probe)
            if not length:
                # Eine Entitaet am Anfang kann als Pseudonym ins Budget passen.
                # Ihr leerer Praefix ist keine Budgetueberschreitung.
                low = probe
                continue
            candidate[-1] = {**chunk, "text": text[:length]}
            partial = page(candidate, raw_position(index, length))
            if partial.chunks and len(partial.text) <= budget:
                low = probe
                fitting = partial
            else:
                high = probe - 1
        low = safe_boundary(index, low)
        if low:
            best = fitting
            # Wenn moeglich an einer Zeilengrenze enden, ohne die Seite zu leeren.
            boundary = text.rfind("\n", low // 2, low)
            length = safe_boundary(index, boundary + 1 if boundary >= 0 else low)
            if not length:
                break
            candidate[-1] = {**chunk, "text": text[:length]}
            partial = page(candidate, raw_position(index, length))
            if partial.chunks and len(partial.text) <= budget:
                best = partial
        break
    return best


def prepare_context(
    chunks: list[dict],
    *,
    budget: int,
    query: str = "",
    max_sources: int = DEFAULT_MAX_CONTEXT_CHUNKS,
    source_diversity: bool = True,
    reranker=None,
    source_numbers: dict[tuple, int] | None = None,
    prepare_text: Callable[[str], str] = str,
) -> PreparedContext:
    """Waehlt Originalpassagen einschliesslich Darstellung innerhalb des Budgets.

    Der unverkuerzte Kontext hat Vorrang. Erst bei Budgetknappheit werden
    Absaetze nach Query-Ueberschneidung geordnet und mit ihren unmittelbaren
    Nachbarn aufgenommen. Uebergrosse Einheiten werden ganz ausgelassen.
    """
    documents = _documents(chunks)
    if source_numbers is None:
        source_numbers = {}
    input_chars = sum(len(doc.text) for doc in documents)
    for doc in documents:
        key = _context_source_key(doc.metadata)
        doc.metadata["context_source_id"] = source_numbers.setdefault(key, len(source_numbers) + 1)
        doc.selected = set(range(len(doc.blocks)))
    full_chunks = [doc.excerpt() for doc in documents]
    full_text = prepare_text(_render(full_chunks, False))
    if len(documents) <= max_sources and len(full_text) <= budget:
        return PreparedContext(full_text, full_chunks, input_chars, False)

    # Optionales Reranking ordnet ganze Dokumente. Es entfernt keine einzelnen
    # Belegzeilen und laeuft fuer alle Provider an derselben Stelle.
    empty_notice = prepare_text(_EMPTY_NOTICE)
    if budget <= len(empty_notice):
        return PreparedContext("", [], input_chars, bool(documents))
    if reranker is not None and documents:
        ranked = reranker.rerank(query, full_chunks, top_k=len(full_chunks))
        rank = {document_key(chunk): i for i, chunk in enumerate(ranked)}
        documents.sort(
            key=lambda doc: rank.get(document_key({"metadata": doc.metadata}), len(rank))
        )

    omitted_sources = len(documents) > max_sources
    all_documents = documents
    documents = documents[:max_sources]
    terms = {
        term for term in re.findall(r"\w+", query.casefold()) if len(term) >= 4 or term.isdigit()
    }
    candidates: list[list[set[int]]] = []
    for doc in documents:
        doc.selected.clear()
        source_share = max(1, budget // max(1, len(documents)))
        doc.split_long_paragraphs(source_share, prepare_text)
        options = []
        for index in range(len(doc.blocks)):
            # Abschnitte mit Nachbarn erhalten Bedingungen und Verneinungen.
            bundle = set(range(max(0, index - 1), min(len(doc.blocks), index + 2)))
            # Kurze aufeinanderfolgende Abschnitte gehoeren oft zu einem
            # gemeinsamen Feld-/Werteblock. Das Fenster darf sie einschliessen,
            # ohne einzelne Werte nach ihrem Inhalt zu bevorzugen.
            end = max(bundle)
            while end + 1 < len(doc.blocks):
                window = doc.text[doc.blocks[min(bundle)][0] : doc.blocks[end + 1][1]]
                if len(prepare_text(window)) > source_share // 2:
                    break
                end += 1
                bundle.add(end)
            # Die letzte Markdown-Ueberschrift bleibt auch bei spaeten Absaetzen sichtbar.
            for heading in range(index, -1, -1):
                if re.match(r"^\s*#{1,6}\s", doc.text[slice(*doc.blocks[heading])]):
                    bundle.add(heading)
                    break
            options.append(bundle)
        options.sort(
            key=lambda bundle: (
                -sum(
                    any(term in doc.text[slice(*doc.blocks[i])].casefold() for i in bundle)
                    for term in terms
                )
            )
        )
        candidates.append(options)

    def selected_context() -> tuple[str, list[dict], bool]:
        selected = [doc.excerpt() for doc in documents if doc.selected]
        omitted = omitted_sources or any(len(doc.selected) < len(doc.blocks) for doc in documents)
        return prepare_text(_render(selected, omitted)), selected, omitted

    # Jede Quelle erhaelt zunaechst eine Chance auf eine vollstaendige Passage.
    # Das Restbudget wird danach weiter verteilt, ohne festen Cap pro Quelle.
    queue = [
        (doc_index, bundle) for doc_index, options in enumerate(candidates) for bundle in options
    ]
    if source_diversity:
        queue = [
            (doc_index, options[round_index])
            for round_index in range(max(map(len, candidates), default=0))
            for doc_index, options in enumerate(candidates)
            if round_index < len(options)
        ]
    text, selected, omitted = selected_context()
    for doc_index, bundle in queue:
        doc = documents[doc_index]
        if bundle.issubset(doc.selected):
            continue
        before = set(doc.selected)
        doc.selected.update(bundle)
        candidate_text, candidate_chunks, candidate_omitted = selected_context()
        if len(candidate_text) > budget:
            doc.selected = before
        else:
            text, selected, omitted = candidate_text, candidate_chunks, candidate_omitted

    if not selected:
        text = empty_notice if len(empty_notice) <= budget else ""
    # Referenzen pro ganz ausgelassener Quelle erhalten, auch neben passenden
    # Auszuegen und jenseits des Quellenlimits. Sie enthalten keine Belegtexte.
    selected_keys = {document_key(chunk) for chunk in selected}
    references = []
    for doc in all_documents:
        if (
            doc.metadata.get("document_ref")
            and document_key({"metadata": doc.metadata}) not in selected_keys
        ):
            candidate = prepare_text(_render(selected, omitted, [*references, doc.metadata]))
            if len(candidate) <= budget:
                references.append(doc.metadata)
                text = candidate
    return PreparedContext(text, selected, input_chars, omitted, references)
