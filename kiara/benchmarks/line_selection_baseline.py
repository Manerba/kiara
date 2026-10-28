"""Historische Zeilenauswahl nur fuer den Benchmark, Stand d37c2854.

Unveraenderte Auswahlhelfer aus agent_chain.py vor #320. Kein Runtime-Pfad.
Der Benchmark zaehlt zusaetzlich Trennzeichen zum gemeinsamen Zeichenbudget.
"""

import re


def select_lines(chunks: list[dict], *, budget: int, query: str) -> tuple[str, list[dict]]:
    """Historische Auswahl unter einem gemeinsamen Ausgabebudget vergleichen.

    Rechnungsgruppierung ist wie im letzten Live-Stand deaktiviert. Trennzeichen
    zaehlen zum Budget. Dies ist eine isolierte Auswahl-Ablation, kein Replay
    des alten Agenten mit seinen weiteren Adapter-Kuerzungen.
    """
    if budget <= 0:
        return "", []
    candidates = _compact_source_diverse_chunks(
        chunks,
        text_budget=budget,
        query=query,
        invoice_number_grouping=False,
    )
    if not candidates:
        candidates = chunks
    selected, remaining = [], budget
    for chunk in candidates:
        separator = 2 if selected else 0
        if remaining <= separator:
            break
        copied, remaining = _copy_chunk_with_text_budget(chunk, remaining - separator)
        if copied is not None:
            selected.append(copied)
    return "\n\n".join(chunk["text"] for chunk in selected), selected


def _copy_chunk_with_text_budget(chunk: dict, text_budget: int) -> tuple[dict | None, int]:
    """Kopiert einen Retrieval-Chunk und kuerzt dessen Text auf das Restbudget."""
    copied = dict(chunk)
    text = str(copied.get("text") or "")
    if not text:
        copied["text"] = ""
        return copied, text_budget
    if len(text) <= text_budget:
        copied["text"] = text
        return copied, text_budget - len(text)
    if text_budget <= 0:
        return None, 0
    suffix = "..."
    if text_budget > len(suffix):
        copied["text"] = text[: text_budget - len(suffix)] + suffix
    else:
        copied["text"] = text[:text_budget]
    return copied, 0


_INVOICE_NUMBER_RE = re.compile(
    r"(?:Rechnung(?:snummer)?[:\s]+|Invoice(?:\s+No\.?)?[:\s]+|_)(\d{8,16})(?:\.pdf)?",
    re.IGNORECASE,
)


_MONEY_RE = re.compile(
    r"\b\d{1,3}(?:[.\s]\d{3})*,\d{2}\s*(?:EUR|\u20ac)\b|"
    r"\b\d+(?:\.\d{2})\s*(?:EUR|\u20ac)\b",
    re.IGNORECASE,
)


_SALIENT_LINE_RE = re.compile(
    r"rechnung|rechnungsdatum|invoice|zu zahlender betrag|zahlender betrag|"
    r"betrag|summe|gesamt|brutto|netto|steuer|faellig|fallig|fällig|"
    r"abbuch|abgebucht|leistungszeitraum|service|projekt|total|amount|due",
    re.IGNORECASE,
)


def _extract_invoice_number(text: str) -> str | None:
    match = _INVOICE_NUMBER_RE.search(text or "")
    return match.group(1) if match else None


def _chunk_source(chunk: dict) -> str:
    return str((chunk.get("metadata") or {}).get("source") or "")


def _chunk_index(chunk: dict) -> int:
    try:
        return int((chunk.get("metadata") or {}).get("chunk_index", 0))
    except (TypeError, ValueError):
        return 0


def _raw_chunk_group_key(
    chunk: dict,
    *,
    invoice_number_grouping: bool = True,
) -> tuple[str, str]:
    """Gruppiert Raw-Chunks nach Belegnummer, sonst nach Quelle.

    Viele Mail-/Attachment-Quellen liegen mehrfach vor (z.B. INBOX und
    normalisierter Cache-Pfad). Bei Rechnungen ist die Rechnungsnummer der
    stabilere Dedupe-Key als der Dateipfad.
    """
    source = _chunk_source(chunk)
    inv = None
    if invoice_number_grouping:
        inv = _extract_invoice_number(f"{source}\n{chunk.get('text') or ''}")
    if inv:
        return ("invoice", inv)
    return ("source", source or repr((chunk.get("metadata") or {})))


def _query_terms(query: str | None) -> set[str]:
    if not query:
        return set()
    return {
        token.lower()
        for token in re.findall(r"[A-Za-zÄÖÜäöüß0-9]{4,}", query)
        if token.lower() not in {"2025", "2026"}
    }


def _score_salient_line(line: str, query_terms: set[str]) -> int:
    lower = line.lower()
    score = 0
    if _MONEY_RE.search(line):
        score += 20
    if "zu zahlender betrag" in lower or "zahlender betrag" in lower:
        score += 18
    if "rechnung" in lower or "invoice" in lower:
        score += 10
    if "rechnungsdatum" in lower:
        score += 10
    if "summe" in lower or "gesamt" in lower or "total" in lower:
        score += 8
    if "brutto" in lower or "netto" in lower or "steuer" in lower:
        score += 6
    if "abbuch" in lower or "faellig" in lower or "fallig" in lower or "fällig" in lower:
        score += 5
    if "leistungszeitraum" in lower or "service" in lower or "projekt" in lower:
        score += 4
    if _SALIENT_LINE_RE.search(line):
        score += 3
    score += sum(1 for term in query_terms if term in lower)
    return score


def _compact_raw_chunk_group(
    group_key: tuple[str, str],
    chunks: list[dict],
    query: str | None,
    per_group_budget: int,
) -> dict:
    """Baut einen kompakten, faktenorientierten Kontext-Chunk pro Quelle/Beleg."""
    ordered = sorted(chunks, key=_chunk_index)
    sources: list[str] = []
    for chunk in ordered:
        source = _chunk_source(chunk)
        if source and source not in sources:
            sources.append(source)

    terms = _query_terms(query)
    candidates: list[tuple[int, int, str]] = []
    fallback_parts: list[str] = []
    seen_lines: set[str] = set()
    order = 0
    for chunk in ordered:
        text = str(chunk.get("text") or "")
        if text and len(" ".join(fallback_parts)) < per_group_budget:
            fallback_parts.append(text.strip())
        for raw_line in text.splitlines():
            line = " ".join(raw_line.split())
            if not line or line in seen_lines:
                continue
            seen_lines.add(line)
            score = _score_salient_line(line, terms)
            if score > 0:
                candidates.append((score, order, line))
            order += 1

    selected: list[tuple[int, str]] = []
    remaining = per_group_budget
    label = "Rechnung" if group_key[0] == "invoice" else "Quelle"
    heading = f"{label} {group_key[1]} - kompakter Auszug"
    selected.append((-1, heading))
    remaining -= len(heading)
    if sources:
        source = sources[0]
        if len(source) > 220:
            source = "..." + source[-217:]
        source_line = f"Quelle: {source}"
        selected.append((-1, source_line))
        remaining -= len(source_line)
        if len(sources) > 1:
            more_sources = f"Quellen dieser Gruppe: {len(sources)}"
            selected.append((-1, more_sources))
            remaining -= len(more_sources)

    for _score, line_order, line in sorted(candidates, key=lambda item: (-item[0], item[1])):
        cost = len(line) + 1
        if cost > remaining:
            continue
        selected.append((line_order, line))
        remaining -= cost
        if remaining <= 80:
            break

    if len(selected) <= 1:
        fallback = "\n".join(fallback_parts).strip()
        if fallback:
            selected.append((0, fallback[: max(remaining, 0)]))

    selected_lines = [line for _, line in sorted(selected, key=lambda item: item[0])]
    metadata = dict((ordered[0].get("metadata") or {}) if ordered else {})
    metadata["chunk_index"] = -1
    metadata["chunk_total"] = 1
    metadata["source_diverse_compact"] = True
    metadata["compact_group_type"] = group_key[0]
    metadata["compact_group_value"] = group_key[1]
    if group_key[0] == "invoice":
        metadata["invoice_number"] = group_key[1]
    return {
        "text": "\n".join(selected_lines).strip(),
        "metadata": metadata,
        "score": ordered[0].get("score", 0.0) if ordered else 0.0,
    }


def _compact_source_diverse_chunks(
    raw_chunks: list[dict],
    text_budget: int,
    query: str | None = None,
    invoice_number_grouping: bool = True,
) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    order: list[tuple[str, str]] = []
    for chunk in raw_chunks:
        if not isinstance(chunk, dict):
            continue
        key = _raw_chunk_group_key(
            chunk,
            invoice_number_grouping=invoice_number_grouping,
        )
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(chunk)

    if len(groups) < 2:
        return []

    group_count = len(groups)
    per_group_budget = min(1600, max(450, text_budget // max(group_count, 1)))
    compacted: list[dict] = []
    remaining = text_budget
    for key in order:
        chunk = _compact_raw_chunk_group(key, groups[key], query, per_group_budget)
        copied, remaining = _copy_chunk_with_text_budget(chunk, remaining)
        if copied is not None:
            compacted.append(copied)
        if remaining <= 0:
            break
    return compacted
