"""Contextual Retrieval: LLM-basierte Chunk-Anreicherung.

Generiert pro Chunk einen Kontext-Prefix, der den Chunk im Gesamtdokument
verortet. Der Prefix wird dem Chunk-Text vorangestellt, sodass sowohl
Vektor-Embeddings als auch BM25-Index profitieren.

Referenz: Anthropic "Contextual Retrieval" (2024)
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from collections import defaultdict
from collections.abc import Callable, Mapping

from kiara.shared.constants import (
    ENRICHMENT_HEADER_CHARS,
    ENRICHMENT_MAX_CONTEXT_CHARS,
    ENRICHMENT_MAX_PREFIX_CHARS,
    ENRICHMENT_MIN_CHUNK_CHARS,
    ENRICHMENT_NEIGHBOR_CHARS,
)

logger = logging.getLogger(__name__)

# --- Bekannte Platzhalter (fuer optionale Python-Validatoren / Services) ---

KNOWN_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "document_context",
        "chunk_text",
        "file_type",
        "source",
        "datasource_name",
    }
)
"""Definiert die bekannten Platzhalter-Namen fuer Enrichment-Templates.
Kann spaeter von Python-seitigen Admin-Validatoren oder Services wiederverwendet werden.
Dieses Sprint-TODO aendert solche Aufrufer selbst nicht.
Nicht intern im Modul verwendet."""

_CURRENT_CHUNK_MARKER = "\n[--- AKTUELLER ABSCHNITT (siehe unten) ---]\n"
_ELLIPSIS = "\n[...]\n"


def _build_document_context(
    full_text: str,
    chunk_text: str,
    chunk_index: int,
    all_chunk_texts: list[str],
    max_chars: int = ENRICHMENT_MAX_CONTEXT_CHARS,
    header_chars: int = ENRICHMENT_HEADER_CHARS,
    neighbor_chars: int = ENRICHMENT_NEIGHBOR_CHARS,
) -> str:
    """Baut kompakten Dokumentkontext fuer das LLM auf.

    Strategie (Prioritaetsreihenfolge):
    1. Dokument-Header (erste ``header_chars`` Zeichen) — Titel, Zusammenfassung
    2. Vorheriger Chunk (lokaler Kontext, max ``neighbor_chars``)
    3. Markierung des aktuellen Chunks (Pflicht-Anker, darf bei Truncation nicht verloren gehen)
    4. Naechster Chunk (lokaler Kontext, max ``neighbor_chars``)
    5. Truncation auf ``max_chars`` wenn noetig

    Boundary-Handling:
    - ``chunk_index == 0``: Kein Vorgaenger-Nachbar
    - ``chunk_index == len(all_chunk_texts) - 1``: Kein Nachfolger-Nachbar
    - ``len(all_chunk_texts) == 1``: Nur Header, keine Nachbarn

    Args:
        full_text:       Vollstaendiger Dokumenttext.
        chunk_text:      Text des zu enrichenden Chunks.
        chunk_index:     0-basierter Index des Chunks.
                         Muss ``0 <= chunk_index < len(all_chunk_texts)`` sein.
        all_chunk_texts: Alle Chunk-Texte des Dokuments (fuer Nachbar-Kontext).
        max_chars:       Max. Gesamtlaenge des Kontexts.
        header_chars:    Zeichen vom Dokumentanfang.
        neighbor_chars:  Max. Zeichen pro Nachbar-Chunk.

    ``chunk_text`` wird absichtlich NICHT in den Kontext kopiert, weil der
    eigentliche Chunk separat ueber ``{chunk_text}`` in den Prompt eingesetzt wird.
    Die Funktion markiert nur dessen Position im Gesamtdokument.

    Returns:
        Kontext-String fuer den LLM-Prompt (max ``max_chars`` Zeichen).
    """
    if max_chars <= 0:
        return ""
    if not all_chunk_texts:
        return full_text[:max_chars]
    if not 0 <= chunk_index < len(all_chunk_texts):
        raise IndexError(f"chunk_index {chunk_index} ausserhalb 0..{len(all_chunk_texts) - 1}")
    if max_chars <= len(_CURRENT_CHUNK_MARKER):
        return _CURRENT_CHUNK_MARKER[:max_chars]

    parts: list[str] = []
    remaining = max_chars - len(_CURRENT_CHUNK_MARKER)

    def _append_with_budget(text: str) -> None:
        nonlocal remaining
        if not text or remaining <= 0:
            return
        if len(text) <= remaining:
            parts.append(text)
            remaining -= len(text)
            return
        if remaining <= len(_ELLIPSIS):
            parts.append(text[:remaining])
        else:
            parts.append(text[: remaining - len(_ELLIPSIS)] + _ELLIPSIS)
        remaining = 0

    header = full_text[:header_chars]
    _append_with_budget(header)

    if chunk_index > 0:
        prev_chunk = all_chunk_texts[chunk_index - 1][:neighbor_chars]
        _append_with_budget(f"{_ELLIPSIS}{prev_chunk}")

    parts.append(_CURRENT_CHUNK_MARKER)

    if chunk_index < len(all_chunk_texts) - 1:
        next_chunk = all_chunk_texts[chunk_index + 1][:neighbor_chars]
        _append_with_budget(f"{next_chunk}{_ELLIPSIS}")

    return "".join(parts)


def _render_prompt(
    template_content: str,
    document_context: str,
    chunk_text: str,
    file_type: str = "",
    source: str = "",
    datasource_name: str = "",
) -> str:
    """Substituiert Platzhalter im Enrichment-Template.

    Verwendet ``str.format_map()`` mit ``defaultdict(str)`` fuer fehlende Keys
    (fehlende oder unbekannte Platzhalter werden durch leere Strings ersetzt).

    Args:
        template_content: Prompt-Template aus ``prompt_templates.content``.
        document_context: Ergebnis von ``_build_document_context()``.
        chunk_text:       Aktueller Chunk-Text.
        file_type:        Dateityp (z.B. ".pdf"). Optional.
        source:           Dateipfad. Optional.
        datasource_name:  Name der Datenquelle. Optional.

    Returns:
        Fertig substituierter Prompt-String.

    Hinweis:
        Einfache unbekannte Platzhalter werden via ``defaultdict(str)`` zu ``""``.
        Malforme Format-Strings oder Ausdruecke wie ``{foo.bar}`` koennen weiterhin
        Exceptions ausloesen und werden bewusst nicht hier verschluckt.
    """
    values = defaultdict(
        str,
        {
            "document_context": document_context,
            "chunk_text": chunk_text,
            "file_type": file_type,
            "source": source,
            "datasource_name": datasource_name,
        },
    )
    return template_content.format_map(values)


def _invoke_progress_callback(
    progress_callback: Callable[[int, int], None] | None,
    current: int,
    total: int,
) -> None:
    """Fuehrt den optionalen Progress-Callback robust aus.

    Callback-Fehler sollen Enrichment nicht abbrechen, weil der Callback nur
    Beobachtungs-/UI-Zwecken dient. Fehler werden daher geloggt und ignoriert.
    """
    if progress_callback is None:
        return
    try:
        progress_callback(current, total)
    except Exception as cb_exc:
        logger.warning(
            "progress_callback fehlgeschlagen bei Chunk %d/%d: %s",
            current,
            total,
            cb_exc,
        )


def _normalize_metadata_value(metadata: Mapping[str, object], key: str) -> str:
    """Liest optionale Metadaten robust aus.

    Fehlerhafte Mapping-Implementierungen oder problematische Werte duerfen den
    Batch nicht schon vor dem eigentlichen Chunk-Loop abbrechen.
    """
    try:
        raw_value = metadata.get(key, "")
    except Exception as meta_exc:
        logger.warning("file_metadata.get(%r) fehlgeschlagen: %s", key, meta_exc)
        return ""

    try:
        return str(raw_value or "")
    except Exception as value_exc:
        logger.warning(
            "file_metadata[%r] liess sich nicht in str umwandeln: %s",
            key,
            value_exc,
        )
        return ""


def enrich_chunk(
    chunk_text: str,
    chunk_index: int,
    full_text: str,
    all_chunk_texts: list[str],
    llm_generate_fn: Callable[[str, str], str],
    prompt_template: str,
    max_context_chars: int = ENRICHMENT_MAX_CONTEXT_CHARS,
    file_type: str = "",
    source: str = "",
    datasource_name: str = "",
) -> str:
    """Reichert einen einzelnen Chunk mit LLM-generiertem Kontext-Prefix an.

    **Wichtig**: Diese Funktion faengt KEINE Exceptions ab. Der Aufrufer
    (``enrich_chunks_batch``) ist fuer Exception-Handling und Skip-Logik
    verantwortlich. Dadurch kann die Batch-Funktion Stats korrekt zaehlen.

    Ablauf:
    1. Dokumentkontext bauen via ``_build_document_context()``
    2. Prompt rendern via ``_render_prompt()``
    3. LLM aufrufen: ``llm_generate_fn("", rendered_prompt) -> prefix``
    4. Prefix validieren: nicht leer, max ``ENRICHMENT_MAX_PREFIX_CHARS``
    5. Ergebnis: ``"<prefix>\\n\\n<original_text>"``

    Args:
        chunk_text:      Originaltext des Chunks.
        chunk_index:     0-basierter Index.
        full_text:       Vollstaendiger Dokumenttext.
        all_chunk_texts: Alle Chunk-Texte des Dokuments.
        llm_generate_fn: ``Callable[[str, str], str]``.
                         Kompatibel mit ``LLMBackend.generate(system_prompt, user_message)``.
        prompt_template: Prompt-Template-Content (aus DB).
        max_context_chars: Optionaler Override fuer ``_build_document_context()``.
                           Wird in Sprint 16.3 ueber ``config.enrichment.max_context_chars``
                           von der Pipeline durchgereicht.
        file_type:       Dateityp (optional).
        source:          Dateipfad (optional).
        datasource_name: Datenquellenname (optional).

    Returns:
        Angereicherter Text: ``"<prefix>\\n\\n<original_text>"``.
        Bei leerem LLM-Output: Original-Text unveraendert.

    Raises:
        Exception: LLM-/Render-Fehler (ConnectionError, RuntimeError, ValueError,
                   TypeError etc.) werden NICHT gefangen — Aufrufer muss try/except verwenden.
    """
    document_context = _build_document_context(
        full_text=full_text,
        chunk_text=chunk_text,
        chunk_index=chunk_index,
        all_chunk_texts=all_chunk_texts,
        max_chars=max_context_chars,
    )
    rendered_prompt = _render_prompt(
        template_content=prompt_template,
        document_context=document_context,
        chunk_text=chunk_text,
        file_type=file_type,
        source=source,
        datasource_name=datasource_name,
    )

    prefix_raw = llm_generate_fn("", rendered_prompt)
    if not isinstance(prefix_raw, str):
        raise TypeError(f"llm_generate_fn muss str liefern, erhielt {type(prefix_raw).__name__}")

    prefix = prefix_raw.strip()
    if not prefix:
        return chunk_text

    if len(prefix) > ENRICHMENT_MAX_PREFIX_CHARS:
        logger.warning(
            "Enrichment-Prefix gekuerzt fuer Chunk %d (von %d auf %d Zeichen)",
            chunk_index + 1,
            len(prefix),
            ENRICHMENT_MAX_PREFIX_CHARS,
        )
        prefix = prefix[:ENRICHMENT_MAX_PREFIX_CHARS].rstrip()
        if not prefix:
            return chunk_text

    return f"{prefix}\n\n{chunk_text}"


def enrich_chunks_batch(
    chunk_texts: list[str],
    full_text: str,
    llm_generate_fn: Callable[[str, str], str],
    prompt_template: str,
    max_context_chars: int = ENRICHMENT_MAX_CONTEXT_CHARS,
    file_metadata: Mapping[str, object] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_workers: int = 2,
    llm_generate_factory: Callable[[], Callable[[str, str], str]] | None = None,
    allow_parallel: bool = False,
) -> tuple[list[str], dict]:
    """Batch-Enrichment mit sicherer, opt-in Parallelisierung.

    Verantwortlichkeiten:
    - Skip-Pruefung: ``len(chunk_text) < ENRICHMENT_MIN_CHUNK_CHARS``
    - Exception-Handling: ``except Exception`` → Original-Text + Warning-Log
    - Stats-Zaehlung: enriched / skipped / unchanged / errors / enrichment_time_ms
    - Optionalen ``progress_callback`` robust aufrufen
    - Parallelisierung nur wenn allow_parallel=True, Factory vorhanden,
      max_workers>1 und len(chunk_texts)>3

    Returns:
        Tuple ``(enriched_texts, stats_dict)``.
    """
    stats = {"enriched": 0, "skipped": 0, "unchanged": 0, "errors": 0, "enrichment_time_ms": 0}
    if not chunk_texts:
        return [], stats

    t0 = time.monotonic()

    if file_metadata is None:
        metadata: Mapping[str, object] = {}
    elif isinstance(file_metadata, Mapping):
        metadata = file_metadata
    else:
        logger.warning(
            "file_metadata hat unerwarteten Typ %s; optionale Platzhalter bleiben leer",
            type(file_metadata).__name__,
        )
        metadata = {}

    file_type = _normalize_metadata_value(metadata, "file_type")
    source = _normalize_metadata_value(metadata, "source")
    datasource_name = _normalize_metadata_value(metadata, "datasource_name")

    common_kwargs = dict(
        full_text=full_text,
        all_chunk_texts=chunk_texts,
        prompt_template=prompt_template,
        max_context_chars=max_context_chars,
        file_type=file_type,
        source=source,
        datasource_name=datasource_name,
    )

    use_parallel = (
        allow_parallel
        and llm_generate_factory is not None
        and max_workers > 1
        and len(chunk_texts) > 3
    )

    if use_parallel:
        results, stats = _enrich_parallel(
            chunk_texts,
            llm_generate_fn,
            llm_generate_factory,
            common_kwargs,
            stats,
            progress_callback,
            max_workers,
        )
    else:
        results, stats = _enrich_sequential(
            chunk_texts,
            llm_generate_fn,
            common_kwargs,
            stats,
            progress_callback,
        )

    stats["enrichment_time_ms"] = int((time.monotonic() - t0) * 1000)
    return results, stats


def _enrich_sequential(chunk_texts, llm_generate_fn, common_kwargs, stats, progress_callback):
    """Sequenzielle Verarbeitung (Standard-Pfad)."""
    total = len(chunk_texts)
    results: list[str] = []

    for i, chunk_text in enumerate(chunk_texts):
        try:
            if len(chunk_text) < ENRICHMENT_MIN_CHUNK_CHARS:
                results.append(chunk_text)
                stats["skipped"] += 1
            else:
                enriched_text = enrich_chunk(
                    chunk_text=chunk_text,
                    chunk_index=i,
                    llm_generate_fn=llm_generate_fn,
                    **common_kwargs,
                )
                results.append(enriched_text)
                if enriched_text != chunk_text:
                    stats["enriched"] += 1
                else:
                    stats["unchanged"] += 1
        except Exception as exc:
            logger.warning("Enrichment fehlgeschlagen fuer Chunk %d/%d: %s", i + 1, total, exc)
            results.append(chunk_text)
            stats["errors"] += 1

        _invoke_progress_callback(progress_callback, i + 1, total)

    return results, stats


def _enrich_parallel(
    chunk_texts,
    llm_generate_fn,
    llm_generate_factory,
    common_kwargs,
    stats,
    progress_callback,
    max_workers,
):
    """Isolierte Parallelisierung — jede Task hat eigene Backend-Instanz."""
    total = len(chunk_texts)
    results: list[str | None] = [None] * total

    def _process_chunk(idx, chunk_text):
        if len(chunk_text) < ENRICHMENT_MIN_CHUNK_CHARS:
            return idx, chunk_text, "skipped"
        try:
            gen_fn = llm_generate_factory()
            enriched = enrich_chunk(
                chunk_text=chunk_text,
                chunk_index=idx,
                llm_generate_fn=gen_fn,
                **common_kwargs,
            )
            if enriched != chunk_text:
                return idx, enriched, "enriched"
            return idx, enriched, "unchanged"
        except Exception as exc:
            logger.warning("Enrichment fehlgeschlagen fuer Chunk %d/%d: %s", idx + 1, total, exc)
            return idx, chunk_text, "errors"

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, chunk_text in enumerate(chunk_texts):
            futures.append(executor.submit(_process_chunk, i, chunk_text))

        # Geordnete Ergebnis-Sammlung (Input-Reihenfolge)
        done_count = 0
        for future in concurrent.futures.as_completed(futures):
            idx, text, status = future.result()
            results[idx] = text
            stats[status] = stats.get(status, 0) + 1
            done_count += 1
            _invoke_progress_callback(progress_callback, done_count, total)

    # Falls Futures irgendwie None liessen (sollte nicht passieren)
    for i in range(total):
        if results[i] is None:
            results[i] = chunk_texts[i]
            stats["errors"] += 1

    return results, stats
