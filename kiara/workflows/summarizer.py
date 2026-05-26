"""Dokumenten-Zusammenfassung fuer KIara.

Unterstuetzt direkte Zusammenfassung kurzer Texte und Map-Reduce
fuer laengere Dokumente.
"""

import asyncio
import logging
import math

from kiara.shared.constants import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_CONTEXT_TOKENS,
    DEFAULT_SUMMARY_CHUNK_SIZE,
    EXCERPT_LENGTH,
)

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """Fasst Dokumente zusammen, wahlweise mit LLM (Map-Reduce).

    Args:
        config: KIara-Konfiguration. Relevante Schluessel:
                ``workflows.max_context_tokens`` (default 6000),
                ``workflows.summary_chunk_size`` (default 4000).
        llm: Optionales Backend-Objekt mit ``generate("", prompt)``-Methode.
    """

    def __init__(self, config: dict, llm=None) -> None:
        workflow_config = config.get("workflows", {})
        self._max_context = workflow_config.get("max_context_tokens", DEFAULT_MAX_CONTEXT_TOKENS)
        if self._max_context <= 0:
            self._max_context = DEFAULT_MAX_CONTEXT_TOKENS
        self._chunk_size = workflow_config.get("summary_chunk_size", DEFAULT_SUMMARY_CHUNK_SIZE)
        if self._chunk_size <= 0:
            self._chunk_size = DEFAULT_SUMMARY_CHUNK_SIZE
        self._llm = llm

    # ------------------------------------------------------------------
    # Oeffentliche API
    # ------------------------------------------------------------------

    async def summarize(
        self,
        text: str,
        focus: str | None = None,
        output_format: str = "paragraph",
        max_length: int = 500,
    ) -> str:
        """Fasst einen Text zusammen.

        Wenn der Text kuerzer als ``max_context`` ist, wird er direkt
        zusammengefasst. Andernfalls wird Map-Reduce eingesetzt.

        Args:
            text: Der zusammenzufassende Text.
            focus: Optionaler inhaltlicher Fokus.
            output_format: ``paragraph``, ``bullet_points`` oder ``structured``.
            max_length: Maximale Laenge der Zusammenfassung (Zeichen).

        Returns:
            Zusammenfassung als String.
        """
        if self._llm is None:
            # Ohne LLM: einfach kuerzen
            limit = min(max_length, self._chunk_size) if max_length else self._chunk_size
            truncated = text[:limit]
            if len(text) > limit:
                truncated += "..."
            return truncated

        if self._estimated_tokens(text) <= self._max_context:
            # Kurzer Text: direkt zusammenfassen
            return await self._direct_summarize(text, focus, output_format, max_length)

        # Langer Text: Map-Reduce
        chunks = self._split_for_map(text)
        partial_summaries = await self._map_phase(chunks, focus)
        return await self._reduce_phase(partial_summaries, focus, max_length)

    async def summarize_multiple(
        self,
        documents: list[dict],
        focus: str | None = None,
    ) -> str:
        """Fasst mehrere Dokumente zusammen mit Quellenangaben.

        Args:
            documents: Liste von Dicts mit ``title``, ``text``, ``source``.
            focus: Optionaler inhaltlicher Fokus.

        Returns:
            Zusammenfassung mit Quellenangaben als Markdown.
        """
        if not documents:
            return "(Keine Dokumente vorhanden)"

        if self._llm is None:
            # Ohne LLM: Titel und erste Zeilen jedes Dokuments
            lines = ["## Dokumentenuebersicht", ""]
            for doc in documents:
                title = doc.get("title", "Unbekannt")
                source = doc.get("source", "")
                text = doc.get("text", "")
                preview = text[:EXCERPT_LENGTH] + "..." if len(text) > EXCERPT_LENGTH else text
                lines.append(f"### {title}")
                if source:
                    lines.append(f"*Quelle: {source}*")
                lines.append("")
                lines.append(preview)
                lines.append("")
            return "\n".join(lines)

        # Mit LLM: Alle Dokumente aufbereiten und zusammenfassen
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get("title", f"Dokument {i}")
            source = doc.get("source", "")
            text = doc.get("text", "")
            header = f"[{title}]"
            if source:
                header += f" (Quelle: {source})"
            doc_texts.append(f"{header}\n{text}")

        combined = "\n\n---\n\n".join(doc_texts)

        # Wenn zu lang: Map-Reduce
        if self._estimated_tokens(combined) > self._max_context:
            chunks = self._split_for_map(combined)
            partial = await self._map_phase(chunks, focus)
            summary = await self._reduce_phase(partial, focus, max_length=800)
        else:
            summary = await self._direct_summarize(combined, focus, "paragraph", 800)

        # Quellenverzeichnis anhaengen
        sources_lines = ["\n\n### Quellen"]
        for doc in documents:
            title = doc.get("title", "Unbekannt")
            source = doc.get("source", "")
            sources_lines.append(f"- {title}" + (f" ({source})" if source else ""))

        return summary + "\n".join(sources_lines)

    # ------------------------------------------------------------------
    # Interne Methoden
    # ------------------------------------------------------------------

    def _split_for_map(self, text: str) -> list[str]:
        """Teilt Text in Chunks, bevorzugt an Absatzgrenzen.

        Args:
            text: Der zu teilende Text.

        Returns:
            Liste von Text-Chunks.
        """
        if len(text) <= self._chunk_size:
            return [text]

        chunks: list[str] = []
        remaining = text

        while remaining:
            if len(remaining) <= self._chunk_size:
                chunks.append(remaining)
                break

            # Versuche an Absatzgrenze zu trennen
            chunk_candidate = remaining[: self._chunk_size]
            last_para = chunk_candidate.rfind("\n\n")

            if last_para > self._chunk_size // 4:
                # Absatzgrenze gefunden und nicht zu weit vorne
                split_at = last_para + 2
            else:
                # Fallback: an letztem Zeilenumbruch trennen
                last_newline = chunk_candidate.rfind("\n")
                if last_newline > self._chunk_size // 4:
                    split_at = last_newline + 1
                else:
                    # Harter Schnitt
                    split_at = self._chunk_size

            chunks.append(remaining[:split_at].rstrip())
            remaining = remaining[split_at:].lstrip()

        return chunks

    def _estimated_tokens(self, text: str) -> int:
        """Schaetzt Token anhand der zentralen chars/token-Konstante."""
        if not text:
            return 0
        return math.ceil(len(text) / CHARS_PER_TOKEN)

    def _prompt_fits_context(self, prompt: str) -> bool:
        return self._estimated_tokens(prompt) <= self._max_context

    def _build_reduce_prompt(
        self,
        combined: str,
        focus: str | None,
        max_length: int,
    ) -> str:
        focus_hint = f"\nFokus: {focus}" if focus else ""
        return (
            f"Die folgenden Teil-Zusammenfassungen stammen aus einem laengeren Dokument. "
            f"Erstelle daraus eine kohaerente Gesamt-Zusammenfassung "
            f"(maximal {max_length} Zeichen).{focus_hint}\n\n"
            f"{combined}\n\n"
            f"Gesamt-Zusammenfassung:"
        )

    def _reduce_body_budget_chars(
        self,
        focus: str | None,
        max_length: int,
    ) -> int:
        context_budget_chars = int(self._max_context * CHARS_PER_TOKEN)
        prompt_overhead = len(self._build_reduce_prompt("", focus, max_length))
        return max(1, context_budget_chars - prompt_overhead)

    @staticmethod
    def _format_reduce_entries(summaries: list[str]) -> list[str]:
        return [f"Teil {i}: {s}" for i, s in enumerate(summaries, 1)]

    @staticmethod
    def _truncate_to_budget(text: str, budget_chars: int) -> str:
        if len(text) <= budget_chars:
            return text
        if budget_chars <= 3:
            return text[:budget_chars]
        truncated = text[: budget_chars - 3].rstrip()
        boundary = max(truncated.rfind("\n\nTeil "), truncated.rfind("\n"))
        if boundary > budget_chars // 2:
            truncated = truncated[:boundary].rstrip()
        return truncated + "..."

    def _build_reduce_batches(
        self,
        summaries: list[str],
        focus: str | None,
        max_length: int,
    ) -> list[str]:
        entries = self._format_reduce_entries(summaries)
        body_budget = self._reduce_body_budget_chars(focus, max_length)
        batches: list[str] = []
        current: list[str] = []

        for entry in entries:
            entry = self._truncate_to_budget(entry, body_budget)
            candidate = "\n\n".join([*current, entry])
            if current and len(candidate) > body_budget:
                batches.append("\n\n".join(current))
                current = [entry]
            else:
                current.append(entry)

        if current:
            batches.append("\n\n".join(current))
        return batches

    def _generate_reduce_body(
        self,
        combined: str,
        focus: str | None,
        max_length: int,
    ) -> str:
        prompt = self._build_reduce_prompt(combined, focus, max_length)
        try:
            result = self._llm.generate("", prompt)
            return result.strip()
        except Exception as exc:
            logger.warning("Reduce-Phase fehlgeschlagen: %s", exc)
            fallback = combined
            if len(fallback) > max_length:
                fallback = fallback[:max_length]
                last_nl = fallback.rfind("\n")
                if last_nl > 0:
                    fallback = fallback[:last_nl]
                return fallback + "..."
            return fallback

    async def _map_phase(self, chunks: list[str], focus: str | None = None) -> list[str]:
        """Map-Phase: Jeden Chunk einzeln zusammenfassen.

        Args:
            chunks: Liste von Text-Chunks.
            focus: Optionaler inhaltlicher Fokus.

        Returns:
            Liste von Teil-Zusammenfassungen.
        """
        summaries = []
        focus_hint = f"\nFokus: {focus}" if focus else ""

        for i, chunk in enumerate(chunks, 1):
            prompt = (
                f"Fasse den folgenden Textabschnitt ({i}/{len(chunks)}) "
                f"kurz und praezise zusammen.{focus_hint}\n\n"
                f"Text:\n{chunk}\n\n"
                f"Zusammenfassung:"
            )
            try:
                result = self._llm.generate("", prompt)
                summaries.append(result.strip())
            except Exception as exc:
                logger.warning("Map-Phase Chunk %d fehlgeschlagen: %s", i, exc)
                # Fallback: ersten Teil des Chunks
                summaries.append(chunk[:500] + "...")

        return summaries

    async def _reduce_phase(
        self,
        summaries: list[str],
        focus: str | None = None,
        max_length: int = 500,
        _depth: int = 0,
    ) -> str:
        """Reduce-Phase: Teil-Zusammenfassungen verdichten.

        Args:
            summaries: Liste von Teil-Zusammenfassungen.
            focus: Optionaler inhaltlicher Fokus.
            max_length: Maximale Laenge der Endzusammenfassung.

        Returns:
            Verdichtete Gesamt-Zusammenfassung.
        """
        combined = "\n\n".join(self._format_reduce_entries(summaries))
        prompt = self._build_reduce_prompt(combined, focus, max_length)

        if not self._prompt_fits_context(prompt):
            if _depth < 4:
                batches = self._build_reduce_batches(summaries, focus, max_length)
                if len(batches) > 1:
                    partial = await asyncio.gather(
                        *[
                            asyncio.to_thread(self._generate_reduce_body, batch, focus, max_length)
                            for batch in batches
                        ]
                    )
                    return await self._reduce_phase(partial, focus, max_length, _depth=_depth + 1)

            body_budget = self._reduce_body_budget_chars(focus, max_length)
            combined = self._truncate_to_budget(combined, body_budget)
            prompt = self._build_reduce_prompt(combined, focus, max_length)

        try:
            result = await asyncio.to_thread(self._llm.generate, "", prompt)
            return result.strip()
        except Exception as exc:
            logger.warning("Reduce-Phase fehlgeschlagen: %s", exc)
            # Fallback: Teile zusammenkleben
            combined = "\n\n".join(summaries)
            if len(combined) > max_length:
                truncated = combined[:max_length]
                last_nl = truncated.rfind("\n")
                if last_nl > 0:
                    truncated = truncated[:last_nl]
                return truncated + "..."
            return combined

    async def _direct_summarize(
        self,
        text: str,
        focus: str | None = None,
        output_format: str = "paragraph",
        max_length: int = 500,
    ) -> str:
        """Direkte Zusammenfassung eines kurzen Textes.

        Args:
            text: Zu fassender Text.
            focus: Optionaler Fokus.
            output_format: Gewuenschtes Ausgabeformat.
            max_length: Maximale Laenge.

        Returns:
            Zusammenfassung.
        """
        format_hints = {
            "paragraph": "als zusammenhaengenden Absatz",
            "bullet_points": "als Aufzaehlung mit Stichpunkten (- )",
            "structured": "strukturiert mit Ueberschriften und Stichpunkten",
        }
        format_hint = format_hints.get(output_format, format_hints["paragraph"])
        focus_hint = f"\nFokus: {focus}" if focus else ""

        prompt = (
            f"Fasse den folgenden Text {format_hint} zusammen "
            f"(maximal {max_length} Zeichen).{focus_hint}\n\n"
            f"Text:\n{text}\n\n"
            f"Zusammenfassung:"
        )

        try:
            result = self._llm.generate("", prompt)
            return result.strip()
        except Exception as exc:
            logger.warning("Direkte Zusammenfassung fehlgeschlagen: %s", exc)
            truncated = text[:max_length]
            if len(text) > max_length:
                truncated += "..."
            return truncated
