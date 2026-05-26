"""Tests fuer den DocumentSummarizer (kiara.workflows.summarizer)."""

import pytest

from kiara.shared.constants import CHARS_PER_TOKEN
from kiara.workflows.summarizer import DocumentSummarizer

# ---------------------------------------------------------------------------
# Mock-LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Simuliert ein LLM mit fester Antwort."""

    def __init__(self, response: str = "Mock-Zusammenfassung") -> None:
        self._response = response
        self.call_count: int = 0
        self.last_prompt: str | None = None
        self.all_prompts: list[str] = []

    def generate(self, system_prompt: str, user_message: str) -> str:
        self.call_count += 1
        self.last_prompt = user_message
        self.all_prompts.append(user_message)
        return self._response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    return {
        "workflows": {
            "max_context_tokens": 200,
            "summary_chunk_size": 100,
        },
    }


@pytest.fixture
def config_large():
    """Config mit groesseren Limits fuer bestimmte Tests."""
    return {
        "workflows": {
            "max_context_tokens": 6000,
            "summary_chunk_size": 4000,
        },
    }


@pytest.fixture
def summarizer_no_llm(config):
    return DocumentSummarizer(config, llm=None)


@pytest.fixture
def mock_llm():
    return MockLLM("Dies ist eine Mock-Zusammenfassung des Dokuments.")


@pytest.fixture
def summarizer_with_llm(config, mock_llm):
    return DocumentSummarizer(config, llm=mock_llm)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSummarizeWithoutLLM:
    @pytest.mark.asyncio
    async def test_summarize_short_doc_without_llm(self, summarizer_no_llm):
        """Ohne LLM wird kurzer Text direkt zurueckgegeben."""
        text = "Ein kurzer Absatz."
        result = await summarizer_no_llm.summarize(text)
        assert result == text

    @pytest.mark.asyncio
    async def test_summarize_long_doc_without_llm(self, summarizer_no_llm):
        """Ohne LLM wird langer Text gekuerzt mit Ellipsis."""
        text = "Wort " * 500  # Deutlich laenger als chunk_size=100
        result = await summarizer_no_llm.summarize(text)
        assert result.endswith("...")
        assert len(result) <= 110  # chunk_size + "..."


class TestSummarizeWithLLM:
    @pytest.mark.asyncio
    async def test_summarize_short_doc_with_llm(self, summarizer_with_llm, mock_llm):
        """Mit LLM: kurzer Text wird direkt zusammengefasst (kein Map-Reduce)."""
        text = "Ein einfacher Satz fuer die Zusammenfassung."
        result = await summarizer_with_llm.summarize(text)
        assert "Mock-Zusammenfassung" in result
        # Nur ein LLM-Aufruf (direkte Zusammenfassung)
        assert mock_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_with_focus(self, summarizer_with_llm, mock_llm):
        """Fokus-Parameter wird an den Prompt weitergegeben."""
        text = "Ein kurzer Text."
        await summarizer_with_llm.summarize(text, focus="Sicherheitsaspekte")
        assert mock_llm.last_prompt is not None
        assert "Sicherheitsaspekte" in mock_llm.last_prompt


class TestSplitForMap:
    def test_split_for_map(self, config):
        """Text wird korrekt in Chunks aufgeteilt."""
        summarizer = DocumentSummarizer(config, llm=None)
        # chunk_size=100, erstelle Text laenger als 100 Zeichen
        text = "Absatz eins.\n\n" + "x" * 80 + "\n\nAbsatz drei hat mehr Text."
        chunks = summarizer._split_for_map(text)
        assert len(chunks) >= 2
        # Alle Chunks zusammen enthalten den gesamten Text (ohne Whitespace)
        combined = "".join(chunks)
        assert "Absatz eins" in combined

    def test_split_for_map_short_text(self, config):
        """Kurzer Text wird nicht aufgeteilt."""
        summarizer = DocumentSummarizer(config, llm=None)
        text = "Kurz."
        chunks = summarizer._split_for_map(text)
        assert len(chunks) == 1
        assert chunks[0] == text


class TestMapReduceFlow:
    @pytest.mark.asyncio
    async def test_map_reduce_flow(self, config, mock_llm):
        """Langer Text triggert Map-Reduce (mehrere LLM-Aufrufe)."""
        summarizer = DocumentSummarizer(config, llm=mock_llm)
        # max_context=200, text laenger als 200 -> Map-Reduce
        text = "Absatz mit viel Text. " * 50  # ca. 1100 Zeichen
        result = await summarizer.summarize(text)
        assert "Mock-Zusammenfassung" in result
        # Map: mehrere Chunks + Reduce: 1 Aufruf -> mehr als 1 Aufruf
        assert mock_llm.call_count > 1

    @pytest.mark.asyncio
    async def test_map_phase_calls_per_chunk(self, config):
        """Map-Phase ruft LLM fuer jeden Chunk auf."""
        llm = MockLLM("Chunk-Zusammenfassung")
        summarizer = DocumentSummarizer(config, llm=llm)

        chunks = ["Chunk 1 Text", "Chunk 2 Text", "Chunk 3 Text"]
        results = await summarizer._map_phase(chunks)
        assert len(results) == 3
        assert llm.call_count == 3
        assert all("Chunk-Zusammenfassung" in r for r in results)

    @pytest.mark.asyncio
    async def test_reduce_phase_batches_to_context_budget(self):
        """Reduce-Prompts bleiben innerhalb von max_context_tokens."""
        llm = MockLLM("Verdichtete Zusammenfassung")
        summarizer = DocumentSummarizer(
            {
                "workflows": {
                    "max_context_tokens": 80,
                    "summary_chunk_size": 100,
                },
            },
            llm=llm,
        )
        summaries = [("Detail " * 25).strip() for _ in range(20)]

        result = await summarizer._reduce_phase(summaries, max_length=200)

        assert result == "Verdichtete Zusammenfassung"
        assert llm.call_count > 1
        max_chars = int(80 * CHARS_PER_TOKEN)
        assert all(len(prompt) <= max_chars for prompt in llm.all_prompts)


class TestSummarizeMultiple:
    @pytest.mark.asyncio
    async def test_summarize_multiple(self, config_large):
        """Mehrere Dokumente werden zusammengefasst mit Quellenangaben."""
        llm = MockLLM("Gesamtzusammenfassung der Dokumente.")
        summarizer = DocumentSummarizer(config_large, llm=llm)

        documents = [
            {"title": "Handbuch", "text": "Inhalt des Handbuchs.", "source": "/docs/handbuch.pdf"},
            {"title": "FAQ", "text": "Haeufig gestellte Fragen.", "source": "/docs/faq.md"},
        ]
        result = await summarizer.summarize_multiple(documents)
        assert "Gesamtzusammenfassung" in result
        assert "Quellen" in result
        assert "Handbuch" in result
        assert "FAQ" in result

    @pytest.mark.asyncio
    async def test_summarize_multiple_without_llm(self, config_large):
        """Ohne LLM: Dokumentuebersicht mit Vorschau."""
        summarizer = DocumentSummarizer(config_large, llm=None)

        documents = [
            {"title": "Dok A", "text": "Inhalt A.", "source": "a.txt"},
            {"title": "Dok B", "text": "Inhalt B.", "source": "b.txt"},
        ]
        result = await summarizer.summarize_multiple(documents)
        assert "Dok A" in result
        assert "Dok B" in result
        assert "Inhalt A" in result

    @pytest.mark.asyncio
    async def test_summarize_multiple_empty(self, config_large):
        """Leere Dokumentliste gibt Hinweis zurueck."""
        summarizer = DocumentSummarizer(config_large, llm=None)
        result = await summarizer.summarize_multiple([])
        assert "Keine Dokumente" in result
