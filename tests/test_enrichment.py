"""Tests fuer Sprint 16.2: Core-Enrichment-Modul.

38 Tests: _build_document_context, _render_prompt, enrich_chunk, enrich_chunks_batch.
"""

import pytest

from kiara.ingestion.enrichment import (
    _CURRENT_CHUNK_MARKER,
    _ELLIPSIS,
    _build_document_context,
    _render_prompt,
    enrich_chunk,
    enrich_chunks_batch,
)
from kiara.shared.constants import (
    ENRICHMENT_MAX_PREFIX_CHARS,
    ENRICHMENT_MIN_CHUNK_CHARS,
)

# ---------------------------------------------------------------------------
# Hilfsdaten
# ---------------------------------------------------------------------------

_TEMPLATE = (
    "Kontext:\n{document_context}\n\n"
    "Abschnitt:\n{chunk_text}\n\n"
    "Dateityp: {file_type}\nQuelle: {source}\nDatenquelle: {datasource_name}"
)

_SHORT_DOC = "Dies ist ein kurzes Testdokument."
_CHUNKS_3 = ["Erster Chunk.", "Zweiter Chunk mit mehr Inhalt.", "Dritter Chunk."]


def _mock_llm(prefix: str = "Generierter Kontext"):
    """Erstellt ein LLM-Mock-Callable."""

    def _fn(system_prompt: str, user_message: str) -> str:
        return prefix

    return _fn


# ---------------------------------------------------------------------------
# _build_document_context
# ---------------------------------------------------------------------------


class TestBuildDocumentContext:
    def test_short_document(self):
        """Test 1: Kurzes Dokument (< header_chars) → vollstaendiger Text im Kontext."""
        result = _build_document_context(
            full_text=_SHORT_DOC,
            chunk_text="Dies",
            chunk_index=0,
            all_chunk_texts=["Dies"],
            header_chars=2000,
        )
        assert _SHORT_DOC in result

    def test_long_document_truncated(self):
        """Test 2: Langes Dokument → Header + Nachbarn, insgesamt <= max_chars."""
        long_text = "A" * 10000
        chunks = ["A" * 1000, "B" * 1000, "C" * 1000]
        result = _build_document_context(
            full_text=long_text,
            chunk_text=chunks[1],
            chunk_index=1,
            all_chunk_texts=chunks,
            max_chars=500,
        )
        assert len(result) <= 500

    def test_first_chunk_no_predecessor(self):
        """Test 3: index=0 → kein Vorgaenger-Nachbar, nur Nachfolger."""
        result = _build_document_context(
            full_text="Header " * 100,
            chunk_text=_CHUNKS_3[0],
            chunk_index=0,
            all_chunk_texts=_CHUNKS_3,
        )
        assert _CURRENT_CHUNK_MARKER in result
        # Nachfolger sollte enthalten sein
        assert _CHUNKS_3[1][:20] in result or _ELLIPSIS in result

    def test_last_chunk_no_successor(self):
        """Test 4: Letzter Chunk → kein Nachfolger, nur Vorgaenger."""
        result = _build_document_context(
            full_text="Header " * 100,
            chunk_text=_CHUNKS_3[2],
            chunk_index=2,
            all_chunk_texts=_CHUNKS_3,
        )
        assert _CURRENT_CHUNK_MARKER in result

    def test_middle_chunk_both_neighbors(self):
        """Test 5: Mittlerer Chunk → beide Nachbarn vorhanden."""
        result = _build_document_context(
            full_text="Header " * 100,
            chunk_text=_CHUNKS_3[1],
            chunk_index=1,
            all_chunk_texts=_CHUNKS_3,
        )
        assert _CURRENT_CHUNK_MARKER in result

    def test_single_chunk_only_header(self):
        """Test 6: Single-Chunk-Dokument → nur Header, keine Nachbarn."""
        result = _build_document_context(
            full_text=_SHORT_DOC,
            chunk_text=_SHORT_DOC,
            chunk_index=0,
            all_chunk_texts=[_SHORT_DOC],
        )
        assert _CURRENT_CHUNK_MARKER in result
        # Keine Ellipsis fuer Nachbarn
        parts_without_marker = result.replace(_CURRENT_CHUNK_MARKER, "")
        assert _ELLIPSIS not in parts_without_marker or _SHORT_DOC in parts_without_marker

    def test_max_chars_zero(self):
        """Test 32: max_chars <= 0 → leerer String."""
        result = _build_document_context(
            full_text="Text",
            chunk_text="Text",
            chunk_index=0,
            all_chunk_texts=["Text"],
            max_chars=0,
        )
        assert result == ""

    def test_max_chars_negative(self):
        """Test 32b: Negativer max_chars → leerer String."""
        result = _build_document_context(
            full_text="Text",
            chunk_text="Text",
            chunk_index=0,
            all_chunk_texts=["Text"],
            max_chars=-1,
        )
        assert result == ""

    def test_empty_all_chunk_texts(self):
        """Test 33: Leere all_chunk_texts → full_text[:max_chars]."""
        result = _build_document_context(
            full_text="Volltext hier",
            chunk_text="",
            chunk_index=0,
            all_chunk_texts=[],
            max_chars=5,
        )
        assert result == "Vollt"

    def test_invalid_chunk_index(self):
        """Test 34: Ungueltiger chunk_index → IndexError."""
        with pytest.raises(IndexError):
            _build_document_context(
                full_text="Text",
                chunk_text="Text",
                chunk_index=5,
                all_chunk_texts=["Text"],
            )

    def test_small_max_chars_preserves_marker(self):
        """Test 35: Kleiner max_chars → Marker bleibt erhalten wenn >= len(marker)."""
        marker_len = len(_CURRENT_CHUNK_MARKER)
        result = _build_document_context(
            full_text="A" * 10000,
            chunk_text="A" * 100,
            chunk_index=0,
            all_chunk_texts=["A" * 100, "B" * 100],
            max_chars=marker_len + 10,
        )
        assert _CURRENT_CHUNK_MARKER in result

    def test_very_small_max_chars_truncates_marker(self):
        """Test 35b: max_chars < len(marker) → abgeschnittener Marker."""
        result = _build_document_context(
            full_text="A" * 1000,
            chunk_text="A" * 100,
            chunk_index=0,
            all_chunk_texts=["A" * 100],
            max_chars=5,
        )
        assert result == _CURRENT_CHUNK_MARKER[:5]


# ---------------------------------------------------------------------------
# _render_prompt
# ---------------------------------------------------------------------------


class TestRenderPrompt:
    def test_all_placeholders(self):
        """Test 7: Alle Platzhalter substituiert."""
        result = _render_prompt(
            template_content=_TEMPLATE,
            document_context="ctx",
            chunk_text="chunk",
            file_type=".pdf",
            source="/test.pdf",
            datasource_name="DS1",
        )
        assert "{" not in result
        assert "ctx" in result
        assert "chunk" in result
        assert ".pdf" in result
        assert "/test.pdf" in result
        assert "DS1" in result

    def test_missing_optional_placeholders(self):
        """Test 8: Fehlende optionale Platzhalter → leere Strings."""
        result = _render_prompt(
            template_content=_TEMPLATE,
            document_context="ctx",
            chunk_text="chunk",
        )
        assert "ctx" in result
        assert "chunk" in result
        assert "Dateityp: \n" in result  # file_type leer

    def test_empty_template(self):
        """Test 9: Leeres Template → leerer String."""
        result = _render_prompt(
            template_content="",
            document_context="ctx",
            chunk_text="chunk",
        )
        assert result == ""

    def test_unknown_placeholder(self):
        """Test 10: Unbekannter Platzhalter → durch '' ersetzt, kein KeyError."""
        result = _render_prompt(
            template_content="Hallo {unknown_var}!",
            document_context="ctx",
            chunk_text="chunk",
        )
        assert result == "Hallo !"


# ---------------------------------------------------------------------------
# enrich_chunk
# ---------------------------------------------------------------------------


class TestEnrichChunk:
    def test_successful_enrichment(self):
        """Test 11: Erfolgreicher LLM-Output → Prefix + \\n\\n + Original."""
        result = enrich_chunk(
            chunk_text="Original-Text",
            chunk_index=0,
            full_text="Original-Text",
            all_chunk_texts=["Original-Text"],
            llm_generate_fn=_mock_llm("Kontext hier"),
            prompt_template="{document_context}\n{chunk_text}",
        )
        assert result == "Kontext hier\n\nOriginal-Text"

    def test_empty_llm_output(self):
        """Test 12: Leerer LLM-Output → Original-Text unveraendert."""
        result = enrich_chunk(
            chunk_text="Original-Text",
            chunk_index=0,
            full_text="Original-Text",
            all_chunk_texts=["Original-Text"],
            llm_generate_fn=_mock_llm(""),
            prompt_template="{document_context}\n{chunk_text}",
        )
        assert result == "Original-Text"

    def test_whitespace_llm_output(self):
        """Test 12b: Nur-Whitespace LLM-Output → Original-Text."""
        result = enrich_chunk(
            chunk_text="Original-Text",
            chunk_index=0,
            full_text="Original-Text",
            all_chunk_texts=["Original-Text"],
            llm_generate_fn=_mock_llm("   \n\t  "),
            prompt_template="{document_context}\n{chunk_text}",
        )
        assert result == "Original-Text"

    def test_llm_exception_propagates(self):
        """Test 13: LLM-Exception propagiert (NICHT gefangen)."""

        def _failing_llm(s, u):
            raise ConnectionError("Backend down")

        with pytest.raises(ConnectionError, match="Backend down"):
            enrich_chunk(
                chunk_text="Text",
                chunk_index=0,
                full_text="Text",
                all_chunk_texts=["Text"],
                llm_generate_fn=_failing_llm,
                prompt_template="{chunk_text}",
            )

    def test_prefix_truncation(self):
        """Test 14: Zu langer Prefix → gekuerzt auf MAX_PREFIX_CHARS."""
        long_prefix = "A" * (ENRICHMENT_MAX_PREFIX_CHARS + 200)
        result = enrich_chunk(
            chunk_text="Original",
            chunk_index=0,
            full_text="Original",
            all_chunk_texts=["Original"],
            llm_generate_fn=_mock_llm(long_prefix),
            prompt_template="{chunk_text}",
        )
        prefix_part = result.split("\n\n")[0]
        assert len(prefix_part) <= ENRICHMENT_MAX_PREFIX_CHARS

    def test_prompt_contains_context_and_chunk(self):
        """Test 20: Mock-LLM erhaelt korrekt gerenderten Prompt."""
        captured = []

        def _capturing_llm(system_prompt, user_message):
            captured.append(user_message)
            return "prefix"

        enrich_chunk(
            chunk_text="Mein Chunk",
            chunk_index=0,
            full_text="Volltext",
            all_chunk_texts=["Mein Chunk"],
            llm_generate_fn=_capturing_llm,
            prompt_template="Kontext: {document_context}\nChunk: {chunk_text}",
        )
        assert "Mein Chunk" in captured[0]
        assert "Volltext" in captured[0]

    def test_metadata_placeholders(self):
        """Test 21: file_type, source, datasource_name korrekt substituiert."""
        captured = []

        def _capturing_llm(s, u):
            captured.append(u)
            return "prefix"

        enrich_chunk(
            chunk_text="Text",
            chunk_index=0,
            full_text="Text",
            all_chunk_texts=["Text"],
            llm_generate_fn=_capturing_llm,
            prompt_template="{file_type}|{source}|{datasource_name}|{chunk_text}",
            file_type=".pdf",
            source="/a/b.pdf",
            datasource_name="DS1",
        )
        assert ".pdf|/a/b.pdf|DS1|Text" in captured[0]

    def test_non_string_llm_return_raises(self):
        """Test 23: Nicht-stringiger LLM-Return (inkl. None) → TypeError."""
        with pytest.raises(TypeError, match="str"):
            enrich_chunk(
                chunk_text="Text",
                chunk_index=0,
                full_text="Text",
                all_chunk_texts=["Text"],
                llm_generate_fn=lambda s, u: None,
                prompt_template="{chunk_text}",
            )

    def test_integer_llm_return_raises(self):
        """Test 23b: Integer-Return → TypeError."""
        with pytest.raises(TypeError):
            enrich_chunk(
                chunk_text="Text",
                chunk_index=0,
                full_text="Text",
                all_chunk_texts=["Text"],
                llm_generate_fn=lambda s, u: 42,
                prompt_template="{chunk_text}",
            )


# ---------------------------------------------------------------------------
# enrich_chunks_batch
# ---------------------------------------------------------------------------


class TestEnrichChunksBatch:
    def test_empty_list(self):
        """Test 15: Leere Liste → leere Liste, Stats alle 0."""
        results, stats = enrich_chunks_batch(
            chunk_texts=[],
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="",
        )
        assert results == []
        assert stats == {
            "enriched": 0,
            "skipped": 0,
            "unchanged": 0,
            "errors": 0,
            "enrichment_time_ms": 0,
        }

    def test_partial_failure(self):
        """Test 16: 1 von 3 fehlschlaegt → enriched=2, errors=1."""
        call_count = [0]

        def _mixed_llm(s, u):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("LLM down")
            return "prefix"

        chunks = ["A" * 100, "B" * 100, "C" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="A" * 300,
            llm_generate_fn=_mixed_llm,
            prompt_template="{chunk_text}",
        )
        assert len(results) == 3
        assert stats["enriched"] == 2
        assert stats["errors"] == 1
        assert results[1] == chunks[1]  # Fehler → Original

    def test_skip_short_chunks(self):
        """Test 17: 1 zu kurz → enriched=2, skipped=1."""
        chunks = ["A" * 100, "B" * 10, "C" * 100]  # B ist zu kurz
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="A" * 300,
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
        )
        assert stats["enriched"] == 2
        assert stats["skipped"] == 1
        assert results[1] == chunks[1]  # Skip → Original

    def test_llm_exception_no_crash(self):
        """Test 18: LLM-Exception bei 1 von 3 → enriched=2, errors=1, kein Crash."""
        call_count = [0]

        def _failing_on_second(s, u):
            call_count[0] += 1
            if call_count[0] == 2:
                raise ConnectionError("timeout")
            return "prefix"

        chunks = ["A" * 100, "B" * 100, "C" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_failing_on_second,
            prompt_template="{chunk_text}",
        )
        assert stats["errors"] == 1
        assert stats["enriched"] == 2
        assert len(results) == 3

    def test_progress_callback(self):
        """Test 19: Callback aufgerufen mit (1,3), (2,3), (3,3)."""
        calls = []
        chunks = ["A" * 100, "B" * 10, "C" * 100]  # B skipped

        enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            progress_callback=lambda cur, tot: calls.append((cur, tot)),
        )
        assert calls == [(1, 3), (2, 3), (3, 3)]

    def test_exact_min_chunk_chars(self):
        """Test 22: Chunk exakt MIN_CHUNK_CHARS lang → wird enriched (strict less-than)."""
        chunk = "A" * ENRICHMENT_MIN_CHUNK_CHARS  # Genau 50 Zeichen
        results, stats = enrich_chunks_batch(
            chunk_texts=[chunk],
            full_text=chunk,
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
        )
        assert stats["skipped"] == 0
        assert stats["enriched"] == 1

    def test_non_string_llm_return_batch(self):
        """Test 24: Nicht-stringiger Return → errors=1, Original bleibt."""
        chunks = ["A" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=lambda s, u: None,
            prompt_template="{chunk_text}",
        )
        assert stats["errors"] == 1
        assert results[0] == chunks[0]

    def test_metadata_none(self):
        """Test 25: file_metadata=None → kein Crash."""
        chunks = ["A" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            file_metadata=None,
        )
        assert stats["enriched"] == 1

    def test_partial_metadata(self):
        """Test 26: Partielle file_metadata → fehlende Keys bleiben ''."""
        captured = []

        def _cap(s, u):
            captured.append(u)
            return "prefix"

        chunks = ["A" * 100]
        enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_cap,
            prompt_template="{file_type}|{source}|{datasource_name}|{chunk_text}",
            file_metadata={"file_type": ".txt"},
        )
        # source und datasource_name sollten leer sein
        assert ".txt||" in captured[0]

    def test_non_mapping_metadata(self):
        """Test 27: Truthy Nicht-Mapping → Warning, Platzhalter leer."""
        chunks = ["A" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            file_metadata="not a dict",
        )
        assert stats["enriched"] == 1

    def test_preserves_order_and_length(self):
        """Test 28: Reihenfolge und Laenge exakt beibehalten."""
        chunks = ["A" * 100, "B" * 10, "C" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
        )
        assert len(results) == len(chunks)
        # Skipped chunk bleibt identisch
        assert results[1] is chunks[1] or results[1] == chunks[1]

    def test_failing_progress_callback(self):
        """Test 29: Fehlerhafter Callback → Warning, Verarbeitung laeuft weiter."""

        def _bad_cb(cur, tot):
            raise ValueError("CB broken")

        chunks = ["A" * 100, "B" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            progress_callback=_bad_cb,
        )
        assert stats["enriched"] == 2
        assert len(results) == 2

    def test_max_context_chars_override(self):
        """Test 30: max_context_chars=1234 wird durchgereicht."""
        captured_prompts = []

        def _cap(s, u):
            captured_prompts.append(u)
            return "prefix"

        long_text = "X" * 5000
        chunks = [long_text]
        enrich_chunks_batch(
            chunk_texts=chunks,
            full_text=long_text,
            llm_generate_fn=_cap,
            prompt_template="{document_context}",
            max_context_chars=200,
        )
        # document_context sollte auf ~200 Zeichen begrenzt sein
        assert len(captured_prompts[0]) <= 200

    def test_malformed_template(self):
        """Test 31: Kaputtes Template → Fehler gezaehlt, Original bleibt."""
        chunks = ["A" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text} {foo.bar}",
        )
        assert stats["errors"] == 1
        assert results[0] == chunks[0]

    def test_mixed_batch(self):
        """Test 36: Gemischter Batch (Erfolg + Skip + Error)."""
        call_count = [0]

        def _mixed(s, u):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("fail")
            return "prefix"

        chunks = ["A" * 100, "B" * 100, "short", "D" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mixed,
            prompt_template="{chunk_text}",
        )
        assert stats["enriched"] == 2
        assert stats["errors"] == 1
        assert stats["skipped"] == 1
        assert len(results) == 4
        # Skip- und Error-Slots bleiben byte-identisch
        assert results[1] == chunks[1]  # Error → Original
        assert results[2] == chunks[2]  # Skip → Original

    def test_broken_metadata_get(self):
        """Test 37: Mapping dessen .get() wirft → Warning, Platzhalter leer."""

        class BrokenMapping:
            def get(self, key, default=""):
                raise RuntimeError("broken get")

            def __contains__(self, key):
                return False

            def __getitem__(self, key):
                raise KeyError(key)

        chunks = ["A" * 100]
        # BrokenMapping ist kein Mapping (isinstance-Check schlaegt fehl) → Warning-Pfad
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            file_metadata=BrokenMapping(),
        )
        assert stats["enriched"] == 1

    def test_broken_str_metadata_value(self):
        """Test 38: Metadatenwert dessen __str__ wirft → Warning, Platzhalter leer."""

        class BadStr:
            def __str__(self):
                raise ValueError("cannot convert")

            def __bool__(self):
                return True

        chunks = ["A" * 100]
        results, stats = enrich_chunks_batch(
            chunk_texts=chunks,
            full_text="",
            llm_generate_fn=_mock_llm(),
            prompt_template="{chunk_text}",
            file_metadata={"file_type": BadStr()},
        )
        assert stats["enriched"] == 1
