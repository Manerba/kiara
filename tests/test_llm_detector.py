"""Tests fuer LLM-basierte PII-Erkennung (Schicht 4).

Alle Tests nutzen gemockte LLM-Aufrufe — kein echter Ollama-Server noetig.
"""

import json
from unittest.mock import MagicMock

from kiara.anonymization.llm_detector import (
    LLMDetection,
    LLMPIIDetector,
    _normalize_category,
    _overlaps,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(enabled=True, threshold=0.7, cache_size=16):
    """Erzeugt eine minimale Config mit llm_detection."""
    return {
        "ollama": {
            "url": "http://localhost:11434",
            "llm_url": "",
            "embedding_url": "",
            "llm_model": "test-model",
            "embedding_model": "test-embed",
            "prompt_template": "",
        },
        "anonymization": {
            "enabled": True,
            "llm_detection": {
                "enabled": enabled,
                "model": "",
                "temperature": 0.1,
                "confidence_threshold": threshold,
                "timeout_seconds": 5,
                "cache_size": cache_size,
                "num_ctx": 2048,
            },
        },
    }


def _mock_llm_response(response_text):
    """Erzeugt einen gepatchten LLM, der den gegebenen Text zurueckgibt."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = response_text
    return mock_llm


# ---------------------------------------------------------------------------
# TestDetection — Grundlegende Erkennung
# ---------------------------------------------------------------------------


class TestDetection:
    """Grundlegende PII-Erkennung per LLM."""

    def test_person_detected(self):
        """LLM erkennt eine Person im Text."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Thomas Bergmann",
                        "category": "PERSON",
                        "confidence": 0.95,
                        "reasoning": "Deutscher Vorname + Nachname",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        text = "Bitte kontaktieren Sie Thomas Bergmann fuer Details."
        results = detector.detect(text, [])

        assert len(results) == 1
        assert results[0].value == "Thomas Bergmann"
        assert results[0].category == "person"
        assert results[0].confidence == 0.95
        assert text[results[0].start : results[0].end] == "Thomas Bergmann"

    def test_address_detected(self):
        """LLM erkennt eine Adresse im Text."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Marienplatz 12",
                        "category": "ADDRESS",
                        "confidence": 0.85,
                        "reasoning": "Deutsche Strassenadresse",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        text = "Lieferung an Marienplatz 12 in Muenchen."
        results = detector.detect(text, [])

        assert len(results) == 1
        assert results[0].category == "address"

    def test_multiple_entities(self):
        """LLM erkennt mehrere Entitaeten gleichzeitig."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Maria Weber",
                        "category": "PERSON",
                        "confidence": 0.9,
                        "reasoning": "Name",
                    },
                    {
                        "text": "Acme Corp",
                        "category": "ORGANIZATION",
                        "confidence": 0.8,
                        "reasoning": "Firmenname",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        text = "Maria Weber arbeitet bei Acme Corp seit 2020."
        results = detector.detect(text, [])

        assert len(results) == 2
        categories = {r.category for r in results}
        assert categories == {"person", "organization"}

    def test_empty_text(self):
        """Leerer Text liefert leere Liste."""
        config = _make_config()
        detector = LLMPIIDetector(config)

        results = detector.detect("", [])
        assert results == []

    def test_disabled_returns_empty(self):
        """Deaktivierter Detector liefert immer leere Liste."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        results = detector.detect("Thomas Bergmann ist hier.", [])
        assert results == []


# ---------------------------------------------------------------------------
# TestConfidence — Schwellwert-Filterung
# ---------------------------------------------------------------------------


class TestConfidence:
    """Confidence-Schwellwert-Filterung."""

    def test_below_threshold_filtered(self):
        """Entitaeten unter dem Threshold werden verworfen."""
        config = _make_config(threshold=0.7)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "ABC",
                        "category": "PERSON",
                        "confidence": 0.5,
                        "reasoning": "Unsicher",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("ABC ist ein Test.", [])
        assert results == []

    def test_above_threshold_kept(self):
        """Entitaeten ueber dem Threshold werden behalten."""
        config = _make_config(threshold=0.7)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Klaus",
                        "category": "PERSON",
                        "confidence": 0.85,
                        "reasoning": "Vorname",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Klaus ruft an.", [])
        assert len(results) == 1

    def test_exact_threshold_kept(self):
        """Entitaeten genau am Threshold werden behalten."""
        config = _make_config(threshold=0.7)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Test",
                        "category": "CUSTOM",
                        "confidence": 0.7,
                        "reasoning": "Grenzfall",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Test Daten hier.", [])
        assert len(results) == 1

    def test_nan_confidence_rejected(self):
        """Confidence=NaN wird auf 0.0 gemappt → Threshold-Filter greift."""
        config = _make_config(threshold=0.5)
        detector = LLMPIIDetector(config)
        # NaN als String-Literal (Python json.loads akzeptiert "NaN")
        detector._llm = _mock_llm_response(
            '[{"text": "Adversarial", "category": "PERSON", "confidence": NaN, "reasoning": "x"}]'
        )
        detector._llm_init_attempted = True

        results = detector.detect("Adversarial im Text.", [])
        assert results == []

    def test_infinity_confidence_rejected(self):
        """Confidence=Infinity wird auf 0.0 gemappt → Threshold-Filter greift."""
        config = _make_config(threshold=0.5)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            '[{"text": "Adversarial", "category": "PERSON", "confidence": Infinity, "reasoning": "x"}]'
        )
        detector._llm_init_attempted = True

        results = detector.detect("Adversarial im Text.", [])
        assert results == []

    def test_out_of_range_confidence_rejected(self):
        """Confidence ausserhalb [0,1] (z.B. 70 als Prozent) wird auf 0.0 gemappt."""
        config = _make_config(threshold=0.5)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Adversarial",
                        "category": "PERSON",
                        "confidence": 70,
                        "reasoning": "Prozent statt 0..1",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Adversarial im Text.", [])
        assert results == []

    def test_negative_confidence_rejected(self):
        """Negative Confidence wird auf 0.0 gemappt."""
        config = _make_config(threshold=0.5)
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Adversarial",
                        "category": "PERSON",
                        "confidence": -1,
                        "reasoning": "Negativ",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Adversarial im Text.", [])
        assert results == []


# ---------------------------------------------------------------------------
# TestJSONParsing — Robustes Parsing
# ---------------------------------------------------------------------------


class TestJSONParsing:
    """Robustes Parsing der LLM-Antworten."""

    def test_direct_json(self):
        """Direktes JSON-Array wird korrekt geparst."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            '[{"text": "Max", "category": "PERSON", "confidence": 0.9, "reasoning": "Name"}]'
        )
        detector._llm_init_attempted = True

        results = detector.detect("Max ist da.", [])
        assert len(results) == 1

    def test_markdown_block(self):
        """JSON in Markdown-Code-Block wird extrahiert."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        response = '```json\n[{"text": "Eva", "category": "PERSON", "confidence": 0.9, "reasoning": "Name"}]\n```'
        detector._llm = _mock_llm_response(response)
        detector._llm_init_attempted = True

        results = detector.detect("Eva kommt spaeter.", [])
        assert len(results) == 1

    def test_json_with_surrounding_text(self):
        """JSON-Array in Freitext wird gefunden."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        response = 'Hier sind die Ergebnisse: [{"text": "Kai", "category": "PERSON", "confidence": 0.8, "reasoning": "Name"}] Das war alles.'
        detector._llm = _mock_llm_response(response)
        detector._llm_init_attempted = True

        results = detector.detect("Kai hat gefragt.", [])
        assert len(results) == 1

    def test_invalid_json(self):
        """Ungueltiges JSON liefert leere Liste (kein Fehler)."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response("Das ist kein JSON.")
        detector._llm_init_attempted = True

        results = detector.detect("Irgendein Text hier.", [])
        assert results == []

    def test_empty_response(self):
        """Leere Antwort liefert leere Liste."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response("")
        detector._llm_init_attempted = True

        results = detector.detect("Ein Text.", [])
        assert results == []

    def test_empty_array_response(self):
        """Leeres JSON-Array liefert leere Liste."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response("[]")
        detector._llm_init_attempted = True

        results = detector.detect("Ein sauberer Text.", [])
        assert results == []

    def test_json_array_with_nested_arrays(self):
        """Issue #1913: Non-greedy Bracket-Regex versagt bei nested arrays.

        Vor dem Fix matchte re.search(r"\\[.*?\\]") nur bis zum ersten ']' —
        bei verschachtelten Arrays brach der Match mitten im JSON ab.
        """
        config = _make_config()
        detector = LLMPIIDetector(config)
        response = '[{"text": "Lea", "category": "PERSON", "confidence": 0.9, "reasoning": "Name", "aliases": ["L.", "Lea M."]}]'
        detector._llm = _mock_llm_response(response)
        detector._llm_init_attempted = True

        results = detector.detect("Lea hat angerufen.", [])
        assert len(results) == 1

    def test_json_array_with_prefix_brackets(self):
        """Issue #1913: Vorspann mit Klammertext darf das Parsen nicht blockieren.

        Vor dem Fix matchte non-greedy '[Hinweis]' und scheiterte am
        json.loads — das echte JSON-Array dahinter wurde nie geparst.
        """
        config = _make_config()
        detector = LLMPIIDetector(config)
        response = 'Bei [Hinweis] folgt: [{"text": "Tom", "category": "PERSON", "confidence": 0.85, "reasoning": "Vorname"}]'
        detector._llm = _mock_llm_response(response)
        detector._llm_init_attempted = True

        results = detector.detect("Tom war hier.", [])
        assert len(results) == 1


# ---------------------------------------------------------------------------
# TestCaching — LRU-Cache-Verhalten
# ---------------------------------------------------------------------------


class TestCaching:
    """LRU-Cache-Verhalten."""

    def test_cache_hit(self):
        """Zweiter Aufruf mit gleichem Text nutzt den Cache."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        mock_llm = _mock_llm_response(
            json.dumps(
                [
                    {"text": "Peter", "category": "PERSON", "confidence": 0.9, "reasoning": "Name"},
                ]
            )
        )
        detector._llm = mock_llm
        detector._llm_init_attempted = True

        text = "Peter ruft an."
        results1 = detector.detect(text, [])
        results2 = detector.detect(text, [])

        assert len(results1) == 1
        assert len(results2) == 1
        # LLM wurde nur einmal aufgerufen
        assert mock_llm.generate.call_count == 1

    def test_cache_miss(self):
        """Unterschiedliche Texte fuehren zu separaten LLM-Aufrufen."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        mock_llm = _mock_llm_response(json.dumps([]))
        detector._llm = mock_llm
        detector._llm_init_attempted = True

        detector.detect("Text eins.", [])
        detector.detect("Text zwei.", [])

        assert mock_llm.generate.call_count == 2

    def test_cache_eviction(self):
        """Cache raeumt aelteste Eintraege bei Ueberlauf."""
        config = _make_config(cache_size=2)
        detector = LLMPIIDetector(config)
        mock_llm = _mock_llm_response(json.dumps([]))
        detector._llm = mock_llm
        detector._llm_init_attempted = True

        detector.detect("Text A.", [])
        detector.detect("Text B.", [])
        detector.detect("Text C.", [])  # Raeumt "Text A" aus dem Cache

        assert len(detector._cache) == 2

        # "Text A" muss jetzt aus dem Cache sein → erneuter LLM-Aufruf
        mock_llm.generate.reset_mock()
        detector.detect("Text A.", [])
        assert mock_llm.generate.call_count == 1


# ---------------------------------------------------------------------------
# TestCategoryNormalization — Kategorie-Normalisierung
# ---------------------------------------------------------------------------


class TestCategoryNormalization:
    """Normalisierung von Kategorie-Bezeichnungen."""

    def test_english_uppercase(self):
        assert _normalize_category("PERSON") == "person"

    def test_english_lowercase(self):
        assert _normalize_category("organization") == "organization"

    def test_german_name(self):
        assert _normalize_category("Unternehmen") == "organization"

    def test_german_adresse(self):
        assert _normalize_category("Adresse") == "address"

    def test_german_personen(self):
        assert _normalize_category("Personen") == "person"

    def test_unknown_falls_back_to_custom(self):
        assert _normalize_category("FOOBAR") == "custom"

    def test_whitespace_stripped(self):
        assert _normalize_category("  person  ") == "person"


# ---------------------------------------------------------------------------
# TestErrorHandling — Fehlerbehandlung
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Fehlerbehandlung: Layer 4 darf nie blockieren."""

    def test_llm_exception(self):
        """LLM-Exception fuehrt zu leerer Liste."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = ConnectionError("Ollama nicht erreichbar")
        detector._llm = mock_llm
        detector._llm_init_attempted = True

        results = detector.detect("Test mit Fehler.", [])
        assert results == []

    def test_llm_unreachable(self):
        """Wenn _create_llm None liefert, wird [] zurueckgegeben."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = None
        detector._llm_init_attempted = True

        results = detector.detect("Kein LLM verfuegbar.", [])
        assert results == []

    def test_invalid_entity_format(self):
        """Ungueltiges Entity-Format wird stillschweigend ignoriert."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {"wrong_key": "Test"},  # Fehlendes "text"-Feld
                    "not a dict",  # Kein Dict
                    42,  # Kein Dict
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Test Text.", [])
        assert results == []

    def test_entity_not_in_original(self):
        """Entitaet, die nicht im Originaltext vorkommt, wird verworfen."""
        config = _make_config()
        detector = LLMPIIDetector(config)
        detector._llm = _mock_llm_response(
            json.dumps(
                [
                    {
                        "text": "Fantasiename",
                        "category": "PERSON",
                        "confidence": 0.95,
                        "reasoning": "Halluzination",
                    },
                ]
            )
        )
        detector._llm_init_attempted = True

        results = detector.detect("Hier steht nur normaler Text.", [])
        assert results == []


# ---------------------------------------------------------------------------
# TestExtractUncovered — Maskierung bereits abgedeckter Bereiche
# ---------------------------------------------------------------------------


class TestExtractUncovered:
    """Maskierung bereits abgedeckter Bereiche vor dem LLM-Aufruf."""

    def test_no_covered_ranges(self):
        """Ohne abgedeckte Bereiche bleibt der Text unveraendert."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        result = detector._extract_uncovered("Hallo Welt", [])
        assert result == "Hallo Welt"

    def test_single_covered_range(self):
        """Ein abgedeckter Bereich wird maskiert."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Rechnung an Mueller GmbH bitte."
        result = detector._extract_uncovered(text, [(12, 24)])
        assert "[MASKIERT]" in result
        assert "Mueller GmbH" not in result

    def test_multiple_covered_ranges(self):
        """Mehrere abgedeckte Bereiche werden alle maskiert."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Von Hans an Petra: Daten."
        result = detector._extract_uncovered(text, [(4, 8), (12, 17)])
        assert "Hans" not in result
        assert "Petra" not in result
        assert result.count("[MASKIERT]") == 2

    def test_overlapping_ranges_are_merged(self):
        """Ueberlappende Bereiche werden gemerged statt korrumpiert.

        Regression: Frueher fuehrte die zweite Substitution mitten in den
        bereits gesetzten [MASKIERT]-Marker und erzeugte Texte wie
        '[MASKIERT]ASKIERT]rld Foo'.
        """
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Hello World Foo"
        # Ranges (0,5)='Hello' und (3,8)='lo Wo' ueberlappen → gemerged (0,8)='Hello Wo'
        result = detector._extract_uncovered(text, [(0, 5), (3, 8)])
        # Genau ein zusammengefasster Marker, kein verstuemmelter zweiter
        assert result.count("[MASKIERT]") == 1
        assert "ASKIERT" not in result.replace("[MASKIERT]", "")
        assert result == "[MASKIERT]rld Foo"

    def test_out_of_range_end_is_clamped(self):
        """end > len(text) wird auf len(text) geklemmt, kein Daten-Verlust dahinter."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Hello World"  # len=11
        result = detector._extract_uncovered(text, [(6, 999)])
        assert result == "Hello [MASKIERT]"

    def test_negative_start_is_clamped(self):
        """Negativer start wird auf 0 geklemmt, kein Tail-Verlust durch result[:-n]."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Hello World"
        result = detector._extract_uncovered(text, [(-3, 5)])
        # start=-3 darf nicht result[:-3] schneiden — sonst entstuende
        # 'Hello Wo' + '[MASKIERT]' + ' World' o.ae.
        assert result == "[MASKIERT] World"

    def test_inverted_or_empty_range_is_dropped(self):
        """start>=end Intervalle werden verworfen, kein leerer Marker eingefuegt."""
        config = _make_config(enabled=False)
        detector = LLMPIIDetector(config)

        text = "Hello World"
        result = detector._extract_uncovered(text, [(5, 5), (8, 3)])
        assert result == text


# ---------------------------------------------------------------------------
# TestOverlap — Ueberlappungserkennung
# ---------------------------------------------------------------------------


class TestOverlap:
    """Ueberlappungserkennung bei Bereichs-Pruefung."""

    def test_no_overlap(self):
        assert not _overlaps(0, 5, [(10, 20)])

    def test_full_overlap(self):
        assert _overlaps(10, 20, [(10, 20)])

    def test_partial_overlap_start(self):
        assert _overlaps(8, 15, [(10, 20)])

    def test_partial_overlap_end(self):
        assert _overlaps(15, 25, [(10, 20)])

    def test_contained(self):
        assert _overlaps(12, 18, [(10, 20)])

    def test_empty_ranges(self):
        assert not _overlaps(0, 5, [])


# ---------------------------------------------------------------------------
# TestAnonymizerIntegration — Roundtrip mit Anonymizer
# ---------------------------------------------------------------------------


class TestAnonymizerIntegration:
    """Integration mit dem Anonymizer (Schicht 1-2-3 + LLM)."""

    def test_anonymizer_with_llm_roundtrip(self, llm_detection_config):
        """LLM-Schicht wird korrekt in den Anonymizer integriert."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.build()

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            LLMDetection(
                start=0,
                end=11,
                value="Max Mueller",
                category="person",
                confidence=0.9,
                reasoning="Deutscher Name",
            ),
        ]

        anon = Anonymizer(automaton, llm_detection_config, llm_detector=mock_detector)

        clean_text = anon.anonymize("Max Mueller hat angerufen.")

        assert "Max Mueller" not in clean_text
        assert anon.last_result.stats["total"] >= 1

    def test_anonymizer_llm_failure_fallback(self, llm_detection_config):
        """Bei LLM-Fehler laeuft die Pipeline trotzdem weiter."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.build()

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("LLM kaputt")

        anon = Anonymizer(automaton, llm_detection_config, llm_detector=mock_detector)

        # Kein Fehler — Pipeline laeuft durch
        clean_text = anon.anonymize("Normaler Text ohne PII.")
        assert clean_text == "Normaler Text ohne PII."


# ---------------------------------------------------------------------------
# TestAnonymizerDeanonymizeIntegration — Integration mit Anonymizer
# ---------------------------------------------------------------------------


class TestAnonymizerDeanonymizeIntegration:
    """Integration mit dem Anonymizer (ehemals Pseudonymizer)."""

    def test_anonymizer_deanonymize(self):
        """LLM-Treffer koennen ueber den Anonymizer de-anonymisiert werden."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton._finalized = True

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            LLMDetection(
                start=0,
                end=12,
                value="Lisa Schmidt",
                category="person",
                confidence=0.9,
                reasoning="Name",
            ),
        ]

        config = _make_config()
        anon = Anonymizer(automaton, config, llm_detector=mock_detector)

        text = "Lisa Schmidt ist Projektleiterin."
        anon_text = anon.anonymize(text)

        assert "Lisa Schmidt" not in anon_text
        assert "\u2039ANON:Person_001\u203a" in anon_text

        # De-Anonymisierung
        restored = anon.deanonymize(anon_text)
        assert "Lisa Schmidt" in restored

    def test_anonymizer_stats(self):
        """LLM-Treffer erscheinen in den Stats."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton._finalized = True

        mock_detector = MagicMock()
        mock_detector.detect.return_value = [
            LLMDetection(
                start=0,
                end=10,
                value="Karl Braun",
                category="person",
                confidence=0.85,
                reasoning="Name",
            ),
        ]

        config = _make_config()
        anon = Anonymizer(automaton, config, llm_detector=mock_detector)

        anon.anonymize("Karl Braun ist neu hier.")

        assert anon.stats["total_entities"] >= 1
        assert "PERSON" in anon.stats["entity_types"]

    def test_anonymizer_llm_failure(self):
        """Anonymizer laeuft bei LLM-Fehler weiter."""
        from kiara.anonymization.anonymizer import Anonymizer
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton._finalized = True

        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("LLM down")

        config = _make_config()
        anon = Anonymizer(automaton, config, llm_detector=mock_detector)

        # Kein Fehler
        result = anon.anonymize("Normaler Text.")
        assert result == "Normaler Text."
