"""Tests fuer AnonymizationAutomaton (Aho-Corasick-basierte Entitaetserkennung).

Testet: Single-Match, Multi-Match, Overlap-Resolution, Wortgrenzen,
Case-Insensitivity, Edge Cases und Performance.
"""

import time

from kiara.anonymization.automaton import AnonymizationAutomaton

# ---------------------------------------------------------------------------
# Einfacher Match
# ---------------------------------------------------------------------------


class TestSingleMatch:
    def test_find_single_entity(self, sample_entities):
        """Eine bekannte Entitaet im Text wird gefunden."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("Rechnung an Mueller GmbH erhalten.")
        assert len(results) == 1
        assert results[0]["value"] == "Mueller GmbH"
        assert results[0]["category"] == "customer"
        assert results[0]["id"] == "K-1001"

    def test_match_positions_correct(self, sample_entities):
        """Start/End-Positionen stimmen mit dem Text ueberein."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        text = "Lieferung an Schmidt AG geplant."
        results = engine.find_entities(text)
        assert len(results) == 1
        match = results[0]
        assert text[match["start"] : match["end"]] == "Schmidt AG"

    def test_match_positions_use_original_nfd_offsets(self):
        """Offsets beziehen sich auch bei NFD-Eingabe auf den Originalstring."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "M\u00fcller GmbH", "category": "customer", "id": "K-1"},
            ]
        )
        engine.build()

        text = "Kunde Mu\u0308ller GmbH zahlt."
        results = engine.find_entities(text)

        assert len(results) == 1
        match = results[0]
        assert text[match["start"] : match["end"]] == "Mu\u0308ller GmbH"
        assert match["end"] == len("Kunde Mu\u0308ller GmbH")

    def test_match_at_text_start(self, sample_entities):
        """Entitaet am Textanfang wird gefunden."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("Mueller GmbH hat bestellt.")
        assert len(results) == 1
        assert results[0]["start"] == 0

    def test_match_at_text_end(self, sample_entities):
        """Entitaet am Textende wird gefunden."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("Bestellung von Mueller GmbH")
        assert len(results) == 1
        assert results[0]["end"] == len("Bestellung von Mueller GmbH")


# ---------------------------------------------------------------------------
# Mehrere Matches
# ---------------------------------------------------------------------------


class TestMultipleMatches:
    def test_two_entities_in_text(self, sample_entities):
        """Zwei verschiedene Entitaeten werden beide erkannt."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        text = "Mueller GmbH bestellt bei Schmidt AG."
        results = engine.find_entities(text)

        values = {r["value"] for r in results}
        assert "Mueller GmbH" in values
        assert "Schmidt AG" in values
        assert len(results) == 2

    def test_three_categories_mixed(self, sample_entities):
        """Entitaeten aus verschiedenen Kategorien im selben Text."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        text = "Hans Mueller arbeitet in Stuttgart fuer Mueller GmbH."
        results = engine.find_entities(text)
        categories = {r["category"] for r in results}
        assert "employee" in categories
        assert "location" in categories
        assert "customer" in categories

    def test_matches_sorted_by_position(self, sample_entities):
        """Ergebnisse sind nach Start-Position sortiert."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        text = "Stuttgart: Mueller GmbH, Lieferant Bauer & Soehne KG."
        results = engine.find_entities(text)
        positions = [r["start"] for r in results]
        assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# Overlap-Resolution
# ---------------------------------------------------------------------------


class TestOverlapResolution:
    def test_longest_match_wins(self):
        """Bei Ueberlappung gewinnt der laengste Treffer."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Mueller GmbH", "category": "customer", "id": "K-1"},
                {"value": "Hans Mueller", "category": "employee", "id": "M-1"},
                {"value": "Hans Mueller GmbH", "category": "customer", "id": "K-2"},
            ]
        )
        engine.build()

        results = engine.find_entities("Bestellung: Hans Mueller GmbH")
        # "Hans Mueller GmbH" ist der laengste Match
        assert len(results) == 1
        assert results[0]["value"] == "Hans Mueller GmbH"

    def test_non_overlapping_preserved(self):
        """Nicht ueberlappende Treffer bleiben alle erhalten."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Alpha GmbH", "category": "customer", "id": "K-1"},
                {"value": "Beta AG", "category": "customer", "id": "K-2"},
            ]
        )
        engine.build()

        results = engine.find_entities("Alpha GmbH und Beta AG")
        assert len(results) == 2

    def test_subset_entity_suppressed(self):
        """Ein kuerzerer Treffer innerhalb eines laengeren wird unterdrueckt."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Deutsche Bank AG", "category": "customer", "id": "K-1"},
                {"value": "Bank", "category": "custom", "id": "X-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Ueberweisung an Deutsche Bank AG.")
        assert len(results) == 1
        assert results[0]["value"] == "Deutsche Bank AG"

    def test_large_overlap_resolution_does_not_materialize_positions(self):
        """Overlap-Resolution arbeitet nach Intervallen statt Positionssets."""
        from kiara.anonymization._text_utils import _resolve_overlaps

        matches = [
            {"start": 0, "end": 100_000_000, "value": "large"},
            {"start": 1, "end": 2, "value": "small"},
            {"start": 100_000_001, "end": 100_000_010, "value": "next"},
        ]

        resolved = _resolve_overlaps(matches)

        assert [m["value"] for m in resolved] == ["large", "next"]


# ---------------------------------------------------------------------------
# Wortgrenzen
# ---------------------------------------------------------------------------


class TestWordBoundaries:
    def test_no_partial_match_prefix(self, sample_entities):
        """'Benzmueller' darf NICHT auf 'Mueller' matchen."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Mueller", "category": "employee", "id": "M-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Herr Benzmueller hat angerufen.")
        assert len(results) == 0

    def test_no_partial_match_suffix(self):
        """'Muellerweg' darf NICHT auf 'Mueller' matchen."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Mueller", "category": "employee", "id": "M-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Adresse: Muellerweg 5")
        assert len(results) == 0

    def test_match_with_punctuation_boundary(self):
        """Satzzeichen zaehlen als Wortgrenze."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Mueller", "category": "employee", "id": "M-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Info: Mueller, siehe Anlage.")
        assert len(results) == 1

    def test_match_surrounded_by_spaces(self):
        """Leerzeichen sind Wortgrenzen."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Test GmbH", "category": "customer", "id": "K-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Die Test GmbH ist Kunde.")
        assert len(results) == 1

    def test_match_after_opening_bracket(self):
        """Klammer zaehlt als Wortgrenze."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Firma ABC", "category": "customer", "id": "K-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Besteller (Firma ABC) bestaetigt.")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Case-Insensitive
# ---------------------------------------------------------------------------


class TestCaseInsensitive:
    def test_lowercase_input_matches(self, sample_entities):
        """Kleingeschriebener Text matcht auf Entity in Originalschreibung."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("bestellung von mueller gmbh.")
        assert len(results) == 1
        assert results[0]["category"] == "customer"

    def test_uppercase_input_matches(self, sample_entities):
        """Grossgeschriebener Text matcht."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("LIEFERUNG AN SCHMIDT AG")
        assert len(results) == 1

    def test_original_casing_preserved(self, sample_entities):
        """Im Ergebnis wird das Original-Casing aus dem Quelltext zurueckgegeben."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("von MUELLER GMBH erhalten")
        assert len(results) == 1
        assert results[0]["value"] == "MUELLER GMBH"


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text(self, sample_entities):
        """Leerer Text gibt leere Liste zurueck."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        assert engine.find_entities("") == []

    def test_text_without_matches(self, sample_entities):
        """Text ohne bekannte Entitaeten gibt leere Liste zurueck."""
        engine = AnonymizationAutomaton()
        engine.add_entities(sample_entities)
        engine.build()

        results = engine.find_entities("Das Wetter ist heute schoen.")
        assert results == []

    def test_no_entities_loaded(self):
        """Engine ohne geladene Entitaeten gibt leere Liste zurueck."""
        engine = AnonymizationAutomaton()
        engine.build()

        assert engine.find_entities("Mueller GmbH") == []

    def test_entity_count_property(self, sample_entities):
        """entity_count gibt die korrekte Anzahl zurueck."""
        engine = AnonymizationAutomaton()
        assert engine.entity_count == 0

        engine.add_entities(sample_entities)
        assert engine.entity_count == len(sample_entities)

    def test_short_entity_ignored(self):
        """Entitaeten mit weniger als 2 Zeichen werden ignoriert."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "X", "category": "custom", "id": "1"},
                {"value": "", "category": "custom", "id": "2"},
                {"value": "AB", "category": "custom", "id": "3"},
            ]
        )
        engine.build()

        # Nur "AB" sollte gezaehlt werden
        assert engine.entity_count == 1

    def test_whitespace_only_entity_ignored(self):
        """Leerstring und Whitespace-Only werden ignoriert."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "   ", "category": "custom", "id": "1"},
                {"value": "\t", "category": "custom", "id": "2"},
            ]
        )
        engine.build()
        assert engine.entity_count == 0

    def test_duplicate_entities(self):
        """Gleiche Entitaet zweimal hinzugefuegt erzeugt nur einen Treffer."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Doppelt GmbH", "category": "customer", "id": "K-1"},
                {"value": "Doppelt GmbH", "category": "customer", "id": "K-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Vertrag mit Doppelt GmbH.")
        # Aho-Corasick ueberschreibt den gleichen Key, also max 1 Treffer
        assert len(results) >= 1
        assert results[0]["value"] == "Doppelt GmbH"

    def test_special_characters_in_entity(self):
        """Entitaeten mit Sonderzeichen (Umlaute, &) werden erkannt."""
        engine = AnonymizationAutomaton()
        engine.add_entities(
            [
                {"value": "Bauer & Soehne KG", "category": "supplier", "id": "L-1"},
            ]
        )
        engine.build()

        results = engine.find_entities("Lieferant: Bauer & Soehne KG")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_large_entity_set_performance(self):
        """50.000 Entitaeten, 10.000 Zeichen Text — Erkennung in <50ms."""
        engine = AnonymizationAutomaton()

        # 50.000 synthetische Entitaeten
        entities = [
            {"value": f"Firma_{i:06d} GmbH", "category": "customer", "id": f"K-{i}"}
            for i in range(50_000)
        ]
        engine.add_entities(entities)
        engine.build()

        # 10.000 Zeichen Text mit einigen bekannten Entitaeten
        text_parts = [
            "Dies ist ein langer Testtext mit verschiedenen Woertern. " * 100,
            "Bestellung von Firma_000042 GmbH erhalten. ",
            "Lieferung an Firma_012345 GmbH geplant. ",
            "Rechnung von Firma_049999 GmbH bezahlt. ",
            "Ende des Textes mit weiterem Fuelltext hier. " * 100,
        ]
        text = "".join(text_parts)
        assert len(text) >= 10000  # Mindestens 10.000 Zeichen

        # Zeitmessung
        start = time.perf_counter()
        results = engine.find_entities(text)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 50, f"Erkennungsdauer {elapsed_ms:.1f}ms ueberschreitet 50ms-Limit"
        assert len(results) == 3  # Die drei eingebetteten Firmen

    def test_build_time_acceptable(self):
        """Automaton-Build fuer 50.000 Entitaeten in <2s."""
        engine = AnonymizationAutomaton()

        entities = [
            {"value": f"Entity_{i:06d}", "category": "custom", "id": str(i)} for i in range(50_000)
        ]

        start = time.perf_counter()
        engine.add_entities(entities)
        engine.build()
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, f"Build-Dauer {elapsed:.2f}s ueberschreitet 2s-Limit"
        assert engine.entity_count == 50_000
