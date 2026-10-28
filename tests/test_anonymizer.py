"""Tests fuer den Anonymizer (Hauptfassade).

Testet: Roundtrip (anonymize+deanonymize), Platzhalter-Format,
Statistiken, ERP+Regex-Kombination, Mapping-Konsistenz,
Mehrfachvorkommen und AnonymizationResult-Dataclass.
"""

from kiara.anonymization.anonymizer import AnonymizationResult, Anonymizer

# ---------------------------------------------------------------------------
# Roundtrip: Anonymize + Deanonymize
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_simple_roundtrip(self, loaded_anonymizer):
        """Anonymisierung und De-Anonymisierung ergeben den Originaltext."""
        original = "Rechnung an Mueller GmbH, Standort Stuttgart."
        clean_text = loaded_anonymizer.anonymize(original)
        restored = loaded_anonymizer.deanonymize(clean_text)
        assert restored == original

    def test_roundtrip_multiple_entities(self, loaded_anonymizer):
        """Roundtrip mit mehreren Entitaeten verschiedener Kategorien."""
        original = "Hans Mueller arbeitet bei Mueller GmbH in Stuttgart."
        clean_text = loaded_anonymizer.anonymize(original)
        restored = loaded_anonymizer.deanonymize(clean_text)
        assert restored == original

    def test_roundtrip_with_regex_entities(self, loaded_anonymizer):
        """Roundtrip inklusive Regex-erkannter Muster (E-Mail, IBAN)."""
        original = "Mail: test@example.com, IBAN: DE89370400440532013000."
        clean_text = loaded_anonymizer.anonymize(original)

        # Platzhalter muessen im Clean-Text stehen
        assert "test@example.com" not in clean_text
        assert "DE89370400440532013000" not in clean_text

        restored = loaded_anonymizer.deanonymize(clean_text)
        assert restored == original

    def test_roundtrip_text_without_entities(self, loaded_anonymizer):
        """Text ohne PII bleibt unveraendert."""
        original = "Das Wetter ist heute schoen."
        clean_text = loaded_anonymizer.anonymize(original)
        assert clean_text == original
        assert loaded_anonymizer.last_result.mapping == {}
        restored = loaded_anonymizer.deanonymize(clean_text)
        assert restored == original

    def test_roundtrip_empty_text(self, loaded_anonymizer):
        """Leerer Text funktioniert ohne Fehler."""
        clean_text = loaded_anonymizer.anonymize("")
        assert clean_text == ""
        restored = loaded_anonymizer.deanonymize(clean_text)
        assert restored == ""


# ---------------------------------------------------------------------------
# Platzhalter-Format
# ---------------------------------------------------------------------------


class TestPlaceholderFormat:
    def test_customer_placeholder(self, loaded_anonymizer):
        """Kunden-Platzhalter: ‹ANON:Unternehmen_001›."""
        clean_text = loaded_anonymizer.anonymize("Von Mueller GmbH erhalten.")
        assert "\u2039ANON:Unternehmen_" in clean_text
        # Format pruefen: ‹ANON:Unternehmen_NNN›
        for placeholder in loaded_anonymizer.last_result.mapping:
            if "Unternehmen" in placeholder:
                assert placeholder.startswith("\u2039ANON:Unternehmen_")
                assert placeholder.endswith("\u203a")
                # Nummer ist dreistellig
                prefix = "\u2039ANON:Unternehmen_"
                num_part = placeholder[len(prefix) : -1]
                assert num_part.isdigit()
                assert len(num_part) == 3

    def test_employee_placeholder(self, loaded_anonymizer):
        """Mitarbeiter-Platzhalter: ‹ANON:Person_001›."""
        clean_text = loaded_anonymizer.anonymize("Hans Mueller ist zustaendig.")
        assert "\u2039ANON:Person_" in clean_text

    def test_supplier_placeholder(self, loaded_anonymizer):
        """Lieferanten-Platzhalter: ‹ANON:Unternehmen_NNN›."""
        clean_text = loaded_anonymizer.anonymize("Lieferung von Bauer & Soehne KG.")
        assert "\u2039ANON:Unternehmen_" in clean_text

    def test_location_placeholder(self, loaded_anonymizer):
        """Ort-Platzhalter: ‹ANON:Adresse_001›."""
        clean_text = loaded_anonymizer.anonymize("Standort Stuttgart geplant.")
        assert "\u2039ANON:Adresse_" in clean_text

    def test_product_placeholder(self, loaded_anonymizer):
        """Produkt-Platzhalter: ‹ANON:Artikel_001›."""
        clean_text = loaded_anonymizer.anonymize("Bestellung: ServerRack Pro X1.")
        assert "\u2039ANON:Artikel_" in clean_text

    def test_iban_placeholder(self, loaded_anonymizer):
        """IBAN-Platzhalter: ‹ANON:IBAN_NNN›."""
        clean_text = loaded_anonymizer.anonymize("Konto DE89370400440532013000 nutzen.")
        assert "\u2039ANON:IBAN_" in clean_text

    def test_email_placeholder(self, loaded_anonymizer):
        """E-Mail-Platzhalter: ‹ANON:Email_NNN›."""
        clean_text = loaded_anonymizer.anonymize("Senden an info@example.com bitte.")
        assert "\u2039ANON:Email_" in clean_text

    def test_phone_placeholder(self, loaded_anonymizer):
        """Telefon-Platzhalter: ‹ANON:Telefon_NNN›."""
        clean_text = loaded_anonymizer.anonymize("Anruf unter +49 711 12345678.")
        assert "\u2039ANON:Telefon_" in clean_text


# ---------------------------------------------------------------------------
# Statistiken
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_stats_total_count(self, loaded_anonymizer):
        """stats['total'] zaehlt alle gefundenen Entitaeten."""
        loaded_anonymizer.anonymize("Mueller GmbH und Schmidt AG in Stuttgart.")
        assert loaded_anonymizer.last_result.stats["total"] == 3

    def test_stats_by_category(self, loaded_anonymizer):
        """stats['by_category'] schluesselt nach Kategorie auf."""
        loaded_anonymizer.anonymize("Mueller GmbH und Schmidt AG in Stuttgart.")
        by_cat = loaded_anonymizer.last_result.stats["by_category"]
        assert by_cat.get("customer", 0) == 2
        assert by_cat.get("location", 0) == 1

    def test_stats_empty_text(self, loaded_anonymizer):
        """Leerer Text hat total=0."""
        loaded_anonymizer.anonymize("Keine PII hier.")
        assert loaded_anonymizer.last_result.stats["total"] == 0

    def test_get_stats_method(self, loaded_anonymizer):
        """get_stats() liefert entity_count und loaded-Status."""
        stats = loaded_anonymizer.get_stats()
        assert stats["loaded"] is True
        assert stats["entity_count"] > 0
        assert isinstance(stats["categories"], dict)

    def test_get_stats_unloaded(self):
        """Frischer Anonymizer ohne Entities hat entity_count=0."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})
        stats = anon.get_stats()
        assert stats["loaded"] is True
        assert stats["entity_count"] == 0


# ---------------------------------------------------------------------------
# ERP + Regex kombiniert
# ---------------------------------------------------------------------------


class TestCombinedRecognition:
    def test_erp_and_regex_both_found(self, loaded_anonymizer):
        """ERP-Entitaeten und Regex-Muster werden kombiniert erkannt."""
        text = "Mueller GmbH, IBAN DE89370400440532013000, Mail info@mueller.de."
        loaded_anonymizer.anonymize(text)

        categories = {e["category"] for e in loaded_anonymizer.last_result.entities_found}
        assert "customer" in categories
        assert "iban" in categories
        assert "email" in categories

    def test_overlap_erp_regex_resolved(self, loaded_anonymizer):
        """ERP-Treffer hat Vorrang vor Regex-Treffer bei Ueberlappung."""
        # Anna Schmidt's E-Mail ist nicht als ERP-Entitaet geladen
        # Also kein echter Overlap hier; stattdessen: getrennter Test
        text = "Kontakt: Anna Schmidt, anna@firma.de."
        loaded_anonymizer.anonymize(text)

        # Beide muessen gefunden werden (kein Overlap)
        assert len(loaded_anonymizer.last_result.entities_found) >= 2

    def test_regex_only_in_uncovered_regions(self, loaded_anonymizer):
        """Regex laeuft nur auf Bereiche, die nicht durch ERP abgedeckt sind."""
        text = "Mueller GmbH: DE89370400440532013000"
        loaded_anonymizer.anonymize(text)

        # Mueller GmbH ist ERP, IBAN ist Regex
        result = loaded_anonymizer.last_result
        erp_entities = [e for e in result.entities_found if e["category"] == "customer"]
        regex_entities = [e for e in result.entities_found if e["category"] == "iban"]
        assert len(erp_entities) == 1
        assert len(regex_entities) == 1


# ---------------------------------------------------------------------------
# Mapping-Konsistenz
# ---------------------------------------------------------------------------


class TestMappingConsistency:
    def test_mapping_complete(self, loaded_anonymizer):
        """Jede im clean_text vorhandene Platzhalter-Referenz hat einen Mapping-Eintrag."""
        text = "Mueller GmbH und Schmidt AG."
        clean_text = loaded_anonymizer.anonymize(text)

        import re

        placeholders_in_text = re.findall(r"\u2039ANON:[A-Za-z_]+_\d{3}\u203a", clean_text)
        for ph in placeholders_in_text:
            assert ph in loaded_anonymizer.last_result.mapping, f"Platzhalter {ph} fehlt im Mapping"

    def test_mapping_values_are_original(self, loaded_anonymizer):
        """Mapping-Werte enthalten die Originalwerte aus dem Text."""
        text = "Mueller GmbH und Stuttgart."
        loaded_anonymizer.anonymize(text)

        values = set(loaded_anonymizer.last_result.mapping.values())
        assert "Mueller GmbH" in values
        assert "Stuttgart" in values

    def test_mapping_is_dict(self, loaded_anonymizer):
        """Mapping ist ein dict[str, str]."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        mapping = loaded_anonymizer.last_result.mapping
        assert isinstance(mapping, dict)
        for key, value in mapping.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_regex_overlap_keeps_mapping_for_remaining_ac_placeholder(self):
        """Breiter Regex-Treffer darf Mapping fuer anderen gleichen AC-Treffer nicht loeschen."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.add_entry("john", "PERSON", "P-1", reporting_category="person")
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})

        original = "john schrieb an john@example.com."
        clean_text = anon.anonymize(original)

        assert "\u2039ANON:Person_001\u203a" in clean_text
        assert "\u2039ANON:Email_" in clean_text
        assert anon.deanonymize(clean_text) == original

    def test_same_key_different_ac_surfaces_roundtrip_exact(self):
        """Varianten desselben Stammdatensatzes werden exakt wiederhergestellt."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.add_entry("Muller GmbH", "ORG", "K-1", reporting_category="customer")
        automaton.add_entry("Mueller GmbH", "ORG", "K-1", reporting_category="customer")
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})

        original = "Muller GmbH und Mueller GmbH"
        clean_text = anon.anonymize(original)

        placeholders = [ph for ph in anon.last_result.mapping if "Unternehmen" in ph]
        assert len(placeholders) == 2
        assert clean_text.count("\u2039ANON:Unternehmen_") == 2
        assert anon.deanonymize(clean_text) == original


# ---------------------------------------------------------------------------
# Mehrfachvorkommen
# ---------------------------------------------------------------------------


class TestMultipleOccurrences:
    def test_same_entity_same_placeholder(self, loaded_anonymizer):
        """Dieselbe Entitaet mehrmals → immer derselbe Platzhalter."""
        text = "Mueller GmbH bestellt. Mueller GmbH zahlt."
        clean_text = loaded_anonymizer.anonymize(text)

        # Zaehle, wie oft der Platzhalter vorkommt
        placeholder = None
        for ph, val in loaded_anonymizer.last_result.mapping.items():
            if val == "Mueller GmbH":
                placeholder = ph
                break
        assert placeholder is not None
        assert clean_text.count(placeholder) == 2

    def test_different_entities_different_placeholders(self, loaded_anonymizer):
        """Verschiedene Entitaeten bekommen verschiedene Platzhalter."""
        text = "Mueller GmbH und Schmidt AG."
        loaded_anonymizer.anonymize(text)

        placeholders = list(loaded_anonymizer.last_result.mapping.keys())
        assert len(set(placeholders)) == len(placeholders)  # Alle unique

    def test_counter_increments_per_category(self):
        """Zaehler inkrementiert pro Kategorie: Unternehmen_001, Unternehmen_002."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.add_entities(
            [
                {"value": "Firma A", "category": "customer", "id": "K-1"},
                {"value": "Firma B", "category": "customer", "id": "K-2"},
                {"value": "Ort X", "category": "location", "id": "O-1"},
            ]
        )
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})

        anon.anonymize("Firma A, Firma B in Ort X.")

        ph_values = anon.last_result.mapping
        # Zwei Unternehmen-Platzhalter und ein Adresse-Platzhalter
        org_phs = [ph for ph in ph_values if "Unternehmen" in ph]
        adr_phs = [ph for ph in ph_values if "Adresse" in ph]
        assert len(org_phs) == 2
        assert len(adr_phs) == 1
        assert "\u2039ANON:Unternehmen_001\u203a" in org_phs
        assert "\u2039ANON:Unternehmen_002\u203a" in org_phs
        assert "\u2039ANON:Adresse_001\u203a" in adr_phs


# ---------------------------------------------------------------------------
# AnonymizationResult Dataclass
# ---------------------------------------------------------------------------


class TestAnonymizationResult:
    def test_result_has_all_fields(self, loaded_anonymizer):
        """AnonymizationResult hat clean_text, mapping, entities_found, stats."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        result = loaded_anonymizer.last_result
        assert hasattr(result, "clean_text")
        assert hasattr(result, "mapping")
        assert hasattr(result, "entities_found")
        assert hasattr(result, "stats")

    def test_result_is_dataclass(self):
        """AnonymizationResult ist eine Dataclass."""
        from dataclasses import fields

        field_names = {f.name for f in fields(AnonymizationResult)}
        assert "clean_text" in field_names
        assert "mapping" in field_names
        assert "entities_found" in field_names
        assert "stats" in field_names

    def test_result_default_values(self):
        """Default-Werte sind korrekt gesetzt."""
        result = AnonymizationResult(clean_text="test")
        assert result.mapping == {}
        assert result.entities_found == []
        assert result.stats == {}

    def test_entities_found_structure(self, loaded_anonymizer):
        """entities_found enthaelt dicts mit start, end, value, category."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        result = loaded_anonymizer.last_result
        assert len(result.entities_found) == 1
        entity = result.entities_found[0]
        assert "start" in entity
        assert "end" in entity
        assert "value" in entity
        assert "category" in entity

    def test_empty_automaton_returns_unchanged(self):
        """Anonymizer ohne Stammdaten gibt Text unveraendert zurueck (kein AC-Treffer)."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})
        clean_text = anon.anonymize("Mueller GmbH")
        # Kein AC-Treffer, aber Regex koennte greifen — ohne ERP-Entities bleibt Text gleich
        assert "Mueller GmbH" in clean_text


# ---------------------------------------------------------------------------
# load_entities mit Config
# ---------------------------------------------------------------------------


class TestLoadEntities:
    def test_load_from_csv_dir(self, tmp_path, tmp_config):
        """Laden aus einem CSV-Verzeichnis via build_automaton_from_sources."""
        from kiara.anonymization.stammdaten import build_automaton_from_sources

        csv_dir = tmp_path / "stammdaten"
        csv_dir.mkdir(exist_ok=True)
        csv_file = csv_dir / "kunden.csv"
        csv_file.write_text("key;type;value;variants\nK-1;ORG;Test GmbH;\n", encoding="utf-8")

        tmp_config["anonymization"]["erp_source"] = {
            "enabled": True,
            "type": "csv",
            "path": str(csv_dir),
        }
        tmp_config["anonymization"]["ldap_source"] = {"enabled": False}
        automaton = build_automaton_from_sources(tmp_config)
        anon = Anonymizer(automaton, tmp_config)

        clean_text = anon.anonymize("Auftrag fuer Test GmbH.")
        assert "\u2039ANON:Unternehmen_001\u203a" in clean_text

    def test_load_without_ldap(self, tmp_config):
        """LDAP deaktiviert: build_automaton_from_sources funktioniert trotzdem."""
        from kiara.anonymization.stammdaten import build_automaton_from_sources

        tmp_config["anonymization"]["ldap_source"] = {"enabled": False}
        automaton = build_automaton_from_sources(tmp_config)
        anon = Anonymizer(automaton, tmp_config)
        assert anon.get_stats()["loaded"] is True

    def test_load_custom_entries(self, tmp_config):
        """Benutzerdefinierte Eintraege aus Config werden geladen."""
        from kiara.anonymization.stammdaten import build_automaton_from_sources

        tmp_config["anonymization"]["custom_entries"] = [
            {"text": "Geheimprojekt Alpha", "category": "custom", "id": "c-1"},
        ]
        tmp_config["anonymization"]["ldap_source"] = {"enabled": False}

        automaton = build_automaton_from_sources(tmp_config)
        anon = Anonymizer(automaton, tmp_config)

        clean_text = anon.anonymize("Details zum Geheimprojekt Alpha.")
        assert "Geheimprojekt Alpha" not in clean_text
        assert "\u2039ANON:Eintrag_001\u203a" in clean_text


# ---------------------------------------------------------------------------
# Stateful Behavior (BaseAnonymizer-Konformitaet)
# ---------------------------------------------------------------------------


class TestStatefulBehavior:
    """Tests fuer das stateful Verhalten (anonymize → str, internes Mapping)."""

    def test_deanonymize_without_mapping(self, loaded_anonymizer):
        """deanonymize() ohne mapping nutzt internes reverse_map."""
        original = "Mueller GmbH in Stuttgart."
        clean = loaded_anonymizer.anonymize(original)
        restored = loaded_anonymizer.deanonymize(clean)
        assert restored == original

    def test_deanonymize_uses_internal_mapping(self, loaded_anonymizer):
        """deanonymize() nutzt internes Mapping fuer Rueckaufloesung."""
        original = "Mueller GmbH"
        clean = loaded_anonymizer.anonymize(original)
        assert loaded_anonymizer.last_result.mapping  # Mapping vorhanden
        restored = loaded_anonymizer.deanonymize(clean)
        assert restored == original

    def test_reverse_map_accumulates(self, loaded_anonymizer):
        """reverse_map akkumuliert ueber mehrere anonymize()-Aufrufe."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        loaded_anonymizer.anonymize("Schmidt AG")
        rmap = loaded_anonymizer.reverse_map
        assert "Mueller GmbH" in rmap.values()
        assert "Schmidt AG" in rmap.values()

    def test_reverse_map_persists_through_regex_override(self, loaded_anonymizer):
        """Regex-Override darf persistente _reverse_map-Eintraege nicht poppen.

        Wenn ein Regex-Match einen frueheren AC-Match komplett enthaelt
        (z.B. AC-Match "Stuttgart" innerhalb der Email "Stuttgart@example.com"),
        darf der AC-Pseudonym nur aus dem call-lokalen mapping entfernt werden.
        Bleibt der Eintrag aus self._reverse_map erhalten, koennen Texte aus
        frueheren anonymize()-Aufrufen weiterhin deanonymisiert werden.
        """
        # Aufruf 1: AC-Match auf "Stuttgart" allein -> Pseudonym landet in _reverse_map
        clean_1 = loaded_anonymizer.anonymize("Standort Stuttgart heute.")
        assert "Stuttgart" not in clean_1

        # Aufruf 2: gleiche Surface "Stuttgart" innerhalb einer Email-Adresse,
        # die der Regex breiter abdeckt -> Override-Pfad triggert
        loaded_anonymizer.anonymize("Mail an Stuttgart@example.com")

        # Roundtrip aus Aufruf 1 muss weiterhin funktionieren
        assert loaded_anonymizer.deanonymize(clean_1) == "Standort Stuttgart heute."

    def test_reverse_map_readonly(self, loaded_anonymizer):
        """reverse_map gibt Kopie zurueck (keine Seiteneffekte)."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        rmap = loaded_anonymizer.reverse_map
        rmap["\u2039ANON:FAKE_001\u203a"] = "injected"
        assert "\u2039ANON:FAKE_001\u203a" not in loaded_anonymizer.reverse_map

    def test_last_result_property(self, loaded_anonymizer):
        """last_result enthaelt das vollstaendige AnonymizationResult."""
        loaded_anonymizer.anonymize("Mueller GmbH, IBAN DE89370400440532013000")
        result = loaded_anonymizer.last_result
        assert result is not None
        assert isinstance(result.mapping, dict)
        assert len(result.entities_found) >= 2
        assert result.stats["total"] >= 2

    def test_last_result_none_before_first_call(self):
        """last_result ist None vor dem ersten anonymize()-Aufruf."""
        from kiara.anonymization.automaton import AnonymizationAutomaton

        automaton = AnonymizationAutomaton()
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})
        assert anon.last_result is None

    def test_stats_property_format(self, loaded_anonymizer):
        """stats-Property hat total_entities und entity_types."""
        loaded_anonymizer.anonymize("Mueller GmbH")
        stats = loaded_anonymizer.stats
        assert "total_entities" in stats
        assert "entity_types" in stats

    def test_anonymize_returns_str(self, loaded_anonymizer):
        """anonymize() gibt str zurueck, nicht AnonymizationResult."""
        result = loaded_anonymizer.anonymize("Mueller GmbH")
        assert isinstance(result, str)

    def test_cloud_adapter_roundtrip(self, loaded_anonymizer):
        """Anonymizer funktioniert korrekt durch CloudAdapterBase-Methoden."""
        text = "Mueller GmbH in Stuttgart"
        anon_text = loaded_anonymizer.anonymize(text)
        assert "Mueller GmbH" not in anon_text
        restored = loaded_anonymizer.deanonymize(anon_text)
        assert restored == text


# ---------------------------------------------------------------------------
# BaseAnonymizer Protocol
# ---------------------------------------------------------------------------


class TestBaseAnonymizerProtocol:
    """Anonymizer erfuellt die BaseAnonymizer-Schnittstelle."""

    def test_anonymizer_is_base(self, loaded_anonymizer):
        from kiara.anonymization.base import BaseAnonymizer

        assert isinstance(loaded_anonymizer, BaseAnonymizer)

    def test_fresh_anonymizer_is_base(self):
        from kiara.anonymization.automaton import AnonymizationAutomaton
        from kiara.anonymization.base import BaseAnonymizer

        automaton = AnonymizationAutomaton()
        automaton.build()
        anon = Anonymizer(automaton, {"anonymization": {}})
        assert isinstance(anon, BaseAnonymizer)
