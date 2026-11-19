"""Wortgrenzen und Umlaut-Umschrift im Anonymisierungs-Automaten (Hausaenderung)."""

import pytest

from kiara.anonymization.anonymizer import Anonymizer
from kiara.anonymization.automaton import AnonymizationAutomaton

CONFIG = {"anonymization": {"min_term_length": 3}}


@pytest.fixture
def anon():
    a = AnonymizationAutomaton(word_boundaries=True)
    a.add_entry(text="Musterfrau", entity_type="PERSON", erp_key="p1")
    a.add_entry(text="Musterfrau-Beispiel", entity_type="PERSON", erp_key="p3")
    a.add_entry(text="Beispielbau", entity_type="ORG", erp_key="o1")
    a.add_entry(text="Jürgen Grünwald", entity_type="PERSON", erp_key="p2")
    a.add_entry(text="Grünwald", entity_type="PERSON", erp_key="p2")
    a.build()
    return lambda text: Anonymizer(a, CONFIG).anonymize(text)


@pytest.mark.parametrize(
    "text",
    [
        "Frau Musterfrau-Lange",
        "/EDV/Musterfrau-Projekt/plan.pdf",
        "Angebot_Musterfrau.pdf",
        "20260409_2097252_Musterfrau.txt",
        "Musterfrau's Notiz",
        "Musterfrau–Bericht",
    ],
)
def test_connectors_are_boundaries(anon, text):
    result = anon(text)
    assert "Musterfrau" not in result
    assert "‹ANON:Person_" in result


def test_company_in_hyphen_compound(anon):
    assert anon("Beispielbau-Werk Nord") == "‹ANON:Unternehmen_001›-Werk Nord"


def test_longest_match_keeps_whole_double_name(anon):
    assert anon("Frau Musterfrau-Beispiel") == "Frau ‹ANON:Person_002›"


@pytest.mark.parametrize("text", ["Benzmusterfrau", "Musterfraus Büro", "Musterfrau2"])
def test_no_partial_words(anon, text):
    assert anon(text) == text


@pytest.mark.parametrize("text", ["Juergen Gruenwald", "JUERGEN GRUENWALD", "Herr Gruenwald"])
def test_umlaut_transliteration(anon, text):
    result = anon(text)
    assert "ruenwald" not in result.lower()
    assert "‹ANON:Person_" in result


def test_transliteration_counts_as_entry_and_keeps_type():
    a = AnonymizationAutomaton(word_boundaries=True)
    assert a.add_entry(text="Müller", entity_type="PERSON", erp_key="p1")
    a.add_entry(text="Maier", entity_type="PERSON", erp_key="p2")
    a.build()
    assert a.entry_count == 3  # müller, mueller, maier
    assert {m["entry"].entity_type for m in a.scan("Mueller")} == {"PERSON"}


def test_transliteration_does_not_override_other_key():
    a = AnonymizationAutomaton(word_boundaries=True)
    a.add_entry(text="Mueller", entity_type="ORG", erp_key="o1")
    a.add_entry(text="Müller", entity_type="PERSON", erp_key="p1")
    a.build()
    assert [m["entry"].erp_key for m in a.scan("Mueller")] == ["o1"]
