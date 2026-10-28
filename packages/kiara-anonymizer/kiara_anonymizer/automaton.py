"""Aho-Corasick-Automat fuer deterministische Anonymisierung.

Durchsucht Text in einem einzigen Durchlauf nach allen bekannten
Stammdaten (ERP + LDAP + Custom). Performance: <1ms pro Chunk.

Pseudonyme werden beim add_entry() vergeben (nicht erst beim Scan).
Wird von stammdaten.build_automaton_from_sources() befuellt und vom
konsolidierten Anonymizer (anonymizer.py) genutzt.

Wortgrenzen-Pruefung und Overlap-Resolution sind in _text_utils.py
ausgelagert.

Compatibility-API:
    add_entities(), find_entities() und entity_count stehen als
    Kompatibilitaetsschicht bereit, damit die bisherige
    AnonymizationEngine-API erhalten bleibt.
"""

import logging
import unicodedata
from dataclasses import dataclass

import ahocorasick

from kiara_anonymizer._text_utils import _is_word_boundary, _resolve_overlaps

logger = logging.getLogger(__name__)


# -- Pseudonym-Praefix-Tabelle ------------------------------------------------

_PSEUDONYM_PREFIXES = {
    "PERSON": "Person",
    "ORG": "Unternehmen",
    "PRODUCT": "Artikel",
    "ADDRESS": "Adresse",
    "EMAIL": "Email",
    "PHONE": "Telefon",
    "IBAN": "IBAN",
    "TAX_ID": "SteuerID",
    "IP_ADDRESS": "IP_Adresse",
    "INTERNAL_CODE": "Zugangsdaten",
    "CREDENTIAL": "Zugangsdaten",
    "CUSTOM": "Eintrag",
}

# -- Reporting-Category: entity_type (UPPERCASE) → oeffentlicher lowercase-Key -

_REPORTING_CATEGORY = {
    "PERSON": "person",
    "ORG": "organization",
    "ADDRESS": "address",
    "PRODUCT": "product",
    "EMAIL": "email",
    "PHONE": "phone",
    "IBAN": "iban",
    "TAX_ID": "tax_id",
    "IP_ADDRESS": "ip_address",
    "INTERNAL_CODE": "credential",
    "CREDENTIAL": "credential",
    "CUSTOM": "custom",
}

# -- Typ-Normalisierung: lowercase-Kategorie → UPPERCASE entity_type ----------

_NORMALIZE_TYPE = {
    "org": "ORG",
    "customer": "ORG",
    "supplier": "ORG",
    "organization": "ORG",
    "employee": "PERSON",
    "person": "PERSON",
    "location": "ADDRESS",
    "address": "ADDRESS",
    "product": "PRODUCT",
    "iban": "IBAN",
    "tax_id": "TAX_ID",
    "email": "EMAIL",
    "phone": "PHONE",
    "ip_address": "IP_ADDRESS",
    "credential": "INTERNAL_CODE",
    "internal_code": "INTERNAL_CODE",
    "custom": "CUSTOM",
}


def normalize_entity_type(category: str) -> str:
    """Normalisiert eine Kategorie auf UPPERCASE entity_type.

    Unbekannte Kategorien fallen auf CUSTOM zurueck.
    """
    return _NORMALIZE_TYPE.get(category.lower(), "CUSTOM") if category else "CUSTOM"


def _normalize_nfc_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """NFC-normalisiert Text und mappt NFC-Positionen auf Original-Offsets."""
    if unicodedata.is_normalized("NFC", text):
        return text, list(range(len(text))), list(range(1, len(text) + 1))

    normalized_parts: list[str] = []
    start_offsets: list[int] = []
    end_offsets: list[int] = []
    cluster: list[str] = []
    cluster_start = 0
    cluster_end = 0

    def flush_cluster() -> None:
        if not cluster:
            return
        normalized = unicodedata.normalize("NFC", "".join(cluster))
        for ch in normalized:
            normalized_parts.append(ch)
            start_offsets.append(cluster_start)
            end_offsets.append(cluster_end)

    for idx, ch in enumerate(text):
        if not cluster:
            cluster = [ch]
            cluster_start = idx
            cluster_end = idx + 1
            continue
        if unicodedata.combining(ch):
            cluster.append(ch)
            cluster_end = idx + 1
            continue
        flush_cluster()
        cluster = [ch]
        cluster_start = idx
        cluster_end = idx + 1

    flush_cluster()
    return "".join(normalized_parts), start_offsets, end_offsets


def _casefold_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    """Casefold mit Offset-Map von gefoldeten Positionen zu Original-Positionen.

    Noetig weil casefold() Laengen aendern kann (z.B. ß → ss).
    start_offsets/end_offsets mappen gefoldete Positionen auf den Eingabetext.
    """
    folded_parts: list[str] = []
    start_offsets: list[int] = []
    end_offsets: list[int] = []
    for orig_idx, ch in enumerate(text):
        for fc in ch.casefold():
            folded_parts.append(fc)
            start_offsets.append(orig_idx)
            end_offsets.append(orig_idx + 1)
    return "".join(folded_parts), start_offsets, end_offsets


@dataclass
class AnonymizationEntry:
    """Ein Eintrag im Anonymisierungs-Woerterbuch."""

    original: str
    entity_type: str
    erp_key: str
    pseudonym: str
    reporting_category: str = ""
    min_length: int = 3


class AnonymizationAutomaton:
    """Aho-Corasick-Automat fuer deterministische Anonymisierung."""

    def __init__(self, word_boundaries: bool = True):
        self._automaton = ahocorasick.Automaton()
        self._entries_by_key: dict[str, AnonymizationEntry] = {}
        self._entries_by_surface: dict[str, AnonymizationEntry] = {}
        self._pseudonym_counters: dict[str, int] = {}
        self._finalized = False
        self._word_boundaries = word_boundaries

    def add_entry(
        self, text: str, entity_type: str, erp_key: str, reporting_category: str | None = None
    ) -> bool:
        """Fuegt einen Stammdaten-Eintrag zum Automaten hinzu.

        Mehrere Texte mit demselben erp_key erhalten dasselbe Pseudonym.

        Args:
            text: Der Klartext-Wert (z.B. "Mueller GmbH").
            entity_type: UPPERCASE Typ (z.B. "PERSON", "ORG").
            erp_key: Eindeutiger Schluessel (Pseudonym-Recycling).
            reporting_category: Optionaler lowercase Reporting-Key.
                Falls None, wird ueber _REPORTING_CATEGORY bestimmt.

        Returns:
            True wenn der Eintrag aufgenommen wurde, False wenn uebersprungen.
        """
        if not isinstance(text, str) or not text.strip():
            return False
        text = unicodedata.normalize("NFC", text.strip())
        if len(text) < 2:
            return False
        text_lower = text.casefold()
        if len(text_lower) < 2:
            return False

        if erp_key in self._entries_by_key:
            pseudonym = self._entries_by_key[erp_key].pseudonym
        else:
            # Counter pro Prefix (nicht pro entity_type), damit Aliase wie
            # INTERNAL_CODE und CREDENTIAL (beide \u2192 "Zugangsdaten") nicht
            # kollidieren und identische Pseudonyme erzeugen.
            prefix = _PSEUDONYM_PREFIXES.get(entity_type, "Eintrag")
            count = self._pseudonym_counters.get(prefix, 0) + 1
            self._pseudonym_counters[prefix] = count
            pseudonym = f"\u2039ANON:{prefix}_{count:03d}\u203a"

        if reporting_category is None:
            reporting_category = _REPORTING_CATEGORY.get(entity_type, "custom")

        entry = AnonymizationEntry(
            original=text,
            entity_type=entity_type,
            erp_key=erp_key,
            pseudonym=pseudonym,
            reporting_category=reporting_category,
            min_length=len(text_lower),
        )

        existing_surface = self._entries_by_surface.get(text_lower)
        if existing_surface and existing_surface.erp_key != erp_key:
            logger.warning(
                "Ambigue Anonymizer-Surface uebersprungen: %r existiert bereits fuer %s, neu waere %s",
                text,
                existing_surface.erp_key,
                erp_key,
            )
            return False

        if erp_key not in self._entries_by_key:
            self._entries_by_key[erp_key] = entry
        self._entries_by_surface[text_lower] = entry
        self._automaton.add_word(text_lower, entry)
        self._finalized = False
        return True

    def build(self) -> None:
        """Finalisiert den Automaten. Muss nach allen add_entry-Aufrufen erfolgen."""
        if len(self._automaton) > 0:
            self._automaton.make_automaton()
        self._finalized = True

    def scan(self, text: str) -> list[dict]:
        """Durchsucht den Text nach allen bekannten Stammdaten.

        Returns:
            Liste von Treffern: [{"start": int, "end": int, "entry": AnonymizationEntry}]
        """
        if not self._finalized:
            raise RuntimeError("Automaton not built. Call build() first.")

        if not isinstance(text, str):
            return []

        if len(self._automaton) == 0:
            return []

        original_text = text
        text, nfc_start_offsets, nfc_end_offsets = _normalize_nfc_with_offsets(text)
        text_folded, fold_start_offsets, fold_end_offsets = _casefold_with_offsets(text)
        raw_matches = []

        for end_idx, entry in self._automaton.iter(text_folded):
            # ahocorasick liefert end_idx als *inklusive* letzte Position des Treffers,
            # daher: start = end - laenge + 1, end_excl = end + 1
            key_len = entry.min_length
            start_idx = end_idx - key_len + 1
            end_idx + 1

            # Positionen aus dem casefold-Raum zurueck in den Original-Raum mappen
            nfc_start = fold_start_offsets[start_idx]
            nfc_end = fold_end_offsets[end_idx]
            if nfc_end <= 0 or nfc_end > len(nfc_end_offsets):
                continue
            orig_start = nfc_start_offsets[nfc_start]
            orig_end = nfc_end_offsets[nfc_end - 1]

            if self._word_boundaries and not _is_word_boundary(original_text, orig_start, orig_end):
                continue

            raw_matches.append(
                {
                    "start": orig_start,
                    "end": orig_end,
                    "entry": entry,
                }
            )

        return _resolve_overlaps(raw_matches)

    @property
    def entry_count(self) -> int:
        return len(self._automaton)

    # -- Compatibility-API (ehem. AnonymizationEngine) -------------------------

    @property
    def entity_count(self) -> int:
        """Alias fuer entry_count (Kompatibilitaet mit AnonymizationEngine)."""
        return self.entry_count

    def add_entities(self, entities: list[dict]) -> None:
        """Batch-API: Fuegt Entitaeten im engine.py-Format hinzu.

        Format: [{"value": "Mueller GmbH", "category": "customer", "id": "K-1234"}, ...]
        """
        for entity in entities:
            value = entity.get("value", "").strip()
            if not value or len(value) < 2:
                continue

            raw_category = entity.get("category", "custom")
            entity_type = normalize_entity_type(raw_category)
            value_nfc = unicodedata.normalize("NFC", value)
            erp_key = entity.get("id", "") or f"_auto_{entity_type}_{value_nfc.casefold()}"

            self.add_entry(
                text=value,
                entity_type=entity_type,
                erp_key=erp_key,
                reporting_category=raw_category,
            )

    def find_entities(self, text: str) -> list[dict]:
        """Findet alle Entitaeten im Text (engine.py-kompatibles Format).

        Returns:
            [{"start": 5, "end": 16, "value": "Mueller GmbH",
              "category": "customer", "id": "K-1234"}, ...]
        """
        matches = self.scan(text)
        result = []
        for match in matches:
            entry = match["entry"]
            # Original-Casing aus dem Quelltext
            original_text = text[match["start"] : match["end"]]
            result.append(
                {
                    "start": match["start"],
                    "end": match["end"],
                    "value": original_text,
                    "category": entry.reporting_category,
                    "id": entry.erp_key,
                }
            )
        return result
