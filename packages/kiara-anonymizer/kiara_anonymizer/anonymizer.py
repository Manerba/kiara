"""Anonymizer — konsolidierte Anonymisierungs- und De-Anonymisierungsfassade.

Orchestriert AnonymizationAutomaton (Aho-Corasick), RegexPatternMatcher und
optionalen LLM-PII-Detector. Haelt Forward- und Reverse-Mapping
ausschliesslich im RAM. Nach Session-Ende ist das Mapping unwiederbringlich
verloren.

Dreistufige Erkennungs-Pipeline:
    1. Aho-Corasick (Stammdaten):  Schnelle Erkennung ueber ERP-/LDAP-Entities.
       Pseudonyme sind bei add_entry() vorregistriert.
    2. Regex-Patterns:  Erkennt strukturierte PII (IBAN, Telefon, Email, IPv4,
       Steuer-ID) in noch nicht abgedeckten Bereichen.
    3. LLM-PII-Erkennung (opt-in):  Erkennt PII per LLM-Analyse.

Platzhalter-Format:
    Einheitlich ``‹ANON:Prefix_NNN›`` (z.B. ``‹ANON:Person_001›``, ``‹ANON:IBAN_002›``).
"""

import logging
import re
import unicodedata
from dataclasses import dataclass, field

from kiara_anonymizer.automaton import (
    _PSEUDONYM_PREFIXES,
    _REPORTING_CATEGORY,
    AnonymizationAutomaton,
    normalize_entity_type,
)
from kiara_anonymizer.base import BaseAnonymizer
from kiara_anonymizer.patterns import RegexPatternMatcher

logger = logging.getLogger(__name__)

_PSEUDONYM_PATTERN = re.compile(r"‹ANON:[A-Za-z0-9_]+_\d+›")


@dataclass
class AnonymizationResult:
    """Ergebnis einer Anonymisierung."""

    clean_text: str
    mapping: dict[str, str] = field(default_factory=dict)
    entities_found: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class Anonymizer(BaseAnonymizer):
    """Konsolidierter Anonymizer (absorbiert den ehemaligen Pseudonymizer).

    Zustandsbehaftet: Eine Instanz akkumuliert Platzhalter-Zuordnungen
    in _reverse_map und _category_counters. Fuer eine neue, unabhaengige
    Anonymisierung muss eine neue Instanz erzeugt werden.

    Verwendung:
        >>> from kiara.anonymization.stammdaten import build_automaton_from_sources
        >>> from kiara.anonymization.anonymizer import Anonymizer
        >>> automaton = build_automaton_from_sources(config)
        >>> anon = Anonymizer(automaton, config)
        >>> clean = anon.anonymize("Mail an Mueller GmbH, IBAN DE89370400440532013000")
        >>> anon.deanonymize(clean)
        'Mail an Mueller GmbH, IBAN DE89370400440532013000'
    """

    def __init__(self, automaton: AnonymizationAutomaton, config: dict, llm_detector=None):
        self.automaton = automaton
        self.config = config
        self._llm_detector = llm_detector
        self._reverse_map: dict[str, str] = {}
        # Counter aus Automaton initialisieren (kollisionsfreie Platzhalter)
        self._category_counters: dict[str, int] = dict(automaton._pseudonym_counters)
        # Separater Zaehler fuer tatsaechliche Runtime-Matches (Audit-Stats)
        self._match_counters: dict[str, int] = {}
        self._last_result: AnonymizationResult | None = None
        self._ac_base_surfaces: dict[str, str] = {}
        self._ac_surface_placeholders: dict[tuple[str, str], str] = {}

        cfg = config if isinstance(config, dict) else {}
        anon_cfg = cfg.get("anonymization", {})
        if not isinstance(anon_cfg, dict):
            anon_cfg = {}
        self._excluded_terms = set(
            t.lower() for t in anon_cfg.get("excluded_terms", []) if isinstance(t, str)
        )
        self._min_term_length = anon_cfg.get("min_term_length", 3)

        # Regex-Matcher mit optionalen Custom-Patterns
        custom_patterns = anon_cfg.get("custom_patterns", [])
        self._regex = RegexPatternMatcher(custom_patterns or None)

    @property
    def reverse_map(self) -> dict[str, str]:
        """Public accessor for the pseudonym->original mapping (read-only copy)."""
        return dict(self._reverse_map)

    @property
    def last_result(self) -> AnonymizationResult | None:
        """Vollstaendiges Ergebnis des letzten anonymize()-Aufrufs."""
        return self._last_result

    @property
    def stats(self) -> dict:
        """Statistiken fuer Audit-Logging."""
        return {
            "total_entities": len(self._reverse_map),
            "entity_types": dict(self._match_counters),
        }

    def get_stats(self) -> dict:
        """Statistik: Wie viele Entitaeten geladen, pro Kategorie."""
        return {
            "entity_count": self.automaton.entry_count,
            "categories": dict(self.automaton._pseudonym_counters),
            "loaded": self.automaton._finalized,
        }

    def anonymize(self, text: str) -> str:
        """Anonymisiert Text in drei Stufen.

        1. Aho-Corasick-Scan: Ersetzt bekannte Stammdaten
        2. Regex-Auffangnetz: Ersetzt strukturierte Muster
        3. LLM-PII-Erkennung (opt-in): Erkennt PII per LLM

        Das vollstaendige AnonymizationResult ist nach dem Aufruf ueber
        die Property ``last_result`` abrufbar.
        """
        if not isinstance(text, str):
            logger.warning(
                "anonymize() erhielt %s statt str — gebe Leerstring zurueck (Caller-Bug?)",
                type(text).__name__,
            )
            return ""
        text = unicodedata.normalize("NFC", text)

        all_entities = []
        masked_positions = set()
        replacements = []
        mapping: dict[str, str] = {}

        # -- Stufe 1: Aho-Corasick --
        ac_matches = self.automaton.scan(text)

        for match in ac_matches:
            entry = match["entry"]
            original_text = text[match["start"] : match["end"]]

            # AC-Post-Filter
            if original_text.lower() in self._excluded_terms:
                continue
            if len(original_text) < self._min_term_length:
                continue

            pseudonym = self._placeholder_for_ac_match(entry, original_text)
            replacements.append((match["start"], match["end"], pseudonym))
            masked_positions.add((match["start"], match["end"]))
            self._reverse_map[pseudonym] = original_text
            mapping[pseudonym] = original_text
            self._match_counters[entry.entity_type] = (
                self._match_counters.get(entry.entity_type, 0) + 1
            )
            all_entities.append(
                {
                    "value": original_text,
                    "category": entry.reporting_category,
                    "start": match["start"],
                    "end": match["end"],
                    "source": "ac",
                }
            )

        # -- Stufe 2: Regex-Auffangnetz --
        # Regex OHNE AC-Abdeckung laufen lassen, Konflikte danach aufloesen.
        # Grund: AC-Teiltreffer (z.B. "john") duerfen nicht verhindern, dass
        # Regex die vollstaendige E-Mail "john.doe@example.com" maskiert.
        regex_matches = self._regex.find_patterns(text)

        for rm in regex_matches:
            rm_start, rm_end = rm["start"], rm["end"]

            # Ueberlappung mit AC-Matches pruefen
            overlapping_ac = [(s, e) for s, e in masked_positions if s < rm_end and rm_start < e]

            if overlapping_ac:
                # AC deckt Regex-Match vollstaendig ab → Regex ueberspringen
                if any(s <= rm_start and rm_end <= e for s, e in overlapping_ac):
                    continue
                # Regex nur verwenden wenn er ALLE ueberlappenden
                # AC-Matches vollstaendig enthaelt (strikt breiter)
                if not all(rm_start <= s and e <= rm_end for s, e in overlapping_ac):
                    # Teilueberlappung → sicherheitshalber ueberspringen
                    continue
                # Regex ist breiter → ueberlappende AC-Matches entfernen
                for ac_span in overlapping_ac:
                    masked_positions.discard(ac_span)
                    removed = [r for r in replacements if (r[0], r[1]) == ac_span]
                    replacements = [r for r in replacements if (r[0], r[1]) != ac_span]
                    for _, _, ps in removed:
                        if not any(r[2] == ps for r in replacements):
                            mapping.pop(ps, None)
                            # _reverse_map ist append-only ueber Session: ein
                            # Pseudonym kann aus einem frueheren anonymize()-
                            # Aufruf stammen, dessen Roundtrip noch gebraucht
                            # wird. Nur das call-lokale mapping shrinkt.
                    all_entities = [
                        ent for ent in all_entities if (ent["start"], ent["end"]) != ac_span
                    ]

            entity_type = rm["entity_type"]  # Already UPPERCASE from patterns.py
            pseudonym = self._generate_placeholder(entity_type)
            replacements.append((rm_start, rm_end, pseudonym))
            masked_positions.add((rm_start, rm_end))
            self._reverse_map[pseudonym] = rm["value"]
            mapping[pseudonym] = rm["value"]
            reporting_cat = _REPORTING_CATEGORY.get(entity_type, "custom")
            self._match_counters[entity_type] = self._match_counters.get(entity_type, 0) + 1
            all_entities.append(
                {
                    "value": rm["value"],
                    "category": reporting_cat,
                    "start": rm_start,
                    "end": rm_end,
                    "source": "regex",
                }
            )

        # -- Stufe 3: LLM-PII-Erkennung --
        if self._llm_detector is not None:
            try:
                covered_list = list(masked_positions)
                llm_detections = self._llm_detector.detect(text, covered_list)
                for det in llm_detections:
                    # Overlap-Check gegen bereits maskierte Positionen
                    if any(s < det.end and det.start < e for s, e in masked_positions):
                        continue
                    entity_type = normalize_entity_type(det.category)
                    pseudonym = self._generate_placeholder(entity_type)
                    replacements.append((det.start, det.end, pseudonym))
                    masked_positions.add((det.start, det.end))
                    self._reverse_map[pseudonym] = det.value
                    mapping[pseudonym] = det.value
                    reporting_cat = _REPORTING_CATEGORY.get(entity_type, "custom")
                    self._match_counters[entity_type] = self._match_counters.get(entity_type, 0) + 1
                    all_entities.append(
                        {
                            "value": det.value,
                            "category": reporting_cat,
                            "start": det.start,
                            "end": det.end,
                            "source": "llm",
                            "confidence": det.confidence,
                        }
                    )
            except Exception as exc:
                logger.warning("LLM-PII-Erkennung fehlgeschlagen: %s", exc)

        # -- Ersetzung (von hinten nach vorne) --
        replacements.sort(key=lambda r: r[0], reverse=True)
        clean_text = text
        for start, end, replacement in replacements:
            clean_text = clean_text[:start] + replacement + clean_text[end:]

        # Statistiken
        by_category: dict[str, int] = {}
        for e in all_entities:
            cat = e["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        result = AnonymizationResult(
            clean_text=clean_text,
            mapping=mapping,
            entities_found=all_entities,
            stats={"total": len(all_entities), "by_category": by_category},
        )
        self._last_result = result
        return clean_text

    def deanonymize(self, text: str) -> str:
        """Ersetzt alle Pseudonyme durch Klarnamen via positions-basiertem Scan.

        Single-Pass-Regex-Substitution: Originaltexte, die Pseudonym-Patterns
        enthalten, werden nicht erneut aufgeloest (keine Mapping-Ketten),
        Mehrfach-Vorkommen werden separat positional ersetzt.
        """
        if not isinstance(text, str):
            return ""
        return _PSEUDONYM_PATTERN.sub(
            lambda m: self._reverse_map.get(m.group(0), m.group(0)),
            text,
        )

    def _generate_placeholder(self, entity_type: str) -> str:
        """Erzeugt einen nummerierten Platzhalter fuer Regex-/LLM-Treffer."""
        # Counter pro Prefix (nicht pro entity_type), damit Aliase wie
        # INTERNAL_CODE und CREDENTIAL (beide \u2192 "Zugangsdaten") nicht
        # kollidieren und identische Pseudonyme erzeugen.
        prefix = _PSEUDONYM_PREFIXES.get(entity_type, "Eintrag")
        count = self._category_counters.get(prefix, 0) + 1
        self._category_counters[prefix] = count
        return f"\u2039ANON:{prefix}_{count:03d}\u203a"

    def _placeholder_for_ac_match(self, entry, original_text: str) -> str:
        """Returns a stable placeholder for an AC surface form."""
        key = (entry.pseudonym, original_text)
        if key in self._ac_surface_placeholders:
            return self._ac_surface_placeholders[key]

        if entry.pseudonym not in self._ac_base_surfaces:
            pseudonym = entry.pseudonym
            self._ac_base_surfaces[entry.pseudonym] = original_text
        else:
            pseudonym = self._generate_placeholder(entry.entity_type)

        self._ac_surface_placeholders[key] = pseudonym
        return pseudonym
