"""Anonymisierung: Erkennung und Pseudonymisierung sensibler Daten.

Schuetzt personenbezogene Daten (PII) vor dem Versand an Cloud-KI-Dienste.

Architektur — konsolidierte Pipeline:

    stammdaten.build_automaton_from_sources(config)
        -> AnonymizationAutomaton (automaton.py)
        -> Anonymizer(automaton, config)

    Der Anonymizer orchestriert Aho-Corasick, Regex-Auffangnetz (patterns.py)
    und optionalen LLM-PII-Detector. Pseudonyme werden beim add_entry() im
    Automaton vergeben. Regex-/LLM-Treffer erhalten nummerierte Platzhalter
    zur Laufzeit.

Beziehungen:
    - generation.backends.factory nutzt Anonymizer fuer Cloud-Backends
    - web.api_chat erstellt Anonymizer pro Session
    - web.api_ki nutzt Anonymizer fuer KI-Endpunkte
    - admin.functions.ki nutzt Anonymizer fuer Admin-KI

Module:
    automaton       — AnonymizationAutomaton (Aho-Corasick + Pseudonym-Vergabe)
    anonymizer      — Anonymizer (konsolidierte Orchestrierung: AC + Regex + LLM)
    patterns        — RegexPatternMatcher (konsolidiertes Pattern-Set)
    stammdaten      — build_automaton_from_sources (ERP/LDAP -> Automaton)
    _text_utils     — Wortgrenzen-Pruefung, Overlap-Resolution
    erp_connector   — ERP-Stammdaten-Loader
    ldap_enricher   — LDAP-Anreicherung
    llm_detector    — LLM-basierte PII-Erkennung (Schicht 4, opt-in)
    audit           — Audit-Logging fuer Anonymisierungsereignisse

Die Kernlogik ist in das Subpackage kiara-anonymizer extrahiert.
Dieses Modul stellt Compat-Shims fuer bestehende Importpfade bereit.
"""

from .anonymizer import Anonymizer
from .base import BaseAnonymizer
