"""kiara-anonymizer: Eigenstaendige Anonymisierungs-Library.

Kernlogik fuer PII-Erkennung und -Anonymisierung:
    - Aho-Corasick-Automat (deterministische Stammdaten-Erkennung)
    - Regex-Patterns (strukturierte PII: IBAN, Telefon, E-Mail, ...)
    - LLM-PII-Detection (opt-in, Backend-Injection)
    - ERP-Stammdaten-Import (CSV/JSON)

Keine Abhaengigkeit auf kiara — kann eigenstaendig installiert werden.
"""

from kiara_anonymizer.anonymizer import AnonymizationResult, Anonymizer
from kiara_anonymizer.automaton import (
    AnonymizationAutomaton,
    AnonymizationEntry,
    normalize_entity_type,
)
from kiara_anonymizer.base import BaseAnonymizer
from kiara_anonymizer.erp_connector import ERPConnector
from kiara_anonymizer.llm_detector import LLMBackend, LLMDetection, LLMPIIDetector
from kiara_anonymizer.patterns import RegexPatternMatcher
