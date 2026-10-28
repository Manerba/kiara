"""Abstrakte Basisklasse fuer Anonymisierung/De-Anonymisierung.

Definiert die einheitliche Schnittstelle, die sowohl Anonymizer (Standalone)
als auch Pseudonymizer (Produktion) implementieren. Dadurch koennen
Cloud-Adapter und andere Caller mit beiden Implementierungen arbeiten,
ohne hasattr-Checks oder Type-Switching.
"""

from abc import ABC, abstractmethod


class BaseAnonymizer(ABC):
    """Einheitliche Schnittstelle fuer Anonymisierung/De-Anonymisierung."""

    @abstractmethod
    def anonymize(self, text: str) -> str:
        """Anonymisiert Text. Gibt den anonymisierten Text als String zurueck."""

    @abstractmethod
    def deanonymize(self, text: str) -> str:
        """Stellt Originaltext wieder her (nutzt internes Mapping)."""

    @property
    @abstractmethod
    def reverse_map(self) -> dict[str, str]:
        """Platzhalter -> Original-Mapping (read-only Kopie)."""

    @property
    @abstractmethod
    def stats(self) -> dict:
        """Statistiken fuer Audit-Logging."""
