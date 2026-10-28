"""Query-Expansion fuer besseres Retrieval.

Erweitert Suchanfragen um alternative Formulierungen und Fachbegriffe,
insbesondere relevant fuer DE/EN-Terminologie in IT-Wissensbasen.
"""

import logging
import re
from typing import TYPE_CHECKING

from kiara.shared.constants import (
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_QUERY_EXPANSION_NUM_CTX,
    QUERY_EXPANSION_TIMEOUT_SECS,
)
from kiara.shared.processing_warnings import report_processing_warning

if TYPE_CHECKING:
    from kiara.generation.backends.base import LLMBackend

logger = logging.getLogger(__name__)


class QueryExpander:
    """Erweitert Suchanfragen fuer besseres Retrieval.

    Besonders relevant fuer DE<->EN-Terminologie:
    "Firewall-Regeln" -> ["Firewall-Regeln", "firewall rules", "Paketfilter", "iptables"]
    """

    def __init__(self, config: dict, backend: "LLMBackend | None" = None):
        """
        Args:
            config:  KIara-Konfiguration.
            backend: LLMBackend-Instanz fuer Query-Expansion.
        """
        self._enabled = config.get("retrieval", {}).get("query_expansion", False)
        self._backend = backend

    def expand(self, query: str) -> list[str]:
        """Generiert alternative Suchbegriffe.

        Returns:
            [original_query, expansion1, expansion2, expansion3]
            Max 4 Queries total.

        Wenn deaktiviert oder kein LLM/Backend: nur Originalquery zurueckgeben.
        """
        normalized_query = (query or "").strip()
        if not normalized_query:
            return []
        if not self._enabled:
            return [normalized_query]
        if self._backend is None:
            report_processing_warning("query_expansion", "missing_backend")
            return [normalized_query]

        if self._backend.provider != "ollama":
            logger.debug(
                "Query-Expansion uebersprungen: provider=%s (nur ollama unterstuetzt)",
                self._backend.provider,
            )
            return [normalized_query]
        try:
            prompt = self._build_expansion_prompt(normalized_query)
            response = self._backend.generate(
                "",
                prompt,
                temperature=DEFAULT_LLM_TEMPERATURE,
                num_ctx=DEFAULT_QUERY_EXPANSION_NUM_CTX,
                timeout=QUERY_EXPANSION_TIMEOUT_SECS,
            )
            expansions = self._parse_expansions(response)
            seen = {normalized_query.lower()}
            unique: list[str] = []
            for e in expansions[:3]:
                e_lower = e.lower()
                if e_lower not in seen:
                    seen.add(e_lower)
                    unique.append(e)
            result = [normalized_query, *unique]
            logger.debug("Query-Expansion: %s -> %s", normalized_query, result)
            return result
        except Exception as exc:
            logger.warning("Query-Expansion fehlgeschlagen: %s", exc)
            report_processing_warning("query_expansion")
            return [normalized_query]

    def _build_expansion_prompt(self, query: str) -> str:
        """Baut den Prompt fuer die Query-Expansion."""
        return (
            "Generiere genau 3 alternative Suchbegriffe fuer die folgende "
            "Suchanfrage. Beruecksichtige sowohl deutsche als auch englische "
            "Fachbegriffe. "
            "Gib NUR die 3 Alternativen zurueck, eine pro Zeile, ohne "
            "Nummerierung oder Erklaerung.\n\n"
            f"Suchanfrage: {query}\n\n"
            "Alternative Suchbegriffe:"
        )

    def _parse_expansions(self, response: str) -> list[str]:
        """Parst die LLM-Antwort in einzelne Suchbegriffe."""
        lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
        # Nummerierung entfernen (falls vorhanden)
        cleaned: list[str] = []
        for line in lines:
            # Entferne Praefixe wie "1.", "2)", "- ", "* ", "– ", "— ", "• " (beliebige Zahl)
            line = re.sub(r"^\d+[.)]\s*", "", line)
            line = re.sub(r"^[-–—*•]\s*", "", line)
            line = line.strip()
            if line and line != "Alternative Suchbegriffe:":
                cleaned.append(line)
        return cleaned[:3]

    @property
    def enabled(self) -> bool:
        """Ob Query-Expansion aktiviert ist."""
        return self._enabled
