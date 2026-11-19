"""Anonymisierung aller MCP-Ausgaben (Retrieval, query) vor der Rueckgabe an den Client.

MCP-Clients sind externe Modelle (z.B. Claude Code): Was ein Werkzeug
zurueckgibt, geht beim Anbieter in die Inferenz. Fuer sie gelten deshalb
dieselben Quellenregeln wie fuer Cloud-Backends:

- ``blocked``-Datenquellen erscheinen nicht in MCP-Ausgaben.
- Treffer aus ``anonymize``-Datenquellen werden mit derselben Engine und
  denselben Stammdaten wie der Cloud-Pfad anonymisiert (``‹ANON:Typ_NNN›``).
- Treffer, die nur aus ``public``-Datenquellen stammen, bleiben im Klartext.

Fail-closed: Laesst sich der Anonymizer nicht bauen oder sind keine Stammdaten
geladen, wird die Anfrage abgelehnt statt im Klartext beantwortet. Je Anfrage
mit Anonymisierung entsteht ein Eintrag in ``audit_anonymization`` (Route
``mcp:<werkzeug>``, Anzahl und Typen, ``correlation_id``); dieselben Zahlen
stehen im Feld ``anonymization`` der Antwort, damit ein Mitschnitt der
Modellanfragen gegen das Audit geprueft werden kann.

Schalter: ``web.mcp_output_cloud_policy`` (Default true). ``anonymization.enabled``
false schaltet wie beim Cloud-Pfad nur die Anonymisierung ab, nicht die Sperre
von ``blocked``-Quellen.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_REF_PREFIX = "src_"

# Technische Felder, die nie Personenbezug tragen und unveraendert bleiben muessen.
_SKIP_KEYS = frozenset(
    {
        "id",
        "source_ref",
        "correlation_id",
        "route",
        "tool",
        "schema",
        "score_type",
        "embedding_model",
        "file_type",
        "code",
        "status",
        "text_mode",
        "scope",
        "anonymization",
    }
)


class McpAnonymizationUnavailable(RuntimeError):
    """Anonymisierung waere noetig, ist aber nicht verfuegbar (fail-closed)."""


def mcp_output_policy_enabled(config: dict) -> bool:
    web = config.get("web") if isinstance(config, dict) else None
    return not (isinstance(web, dict) and web.get("mcp_output_cloud_policy") is False)


def anonymization_enabled(config: dict) -> bool:
    anon = config.get("anonymization") if isinstance(config, dict) else None
    return not (isinstance(anon, dict) and anon.get("enabled") is False)


def datasource_restriction_levels() -> dict[int, str]:
    """restriction_level je Datenquelle (alle, auch inaktive)."""
    from kiara.shared.db import db_query

    with db_query() as db:
        rows = db.execute("SELECT id, restriction_level FROM datasources").fetchall()
    return {int(r["id"]): str(r["restriction_level"] or "anonymize") for r in rows}


def source_ref(source: str) -> str:
    """Opaker, stabiler Verweis auf eine Quelle (Pfade koennen Namen enthalten)."""
    digest = hashlib.sha256(source.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"{SOURCE_REF_PREFIX}{digest[:40]}"


def is_source_ref(value: str) -> bool:
    rest = value[len(SOURCE_REF_PREFIX) :]
    return (
        value.startswith(SOURCE_REF_PREFIX)
        and len(rest) == 40
        and all(c in "0123456789abcdef" for c in rest)
    )


def resolve_source_ref(ref: str, datasource_id: int) -> str | None:
    """source_ref einer Datenquelle zurueck in den Quellpfad uebersetzen."""
    from kiara.shared.db import db_query

    with db_query() as db:
        rows = db.execute(
            "SELECT DISTINCT file_path FROM file_hashes WHERE datasource_id = %s",
            (datasource_id,),
        ).fetchall()
    for row in rows:
        path = row["file_path"]
        if path and source_ref(path) == ref:
            return path
    return None


class McpOutputAnonymizer:
    """Ein Anonymizer je MCP-Anfrage: gleiche Werte erhalten gleiche Platzhalter."""

    def __init__(self, config: dict, levels: dict[int, str] | None = None):
        self._config = config
        self._levels = levels
        self._anonymizer = None
        self.enabled = mcp_output_policy_enabled(config) and anonymization_enabled(config)

    # --- Quellenregeln ---

    @property
    def levels(self) -> dict[int, str]:
        if self._levels is None:
            self._levels = datasource_restriction_levels()
        return self._levels

    def blocked_ids(self) -> set[int]:
        if not mcp_output_policy_enabled(self._config):
            return set()
        return {ds_id for ds_id, level in self.levels.items() if level == "blocked"}

    def level_for(self, ds_ids: list[int]) -> str:
        """Strengste Regel der beteiligten Datenquellen (unbekannt = anonymize)."""
        if not mcp_output_policy_enabled(self._config):
            return "public"
        levels = {self.levels.get(int(ds_id), "anonymize") for ds_id in ds_ids}
        if "blocked" in levels:
            return "blocked"
        if levels and levels <= {"public"}:
            return "public"
        return "anonymize"

    def without_blocked(self, ds_ids: list[int] | None) -> list[int] | None:
        """Datenquellen-Filter ohne blocked-Quellen (None = alle Datenquellen)."""
        blocked = self.blocked_ids()
        if not blocked:
            return ds_ids
        if ds_ids is None:
            return sorted(ds_id for ds_id in self.levels if ds_id not in blocked)
        return [ds_id for ds_id in ds_ids if ds_id not in blocked]

    def needs_anonymization(self, ds_ids: list[int] | None) -> bool:
        """Ob eine Anfrage ueber diese Datenquellen anonymisierte Ausgabe braucht."""
        if not self.enabled:
            return False
        scope = list(self.levels) if ds_ids is None else ds_ids
        return any(self.level_for([ds_id]) != "public" for ds_id in scope)

    # --- Anonymisierung ---

    def _engine(self):
        if self._anonymizer is None:
            try:
                from kiara.anonymization.anonymizer import Anonymizer
                from kiara.anonymization.llm_detector import build_llm_detector
                from kiara.anonymization.stammdaten import cached_automaton
                from kiara.generation.backends.factory import _normalize_config_for_anonymizer

                normalized = _normalize_config_for_anonymizer(self._config)
                automaton = cached_automaton(normalized)
                if automaton is None or automaton.entry_count == 0:
                    raise RuntimeError("keine Stammdaten geladen")
                self._anonymizer = Anonymizer(
                    automaton, normalized, llm_detector=build_llm_detector(normalized)
                )
            except Exception as exc:
                logger.error("MCP-Anonymisierung nicht verfuegbar: %s", exc)
                raise McpAnonymizationUnavailable(
                    f"Anonymisierung der MCP-Ausgabe nicht verfuegbar: {exc}"
                ) from exc
        return self._anonymizer

    @property
    def used(self) -> bool:
        return self._anonymizer is not None

    def text(self, value: str) -> str:
        if not isinstance(value, str) or not value:
            return value
        return self._engine().anonymize(value)

    def tree(self, value: Any, key: str | None = None) -> Any:
        """Alle Strings einer JSON-Struktur anonymisieren (technische Felder ausgenommen)."""
        if key in _SKIP_KEYS:
            return value
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {k: self.tree(v, str(k)) for k, v in value.items()}
        if isinstance(value, list):
            return [self.tree(v) for v in value]
        return value

    def summary(self, *, raw_items: int = 0) -> dict[str, Any]:
        """Kennzahlen fuer Antwort und Audit (ohne Klartextwerte)."""
        if not self.used:
            return {
                "applied": False,
                "entity_count": 0,
                "entity_types": {},
                "raw_items": raw_items,
            }
        stats = self._anonymizer.stats
        return {
            "applied": True,
            "entity_count": int(stats.get("total_entities", 0)),
            "entity_types": dict(stats.get("entity_types", {})),
            "placeholder_format": "‹ANON:Typ_NNN›",
            "raw_items": raw_items,
        }

    def audit(self, user: dict, tool: str, query: str, correlation_id: str) -> None:
        """audit_anonymization-Eintrag; wirft bei Fehler (fail-closed)."""
        if not self.used:
            return
        from kiara.anonymization.audit import log_anonymization_event

        summary = self.summary()
        log_anonymization_event(
            user_id=user.get("id"),
            session_id=correlation_id,
            question=query,
            entity_count=summary["entity_count"],
            entity_types=summary["entity_types"],
            route=f"mcp:{tool}",
            correlation_id=correlation_id,
        )
