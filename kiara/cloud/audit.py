"""Protokolliert alle Cloud-KI-Anfragen fuer DSGVO-Compliance.

Speichert in Tabelle 'cloud_audit_log'.
WICHTIG: Speichert NUR den anonymisierten Text, NIEMALS das Original.
"""

import json
import logging
from datetime import datetime, timedelta

from kiara.repositories.audit import CloudAuditLogRepository

logger = logging.getLogger(__name__)

# Module-level Repo-Instanz
_cloud_audit_repo = CloudAuditLogRepository()

# Kosten-Schaetzungen pro 1K Tokens (EUR, grob)
_COST_PER_1K = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "claude-sonnet-4-6": {"input": 0.003, "output": 0.015},
    "claude-haiku-4-5-20251001": {"input": 0.0008, "output": 0.004},
    "claude-opus-4-6": {"input": 0.015, "output": 0.075},
}

_COST_FAMILY_FALLBACKS = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "haiku": {"input": 0.0008, "output": 0.004},
    "sonnet": {"input": 0.003, "output": 0.015},
    "opus": {"input": 0.015, "output": 0.075},
}


def _stats_since_date(period: str, now: datetime | None = None) -> str:
    """Startdatum fuer Audit-Statistiken als YYYY-MM-DD."""
    current = now or datetime.now()
    if period == "month":
        return current.replace(day=1).strftime("%Y-%m-%d")
    days = {"day": 1, "week": 7}.get(period, 30)
    return (current - timedelta(days=days)).strftime("%Y-%m-%d")


def _costs_for_model(model_name: str) -> dict | None:
    """Kostenprofil fuer ein Modell oder None, wenn die Kosten unbekannt sind."""
    costs = _COST_PER_1K.get(model_name)
    if costs is not None:
        return costs
    model_l = (model_name or "").lower()
    for family, family_costs in _COST_FAMILY_FALLBACKS.items():
        if family in model_l:
            return family_costs
    return None


class AuditLogger:
    """Protokolliert alle Cloud-KI-Anfragen fuer DSGVO-Compliance.

    Speichert in Tabelle 'cloud_audit_log'.
    """

    def __init__(self, config: dict):
        self._config = config
        budget = config.get("cloud", {}).get("budget", {})
        self._monthly_limit = budget.get("monthly_limit_eur", 50.0)
        self._alert_pct = budget.get("alert_threshold_pct", 80)

    def log_request(
        self,
        user_id: int,
        question_anonymized: str,
        answer_anonymized: str,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        entities_anonymized: int,
        duration_ms: int,
        context_sources: list[dict] | None = None,
        correlation_id: str = "",
    ) -> int:
        """Loggt einen Cloud-Request. Gibt audit_id zurueck.

        WICHTIG: Speichert NUR den anonymisierten Text, NIEMALS das Original.

        Args:
            context_sources: Optionale Liste von RAG-Kontext-Metadaten
                (Dateiname, Datasource, Chunk-Anzahl). Kein Volltext.
        """
        try:
            sources_json = (
                json.dumps(context_sources, ensure_ascii=False) if context_sources else None
            )
            kwargs = dict(
                user_id=user_id,
                question_anonymized=question_anonymized,
                answer_anonymized=answer_anonymized,
                provider=provider,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                entities_anonymized=entities_anonymized,
                duration_ms=duration_ms,
                context_sources=sources_json,
            )
            if correlation_id:
                kwargs["correlation_id"] = correlation_id
            audit_id = _cloud_audit_repo.create(**kwargs)
            return audit_id
        except Exception as exc:
            logger.critical(
                "Cloud-Audit-Log Persistenz fehlgeschlagen: %s",
                exc,
                exc_info=True,
            )
            raise RuntimeError("Cloud-Audit-Log Persistenz fehlgeschlagen") from exc

    def get_log(
        self, limit: int = 100, offset: int = 0, user_id: int = None, date_from: str = None
    ) -> list[dict]:
        """Liest Audit-Log mit optionaler Filterung.

        Args:
            limit: Max. Anzahl Eintraege.
            offset: Offset fuer Paginierung.
            user_id: Optional: nur Eintraege eines Users.
            date_from: Optional: nur Eintraege ab Datum (ISO-Format).

        Returns:
            Liste von Audit-Eintraegen.
        """
        try:
            return _cloud_audit_repo.list_with_user(
                user_id=user_id,
                date_from=date_from,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            logger.error("Audit-Log Abfrage fehlgeschlagen: %s", exc)
            return []

    def get_stats(self, period: str = "month") -> dict:
        """Aggregierte Statistiken: Requests/Tag, Token-Verbrauch, Kosten-Schaetzung.

        Args:
            period: 'day', 'week' oder 'month'.

        Returns:
            Dict mit Statistiken.
        """
        since = _stats_since_date(period)

        try:
            row = _cloud_audit_repo.get_aggregate_stats(since)
            daily = _cloud_audit_repo.get_daily_counts(since)
            by_model = _cloud_audit_repo.get_by_model(since)
        except Exception as exc:
            logger.error("Audit-Statistik fehlgeschlagen: %s", exc)
            return {}

        stats = dict(row) if row else {}
        stats["period"] = period
        stats["since"] = since
        stats["daily"] = daily
        stats["by_model"] = by_model

        # Kosten-Schaetzung
        total_cost = 0.0
        unknown_cost_models: set[str] = set()
        for m in by_model:
            model_name = m["model"]
            costs = _costs_for_model(model_name)
            if costs is None:
                logger.warning(
                    "Unbekanntes Modell '%s' in Kostenberechnung — "
                    "Kosten werden als unbekannt markiert",
                    model_name,
                )
                unknown_cost_models.add(model_name)
                continue
            cost = (m["tokens_in"] or 0) / 1000 * costs["input"] + (
                m["tokens_out"] or 0
            ) / 1000 * costs["output"]
            total_cost += cost

        stats["estimated_cost_eur"] = round(total_cost, 4)
        stats["estimated_cost_unknown"] = bool(unknown_cost_models)
        stats["unknown_cost_models"] = sorted(unknown_cost_models)
        stats["budget_limit_eur"] = self._monthly_limit
        stats["budget_used_pct"] = round(
            (total_cost / self._monthly_limit * 100) if self._monthly_limit > 0 else 0, 1
        )

        return stats

    def check_budget(self) -> tuple[bool, float]:
        """Prueft ob das monatliche Budget noch ausreicht.

        Returns:
            Tuple (erlaubt, verbleibend_eur).
        """
        stats = self.get_stats("month")
        cost = stats.get("estimated_cost_eur", 0)
        remaining = self._monthly_limit - cost
        allowed = remaining > 0 and not stats.get("estimated_cost_unknown", False)
        return allowed, round(remaining, 4)
