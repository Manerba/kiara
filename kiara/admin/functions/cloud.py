"""Admin-Funktionen fuer Cloud-KI-Konfiguration."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.enums import RoutingMode

logger = logging.getLogger(__name__)


@admin_function(
    name="cloud.config.read",
    category="cloud",
    description="Cloud-KI-Konfiguration anzeigen",
    help="Zeigt die aktuelle Cloud-KI-Konfiguration (Provider, Routing, Budget). "
    "Der API-Key wird maskiert.",
    examples=[
        "Zeige Cloud-KI-Konfiguration",
        "Wie ist die Cloud-KI konfiguriert?",
        "Cloud-Einstellungen?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Cloud-KI-Konfiguration (API-Key maskiert)",
    raw_response=True,
    http=("GET", "/api/admin/cloud/config"),
)
async def cloud_config_read(user: dict) -> AdminResult:
    from kiara.config import load_config
    from kiara.shared.constants import (
        CLOUD_BUDGET_ALERT_THRESHOLD_PCT,
        CLOUD_MONTHLY_BUDGET_EUR,
        CLOUD_ROUTING_CONFIDENCE_THRESHOLD,
        CLOUD_ROUTING_MIN_CHUNKS,
        DEFAULT_LLM_TEMPERATURE,
        DEFAULT_MAX_TOKENS,
        DEFAULT_OPENAI_MODEL,
    )

    config = load_config()
    cloud = config.get("cloud", {})
    routing = cloud.get("routing", {})
    budget = cloud.get("budget", {})

    return AdminResult.ok(
        "Cloud-KI-Konfiguration",
        data={
            "config": {
                "enabled": cloud.get("enabled", False),
                "provider": cloud.get("provider", "openai"),
                "api_key": PASSWORD_MASK if cloud.get("api_key") else "",
                "model": cloud.get("model", DEFAULT_OPENAI_MODEL),
                "base_url": cloud.get("base_url", ""),
                "max_tokens": cloud.get("max_tokens", DEFAULT_MAX_TOKENS),
                "temperature": cloud.get("temperature", DEFAULT_LLM_TEMPERATURE),
                "routing_mode": cloud.get("mode", routing.get("mode", RoutingMode.AUTO.value)),
                "confidence_threshold": routing.get(
                    "confidence_threshold", CLOUD_ROUTING_CONFIDENCE_THRESHOLD
                ),
                "min_relevant_chunks": routing.get("min_relevant_chunks", CLOUD_ROUTING_MIN_CHUNKS),
                "cloud_keywords": routing.get("cloud_keywords", []),
                "monthly_limit_eur": budget.get("monthly_limit_eur", CLOUD_MONTHLY_BUDGET_EUR),
                "alert_threshold_pct": budget.get(
                    "alert_threshold_pct", CLOUD_BUDGET_ALERT_THRESHOLD_PCT
                ),
            }
        },
    )


@admin_function(
    name="cloud.config.update",
    category="cloud",
    description="Cloud-KI-Konfiguration speichern",
    help="Speichert die Cloud-KI-Konfiguration. Bei maskiertem API-Key "
    "(********) wird der bestehende Key beibehalten. Klartext-Keys "
    "werden opportunistisch verschlüsselt.",
    examples=[
        "Aktiviere Cloud-KI mit Provider anthropic",
        "Cloud-Konfiguration speichern",
        "Setze Cloud-Budget auf 50 EUR",
    ],
    parameters={
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean", "description": "Cloud-KI aktiviert"},
            "provider": {"type": "string", "description": "KI-Provider (openai, anthropic)"},
            "api_key": {
                "type": "string",
                "description": "API-Schluessel (PASSWORD_MASK = beibehalten)",
            },
            "model": {"type": "string", "description": "LLM-Modellname"},
            "base_url": {"type": "string", "description": "Custom Base-URL"},
            "max_tokens": {"type": "integer", "description": "Max. Token pro Anfrage"},
            "temperature": {"type": "number", "description": "LLM-Temperatur (0.0-2.0)"},
            "routing_mode": {
                "type": "string",
                "enum": ["auto", "local", "cloud"],
                "description": "Routing-Modus (auto, local, cloud)",
            },
            "confidence_threshold": {
                "type": "number",
                "description": "Confidence-Schwelle fuer Cloud-Routing",
            },
            "min_relevant_chunks": {
                "type": "integer",
                "description": "Min. relevante Chunks fuer Routing",
            },
            "cloud_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cloud-Routing Schluesselwoerter",
            },
            "monthly_limit_eur": {"type": "number", "description": "Monatliches Budget in EUR"},
            "alert_threshold_pct": {"type": "integer", "description": "Budget-Alert-Schwelle (%)"},
        },
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="cloud-konfiguration aktualisieren",
    changelog_area="cloud_config",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/cloud/config"),
)
async def cloud_config_update(
    user: dict,
    enabled: bool | None = None,
    provider: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    routing_mode: str | None = None,
    confidence_threshold: float | None = None,
    min_relevant_chunks: int | None = None,
    cloud_keywords: list[str] | None = None,
    monthly_limit_eur: float | None = None,
    alert_threshold_pct: int | None = None,
) -> AdminResult:
    from kiara.config import load_config, save_config
    from kiara.shared.constants import (
        CLOUD_BUDGET_ALERT_THRESHOLD_PCT,
        CLOUD_MONTHLY_BUDGET_EUR,
        CLOUD_ROUTING_CONFIDENCE_THRESHOLD,
        CLOUD_ROUTING_MIN_CHUNKS,
        DEFAULT_LLM_TEMPERATURE,
        DEFAULT_MAX_TOKENS,
        DEFAULT_OPENAI_MODEL,
    )
    from kiara.shared.crypto import encrypt, is_encrypted

    valid_modes = {mode.value for mode in RoutingMode}
    if routing_mode is not None and routing_mode not in valid_modes:
        return AdminResult.fail("Ungueltiger Cloud-Routing-Modus. Erlaubt sind: auto, local, cloud")

    config = load_config()
    current_cloud = config.get("cloud", {})
    if not isinstance(current_cloud, dict):
        current_cloud = {}

    cloud = dict(current_cloud)
    routing = dict(cloud.get("routing", {}) or {})
    budget = dict(cloud.get("budget", {}) or {})

    def _default(container: dict, key: str, value):
        if container.get(key) is None:
            container[key] = value

    if enabled is not None:
        cloud["enabled"] = enabled
    _default(cloud, "enabled", False)

    if provider is not None:
        cloud["provider"] = provider
    _default(cloud, "provider", "openai")

    if api_key == PASSWORD_MASK:
        resolved_key = current_cloud.get("api_key", "")
        # Alten Klartext-Key opportunistisch verschluesseln
        if resolved_key and not is_encrypted(resolved_key):
            resolved_key = encrypt(resolved_key)
        cloud["api_key"] = resolved_key
    elif api_key is not None:
        resolved_key = encrypt(api_key) if api_key else ""
        cloud["api_key"] = resolved_key
    _default(cloud, "api_key", "")

    if model is not None:
        cloud["model"] = model
    _default(cloud, "model", DEFAULT_OPENAI_MODEL)

    if base_url is not None:
        cloud["base_url"] = base_url
    _default(cloud, "base_url", "")

    if max_tokens is not None:
        cloud["max_tokens"] = max_tokens
    _default(cloud, "max_tokens", DEFAULT_MAX_TOKENS)

    if temperature is not None:
        cloud["temperature"] = temperature
    _default(cloud, "temperature", DEFAULT_LLM_TEMPERATURE)

    if routing_mode is not None:
        effective_mode = routing_mode
    else:
        effective_mode = cloud.get("mode") or routing.get("mode") or RoutingMode.AUTO.value
    cloud["mode"] = effective_mode
    routing["mode"] = effective_mode

    if confidence_threshold is not None:
        routing["confidence_threshold"] = confidence_threshold
    _default(routing, "confidence_threshold", CLOUD_ROUTING_CONFIDENCE_THRESHOLD)

    if min_relevant_chunks is not None:
        routing["min_relevant_chunks"] = min_relevant_chunks
    _default(routing, "min_relevant_chunks", CLOUD_ROUTING_MIN_CHUNKS)

    if cloud_keywords is not None:
        routing["cloud_keywords"] = cloud_keywords
    _default(routing, "cloud_keywords", [])

    if monthly_limit_eur is not None:
        budget["monthly_limit_eur"] = monthly_limit_eur
    _default(budget, "monthly_limit_eur", CLOUD_MONTHLY_BUDGET_EUR)

    if alert_threshold_pct is not None:
        budget["alert_threshold_pct"] = alert_threshold_pct
    _default(budget, "alert_threshold_pct", CLOUD_BUDGET_ALERT_THRESHOLD_PCT)

    cloud["routing"] = routing
    cloud["budget"] = budget
    config["cloud"] = cloud

    save_config(config)
    logger.info("Cloud-Konfiguration gespeichert von %s", user["username"])
    return AdminResult.ok(
        "Cloud-Konfiguration gespeichert",
        data={"message": "Cloud-Konfiguration gespeichert"},
    )


@admin_function(
    name="cloud.audit",
    category="cloud",
    description="Cloud-Audit-Log anzeigen",
    help="Zeigt das Cloud-KI-Audit-Log mit Filterung und Paginierung.",
    examples=[
        "Zeige Cloud-Audit-Log",
        "Cloud-Anfragen der letzten Woche",
        "Welche Cloud-Anfragen hat Benutzer 3 gestellt?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (Standard: 1)", "default": 1},
            "per_page": {
                "type": "integer",
                "description": "Eintraege pro Seite (max 100)",
                "default": 20,
            },
            "user_id": {"type": "integer", "description": "Filter nach Benutzer-ID"},
            "date_from": {"type": "string", "description": "Filter ab Datum (YYYY-MM-DD)"},
        },
    },
    returns="Paginiertes Cloud-Audit-Log",
    raw_response=True,
    http=("GET", "/api/admin/cloud/audit"),
)
async def cloud_audit(
    user: dict,
    page: int = 1,
    per_page: int = 20,
    user_id: int | None = None,
    date_from: str | None = None,
) -> AdminResult:
    from kiara.config import load_config
    from kiara.shared.db import db_query

    config = load_config()
    from kiara.cloud.audit import AuditLogger

    audit = AuditLogger(config)

    offset = (page - 1) * per_page
    entries = audit.get_log(
        limit=per_page,
        offset=offset,
        user_id=user_id,
        date_from=date_from,
    )

    try:
        conditions: list[str] = []
        params: list = []
        if user_id is not None:
            conditions.append("user_id = %s")
            params.append(user_id)
        if date_from:
            conditions.append("created_at >= %s")
            params.append(date_from)
        sql = "SELECT COUNT(*) AS count FROM cloud_audit_log"
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        with db_query() as db:
            total = db.execute(sql, params).fetchone()["count"]
    except Exception:
        total = len(entries)

    return AdminResult.ok(
        f"{len(entries)} Cloud-Audit-Eintraege",
        data={
            "data": entries,
            "total": total,
            "page": page,
        },
    )


@admin_function(
    name="cloud.stats",
    category="cloud",
    description="Cloud-Nutzungsstatistiken anzeigen",
    help="Zeigt aggregierte Cloud-KI-Statistiken: Requests/Tag, "
    "Token-Verbrauch, Kosten-Schätzung und Budget-Status.",
    examples=[
        "Cloud-Statistiken",
        "Wie viele Cloud-Tokens wurden verbraucht?",
        "Cloud-Kosten diesen Monat",
    ],
    parameters={
        "type": "object",
        "properties": {
            "period": {
                "type": "string",
                "description": "Zeitraum (month, week, day)",
                "default": "month",
            },
        },
    },
    returns="Cloud-Nutzungsstatistiken (12 Felder)",
    raw_response=True,
    http=("GET", "/api/admin/cloud/stats"),
)
async def cloud_stats(user: dict, period: str = "month") -> AdminResult:
    from kiara.config import load_config

    config = load_config()
    from kiara.cloud.audit import AuditLogger

    audit = AuditLogger(config)
    stats = audit.get_stats(period)

    return AdminResult.ok("Cloud-Statistiken", data=stats)


@admin_function(
    name="cloud.test",
    category="cloud",
    description="Cloud-API-Verbindungstest",
    help="Sendet eine Test-Anfrage an die Cloud-API um die Verbindung und den API-Key zu prüfen.",
    examples=[
        "Teste Cloud-API-Verbindung",
        "Funktioniert der Cloud-API-Key?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "provider": {
                "type": "string",
                "description": "Cloud-Provider (openai, anthropic)",
                "default": "openai",
            },
            "api_key": {"type": "string", "description": "API-Schluessel (Pflichtfeld)"},
            "model": {"type": "string", "description": "Modellname (optional)"},
            "base_url": {"type": "string", "description": "Custom Base-URL (optional)"},
        },
        "required": ["api_key"],
    },
    returns="Verbindungstest-Ergebnis (success, response, model, provider)",
    raw_response=True,
    http=("POST", "/api/admin/cloud/test"),
)
async def cloud_test(
    user: dict,
    api_key: str,
    provider: str = "openai",
    model: str = "",
    base_url: str = "",
) -> AdminResult:
    if not api_key:
        return AdminResult.fail("API-Key ist erforderlich")

    if provider not in {"openai", "anthropic"}:
        return AdminResult.fail("Ungueltiger Provider. Erlaubt sind: openai, anthropic")

    from kiara.cloud.client import CloudKIClient
    from kiara.shared.constants import DEFAULT_CLOUD_MODEL, DEFAULT_OPENAI_MODEL

    client = CloudKIClient(
        provider=provider,
        api_key=api_key,
        model=model or (DEFAULT_OPENAI_MODEL if provider == "openai" else DEFAULT_CLOUD_MODEL),
        base_url=base_url,
        max_tokens=10,
    )

    try:
        response = await client.generate(
            prompt="Antworte mit einem einzigen Wort: OK",
            temperature=0.0,
        )
        return AdminResult.ok(
            "Cloud-Verbindung erfolgreich",
            data={
                "success": True,
                "response": response.strip(),
                "model": client.model,
                "provider": client.provider,
            },
        )
    except Exception as exc:
        return AdminResult.fail(f"Cloud-Verbindung fehlgeschlagen: {exc}")
