"""Admin-Funktionen fuer DSGVO-Compliance."""

import logging
from datetime import datetime

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


def _validate_audit_date_filter(value: str | None, field_name: str) -> AdminResult | None:
    """Validiert optionale Audit-Datumsfilter im erwarteten YYYY-MM-DD-Format."""
    if value is None:
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except (TypeError, ValueError):
        return AdminResult.fail(
            f"{field_name} muss im Format YYYY-MM-DD angegeben werden",
            http_status=400,
        )
    return None


def _load_cloud_backends_for_report() -> list[dict] | None:
    """Liest aktive Cloud-Backends fuer VVT/DSFA, ohne Reports bei DB-Problemen zu blockieren."""
    try:
        from kiara.repositories.backend import KiBackendRepository

        return KiBackendRepository().list_by_category("cloud", active_only=True)
    except Exception as exc:
        logger.warning("Cloud-Backend-Flags fuer Compliance-Report nicht lesbar: %s", exc)
        return None


@admin_function(
    name="compliance.audit",
    category="compliance",
    description="Audit-Log abfragen (paginiert, filterbar)",
    help="Zeigt alle Anfragen aus dem Query-Log. Unterstützt Pagination, "
    "Sortierung und Filter nach Benutzer, Datum, Routing-Ziel und Chain-Typ.",
    examples=[
        "Zeige das Audit-Log",
        "Audit-Log der letzten Woche",
        "Welche Anfragen hat Benutzer 5 gestellt?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (Standard: 1)", "default": 1},
            "page_size": {
                "type": "integer",
                "description": "Eintraege pro Seite (Standard: 50, max 500)",
                "default": 50,
            },
            "sort_field": {"type": "string", "description": "Sortierfeld", "default": "created_at"},
            "sort_dir": {
                "type": "string",
                "description": "Sortierrichtung",
                "enum": ["asc", "desc"],
                "default": "desc",
            },
            "user_id": {"type": "integer", "description": "Filter nach Benutzer-ID"},
            "date_from": {"type": "string", "description": "Filter ab Datum (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "Filter bis Datum (YYYY-MM-DD)"},
            "routed_to": {"type": "string", "description": "Filter nach Routing-Ziel"},
            "chain_type": {"type": "string", "description": "Filter nach Chain-Typ"},
            "search_text": {"type": "string", "description": "Volltextsuche in Fragen"},
        },
    },
    returns="Paginierte Liste der Audit-Eintraege",
    raw_response=True,
    http=("GET", "/api/admin/compliance/audit"),
)
async def compliance_audit(
    user: dict,
    page: int = 1,
    page_size: int = 50,
    sort_field: str = "created_at",
    sort_dir: str = "desc",
    user_id: int | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    routed_to: str | None = None,
    chain_type: str | None = None,
    search_text: str | None = None,
) -> AdminResult:
    from kiara.repositories.audit import QueryLogRepository

    invalid_date = _validate_audit_date_filter(
        date_from, "date_from"
    ) or _validate_audit_date_filter(date_to, "date_to")
    if invalid_date is not None:
        return invalid_date

    audit_repo = QueryLogRepository()
    try:
        result = audit_repo.list_with_joins(
            user_id=user_id,
            date_from=date_from,
            date_to=date_to,
            routed_to=routed_to,
            chain_type=chain_type,
            search_text=search_text,
            sort_field=sort_field,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size,
        )
    except Exception as exc:
        logger.error("Compliance-Audit Abfrage fehlgeschlagen: %s", exc)
        result = {
            "data": [],
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_items": 0,
                "total_pages": 0,
            },
        }

    return AdminResult.ok(
        f"{len(result.get('data', []))} Audit-Eintraege",
        data=result,
    )


@admin_function(
    name="compliance.audit_detail",
    category="compliance",
    description="Einzelnen Audit-Eintrag mit Details anzeigen",
    help="Zeigt einen einzelnen Audit-Eintrag mit allen Details an, "
    "inklusive Tool-Calls, Context-Sources und Cloud-Audit-Daten.",
    examples=[
        "Zeige Audit-Eintrag 42",
        "Details zu Audit-ID 100",
    ],
    parameters={
        "type": "object",
        "properties": {
            "audit_id": {"type": "integer", "description": "ID des Audit-Eintrags"},
        },
        "required": ["audit_id"],
    },
    returns="Audit-Eintrag mit Tool-Calls, Context-Sources und Cloud-Audit",
    raw_response=True,
    http=("GET", "/api/admin/compliance/audit/{audit_id}"),
)
async def compliance_audit_detail(user: dict, audit_id: int) -> AdminResult:
    import json

    from kiara.repositories.audit import CloudAuditLogRepository, QueryLogRepository

    audit_repo = QueryLogRepository()
    cloud_audit_repo = CloudAuditLogRepository()

    try:
        result = audit_repo.get_detail_with_joins(audit_id)
    except Exception as exc:
        logger.error("Audit-Detail Abfrage fehlgeschlagen: %s", exc)
        return AdminResult.fail("DB-Fehler bei Audit-Detail-Abfrage")

    if not result:
        return AdminResult.fail("Audit-Eintrag nicht gefunden")

    # Tool-Calls als Array parsen
    if result.get("tool_calls"):
        try:
            result["tool_calls"] = json.loads(result["tool_calls"])
        except (json.JSONDecodeError, TypeError):
            result["tool_calls"] = []
    else:
        result["tool_calls"] = []

    # context_sources parsen (RAG-Quellen)
    if result.get("context_sources"):
        try:
            result["context_sources"] = json.loads(result["context_sources"])
        except (json.JSONDecodeError, TypeError):
            result["context_sources"] = []
    else:
        result["context_sources"] = []

    # Cloud-Audit-Daten ueber correlation_id matchen
    corr_id = result.get("correlation_id")
    if corr_id:
        try:
            cloud_row = cloud_audit_repo.find_matching_entry(corr_id)
            if cloud_row:
                result["cloud_audit"] = cloud_row
        except Exception as exc:
            logger.debug("Cloud-Audit-Daten laden fehlgeschlagen: %s", exc)

    return AdminResult.ok("Audit-Detail", data=result)


@admin_function(
    name="compliance.permissions_matrix",
    category="compliance",
    description="Permission-Matrix (Benutzergruppen x Datenquellen)",
    help="Zeigt die Berechtigungsmatrix: Welche LDAP-Gruppe hat Zugriff "
    "auf welche Datenquellen und Tools.",
    examples=[
        "Zeige die Permission-Matrix",
        "Welche Gruppen haben Zugriff auf welche Datenquellen?",
        "Berechtigungsübersicht",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Gruppen mit Datenquellen, Mitgliedern und Tools",
    raw_response=True,
    http=("GET", "/api/admin/compliance/permissions-matrix"),
)
async def compliance_permissions_matrix(user: dict) -> AdminResult:
    from kiara.config import load_config
    from kiara.repositories.group import GroupRepository
    from kiara.repositories.permission import PermissionRepository

    config = load_config()
    cloud_enabled = config.get("cloud", {}).get("enabled", False)

    group_repo = GroupRepository()
    perm_repo = PermissionRepository()

    try:
        groups = group_repo.list_all(sort="cn ASC")

        result_groups = []
        for g in groups:
            group_id = g["id"]

            ds_rows = perm_repo.get_group_datasources(group_id)
            datasources = [
                {
                    "id": ds["id"],
                    "name": ds["name"],
                    "type": ds["type"],
                    "is_active": bool(ds["is_active"]),
                    "restriction_level": ds.get("restriction_level") or "anonymize",
                }
                for ds in ds_rows
            ]

            member_rows = group_repo.get_members(group_id)
            members = [
                {
                    "id": m["id"],
                    "username": m["username"],
                    "vorname": m["vorname"],
                    "nachname": m["nachname"],
                }
                for m in member_rows
            ]

            tools = ["local_rag"]
            if cloud_enabled:
                tools.append("cloud_ki")

            result_groups.append(
                {
                    "id": group_id,
                    "name": g["cn"],
                    "dn": g["dn"],
                    "member_count": g["member_count"],
                    "members": members,
                    "datasources": datasources,
                    "tools": tools,
                }
            )

    except Exception as exc:
        logger.error("Permissions-Matrix Abfrage fehlgeschlagen: %s", exc)
        result_groups = []

    return AdminResult.ok(
        f"{len(result_groups)} Gruppen",
        data={"groups": result_groups},
    )


@admin_function(
    name="compliance.anonymization_stats",
    category="compliance",
    description="Anonymisierungs-Statistiken anzeigen",
    help="Zeigt Statistiken zur Anonymisierung: Gesamtzahl anonymisierter "
    "Entitäten, Aufschlüsselung nach Kategorie, 30-Tage-Trend.",
    examples=[
        "Anonymisierungs-Statistiken",
        "Wie viele Entitäten wurden anonymisiert?",
        "Zeige Anonymisierungstrend",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Anonymisierungs-Statistiken mit Trend-Daten",
    raw_response=True,
    http=("GET", "/api/admin/compliance/anonymization-stats"),
)
async def compliance_anonymization_stats(user: dict) -> AdminResult:
    from datetime import datetime, timedelta

    from kiara.repositories.audit import AuditAnonymizationRepository
    from kiara.web.config_utils import safe_json_loads

    anon_audit_repo = AuditAnonymizationRepository()

    try:
        total_entities = anon_audit_repo.total_entities()

        type_rows = anon_audit_repo.all_entity_types()
        by_category = {}
        for row in type_rows:
            types = safe_json_loads(row["entity_types"])
            if types:
                for cat, count in types.items():
                    by_category[cat] = by_category.get(cat, 0) + count

        last_updated = anon_audit_repo.last_timestamp()

        since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        stats_30d = anon_audit_repo.stats_since(since_30d)

        daily = anon_audit_repo.daily_since(since_30d)
        by_route = anon_audit_repo.by_route()

    except Exception as exc:
        logger.error("Anonymisierungs-Statistik fehlgeschlagen: %s", exc)
        return AdminResult.ok(
            "Anonymisierungs-Statistiken (Fehler)",
            data={
                "total_entities": 0,
                "by_category": {},
                "last_updated": None,
                "total_30d": 0,
                "requests_30d": 0,
                "daily": [],
                "by_route": [],
            },
        )

    return AdminResult.ok(
        "Anonymisierungs-Statistiken",
        data={
            "total_entities": total_entities,
            "by_category": by_category,
            "last_updated": last_updated,
            "total_30d": stats_30d["total_30d"] if stats_30d else 0,
            "requests_30d": stats_30d["requests_30d"] if stats_30d else 0,
            "daily": daily,
            "by_route": by_route,
        },
    )


@admin_function(
    name="compliance.governance",
    category="compliance",
    description="Datenquellen-Governance-Übersicht",
    help="Zeigt eine Übersicht aller Datenquellen mit Zugriffsgruppen, "
    "Nutzungsstatistiken und Anonymisierungslevel.",
    examples=[
        "Zeige Datenquellen-Governance",
        "Governance-Übersicht",
        "Welche Datenquellen haben welche Zugriffsgruppen?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Datenquellen mit Zugriffsgruppen und Nutzungsdaten",
    raw_response=True,
    http=("GET", "/api/admin/compliance/datasource-governance"),
)
async def compliance_governance(user: dict) -> AdminResult:
    from datetime import datetime, timedelta

    from kiara.repositories.audit import QueryLogRepository
    from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
    from kiara.repositories.permission import PermissionRepository

    audit_repo = QueryLogRepository()
    ds_repo = DatasourceRepository()
    embedding_repo = DatasourceEmbeddingRepository()
    perm_repo = PermissionRepository()

    try:
        since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        datasources = ds_repo.list_all(sort="name ASC")

        # Embedding-Aggregate pro DS laden
        from kiara.shared.db import db_query

        with db_query() as db:
            emb_agg_rows = db.execute(
                "SELECT datasource_id, "
                "COALESCE(SUM(chunk_count), 0) AS chunk_count, "
                "COALESCE(SUM(file_count), 0) AS file_count, "
                "MAX(last_ingested_at) AS last_ingested_at "
                "FROM datasource_embeddings "
                "GROUP BY datasource_id"
            ).fetchall()
        emb_agg = {r["datasource_id"]: dict(r) for r in emb_agg_rows}

        # Gesamt-Queries (nicht pro DS — query_log hat keinen DS-Bezug)
        total_queries_30d = 0
        try:
            total_queries_30d = audit_repo.count_since(since_30d)
        except Exception as exc:
            logger.debug("Query-Log-Zaehlung fehlgeschlagen: %s", exc)

        result = []
        for ds in datasources:
            ds_id = ds["id"]

            group_rows = perm_repo.get_datasource_groups(ds_id)
            access_groups = [{"id": gr["id"], "name": gr["cn"]} for gr in group_rows]

            emb_rows, _ = embedding_repo.list_filtered(
                {"datasource_id": ds_id},
                limit=100,
                offset=0,
            )

            agg = emb_agg.get(ds_id, {})
            result.append(
                {
                    "datasource_id": ds_id,
                    "name": ds["name"],
                    "type": ds["type"],
                    "is_active": bool(ds["is_active"]),
                    "path": ds["path"],
                    "last_sync": agg.get("last_ingested_at"),
                    "document_count": agg.get("file_count", 0),
                    "chunk_count": agg.get("chunk_count", 0),
                    "restriction_level": ds.get("restriction_level") or "anonymize",
                    "access_groups": access_groups,
                    "embeddings": emb_rows,
                }
            )

    except Exception as exc:
        logger.error("Datasource-Governance Abfrage fehlgeschlagen: %s", exc)
        result = []

    return AdminResult.ok(
        f"{len(result)} Datenquellen",
        data={"datasources": result, "total_queries_30d": total_queries_30d},
    )


@admin_function(
    name="compliance.vvt",
    category="compliance",
    description="Verzeichnis der Verarbeitungstätigkeiten (VVT)",
    help="Generiert den VVT-Eintrag gemäß DSGVO Art. 30. "
    "Enthält Zweck, Rechtsgrundlage, Betroffene, Datenkategorien, "
    "Empfänger, Löschfristen, TOMs und Anonymisierungsstatus.",
    examples=[
        "Zeige das VVT",
        "Verzeichnis der Verarbeitungstätigkeiten",
        "VVT-Eintrag anzeigen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="VVT-Eintrag mit 11 Feldern (inkl. Drittlandtransfer gemaess Art. 30 Abs. 1 lit. e DSGVO)",
    raw_response=True,
    http=("GET", "/api/admin/compliance/vvt"),
)
async def compliance_vvt(user: dict) -> AdminResult:
    from kiara.compliance import ComplianceReportGenerator
    from kiara.config import load_config

    config = load_config()
    generator = ComplianceReportGenerator(config, cloud_backends=_load_cloud_backends_for_report())
    vvt = generator.generate_vvt_entry()

    return AdminResult.ok("VVT-Eintrag", data=vvt)


@admin_function(
    name="compliance.dsfa",
    category="compliance",
    description="Datenschutz-Folgenabschätzung (DSFA)",
    help="Generiert die Datenschutz-Folgenabschätzung gemäß DSGVO Art. 35. "
    "Enthält Erforderlichkeit, Systembeschreibung, Notwendigkeit, "
    "Risiken und Ergebnis.",
    examples=[
        "Zeige die DSFA",
        "Datenschutz-Folgenabschätzung",
        "Ist eine DSFA erforderlich?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="DSFA mit 7 Feldern",
    raw_response=True,
    http=("GET", "/api/admin/compliance/dsfa"),
)
async def compliance_dsfa(user: dict) -> AdminResult:
    from kiara.compliance import ComplianceReportGenerator
    from kiara.config import load_config

    config = load_config()
    generator = ComplianceReportGenerator(config, cloud_backends=_load_cloud_backends_for_report())
    dsfa = generator.generate_dsfa()

    return AdminResult.ok("DSFA", data=dsfa)


@admin_function(
    name="compliance.dashboard",
    category="compliance",
    description="Compliance-KPIs (30-Tage-Dashboard)",
    help="Aggregierte Compliance-Kennzahlen: Anfragen, Cloud-Nutzung, "
    "Anonymisierungsstatistiken, Budget-Status und Warnungen.",
    examples=[
        "Compliance-Dashboard",
        "Zeige Compliance-KPIs",
        "Wie ist der Compliance-Status?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Compliance-KPIs mit Warnungen",
    raw_response=True,
    http=("GET", "/api/admin/compliance/dashboard"),
)
async def compliance_dashboard(user: dict) -> AdminResult:
    from datetime import datetime, timedelta

    from kiara.config import load_config
    from kiara.repositories.audit import AuditAnonymizationRepository, QueryLogRepository
    from kiara.repositories.datasource import DatasourceRepository
    from kiara.repositories.group import GroupRepository
    from kiara.repositories.permission import PermissionRepository

    config = load_config()
    since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    audit_repo = QueryLogRepository()
    anon_audit_repo = AuditAnonymizationRepository()
    ds_repo = DatasourceRepository()
    group_repo = GroupRepository()
    perm_repo = PermissionRepository()

    try:
        total_queries_30d = audit_repo.count_since(since_30d)
        cloud_queries_30d = audit_repo.count_since(since_30d, routed_to="cloud")
        cloud_percentage = round(
            (cloud_queries_30d / total_queries_30d * 100) if total_queries_30d > 0 else 0.0, 1
        )
        anon_stats = anon_audit_repo.stats_since(since_30d)
        entities_anonymized_30d = anon_stats["total_30d"] if anon_stats else 0
        unique_users_30d = audit_repo.count_distinct_users_since(since_30d)
        datasource_count = ds_repo.count()
        group_count = group_repo.count()
        last_anonymization_update = anon_audit_repo.last_timestamp()
    except Exception as exc:
        logger.error("Compliance-Dashboard Abfrage fehlgeschlagen: %s", exc)
        return AdminResult.ok(
            "Compliance-Dashboard (Fehler)",
            data={
                "total_queries_30d": 0,
                "cloud_queries_30d": 0,
                "cloud_percentage": 0.0,
                "entities_anonymized_30d": 0,
                "unique_users_30d": 0,
                "datasource_count": 0,
                "group_count": 0,
                "cloud_budget_used": 0.0,
                "cloud_budget_limit": 0.0,
                "last_anonymization_update": None,
                "warnings": ["Datenbankfehler bei der Abfrage"],
            },
        )

    # Cloud-Budget
    warnings: list[str] = []
    cloud_budget_used = 0.0
    cloud_budget_limit = 0.0
    try:
        from kiara.cloud.audit import AuditLogger

        audit = AuditLogger(config)
        stats = audit.get_stats("month")
        cloud_budget_used = stats.get("estimated_cost_eur", 0.0)
        cloud_budget_limit = stats.get("budget_limit_eur", 0.0)
        budget_pct = stats.get("budget_used_pct", 0.0)
        if budget_pct >= 90:
            warnings.append(f"Cloud-Budget zu {budget_pct}% ausgeschoepft!")
        elif budget_pct >= 80:
            warnings.append(f"Cloud-Budget zu {budget_pct}% ausgeschoepft (Warnschwelle erreicht)")
    except Exception as exc:
        logger.warning("Cloud-Budget-Abfrage fehlgeschlagen: %s", exc)

    # Compliance-Warnungen
    cloud_enabled = config.get("cloud", {}).get("enabled", False)
    anon_enabled = config.get("anonymization", {}).get("enabled", True)
    if cloud_enabled and not anon_enabled:
        warnings.append("KRITISCH: Cloud-KI ist aktiviert, aber Anonymisierung ist deaktiviert!")
    try:
        unassigned_count = perm_repo.count_unassigned_datasources()
        if unassigned_count > 0:
            warnings.append(f"{unassigned_count} aktive Datenquelle(n) ohne Gruppenzuordnung")
    except Exception as exc:
        logger.debug("Compliance-Warnings-Pruefung fehlgeschlagen: %s", exc)

    return AdminResult.ok(
        "Compliance-Dashboard",
        data={
            "total_queries_30d": total_queries_30d,
            "cloud_queries_30d": cloud_queries_30d,
            "cloud_percentage": cloud_percentage,
            "entities_anonymized_30d": entities_anonymized_30d,
            "unique_users_30d": unique_users_30d,
            "datasource_count": datasource_count,
            "group_count": group_count,
            "cloud_budget_used": cloud_budget_used,
            "cloud_budget_limit": cloud_budget_limit,
            "last_anonymization_update": last_anonymization_update,
            "warnings": warnings,
        },
    )


@admin_function(
    name="compliance.prompt_defense",
    category="compliance",
    description="Prompt-Defense-Events abfragen (paginiert, filterbar)",
    help="Zeigt erkannte Prompt-Injection-Versuche mit Filtern nach Schweregrad, "
    "Quelle und Zeitraum. Jeder Eintrag enthaelt die erkannten Patterns, "
    "entfernte Zeichen und die Quelle (User-Eingabe, Dokument, Tool, History).",
    examples=[
        "Zeige Prompt-Defense-Events",
        "Hochkritische Injection-Versuche der letzten Woche",
    ],
    parameters={
        "type": "object",
        "properties": {
            "page": {"type": "integer", "description": "Seite (Standard: 1)", "default": 1},
            "page_size": {
                "type": "integer",
                "description": "Eintraege pro Seite (Standard: 50)",
                "default": 50,
            },
            "severity": {
                "type": "string",
                "description": "Filter nach Schweregrad",
                "enum": ["niedrig", "mittel", "hoch"],
            },
            "source_type": {
                "type": "string",
                "description": "Filter nach Quelle",
                "enum": ["user_input", "retrieval_context", "tool_result", "chat_history"],
            },
            "date_from": {"type": "string", "description": "Filter ab Datum (YYYY-MM-DD)"},
            "date_to": {"type": "string", "description": "Filter bis Datum (YYYY-MM-DD)"},
        },
    },
    returns="Paginierte Liste der Prompt-Defense-Events",
    raw_response=True,
    http=("GET", "/api/admin/compliance/prompt-defense"),
)
async def compliance_prompt_defense(
    user: dict,
    page: int = 1,
    page_size: int = 50,
    severity: str = "",
    source_type: str = "",
    date_from: str = "",
    date_to: str = "",
) -> AdminResult:
    from kiara.repositories.audit import PromptDefenseRepository

    repo = PromptDefenseRepository()
    result = repo.list_filtered(
        page=page,
        page_size=page_size,
        severity=severity or None,
        source_type=source_type or None,
        date_from=date_from or None,
        date_to=date_to or None,
    )
    return AdminResult.ok(
        f"{len(result.get('data', []))} Prompt-Defense-Events",
        data=result,
    )


@admin_function(
    name="compliance.prompt_defense_stats",
    category="compliance",
    description="Prompt-Defense-Statistiken (Trend, haeufigste Patterns)",
    help="Aggregierte Statistiken ueber Prompt-Injection-Events der letzten 30 Tage: "
    "Gesamtzahl, Anteil hochkritischer Events, haeufigste Quelle und Pattern.",
    examples=[
        "Prompt-Defense-Statistiken",
        "Wie viele Injection-Versuche gab es?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Aggregierte Prompt-Defense-Statistiken (30 Tage)",
    raw_response=True,
    http=("GET", "/api/admin/compliance/prompt-defense-stats"),
)
async def compliance_prompt_defense_stats(user: dict) -> AdminResult:
    from kiara.repositories.audit import PromptDefenseRepository

    repo = PromptDefenseRepository()
    stats = repo.get_stats(days=30)
    return AdminResult.ok("Prompt-Defense-Statistiken", data=stats)
