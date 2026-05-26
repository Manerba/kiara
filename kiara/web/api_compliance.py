"""
KIara Admin - DSGVO-Compliance API-Router
Audit-Log, Permissions-Matrix, Anonymisierungs-Statistiken,
Datenquellen-Governance, VVT, DSFA, Dashboard.
"""

import logging

from fastapi import APIRouter

from kiara.repositories.audit import (
    AuditAnonymizationRepository,
    QueryLogRepository,
)
from kiara.repositories.datasource import DatasourceEmbeddingRepository, DatasourceRepository
from kiara.repositories.group import GroupRepository
from kiara.repositories.permission import PermissionRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/compliance", tags=["compliance"])

# --- Repository-Instanzen (Modul-Ebene) ---
audit_repo = QueryLogRepository()
anon_audit_repo = AuditAnonymizationRepository()
ds_repo = DatasourceRepository()
embedding_repo = DatasourceEmbeddingRepository()
group_repo = GroupRepository()
perm_repo = PermissionRepository()


# --- Migriert zu Admin-Registry: compliance.get_compliance_audit (Sprint 09) ---
# @router.get("/audit")
# async def get_compliance_audit(
#     user: dict = Depends(get_admin_user),
#     page: int = Query(1, ge=1),
#     page_size: int = Query(50, ge=1, le=500),
#     sort_field: str = Query("created_at"),
#     sort_dir: str = Query("desc"),
#     user_id: int = Query(None),
#     date_from: str = Query(None),
#     date_to: str = Query(None),
#     routed_to: str = Query(None),
#     chain_type: str = Query(None),
#     search_text: str = Query(None),
# ) -> dict:
#     """Alle Anfragen aus query_log, paginiert, sortierbar und filterbar."""
#     try:
#         result = audit_repo.list_with_joins(
#             user_id=user_id,
#             date_from=date_from,
#             date_to=date_to,
#             routed_to=routed_to,
#             chain_type=chain_type,
#             search_text=search_text,
#             sort_field=sort_field,
#             sort_dir=sort_dir,
#             page=page,
#             page_size=page_size,
#         )
#     except Exception as exc:
#         logger.error("Compliance-Audit Abfrage fehlgeschlagen: %s", exc)
#         result = {
#             "data": [],
#             "pagination": {
#                 "page": page,
#                 "page_size": page_size,
#                 "total_items": 0,
#                 "total_pages": 0,
#             },
#         }
#
#     return result


# --- Migriert zu Admin-Registry: compliance.get_permissions_matrix (Sprint 09) ---
# @router.get("/permissions-matrix")
# async def get_permissions_matrix(user: dict = Depends(get_admin_user)) -> dict:
#     """Matrix: LDAP-Gruppen -> Datenquellen + Tools."""
#     try:
#         config = load_config()
#         cloud_enabled = config.get("cloud", {}).get("enabled", False)
#
#         groups = group_repo.list_all(sort="cn ASC")
#
#         result_groups = []
#         for g in groups:
#             group_id = g["id"]
#
#             # Datenquellen fuer diese Gruppe
#             ds_rows = perm_repo.get_group_datasources(group_id)
#             datasources = [
#                 {
#                     "id": ds["id"],
#                     "name": ds["name"],
#                     "type": ds["type"],
#                     "is_active": bool(ds["is_active"]),
#                     "restriction_level": ds.get("restriction_level") or "anonymize",
#                 }
#                 for ds in ds_rows
#             ]
#
#             # Mitglieder der Gruppe
#             member_rows = group_repo.get_members(group_id)
#             members = [
#                 {
#                     "id": m["id"],
#                     "username": m["username"],
#                     "vorname": m["vorname"],
#                     "nachname": m["nachname"],
#                 }
#                 for m in member_rows
#             ]
#
#             # Tools-Zugriff: Abgeleitet aus Cloud-Konfiguration
#             tools = ["local_rag"]
#             if cloud_enabled:
#                 tools.append("cloud_ki")
#
#             result_groups.append({
#                 "id": group_id,
#                 "name": g["cn"],
#                 "dn": g["dn"],
#                 "member_count": g["member_count"],
#                 "members": members,
#                 "datasources": datasources,
#                 "tools": tools,
#             })
#
#     except Exception as exc:
#         logger.error("Permissions-Matrix Abfrage fehlgeschlagen: %s", exc)
#         result_groups = []
#
#     return {"groups": result_groups}

# --- Migriert zu Admin-Registry: compliance.get_anonymization_stats (Sprint 09) ---
# @router.get("/anonymization-stats")
# async def get_anonymization_stats(user: dict = Depends(get_admin_user)) -> dict:
#     """Anonymisierungs-Statistiken aus audit_anonymization Tabelle."""
#     try:
#         # Gesamtzahl anonymisierter Entitaeten
#         total_entities = anon_audit_repo.total_entities()
#
#         # Aufschluesselung nach Kategorie
#         # entity_types ist als JSON gespeichert, z.B. '{"Kunden": 3, "IBAN": 1}'
#         type_rows = anon_audit_repo.all_entity_types()
#
#         by_category = {}
#         for row in type_rows:
#             types = safe_json_loads(row["entity_types"])
#             if types:
#                 for cat, count in types.items():
#                     by_category[cat] = by_category.get(cat, 0) + count
#
#         # Letztes Update
#         last_updated = anon_audit_repo.last_timestamp()
#
#         # Statistik letzte 30 Tage
#         since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
#         stats_30d = anon_audit_repo.stats_since(since_30d)
#
#         # Taeglich (letzte 30 Tage)
#         daily = anon_audit_repo.daily_since(since_30d)
#
#         # Route-Verteilung
#         by_route = anon_audit_repo.by_route()
#
#     except Exception as exc:
#         logger.error("Anonymisierungs-Statistik fehlgeschlagen: %s", exc)
#         return {
#             "total_entities": 0,
#             "by_category": {},
#             "last_updated": None,
#             "total_30d": 0,
#             "requests_30d": 0,
#             "daily": [],
#             "by_route": [],
#         }
#
#     return {
#         "total_entities": total_entities,
#         "by_category": by_category,
#         "last_updated": last_updated,
#         "total_30d": stats_30d["total_30d"] if stats_30d else 0,
#         "requests_30d": stats_30d["requests_30d"] if stats_30d else 0,
#         "daily": daily,
#         "by_route": by_route,
#     }

# --- Migriert zu Admin-Registry: compliance.get_datasource_governance (Sprint 09) ---
# @router.get("/datasource-governance")
# async def get_datasource_governance(user: dict = Depends(get_admin_user)) -> dict:
#     """Datenquellen-Governance: Uebersicht aller Datenquellen mit Zugriffsgruppen und Nutzung."""
#     try:
#         since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
#
#         datasources = ds_repo.list_all(sort="name ASC")
#
#         result = []
#         for ds in datasources:
#             ds_id = ds["id"]
#
#             # Zugriffsgruppen
#             group_rows = perm_repo.get_datasource_groups(ds_id)
#             access_groups = [{"id": gr["id"], "name": gr["cn"]} for gr in group_rows]
#
#             # Anfragen letzte 30 Tage (aus query_log, matchen ueber sources)
#             # Da query_log kein direktes datasource_id Feld hat,
#             # zaehlen wir alle Queries der letzten 30 Tage
#             queries_30d = 0
#             try:
#                 queries_30d = audit_repo.count_since(since_30d)
#             except Exception as exc:
#                 logger.debug("Query-Log-Zaehlung fehlgeschlagen: %s", exc)
#
#             # Embedding-Modelle fuer diese Datenquelle
#             emb_rows, _ = embedding_repo.list_filtered(
#                 {"datasource_id": ds_id}, limit=100, offset=0,
#             )
#
#             result.append({
#                 "datasource_id": ds_id,
#                 "name": ds["name"],
#                 "type": ds["type"],
#                 "is_active": bool(ds["is_active"]),
#                 "path": ds["path"],
#                 "last_sync": ds["last_ingested_at"],
#                 "document_count": ds["file_count"] or 0,
#                 "chunk_count": ds["chunk_count"] or 0,
#                 "restriction_level": ds.get("restriction_level") or "anonymize",
#                 "access_groups": access_groups,
#                 "queries_last_30d": queries_30d,
#                 "embeddings": emb_rows,
#             })
#
#     except Exception as exc:
#         logger.error("Datasource-Governance Abfrage fehlgeschlagen: %s", exc)
#         result = []
#
#     return {"datasources": result}

# --- Migriert zu Admin-Registry: compliance.get_vvt (Sprint 09) ---
# @router.get("/vvt")
# async def get_vvt(user: dict = Depends(get_admin_user)) -> dict:
#     """VVT-Eintrag (Verzeichnis der Verarbeitungstaetigkeiten) als JSON."""
#     config = load_config()
#     generator = ComplianceReportGenerator(config)
#     return generator.generate_vvt_entry()

# --- Migriert zu Admin-Registry: compliance.get_dsfa (Sprint 09) ---
# @router.get("/dsfa")
# async def get_dsfa(user: dict = Depends(get_admin_user)) -> dict:
#     """Datenschutz-Folgenabschaetzung als JSON."""
#     config = load_config()
#     generator = ComplianceReportGenerator(config)
#     return generator.generate_dsfa()


def _query_dashboard_stats(since_30d: str) -> dict:
    """DB-Abfragen fuer das Compliance-Dashboard (30-Tage-Fenster).

    Returns:
        Dict mit total_queries_30d, cloud_queries_30d, cloud_percentage,
        entities_anonymized_30d, unique_users_30d, datasource_count,
        group_count, last_anonymization_update.
    """
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

    return {
        "total_queries_30d": total_queries_30d,
        "cloud_queries_30d": cloud_queries_30d,
        "cloud_percentage": cloud_percentage,
        "entities_anonymized_30d": entities_anonymized_30d,
        "unique_users_30d": unique_users_30d,
        "datasource_count": datasource_count,
        "group_count": group_count,
        "last_anonymization_update": last_anonymization_update,
    }


def _get_cloud_budget(config: dict) -> tuple[float, float, list[str]]:
    """Cloud-Budget-Status aus AuditLogger abfragen.

    Returns:
        Tuple von (budget_used, budget_limit, warnings).
    """
    warnings: list[str] = []
    try:
        from kiara.cloud.audit import AuditLogger

        audit = AuditLogger(config)
        stats = audit.get_stats("month")
        budget_used = stats.get("estimated_cost_eur", 0.0)
        budget_limit = stats.get("budget_limit_eur", 0.0)

        budget_pct = stats.get("budget_used_pct", 0.0)
        if budget_pct >= 90:
            warnings.append(f"Cloud-Budget zu {budget_pct}% ausgeschoepft!")
        elif budget_pct >= 80:
            warnings.append(f"Cloud-Budget zu {budget_pct}% ausgeschoepft (Warnschwelle erreicht)")
        return budget_used, budget_limit, warnings
    except Exception as exc:
        logger.warning("Cloud-Budget-Abfrage fehlgeschlagen: %s", exc)
        return 0.0, 0.0, warnings


def _check_compliance_warnings(config: dict) -> list[str]:
    """Compliance-Warnungen pruefen (Cloud/Anon-Status, unzugeordnete Datenquellen).

    Returns:
        Liste von Warnungs-Strings.
    """
    warnings: list[str] = []

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

    return warnings


# --- Migriert zu Admin-Registry: compliance.get_compliance_dashboard (Sprint 09) ---
# @router.get("/dashboard")
# async def get_compliance_dashboard(user: dict = Depends(get_admin_user)) -> dict:
#     """Aggregierte Compliance-Kennzahlen fuer das Dashboard."""
#     config = load_config()
#     since_30d = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
#
#     try:
#         stats = _query_dashboard_stats(since_30d)
#     except Exception as exc:
#         logger.error("Compliance-Dashboard Abfrage fehlgeschlagen: %s", exc)
#         return {
#             "total_queries_30d": 0, "cloud_queries_30d": 0, "cloud_percentage": 0.0,
#             "entities_anonymized_30d": 0, "unique_users_30d": 0,
#             "datasource_count": 0, "group_count": 0,
#             "cloud_budget_used": 0.0, "cloud_budget_limit": 0.0,
#             "last_anonymization_update": None,
#             "warnings": ["Datenbankfehler bei der Abfrage"],
#         }
#
#     cloud_budget_used, cloud_budget_limit, budget_warnings = _get_cloud_budget(config)
#     compliance_warnings = _check_compliance_warnings(config)
#
#     return {
#         **stats,
#         "cloud_budget_used": cloud_budget_used,
#         "cloud_budget_limit": cloud_budget_limit,
#         "warnings": budget_warnings + compliance_warnings,
#     }
