"""
KIara Admin - Tools API-Router
Verwaltung, Test und Nutzungsstatistiken fuer registrierte Tools.
"""

import logging
import secrets
import threading
import time
from dataclasses import dataclass

from fastapi import APIRouter
from pydantic import BaseModel

from kiara.config import load_config
from kiara.shared.constants import (
    ADMIN_TOOL_TEST_MAX_SESSIONS,
    ADMIN_TOOL_TEST_SESSION_TTL_SECS,
    PASSWORD_MASK,
)
from kiara.shared.crypto import encrypt_secret_fields
from kiara.tools.action_tools import GenerateChecklistTool, GenerateStatusMailTool
from kiara.tools.builtin_tools import CalculateTool, ListDatasourcesTool, SearchDocumentsTool
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool
from kiara.tools.erp_backend import ERPBackend
from kiara.tools.erp_tools import (
    ERPCustomerInfoTool,
    ERPOrderStatusTool,
    ERPOrdersTool,
    ERPProductionStatusTool,
    ERPStockLevelTool,
)
from kiara.tools.registry import ToolRegistry, load_config_tool_permissions
from kiara.tools.workflow_tools import (
    ExportDocumentTool,
    GenerateFromTemplateTool,
    SummarizeDocumentTool,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/tools", tags=["tools"])

# Singleton-Cache fuer die ToolRegistry
_registry: ToolRegistry | None = None
_registry_lock = threading.Lock()

# Singleton-Cache fuer die ConnectionRegistry
_connection_registry = None
_connection_registry_lock = threading.Lock()


def _create_document_tools(config: dict, *, retriever=None, reranker=None) -> dict:
    if retriever is None:
        from kiara.retrieval.retriever import HybridRetriever

        retriever = HybridRetriever(config)
        try:
            from kiara.retrieval.reranker import create_reranker

            reranker = create_reranker(config)
        except Exception as exc:
            logger.warning("Reranker nicht verfuegbar: %s", exc)
    references = DocumentReferences(str(retriever._store.get_collection_name()))
    return {
        "search_documents": SearchDocumentsTool(
            retriever, reranker, config, document_references=references
        ),
        "read_document": ReadDocumentTool(retriever, config, references),
    }


@dataclass
class _DocumentTestSession:
    user_id: int
    origin: ToolRegistry
    tools: dict
    last_used: float


_document_test_sessions: dict[str, _DocumentTestSession] = {}
_document_test_sessions_lock = threading.Lock()


def _get_document_test_tool(registry, name, user_id, session_id=None):
    """Such-/Lesefolge pro Admin und expliziter, zeitlich begrenzter Testsitzung."""
    config = load_config()
    settings = config.get("tools", {})
    ttl = settings.get("test_session_ttl_secs", ADMIN_TOOL_TEST_SESSION_TTL_SECS)
    maximum = settings.get("test_max_sessions", ADMIN_TOOL_TEST_MAX_SESSIONS)
    now = time.monotonic()
    with _document_test_sessions_lock:
        for key, session in list(_document_test_sessions.items()):
            if now - session.last_used >= ttl or session.origin is not registry:
                del _document_test_sessions[key]
        if session_id:
            session = _document_test_sessions.get(session_id)
            if session is None or session.user_id != user_id:
                raise ValueError(
                    "Testsitzung ungueltig oder abgelaufen. Suche ohne session_id starten."
                )
        else:
            if name != "search_documents":
                raise ValueError("Zuerst search_documents testen und dessen session_id uebergeben.")
            # Infrastruktur teilen, nur Referenzen und Such-Deduplizierung
            # gehoeren der Sitzung. Ablaufende Sitzungen halten keine eigenen Clients.
            shared = registry.get_tool("search_documents")
            tools = _create_document_tools(
                config, retriever=shared._retriever, reranker=shared._reranker
            )
            session = _DocumentTestSession(user_id, registry, tools, now)
            while len(_document_test_sessions) >= maximum:
                oldest = min(
                    _document_test_sessions, key=lambda key: _document_test_sessions[key].last_used
                )
                del _document_test_sessions[oldest]
            session_id = secrets.token_urlsafe(24)
            _document_test_sessions[session_id] = session
        session.last_used = now
        return session.tools[name], session_id


def _get_connection_registry():
    """Lazy-initialisiert die ConnectionRegistry aus der Config."""
    global _connection_registry
    if _connection_registry is not None:
        return _connection_registry

    with _connection_registry_lock:
        if _connection_registry is not None:
            return _connection_registry
        from kiara.tools.erp_connectors import ConnectionRegistry

        config = load_config()
        registry = ConnectionRegistry()
        registry.load_from_config(config)
        _connection_registry = registry
    return _connection_registry


def _get_registry() -> ToolRegistry:
    """Lazy-initialisiert und cached die ToolRegistry als Singleton.

    Laedt die Config, erstellt das ERPBackend und registriert alle Tools
    (Builtin + ERP + Action + Workflow).

    Returns:
        Die gecachte ToolRegistry-Instanz.
    """
    global _registry
    if _registry is not None:
        return _registry

    with _registry_lock:
        if _registry is not None:
            return _registry

        config = load_config()
        conn_registry = _get_connection_registry()
        backend = ERPBackend(config, connection_registry=conn_registry)

        registry = ToolRegistry(permission_provider=load_config_tool_permissions)

        # Builtin-Tools (immer verfuegbar, nicht konfigurierbar)
        registry.register(CalculateTool())
        registry.register(ListDatasourcesTool())
        try:
            for tool in _create_document_tools(config).values():
                registry.register(tool)
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.warning("SearchDocumentsTool konnte nicht registriert werden: %s", exc)

        # ERP-Tools
        registry.register(ERPOrderStatusTool(backend))
        registry.register(ERPOrdersTool(backend))
        registry.register(ERPStockLevelTool(backend))
        registry.register(ERPCustomerInfoTool(backend))
        registry.register(ERPProductionStatusTool(backend))
        registry.register(GenerateStatusMailTool(backend))
        registry.register(GenerateChecklistTool())

        # Phase 3: Workflow-Tools registrieren
        try:
            from kiara.retrieval.retriever import HybridRetriever as _Retriever
            from kiara.workflows.exporter import DocumentExporter
            from kiara.workflows.summarizer import DocumentSummarizer
            from kiara.workflows.template_engine import TemplateEngine

            template_engine = TemplateEngine(config)
            summarizer = DocumentSummarizer(config)
            exporter = DocumentExporter(config)
            wf_retriever = _Retriever(config)

            registry.register(SummarizeDocumentTool(wf_retriever, summarizer))
            registry.register(GenerateFromTemplateTool(template_engine))
            registry.register(ExportDocumentTool(exporter))
        except Exception as exc:
            from kiara.retrieval.index_compatibility import IndexCompatibilityError

            if isinstance(exc, IndexCompatibilityError):
                raise
            logger.warning("Workflow-Tools konnten nicht registriert werden: %s", exc)

        _registry = registry
        logger.info(
            "ToolRegistry initialisiert mit %d Tools.", len(registry.list_tools(user_groups=["*"]))
        )
    return _registry


class PermissionsUpdateRequest(BaseModel):
    """Request zum Aktualisieren der Tool-Berechtigungen einer Gruppe."""

    group: str
    tools: list[str]


class ToolTestRequest(BaseModel):
    """Request zum manuellen Testen eines registrierten Tools."""

    params: dict


# --- Migriert zu Admin-Registry: tools.list (Sprint 09) ---
# @router.get("/list")
# async def list_tools(user=Depends(get_admin_user)):
#     """Alle registrierten Tools mit Status auflisten.
#
#     Returns:
#         {"tools": [{"name", "description", "requires_confirmation",
#                      "required_permissions", "parameters"}]}
#     """
#     registry = _get_registry()
#     definitions = registry.list_tools(user_groups=["*"])
#
#     tools = []
#     for defn in definitions:
#         tools.append({
#             "name": defn.name,
#             "description": defn.description,
#             "requires_confirmation": defn.requires_confirmation,
#             "required_permissions": defn.required_permissions,
#             "parameters": defn.parameters,
#             "category": defn.category,
#         })
#
#     return {"tools": tools}

# --- Migriert zu Admin-Registry: tools.permissions (Sprint 09) ---
# @router.get("/permissions")
# async def get_permissions(user=Depends(get_admin_user)):
#     """Berechtigungs-Matrix aus der Config lesen.
#
#     Returns:
#         {"permissions": {"ERP-Leser": ["erp_order_status", ...], ...}}
#     """
#     config = load_config()
#     tools_cfg = config.get("tools", {})
#     permissions = tools_cfg.get("permissions", {})
#
#     return {"permissions": permissions}

# --- Migriert zu Admin-Registry: tools.permissions.update (Sprint 09) ---
# @router.post("/permissions")
# async def update_permissions(data: PermissionsUpdateRequest, user=Depends(get_admin_user)):
#     """Berechtigungen fuer eine Gruppe aktualisieren.
#
#     Body: {"group": str, "tools": list[str]}
#     Speichert die Aenderungen in der Config-Datei.
#     """
#     config = load_config()
#
#     if "tools" not in config:
#         config["tools"] = {}
#     if "permissions" not in config["tools"]:
#         config["tools"]["permissions"] = {}
#
#     config["tools"]["permissions"][data.group] = data.tools
#
#     save_config(config)
#     logger.info(
#         "Tool-Berechtigungen fuer Gruppe '%s' aktualisiert von %s: %s",
#         data.group, user["username"], data.tools,
#     )
#
#     return {"message": f"Berechtigungen fuer Gruppe '{data.group}' gespeichert."}

# --- Migriert zu Admin-Registry: tools.test (Sprint 09) ---
# @router.post("/test/{name}")
# async def test_tool(name: str, data: ToolTestRequest, user=Depends(get_admin_user)):
#     """Tool manuell testen (Admin-only).
#
#     Fuehrt das Tool mit den uebergebenen Parametern aus und gibt
#     das Ergebnis zurueck.
#     """
#     registry = _get_registry()
#     tool = registry.get_tool(name)
#
#     if tool is None:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Tool '{name}' nicht gefunden.",
#         )
#
#     # Parameter validieren
#     valid, msg = tool.validate_params(data.params)
#     if not valid:
#         raise HTTPException(
#             status_code=422,
#             detail=f"Parameter-Validierung fehlgeschlagen: {msg}",
#         )
#
#     try:
#         result = await tool.execute(data.params, user_id=user["id"])
#     except Exception as exc:
#         logger.error("Fehler bei Tool-Test '%s': %s", name, exc)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Tool-Ausfuehrung fehlgeschlagen: {exc}",
#         )
#
#     return {
#         "success": result.success,
#         "data": result.data,
#         "error": result.error,
#         "display_text": result.display_text,
#     }


class ERPConfigRequest(BaseModel):
    """Request zum Speichern der ERP-Backend-Konfiguration."""

    backend: str = "mock"
    sql_dsn: str = ""
    sql_driver: str = "auto"
    mapping: dict = {}


# --- Migriert zu Admin-Registry: tools.erp.config.read (Sprint 09) ---
# @router.get("/erp-config")
# async def get_erp_config(user=Depends(get_admin_user)):
#     """ERP-Backend-Konfiguration auslesen."""
#     config = load_config()
#     erp_cfg = config.get("tools", {}).get("erp", {})
#     return {
#         "backend": erp_cfg.get("backend", "mock"),
#         "sql_dsn": PASSWORD_MASK if erp_cfg.get("sql_dsn") else "",
#         "sql_driver": erp_cfg.get("sql_driver", "auto"),
#         "mapping": erp_cfg.get("mapping", {}),
#     }

# --- Migriert zu Admin-Registry: tools.erp.config.update (Sprint 09) ---
# @router.post("/erp-config")
# async def save_erp_config(data: ERPConfigRequest, user=Depends(get_admin_user)):
#     """ERP-Backend-Konfiguration speichern."""
#     config = load_config()
#     erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
#     erp_cfg["backend"] = data.backend
#     if data.sql_dsn and data.sql_dsn != PASSWORD_MASK:
#         erp_cfg["sql_dsn"] = data.sql_dsn
#     erp_cfg["sql_driver"] = data.sql_driver
#     erp_cfg["mapping"] = data.mapping
#     save_config(config)
#     # Registry-Cache invalidieren
#     global _registry, _connection_registry
#     _registry = None
#     _connection_registry = None
#     logger.info("ERP-Config aktualisiert von %s: backend=%s", user["username"], data.backend)
#     return {"message": "ERP-Konfiguration gespeichert"}

# --- Migriert zu Admin-Registry: tools.erp.test_connection (Sprint 09) ---
# @router.post("/erp-test-connection")
# def test_erp_connection(user=Depends(get_admin_user)):
#     """Verbindungstest fuer das konfigurierte ERP-SQL-Backend."""
#     config = load_config()
#     erp_cfg = config.get("tools", {}).get("erp", {})
#     if erp_cfg.get("backend") != "sql":
#         raise HTTPException(status_code=400, detail="Backend ist nicht SQL")
#     dsn = erp_cfg.get("sql_dsn", "")
#     if not dsn:
#         raise HTTPException(status_code=400, detail="Kein SQL-DSN konfiguriert")
#     from kiara.connectors.sql_connector import SQLConnector
#     connector = SQLConnector(dsn, erp_cfg.get("sql_driver", "auto"))
#     result = connector.test_connection()
#     if not result.get("success"):
#         raise HTTPException(status_code=502, detail=result.get("message", "Verbindungstest fehlgeschlagen"))
#     return result

# --- Migriert zu Admin-Registry: tools.erp.table_schema (Sprint 09) ---
# @router.get("/erp-table-schema/{table_name}")
# def get_erp_table_schema(table_name: str, user=Depends(get_admin_user)):
#     """Spalten einer Tabelle im ERP-SQL-Backend auslesen."""
#     config = load_config()
#     erp_cfg = config.get("tools", {}).get("erp", {})
#     dsn = erp_cfg.get("sql_dsn", "")
#     if not dsn:
#         return {"columns": [], "error": "Kein SQL-DSN konfiguriert"}
#     from kiara.connectors.sql_connector import SQLConnector
#     connector = SQLConnector(dsn, erp_cfg.get("sql_driver", "auto"))
#     try:
#         schema = connector.get_table_schema(table_name)
#         return {"columns": schema}
#     except Exception as exc:
#         return {"columns": [], "error": str(exc)}

# ------------------------------------------------------------------
# Connection-Registry Endpoints
# ------------------------------------------------------------------


def _encrypt_connection_config(
    new_config: dict,
    config_fields: list[dict],
    existing_config: dict | None = None,
) -> dict:
    """Verschluesselt Secret-Felder in der Connection-Config.

    Delegiert an ``encrypt_secret_fields()`` aus ``shared.crypto``.
    """
    secret_names = [f["name"] for f in config_fields if f.get("type") == "password"]
    return encrypt_secret_fields(new_config, secret_names, existing_config)


def _mask_connection_config(config: dict, config_fields: list[dict]) -> dict:
    """Ersetzt Secret-Felder durch PASSWORD_MASK fuer API-Responses."""
    masked = dict(config)
    for field in config_fields:
        if field.get("type") == "password" and masked.get(field["name"]):
            masked[field["name"]] = PASSWORD_MASK
    return masked


def _get_connector_fields(connector_type: str) -> list[dict]:
    """Config-Felder fuer einen Konnektor-Typ ermitteln."""
    from kiara.tools.erp_connectors import ConnectionRegistry

    for cls in ConnectionRegistry._connector_types.values():
        defn = cls.connector_definition()
        if defn.type_id == connector_type:
            return defn.config_fields
    return []


class ConnectionRequest(BaseModel):
    """Request zum Anlegen oder Bearbeiten einer ERP-Verbindung."""

    name: str
    type: str
    config: dict = {}
    enabled: bool = True


class AssignmentsRequest(BaseModel):
    """Request zum Speichern der Tool-zu-Verbindung-Zuordnungen."""

    assignments: dict


# --- Migriert zu Admin-Registry: tools.connector_types (Sprint 09) ---
# @router.get("/connector-types")
# async def list_connector_types(user=Depends(get_admin_user)):
#     """Verfuegbare Konnektor-Typen + Config-Schema."""
#     conn_reg = _get_connection_registry()
#     return {"types": conn_reg.list_connector_types()}


# --- Migriert zu Admin-Registry: tools.connections.list (Sprint 09) ---
# @router.get("/connections")
# async def list_connections(user=Depends(get_admin_user)):
#     """Alle Verbindungen listen (Passwoerter maskiert)."""
#     conn_reg = _get_connection_registry()
#     connections = []
#     for conn in conn_reg.list_connections():
#         config_fields = _get_connector_fields(conn.connector_type)
#         connections.append({
#             "id": conn.id,
#             "name": conn.name,
#             "type": conn.connector_type,
#             "config": _mask_connection_config(conn.config, config_fields),
#             "enabled": conn.enabled,
#         })
#     return {"connections": connections}


# --- Migriert zu Admin-Registry: tools.connections.create (Sprint 09) ---
# @router.post("/connections")
# async def create_connection(data: ConnectionRequest, user=Depends(get_admin_user)):
#     """Verbindung anlegen."""
#     from kiara.tools.erp_connectors.base import ConnectionInfo
#
#     config_fields = _get_connector_fields(data.type)
#     encrypted_config = _encrypt_connection_config(data.config, config_fields)
#
#     conn_id = f"conn-{uuid.uuid4().hex[:8]}"
#     info = ConnectionInfo(
#         id=conn_id,
#         name=data.name,
#         connector_type=data.type,
#         config=encrypted_config,
#         enabled=data.enabled,
#     )
#
#     conn_reg = _get_connection_registry()
#     conn_reg.add_connection(info)
#
#     # In Config persistieren
#     config = load_config()
#     erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
#     registry_data = conn_reg.save_to_config()
#     erp_cfg["connections"] = registry_data["connections"]
#     save_config(config)
#
#     # Registry-Cache invalidieren
#     global _registry, _connection_registry
#     _registry = None
#     _connection_registry = None
#
#     logger.info("ERP-Verbindung '%s' (%s) angelegt von %s", data.name, data.type, user["username"])
#     return {"id": conn_id, "message": "Verbindung angelegt"}


# --- Migriert zu Admin-Registry: tools.connections.update (Sprint 09) ---
# @router.put("/connections/{connection_id}")
# async def update_connection(connection_id: str, data: ConnectionRequest, user=Depends(get_admin_user)):
#     """Verbindung bearbeiten."""
#     from kiara.tools.erp_connectors.base import ConnectionInfo
#
#     conn_reg = _get_connection_registry()
#     existing = None
#     for conn in conn_reg.list_connections():
#         if conn.id == connection_id:
#             existing = conn
#             break
#     if not existing:
#         raise HTTPException(status_code=404, detail=f"Verbindung '{connection_id}' nicht gefunden.")
#
#     config_fields = _get_connector_fields(data.type)
#     encrypted_config = _encrypt_connection_config(
#         data.config, config_fields, existing_config=existing.config
#     )
#
#     info = ConnectionInfo(
#         id=connection_id,
#         name=data.name,
#         connector_type=data.type,
#         config=encrypted_config,
#         enabled=data.enabled,
#     )
#     conn_reg.add_connection(info)
#
#     # In Config persistieren
#     config = load_config()
#     erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
#     registry_data = conn_reg.save_to_config()
#     erp_cfg["connections"] = registry_data["connections"]
#     save_config(config)
#
#     global _registry, _connection_registry
#     _registry = None
#     _connection_registry = None
#
#     logger.info("ERP-Verbindung '%s' aktualisiert von %s", connection_id, user["username"])
#     return {"message": "Verbindung aktualisiert"}


# --- Migriert zu Admin-Registry: tools.connections.delete (Sprint 09) ---
# @router.delete("/connections/{connection_id}")
# async def delete_connection(connection_id: str, user=Depends(get_admin_user)):
#     """Verbindung loeschen."""
#     conn_reg = _get_connection_registry()
#     conn_reg.remove_connection(connection_id)
#
#     # In Config persistieren
#     config = load_config()
#     erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
#     registry_data = conn_reg.save_to_config()
#     erp_cfg["connections"] = registry_data["connections"]
#     erp_cfg["tool_assignments"] = registry_data["tool_assignments"]
#     save_config(config)
#
#     global _registry, _connection_registry
#     _registry = None
#     _connection_registry = None
#
#     logger.info("ERP-Verbindung '%s' geloescht von %s", connection_id, user["username"])
#     return {"message": "Verbindung geloescht"}


# --- Migriert zu Admin-Registry: tools.connections.test (Sprint 09) ---
# @router.post("/connections/{connection_id}/test")
# async def test_connection(connection_id: str, user=Depends(get_admin_user)):
#     """Verbindung testen."""
#     conn_reg = _get_connection_registry()
#     try:
#         connector = conn_reg.get_connector(connection_id)
#     except (KeyError, ValueError) as exc:
#         raise HTTPException(status_code=404, detail=str(exc))
#     try:
#         result = await connector.test_connection()
#         return result
#     except Exception as exc:
#         raise HTTPException(status_code=502, detail=f"Testfehler: {exc}")


# --- Migriert zu Admin-Registry: tools.assignments (Sprint 09) ---
# @router.get("/assignments")
# async def get_assignments(user=Depends(get_admin_user)):
#     """Aktuelle Tool→Verbindung Zuordnungen."""
#     conn_reg = _get_connection_registry()
#     return {"assignments": conn_reg.get_tool_assignments()}


# --- Migriert zu Admin-Registry: tools.assignments.update (Sprint 09) ---
# @router.post("/assignments")
# async def save_assignments(data: AssignmentsRequest, user=Depends(get_admin_user)):
#     """Zuordnungen speichern."""
#     conn_reg = _get_connection_registry()
#     # Bestehende Zuordnungen loeschen und neu setzen
#     conn_reg._tool_assignments.clear()
#     for tool_name, conn_id in data.assignments.items():
#         if conn_id:
#             conn_reg.assign_tool(tool_name, conn_id)
#
#     # In Config persistieren
#     config = load_config()
#     erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
#     registry_data = conn_reg.save_to_config()
#     erp_cfg["tool_assignments"] = registry_data["tool_assignments"]
#     save_config(config)
#
#     global _registry, _connection_registry
#     _registry = None
#     _connection_registry = None
#
#     logger.info("Tool-Zuordnungen aktualisiert von %s: %s", user["username"], data.assignments)
#     return {"message": "Zuordnungen gespeichert"}


# --- Migriert zu Admin-Registry: tools.usage (Sprint 09) ---
# @router.get("/usage")
# async def get_tool_usage(user=Depends(get_admin_user)):
#     """Nutzungsstatistiken pro Tool aus dem query_log.
#
#     Liest die tools_used Spalte (JSON-Array) und aggregiert
#     Nutzungshaeufigkeit und letztes Verwendungsdatum.
#
#     Returns:
#         {"tools": [{"name": str, "usage_count": int, "last_used": str}]}
#     """
#     try:
#         with db_query() as db:
#             rows = db.execute(
#                 "SELECT tools_used, created_at FROM query_log "
#                 "WHERE tools_used IS NOT NULL AND tools_used != '' "
#                 "ORDER BY created_at DESC"
#             ).fetchall()
#     except Exception as exc:
#         logger.error("Fehler beim Lesen der Tool-Nutzungsstatistiken: %s", exc)
#         return {"tools": []}
#
#     # Aggregieren: Zaehler und letztes Datum pro Tool-Name
#     usage: dict[str, dict] = {}
#
#     for row in rows:
#         tools_used_raw = row["tools_used"]
#         created_at = row["created_at"]
#
#         # tools_used ist ein JSON-Array (z.B. '["erp_order_status", "erp_stock_level"]')
#         tool_names = safe_json_loads(tools_used_raw)
#         if not isinstance(tool_names, list):
#             continue
#
#         for tool_name in tool_names:
#             if not isinstance(tool_name, str):
#                 continue
#             if tool_name not in usage:
#                 usage[tool_name] = {
#                     "name": tool_name,
#                     "usage_count": 0,
#                     "last_used": created_at,
#                 }
#             usage[tool_name]["usage_count"] += 1
#             # Da Rows nach created_at DESC sortiert sind, ist der erste
#             # Eintrag bereits der neueste — nur setzen wenn noch nicht gesetzt
#             if usage[tool_name]["last_used"] is None:
#                 usage[tool_name]["last_used"] = created_at
#
#     # Nach Nutzungshaeufigkeit absteigend sortieren
#     tools_list = sorted(usage.values(), key=lambda t: t["usage_count"], reverse=True)
#
#     return {"tools": tools_list}
