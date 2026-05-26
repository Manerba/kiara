"""Admin-Funktionen fuer Tool-Verwaltung, ERP-Konfiguration und Verbindungen."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.shared.constants import PASSWORD_MASK

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Tool-Registry
# ──────────────────────────────────────────────────────────────────────


@admin_function(
    name="tools.list",
    category="tools",
    description="Alle registrierten Tools auflisten",
    help="Listet alle registrierten Tools mit Name, Beschreibung, "
    "Berechtigungen und Parameter-Schema auf.",
    examples=[
        "Welche Tools sind registriert?",
        "Zeige alle Tools",
        "Tool-Liste",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der Tools mit Details",
    raw_response=True,
    http=("GET", "/api/admin/tools/list"),
)
async def tools_list(user: dict) -> AdminResult:
    from kiara.web.api_tools import _get_registry

    registry = _get_registry()
    definitions = registry.list_tools(user_groups=["*"])

    tools = []
    for defn in definitions:
        tools.append(
            {
                "name": defn.name,
                "description": defn.description,
                "requires_confirmation": defn.requires_confirmation,
                "required_permissions": defn.required_permissions,
                "parameters": defn.parameters,
                "category": defn.category,
            }
        )

    return AdminResult.ok(f"{len(tools)} Tools registriert", data={"tools": tools})


@admin_function(
    name="tools.permissions",
    category="tools",
    description="Tool-Berechtigungen pro Gruppe anzeigen",
    help="Zeigt die Berechtigungsmatrix: Welche Gruppe darf welche Tools nutzen.",
    examples=[
        "Zeige Tool-Berechtigungen",
        "Welche Gruppen dürfen welche Tools nutzen?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Berechtigungsmatrix (Gruppe → Tool-Liste)",
    raw_response=True,
    http=("GET", "/api/admin/tools/permissions"),
)
async def tools_permissions(user: dict) -> AdminResult:
    from kiara.config import load_config

    config = load_config()
    tools_cfg = config.get("tools", {})
    permissions = tools_cfg.get("permissions", {})

    return AdminResult.ok("Tool-Berechtigungen", data={"permissions": permissions})


@admin_function(
    name="tools.permissions.update",
    category="tools",
    description="Tool-Berechtigungen für eine Gruppe speichern",
    help="Aktualisiert die Tool-Berechtigungen für eine bestimmte Gruppe. "
    "Erwartet den Gruppennamen und eine Liste erlaubter Tool-Namen.",
    examples=[
        "Setze Tool-Berechtigungen für Gruppe ERP-Leser",
        "Erlaube Gruppe X die Tools erp_order_status und erp_stock_level",
    ],
    parameters={
        "type": "object",
        "properties": {
            "group": {"type": "string", "description": "Gruppenname (DN oder Anzeigename)"},
            "tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Liste erlaubter Tool-Namen",
            },
        },
        "required": ["group", "tools"],
    },
    returns="Bestätigung der Speicherung",
    pk_param="group",
    changelog_area="tools",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/tools/permissions"),
)
async def tools_permissions_update(
    user: dict,
    group: str,
    tools: list[str],
) -> AdminResult:
    from kiara.config import load_config, save_config

    config = load_config()

    if "tools" not in config:
        config["tools"] = {}
    if "permissions" not in config["tools"]:
        config["tools"]["permissions"] = {}

    config["tools"]["permissions"][group] = tools

    save_config(config)
    logger.info(
        "Tool-Berechtigungen fuer Gruppe '%s' aktualisiert von %s: %s",
        group,
        user["username"],
        tools,
    )

    return AdminResult.ok(
        f"Berechtigungen fuer Gruppe '{group}' gespeichert",
        data={"message": f"Berechtigungen fuer Gruppe '{group}' gespeichert."},
    )


@admin_function(
    name="tools.test",
    category="tools",
    description="Tool manuell testen",
    help="Führt ein registriertes Tool mit den übergebenen Parametern "
    "aus und gibt das Ergebnis zurück.",
    examples=[
        "Teste Tool erp_order_status mit Auftrag 12345",
        "Tool-Test für calculate",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name des zu testenden Tools"},
            "params": {
                "type": "object",
                "description": "Test-Eingabeparameter fuer das Tool (tool-abhaengig)",
                "default": {},
            },
        },
        "required": ["name"],
    },
    returns="Tool-Ergebnis (success, data, error, display_text)",
    raw_response=True,
    http=("POST", "/api/admin/tools/test/{name}"),
)
async def tools_test(
    user: dict,
    name: str,
    params: dict | None = None,
) -> AdminResult:
    from kiara.web.api_tools import _get_registry

    if params is None:
        params = {}

    registry = _get_registry()
    tool = registry.get_tool(name)

    if tool is None:
        return AdminResult.fail(f"Tool '{name}' nicht gefunden")

    valid, msg = tool.validate_params(params)
    if not valid:
        return AdminResult.fail(f"Parameter-Validierung fehlgeschlagen: {msg}")

    try:
        result = await tool.execute(params, user_id=user["id"])
    except Exception as exc:
        logger.error("Fehler bei Tool-Test '%s': %s", name, exc)
        return AdminResult.fail(f"Tool-Ausfuehrung fehlgeschlagen: {exc}")

    return AdminResult.ok(
        f"Tool '{name}' ausgefuehrt",
        data={
            "success": result.success,
            "data": result.data,
            "error": result.error,
            "display_text": result.display_text,
        },
    )


@admin_function(
    name="tools.usage",
    category="tools",
    description="Tool-Nutzungsstatistiken anzeigen",
    help="Zeigt Nutzungshäufigkeit und letztes Verwendungsdatum pro Tool aus dem Query-Log.",
    examples=[
        "Welche Tools werden am häufigsten genutzt?",
        "Tool-Nutzungsstatistiken",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Tool-Nutzungsstatistiken (Name, Häufigkeit, letztes Datum)",
    raw_response=True,
    http=("GET", "/api/admin/tools/usage"),
)
async def tools_usage(user: dict) -> AdminResult:
    from kiara.shared.db import db_query
    from kiara.web.config_utils import safe_json_loads

    try:
        with db_query() as db:
            rows = db.execute(
                "SELECT tools_used, created_at FROM query_log "
                "WHERE tools_used IS NOT NULL AND tools_used != '' "
                "ORDER BY created_at DESC"
            ).fetchall()
    except Exception as exc:
        logger.error("Fehler beim Lesen der Tool-Nutzungsstatistiken: %s", exc)
        return AdminResult.ok("Keine Nutzungsdaten", data={"tools": []})

    usage: dict[str, dict] = {}

    for row in rows:
        tools_used_raw = row["tools_used"]
        created_at = row["created_at"]

        tool_names = safe_json_loads(tools_used_raw)
        if not isinstance(tool_names, list):
            continue

        for tool_name in tool_names:
            if not isinstance(tool_name, str):
                continue
            if tool_name not in usage:
                usage[tool_name] = {
                    "name": tool_name,
                    "usage_count": 0,
                    "last_used": created_at,
                }
            usage[tool_name]["usage_count"] += 1
            if usage[tool_name]["last_used"] is None:
                usage[tool_name]["last_used"] = created_at

    tools_list = sorted(usage.values(), key=lambda t: t["usage_count"], reverse=True)

    return AdminResult.ok(
        f"{len(tools_list)} Tools mit Nutzungsdaten",
        data={"tools": tools_list},
    )


@admin_function(
    name="tools.connector_types",
    category="tools",
    description="Verfügbare Konnektortypen auflisten",
    help="Listet alle verfügbaren Konnektortypen mit Config-Schema auf.",
    examples=[
        "Welche Konnektortypen gibt es?",
        "Verfügbare Verbindungstypen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der Konnektortypen mit Config-Schema",
    raw_response=True,
    http=("GET", "/api/admin/tools/connector-types"),
)
async def tools_connector_types(user: dict) -> AdminResult:
    from kiara.web.api_tools import _get_connection_registry

    conn_reg = _get_connection_registry()
    return AdminResult.ok(
        "Konnektortypen",
        data={"types": conn_reg.list_connector_types()},
    )


# ──────────────────────────────────────────────────────────────────────
# ERP-Konfiguration
# ──────────────────────────────────────────────────────────────────────


@admin_function(
    name="tools.erp.config.read",
    category="tools",
    description="ERP-Backend-Konfiguration anzeigen",
    help="Zeigt die aktuelle ERP-Backend-Konfiguration. SQL-DSN wird maskiert.",
    examples=[
        "Zeige ERP-Konfiguration",
        "Wie ist das ERP-Backend konfiguriert?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="ERP-Konfiguration (backend, sql_dsn maskiert, sql_driver, mapping)",
    raw_response=True,
    http=("GET", "/api/admin/tools/erp-config"),
)
async def tools_erp_config_read(user: dict) -> AdminResult:
    from kiara.config import load_config

    config = load_config()
    erp_cfg = config.get("tools", {}).get("erp", {})
    return AdminResult.ok(
        "ERP-Konfiguration",
        data={
            "backend": erp_cfg.get("backend", "mock"),
            "sql_dsn": PASSWORD_MASK if erp_cfg.get("sql_dsn") else "",
            "sql_driver": erp_cfg.get("sql_driver", "auto"),
            "mapping": erp_cfg.get("mapping", {}),
        },
    )


@admin_function(
    name="tools.erp.config.update",
    category="tools",
    description="ERP-Konfiguration speichern",
    help="Speichert die ERP-Backend-Konfiguration. Bei maskiertem SQL-DSN "
    "(********) wird der bestehende Wert beibehalten.",
    examples=[
        "Setze ERP-Backend auf SQL",
        "ERP-Konfiguration speichern",
    ],
    parameters={
        "type": "object",
        "properties": {
            "backend": {
                "type": "string",
                "description": "ERP-Backend-Typ (mock, sql)",
                "default": "mock",
            },
            "sql_dsn": {
                "type": "string",
                "description": "SQL-Verbindungsstring (PASSWORD_MASK = beibehalten)",
                "default": "",
            },
            "sql_driver": {
                "type": "string",
                "description": "SQL-Treiber (auto, mssql, postgresql, mysql)",
                "default": "auto",
            },
            "mapping": {"type": "object", "description": "Tabellen-/Feld-Zuordnung", "default": {}},
        },
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="erp-konfiguration aktualisieren",
    changelog_area="erp_config",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/tools/erp-config"),
)
async def tools_erp_config_update(
    user: dict,
    backend: str = "mock",
    sql_dsn: str = "",
    sql_driver: str = "auto",
    mapping: dict | None = None,
) -> AdminResult:
    from kiara.config import load_config, save_config

    if mapping is None:
        mapping = {}

    config = load_config()
    erp_cfg = config.setdefault("tools", {}).setdefault("erp", {})
    erp_cfg["backend"] = backend
    # PASSWORD_MASK Drei-Wege-Logik (konsistent mit cloud.py)
    if sql_dsn == PASSWORD_MASK:
        # Ungeaendert → bestehenden (ggf. verschluesselten) Wert behalten
        sql_dsn = config.get("tools", {}).get("erp", {}).get("sql_dsn", "")
        if sql_dsn:
            from kiara.shared.crypto import encrypt, is_encrypted

            if not is_encrypted(sql_dsn):
                sql_dsn = encrypt(sql_dsn)
    elif sql_dsn:
        from kiara.shared.crypto import encrypt

        sql_dsn = encrypt(sql_dsn)
    # else: leerer String → DSN loeschen
    erp_cfg["sql_dsn"] = sql_dsn
    erp_cfg["sql_driver"] = sql_driver
    erp_cfg["mapping"] = mapping
    save_config(config)

    # Registry-Cache invalidieren
    import kiara.web.api_tools as _api_tools

    _api_tools._registry = None
    _api_tools._connection_registry = None

    logger.info("ERP-Config aktualisiert von %s: backend=%s", user["username"], backend)
    return AdminResult.ok(
        "ERP-Konfiguration gespeichert",
        data={"message": "ERP-Konfiguration gespeichert"},
    )


@admin_function(
    name="tools.erp.test_connection",
    category="tools",
    description="ERP-SQL-Verbindungstest durchführen",
    help="Testet die Verbindung zum konfigurierten ERP-SQL-Backend.",
    examples=[
        "Teste ERP-Verbindung",
        "Funktioniert die ERP-SQL-Verbindung?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Verbindungstest-Ergebnis (success, tables, driver)",
    raw_response=True,
    http=("POST", "/api/admin/tools/erp-test-connection"),
)
async def tools_erp_test_connection(user: dict) -> AdminResult:
    from kiara.config import load_config

    config = load_config()
    erp_cfg = config.get("tools", {}).get("erp", {})
    if erp_cfg.get("backend") != "sql":
        return AdminResult.fail("Backend ist nicht SQL")
    dsn = erp_cfg.get("sql_dsn", "")
    if not dsn:
        return AdminResult.fail("Kein SQL-DSN konfiguriert")

    from kiara.shared.crypto import decrypt

    dsn = decrypt(dsn) if dsn.startswith("gAAAAA") else dsn

    from kiara.connectors.sql_connector import SQLConnector

    connector = SQLConnector(dsn, erp_cfg.get("sql_driver", "auto"))
    result = connector.test_connection()
    if not result.get("success"):
        return AdminResult.fail(result.get("message", "Verbindungstest fehlgeschlagen"))

    return AdminResult.ok("ERP-Verbindung erfolgreich", data=result)


@admin_function(
    name="tools.erp.table_schema",
    category="tools",
    description="ERP-Tabellen-Schema auslesen",
    help="Zeigt die Spalten einer Tabelle im ERP-SQL-Backend.",
    examples=[
        "Zeige Schema der Tabelle Auftraege",
        "Welche Spalten hat die ERP-Tabelle Kunden?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "table_name": {"type": "string", "description": "Name der ERP-Tabelle"},
        },
        "required": ["table_name"],
    },
    returns="Tabellen-Schema (columns, error)",
    raw_response=True,
    http=("GET", "/api/admin/tools/erp-table-schema/{table_name}"),
)
async def tools_erp_table_schema(user: dict, table_name: str) -> AdminResult:
    from kiara.config import load_config

    config = load_config()
    erp_cfg = config.get("tools", {}).get("erp", {})
    dsn = erp_cfg.get("sql_dsn", "")
    if not dsn:
        return AdminResult.ok(
            "Kein SQL-DSN konfiguriert",
            data={"columns": [], "error": "Kein SQL-DSN konfiguriert"},
        )

    from kiara.shared.crypto import decrypt

    dsn = decrypt(dsn) if dsn.startswith("gAAAAA") else dsn

    from kiara.connectors.sql_connector import SQLConnector

    connector = SQLConnector(dsn, erp_cfg.get("sql_driver", "auto"))
    try:
        schema = connector.get_table_schema(table_name)
        return AdminResult.ok(
            f"Schema fuer Tabelle '{table_name}'",
            data={"columns": schema},
        )
    except Exception as exc:
        return AdminResult.ok(
            "Schema-Abfrage fehlgeschlagen",
            data={"columns": [], "error": str(exc)},
        )


# ──────────────────────────────────────────────────────────────────────
# Connection-Registry
# ──────────────────────────────────────────────────────────────────────


@admin_function(
    name="tools.connections.list",
    category="tools",
    description="ERP-Verbindungen auflisten",
    help="Zeigt alle konfigurierten ERP-Verbindungen. Passwort-Felder werden maskiert.",
    examples=[
        "Zeige ERP-Verbindungen",
        "Welche Verbindungen sind konfiguriert?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der Verbindungen (Passwörter maskiert)",
    raw_response=True,
    http=("GET", "/api/admin/tools/connections"),
)
async def tools_connections_list(user: dict) -> AdminResult:
    from kiara.web.api_tools import (
        _get_connection_registry,
        _get_connector_fields,
        _mask_connection_config,
    )

    conn_reg = _get_connection_registry()
    connections = []
    for conn in conn_reg.list_connections():
        config_fields = _get_connector_fields(conn.connector_type)
        connections.append(
            {
                "id": conn.id,
                "name": conn.name,
                "type": conn.connector_type,
                "config": _mask_connection_config(conn.config, config_fields),
                "enabled": conn.enabled,
            }
        )

    return AdminResult.ok(
        f"{len(connections)} Verbindungen",
        data={"connections": connections},
    )


@admin_function(
    name="tools.connections.create",
    category="tools",
    description="Neue ERP-Verbindung erstellen",
    help="Erstellt eine neue ERP-Verbindung. Passwort-Felder werden automatisch verschlüsselt.",
    examples=[
        "Erstelle eine neue SQL-Verbindung 'Produktions-DB'",
        "Neue ERP-Verbindung anlegen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name der Verbindung"},
            "type": {"type": "string", "description": "Konnektor-Typ (sql, api, ...)"},
            "config": {
                "type": "object",
                "description": "Verbindungskonfiguration (typabhaengig)",
                "default": {},
            },
            "enabled": {"type": "boolean", "description": "Verbindung aktiv", "default": True},
        },
        "required": ["name", "type"],
    },
    returns="Bestätigung mit Verbindungs-ID",
    requires_confirmation=True,
    confirm_template="erp-verbindung {name} erstellen",
    pk_param="name",
    changelog_area="connections",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/tools/connections"),
)
async def tools_connections_create(
    user: dict,
    name: str,
    type: str,
    config: dict | None = None,
    enabled: bool = True,
) -> AdminResult:
    import uuid

    from kiara.config import load_config, save_config
    from kiara.web.api_tools import (
        _encrypt_connection_config,
        _get_connection_registry,
        _get_connector_fields,
    )

    if config is None:
        config = {}

    config_fields = _get_connector_fields(type)
    encrypted_config = _encrypt_connection_config(config, config_fields)

    conn_id = f"conn-{uuid.uuid4().hex[:8]}"

    from kiara.tools.erp_connectors.base import ConnectionInfo

    info = ConnectionInfo(
        id=conn_id,
        name=name,
        connector_type=type,
        config=encrypted_config,
        enabled=enabled,
    )

    conn_reg = _get_connection_registry()
    conn_reg.add_connection(info)

    # In Config persistieren
    cfg = load_config()
    erp_cfg = cfg.setdefault("tools", {}).setdefault("erp", {})
    registry_data = conn_reg.save_to_config()
    erp_cfg["connections"] = registry_data["connections"]
    save_config(cfg)

    # Registry-Cache invalidieren
    import kiara.web.api_tools as _api_tools

    _api_tools._registry = None
    _api_tools._connection_registry = None

    logger.info("ERP-Verbindung '%s' (%s) angelegt von %s", name, type, user["username"])
    return AdminResult.ok(
        f"Verbindung '{name}' angelegt",
        data={"id": conn_id, "message": "Verbindung angelegt"},
    )


@admin_function(
    name="tools.connections.update",
    category="tools",
    description="ERP-Verbindung aktualisieren",
    help="Aktualisiert eine bestehende ERP-Verbindung. Bei maskierten "
    "Passwörtern (********) wird das bestehende Passwort beibehalten.",
    examples=[
        "Aktualisiere Verbindung conn-a1b2c3d4",
        "Ändere den Namen der Verbindung",
    ],
    parameters={
        "type": "object",
        "properties": {
            "connection_id": {
                "type": "string",
                "description": "ID der Verbindung (z.B. conn-a1b2c3d4)",
            },
            "name": {"type": "string", "description": "Name der Verbindung"},
            "type": {"type": "string", "description": "Konnektor-Typ"},
            "config": {"type": "object", "description": "Verbindungskonfiguration", "default": {}},
            "enabled": {"type": "boolean", "description": "Verbindung aktiv", "default": True},
        },
        "required": ["connection_id", "name", "type"],
    },
    returns="Bestätigung der Aktualisierung",
    requires_confirmation=True,
    confirm_template="verbindung {connection_id} aktualisieren",
    pk_param="connection_id",
    changelog_area="connections",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/tools/connections/{connection_id}"),
)
async def tools_connections_update(
    user: dict,
    connection_id: str,
    name: str,
    type: str,
    config: dict | None = None,
    enabled: bool = True,
) -> AdminResult:
    from kiara.config import load_config, save_config
    from kiara.tools.erp_connectors.base import ConnectionInfo
    from kiara.web.api_tools import (
        _encrypt_connection_config,
        _get_connection_registry,
        _get_connector_fields,
    )

    if config is None:
        config = {}

    conn_reg = _get_connection_registry()
    existing = None
    for conn in conn_reg.list_connections():
        if conn.id == connection_id:
            existing = conn
            break
    if not existing:
        return AdminResult.fail(f"Verbindung '{connection_id}' nicht gefunden")

    # Merge: existing.config als Basis, uebergebene Felder ueberschreiben.
    # Verhindert, dass bei teilweisem/leerem config (z.B. LLM-Aufruf via
    # Admin-Chat) Nicht-Secret-Felder verloren gehen.
    merged_config = {**existing.config, **config}

    config_fields = _get_connector_fields(type)
    encrypted_config = _encrypt_connection_config(
        merged_config,
        config_fields,
        existing_config=existing.config,
    )

    info = ConnectionInfo(
        id=connection_id,
        name=name,
        connector_type=type,
        config=encrypted_config,
        enabled=enabled,
    )
    conn_reg.add_connection(info)

    # In Config persistieren
    cfg = load_config()
    erp_cfg = cfg.setdefault("tools", {}).setdefault("erp", {})
    registry_data = conn_reg.save_to_config()
    erp_cfg["connections"] = registry_data["connections"]
    save_config(cfg)

    import kiara.web.api_tools as _api_tools

    _api_tools._registry = None
    _api_tools._connection_registry = None

    logger.info("ERP-Verbindung '%s' aktualisiert von %s", connection_id, user["username"])
    return AdminResult.ok(
        "Verbindung aktualisiert",
        data={"message": "Verbindung aktualisiert"},
    )


@admin_function(
    name="tools.connections.delete",
    category="tools",
    description="ERP-Verbindung löschen",
    help="Löscht eine bestehende ERP-Verbindung und entfernt die zugehörigen Tool-Zuordnungen.",
    examples=[
        "Lösche Verbindung conn-a1b2c3d4",
        "ERP-Verbindung entfernen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "connection_id": {"type": "string", "description": "ID der zu loeschenden Verbindung"},
        },
        "required": ["connection_id"],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="verbindung {connection_id} löschen",
    pk_param="connection_id",
    changelog_area="connections",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/tools/connections/{connection_id}"),
)
async def tools_connections_delete(user: dict, connection_id: str) -> AdminResult:
    from kiara.config import load_config, save_config
    from kiara.web.api_tools import _get_connection_registry

    conn_reg = _get_connection_registry()
    conn_reg.remove_connection(connection_id)

    # In Config persistieren
    cfg = load_config()
    erp_cfg = cfg.setdefault("tools", {}).setdefault("erp", {})
    registry_data = conn_reg.save_to_config()
    erp_cfg["connections"] = registry_data["connections"]
    erp_cfg["tool_assignments"] = registry_data["tool_assignments"]
    save_config(cfg)

    import kiara.web.api_tools as _api_tools

    _api_tools._registry = None
    _api_tools._connection_registry = None

    logger.info("ERP-Verbindung '%s' geloescht von %s", connection_id, user["username"])
    return AdminResult.ok(
        "Verbindung geloescht",
        data={"message": "Verbindung geloescht"},
    )


@admin_function(
    name="tools.connections.test",
    category="tools",
    description="ERP-Verbindung testen",
    help="Testet eine bestehende ERP-Verbindung und gibt das Ergebnis zurück.",
    examples=[
        "Teste Verbindung conn-a1b2c3d4",
        "Funktioniert die ERP-Verbindung?",
    ],
    parameters={
        "type": "object",
        "properties": {
            "connection_id": {"type": "string", "description": "ID der zu testenden Verbindung"},
        },
        "required": ["connection_id"],
    },
    returns="Verbindungstest-Ergebnis",
    raw_response=True,
    http=("POST", "/api/admin/tools/connections/{connection_id}/test"),
)
async def tools_connections_test(user: dict, connection_id: str) -> AdminResult:
    from kiara.web.api_tools import _get_connection_registry

    conn_reg = _get_connection_registry()
    try:
        connector = conn_reg.get_connector(connection_id)
    except (KeyError, ValueError) as exc:
        return AdminResult.fail(str(exc))
    try:
        result = await connector.test_connection()
        return AdminResult.ok("Verbindungstest", data=result)
    except Exception as exc:
        return AdminResult.fail(f"Testfehler: {exc}")


@admin_function(
    name="tools.assignments",
    category="tools",
    description="Tool→Verbindung Zuordnungen anzeigen",
    help="Zeigt die aktuellen Zuordnungen von Tools zu ERP-Verbindungen.",
    examples=[
        "Zeige Tool-Zuordnungen",
        "Welches Tool nutzt welche Verbindung?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Zuordnung Tool-Name → Connection-ID",
    raw_response=True,
    http=("GET", "/api/admin/tools/assignments"),
)
async def tools_assignments(user: dict) -> AdminResult:
    from kiara.web.api_tools import _get_connection_registry

    conn_reg = _get_connection_registry()
    return AdminResult.ok(
        "Tool-Zuordnungen",
        data={"assignments": conn_reg.get_tool_assignments()},
    )


@admin_function(
    name="tools.assignments.update",
    category="tools",
    description="Tool→Verbindung Zuordnungen speichern",
    help="Speichert die Zuordnungen von Tools zu ERP-Verbindungen. "
    "Erwartet ein Dictionary mit Tool-Name → Connection-ID.",
    examples=[
        "Ordne erp_order_status der Verbindung conn-abc zu",
        "Tool-Zuordnungen aktualisieren",
    ],
    parameters={
        "type": "object",
        "properties": {
            "assignments": {
                "type": "object",
                "description": "Zuordnung Tool-Name → Connection-ID (dict)",
            },
        },
        "required": ["assignments"],
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="tool-zuordnungen aktualisieren",
    changelog_area="assignments",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/tools/assignments"),
)
async def tools_assignments_update(user: dict, assignments: dict) -> AdminResult:
    from kiara.config import load_config, save_config
    from kiara.web.api_tools import _get_connection_registry

    conn_reg = _get_connection_registry()
    conn_reg._tool_assignments.clear()
    for tool_name, conn_id in assignments.items():
        if conn_id:
            conn_reg.assign_tool(tool_name, conn_id)

    # In Config persistieren
    cfg = load_config()
    erp_cfg = cfg.setdefault("tools", {}).setdefault("erp", {})
    registry_data = conn_reg.save_to_config()
    erp_cfg["tool_assignments"] = registry_data["tool_assignments"]
    save_config(cfg)

    import kiara.web.api_tools as _api_tools

    _api_tools._registry = None
    _api_tools._connection_registry = None

    logger.info("Tool-Zuordnungen aktualisiert von %s: %s", user["username"], assignments)
    return AdminResult.ok(
        "Zuordnungen gespeichert",
        data={"message": "Zuordnungen gespeichert"},
    )
