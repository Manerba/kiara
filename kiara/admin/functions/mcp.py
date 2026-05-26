"""Admin-Funktionen fuer MCP-Server-Management."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="mcp.status",
    category="mcp",
    description="MCP-Server-Status anzeigen",
    help="Zeigt den Status des MCP-Servers an (aktiv/inaktiv, Port, Konfiguration).",
    examples=[
        "Ist der MCP-Server aktiv?",
        "MCP-Status?",
        "Wie ist der MCP-Server konfiguriert?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="MCP-Server-Status und Konfiguration",
    raw_response=True,
    http=("GET", "/api/admin/mcp/status"),
)
async def mcp_status(user: dict) -> AdminResult:
    import asyncio

    from kiara.config import load_config
    from kiara.shared.constants import MCP_PORT_DEFAULT

    cfg = load_config()
    config_enabled = cfg.get("web", {}).get("mcp_enabled", False)
    port = cfg.get("web", {}).get("mcp_port", MCP_PORT_DEFAULT)

    service_status = "unknown"
    service_running = False
    pid = None
    uptime = None

    try:
        proc = await asyncio.create_subprocess_exec(
            "systemctl",
            "is-active",
            "kiara-mcp",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        service_status = stdout.decode().strip()
        service_running = proc.returncode == 0

        proc2 = await asyncio.create_subprocess_exec(
            "systemctl",
            "show",
            "kiara-mcp",
            "--property=MainPID,ActiveEnterTimestampMonotonic",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout2, _ = await proc2.communicate()
        prop_dict = {}
        for line in stdout2.decode().split("\n"):
            if "=" in line:
                k, v = line.split("=", 1)
                prop_dict[k] = v

        raw_pid = int(prop_dict.get("MainPID") or "0")
        pid = raw_pid if raw_pid > 0 else None

        ts_mono = prop_dict.get("ActiveEnterTimestampMonotonic", "0")
        if pid and ts_mono and ts_mono != "0":
            try:
                service_start_us = int(ts_mono)
                with open("/proc/uptime") as f:
                    system_uptime_s = float(f.read().split()[0])
                system_uptime_us = int(system_uptime_s * 1_000_000)
                service_uptime_s = (system_uptime_us - service_start_us) // 1_000_000
                if service_uptime_s >= 0:
                    hours, remainder = divmod(service_uptime_s, 3600)
                    minutes = remainder // 60
                    uptime = f"{hours}h {minutes}min"
            except (ValueError, TypeError, OSError, IndexError):
                uptime = None
    except (FileNotFoundError, OSError, ValueError):
        pass

    return AdminResult.ok(
        "MCP-Status",
        data={
            "config_enabled": config_enabled,
            "service_running": service_running,
            "service_status": service_status,
            "port": port,
            "uptime": uptime,
            "pid": pid,
        },
    )


@admin_function(
    name="mcp.toggle",
    category="mcp",
    description="MCP-Server ein-/ausschalten",
    help="Aktiviert oder deaktiviert den MCP-Server. Ändert die Konfiguration "
    "und startet/stoppt den systemd-Service.",
    examples=[
        "Aktiviere den MCP-Server",
        "MCP ausschalten",
        "Schalte MCP ein",
    ],
    parameters={
        "type": "object",
        "properties": {
            "enabled": {
                "type": "boolean",
                "description": "True = aktivieren, False = deaktivieren",
            },
        },
        "required": ["enabled"],
    },
    returns="Bestätigung der Änderung",
    requires_confirmation=True,
    confirm_template="mcp-server {enabled}",
    changelog_area="mcp",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/mcp/toggle"),
)
async def mcp_toggle(user: dict, enabled: bool) -> AdminResult:
    import asyncio

    from kiara.config import load_config, save_config

    # 1. Config ändern (wie api_mcp.py Zeile 59-61)
    cfg = load_config()
    old_enabled = cfg.get("web", {}).get("mcp_enabled", False)
    cfg.setdefault("web", {})["mcp_enabled"] = enabled

    def _rollback_config() -> None:
        try:
            cfg["web"]["mcp_enabled"] = old_enabled
            save_config(cfg)
        except Exception as exc:
            logger.warning("MCP-Config-Rollback fehlgeschlagen: %s", exc)

    try:
        save_config(cfg)
    except (PermissionError, OSError, ValueError) as exc:
        return AdminResult.fail(f"Config-Fehler: {exc}")

    # 2. Service steuern (wie api_mcp.py Zeile 66-77)
    svc_warning = None
    try:
        action = "enable" if enabled else "disable"
        proc = await asyncio.create_subprocess_exec(
            "sudo",
            "systemctl",
            action,
            "--now",
            "kiara-mcp",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            svc_warning = stderr.decode().strip()
            logger.warning("systemctl %s --now fehlgeschlagen: %s", action, svc_warning)
    except (FileNotFoundError, OSError) as exc:
        svc_warning = str(exc)
        logger.warning("systemctl fehlgeschlagen: %s", exc)

    # 3. Status prüfen — Polling statt fixem Sleep (deterministisch)
    service_status = "unknown"
    expected = "active" if enabled else "inactive"
    for _ in range(10):
        await asyncio.sleep(0.5)
        try:
            proc = await asyncio.create_subprocess_exec(
                "systemctl",
                "is-active",
                "kiara-mcp",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await proc.communicate()
            service_status = stdout.decode().strip()
            if service_status == expected:
                break
        except Exception:
            service_status = "unknown"

    if svc_warning:
        # Config zurückrollen — Service-Zustand und Config konsistent halten
        _rollback_config()
        return AdminResult.fail(
            f"Service-Steuerung fehlgeschlagen: {svc_warning}",
            data={"enabled": old_enabled, "service_status": service_status},
        )

    if service_status != expected:
        _rollback_config()
        return AdminResult.fail(
            f"Service-Status inkonsistent: {service_status} (erwartet: {expected})",
            data={
                "enabled": old_enabled,
                "requested_enabled": enabled,
                "service_status": service_status,
                "expected_status": expected,
            },
        )

    status_text = "aktiviert" if enabled else "deaktiviert"
    msg = f"MCP-Server {status_text}"
    return AdminResult.ok(
        msg,
        data={
            "enabled": enabled,
            "service_status": service_status,
            "message": msg,
        },
    )


@admin_function(
    name="mcp.keys.list",
    category="mcp",
    description="API-Keys auflisten",
    help="Listet alle MCP-API-Keys auf (ohne die Keys selbst — nur Prefix, Name, Status).",
    examples=[
        "Welche API-Keys gibt es?",
        "Zeig mir die MCP-Keys",
        "API-Key-Liste",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der API-Keys",
    raw_response=True,
    http=("GET", "/api/admin/mcp/keys"),
)
async def mcp_keys_list(user: dict) -> AdminResult:
    from kiara.shared.db import db_query

    with db_query() as db:
        rows = db.execute(
            """SELECT ak.id, ak.name, ak.key_prefix, ak.is_active, ak.expires_at,
                      ak.last_used_at, ak.created_at, ak.user_id, u.username
               FROM api_keys ak
               JOIN users u ON ak.user_id = u.id
               ORDER BY ak.created_at DESC"""
        ).fetchall()
    keys = [dict(r) for r in rows]
    return AdminResult.ok(f"{len(keys)} API-Key(s)", data={"data": keys})


@admin_function(
    name="mcp.keys.create",
    category="mcp",
    description="Neuen MCP-API-Key generieren",
    help="Generiert einen neuen API-Key für den MCP-Server. Der vollständige Key "
    "wird nur einmal angezeigt — danach ist nur noch der Prefix sichtbar.",
    examples=[
        "Erstelle einen neuen API-Key",
        "Generiere einen MCP-Key für User 1",
        "Neuer API-Key mit Name 'Produktions-Key'",
    ],
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Anzeigename des Keys",
            },
            "user_id": {
                "type": "integer",
                "description": "Benutzer-ID dem der Key zugeordnet wird",
            },
            "expires_at": {
                "type": "string",
                "description": "Ablaufdatum im ISO-Format (optional, z.B. 2026-12-31)",
            },
        },
        "required": ["name", "user_id"],
    },
    returns="Generierter API-Key (wird nur einmal angezeigt!)",
    pk_param="name",
    changelog_area="mcp",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/mcp/keys"),
)
async def mcp_keys_create(
    user: dict, name: str, user_id: int, expires_at: str | None = None
) -> AdminResult:
    from kiara.repositories.api_key import ApiKeyRepository
    from kiara.repositories.user import UserRepository

    # User validieren
    user_repo = UserRepository()
    target_user = user_repo.get_by_id(user_id)
    if not target_user:
        return AdminResult.fail(f"Benutzer {user_id} nicht gefunden")

    # Key generieren — Logik inline (aus api_mcp.py Zeile 149-159 extrahiert)
    import hashlib
    import secrets

    raw = secrets.token_hex(24)  # 48 Hex-Zeichen = 192 Bit
    full_key = f"orag_{raw}"
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()
    key_prefix = full_key[:12] + "..."

    repo = ApiKeyRepository()
    repo.create(
        name=name,
        user_id=user_id,
        key_hash=key_hash,
        key_prefix=key_prefix,
        expires_at=expires_at,
    )

    return AdminResult.ok(
        f"API-Key '{name}' erstellt",
        data={
            "key": full_key,
            "prefix": key_prefix,
            "message": "Key nur jetzt sichtbar — sicher aufbewahren!",
        },
    )


@admin_function(
    name="mcp.keys.delete",
    category="mcp",
    description="API-Key löschen",
    help="Widerruft einen MCP-API-Key. Der Key kann danach nicht mehr "
    "für Authentifizierung verwendet werden.",
    examples=[
        "Lösche API-Key 5",
        "Widerrufe den alten MCP-Key",
    ],
    parameters={
        "type": "object",
        "properties": {
            "key_id": {
                "type": "integer",
                "description": "ID des API-Keys",
            },
        },
        "required": ["key_id"],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="api-key {key_id} löschen",
    pk_param="key_id",
    changelog_area="mcp",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/mcp/keys/{key_id}"),
)
async def mcp_keys_delete(user: dict, key_id: int) -> AdminResult:
    from kiara.repositories.api_key import ApiKeyRepository

    repo = ApiKeyRepository()
    if not repo.exists(key_id):
        return AdminResult.fail(f"API-Key {key_id} nicht gefunden")
    repo.delete(key_id)
    return AdminResult.ok(
        f"API-Key {key_id} gelöscht",
        data={"message": "API-Key geloescht"},
    )


@admin_function(
    name="mcp.keys.toggle",
    category="mcp",
    description="API-Key aktivieren/deaktivieren",
    help="Aktiviert oder deaktiviert einen MCP-API-Key ohne ihn zu löschen.",
    examples=[
        "Deaktiviere API-Key 3",
        "Aktiviere den API-Key wieder",
    ],
    parameters={
        "type": "object",
        "properties": {
            "key_id": {
                "type": "integer",
                "description": "ID des API-Keys",
            },
        },
        "required": ["key_id"],
    },
    returns="Neuer Status des Keys",
    requires_confirmation=True,
    confirm_template="api-key {key_id} umschalten",
    pk_param="key_id",
    changelog_area="mcp",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/mcp/keys/{key_id}/toggle"),
)
async def mcp_keys_toggle(user: dict, key_id: int) -> AdminResult:
    from kiara.repositories.api_key import ApiKeyRepository

    repo = ApiKeyRepository()
    key = repo.get_by_id(key_id)
    if not key:
        return AdminResult.fail(f"API-Key {key_id} nicht gefunden")
    new_status = not key["is_active"]
    repo.update(key_id, is_active=new_status)
    status_text = "aktiviert" if new_status else "deaktiviert"
    return AdminResult.ok(
        f"API-Key {key_id} {status_text}",
        data={"is_active": new_status},
    )
