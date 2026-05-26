"""Admin-Funktionen fuer LDAP-Verwaltung."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.shared.constants import PASSWORD_MASK

logger = logging.getLogger(__name__)


@admin_function(
    name="ldap.config.read",
    category="ldap",
    description="LDAP-Konfiguration anzeigen",
    help="Zeigt die aktuelle LDAP-Konfiguration an. Das Bind-Passwort "
    "wird aus Sicherheitsgründen maskiert.",
    examples=[
        "Zeig mir die LDAP-Konfiguration",
        "LDAP-Einstellungen?",
        "Wie ist LDAP konfiguriert?",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="LDAP-Konfiguration (Passwort maskiert)",
    raw_response=True,
    http=("GET", "/api/admin/ldap/config"),
)
async def ldap_config_read(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    config = repo.get("ldap_config")
    if not config:
        return AdminResult.ok("Keine LDAP-Konfiguration vorhanden", data={"config": None})

    # Passwort maskieren
    if config.get("bind_password"):
        config["bind_password"] = PASSWORD_MASK
    return AdminResult.ok("LDAP-Konfiguration", data={"config": config})


@admin_function(
    name="ldap.config.update",
    category="ldap",
    description="LDAP-Konfiguration speichern",
    help="Speichert die LDAP-Verbindungskonfiguration. Testet die Verbindung "
    "vor dem Speichern. Bei maskiertem Passwort (********) wird das "
    "bestehende Passwort beibehalten.",
    examples=[
        "Konfiguriere LDAP mit Server ldaps://ad.firma.de:636",
        "LDAP-Konfiguration speichern",
    ],
    parameters={
        "type": "object",
        "properties": {
            "server_url": {
                "type": "string",
                "description": "LDAP-Server-URL (z.B. ldaps://ad.firma.de)",
            },
            "port": {
                "type": "integer",
                "description": "Port (Standard: 389, LDAPS: 636)",
                "default": 389,
            },
            "use_ssl": {
                "type": "integer",
                "description": "SSL verwenden (0=nein, 1=ja)",
                "default": 0,
            },
            "bind_dn": {
                "type": "string",
                "description": "Bind-DN (z.B. cn=kiara,ou=services,dc=firma,dc=de)",
            },
            "bind_password": {
                "type": "string",
                "description": "Bind-Passwort (******** = bestehendes beibehalten)",
            },
            "search_base": {
                "type": "string",
                "description": "Such-Basis (z.B. ou=users,dc=firma,dc=de)",
            },
        },
        "required": ["server_url", "bind_dn", "bind_password", "search_base"],
    },
    returns="Bestätigung der Speicherung",
    requires_confirmation=True,
    confirm_template="ldap-konfiguration speichern",
    changelog_area="ldap",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/ldap/config"),
)
async def ldap_config_update(
    user: dict,
    server_url: str,
    bind_dn: str,
    bind_password: str,
    search_base: str,
    port: int = 389,
    use_ssl: int = 0,
) -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.shared.crypto import encrypt

    repo = ConfigRepository()

    # PASSWORD_MASK Drei-Wege-Logik (Spec §5):
    # Leer/None → bestehendes beibehalten, PASSWORD_MASK → beibehalten, Neuer Wert → verschlüsseln
    effective_password = bind_password
    if bind_password == PASSWORD_MASK or not bind_password:
        existing = repo.get("ldap_config")
        if existing and existing.get("bind_password"):
            effective_password = existing["bind_password"]
        elif bind_password == PASSWORD_MASK:
            return AdminResult.fail("Kein bestehendes Passwort vorhanden")
        else:
            effective_password = ""  # Leer bei Create ohne bestehendes PW
    else:
        effective_password = encrypt(bind_password)

    # Verbindungstest vor dem Speichern (wie api_ldap.py Zeile 96)
    # WICHTIG: decrypt() erfordert vorherige is_encrypted()-Prüfung (crypto.py Zeile 87)
    from kiara.shared.crypto import decrypt, is_encrypted

    if bind_password == PASSWORD_MASK or not bind_password:
        # Bestehendes (verschlüsseltes) Passwort für Test entschlüsseln
        if effective_password and is_encrypted(effective_password):
            try:
                test_password = decrypt(effective_password)
            except Exception as exc:
                return AdminResult.fail(f"Passwort-Entschlüsselung fehlgeschlagen: {exc}")
        else:
            test_password = effective_password or ""
    else:
        test_password = bind_password  # Klartext, noch nicht verschlüsselt

    test_cfg = {
        "server": server_url,
        "port": port,
        "use_ssl": use_ssl,
        "bind_dn": bind_dn,
        "bind_password": test_password,
        "search_base": search_base,
    }
    import asyncio

    from kiara.shared.ldap import test_connection

    success, error_msg = await asyncio.to_thread(test_connection, test_cfg)
    if not success:
        return AdminResult.fail(f"LDAP-Verbindungstest fehlgeschlagen: {error_msg}")

    repo.set(
        "ldap_config",
        {
            "server_url": server_url,
            "port": port,
            "use_ssl": use_ssl,
            "bind_dn": bind_dn,
            "bind_password": effective_password,
            "search_base": search_base,
        },
    )

    return AdminResult.ok(
        "LDAP-Konfiguration gespeichert",
        data={"message": "LDAP-Konfiguration gespeichert"},
    )


@admin_function(
    name="ldap.config.delete",
    category="ldap",
    description="LDAP-Konfiguration löschen",
    help="ACHTUNG: Löscht die LDAP-Konfiguration und deaktiviert alle "
    "LDAP-importierten Benutzer. Diese können sich nicht mehr anmelden!",
    examples=[
        "Lösche die LDAP-Konfiguration",
        "LDAP deaktivieren",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung der Löschung",
    requires_confirmation=True,
    confirm_template="ldap-konfiguration löschen",
    changelog_area="ldap",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ldap/config"),
)
async def ldap_config_delete(user: dict) -> AdminResult:
    from kiara.shared.db import db_transaction

    # LDAP-Config löschen + LDAP-User deaktivieren in EINER Transaktion
    with db_transaction() as db:
        db.execute("DELETE FROM ldap_config WHERE id = 1")
        db.execute("UPDATE users SET is_active = false WHERE auth_source = 'ldap'")

    return AdminResult.ok(
        "LDAP-Konfiguration gelöscht, LDAP-User deaktiviert",
        data={"message": "LDAP-Konfiguration gelöscht, LDAP-User deaktiviert"},
    )


@admin_function(
    name="ldap.groups",
    category="ldap",
    description="Verfügbare AD-Gruppen auflisten",
    help="Verbindet sich mit dem LDAP-Server und listet alle verfügbaren "
    "Active-Directory-Gruppen auf.",
    examples=[
        "Welche AD-Gruppen gibt es?",
        "Zeig mir die LDAP-Gruppen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste der verfügbaren AD-Gruppen",
    raw_response=True,
    http=("GET", "/api/admin/ldap/groups"),
)
async def ldap_groups(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    config = repo.get("ldap_config")
    if not config:
        return AdminResult.fail("Keine LDAP-Konfiguration vorhanden")

    # LDAP-Verbindung + Gruppensuche
    from kiara.shared.crypto import decrypt, is_encrypted

    password = config.get("bind_password", "")
    try:
        if password and is_encrypted(password):
            password = decrypt(password)
    except Exception as exc:
        return AdminResult.fail(f"Passwort-Entschlüsselung fehlgeschlagen: {exc}")
    conn_cfg = {
        "server": config["server_url"],
        "port": config.get("port", 389),
        "use_ssl": config.get("use_ssl", 0),
        "bind_dn": config["bind_dn"],
        "bind_password": password,
        "search_base": config["search_base"],
    }

    # WICHTIG: get_groups() ist blockierend (Netzwerk) → asyncio.to_thread!
    import asyncio

    try:
        from kiara.shared.ldap import get_groups

        groups = await asyncio.to_thread(get_groups, conn_cfg)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Gruppenabfrage fehlgeschlagen: {exc}")

    return AdminResult.ok(
        f"{len(groups)} Gruppe(n) gefunden",
        data={"groups": groups},
    )


@admin_function(
    name="ldap.select_group",
    category="ldap",
    description="AD-Gruppe auswählen und Sync starten",
    help="Wählt eine AD-Gruppe für die Benutzer-Synchronisation aus. "
    "Startet automatisch einen initialen Sync.",
    examples=[
        "Wähle die Gruppe CN=KIara-Users aus",
        "LDAP-Gruppe KIara-Users für Sync auswählen",
    ],
    parameters={
        "type": "object",
        "properties": {
            "group_dn": {
                "type": "string",
                "description": "Distinguished Name der Gruppe",
            },
            "group_name": {
                "type": "string",
                "description": "Anzeigename der Gruppe",
            },
        },
        "required": ["group_dn", "group_name"],
    },
    returns="Bestätigung + Sync-Ergebnis",
    requires_confirmation=True,
    confirm_template="ldap-gruppe {group_name} auswählen",
    changelog_area="ldap",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/ldap/group"),
)
async def ldap_select_group(user: dict, group_dn: str, group_name: str) -> AdminResult:
    import asyncio

    from kiara.repositories.config import ConfigRepository

    repo = ConfigRepository()
    config = repo.get("ldap_config")
    if not config:
        return AdminResult.fail("Keine LDAP-Konfiguration vorhanden")

    # Whitelist der Felder, die wir hier verwalten — Rollback und Update
    # nutzen exakt dieselbe Feldmenge, damit kein id/last_sync_at/updated_at
    # ungewollt überschrieben wird.
    base_fields = {
        "server_url": config["server_url"],
        "port": config["port"],
        "use_ssl": config.get("use_ssl", 0),
        "bind_dn": config["bind_dn"],
        "bind_password": config["bind_password"],
        "search_base": config["search_base"],
    }
    original_fields = {
        **base_fields,
        "group_dn": config.get("group_dn"),
        "group_name": config.get("group_name"),
    }

    # Bestehende Config um Gruppen-Felder erweitern
    repo.set("ldap_config", {**base_fields, "group_dn": group_dn, "group_name": group_name})

    # Initialen Sync starten — Rollback bei Exception ODER Fehler-Result
    from kiara.web.ldap_sync import sync_ldap_users

    try:
        result = await asyncio.to_thread(sync_ldap_users)
    except Exception as exc:
        repo.set("ldap_config", original_fields)
        return AdminResult.fail(f"Sync fehlgeschlagen: {exc}")

    if "error" in result:
        repo.set("ldap_config", original_fields)
        return AdminResult.fail(f"Sync fehlgeschlagen: {result['error']}")

    return AdminResult.ok(
        f"Gruppe '{group_name}' ausgewählt und synchronisiert",
        data={"message": f"Gruppe '{group_name}' ausgewählt und synchronisiert", "sync": result},
    )


@admin_function(
    name="ldap.deselect_group",
    category="ldap",
    description="LDAP-Gruppenauswahl aufheben",
    help="ACHTUNG: Hebt die Gruppenauswahl für LDAP-Sync auf und "
    "deaktiviert alle LDAP-synchronisierten Benutzer!",
    examples=[
        "Hebe die LDAP-Gruppenauswahl auf",
        "LDAP-Gruppe abwählen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung",
    requires_confirmation=True,
    confirm_template="ldap-gruppenauswahl aufheben",
    changelog_area="ldap",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/ldap/group"),
)
async def ldap_deselect_group(user: dict) -> AdminResult:
    from kiara.shared.db import db_transaction

    with db_transaction() as db:
        db.execute("UPDATE ldap_config SET group_dn = NULL, group_name = NULL WHERE id = 1")
        db.execute("UPDATE users SET is_active = false WHERE auth_source = 'ldap'")

    return AdminResult.ok(
        "Gruppenauswahl aufgehoben, LDAP-Benutzer deaktiviert",
        data={"message": "Gruppenauswahl aufgehoben, LDAP-Benutzer deaktiviert"},
    )


@admin_function(
    name="ldap.sync",
    category="ldap",
    description="Manuellen LDAP-Sync starten",
    help="Synchronisiert Benutzer aus der konfigurierten AD-Gruppe. "
    "Erstellt neue Benutzer und deaktiviert entfernte.",
    examples=[
        "Starte einen LDAP-Sync",
        "Synchronisiere die LDAP-Benutzer",
        "LDAP-Sync manuell auslösen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Sync-Ergebnis (erstellt, aktualisiert, deaktiviert)",
    requires_confirmation=True,
    confirm_template="ldap-sync starten",
    changelog_area="ldap",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/ldap/sync"),
)
async def ldap_sync(user: dict) -> AdminResult:
    import asyncio

    from kiara.web.ldap_sync import sync_ldap_users

    try:
        result = await asyncio.to_thread(sync_ldap_users)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Sync fehlgeschlagen: {exc}")

    if "error" in result:
        return AdminResult.fail(f"LDAP-Sync fehlgeschlagen: {result['error']}")

    return AdminResult.ok(
        "LDAP-Sync abgeschlossen",
        data={"message": "Synchronisierung abgeschlossen", "sync": result},
    )
