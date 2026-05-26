"""Admin-Funktionen fuer Gruppenverwaltung (Sprint 06)."""

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="groups.list",
    category="groups",
    description="Alle importierten Gruppen auflisten",
    help="Listet alle in die lokale DB importierten LDAP-Gruppen mit "
    "Mitglieder- und Datenquellen-Anzahl auf.",
    examples=["Zeige alle Gruppen", "Welche Gruppen gibt es?"],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste aller Gruppen mit Zaehlern",
    raw_response=True,
    http=("GET", "/api/admin/groups"),
)
async def groups_list(user: dict) -> AdminResult:
    from kiara.repositories.group import GroupRepository

    group_repo = GroupRepository()
    rows = group_repo.list_with_counts()
    return AdminResult.ok(f"{len(rows)} Gruppen", data={"data": rows})


@admin_function(
    name="groups.members",
    category="groups",
    description="Mitglieder einer Gruppe anzeigen",
    help="Listet alle Benutzer auf, die Mitglied einer bestimmten Gruppe sind.",
    examples=["Zeige Mitglieder von Gruppe 3", "Wer ist in der IT-Gruppe?"],
    parameters={
        "type": "object",
        "properties": {
            "group_id": {"type": "integer", "description": "Gruppen-ID"},
        },
        "required": ["group_id"],
    },
    returns="Liste der Gruppen-Mitglieder",
    raw_response=True,
    http=("GET", "/api/admin/groups/{group_id}/members"),
)
async def groups_members(user: dict, group_id: int) -> AdminResult:
    from kiara.repositories.group import GroupRepository

    group_repo = GroupRepository()
    if not group_repo.exists(group_id):
        return AdminResult.fail("Gruppe nicht gefunden")
    members = group_repo.get_members(group_id)
    return AdminResult.ok(f"{len(members)} Mitglieder", data={"members": members})


@admin_function(
    name="groups.delete",
    category="groups",
    description="Importierte Gruppe loeschen",
    help="Loescht eine importierte Gruppe und alle Zuordnungen (Cascade). "
    "Betroffene Datenquellen-Berechtigungen werden entfernt.",
    examples=["Loesche Gruppe 5", "Entferne die HR-Gruppe"],
    parameters={
        "type": "object",
        "properties": {
            "group_id": {"type": "integer", "description": "Gruppen-ID"},
        },
        "required": ["group_id"],
    },
    returns="Loeschbestaetigung",
    requires_confirmation=True,
    confirm_template="gruppe {group_id} löschen",
    pk_param="group_id",
    changelog_area="groups",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/groups/{group_id}"),
)
async def groups_delete(user: dict, group_id: int) -> AdminResult:
    from kiara.repositories.group import GroupRepository

    group_repo = GroupRepository()
    if not group_repo.exists(group_id):
        return AdminResult.fail("Gruppe nicht gefunden")
    group_repo.delete_cascade(group_id)
    return AdminResult.ok(
        f"Gruppe {group_id} entfernt",
        data={"message": "Gruppe entfernt", "name": f"group_{group_id}"},
    )


@admin_function(
    name="groups.import_ldap",
    category="groups",
    description="Gruppen aus LDAP importieren",
    help="Importiert LDAP-Gruppen in die lokale Datenbank. Bereits vorhandene "
    "Gruppen werden aktualisiert (Upsert nach DN). Optional kann ein "
    "alternativer base_dn oder spezifische group_dns angegeben werden.",
    examples=[
        "Importiere Gruppen aus LDAP",
        "Importiere nur die Gruppen cn=IT und cn=HR aus dem LDAP",
    ],
    parameters={
        "type": "object",
        "properties": {
            "base_dn": {"type": "string", "description": "Alternativer LDAP Base-DN (optional)"},
            "group_dns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Nur diese DNs importieren (optional, leer = alle)",
            },
        },
        "required": [],
    },
    returns="Anzahl importierter Gruppen",
    requires_confirmation=True,
    confirm_template="ldap gruppen importieren",
    changelog_area="groups",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/groups/import"),
)
async def groups_import_ldap(
    user: dict, base_dn: str = "", group_dns: list[str] | None = None
) -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.repositories.group import GroupRepository
    from kiara.web.config_utils import build_ldap_config
    from kiara.web.ldap_utils import get_groups

    config_repo = ConfigRepository()
    row = config_repo.get("ldap_config")
    if not row:
        return AdminResult.fail("LDAP nicht konfiguriert")
    try:
        ldap_config = build_ldap_config(row)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Config fehlerhaft (Entschluesselung?): {exc}")

    if base_dn:
        ldap_config["search_base"] = base_dn

    try:
        ldap_groups = get_groups(ldap_config)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Abfrage fehlgeschlagen: {exc}")
    if not ldap_groups:
        return AdminResult.ok(
            "Keine Gruppen gefunden", data={"imported": 0, "message": "Keine Gruppen gefunden"}
        )

    # Filter nach spezifischen DNs
    if group_dns:
        dn_set = set(group_dns)
        ldap_groups = [g for g in ldap_groups if g["dn"] in dn_set]
        if not ldap_groups:
            return AdminResult.fail(
                f"Keine der angefragten Gruppen im LDAP gefunden: {', '.join(group_dns)}"
            )

    missing_sid = [
        g.get("dn") or g.get("cn") or "<unbekannt>" for g in ldap_groups if not g.get("sid")
    ]
    if missing_sid:
        return AdminResult.fail(
            "LDAP-Gruppen ohne objectSid werden nicht importiert: " + ", ".join(missing_sid)
        )

    group_repo = GroupRepository()
    imported = 0
    for g in ldap_groups:
        existing = group_repo.get_by_sid(g["sid"])
        if not existing:
            imported += 1
        group_repo.upsert_ldap(
            sid=g["sid"],
            cn=g["cn"],
            dn=g["dn"],
            member_count=g["member_count"],
        )

    return AdminResult.ok(
        f"{imported} neue Gruppen importiert",
        data={
            "imported": imported,
            "total": len(ldap_groups),
            "message": f"{imported} neue Gruppen importiert",
            "name": "LDAP-Import",
        },
    )


@admin_function(
    name="groups.sync_members",
    category="groups",
    description="LDAP-Gruppenmitgliedschaften synchronisieren",
    help="Synchronisiert die Gruppenzugehoerigkeit aller aktiven LDAP-Benutzer "
    "mit dem LDAP-Verzeichnis. Betrifft nur Benutzer mit auth_source='ldap'.",
    examples=[
        "Synchronisiere LDAP-Mitgliedschaften",
        "Aktualisiere Gruppenzuordnungen aus dem LDAP",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Synchronisierungsstatistik",
    requires_confirmation=True,
    confirm_template="ldap mitgliedschaften synchronisieren",
    changelog_area="groups",
    changelog_action="update",
    raw_response=True,
    http=("POST", "/api/admin/groups/sync-members"),
)
async def groups_sync_members(user: dict) -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.web.config_utils import build_ldap_config
    from kiara.web.ldap_sync import sync_user_groups

    config_repo = ConfigRepository()
    row = config_repo.get("ldap_config")
    if not row:
        return AdminResult.fail("LDAP nicht konfiguriert")
    try:
        ldap_config = build_ldap_config(row)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Config fehlerhaft (Entschluesselung?): {exc}")

    stats = sync_user_groups(ldap_config)
    errors = stats.get("errors", 0) if isinstance(stats, dict) else 0
    error_msg = stats.get("error") if isinstance(stats, dict) else None
    if errors or error_msg:
        detail = error_msg if error_msg else f"{stats.get('synced', 0)} OK, {errors} Fehler"
        return AdminResult.fail(
            f"Fehler bei Synchronisierung: {detail}",
            data={"stats": stats, "name": "LDAP-Sync"},
        )
    return AdminResult.ok(
        "Mitgliedschaften synchronisiert",
        data={"message": "Mitgliedschaften synchronisiert", "stats": stats, "name": "LDAP-Sync"},
    )


@admin_function(
    name="groups.ldap_preview",
    category="groups",
    description="LDAP-Gruppen vorab anzeigen (ohne Import)",
    help="Zeigt alle Gruppen im LDAP-Verzeichnis an, ohne sie zu importieren. "
    "Bereits importierte Gruppen werden markiert.",
    examples=[
        "Welche Gruppen gibt es im LDAP?",
        "Zeige LDAP-Gruppen Vorschau",
    ],
    parameters={
        "type": "object",
        "properties": {
            "base_dn": {"type": "string", "description": "Alternativer LDAP Base-DN (optional)"},
        },
        "required": [],
    },
    returns="Liste aller LDAP-Gruppen mit Import-Status",
    raw_response=True,
    http=("GET", "/api/admin/groups/ldap"),
)
async def groups_ldap_preview(user: dict, base_dn: str = "") -> AdminResult:
    from kiara.repositories.config import ConfigRepository
    from kiara.repositories.group import GroupRepository
    from kiara.web.config_utils import build_ldap_config
    from kiara.web.ldap_utils import get_groups

    config_repo = ConfigRepository()
    row = config_repo.get("ldap_config")
    if not row:
        return AdminResult.fail("LDAP nicht konfiguriert")
    try:
        ldap_config = build_ldap_config(row)
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Config fehlerhaft (Entschluesselung?): {exc}")

    if base_dn:
        ldap_config["search_base"] = base_dn

    try:
        ldap_groups = get_groups(ldap_config) or []
    except Exception as exc:
        return AdminResult.fail(f"LDAP-Abfrage fehlgeschlagen: {exc}")
    group_repo = GroupRepository()
    imported_sids = set(group_repo.get_all_sids())

    for g in ldap_groups:
        g["imported"] = bool(g.get("sid")) and g["sid"] in imported_sids

    return AdminResult.ok(f"{len(ldap_groups)} LDAP-Gruppen", data={"groups": ldap_groups})
