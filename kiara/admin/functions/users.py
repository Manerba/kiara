"""Admin-Funktionen fuer Benutzerverwaltung (Sprint 06)."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="users.list",
    category="users",
    description="Alle Benutzer auflisten",
    help="Listet alle lokalen und LDAP-Benutzer mit Status, Rolle und Auth-Quelle. "
    "Passwort-Hashes werden nicht zurueckgegeben (list_safe).",
    examples=[
        "Zeige alle Benutzer",
        "Wer sind die Admins?",
        "Welche Benutzer sind gesperrt?",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Liste aller Benutzer mit Metadaten",
    raw_response=True,
    http=("GET", "/api/admin/users"),
)
async def users_list(user: dict) -> AdminResult:
    from kiara.repositories.user import UserRepository

    user_repo = UserRepository()
    rows = user_repo.list_safe()
    data = [
        {
            "id": r["id"],
            "username": r["username"],
            "vorname": r["vorname"],
            "nachname": r["nachname"],
            "email": r["email"],
            "is_admin": bool(r["is_admin"]),
            "auth_source": r["auth_source"],
            "ldap_dn": r["ldap_dn"],
            "is_active": bool(r["is_active"]),
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    return AdminResult.ok(f"{len(data)} Benutzer", data={"data": data})


@admin_function(
    name="users.create",
    category="users",
    description="Neuen lokalen Benutzer erstellen",
    help="Erstellt einen neuen lokalen Benutzer. LDAP-Benutzer werden ueber "
    "die LDAP-Integration importiert, nicht hier erstellt. "
    "Benutzername mind. 2 Zeichen. Die Passwort-Komplexitaetspruefung ist "
    "in der Entwicklungsumgebung deaktiviert.",
    examples=[
        "Erstelle Benutzer testuser mit Passwort abc12345",
        "Neuen Admin-Account anlegen: admin2, Passwort geheim123",
    ],
    parameters={
        "type": "object",
        "properties": {
            "username": {"type": "string", "description": "Benutzername (mind. 2 Zeichen)"},
            "password": {"type": "string", "description": "Passwort"},
            "vorname": {"type": "string", "description": "Vorname"},
            "nachname": {"type": "string", "description": "Nachname"},
            "email": {"type": "string", "description": "E-Mail-Adresse"},
            "is_admin": {"type": "boolean", "description": "Admin-Rolle (Default: false)"},
        },
        "required": ["username", "password"],
    },
    returns="ID und Bestaetigungsmeldung",
    pk_param="username",
    changelog_area="users",
    changelog_action="create",
    raw_response=True,
    http=("POST", "/api/admin/users"),
)
async def users_create(
    user: dict,
    username: str,
    password: str,
    vorname: str = "",
    nachname: str = "",
    email: str = "",
    is_admin: bool = False,
) -> AdminResult:
    from kiara.repositories.user import UserRepository
    from kiara.web.auth import hash_password

    if not username or len(username) < 2:
        return AdminResult.fail("Benutzername muss mindestens 2 Zeichen haben")

    user_repo = UserRepository()
    if user_repo.username_exists(username):
        return AdminResult.fail(f"Benutzername '{username}' existiert bereits")

    # Dev-Umgebung: Passwort-Komplexitaetspruefung deaktiviert (User-Entscheidung 2026-07-11).
    # from kiara.web.auth import validate_password
    #
    # pw_err = validate_password(password)
    # if pw_err:
    #     return AdminResult.fail(pw_err)

    pw_hash = hash_password(password)
    user_id = user_repo.create(
        username=username,
        password_hash=pw_hash,
        vorname=vorname,
        nachname=nachname,
        email=email,
        is_admin=bool(is_admin),
    )
    return AdminResult.ok(
        f"Benutzer '{username}' erstellt (ID {user_id})",
        data={"id": user_id, "message": "Benutzer erstellt", "name": username},
    )


@admin_function(
    name="users.update",
    category="users",
    description="Benutzer aktualisieren",
    help="Aktualisiert Benutzerdaten. Nur uebergebene Felder werden geaendert "
    "(Partial Update). Bei Deaktivierung (is_active=false) werden ALLE "
    "aktiven Sessions des Benutzers sofort invalidiert. "
    "Selbst-Deaktivierung ist nicht erlaubt.",
    examples=[
        "Mache Benutzer 5 zum Admin",
        "Deaktiviere Benutzer 3",
        "Aendere die E-Mail von Benutzer 7",
    ],
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Benutzer-ID"},
            "vorname": {"type": "string", "description": "Vorname"},
            "nachname": {"type": "string", "description": "Nachname"},
            "email": {"type": "string", "description": "E-Mail-Adresse"},
            "is_admin": {"type": "boolean", "description": "Admin-Rolle"},
            "is_active": {"type": "boolean", "description": "Aktiv-Status"},
            "password": {"type": "string", "description": "Neues Passwort"},
        },
        "required": ["user_id"],
    },
    returns="Bestaetigungsmeldung",
    pk_param="user_id",
    changelog_area="users",
    changelog_action="update",
    raw_response=True,
    http=("PUT", "/api/admin/users/{user_id}"),
)
async def users_update(
    user: dict,
    user_id: int,
    vorname: str | None = None,
    nachname: str | None = None,
    email: str | None = None,
    is_admin: bool | None = None,
    is_active: bool | None = None,
    password: str | None = None,
) -> AdminResult:
    from kiara.repositories.user import SessionRepository, UserRepository
    from kiara.web.auth import hash_password

    user_repo = UserRepository()
    if not user_repo.exists(user_id):
        return AdminResult.fail("Benutzer nicht gefunden")

    fields = {}
    if vorname is not None:
        fields["vorname"] = vorname
    if nachname is not None:
        fields["nachname"] = nachname
    if email is not None:
        fields["email"] = email
    if is_admin is not None:
        fields["is_admin"] = bool(is_admin)
    if is_active is not None:
        fields["is_active"] = bool(is_active)
    if password:
        # Dev-Umgebung: Passwort-Komplexitaetspruefung deaktiviert (User-Entscheidung 2026-07-11).
        # from kiara.web.auth import validate_password
        #
        # pw_err = validate_password(password)
        # if pw_err:
        #     return AdminResult.fail(pw_err)
        fields["password_hash"] = hash_password(password)

    if fields:
        needs_last_admin_guard = (
            is_admin is not None and not is_admin and user_id == user.get("id")
        ) or (is_active is not None and not is_active)
        needs_session_invalidation = (is_active is not None and not is_active) or bool(password)

        if needs_last_admin_guard or needs_session_invalidation:
            from kiara.shared.db import db_transaction

            session_repo = SessionRepository()
            with db_transaction() as conn:
                conn.execute("LOCK TABLE users IN SHARE ROW EXCLUSIVE MODE")

                # Selbst-Deaktivierung verhindern
                if is_active is not None and not is_active and user_id == user.get("id"):
                    return AdminResult.fail("Eigenen Account nicht deaktivieren")

                if is_admin is not None and not is_admin and user_id == user.get("id"):
                    admin_count = user_repo.count({"is_admin": True, "is_active": True}, conn=conn)
                    if admin_count <= 1:
                        return AdminResult.fail("Letzter Admin kann sich nicht selbst degradieren")

                if is_active is not None and not is_active:
                    target = user_repo.get_by_id(user_id, conn=conn)
                    if target and target.get("is_admin") and target.get("is_active"):
                        admin_count = user_repo.count(
                            {"is_admin": True, "is_active": True}, conn=conn
                        )
                        if admin_count <= 1:
                            return AdminResult.fail(
                                "Letzter aktiver Admin kann nicht deaktiviert werden"
                            )

                user_repo.update(user_id, conn=conn, **fields)
                if needs_session_invalidation:
                    session_repo.delete_by_user(user_id, conn=conn)
        else:
            user_repo.update(user_id, **fields)

    return AdminResult.ok(
        f"Benutzer {user_id} aktualisiert",
        data={"message": "Benutzer aktualisiert", "name": f"user_{user_id}"},
    )


@admin_function(
    name="users.delete",
    category="users",
    description="Benutzer loeschen",
    help="Loescht einen Benutzer und alle zugehoerigen Sessions (Cascade). "
    "Der aktuelle Admin kann sich nicht selbst loeschen.",
    examples=[
        "Loesche Benutzer 5",
        "Entferne den Account von testuser",
    ],
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Benutzer-ID"},
        },
        "required": ["user_id"],
    },
    returns="Loeschbestaeigung",
    requires_confirmation=True,
    confirm_template="benutzer {user_id} löschen",
    pk_param="user_id",
    changelog_area="users",
    changelog_action="delete",
    raw_response=True,
    http=("DELETE", "/api/admin/users/{user_id}"),
)
async def users_delete(user: dict, user_id: int) -> AdminResult:
    from kiara.repositories.user import UserRepository

    if not user or not user.get("id"):
        return AdminResult.fail("Authentifizierung erforderlich")
    if user_id == user.get("id"):
        return AdminResult.fail("Eigenen Account nicht loeschen")

    user_repo = UserRepository()
    if not user_repo.exists(user_id):
        return AdminResult.fail("Benutzer nicht gefunden")

    # Letzten aktiven Admin schuetzen: Cross-Admin-Loeschung darf das System nicht aussperren
    target = user_repo.get_by_id(user_id)
    if target and target.get("is_admin") and target.get("is_active"):
        admin_count = user_repo.count({"is_admin": True, "is_active": True})
        if admin_count <= 1:
            return AdminResult.fail("Letzter aktiver Admin kann nicht geloescht werden")

    user_repo.delete_with_sessions(user_id)
    return AdminResult.ok(
        f"Benutzer {user_id} geloescht",
        data={"message": "Benutzer geloescht", "name": f"user_{user_id}"},
    )


@admin_function(
    name="users.reset_password",
    category="users",
    description="Benutzer-Passwort zuruecksetzen (Admin-Override)",
    help="Setzt das Passwort eines Benutzers ohne das alte Passwort zu kennen. "
    "Dies ist ein Admin-Override — der regulaere PUT /api/auth/password "
    "Endpunkt erfordert old_password (Self-Service). "
    "Neues Passwort mind. 8 Zeichen.",
    examples=[
        "Setze das Passwort von Benutzer 5 auf Neu12345",
        "Passwort-Reset fuer testuser",
    ],
    parameters={
        "type": "object",
        "properties": {
            "user_id": {"type": "integer", "description": "Benutzer-ID"},
            "new_password": {"type": "string", "description": "Neues Passwort (mind. 8 Zeichen)"},
        },
        "required": ["user_id", "new_password"],
    },
    returns="Bestaetigungsmeldung",
    requires_confirmation=True,
    confirm_template="passwort für benutzer {user_id} zurücksetzen",
    pk_param="user_id",
    changelog_area="users",
    changelog_action="update",
)
async def users_reset_password(user: dict, user_id: int, new_password: str) -> AdminResult:
    from kiara.repositories.user import UserRepository
    from kiara.web.auth import hash_password, validate_password

    pw_err = validate_password(new_password)
    if pw_err:
        return AdminResult.fail(pw_err)

    user_repo = UserRepository()
    if not user_repo.exists(user_id):
        return AdminResult.fail("Benutzer nicht gefunden")

    pw_hash = hash_password(new_password)

    # Passwort-Aenderung + Session-Invalidierung atomar ausfuehren
    from kiara.repositories.user import SessionRepository
    from kiara.shared.db import db_transaction

    session_repo = SessionRepository()
    with db_transaction() as conn:
        user_repo.set_password(user_id, pw_hash, conn=conn)
        session_repo.delete_by_user(user_id, conn=conn)

    return AdminResult.ok(
        f"Passwort fuer Benutzer {user_id} zurueckgesetzt",
        data={"user_id": user_id, "name": f"user_{user_id}"},
    )
