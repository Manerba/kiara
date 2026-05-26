"""
KIara - LDAP Synchronisierung
Synchronisiert AD-Gruppenmitglieder mit der lokalen users-Tabelle.
"""

import logging
import secrets
from datetime import datetime, timezone

from kiara.repositories.config import ConfigRepository
from kiara.repositories.group import GroupRepository
from kiara.repositories.permission import PermissionRepository
from kiara.repositories.user import UserRepository
from kiara.shared.db import db_transaction
from kiara.web.auth import hash_password
from kiara.web.config_utils import build_ldap_config
from kiara.web.ldap_utils import get_group_members

logger = logging.getLogger(__name__)

config_repo = ConfigRepository()
user_repo = UserRepository()
group_repo = GroupRepository()
perm_repo = PermissionRepository()


def sync_ldap_users() -> dict:
    """Hauptfunktion fuer LDAP-Sync:
    1. LDAP-Config aus DB laden
    2. Gruppenmitglieder abrufen
    3. Benutzer anlegen/aktualisieren/deaktivieren
    4. last_sync_at aktualisieren
    Liefert dict mit Statistiken: {created, updated, deactivated} oder {error}.
    """
    try:
        # 1. Config laden
        config_row = config_repo.get("ldap_config")
        if not config_row:
            return {"error": "Keine LDAP-Konfiguration vorhanden"}

        if not config_row.get("group_dn"):
            return {"error": "Keine AD-Gruppe ausgewaehlt"}

        config = build_ldap_config(config_row)

        # 2. Gruppenmitglieder abrufen
        try:
            members = get_group_members(config)
        except Exception as exc:
            return {"error": f"LDAP-Abfrage fehlgeschlagen: {exc}"}

        stats = {"created": 0, "updated": 0, "deactivated": 0}
        member_usernames = set()

        # 3. Fuer jeden AD-Benutzer
        with db_transaction() as db:
            for member in members:
                username = member["sAMAccountName"]
                if not username:
                    continue

                member_usernames.add(username.lower())

                existing = user_repo.get_by_username(username, conn=db)

                if existing:
                    if existing["auth_source"] == "ldap":
                        # Attribute aktualisieren, aktiv setzen
                        try:
                            db.execute("SAVEPOINT user_update")
                            user_repo.update(
                                existing["id"],
                                conn=db,
                                vorname=member["givenName"],
                                nachname=member["sn"],
                                email=member["mail"],
                                ldap_dn=member["dn"],
                                is_active=True,
                            )
                            db.execute("RELEASE SAVEPOINT user_update")
                            stats["updated"] += 1
                        except Exception as exc:
                            db.execute("ROLLBACK TO SAVEPOINT user_update")
                            logger.warning(
                                "LDAP-Sync: User '%s' Update fehlgeschlagen: %s", username, exc
                            )
                            stats.setdefault("errors", 0)
                            stats["errors"] += 1
                    # Lokale Benutzer (auth_source='local') werden NICHT angefasst
                else:
                    # Neuen LDAP-User anlegen
                    try:
                        db.execute("SAVEPOINT user_create")
                        user_repo.create(
                            conn=db,
                            username=username,
                            password_hash=hash_password(secrets.token_hex(32)),
                            vorname=member["givenName"],
                            nachname=member["sn"],
                            email=member["mail"],
                            auth_source="ldap",
                            ldap_dn=member["dn"],
                            is_active=True,
                            is_admin=False,
                        )
                        db.execute("RELEASE SAVEPOINT user_create")
                        stats["created"] += 1
                    except Exception as exc:
                        db.execute("ROLLBACK TO SAVEPOINT user_create")
                        logger.warning(
                            "LDAP-Sync: User '%s' Create fehlgeschlagen: %s", username, exc
                        )
                        stats.setdefault("errors", 0)
                        stats["errors"] += 1

            # 4. LDAP-Benutzer deaktivieren die nicht mehr in der Gruppe sind
            ldap_users = db.execute(
                "SELECT id, username FROM users WHERE auth_source = 'ldap' AND is_active = true"
            ).fetchall()

            # Safety-Threshold: Wenn >50% der LDAP-User deaktiviert wuerden,
            # ist wahrscheinlich die LDAP-Abfrage fehlgeschlagen (leeres Ergebnis).
            to_deactivate = [u for u in ldap_users if u["username"].lower() not in member_usernames]
            if ldap_users and len(to_deactivate) > len(ldap_users) * 0.5:
                logger.warning(
                    "LDAP-Sync abgebrochen: %d von %d Usern wuerden deaktiviert (>50%%). "
                    "Vermutlich LDAP-Abfragefehler.",
                    len(to_deactivate),
                    len(ldap_users),
                )
                stats["error"] = (
                    f"Abbruch: {len(to_deactivate)}/{len(ldap_users)} User betroffen (>50%)"
                )
            else:
                for ldap_user in to_deactivate:
                    user_repo.update(ldap_user["id"], conn=db, is_active=False)
                    stats["deactivated"] += 1

            # 5. last_sync_at nur bei erfolgreichem Sync aktualisieren
            if not stats.get("error"):
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                db.execute("UPDATE ldap_config SET last_sync_at = %s WHERE id = 1", (now,))

        logger.info("LDAP sync: %s", stats)
        return stats

    except Exception as exc:
        logger.error("LDAP sync error: %s", exc)
        return {"error": str(exc)}


def sync_user_groups(ldap_config: dict) -> dict:
    """Sync LDAP group memberships for all active LDAP users.
    Used by the groups API for permission-based datasource filtering.
    """
    from kiara.web.ldap_utils import get_user_groups

    synced = 0
    errors = 0
    users = []
    _threshold_error = ""

    try:
        with db_transaction() as db:
            users = db.execute(
                "SELECT id, username, ldap_dn FROM users WHERE auth_source = 'ldap' AND is_active = true AND ldap_dn != ''"
            ).fetchall()

            group_rows = db.execute(
                "SELECT id, sid FROM groups WHERE sid IS NOT NULL AND sid != ''"
            ).fetchall()
            group_sid_to_id = {r["sid"]: r["id"] for r in group_rows}

            for user in users:
                try:
                    db.execute("SAVEPOINT group_sync")
                    user_group_sids = get_user_groups(ldap_config, user["ldap_dn"])
                    perm_repo.set_user_groups(
                        user["id"],
                        [group_sid_to_id[sid] for sid in user_group_sids if sid in group_sid_to_id],
                        conn=db,
                    )
                    db.execute("RELEASE SAVEPOINT group_sync")
                    synced += 1
                except Exception as exc:
                    db.execute("ROLLBACK TO SAVEPOINT group_sync")
                    logger.error("Group sync failed for user %s: %s", user["username"], exc)
                    errors += 1

            # Safety-Threshold VOR dem Commit pruefen — bei >50% Fehlern Rollback
            if users and errors > len(users) * 0.5:
                _threshold_error = f"Abbruch: {errors}/{len(users)} User fehlgeschlagen (>50%)"
                raise RuntimeError(_threshold_error)
    except RuntimeError:
        if _threshold_error:
            logger.warning(
                "Group-Sync abgebrochen: %d von %d Usern fehlgeschlagen (>50%%). "
                "Vermutlich LDAP-Verbindungsfehler. Rollback.",
                errors,
                len(users),
            )
            return {"error": _threshold_error}
        raise

    stats = {"synced": synced, "errors": errors}
    logger.info("User group sync: %s", stats)
    return stats
