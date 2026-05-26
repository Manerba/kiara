"""API-Key-Validierung fuer den MCP-Server.

Eigenes Modul, getrennt von den Admin-CRUD-Endpunkten (api_mcp.py),
um zirkulaere Imports zwischen MCP-Server und Admin-App zu vermeiden.
"""

import hashlib
import logging
import re
from datetime import datetime, timezone

from kiara.repositories.api_key import ApiKeyRepository
from kiara.shared.db import db_query
from kiara.web.config_utils import load_user_permissions

logger = logging.getLogger(__name__)

api_key_repo = ApiKeyRepository()

_API_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{32,}$")


def validate_api_key(key: str) -> dict | None:
    """Validiert einen API-Key und gibt den zugehoerigen User zurueck.

    Liest Key+User-Daten (read-only), prueft Gueltigkeit,
    und aktualisiert last_used_at in einer separaten Transaktion.

    Returns:
        User-Dict (wie validate_session) oder None.
    """
    if not isinstance(key, str) or not _API_KEY_PATTERN.match(key):
        return None

    key_hash = hashlib.sha256(key.encode()).hexdigest()

    # --- Phase 1: Read-Only Lookup ---
    with db_query() as db:
        row = db.execute(
            """SELECT ak.id as key_id, ak.user_id, ak.is_active, ak.expires_at,
                      u.id as uid, u.username, u.vorname, u.nachname,
                      u.email, u.is_admin, u.is_active as user_active
               FROM api_keys ak
               JOIN users u ON ak.user_id = u.id
               WHERE ak.key_hash = %s""",
            (key_hash,),
        ).fetchone()

        if not row:
            return None

        # Key aktiv?
        if not row["is_active"]:
            return None

        # User aktiv? (Korrektur: Plan prueft nur Key-Status)
        if not row["user_active"]:
            return None

        # Ablaufdatum pruefen
        if row["expires_at"]:
            exp_val = row["expires_at"]
            expires = exp_val if isinstance(exp_val, datetime) else datetime.fromisoformat(exp_val)
            # Naive Timestamps (ohne Timezone-Info) als UTC interpretieren,
            # da NOW() als TEXT gespeichert kein tzinfo liefert.
            if expires.tzinfo is None:
                expires = expires.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) > expires:
                return None

        key_id = row["key_id"]
        user_id = row["uid"]

        # Gruppen + Datasource-IDs laden
        group_ids, datasource_ids = load_user_permissions(db, user_id)

    # --- Phase 2: last_used_at aktualisieren (eigene Transaktion) ---
    try:
        api_key_repo.touch_last_used(key_id)
    except Exception as exc:
        # Nicht-kritisch: last_used_at ist nur fuer die Admin-Anzeige.
        # Nicht-kritisch: bei DB-Fehler nicht den Request abbrechen.
        logger.debug("last_used_at Update fehlgeschlagen: %s", exc)

    return {
        "id": user_id,
        "username": row["username"],
        "vorname": row["vorname"],
        "nachname": row["nachname"],
        "email": row["email"],
        "is_admin": row["is_admin"],
        "group_ids": group_ids,
        "datasource_ids": datasource_ids,
        "api_key_id": key_id,
    }
