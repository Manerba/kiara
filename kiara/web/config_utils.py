"""Zentrale Hilfsfunktionen fuer Config-Operationen.

Beinhaltet:
- get_masked_config / resolve_masked_field: Singleton-Config lesen/schreiben mit Passwort-Maskierung
- safe_json_loads: JSON-Parsing mit Fallback
- load_user_permissions: User-Berechtigungen (Gruppen + Datasources) laden
- build_ldap_config: LDAP-Verbindungs-Config aus DB-Row aufbauen
"""

import json
import logging
from typing import Any

from fastapi import HTTPException

from kiara.repositories.config import ConfigRepository
from kiara.repositories.permission import PermissionRepository
from kiara.shared.constants import PASSWORD_MASK
from kiara.shared.crypto import decrypt, encrypt, is_encrypted
from kiara.shared.db import (
    ALLOWED_COLUMNS,
    ALLOWED_TABLES,
    validate_identifier,
)

logger = logging.getLogger(__name__)

_config_repo = ConfigRepository()
_perm_repo = PermissionRepository()


def get_masked_config(table: str, mask_fields: list[str] | None = None) -> dict | None:
    """Singleton-Config (id=1) aus einer Tabelle lesen, Passwort-Felder maskieren."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    if mask_fields:
        for field in mask_fields:
            validate_identifier(field, ALLOWED_COLUMNS, "column")

    config = _config_repo.get(table)
    if not config:
        return None
    if mask_fields:
        for field in mask_fields:
            if config.get(field):
                config[field] = PASSWORD_MASK
    return config


def resolve_masked_field(db: Any, table: str, field: str, new_value: str) -> str:
    """Loest ein maskiertes Feld auf: Neuen Wert verschluesseln oder bestehenden beibehalten."""
    validate_identifier(table, ALLOWED_TABLES, "table")
    validate_identifier(field, ALLOWED_COLUMNS, "column")

    if new_value != PASSWORD_MASK:
        return encrypt(new_value)

    row = db.execute(f"SELECT {field} FROM {table} WHERE id = 1").fetchone()
    if row and row[field]:
        return row[field]
    raise HTTPException(status_code=400, detail="Kein gespeichertes Passwort vorhanden")


def safe_json_loads(value: str, default: Any = None) -> Any:
    """JSON-Parsing mit Fallback bei Fehlern."""
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def load_user_permissions(db: Any, user_id: int) -> tuple[list[int], list[int]]:
    """Lade group_ids und datasource_ids fuer einen User.

    Datasource-IDs sind auf is_active=true gefiltert: deaktivierte
    Datenquellen erscheinen nicht in der User-ACL und sind damit fuer
    Retrieval, MCP und Tool-Filter unsichtbar.

    Returns:
        Tuple von (group_ids, datasource_ids).
    """
    group_rows = db.execute(
        "SELECT group_id FROM user_groups WHERE user_id = %s", (user_id,)
    ).fetchall()
    group_ids = [r["group_id"] for r in group_rows]

    datasource_ids = _perm_repo.get_user_datasource_ids(user_id, conn=db)

    return group_ids, datasource_ids


def build_ldap_config(row: Any) -> dict:
    """Erstelle LDAP-Verbindungs-Config aus DB-Row mit Passwort-Entschluesselung.

    Args:
        row: DB-Row aus ldap_config (dict).

    Returns:
        Dict mit server, port, use_ssl, bind_dn, bind_password, search_base, group_dn, group_name.
    """
    password = row["bind_password"]
    if password and is_encrypted(password):
        password = decrypt(password)

    return {
        "server": row["server_url"],
        "port": row["port"] or 389,
        "use_ssl": row.get("use_ssl", 0),
        "bind_dn": row["bind_dn"],
        "bind_password": password,
        "search_base": row["search_base"],
        "group_dn": row.get("group_dn"),
        "group_name": row.get("group_name"),
    }
