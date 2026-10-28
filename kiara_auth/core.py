"""Produktneutraler Auth-Core.

Passwort-Policy, bcrypt-Hashing, Token-Hashing, AuthenticatedPrincipal.
Keine Imports aus kiara.web, kiara.admin, kiara.shared, kiara_db.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

import bcrypt

# ---------------------------------------------------------------------------
# Passwort-Policy
# ---------------------------------------------------------------------------

_PW_MIN_LENGTH = 8
_PW_MAX_BYTES = 72  # bcrypt schneidet nach 72 Bytes ab
_PW_CATEGORIES = [
    re.compile(r"[A-Z]"),  # Grossbuchstabe
    re.compile(r"[a-z]"),  # Kleinbuchstabe
    re.compile(r"\d"),  # Zahl
    re.compile(r"[^A-Za-z0-9]"),  # Sonderzeichen
]
_PW_MIN_CATEGORIES = 3


def validate_password(password: str) -> str | None:
    """Prueft Passwort-Policy: min 8 Zeichen, max 72 Bytes, 3 von 4 Kategorien.

    Returns:
        None wenn OK, sonst Fehlermeldung.
    """
    if not password or len(password) < _PW_MIN_LENGTH:
        return f"Passwort muss mindestens {_PW_MIN_LENGTH} Zeichen haben"
    if len(password.encode("utf-8")) > _PW_MAX_BYTES:
        return f"Passwort darf maximal {_PW_MAX_BYTES} Bytes haben (bcrypt-Grenze)"
    met = sum(1 for cat in _PW_CATEGORIES if cat.search(password))
    if met < _PW_MIN_CATEGORIES:
        return (
            "Passwort muss mindestens 3 der 4 Kategorien enthalten: "
            "Grossbuchstabe, Kleinbuchstabe, Zahl, Sonderzeichen"
        )
    return None


# ---------------------------------------------------------------------------
# Passwort-Hashing (bcrypt)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hasht ein Passwort mit bcrypt (automatisches Salt)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    """Prueft ein Passwort gegen einen bcrypt-Hash."""
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Token-Hashing (SHA-256)
# ---------------------------------------------------------------------------


def hash_token(token: str) -> str:
    """Erzeugt einen SHA-256-Hex-Hash eines Klartext-Tokens.

    Wird fuer Session-Tokens und API-Keys verwendet, damit
    nur Hashes in der DB gespeichert werden.
    """
    return hashlib.sha256(token.encode()).hexdigest()


# ---------------------------------------------------------------------------
# AuthenticatedPrincipal
# ---------------------------------------------------------------------------


@dataclass
class AuthenticatedPrincipal:
    """Produktneutraler authentifizierter Benutzer.

    Repraesentiert die Identitaet nach erfolgreicher Authentifizierung.
    Produktspezifische Rechte (KIara-Datasources, Wiki-Spaces) gehoeren
    nicht hierhin, sondern werden vom jeweiligen Produkt ergaenzt.
    """

    id: int
    username: str
    display_name: str = ""
    email: str = ""
    roles: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    is_active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


def has_role(principal: AuthenticatedPrincipal, role: str) -> bool:
    """Prueft ob ein Principal eine bestimmte Rolle hat."""
    return role in principal.roles


def has_scope(principal: AuthenticatedPrincipal, scope: str) -> bool:
    """Prueft ob ein Principal einen bestimmten Scope hat."""
    return scope in principal.scopes
