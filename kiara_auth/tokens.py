"""Produktneutrale Token-Primitives fuer API-Keys und Bearer-Tokens.

Keine Imports aus kiara.web, kiara.admin, kiara.shared, kiara_db.
"""

from __future__ import annotations

import secrets

from kiara_auth.core import hash_token


def generate_api_key(
    *,
    prefix: str = "orag_",
    entropy_bytes: int = 24,
) -> tuple[str, str]:
    """Generiert einen API-Key mit Prefix und SHA-256-Hash.

    Args:
        prefix: Key-Prefix (z.B. "orag_").
        entropy_bytes: Anzahl Zufallsbytes (24 = 48 Hex-Zeichen = 192 Bit).

    Returns:
        (full_key, key_hash) — der Klartext-Key und sein SHA-256-Hash.
        Der Klartext darf nur einmal angezeigt werden.
    """
    raw = secrets.token_hex(entropy_bytes)
    full_key = f"{prefix}{raw}"
    key_hash = hash_token(full_key)
    return full_key, key_hash


def make_key_prefix(key: str, length: int) -> str:
    """Erzeugt einen Anzeige-Prefix fuer einen API-Key.

    Args:
        key: Vollstaendiger Klartext-Key.
        length: Anzahl sichtbarer Zeichen vor "...".

    Returns:
        z.B. "orag_abc123..." bei length=12.
    """
    return key[:length] + "..."
