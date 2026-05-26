"""Repository-Protocols fuer kiara_auth.

Definiert die Schnittstellen, die Produkte (KIara, LLM-Wiki)
mit ihren eigenen DB-Implementierungen erfuellen.
Keine Abhaengigkeit auf kiara.shared.db oder kiara_db.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SessionWriter(Protocol):
    """Schreibt und loescht Sessions in der Persistenzschicht."""

    def create(
        self,
        *,
        conn: Any = None,
        user_id: int,
        token: str,
        expires_at: str,
        csrf_token_hash: str | None = None,
    ) -> Any:
        """Erzeugt eine neue Session-Zeile.

        `csrf_token_hash` ist optional (NULL-Default), damit andere
        Product-Implementations nicht brechen, die die Spalte noch nicht
        kennen; im KIara-Produktionspfad wird der Hash immer gesetzt.
        """
        ...

    def delete_by_token(self, token: str, *, conn: Any = None) -> Any:
        """Loescht eine Session anhand des Token-Hashes."""
        ...

    def delete_by_user(
        self, user_id: int, *, except_token: str | None = None, conn: Any = None
    ) -> Any:
        """Loescht alle Sessions eines Users (optional eine ausnehmen)."""
        ...

    def cleanup_expired(self, *, conn: Any = None) -> Any:
        """Entfernt abgelaufene Sessions."""
        ...


@runtime_checkable
class UserSessionResolver(Protocol):
    """Liest Session+User-Daten atomar (fuer validate_session).

    Produkte implementieren dies mit ihrer eigenen DB-Logik,
    z.B. JOIN auf sessions+users inkl. Aktivstatus und Ablaufpruefung.
    """

    def resolve(self, token_hash: str) -> dict | None:
        """Gibt User-Dict zurueck wenn Session gueltig, sonst None.

        Das zurueckgegebene Dict muss mindestens enthalten:
        id, username. Weitere Felder sind produktspezifisch.
        """
        ...


@runtime_checkable
class TokenRepository(Protocol):
    """Liest und verwaltet API-/Bearer-Tokens."""

    def get_by_hash(self, key_hash: str, *, conn: Any = None) -> dict | None:
        """Findet einen Token-Eintrag anhand seines SHA-256-Hashes."""
        ...

    def touch_last_used(self, key_id: int, *, conn: Any = None) -> None:
        """Aktualisiert den last_used_at-Zeitstempel."""
        ...
