"""Produktneutraler Session-Service.

Kapselt Token-Erzeugung, Hashing und Delegation an Repository-Protocols.
Keine Imports aus kiara.web, kiara.admin, kiara.shared, kiara_db.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

from kiara_auth.core import hash_token
from kiara_auth.repositories import SessionWriter, UserSessionResolver


class SessionService:
    """Verwaltet Sessions ueber injizierte Repositories.

    Args:
        session_writer: Protocol-Implementierung fuer Session-CRUD.
        session_max_age: Session-Lebensdauer in Sekunden.
        user_session_resolver: Optional - liest Session+User atomar.
        post_auth_hook: Optional - Callback nach erfolgreicher Validierung,
            erhaelt (resolver_result_dict) und kann es anreichern.
    """

    def __init__(
        self,
        session_writer: SessionWriter,
        *,
        session_max_age: int = 365 * 86400,
        user_session_resolver: UserSessionResolver | None = None,
        post_auth_hook: Any = None,
    ):
        self._writer = session_writer
        self._max_age = session_max_age
        self._resolver = user_session_resolver
        self._post_auth_hook = post_auth_hook

    def create(self, user_id: int, *, conn: Any = None) -> tuple[str, str]:
        """Erstellt eine neue Session und gibt (Session-Token, CSRF-Token) zurueck.

        Beide Tokens sind Klartext. Vor der DB-Speicherung werden beide mit
        SHA-256 gehasht; das Token wird als `token`-Spalte, der CSRF-Token-Hash
        als `csrf_token_hash`-Spalte abgelegt.
        """
        token = secrets.token_hex(32)
        token_hash = hash_token(token)
        csrf_plain = secrets.token_hex(32)
        csrf_hash = hash_token(csrf_plain)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=self._max_age)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self._writer.create(
            conn=conn,
            user_id=user_id,
            token=token_hash,
            expires_at=expires_at,
            csrf_token_hash=csrf_hash,
        )
        return token, csrf_plain

    def validate(self, token: str) -> dict | None:
        """Validiert ein Klartext-Token ueber den UserSessionResolver.

        Returns:
            User-Dict vom Resolver oder None bei ungueltiger Session.
        """
        if not self._resolver:
            raise NotImplementedError(
                "SessionService benoetigt einen user_session_resolver fuer validate()"
            )
        token_hash = hash_token(token)
        return self._resolver.resolve(token_hash)

    def delete(self, token: str, *, conn: Any = None) -> None:
        """Loescht eine Session anhand des Klartext-Tokens."""
        token_hash = hash_token(token)
        self._writer.delete_by_token(token_hash, conn=conn)

    def delete_user_sessions(
        self,
        user_id: int,
        *,
        except_token: str | None = None,
        conn: Any = None,
    ) -> Any:
        """Loescht alle Sessions eines Users, optional eine ausnehmen.

        except_token ist ein Klartext-Token, das intern gehasht wird.
        """
        hashed_except = hash_token(except_token) if except_token else None
        return self._writer.delete_by_user(
            user_id,
            except_token=hashed_except,
            conn=conn,
        )

    def cleanup_expired(self, *, conn: Any = None) -> Any:
        """Entfernt abgelaufene Sessions."""
        return self._writer.cleanup_expired(conn=conn)
