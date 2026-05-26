"""Test-Helpers fuer kiara_auth.

Fake-Repositories und Factories fuer Unit-Tests.
Keine Abhaengigkeit auf Produktcode.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kiara_auth.core import AuthenticatedPrincipal

# ---------------------------------------------------------------------------
# Fake-Repositories
# ---------------------------------------------------------------------------


class FakeSessionWriter:
    """In-Memory Session-Store fuer Tests."""

    def __init__(self) -> None:
        self.sessions: dict[str, dict] = {}
        self._next_id = 1

    def create(
        self,
        *,
        conn: Any = None,
        user_id: int,
        token: str,
        expires_at: str,
        csrf_token_hash: str | None = None,
    ) -> int:
        sid = self._next_id
        self._next_id += 1
        self.sessions[token] = {
            "id": sid,
            "user_id": user_id,
            "token": token,
            "expires_at": expires_at,
            "csrf_token_hash": csrf_token_hash,
        }
        return sid

    def delete_by_token(self, token: str, *, conn: Any = None) -> bool:
        return self.sessions.pop(token, None) is not None

    def delete_by_user(
        self, user_id: int, *, except_token: str | None = None, conn: Any = None
    ) -> int:
        to_delete = [
            t for t, s in self.sessions.items() if s["user_id"] == user_id and t != except_token
        ]
        for t in to_delete:
            del self.sessions[t]
        return len(to_delete)

    def cleanup_expired(self, *, conn: Any = None) -> int:
        now = datetime.now(timezone.utc)
        to_delete = []
        for token, session in self.sessions.items():
            expires = datetime.strptime(session["expires_at"], "%Y-%m-%d %H:%M:%S")
            expires = expires.replace(tzinfo=timezone.utc)
            if now > expires:
                to_delete.append(token)
        for t in to_delete:
            del self.sessions[t]
        return len(to_delete)


class FakeUserSessionResolver:
    """Fake-Resolver der feste User-Daten zurueckgibt.

    Args:
        users: Dict von user_id -> User-Dict.
        sessions: Referenz auf FakeSessionWriter.sessions
            (damit resolve() pruefen kann ob die Session existiert).
    """

    def __init__(self, users: dict[int, dict], sessions: dict[str, dict]) -> None:
        self._users = users
        self._sessions = sessions

    def resolve(self, token_hash: str) -> dict | None:
        session = self._sessions.get(token_hash)
        if not session:
            return None
        user = self._users.get(session["user_id"])
        if not user:
            return None
        if not user.get("is_active", True):
            return None
        return dict(user)


# ---------------------------------------------------------------------------
# Factory-Helpers
# ---------------------------------------------------------------------------


def make_test_principal(**kwargs: Any) -> AuthenticatedPrincipal:
    """Erzeugt einen AuthenticatedPrincipal mit sinnvollen Defaults.

    Beispiel:
        p = make_test_principal(username="alice", roles=["admin"])
    """
    defaults = {
        "id": 1,
        "username": "testuser",
        "display_name": "Test User",
        "email": "test@example.com",
        "roles": [],
        "scopes": [],
        "is_active": True,
        "metadata": {},
    }
    defaults.update(kwargs)
    return AuthenticatedPrincipal(**defaults)
