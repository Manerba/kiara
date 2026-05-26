"""Optionaler FastAPI-Adapter fuer Cookie- und Bearer-Token-Auth.

Keine Imports aus kiara.web, kiara.admin, kiara.shared, kiara_db.
Abhaengigkeit auf fastapi und itsdangerous ist optional —
nur importieren, wenn FastAPI tatsaechlich genutzt wird.
"""

from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from itsdangerous import BadSignature, URLSafeSerializer


class CookieAuthAdapter:
    """Kapselt Cookie-Signing, Cookie-Handling und Token-Extraktion.

    Args:
        secret_key_provider: Callable das den Secret-Key liefert.
            Wird lazy aufgerufen, damit der Key bei Bedarf rotiert werden kann.
        cookie_name: Initialer Cookie-Name (dynamisch aenderbar).
        session_max_age: Cookie-Max-Age in Sekunden.
        session_salt: Salt fuer den URLSafeSerializer.
        cookie_secure: Ob das Secure-Flag gesetzt wird (default False fuer
            HTTP im internen Netz).
        cookie_samesite: SameSite-Policy (default "lax").
    """

    def __init__(
        self,
        secret_key_provider: Callable[[], str],
        *,
        cookie_name: str = "kiara_session",
        session_max_age: int = 365 * 86400,
        session_salt: str = "kiara-session",
        cookie_secure: bool = False,
        cookie_samesite: str = "lax",
    ):
        self._secret_key_provider = secret_key_provider
        self._cookie_name = cookie_name
        self._max_age = session_max_age
        self._salt = session_salt
        self._secure = cookie_secure
        self._samesite = cookie_samesite

    @property
    def cookie_name(self) -> str:
        """Aktueller Cookie-Name (dynamisch lesbar)."""
        return self._cookie_name

    @cookie_name.setter
    def cookie_name(self, name: str) -> None:
        """Setzt den Cookie-Namen dynamisch."""
        self._cookie_name = name

    def _get_serializer(self) -> URLSafeSerializer:
        """Erstellt einen URLSafeSerializer mit aktuellem Secret-Key."""
        return URLSafeSerializer(self._secret_key_provider(), salt=self._salt)

    def sign_token(self, token: str) -> str:
        """Signiert ein Token mit HMAC fuer Cookie-Sicherheit."""
        return self._get_serializer().dumps(token)

    def unsign_token(self, signed: str) -> str | None:
        """Verifiziert und entpackt ein signiertes Token.

        Returns:
            Klartext-Token oder None bei ungueltiger Signatur.
        """
        try:
            return self._get_serializer().loads(signed)
        except BadSignature:
            return None

    def set_session_cookie(self, response: Response, token: str) -> None:
        """Setzt das Session-Cookie mit HMAC-signiertem Token."""
        signed = self.sign_token(token)
        response.set_cookie(
            key=self._cookie_name,
            value=signed,
            max_age=self._max_age,
            httponly=True,
            secure=self._secure,
            samesite=self._samesite,
            path="/",
        )

    def delete_session_cookie(self, response: Response) -> None:
        """Loescht das Session-Cookie."""
        response.delete_cookie(key=self._cookie_name, path="/")

    def extract_token_from_request(self, request: Request) -> str | None:
        """Liest und verifiziert das Session-Token aus dem Request-Cookie.

        Returns:
            Klartext-Token oder None.
        """
        signed = request.cookies.get(self._cookie_name)
        if not signed:
            return None
        return self.unsign_token(signed)
