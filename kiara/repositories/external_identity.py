"""Repository fuer lokale Zuordnung externer Identitaeten zu KIara-Usern."""

from __future__ import annotations

from typing import Any

from kiara.shared.ad_sid import normalize_ad_sid
from kiara.shared.db import db_query, db_transaction

OPENDESK_PROVIDER = "opendesk-keycloak"
OPENDESK_IDENTITY_CLAIM = "ad_sid"


class ExternalIdentityRepository:
    """CRUD-Helfer fuer ``user_external_identities``."""

    def get_mapping_with_user(
        self,
        *,
        provider: str,
        issuer: str,
        identity_claim: str,
        external_id: str,
        conn=None,
    ) -> dict[str, Any] | None:
        normalized_id = normalize_ad_sid(external_id)
        query = (
            "SELECT "
            "uei.id AS mapping_id, uei.provider, uei.issuer, "
            "uei.identity_claim, uei.external_id, uei.user_id, "
            "uei.is_active AS mapping_active, "
            "u.id, u.username, u.email, u.is_admin, u.is_active AS user_active "
            "FROM user_external_identities uei "
            "JOIN users u ON u.id = uei.user_id "
            "WHERE uei.provider = %s "
            "AND uei.issuer = %s "
            "AND uei.identity_claim = %s "
            "AND uei.external_id = %s "
            "LIMIT 1"
        )
        params = (provider, issuer, identity_claim, normalized_id)
        if conn is not None:
            row = conn.execute(query, params).fetchone()
            return dict(row) if row else None
        with db_query() as db:
            row = db.execute(query, params).fetchone()
            return dict(row) if row else None

    def upsert_identity(
        self,
        *,
        provider: str,
        issuer: str,
        identity_claim: str,
        external_id: str,
        user_id: int,
        is_active: bool = True,
        conn=None,
    ) -> dict[str, Any]:
        """Transaktionaler Importpfad fuer ein AD-SID-Mapping.

        Der Ziel-User muss existieren und aktiv sein. ``external_id`` wird vor
        dem Schreiben auf den kanonischen SID-String normalisiert.
        """
        normalized_id = normalize_ad_sid(external_id)
        if conn is not None:
            return self._upsert_identity_conn(
                conn,
                provider=provider,
                issuer=issuer,
                identity_claim=identity_claim,
                external_id=normalized_id,
                user_id=user_id,
                is_active=is_active,
            )
        with db_transaction() as db:
            return self._upsert_identity_conn(
                db,
                provider=provider,
                issuer=issuer,
                identity_claim=identity_claim,
                external_id=normalized_id,
                user_id=user_id,
                is_active=is_active,
            )

    def _upsert_identity_conn(
        self,
        conn,
        *,
        provider: str,
        issuer: str,
        identity_claim: str,
        external_id: str,
        user_id: int,
        is_active: bool,
    ) -> dict[str, Any]:
        user = conn.execute(
            "SELECT id, is_active FROM users WHERE id = %s",
            (user_id,),
        ).fetchone()
        if not user:
            raise ValueError(f"User {user_id} nicht gefunden")
        if user["is_active"] is not True:
            raise ValueError(f"User {user_id} ist nicht aktiv")
        row = conn.execute(
            """
            INSERT INTO user_external_identities
                (provider, issuer, identity_claim, external_id, user_id, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (provider, issuer, identity_claim, external_id)
            DO UPDATE SET
                is_active = EXCLUDED.is_active,
                updated_at = now()
            WHERE user_external_identities.user_id = EXCLUDED.user_id
            RETURNING *
            """,
            (provider, issuer, identity_claim, external_id, user_id, is_active),
        ).fetchone()
        if not row:
            raise ValueError("Externe Identitaet ist bereits einem anderen KIara-User zugeordnet")
        return dict(row)

    def duplicate_report(self, *, conn=None) -> dict[str, Any]:
        """Liefert einfache Diagnose fuer doppelte externe IDs oder aktive User-Mappings."""
        query = """
            SELECT provider, issuer, identity_claim, external_id, COUNT(*) AS count
            FROM user_external_identities
            GROUP BY provider, issuer, identity_claim, external_id
            HAVING COUNT(*) > 1
        """
        active_query = """
            SELECT provider, issuer, identity_claim, user_id, COUNT(*) AS count
            FROM user_external_identities
            WHERE is_active
            GROUP BY provider, issuer, identity_claim, user_id
            HAVING COUNT(*) > 1
        """
        if conn is not None:
            duplicates = [dict(row) for row in conn.execute(query).fetchall()]
            active_duplicates = [dict(row) for row in conn.execute(active_query).fetchall()]
        else:
            with db_query() as db:
                duplicates = [dict(row) for row in db.execute(query).fetchall()]
                active_duplicates = [dict(row) for row in db.execute(active_query).fetchall()]
        return {
            "duplicate_external_ids": duplicates,
            "duplicate_active_user_mappings": active_duplicates,
        }
