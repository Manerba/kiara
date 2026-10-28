from __future__ import annotations

import base64

import pytest

from kiara.repositories.external_identity import (
    OPENDESK_IDENTITY_CLAIM,
    OPENDESK_PROVIDER,
    ExternalIdentityRepository,
)
from kiara.shared.ad_sid import normalize_ad_sid
from kiara.web.external_identity_migration import migrate_external_identity_schema
from kiara.web.schema import create_tables

ISSUER = "https://id.opendesk.hirschgasse5.de/realms/opendesk"
SID_ONE = "S-1-5-21-111-222-333-1001"
SID_TWO = "S-1-5-21-111-222-333-1002"


def _sid_to_base64(sid: str) -> str:
    parts = [int(p) for p in sid.split("-")[1:]]
    raw = bytes([parts[0], len(parts[2:])])
    raw += parts[1].to_bytes(6, byteorder="big")
    for subauthority in parts[2:]:
        raw += subauthority.to_bytes(4, byteorder="little")
    return base64.b64encode(raw).decode("ascii")


def _insert_user(conn, username="oidc-user", *, active=True):
    row = conn.execute(
        """
        INSERT INTO users (username, password_hash, email, is_active, is_admin)
        VALUES (%s, '', %s, %s, false)
        RETURNING id
        """,
        (username, f"{username}@example.test", active),
    ).fetchone()
    return int(row["id"])


def test_ad_sid_normalization_accepts_canonical_and_base64_objectsid():
    assert normalize_ad_sid("s-1-5-21-111-222-333-1001") == SID_ONE
    assert normalize_ad_sid(_sid_to_base64(SID_ONE)) == SID_ONE


def test_external_identity_migration_is_idempotent_and_preserves_existing_data(test_db):
    user_id = _insert_user(test_db)
    test_db.execute("DROP TABLE public.user_external_identities")

    create_tables(test_db)
    create_tables(test_db)

    row = test_db.execute(
        "SELECT id, username FROM users WHERE id = %s",
        (user_id,),
    ).fetchone()
    assert row["username"] == "oidc-user"
    table = test_db.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema='public' AND table_name='user_external_identities'"
    ).fetchone()
    assert table is not None


def test_external_identity_repository_upsert_and_resolve(test_db):
    migrate_external_identity_schema(test_db)
    user_id = _insert_user(test_db)
    repo = ExternalIdentityRepository()

    row = repo.upsert_identity(
        provider=OPENDESK_PROVIDER,
        issuer=ISSUER,
        identity_claim=OPENDESK_IDENTITY_CLAIM,
        external_id=_sid_to_base64(SID_ONE),
        user_id=user_id,
        conn=test_db,
    )

    assert row["external_id"] == SID_ONE
    resolved = repo.get_mapping_with_user(
        provider=OPENDESK_PROVIDER,
        issuer=ISSUER,
        identity_claim=OPENDESK_IDENTITY_CLAIM,
        external_id=SID_ONE,
        conn=test_db,
    )
    assert resolved["id"] == user_id
    assert resolved["mapping_active"] is True
    assert resolved["user_active"] is True


def test_external_identity_upsert_rejects_silent_reassignment(test_db):
    migrate_external_identity_schema(test_db)
    first_user_id = _insert_user(test_db, "oidc-user-a")
    second_user_id = _insert_user(test_db, "oidc-user-b")
    repo = ExternalIdentityRepository()

    repo.upsert_identity(
        provider=OPENDESK_PROVIDER,
        issuer=ISSUER,
        identity_claim=OPENDESK_IDENTITY_CLAIM,
        external_id=SID_ONE,
        user_id=first_user_id,
        conn=test_db,
    )

    with pytest.raises(ValueError, match="anderen KIara-User"):
        repo.upsert_identity(
            provider=OPENDESK_PROVIDER,
            issuer=ISSUER,
            identity_claim=OPENDESK_IDENTITY_CLAIM,
            external_id=SID_ONE,
            user_id=second_user_id,
            conn=test_db,
        )

    resolved = repo.get_mapping_with_user(
        provider=OPENDESK_PROVIDER,
        issuer=ISSUER,
        identity_claim=OPENDESK_IDENTITY_CLAIM,
        external_id=SID_ONE,
        conn=test_db,
    )
    assert resolved["id"] == first_user_id


def test_external_identity_constraints_reject_invalid_and_duplicate_active_user(test_db):
    migrate_external_identity_schema(test_db)
    user_id = _insert_user(test_db)
    repo = ExternalIdentityRepository()

    with pytest.raises(ValueError):
        repo.upsert_identity(
            provider=OPENDESK_PROVIDER,
            issuer=ISSUER,
            identity_claim=OPENDESK_IDENTITY_CLAIM,
            external_id="not-a-sid",
            user_id=user_id,
            conn=test_db,
        )

    repo.upsert_identity(
        provider=OPENDESK_PROVIDER,
        issuer=ISSUER,
        identity_claim=OPENDESK_IDENTITY_CLAIM,
        external_id=SID_ONE,
        user_id=user_id,
        conn=test_db,
    )
    with pytest.raises(Exception):
        repo.upsert_identity(
            provider=OPENDESK_PROVIDER,
            issuer=ISSUER,
            identity_claim=OPENDESK_IDENTITY_CLAIM,
            external_id=SID_TWO,
            user_id=user_id,
            conn=test_db,
        )
    test_db.rollback()
