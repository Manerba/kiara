from __future__ import annotations

import base64
import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from kiara.web.oidc_auth import (
    OIDCRetrievalAuthenticator,
    OIDCUnauthorizedError,
    OIDCUnavailableError,
    _HTTPSOnlyRedirectHandler,
    audit_oidc_decision,
    sid_audit_hmac,
)

ISSUER = "https://id.opendesk.hirschgasse5.de/realms/opendesk"
DISCOVERY_URL = f"{ISSUER}/.well-known/openid-configuration"
JWKS_URL = f"{ISSUER}/protocol/openid-connect/certs"
CANONICAL_SID = "S-1-5-21-111-222-333-1001"


def _config():
    return {
        "web": {
            "oidc_retrieval": {
                "enabled": True,
                "issuer": ISSUER,
                "discovery_url": DISCOVERY_URL,
                "jwks_url": "",
                "audience": "kiara-retrieval",
                "client_binding_claim": "azp",
                "client_binding_value": "open-webui",
                "algorithms": ["RS256"],
                "identity_claim": "ad_sid",
                "identity_claim_format": "ad_object_sid_or_base64",
                "required_scopes": [],
                "required_roles": [],
                "clock_skew_seconds": 60,
                "ca_bundle_path": "",
                "allow_admin_full_access": False,
            }
        }
    }


def _sid_to_base64(sid: str) -> str:
    parts = [int(p) for p in sid.split("-")[1:]]
    revision = parts[0]
    authority = parts[1]
    subauthorities = parts[2:]
    raw = bytes([revision, len(subauthorities)])
    raw += authority.to_bytes(6, byteorder="big")
    for subauthority in subauthorities:
        raw += subauthority.to_bytes(4, byteorder="little")
    return base64.b64encode(raw).decode("ascii")


class FakeRepo:
    def __init__(self, row=None):
        self.row = row
        self.calls = []

    def get_mapping_with_user(self, **kwargs):
        self.calls.append(kwargs)
        return self.row


class FakeOIDCHTTP:
    def __init__(self, jwk, *, discovery_issuer=ISSUER):
        self.jwk = jwk
        self.discovery_issuer = discovery_issuer
        self.calls = []

    def __call__(self, url):
        self.calls.append(url)
        if url == DISCOVERY_URL:
            return {"issuer": self.discovery_issuer, "jwks_uri": JWKS_URL}, {}
        if url == JWKS_URL:
            return {"keys": [self.jwk]}, {"cache-control": "max-age=120"}
        raise AssertionError(f"Unexpected URL: {url}")


@contextmanager
def _fake_permissions():
    @contextmanager
    def fake_db_query():
        yield object()

    with (
        patch("kiara.shared.db.db_query", fake_db_query),
        patch("kiara.web.config_utils.load_user_permissions", return_value=([10], [20])),
        patch("kiara.web.oidc_auth.get_secret_key", return_value="test-secret"),
        patch("kiara.web.oidc_auth.ensure_security_log_handler", return_value=None),
    ):
        yield


def _key_material(kid="kid-1"):
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwk["kid"] = kid
    jwk["alg"] = "RS256"
    return private_key, jwk


def _token(private_key, kid="kid-1", **overrides):
    now = int(time.time())
    payload = {
        "iss": ISSUER,
        "aud": ["kiara-retrieval", "open-webui"],
        "azp": "open-webui",
        "typ": "Bearer",
        "ad_sid": _sid_to_base64(CANONICAL_SID),
        "sub": "user-id",
        "exp": now + 300,
        "nbf": now - 10,
    }
    payload.update(overrides)
    return jwt.encode(payload, private_key, algorithm="RS256", headers={"kid": kid})


def test_oidc_auth_valid_token_maps_base64_ad_sid_to_principal():
    private_key, jwk = _key_material()
    repo = FakeRepo(
        {
            "mapping_active": True,
            "user_active": True,
            "id": 7,
            "username": "alice",
            "email": "alice@example.test",
            "is_admin": False,
        }
    )
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=repo,
        http_get_json=FakeOIDCHTTP(jwk),
    )

    with _fake_permissions():
        principal = auth.authenticate(f"Bearer {_token(private_key)}", path="/api/retrieval/list")

    assert principal["id"] == 7
    assert principal["group_ids"] == [10]
    assert principal["datasource_ids"] == [20]
    assert principal["auth_source"] == "oidc"
    assert principal["oidc_external_id"] == CANONICAL_SID
    assert repo.calls[0]["external_id"] == CANONICAL_SID


def test_oidc_auth_rejects_missing_ad_sid_even_with_oidc_sid():
    private_key, jwk = _key_material()
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=FakeOIDCHTTP(jwk),
    )

    with pytest.raises(OIDCUnauthorizedError) as exc:
        auth.authenticate(
            f"Bearer {_token(private_key, ad_sid=None, sid='session-id')}",
            path="/api/retrieval/list",
        )

    assert exc.value.reason == "invalid_identity_claim"


def test_oidc_auth_rejects_wrong_typ_and_client_binding():
    private_key, jwk = _key_material()
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=FakeOIDCHTTP(jwk),
    )

    with pytest.raises(OIDCUnauthorizedError) as typ_exc:
        auth.authenticate(f"Bearer {_token(private_key, typ='ID')}", path="/api/retrieval/list")
    assert typ_exc.value.reason == "invalid_token_type"

    with pytest.raises(OIDCUnauthorizedError) as azp_exc:
        auth.authenticate(
            f"Bearer {_token(private_key, azp='other-client')}",
            path="/api/retrieval/list",
        )
    assert azp_exc.value.reason == "invalid_client_binding"


def test_oidc_auth_discovery_issuer_mismatch_is_503_class():
    private_key, jwk = _key_material()
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=FakeOIDCHTTP(jwk, discovery_issuer="https://wrong.example/realm"),
    )

    with pytest.raises(OIDCUnavailableError) as exc:
        auth.authenticate(f"Bearer {_token(private_key)}", path="/api/retrieval/list")

    assert exc.value.reason == "discovery_issuer_mismatch"


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"aud": "other-audience"}, "jwt_rejected"),
        ({"iss": "https://id.opendesk.hirschgasse5.de/realms/other"}, "jwt_rejected"),
        ({"exp": int(time.time()) - 120, "nbf": int(time.time()) - 300}, "jwt_rejected"),
        ({"nbf_offset_seconds": 120}, "jwt_rejected"),
        ({"ad_sid": 12345}, "invalid_identity_claim"),
        ({"azp": ["open-webui"]}, "invalid_client_binding"),
    ],
)
def test_oidc_auth_rejects_invalid_access_token_claims(overrides, reason):
    overrides = dict(overrides)
    nbf_offset = overrides.pop("nbf_offset_seconds", None)
    if nbf_offset is not None:
        # Erst bei Testausfuehrung berechnen: Collection kann in der Vollsuite
        # viele Minuten vor diesem parametrisierten Fall stattfinden.
        overrides["nbf"] = int(time.time()) + nbf_offset

    private_key, jwk = _key_material()
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=FakeOIDCHTTP(jwk),
    )

    with _fake_permissions(), pytest.raises(OIDCUnauthorizedError) as exc:
        auth.authenticate(
            f"Bearer {_token(private_key, **overrides)}",
            path="/api/retrieval/list",
        )

    assert exc.value.reason == reason


@pytest.mark.parametrize(
    ("config_patch", "token_overrides", "reason"),
    [
        (
            {"required_scopes": ["kiara-retrieval"]},
            {"scope": ["kiara-retrieval"]},
            "invalid_scope_claim",
        ),
        (
            {"required_roles": ["kiara-retrieval"]},
            {"realm_access": "invalid"},
            "invalid_role_claim",
        ),
        (
            {"required_roles": ["kiara-retrieval"]},
            {"realm_access": {"roles": [123]}},
            "invalid_role_claim",
        ),
        (
            {"required_roles": ["kiara-retrieval"]},
            {"resource_access": "invalid"},
            "invalid_role_claim",
        ),
    ],
)
def test_oidc_auth_rejects_scope_and_role_type_errors(config_patch, token_overrides, reason):
    private_key, jwk = _key_material()
    config = _config()
    config["web"]["oidc_retrieval"].update(config_patch)
    auth = OIDCRetrievalAuthenticator(
        config,
        external_identity_repo=FakeRepo(),
        http_get_json=FakeOIDCHTTP(jwk),
    )

    with _fake_permissions(), pytest.raises(OIDCUnauthorizedError) as exc:
        auth.authenticate(
            f"Bearer {_token(private_key, **token_overrides)}",
            path="/api/retrieval/list",
        )

    assert exc.value.reason == reason


def test_oidc_auth_blocks_non_https_redirects():
    handler = _HTTPSOnlyRedirectHandler()

    with pytest.raises(OIDCUnavailableError) as exc:
        handler.redirect_request(None, None, 302, "Found", {}, "http://issuer.example/certs")

    assert exc.value.reason == "oidc_redirect_non_https"


def test_oidc_auth_rate_limits_unknown_kid_refreshes():
    private_key, jwk = _key_material(kid="known-kid")
    http = FakeOIDCHTTP(jwk)
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=http,
    )

    with _fake_permissions():
        with pytest.raises(OIDCUnauthorizedError) as first:
            auth.authenticate(
                f"Bearer {_token(private_key, kid='unknown-kid-1')}",
                path="/api/retrieval/list",
            )
        with pytest.raises(OIDCUnauthorizedError) as second:
            auth.authenticate(
                f"Bearer {_token(private_key, kid='unknown-kid-2')}",
                path="/api/retrieval/list",
            )

    assert first.value.reason == "unknown_kid"
    assert second.value.reason == "unknown_kid"
    assert http.calls == [DISCOVERY_URL, JWKS_URL]


def test_oidc_auth_jwks_refresh_is_singleflight_for_parallel_requests():
    private_key, jwk = _key_material(kid="shared-kid")
    http = FakeOIDCHTTP(jwk)
    repo = FakeRepo(
        {
            "mapping_active": True,
            "user_active": True,
            "id": 7,
            "username": "alice",
            "email": "alice@example.test",
            "is_admin": False,
        }
    )
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=repo,
        http_get_json=http,
    )
    token = f"Bearer {_token(private_key, kid='shared-kid')}"

    def authenticate_once():
        return auth.authenticate(token, path="/api/retrieval/list")["id"]

    with _fake_permissions(), ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: authenticate_once(), range(8)))

    assert results == [7] * 8
    assert http.calls == [DISCOVERY_URL, JWKS_URL]


def test_oidc_auth_ignores_jwks_encryption_keys():
    _, signing_jwk = _key_material(kid="signing-kid")
    enc_jwk = dict(signing_jwk)
    enc_jwk["kid"] = "enc-kid"
    enc_jwk["use"] = "enc"
    enc_jwk["alg"] = "RSA-OAEP"
    http = FakeOIDCHTTP(signing_jwk)
    auth = OIDCRetrievalAuthenticator(
        _config(),
        external_identity_repo=FakeRepo(),
        http_get_json=http,
    )
    http.jwk = enc_jwk

    with pytest.raises(OIDCUnavailableError) as exc:
        auth._refresh_jwks_locked()

    assert exc.value.reason == "empty_jwks"


def test_oidc_audit_file_init_failure_is_fail_closed(monkeypatch):
    import logging

    import kiara.web.oidc_auth as oidc_auth

    monkeypatch.setattr(oidc_auth, "_SECURITY_HANDLER_INSTALLED", False)
    monkeypatch.setattr(
        logging, "FileHandler", lambda *a, **kw: (_ for _ in ()).throw(OSError("no log"))
    )

    with pytest.raises(OIDCUnavailableError) as exc:
        audit_oidc_decision(
            request_id="req",
            path="/api/retrieval/list",
            outcome="deny",
            reason="test",
        )

    assert exc.value.reason == "security_audit_unavailable"


def test_sid_audit_hmac_is_deterministic_and_does_not_contain_raw_sid():
    with patch("kiara.web.oidc_auth.get_secret_key", return_value="test-secret"):
        first = sid_audit_hmac(CANONICAL_SID)
        second = sid_audit_hmac(CANONICAL_SID)

    assert first == second
    assert len(first) == 64
    assert CANONICAL_SID not in first
