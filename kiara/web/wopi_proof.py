"""WOPI proof-key validation for OnlyOffice requests."""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kiara.config import DEFAULT_TIMEOUTS

logger = logging.getLogger(__name__)

DISCOVERY_CACHE_TTL_SECONDS = 24 * 3600
WOPI_TIMESTAMP_TOLERANCE_SECONDS = 20 * 60
DOTNET_EPOCH_TICKS = 621355968000000000
TICKS_PER_SECOND = 10_000_000
MAX_DISCOVERY_XML_BYTES = 1024 * 1024


@dataclass(frozen=True)
class WopiProofKeys:
    """Cached WOPI proof public keys from the Document Server discovery XML."""

    current_key: rsa.RSAPublicKey
    old_key: rsa.RSAPublicKey | None
    fetched_at: float
    expires_at: float


class WopiProofValidationError(RuntimeError):
    """Raised when WOPI proof-key validation cannot be completed."""


def dotnet_ticks_now() -> int:
    """Return current UTC time as .NET ticks, as used by X-WOPI-TimeStamp."""
    return int(time.time() * TICKS_PER_SECOND) + DOTNET_EPOCH_TICKS


def build_expected_proof(access_token: str, request_url: str, timestamp: int) -> bytes:
    """Build the signed WOPI proof byte sequence."""
    access_token_bytes = access_token.encode("utf-8")
    request_url_bytes = request_url.upper().encode("utf-8")
    timestamp_bytes = int(timestamp).to_bytes(8, "big", signed=False)
    return b"".join(
        (
            len(access_token_bytes).to_bytes(4, "big", signed=False),
            access_token_bytes,
            len(request_url_bytes).to_bytes(4, "big", signed=False),
            request_url_bytes,
            len(timestamp_bytes).to_bytes(4, "big", signed=False),
            timestamp_bytes,
        )
    )


def _b64decode(value: str) -> bytes:
    raw = (value or "").strip()
    if not raw:
        raise ValueError("empty base64 value")
    raw += "=" * (-len(raw) % 4)
    return base64.b64decode(raw.encode("ascii"), validate=False)


def _public_key_from_components(modulus_b64: str, exponent_b64: str) -> rsa.RSAPublicKey:
    modulus = int.from_bytes(_b64decode(modulus_b64), "big", signed=False)
    exponent = int.from_bytes(_b64decode(exponent_b64), "big", signed=False)
    if modulus <= 0 or exponent <= 1:
        raise ValueError("invalid RSA public key components")
    return rsa.RSAPublicNumbers(exponent, modulus).public_key()


def _find_proof_key_element(root: ET.Element) -> ET.Element | None:
    proof_key = root.find(".//{*}proof-key")
    if proof_key is not None:
        return proof_key
    return root.find(".//proof-key")


def _find_old_proof_key_element(root: ET.Element) -> ET.Element | None:
    old_key = root.find(".//{*}proof-key-old")
    if old_key is not None:
        return old_key
    return root.find(".//proof-key-old")


def parse_discovery_keys(
    xml_content: bytes, *, ttl_seconds: int = DISCOVERY_CACHE_TTL_SECONDS
) -> WopiProofKeys:
    """Parse current and old WOPI proof keys from a discovery XML document."""
    if len(xml_content) > MAX_DISCOVERY_XML_BYTES:
        raise WopiProofValidationError("WOPI discovery XML zu gross")
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        raise WopiProofValidationError("WOPI discovery XML ungueltig") from exc

    proof_key = _find_proof_key_element(root)
    if proof_key is None:
        raise WopiProofValidationError("WOPI discovery enthaelt keinen proof-key")

    attrs = proof_key.attrib
    try:
        current_key = _public_key_from_components(attrs["modulus"], attrs["exponent"])
    except KeyError as exc:
        raise WopiProofValidationError("WOPI proof-key ohne modulus/exponent") from exc
    except ValueError as exc:
        raise WopiProofValidationError("WOPI proof-key ungueltig") from exc

    old_key = None
    old_modulus = attrs.get("oldmodulus")
    old_exponent = attrs.get("oldexponent")
    if not old_modulus or not old_exponent:
        old_node = _find_old_proof_key_element(root)
        if old_node is not None:
            old_modulus = old_node.attrib.get("modulus")
            old_exponent = old_node.attrib.get("exponent")
    if old_modulus and old_exponent:
        try:
            old_key = _public_key_from_components(old_modulus, old_exponent)
        except ValueError as exc:
            raise WopiProofValidationError("Alter WOPI proof-key ungueltig") from exc

    now = time.time()
    return WopiProofKeys(
        current_key=current_key,
        old_key=old_key,
        fetched_at=now,
        expires_at=now + ttl_seconds,
    )


def _verify_signature(public_key: rsa.RSAPublicKey, proof_b64: str, expected_proof: bytes) -> bool:
    if not proof_b64:
        return False
    try:
        signature = _b64decode(proof_b64)
        public_key.verify(signature, expected_proof, padding.PKCS1v15(), hashes.SHA256())
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


class WopiProofValidator:
    """Fetches WOPI discovery keys and validates X-WOPI-Proof headers."""

    def __init__(
        self,
        *,
        ttl_seconds: int = DISCOVERY_CACHE_TTL_SECONDS,
        fetcher=None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self._fetcher = fetcher
        self._cache: dict[str, WopiProofKeys] = {}
        self._lock = threading.RLock()

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    async def refresh(self, document_server_url: str) -> WopiProofKeys:
        """Force-refresh discovery keys for a Document Server URL."""
        cache_key = self._cache_key(document_server_url)
        discovery_url = f"{cache_key}/hosting/discovery"
        xml_content = await self._fetch_discovery(discovery_url)
        keys = parse_discovery_keys(xml_content, ttl_seconds=self.ttl_seconds)
        with self._lock:
            self._cache[cache_key] = keys
        return keys

    async def validate(
        self,
        *,
        document_server_url: str,
        access_token: str,
        request_url: str,
        timestamp: str,
        proof: str,
        proof_old: str = "",
    ) -> bool:
        """Return True when the request has a valid current or rotation proof."""
        if not access_token or not request_url or not proof or not timestamp:
            return False
        try:
            timestamp_int = int(timestamp)
        except (TypeError, ValueError):
            return False
        if not self._timestamp_is_fresh(timestamp_int):
            return False

        keys = await self._get_keys(document_server_url)
        expected = build_expected_proof(access_token, request_url, timestamp_int)
        if self._verify_with_keys(keys, proof, proof_old, expected):
            return True

        try:
            keys = await self.refresh(document_server_url)
        except WopiProofValidationError:
            logger.warning("WOPI proof refresh nach Verify-Fehler fehlgeschlagen", exc_info=True)
            return False
        return self._verify_with_keys(keys, proof, proof_old, expected)

    async def _get_keys(self, document_server_url: str) -> WopiProofKeys:
        cache_key = self._cache_key(document_server_url)
        now = time.time()
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return cached
        return await self.refresh(cache_key)

    async def _fetch_discovery(self, discovery_url: str) -> bytes:
        if self._fetcher is not None:
            try:
                content = await self._fetcher(discovery_url)
            except Exception as exc:
                raise WopiProofValidationError("WOPI discovery nicht abrufbar") from exc
            if len(content) > MAX_DISCOVERY_XML_BYTES:
                raise WopiProofValidationError("WOPI discovery XML zu gross")
            return content
        try:
            async with httpx.AsyncClient(
                timeout=DEFAULT_TIMEOUTS["http_api"],
                follow_redirects=False,
            ) as client:
                resp = await client.get(discovery_url)
                resp.raise_for_status()
        except Exception as exc:
            raise WopiProofValidationError("WOPI discovery nicht abrufbar") from exc
        if len(resp.content) > MAX_DISCOVERY_XML_BYTES:
            raise WopiProofValidationError("WOPI discovery XML zu gross")
        return resp.content

    def _cache_key(self, document_server_url: str) -> str:
        parsed = urlparse(str(document_server_url or "").strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise WopiProofValidationError("OnlyOffice Document Server URL ungueltig")
        path = parsed.path.rstrip("/")
        return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")

    def _timestamp_is_fresh(self, timestamp: int) -> bool:
        now = dotnet_ticks_now()
        tolerance = WOPI_TIMESTAMP_TOLERANCE_SECONDS * TICKS_PER_SECOND
        return abs(now - timestamp) <= tolerance

    def _verify_with_keys(
        self,
        keys: WopiProofKeys,
        proof: str,
        proof_old: str,
        expected_proof: bytes,
    ) -> bool:
        return (
            _verify_signature(keys.current_key, proof, expected_proof)
            or _verify_signature(keys.current_key, proof_old, expected_proof)
            or (keys.old_key is not None and _verify_signature(keys.old_key, proof, expected_proof))
        )
