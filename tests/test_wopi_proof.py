import base64

import pytest
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from kiara.web.wopi_proof import (
    DOTNET_EPOCH_TICKS,
    TICKS_PER_SECOND,
    WopiProofValidationError,
    WopiProofValidator,
    build_expected_proof,
    dotnet_ticks_now,
)


def _b64_int(value: int) -> str:
    return base64.b64encode(value.to_bytes((value.bit_length() + 7) // 8, "big")).decode("ascii")


def _keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _discovery_xml(current_private, old_private=None) -> bytes:
    current = current_private.public_key().public_numbers()
    attrs = {
        "modulus": _b64_int(current.n),
        "exponent": _b64_int(current.e),
    }
    if old_private:
        old = old_private.public_key().public_numbers()
        attrs["oldmodulus"] = _b64_int(old.n)
        attrs["oldexponent"] = _b64_int(old.e)
    attr_text = " ".join(f'{key}="{value}"' for key, value in attrs.items())
    return f"<wopi-discovery><proof-key {attr_text}/></wopi-discovery>".encode("utf-8")


def _sign(private_key, access_token: str, request_url: str, timestamp: int) -> str:
    signature = private_key.sign(
        build_expected_proof(access_token, request_url, timestamp),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("ascii")


async def _validator_for(xml: bytes, calls: list[str] | None = None) -> WopiProofValidator:
    async def fetcher(url):
        if calls is not None:
            calls.append(url)
        return xml

    return WopiProofValidator(fetcher=fetcher)


@pytest.mark.asyncio
async def test_valid_signature_passes():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now()
    proof = _sign(
        private_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


@pytest.mark.asyncio
async def test_invalid_signature_fails():
    private_key = _keypair()
    other_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now()
    proof = _sign(
        other_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert not await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


@pytest.mark.asyncio
async def test_expired_timestamp_fails():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now() - (21 * 60 * TICKS_PER_SECOND)
    proof = _sign(
        private_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert not await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


@pytest.mark.asyncio
async def test_future_timestamp_fails():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now() + (21 * 60 * TICKS_PER_SECOND)
    proof = _sign(
        private_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert not await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


@pytest.mark.asyncio
async def test_proof_old_fallback_with_current_key_passes():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now()
    proof_old = _sign(
        private_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof="invalid",
        proof_old=proof_old,
    )


@pytest.mark.asyncio
async def test_current_proof_with_old_key_passes():
    current_key = _keypair()
    old_key = _keypair()
    validator = await _validator_for(_discovery_xml(current_key, old_key))
    timestamp = dotnet_ticks_now()
    proof = _sign(
        old_key, "tok", "https://kiara.example/api/wopi/files/1?access_token=tok", timestamp
    )

    assert await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


@pytest.mark.asyncio
async def test_missing_proof_header_fails():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))

    assert not await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(dotnet_ticks_now()),
        proof="",
    )


@pytest.mark.asyncio
async def test_discovery_unreachable_fails_closed():
    async def failing_fetcher(url):
        raise RuntimeError("offline")

    validator = WopiProofValidator(fetcher=failing_fetcher)
    with pytest.raises(WopiProofValidationError):
        await validator.validate(
            document_server_url="https://onlyoffice.example",
            access_token="tok",
            request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
            timestamp=str(dotnet_ticks_now()),
            proof="abc",
        )


@pytest.mark.asyncio
async def test_discovery_cached_after_first_fetch():
    private_key = _keypair()
    calls: list[str] = []
    validator = await _validator_for(_discovery_xml(private_key), calls)
    timestamp = dotnet_ticks_now()
    request_url = "https://kiara.example/api/wopi/files/1?access_token=tok"
    proof = _sign(private_key, "tok", request_url, timestamp)

    assert await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url=request_url,
        timestamp=str(timestamp),
        proof=proof,
    )
    assert await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url=request_url,
        timestamp=str(timestamp),
        proof=proof,
    )
    assert calls == ["https://onlyoffice.example/hosting/discovery"]


@pytest.mark.asyncio
async def test_url_query_string_is_included_in_signature():
    private_key = _keypair()
    validator = await _validator_for(_discovery_xml(private_key))
    timestamp = dotnet_ticks_now()
    signed_url = "https://kiara.example/api/wopi/files/1?access_token=tok&x=1"
    proof = _sign(private_key, "tok", signed_url, timestamp)

    assert not await validator.validate(
        document_server_url="https://onlyoffice.example",
        access_token="tok",
        request_url="https://kiara.example/api/wopi/files/1?access_token=tok",
        timestamp=str(timestamp),
        proof=proof,
    )


def test_dotnet_ticks_are_used_for_timestamp_shape():
    assert dotnet_ticks_now() > DOTNET_EPOCH_TICKS
