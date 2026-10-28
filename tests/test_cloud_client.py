"""Tests fuer den Cloud-KI-Client (kiara.cloud.client).

Testet API-Format-Erzeugung (OpenAI/Anthropic), Retry-Logik,
Timeout-Handling und Token-Counting.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kiara.cloud.client import (
    _ANTHROPIC_BASE,
    _OPENAI_BASE,
    CloudKIClient,
)
from kiara.shared.constants import (
    ANTHROPIC_API_VERSION as _ANTHROPIC_VERSION,
)
from kiara.shared.constants import (
    CLOUD_MAX_RETRIES as _MAX_RETRIES,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_client():
    """OpenAI-Client mit Testdaten."""
    return CloudKIClient(
        provider="openai",
        api_key="sk-test-openai-key",
        model="gpt-4o-mini",
        max_tokens=2048,
    )


@pytest.fixture
def anthropic_client():
    """Anthropic-Client mit Testdaten."""
    return CloudKIClient(
        provider="anthropic",
        api_key="sk-ant-test-key",
        model="claude-sonnet-4-6",
        max_tokens=4096,
    )


@pytest.fixture
def custom_base_url_client():
    """Client mit benutzerdefinierter Base-URL."""
    return CloudKIClient(
        provider="openai",
        api_key="sk-test-key",
        model="local-model",
        base_url="https://my-proxy.example.com",
    )


# ---------------------------------------------------------------------------
# OpenAI API-Format
# ---------------------------------------------------------------------------


class TestOpenAIFormat:
    def test_openai_url_default(self, openai_client):
        """Standard OpenAI-URL."""
        url = openai_client._openai_url()
        assert url == f"{_OPENAI_BASE}/v1/chat/completions"

    def test_openai_url_custom_base(self, custom_base_url_client):
        """Benutzerdefinierte Base-URL."""
        url = custom_base_url_client._openai_url()
        assert url == "https://my-proxy.example.com/v1/chat/completions"

    def test_openai_headers(self, openai_client):
        """Authorization-Header korrekt gesetzt."""
        headers = openai_client._openai_headers()
        assert headers["Authorization"] == "Bearer sk-test-openai-key"
        assert headers["Content-Type"] == "application/json"

    def test_openai_body_with_context(self, openai_client):
        """Request-Body mit Kontext korrekt formatiert."""
        body = openai_client._openai_body(
            prompt="Was ist RAG?",
            context="RAG steht fuer Retrieval-Augmented Generation.",
            temperature=0.5,
            stream=False,
        )
        assert body["model"] == "gpt-4o-mini"
        assert body["max_tokens"] == 2048
        assert body["temperature"] == 0.5
        assert body["stream"] is False
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"
        assert "Kontext:" in body["messages"][1]["content"]
        assert "Was ist RAG?" in body["messages"][1]["content"]

    def test_openai_body_without_context(self, openai_client):
        """Request-Body ohne Kontext: Nur Prompt im User-Message."""
        body = openai_client._openai_body(
            prompt="Einfache Frage",
            context="",
            temperature=0.3,
            stream=True,
        )
        assert body["messages"][1]["content"] == "Einfache Frage"
        assert body["stream"] is True

    def test_openai_body_stream_flag(self, openai_client):
        """Stream-Flag wird korrekt gesetzt."""
        body_stream = openai_client._openai_body("p", "c", 0.3, stream=True)
        body_no_stream = openai_client._openai_body("p", "c", 0.3, stream=False)
        assert body_stream["stream"] is True
        assert body_no_stream["stream"] is False


# ---------------------------------------------------------------------------
# Anthropic API-Format
# ---------------------------------------------------------------------------


class TestAnthropicFormat:
    def test_anthropic_url_default(self, anthropic_client):
        """Standard Anthropic-URL."""
        url = anthropic_client._anthropic_url()
        assert url == f"{_ANTHROPIC_BASE}/v1/messages"

    def test_anthropic_headers(self, anthropic_client):
        """Anthropic-spezifische Headers."""
        headers = anthropic_client._anthropic_headers()
        assert headers["x-api-key"] == "sk-ant-test-key"
        assert headers["anthropic-version"] == _ANTHROPIC_VERSION
        assert headers["Content-Type"] == "application/json"
        # Anthropic nutzt KEIN Authorization-Header
        assert "Authorization" not in headers

    def test_anthropic_body_with_context(self, anthropic_client):
        """Anthropic-Body: system als Top-Level-Feld, nicht in messages."""
        body = anthropic_client._anthropic_body(
            prompt="Analyse bitte",
            context="Kontext-Daten hier",
            temperature=0.3,
            stream=False,
        )
        assert body["model"] == "claude-sonnet-4-6"
        assert body["max_tokens"] == 4096
        assert "system" in body
        assert body["system"] == anthropic_client.system_prompt
        # Nur user-message, kein system in messages
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert "Kontext:" in body["messages"][0]["content"]

    def test_anthropic_body_without_context(self, anthropic_client):
        """Anthropic-Body ohne Kontext."""
        body = anthropic_client._anthropic_body(
            prompt="Direkte Frage",
            context="",
            temperature=0.5,
            stream=False,
        )
        assert body["messages"][0]["content"] == "Direkte Frage"


# ---------------------------------------------------------------------------
# Generate (non-streaming) mit gemocktem HTTP
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_openai_generate_success(self, openai_client):
        """Erfolgreiche OpenAI-Anfrage: Antwort extrahiert, Tokens getrackt."""
        mock_response = {
            "choices": [{"message": {"content": "Das ist die Antwort."}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 45},
        }

        with patch.object(openai_client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await openai_client.generate("Frage?", context="Kontext")

        assert result == "Das ist die Antwort."
        assert openai_client.last_usage["tokens_in"] == 120
        assert openai_client.last_usage["tokens_out"] == 45
        assert openai_client.last_usage["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_anthropic_generate_success(self, anthropic_client):
        """Erfolgreiche Anthropic-Anfrage."""
        mock_response = {
            "content": [{"type": "text", "text": "Anthropic-Antwort hier."}],
            "usage": {"input_tokens": 80, "output_tokens": 30},
        }

        with patch.object(anthropic_client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await anthropic_client.generate("Frage?")

        assert result == "Anthropic-Antwort hier."
        assert anthropic_client.last_usage["tokens_in"] == 80
        assert anthropic_client.last_usage["tokens_out"] == 30

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self):
        """Unbekannter Provider wirft GenerationError."""
        from kiara.exceptions import GenerationError

        client = CloudKIClient(
            provider="unknown",
            api_key="key",
            model="model",
        )
        with pytest.raises(GenerationError, match="Unbekannter Provider"):
            await client.generate("Frage?")

    @pytest.mark.asyncio
    async def test_anthropic_empty_content_returns_empty(self, anthropic_client):
        """Leere Anthropic-Response gibt leeren String zurueck."""
        mock_response = {
            "content": [],
            "usage": {"input_tokens": 10, "output_tokens": 0},
        }

        with patch.object(anthropic_client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            result = await anthropic_client.generate("Frage?")

        assert result == ""


# ---------------------------------------------------------------------------
# Retry-Logik
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retry_on_429(self, openai_client):
        """Retry bei HTTP 429 (Rate Limit)."""
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "choices": [{"message": {"content": "OK"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        mock_resp_200.raise_for_status = MagicMock()

        call_count = 0

        async def mock_request(method, url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_resp_429
            return mock_resp_200

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.request = mock_request
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await openai_client._request(
                "POST",
                "https://api.example.com/v1/test",
                {"Authorization": "Bearer test"},
                {"model": "test"},
            )

        assert call_count == 2
        assert result == mock_resp_200.json.return_value

    @pytest.mark.asyncio
    async def test_retry_on_500(self, openai_client):
        """Retry bei HTTP 500 (Server Error)."""
        mock_resp_500 = MagicMock()
        mock_resp_500.status_code = 500

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {"choices": [{"message": {"content": "OK"}}]}
        mock_resp_200.raise_for_status = MagicMock()

        call_count = 0

        async def mock_request(method, url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock_resp_500
            return mock_resp_200

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.request = mock_request
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await openai_client._request(
                "POST",
                "https://api.example.com/v1/test",
                {},
                {"model": "test"},
            )

        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded_raises(self, openai_client):
        """Nach _MAX_RETRIES Versuchen: raise_for_status."""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500", request=MagicMock(), response=mock_resp
        )

        async def mock_request(method, url, headers=None, json=None):
            return mock_resp

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.request = mock_request
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.HTTPStatusError):
                await openai_client._request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {"model": "test"},
                )


# ---------------------------------------------------------------------------
# Stream-Mock-Helpers
# ---------------------------------------------------------------------------


class MockAsyncLineIterator:
    """Async-Iterator fuer resp.aiter_lines() Mocking."""

    def __init__(self, lines: list):
        self._lines = lines
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._lines):
            raise StopAsyncIteration
        line = self._lines[self._index]
        self._index += 1
        if isinstance(line, Exception):
            raise line
        return line


def _make_stream_ctx(status_code: int = 200, lines: list | None = None, enter_effect=None):
    """Erzeugt async context manager Mock fuer client.stream()."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.raise_for_status = MagicMock()
    if status_code >= 400:
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            str(status_code), request=MagicMock(), response=mock_resp
        )
    if lines is not None:
        mock_resp.aiter_lines = lambda: MockAsyncLineIterator(lines)

    ctx = AsyncMock()
    if enter_effect:
        ctx.__aenter__ = AsyncMock(side_effect=enter_effect)
    else:
        ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


# ---------------------------------------------------------------------------
# Stream-Retry-Logik
# ---------------------------------------------------------------------------


class TestStreamRetryLogic:
    @pytest.mark.asyncio
    async def test_stream_retry_on_503(self, openai_client):
        """503 beim Connect → Retry → Erfolg beim zweiten Versuch."""
        call_count = 0

        def mock_stream(method, url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_stream_ctx(503)
            return _make_stream_ctx(200, lines=["data: hello", "data: world"])

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.stream = mock_stream
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            lines = [
                line
                async for line in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {"Authorization": "Bearer test"},
                    {"model": "test"},
                )
            ]

        assert call_count == 2
        assert lines == ["data: hello", "data: world"]

    @pytest.mark.asyncio
    async def test_stream_retry_on_connect_timeout(self, openai_client):
        """ConnectTimeout → Retry → Erfolg beim zweiten Versuch."""
        call_count = 0

        def mock_stream(method, url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_stream_ctx(enter_effect=httpx.ConnectTimeout("timeout"))
            return _make_stream_ctx(200, lines=["data: ok"])

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = mock_stream
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            lines = [
                line
                async for line in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                )
            ]

        assert call_count == 2
        assert lines == ["data: ok"]
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_no_retry_after_yield(self, openai_client):
        """ReadTimeout NACH erstem yield → Exception propagiert, kein Retry."""
        # aiter_lines liefert einen Chunk, dann ReadTimeout
        lines_with_error = ["data: first-chunk", httpx.ReadTimeout("mid-stream")]

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(200, lines=lines_with_error)
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.ReadTimeout):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

        mock_sleep.assert_not_called()  # Kein Retry-Sleep!

    @pytest.mark.asyncio
    async def test_stream_max_retries_raises_status_error(self, openai_client):
        """3x 503 → letzter Versuch raise_for_status() → HTTPStatusError."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            # Alle Versuche liefern 503
            client_instance.stream = MagicMock(return_value=_make_stream_ctx(503))
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.HTTPStatusError):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

    @pytest.mark.asyncio
    async def test_stream_success_no_retry(self, openai_client):
        """200 beim ersten Versuch — Retry-Loop stoert Happy-Path nicht."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(200, lines=["data: success"])
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            lines = [
                line
                async for line in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                )
            ]

        assert lines == ["data: success"]
        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_stream_max_retries_timeout_raises(self, openai_client):
        """3x ConnectTimeout → letzter Versuch raise → TimeoutException."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(enter_effect=httpx.ConnectTimeout("timeout"))
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.ConnectTimeout):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

        # 2 Retry-Sleeps (Attempt 1 + 2), Attempt 3 raised direkt
        assert mock_sleep.call_count == 2

    @pytest.mark.asyncio
    async def test_stream_backoff_values(self, openai_client):
        """Backoff-Werte sind 2^attempt: sleep(2), sleep(4)."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(enter_effect=httpx.ConnectTimeout("timeout"))
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.ConnectTimeout):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

        # 2^1=2, 2^2=4 (Attempt 3 raised direkt ohne sleep)
        assert mock_sleep.call_args_list == [
            ((2,),),
            ((4,),),
        ]

    @pytest.mark.asyncio
    async def test_stream_non_retry_status_raises_immediately(self, openai_client):
        """400/401 sind nicht in _RETRY_STATUSES → sofort HTTPStatusError, kein Retry."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(return_value=_make_stream_ctx(401))
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.HTTPStatusError):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

        mock_sleep.assert_not_called()  # Kein Retry bei nicht-retryable Status

    @pytest.mark.asyncio
    async def test_stream_connect_error_propagates_immediately(self, openai_client):
        """ConnectError (DNS/Port) ist kein TimeoutException → sofort raise, kein Retry."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(enter_effect=httpx.ConnectError("Connection refused"))
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.ConnectError):
                async for _ in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                ):
                    pass

        mock_sleep.assert_not_called()  # ConnectError wird nie retried

    @pytest.mark.asyncio
    async def test_stream_empty_lines_filtered(self, openai_client):
        """Leere Zeilen und Whitespace-only Zeilen werden nicht yielded."""
        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.stream = MagicMock(
                return_value=_make_stream_ctx(
                    200,
                    lines=[
                        "data: first",
                        "",
                        "  ",
                        "\t",
                        "data: second",
                        "",
                    ],
                )
            )
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            lines = [
                line
                async for line in openai_client._stream_request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {},
                )
            ]

        assert lines == ["data: first", "data: second"]


# ---------------------------------------------------------------------------
# Timeout-Handling
# ---------------------------------------------------------------------------


class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout_retries_then_raises(self, openai_client):
        """Timeout fuehrt zu Retry, nach Max-Retries wird TimeoutException geworfen."""

        async def mock_request(method, url, headers=None, json=None):
            raise httpx.TimeoutException("Connection timed out")

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            client_instance = AsyncMock()
            client_instance.request = mock_request
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            with pytest.raises(httpx.TimeoutException):
                await openai_client._request(
                    "POST",
                    "https://api.example.com/v1/test",
                    {},
                    {"model": "test"},
                )

            # sleep wurde (MAX_RETRIES - 1) mal aufgerufen
            assert mock_sleep.call_count == _MAX_RETRIES - 1

    @pytest.mark.asyncio
    async def test_timeout_recovers_on_retry(self, openai_client):
        """Timeout beim ersten Versuch, Erfolg beim zweiten."""
        call_count = 0

        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "choices": [{"message": {"content": "Recovered"}}],
            "usage": {},
        }
        mock_resp_200.raise_for_status = MagicMock()

        async def mock_request(method, url, headers=None, json=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("timeout")
            return mock_resp_200

        with (
            patch("kiara.cloud.client.httpx.AsyncClient") as MockClient,
            patch("kiara.cloud.client.asyncio.sleep", new_callable=AsyncMock),
        ):
            client_instance = AsyncMock()
            client_instance.request = mock_request
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance

            result = await openai_client._request(
                "POST",
                "https://api.example.com/v1/test",
                {},
                {"model": "test"},
            )

        assert result["choices"][0]["message"]["content"] == "Recovered"
        assert call_count == 2


# ---------------------------------------------------------------------------
# Token-Counting (last_usage)
# ---------------------------------------------------------------------------


class TestTokenCounting:
    def test_initial_usage_zeros(self, openai_client):
        """Anfangs-Tokens sind 0."""
        usage = openai_client.last_usage
        assert usage["tokens_in"] == 0
        assert usage["tokens_out"] == 0
        assert usage["duration_ms"] == 0

    @pytest.mark.asyncio
    async def test_usage_updated_after_openai_generate(self, openai_client):
        """Token-Verbrauch wird nach OpenAI-Anfrage aktualisiert."""
        mock_response = {
            "choices": [{"message": {"content": "Antwort"}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 80},
        }

        with patch.object(openai_client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await openai_client.generate("Test-Prompt")

        usage = openai_client.last_usage
        assert usage["tokens_in"] == 200
        assert usage["tokens_out"] == 80
        assert usage["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_usage_updated_after_anthropic_generate(self, anthropic_client):
        """Token-Verbrauch wird nach Anthropic-Anfrage aktualisiert."""
        mock_response = {
            "content": [{"type": "text", "text": "Claude-Antwort"}],
            "usage": {"input_tokens": 150, "output_tokens": 60},
        }

        with patch.object(anthropic_client, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            await anthropic_client.generate("Test")

        usage = anthropic_client.last_usage
        assert usage["tokens_in"] == 150
        assert usage["tokens_out"] == 60

    def test_last_usage_returns_dict(self, openai_client):
        """last_usage gibt immer ein Dict mit den drei Keys zurueck."""
        usage = openai_client.last_usage
        assert isinstance(usage, dict)
        assert "tokens_in" in usage
        assert "tokens_out" in usage
        assert "duration_ms" in usage


# ---------------------------------------------------------------------------
# Konstruktor und Konfiguration
# ---------------------------------------------------------------------------


class TestClientInit:
    def test_default_system_prompt(self, openai_client):
        """Ohne expliziten system_prompt ist der Default leer (wird per DB gesetzt)."""
        assert openai_client.system_prompt == ""

    def test_custom_system_prompt(self):
        """Benutzerdefinierter System-Prompt."""
        client = CloudKIClient(
            provider="openai",
            api_key="key",
            model="model",
            system_prompt="Du bist ein Test-Bot.",
        )
        assert client.system_prompt == "Du bist ein Test-Bot."

    def test_custom_base_url(self):
        """Benutzerdefinierte Base-URL wird gespeichert."""
        client = CloudKIClient(
            provider="openai",
            api_key="key",
            model="model",
            base_url="https://proxy.example.com",
        )
        assert client.base_url == "https://proxy.example.com"

    def test_default_max_tokens(self):
        """Standard max_tokens ist 4096."""
        client = CloudKIClient(provider="openai", api_key="k", model="m")
        assert client.max_tokens == 4096

    def test_provider_and_model_stored(self, openai_client):
        """Provider und Model sind als Properties zugaenglich."""
        assert openai_client.provider == "openai"
        assert openai_client.model == "gpt-4o-mini"
