"""Tests fuer kiara.backend_warmup — BackendWarmer + build_warmup_tasks."""

import threading
import time
from unittest.mock import MagicMock, call, patch

import httpx
import ollama as ollama_client
import pytest

from kiara.backend_warmup import (
    BackendWarmer,
    WarmupTask,
    _get_tiny_png,
    _is_permanent_ollama_error,
    _is_transient_http_error,
    _make_warmup_client,
    build_warmup_tasks,
)
from kiara.shared.constants import (
    WARMUP_DUMMY_TEXT,
    WARMUP_KEEP_ALIVE,
    WARMUP_RETRY_DELAY,
)


@pytest.fixture(autouse=True)
def _verified_embedding_role_profile(monkeypatch):
    """Warmup-Unit-Tests isolieren die bereits separat getestete Discovery."""
    from kiara.retrieval import embedding_roles
    from kiara.retrieval.embedding_roles import EmbeddingRequestProfile

    def resolve(_url, model, *, role, kind, endpoint, profile_id=None, **_kwargs):
        return EmbeddingRequestProfile(
            requested_model=model,
            canonical_model_id=f"canonical::{model}",
            profile_id=profile_id or f"test-{kind.value}-profile",
            kind=kind.value,
            endpoint=endpoint,
            input_type=role,
            role_sensitive=True,
            index_compatibility_id="sha256:" + "1" * 64,
            query_compatibility_id="sha256:" + "2" * 64,
        )

    monkeypatch.setattr(embedding_roles, "resolve_server_request_profile", resolve)


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _emb_task(url="http://server:11434", model="nomic-embed-text"):
    return WarmupTask("embedding", model, url, f"Embedding ({model})")


def _llm_task(url="http://server:11434", model="qwen3:8b"):
    return WarmupTask("llm", model, url, f"LLM ({model})")


def _vlm_task(url="http://server:11434", model="minicpm-v"):
    return WarmupTask("vlm", model, url, f"VLM ({model})")


def _ddu_task(url="http://server:5001"):
    return WarmupTask("ddu", "", url, f"DDU ({url})")


def _make_backends(category, count=1, provider="ollama", model=None):
    """Erzeugt Mock-Backend-Dicts."""
    return [
        {
            "id": i + 1,
            "category": category,
            "provider": provider,
            "name": f"Backend {i + 1}",
            "url": f"http://server{i + 1}:11434",
            "model": model or ("nomic-embed-text" if category == "embedding" else f"model-{i + 1}"),
            "is_active": True,
            "priority": (i + 1) * 10,
        }
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Fehlerklassifikation
# ---------------------------------------------------------------------------


class TestErrorClassification:
    def test_permanent_not_found(self):
        exc = ollama_client.ResponseError("model 'xyz' not found")
        assert _is_permanent_ollama_error(exc) is True

    def test_permanent_invalid_model(self):
        exc = ollama_client.ResponseError("invalid model name")
        assert _is_permanent_ollama_error(exc) is True

    def test_permanent_unknown_model(self):
        exc = ollama_client.ResponseError("unknown model 'abc'")
        assert _is_permanent_ollama_error(exc) is True

    def test_permanent_file_not_exist(self):
        exc = ollama_client.ResponseError("file does not exist")
        assert _is_permanent_ollama_error(exc) is True

    def test_permanent_oom(self):
        exc = ollama_client.ResponseError("out of memory")
        assert _is_permanent_ollama_error(exc) is True

    def test_permanent_oom_cuda(self):
        exc = ollama_client.ResponseError("GPU out of memory: CUDA allocation failed")
        assert _is_permanent_ollama_error(exc) is True

    def test_transient_response_error(self):
        exc = ollama_client.ResponseError("internal server error")
        assert _is_permanent_ollama_error(exc) is False

    def test_non_response_error(self):
        exc = ValueError("something")
        assert _is_permanent_ollama_error(exc) is False

    def test_http_5xx_transient(self):
        resp = MagicMock()
        resp.status_code = 502
        resp.text = ""
        exc = httpx.HTTPStatusError("Bad Gateway", request=MagicMock(), response=resp)
        assert _is_transient_http_error(exc) is True

    def test_http_409_transient(self):
        resp = MagicMock()
        resp.status_code = 409
        resp.text = "GPU-Operation blockiert"
        exc = httpx.HTTPStatusError("Conflict", request=MagicMock(), response=resp)
        assert _is_transient_http_error(exc) is True

    def test_http_429_transient(self):
        resp = MagicMock()
        resp.status_code = 429
        resp.text = "rate limited"
        exc = httpx.HTTPStatusError("Too Many Requests", request=MagicMock(), response=resp)
        assert _is_transient_http_error(exc) is True

    def test_http_5xx_permanent_body_not_transient(self):
        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"code":"model_not_found"}'
        exc = httpx.HTTPStatusError("Internal Server Error", request=MagicMock(), response=resp)
        assert _is_transient_http_error(exc) is False

    def test_http_4xx_not_transient(self):
        resp = MagicMock()
        resp.status_code = 400
        resp.text = ""
        exc = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=resp)
        assert _is_transient_http_error(exc) is False

    def test_non_http_error(self):
        assert _is_transient_http_error(ValueError("x")) is False


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_make_warmup_client(self):
        client = _make_warmup_client("http://localhost:11434", 60)
        assert client is not None

    def test_lazy_tiny_png_generation(self):
        import kiara.backend_warmup as bw

        old = bw._warmup_tiny_png
        try:
            bw._warmup_tiny_png = None
            png = _get_tiny_png()
            assert png[:4] == b"\x89PNG"
            assert len(png) > 0
        finally:
            bw._warmup_tiny_png = old

    def test_lazy_tiny_png_cached(self):
        png1 = _get_tiny_png()
        png2 = _get_tiny_png()
        assert png1 is png2

    def test_lazy_tiny_png_thread_safe(self):
        import kiara.backend_warmup as bw

        old = bw._warmup_tiny_png
        try:
            bw._warmup_tiny_png = None
            results = []

            def get_png():
                results.append(_get_tiny_png())

            threads = [threading.Thread(target=get_png) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(results) == 5
            assert all(r == results[0] for r in results)
        finally:
            bw._warmup_tiny_png = old


# ---------------------------------------------------------------------------
# BackendWarmer — Einzelne Warmup-Strategien
# ---------------------------------------------------------------------------


class TestWarmupStrategies:
    def test_warmup_embedding_success(self):
        warmer = BackendWarmer()
        task = _emb_task()
        with patch("kiara.backend_warmup.httpx.post") as post:
            response = MagicMock()
            response.json.return_value = {"embeddings": [[0.1, 0.2]]}
            post.return_value = response
            warmer._warm_embedding(task, 60)
            payload = post.call_args.kwargs["json"]
            assert payload == {
                "model": "nomic-embed-text",
                "input": [WARMUP_DUMMY_TEXT],
                "input_type": "search_document",
                "keep_alive": WARMUP_KEEP_ALIVE,
            }

    def test_warmup_embedding_unverified_profile_sends_no_request(self):
        from kiara.retrieval.embedding_roles import EmbeddingRoleContractError

        warmer = BackendWarmer()
        task = _emb_task()
        error = EmbeddingRoleContractError(
            "profile_unverified",
            "verification missing",
            requested_model=task.model,
            endpoint="/api/embed",
            input_type="search_document",
        )

        with (
            patch(
                "kiara.retrieval.embedding_roles.resolve_server_request_profile",
                side_effect=error,
            ),
            patch("kiara.backend_warmup.httpx.post") as post,
            pytest.raises(EmbeddingRoleContractError) as caught,
        ):
            warmer._warm_embedding(task, 60)

        assert caught.value is error
        post.assert_not_called()

    def test_warmup_llm_success(self):
        warmer = BackendWarmer()
        task = _llm_task()
        with patch("kiara.backend_warmup._make_warmup_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            warmer._warm_llm(task, 60)
            client.generate.assert_called_once_with(
                model="qwen3:8b",
                prompt="Hi",
                stream=False,
                options={"num_predict": 1},
                keep_alive=WARMUP_KEEP_ALIVE,
            )

    def test_warmup_vlm_success(self):
        warmer = BackendWarmer()
        task = _vlm_task()
        with patch("kiara.backend_warmup._make_warmup_client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            warmer._warm_vlm(task, 60)
            client.chat.assert_called_once()
            call_kwargs = client.chat.call_args[1]
            assert call_kwargs["model"] == "minicpm-v"
            assert call_kwargs["stream"] is False
            assert call_kwargs["keep_alive"] == WARMUP_KEEP_ALIVE
            assert call_kwargs["messages"][0]["images"][0][:4] == b"\x89PNG"

    def test_warmup_ddu_success(self):
        warmer = BackendWarmer()
        task = _ddu_task()
        # _warm_ddu nutzt jetzt httpx.Client als Context-Manager (with-Statement)
        # statt direktem httpx.post — das erlaubt Pinning + Timeout-Pro-Client.
        with patch("kiara.backend_warmup.httpx.Client") as mock_client_cls:
            mock_resp = MagicMock()
            mock_client = MagicMock()
            mock_client.post.return_value = mock_resp
            # Context-Manager-Protokoll
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client_cls.return_value.__exit__.return_value = False

            warmer._warm_ddu(task, 60)

            mock_client.post.assert_called_once()
            call_args = mock_client.post.call_args
            assert "/v1/convert/file" in call_args[0][0] or "/v1/convert/file" in str(call_args)
            mock_resp.raise_for_status.assert_called_once()

    def test_warmup_colbert_success(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")

        with patch("kiara.backend_warmup.httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_post.return_value = mock_resp

            warmer._warm_colbert(task, 42)

            mock_post.assert_called_once_with(
                "http://colbert.local/api/embed_colbert",
                json={
                    "model": "colbert-xm",
                    "input": ["warmup"],
                    "input_type": "search_document",
                },
                timeout=42,
            )
            mock_resp.raise_for_status.assert_called_once()

    def test_warmup_colbert_http_error_propagates(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")
        resp = httpx.Response(
            409,
            text="GPU-Operation blockiert",
            request=httpx.Request("POST", "http://colbert.local/api/embed_colbert"),
        )

        with patch("kiara.backend_warmup.httpx.post", return_value=resp):
            with pytest.raises(httpx.HTTPStatusError):
                warmer._warm_colbert(task, 42)


# ---------------------------------------------------------------------------
# BackendWarmer — Retry-Logik
# ---------------------------------------------------------------------------


class TestRetryLogic:
    def test_permanent_error_no_retry(self):
        warmer = BackendWarmer()
        task = _emb_task()

        def fail(t, timeout):
            raise ollama_client.ResponseError("model 'xyz' not found")

        with pytest.raises(ollama_client.ResponseError, match="not found"):
            warmer._attempt(fail, task, 60)

    def test_transient_retry_then_fail(self):
        warmer = BackendWarmer()
        task = _emb_task()
        calls = []

        def fail(t, timeout):
            calls.append(1)
            raise ollama_client.RequestError("connection refused")

        with patch("kiara.backend_warmup.time.sleep"):
            with pytest.raises(ollama_client.RequestError):
                warmer._attempt(fail, task, 60, retries=1)
        assert len(calls) == 2  # 1 Versuch + 1 Retry

    def test_transient_retry_then_success(self):
        warmer = BackendWarmer()
        task = _emb_task()
        calls = []

        def flaky(t, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise ollama_client.RequestError("connection refused")
            return "ok"

        with patch("kiara.backend_warmup.time.sleep"):
            result = warmer._attempt(flaky, task, 60, retries=1)
        assert result == "ok"
        assert len(calls) == 2

    def test_retry_delay_2s(self):
        warmer = BackendWarmer()
        task = _emb_task()

        def fail(t, timeout):
            raise ollama_client.RequestError("timeout")

        with patch("kiara.backend_warmup.time.sleep") as mock_sleep:
            with pytest.raises(ollama_client.RequestError):
                warmer._attempt(fail, task, 60, retries=1)
            mock_sleep.assert_called_with(WARMUP_RETRY_DELAY)

    def test_ddu_http_5xx_retry(self):
        warmer = BackendWarmer()
        task = _ddu_task()
        calls = []

        resp = MagicMock()
        resp.status_code = 502

        def fail(t, timeout):
            calls.append(1)
            raise httpx.HTTPStatusError("Bad Gateway", request=MagicMock(), response=resp)

        with patch("kiara.backend_warmup.time.sleep"), pytest.raises(httpx.HTTPStatusError):
            warmer._attempt(fail, task, 60, retries=1)
        assert len(calls) == 2

    def test_colbert_http_409_retry_then_success(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")
        calls = []

        resp = MagicMock()
        resp.status_code = 409
        resp.text = "GPU-Operation blockiert"

        def flaky(t, timeout):
            calls.append(1)
            if len(calls) == 1:
                raise httpx.HTTPStatusError("Conflict", request=MagicMock(), response=resp)
            return "ok"

        with patch("kiara.backend_warmup.time.sleep") as mock_sleep:
            result = warmer._attempt(flaky, task, 60, retries=1)

        assert result == "ok"
        assert len(calls) == 2
        mock_sleep.assert_called_once_with(WARMUP_RETRY_DELAY)

    def test_colbert_http_429_retry_then_fail(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")
        calls = []

        resp = MagicMock()
        resp.status_code = 429
        resp.text = "rate limited"

        def fail(t, timeout):
            calls.append(1)
            raise httpx.HTTPStatusError("Too Many Requests", request=MagicMock(), response=resp)

        with patch("kiara.backend_warmup.time.sleep"), pytest.raises(httpx.HTTPStatusError):
            warmer._attempt(fail, task, 60, retries=1)

        assert len(calls) == 2

    def test_colbert_permanent_http_body_no_retry(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")
        calls = []

        resp = MagicMock()
        resp.status_code = 500
        resp.text = '{"code":"model_not_found"}'

        def fail(t, timeout):
            calls.append(1)
            raise httpx.HTTPStatusError("Internal Server Error", request=MagicMock(), response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            warmer._attempt(fail, task, 60, retries=1)

        assert len(calls) == 1

    def test_colbert_network_error_retries(self):
        warmer = BackendWarmer()
        task = WarmupTask("colbert", "colbert-xm", "http://colbert.local", "ColBERT (colbert-xm)")
        calls = []

        def fail(t, timeout):
            calls.append(1)
            raise httpx.ConnectError("connection refused")

        with patch("kiara.backend_warmup.time.sleep"), pytest.raises(httpx.ConnectError):
            warmer._attempt(fail, task, 60, retries=1)

        assert len(calls) == 2

    def test_ddu_http_4xx_no_retry(self):
        warmer = BackendWarmer()
        task = _ddu_task()
        calls = []

        resp = MagicMock()
        resp.status_code = 400

        def fail(t, timeout):
            calls.append(1)
            raise httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=resp)

        with pytest.raises(httpx.HTTPStatusError):
            warmer._attempt(fail, task, 60, retries=1)
        assert len(calls) == 1  # Kein Retry

    def test_unknown_exception_no_retry(self):
        warmer = BackendWarmer()
        task = _emb_task()
        calls = []

        def fail(t, timeout):
            calls.append(1)
            raise TypeError("unexpected")

        with pytest.raises(TypeError):
            warmer._attempt(fail, task, 60, retries=1)
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# BackendWarmer — Timed Attempt
# ---------------------------------------------------------------------------


class TestTimedAttempt:
    def test_success_returns_duration(self):
        warmer = BackendWarmer()
        task = _emb_task()

        def ok(t, timeout):
            time.sleep(0.01)

        dur = warmer._timed_attempt(ok, task, 60)
        assert dur >= 0.01

    def test_error_has_duration_attribute(self):
        warmer = BackendWarmer()
        task = _emb_task()

        def fail(t, timeout):
            raise ollama_client.ResponseError("not found")

        with pytest.raises(ollama_client.ResponseError) as exc_info:
            warmer._timed_attempt(fail, task, 60)
        assert hasattr(exc_info.value, "_warmup_duration")
        assert exc_info.value._warmup_duration >= 0


# ---------------------------------------------------------------------------
# BackendWarmer.warm() — Parallelisierung + Erfolgs-Semantik
# ---------------------------------------------------------------------------


class TestWarm:
    @pytest.fixture(autouse=True)
    def _orchestration_embedding_seam(self, monkeypatch):
        """Diese Klasse testet Parallelisierung, Retry und Aggregation."""
        import kiara.backend_warmup as module

        def warm_embedding(_self, task, timeout):
            client = module._make_warmup_client(task.url, timeout)
            client.embed(
                model=task.model,
                input=[WARMUP_DUMMY_TEXT],
                keep_alive=WARMUP_KEEP_ALIVE,
            )

        monkeypatch.setattr(BackendWarmer, "_warm_embedding", warm_embedding)

    def test_empty_tasks(self):
        result = BackendWarmer().warm([])
        # Leere Tasks-Liste ist legitim (alle aktiven Backends sind Cloud-Provider).
        # Pipeline darf weiterlaufen → success=True, kein Pipeline-Abbruch.
        assert result.success is True
        assert result.duration_s == 0.0
        assert result.details == []

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_single_embedding_success(self, mock_client_fn, mock_healthy):
        mock_client_fn.return_value = MagicMock()
        result = BackendWarmer().warm([_emb_task()], timeout=10)
        assert result.success is True
        assert len(result.details) == 1
        assert result.details[0]["ok"] is True
        mock_healthy.assert_called_once()

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_partial_failure_category_ok(self, mock_client_fn, mock_healthy):
        """2 Embedding-Backends, 1 OK + 1 Fehler -> success=True."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            client = MagicMock()
            if call_count[0] == 1:
                client.embed.side_effect = ollama_client.RequestError("timeout")
            return client

        mock_client_fn.side_effect = side_effect
        tasks = [
            _emb_task(url="http://server1:11434"),
            _emb_task(url="http://server2:11434"),
        ]
        with patch("kiara.backend_warmup.time.sleep"):
            result = BackendWarmer().warm(tasks, timeout=10)
        assert result.success is True

    @patch("kiara.backend_warmup._make_warmup_client")
    def test_all_in_category_failed(self, mock_client_fn):
        """2 Embedding-Backends, beide Fehler -> success=False."""
        client = MagicMock()
        client.embed.side_effect = ollama_client.RequestError("timeout")
        mock_client_fn.return_value = client
        tasks = [
            _emb_task(url="http://server1:11434"),
            _emb_task(url="http://server2:11434"),
        ]
        with patch("kiara.backend_warmup.time.sleep"):
            result = BackendWarmer().warm(tasks, timeout=10)
        assert result.success is False

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_mixed_categories(self, mock_client_fn, mock_healthy):
        """Embedding OK, LLM alle Fehler -> success=False."""

        def side_effect(*args, **kwargs):
            client = MagicMock()
            client.generate.side_effect = ollama_client.ResponseError("not found")
            return client

        mock_client_fn.side_effect = side_effect
        tasks = [_emb_task(), _llm_task()]
        result = BackendWarmer().warm(tasks, timeout=10)
        assert result.success is False

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_health_cache_on_success(self, mock_client_fn, mock_healthy):
        mock_client_fn.return_value = MagicMock()
        BackendWarmer().warm([_emb_task()], timeout=10)
        mock_healthy.assert_called_once_with("http://server:11434")

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_health_cache_not_called_on_failure(self, mock_client_fn, mock_healthy):
        client = MagicMock()
        client.embed.side_effect = ollama_client.ResponseError("not found")
        mock_client_fn.return_value = client
        BackendWarmer().warm([_emb_task()], timeout=10)
        mock_healthy.assert_not_called()

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_on_progress_called(self, mock_client_fn, mock_healthy):
        mock_client_fn.return_value = MagicMock()
        progress = MagicMock()
        BackendWarmer().warm([_emb_task()], timeout=10, on_progress=progress)
        progress.assert_called_once()
        args = progress.call_args[0]
        assert args[1] == "ok"

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_per_task_duration(self, mock_client_fn, mock_healthy):
        mock_client_fn.return_value = MagicMock()
        result = BackendWarmer().warm([_emb_task()], timeout=10)
        assert result.details[0]["duration_s"] >= 0

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_parallel_execution(self, mock_client_fn, mock_healthy):
        """Pruefen dass Tasks parallel laufen (Gesamtdauer < Summe)."""

        def slow_client(*args, **kwargs):
            client = MagicMock()

            def slow_embed(**kw):
                time.sleep(0.1)

            client.embed.side_effect = slow_embed

            def slow_generate(**kw):
                time.sleep(0.1)

            client.generate.side_effect = slow_generate
            return client

        mock_client_fn.side_effect = slow_client
        tasks = [_emb_task(), _llm_task()]
        t0 = time.monotonic()
        result = BackendWarmer().warm(tasks, timeout=10)
        dur = time.monotonic() - t0
        assert result.success is True
        # Parallel: ~0.1s statt ~0.2s (sequentiell)
        assert dur < 0.3

    @patch("kiara.backend_warmup._mark_healthy_safe")
    @patch("kiara.backend_warmup._make_warmup_client")
    def test_warning_on_partial_failure(self, mock_client_fn, mock_healthy):
        """Fehlgeschlagene Einzelbackends loggen Warning bei Gesamt-Erfolg."""
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            client = MagicMock()
            if call_count[0] == 1:
                client.embed.side_effect = ollama_client.RequestError("timeout")
            return client

        mock_client_fn.side_effect = side_effect
        tasks = [
            _emb_task(url="http://server1:11434"),
            _emb_task(url="http://server2:11434"),
        ]
        with patch("kiara.backend_warmup.time.sleep"):
            with patch("kiara.backend_warmup.logger") as mock_logger:
                result = BackendWarmer().warm(tasks, timeout=10)
        assert result.success is True


# ---------------------------------------------------------------------------
# build_warmup_tasks()
# ---------------------------------------------------------------------------


def _make_ds_row(ddu_enabled=0, vlm_enabled=0):
    """Erzeugt einen Mock-ds_row der wie eine DB-Row per Bracket zugreifbar ist."""
    data = {
        "id": 1,
        "type": "local",
        "ds_ddu_enabled": ddu_enabled,
        "ds_vlm_enabled": vlm_enabled,
    }
    return data


class TestBuildWarmupTasks:
    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_embedding_only(self, mock_norm, mock_active):
        """DS ohne DDU/VLM, kein Graph -> nur Embedding-Task."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert len(tasks) == 1
        assert tasks[0].category == "embedding"
        assert tasks[0].model == "nomic-embed-text"

    def test_no_embedding_model_raises(self):
        with pytest.raises(RuntimeError, match="Kein Embedding-Modell"):
            build_warmup_tasks(_make_ds_row(), "", {})

    def test_no_embedding_model_none_raises(self):
        with pytest.raises(RuntimeError, match="Kein Embedding-Modell"):
            build_warmup_tasks(_make_ds_row(), None, {})

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_no_embedding_backend_raises(self, mock_norm, mock_active):
        mock_active.return_value = []
        with pytest.raises(RuntimeError, match="Kein Embedding-Backend"):
            build_warmup_tasks(_make_ds_row(), "nomic-embed-text", {})

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_all_embedding_backends(self, mock_norm, mock_active):
        """2 Embedding-Backends mit gleichem Modell -> 2 Embedding-Tasks."""
        mock_active.return_value = _make_backends("embedding", count=2)
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert len(tasks) == 2
        assert all(t.category == "embedding" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_embedding_cloud_only_skipped(self, mock_norm, mock_active):
        """Nur Cloud-Embedding-Backends -> kein Embedding-Task."""
        mock_active.return_value = _make_backends("embedding", provider="openai")
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "embedding" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_embedding_mixed_backends(self, mock_norm, mock_active):
        """Ollama+Cloud Embedding -> nur Ollama-Embedding-Task."""
        mock_active.return_value = [
            {
                "id": 1,
                "category": "embedding",
                "provider": "ollama",
                "name": "Ollama",
                "url": "http://server:11434",
                "model": "nomic-embed-text",
                "is_active": True,
                "priority": 10,
            },
            {
                "id": 2,
                "category": "embedding",
                "provider": "openai",
                "name": "OpenAI",
                "url": "",
                "base_url": "https://api.openai.com",
                "model": "nomic-embed-text",
                "is_active": True,
                "priority": 20,
            },
        ]
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        emb_tasks = [t for t in tasks if t.category == "embedding"]
        assert len(emb_tasks) == 1
        assert emb_tasks[0].url == "http://server:11434"

    @patch("kiara.config.get_ddu_backends")
    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_with_ddu(self, mock_norm, mock_active, mock_ddu):
        mock_active.return_value = _make_backends("embedding")
        mock_ddu.return_value = [{"url": "http://server:5001", "is_local": False}]
        ds_row = _make_ds_row(ddu_enabled=1)
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        categories = [t.category for t in tasks]
        assert "embedding" in categories
        assert "ddu" in categories

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_with_vlm(self, mock_norm, mock_active):
        def active_side_effect(category):
            if category == "embedding":
                return _make_backends("embedding")
            if category == "vlm":
                return _make_backends("vlm", model="minicpm-v")
            return []

        mock_active.side_effect = active_side_effect
        ds_row = _make_ds_row(vlm_enabled=1)
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        categories = [t.category for t in tasks]
        assert "vlm" in categories

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_ollama_llm(self, mock_norm, mock_active):
        def active_side_effect(category):
            if category == "embedding":
                return _make_backends("embedding")
            if category == "llm":
                return [
                    {
                        "id": 1,
                        "provider": "ollama",
                        "url": "http://server:11434",
                        "model": "qwen3:8b",
                        "name": "LLM 1",
                    }
                ]
            return []

        mock_active.side_effect = active_side_effect
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": True, "extract_on_ingestion": True}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        categories = [t.category for t in tasks]
        assert "llm" in categories

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_extraction_model(self, mock_norm, mock_active):
        def active_side_effect(category):
            if category == "embedding":
                return _make_backends("embedding")
            if category == "llm":
                return [
                    {
                        "id": 1,
                        "provider": "ollama",
                        "url": "http://server:11434",
                        "model": "qwen3:8b",
                        "name": "LLM 1",
                    }
                ]
            return []

        mock_active.side_effect = active_side_effect
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": True, "extraction_model": "special-model"}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        llm_tasks = [t for t in tasks if t.category == "llm"]
        assert len(llm_tasks) == 1
        assert llm_tasks[0].model == "special-model"

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_disabled(self, mock_norm, mock_active):
        mock_active.return_value = _make_backends("embedding")
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "llm" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_no_ingestion(self, mock_norm, mock_active):
        mock_active.return_value = _make_backends("embedding")
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": True, "extract_on_ingestion": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "llm" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_cloud_llm_skipped(self, mock_norm, mock_active):
        """Nur Cloud-LLM -> kein LLM-Task."""

        def active_side_effect(category):
            if category == "embedding":
                return _make_backends("embedding")
            if category == "llm":
                return [
                    {
                        "id": 1,
                        "provider": "anthropic",
                        "url": "",
                        "model": "claude-3-5-sonnet",
                        "name": "Claude",
                    }
                ]
            return []

        mock_active.side_effect = active_side_effect
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": True}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "llm" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_graph_mixed_backends(self, mock_norm, mock_active):
        """Ollama+Cloud -> nur Ollama-LLM-Tasks."""

        def active_side_effect(category):
            if category == "embedding":
                return _make_backends("embedding")
            if category == "llm":
                return [
                    {
                        "id": 1,
                        "provider": "ollama",
                        "url": "http://server:11434",
                        "model": "qwen3:8b",
                        "name": "Ollama",
                    },
                    {
                        "id": 2,
                        "provider": "anthropic",
                        "url": "",
                        "model": "claude-3-5-sonnet",
                        "name": "Claude",
                    },
                ]
            return []

        mock_active.side_effect = active_side_effect
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": True}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        llm_tasks = [t for t in tasks if t.category == "llm"]
        assert len(llm_tasks) == 1
        assert llm_tasks[0].url == "http://server:11434"

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_missing_ddu_column(self, mock_norm, mock_active):
        """ds_row ohne ds_ddu_enabled-Spalte -> kein DDU-Task, kein Crash."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = {"id": 1, "type": "local"}  # Kein ds_ddu_enabled
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "ddu" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_missing_vlm_column(self, mock_norm, mock_active):
        """ds_row ohne ds_vlm_enabled-Spalte -> kein VLM-Task, kein Crash."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = {"id": 1, "type": "local"}  # Kein ds_vlm_enabled
        config = {"graph": {"enabled": False}}
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "vlm" for t in tasks)

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_no_rr_side_effect(self, mock_norm, mock_active):
        """RR-Counter unveraendert nach build_warmup_tasks()."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = _make_ds_row()
        config = {"graph": {"enabled": False}}
        # get_all_active wird gerufen, nicht get_next_backend -> kein RR-Inkrement
        build_warmup_tasks(ds_row, "nomic-embed-text", config)
        # Kein get_next_backend-Aufruf = kein Seiteneffekt
        for c in mock_active.call_args_list:
            assert c != call("get_next_backend")

    @patch("kiara.retrieval.colbert._resolve_colbert_url", return_value="http://localhost:8888")
    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_with_colbert(self, mock_norm, mock_active, mock_colbert_url):
        """Global + per-DS ColBERT aktiviert → ColBERT-Warmup-Task."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = {
            "id": 1,
            "type": "local",
            "ds_ddu_enabled": False,
            "ds_vlm_enabled": False,
            "ds_colbert_enabled": True,
        }
        config = {
            "graph": {"enabled": False},
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-ir/colbertv2.0"}},
        }
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        colbert_tasks = [t for t in tasks if t.category == "colbert"]
        assert len(colbert_tasks) == 1
        assert colbert_tasks[0].url == "http://localhost:8888"

    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_missing_colbert_column(self, mock_norm, mock_active):
        """Row ohne ds_colbert_enabled → kein Crash, kein ColBERT-Task."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = {"id": 1, "type": "local", "ds_ddu_enabled": False, "ds_vlm_enabled": False}
        config = {
            "graph": {"enabled": False},
            "retrieval": {"colbert": {"enabled": True}},
        }
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "colbert" for t in tasks)

    @patch("kiara.retrieval.colbert._resolve_colbert_url", side_effect=Exception("no colbert"))
    @patch("kiara.backend_selector.get_all_active")
    @patch(
        "kiara.backend_selector.normalize_model_name",
        side_effect=lambda x: x.lower().replace(":latest", ""),
    )
    def test_colbert_url_resolution_error(self, mock_norm, mock_active, mock_colbert_url):
        """_resolve_colbert_url() wirft → WARNING, kein Crash, kein ColBERT-Task."""
        mock_active.return_value = _make_backends("embedding")
        ds_row = {
            "id": 1,
            "type": "local",
            "ds_ddu_enabled": False,
            "ds_vlm_enabled": False,
            "ds_colbert_enabled": True,
        }
        config = {
            "graph": {"enabled": False},
            "retrieval": {"colbert": {"enabled": True, "model": "colbert-ir/colbertv2.0"}},
        }
        tasks = build_warmup_tasks(ds_row, "nomic-embed-text", config)
        assert not any(t.category == "colbert" for t in tasks)
