"""Tests fuer KironReranker und create_reranker (kiara.retrieval.reranker)."""

import json
import logging
import sys
import types
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kiara.retrieval.reranker import (
    CrossEncoderReranker,
    KironReranker,
    RerankBackendError,
    create_reranker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kiron_config(**overrides) -> dict:
    retrieval = {
        "reranking": True,
        "rerank_backend": "kiron",
        "rerank_url": "http://kiron.test:11434",
        "rerank_model": "bge-reranker-v2-m3",
        "rerank_top_k": 3,
        "rerank_fail_mode": "degrade",
        "rerank_timeout_s": 5,
    }
    retrieval.update(overrides)
    return {"retrieval": retrieval}


def make_results(n: int) -> list[dict]:
    return [
        {"text": f"Dokument {i}", "metadata": {"source": f"/f{i}.txt"}, "score": 1.0 - i * 0.1}
        for i in range(n)
    ]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        if text is not None:
            self.text = text
        else:
            self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("kein JSON")
        return self._payload


def service_payload(scores: dict[int, float], documents: list[str] | None = None) -> dict:
    """Baut eine Kiron-Rerank-Antwort: results nach Score absteigend sortiert."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "model": "bge-reranker-v2-m3",
        "results": [
            {
                "index": idx,
                "score": score,
                "document": documents[idx] if documents else f"Dokument {idx}",
            }
            for idx, score in ranked
        ],
    }


# ---------------------------------------------------------------------------
# Konstruktor / enabled
# ---------------------------------------------------------------------------


class TestKironRerankerSetup:
    def test_disabled_when_reranking_false(self):
        rr = KironReranker(kiron_config(reranking=False))
        assert rr.enabled is False

    def test_disabled_when_url_missing(self, caplog):
        rr = KironReranker(kiron_config(rerank_url=""))
        assert rr.enabled is False

    def test_disabled_returns_top_k_without_http(self):
        rr = KironReranker(kiron_config(reranking=False))
        results = make_results(5)
        with patch("kiara.retrieval.reranker.httpx.post") as post:
            out = rr.rerank("frage", results)
        post.assert_not_called()
        assert out == results[:3]

    def test_trailing_slash_stripped(self):
        rr = KironReranker(kiron_config(rerank_url="http://kiron.test:11434/"))
        assert rr._base_url == "http://kiron.test:11434"


# ---------------------------------------------------------------------------
# Erfolgspfad
# ---------------------------------------------------------------------------


class TestKironRerankerSuccess:
    def test_scores_sorted_and_truncated(self):
        rr = KironReranker(kiron_config())
        results = make_results(4)
        payload = service_payload({0: -2.0, 1: 5.0, 2: 1.5, 3: -7.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ) as post:
            out = rr.rerank("frage", results)

        assert [r["text"] for r in out] == ["Dokument 1", "Dokument 2", "Dokument 0"]
        assert out[0]["rerank_score"] == 5.0
        assert rr.last_diagnostics["status"] == "success"
        assert rr.last_diagnostics["scored"] == 4
        body = post.call_args.kwargs["json"]
        assert body["model"] == "bge-reranker-v2-m3"
        assert body["query"] == "frage"
        assert len(body["documents"]) == 4
        assert "top_k" not in body

    def test_top_k_override(self):
        rr = KironReranker(kiron_config())
        results = make_results(4)
        payload = service_payload({0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ):
            out = rr.rerank("frage", results, top_k=1)
        assert len(out) == 1
        assert out[0]["text"] == "Dokument 3"

    def test_non_finite_score_becomes_neg_inf(self):
        rr = KironReranker(kiron_config())
        results = make_results(2)
        payload = service_payload({0: float("inf"), 1: 2.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ):
            out = rr.rerank("frage", results)
        assert out[0]["text"] == "Dokument 1"
        assert out[1]["rerank_score"] == float("-inf")

    def test_no_text_results_skipped(self):
        rr = KironReranker(kiron_config())
        results = [{"text": "", "metadata": {}}, *make_results(2)]
        payload = service_payload({0: 1.0, 1: 2.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ) as post:
            out = rr.rerank("frage", results)
        assert len(post.call_args.kwargs["json"]["documents"]) == 2
        assert all(r["text"] for r in out)

    def test_all_results_without_text_returns_original(self):
        rr = KironReranker(kiron_config())
        results = [{"text": "", "metadata": {}} for _ in range(4)]
        with patch("kiara.retrieval.reranker.httpx.post") as post:
            out = rr.rerank("frage", results)
        post.assert_not_called()
        assert out == results[:3]
        assert rr.last_diagnostics["status"] == "skipped"

    def test_candidates_capped_to_api_limit(self):
        rr = KironReranker(kiron_config())
        results = make_results(5)
        payload = service_payload({0: 3.0, 1: 2.0, 2: 1.0})
        with (
            patch("kiara.retrieval.reranker.RERANK_REMOTE_MAX_DOCUMENTS", 3),
            patch(
                "kiara.retrieval.reranker.httpx.post",
                return_value=FakeResponse(payload=payload),
            ) as post,
        ):
            out = rr.rerank("frage", results)
        assert len(post.call_args.kwargs["json"]["documents"]) == 3
        assert [r["text"] for r in out] == ["Dokument 0", "Dokument 1", "Dokument 2"]

    def test_documents_truncated_to_char_limit(self):
        rr = KironReranker(kiron_config())
        results = [{"text": "x" * 100, "metadata": {}}, {"text": "kurz", "metadata": {}}]
        payload = service_payload({0: 1.0, 1: 2.0})
        with (
            patch("kiara.retrieval.reranker.RERANK_REMOTE_MAX_DOC_CHARS", 10),
            patch(
                "kiara.retrieval.reranker.httpx.post",
                return_value=FakeResponse(payload=payload),
            ) as post,
        ):
            rr.rerank("frage", results)
        docs = post.call_args.kwargs["json"]["documents"]
        assert docs[0] == "x" * 10
        assert docs[1] == "kurz"


# ---------------------------------------------------------------------------
# Fehlerpfade: degrade vs. error
# ---------------------------------------------------------------------------


class TestKironRerankerFailure:
    def test_http_500_degrades_to_original_order(self):
        rr = KironReranker(kiron_config())
        results = make_results(5)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            return_value=FakeResponse(status_code=500, text="kaputt"),
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert all("rerank_score" not in r for r in out)
        assert rr.last_diagnostics["status"] == "degraded"
        assert rr.last_diagnostics["fallback_reason"] == "http_500"

    def test_http_500_raises_in_error_mode(self):
        rr = KironReranker(kiron_config(rerank_fail_mode="error"))
        results = make_results(5)
        with (
            patch(
                "kiara.retrieval.reranker.httpx.post",
                return_value=FakeResponse(status_code=500, text="kaputt"),
            ),
            pytest.raises(RerankBackendError),
        ):
            rr.rerank("frage", results)
        assert rr.last_diagnostics["status"] == "error"

    def test_gpu_gate_409_degrades(self):
        rr = KironReranker(kiron_config())
        results = make_results(4)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            return_value=FakeResponse(status_code=409, text="GPU-Gate blockiert"),
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert rr.last_diagnostics["fallback_reason"] == "http_409"

    def test_timeout_degrades(self):
        rr = KironReranker(kiron_config())
        results = make_results(4)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            side_effect=httpx.ReadTimeout("timeout"),
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert rr.last_diagnostics["fallback_reason"] == "ReadTimeout"

    def test_connect_error_retried_once_then_success(self):
        rr = KironReranker(kiron_config())
        results = make_results(2)
        payload = service_payload({0: 1.0, 1: 2.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            side_effect=[httpx.ConnectError("nein"), FakeResponse(payload=payload)],
        ) as post:
            out = rr.rerank("frage", results)
        assert post.call_count == 2
        assert rr.last_diagnostics["status"] == "success"
        assert out[0]["text"] == "Dokument 1"

    def test_connect_error_twice_degrades(self):
        rr = KironReranker(kiron_config())
        results = make_results(2)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            side_effect=httpx.ConnectError("nein"),
        ) as post:
            out = rr.rerank("frage", results)
        assert post.call_count == 2
        assert out == results[:3]
        assert rr.last_diagnostics["status"] == "degraded"

    def test_malformed_response_no_partial_scores(self):
        rr = KironReranker(kiron_config())
        results = make_results(3)
        payload = {
            "model": "bge-reranker-v2-m3",
            "results": [
                {"index": 0, "score": 1.0, "document": "a"},
                {"index": 99, "score": 2.0, "document": "b"},
            ],
        }
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert all("rerank_score" not in r for r in results)
        assert rr.last_diagnostics["fallback_reason"] == "malformed_response"

    def test_missing_results_key_is_malformed(self):
        rr = KironReranker(kiron_config())
        results = make_results(2)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            return_value=FakeResponse(payload={"model": "x"}),
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert rr.last_diagnostics["fallback_reason"] == "malformed_response"

    def test_score_count_mismatch_detected(self):
        rr = KironReranker(kiron_config())
        results = make_results(3)
        payload = service_payload({0: 1.0})
        with patch(
            "kiara.retrieval.reranker.httpx.post", return_value=FakeResponse(payload=payload)
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert rr.last_diagnostics["fallback_reason"] == "score_count_mismatch"
        assert all("rerank_score" not in r for r in results)

    def test_invalid_json_body_is_malformed(self):
        rr = KironReranker(kiron_config())
        results = make_results(2)
        with patch(
            "kiara.retrieval.reranker.httpx.post",
            return_value=FakeResponse(payload=None, text="html-fehlerseite"),
        ):
            out = rr.rerank("frage", results)
        assert out == results[:3]
        assert rr.last_diagnostics["fallback_reason"] == "malformed_response"


# ---------------------------------------------------------------------------
# System-Log: echter DB-Handler mit abgefangenem INSERT
# ---------------------------------------------------------------------------


class TestKironRerankerSystemLog:
    @pytest.fixture
    def event_insert(self, monkeypatch):
        from kiara.web.logging_db_handler import SystemEventDBHandler

        monkeypatch.delenv("KIARA_SKIP_DB_LOG_HANDLER", raising=False)
        handler = SystemEventDBHandler()
        handler._min_level_cache = logging.INFO
        system_log = logging.Logger("kiara.security", logging.INFO)
        system_log.addHandler(handler)
        monkeypatch.setattr("kiara.retrieval.reranker.security_log", system_log)
        conn = MagicMock()
        with patch("kiara.shared.db.db_transaction") as transaction:
            transaction.return_value.__enter__.return_value = conn
            yield conn.execute

    @pytest.mark.parametrize("fail_mode", ["degrade", "error"])
    def test_kiron_gpu_error_is_persisted_in_full(self, event_insert, fail_mode):
        payload = {
            "error": "Nicht genuegend freier GPU-Speicher",
            "hint": "Speicher freigeben und die Anfrage erneut senden. " * 8,
            "vram_lease": "active",
            "reason": "resource_exhausted",
            "marker_kind": None,
            "lease_state": "admission",
            "detail": 'Backend-Text mit reason=anderer_grund user=fremd und "Zitat".',
        }
        body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        rr = KironReranker(kiron_config(rerank_fail_mode=fail_mode))
        results = make_results(5)
        with patch.object(rr, "_post_rerank", return_value=httpx.Response(503, text=body)):
            if fail_mode == "error":
                with pytest.raises(RerankBackendError):
                    rr.rerank("frage", results)
            else:
                assert rr.rerank("frage", results) == results[:3]

        event_insert.assert_called_once()
        sql, params = event_insert.call_args.args
        assert "INSERT INTO system_events" in sql
        assert params[:6] == (
            "ERROR" if fail_mode == "error" else "WARNING",
            "security",
            "RERANK_BACKEND_FAILED",
            "http_503",
            None,
            None,
        )
        assert payload["error"] in params[6]
        details = params[7].adapted
        assert details["backend_response"] == payload
        assert details["response_body"] == body
        assert details["http_status"] == 503
        assert details["endpoint"] == "http://kiron.test:11434/api/rerank"
        assert details["model"] == "bge-reranker-v2-m3"
        assert details["candidates_in"] == 5
        assert details["fail_mode"] == fail_mode

    def test_non_json_http_error_is_persisted_in_full(self, event_insert):
        body = "<html>Gatewayfehler\n" + "Details " * 70 + "</html>"
        rr = KironReranker(kiron_config())
        with patch.object(rr, "_post_rerank", return_value=httpx.Response(502, text=body)):
            rr.rerank("frage", make_results(2))
        event_insert.assert_called_once()
        params = event_insert.call_args.args[1]
        assert params[3] == "http_502"
        assert params[7].adapted["response_body"] == body
        assert "backend_response" not in params[7].adapted

    def test_timeout_is_persisted_without_http_response(self, event_insert):
        rr = KironReranker(kiron_config())
        with patch.object(rr, "_post_rerank", side_effect=httpx.ReadTimeout("Zeitlimit erreicht")):
            rr.rerank("frage", make_results(2))
        event_insert.assert_called_once()
        params = event_insert.call_args.args[1]
        assert params[3] == "ReadTimeout"
        assert "Zeitlimit erreicht" in params[6]
        assert "http_status" not in params[7].adapted

    def test_invalid_scores_keep_backend_response(self, event_insert):
        payload = service_payload({0: 1.0})
        rr = KironReranker(kiron_config())
        with patch.object(rr, "_post_rerank", return_value=httpx.Response(200, json=payload)):
            rr.rerank("frage", make_results(2))
        event_insert.assert_called_once()
        params = event_insert.call_args.args[1]
        assert params[3] == "score_count_mismatch"
        assert params[7].adapted["http_status"] == 200
        assert params[7].adapted["backend_response"] == payload

    def test_success_does_not_write_failure_event(self, event_insert):
        rr = KironReranker(kiron_config())
        with patch.object(
            rr, "_post_rerank", return_value=httpx.Response(200, json=service_payload({0: 1.0}))
        ):
            rr.rerank("frage", make_results(1))
        event_insert.assert_not_called()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateReranker:
    @pytest.fixture(autouse=True)
    def _patch_sentence_transformers(self):
        fake_module = types.ModuleType("sentence_transformers")
        fake_module.CrossEncoder = MagicMock()
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            yield

    def test_default_is_local(self):
        rr = create_reranker({"retrieval": {"reranking": False}})
        assert isinstance(rr, CrossEncoderReranker)

    def test_kiron_backend_selected(self):
        rr = create_reranker(kiron_config())
        assert isinstance(rr, KironReranker)

    def test_backend_value_normalized(self):
        rr = create_reranker(kiron_config(rerank_backend="  KIRON  "))
        assert isinstance(rr, KironReranker)

    def test_empty_backend_falls_back_to_local(self):
        rr = create_reranker(kiron_config(rerank_backend=""))
        assert isinstance(rr, CrossEncoderReranker)
