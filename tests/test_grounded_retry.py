"""Tests fuer NLI-Grounding (kiara.generation.grounding) und den Abstain-Retry."""

from unittest.mock import MagicMock, patch

import httpx

from kiara.generation import grounding
from kiara.generation.grounding import check_entailment, resolve_grounding_config
from kiara.shared.constants import ABSTAIN_MESSAGE_DEFAULT

PROMPT_TEMPLATE = "Kontext:\n{context}\n\nFrage: {question}"


def _config(**over) -> dict:
    retrieval = {
        "reranking": True,
        "rerank_backend": "kiron",
        "rerank_url": "http://kiron.test:11434",
        "rerank_top_k": 5,
        "abstain_instruction": True,
        "grounding_retry": True,
        "grounding_min_entailment": 0.5,
    }
    retrieval.update(over)
    return {"ollama": {"url": "http://x", "llm_model": "test"}, "retrieval": retrieval}


def _chunks(n=3):
    return [
        {
            "text": f"Chunk {i} mit Inhalt.",
            "metadata": {"source": f"/d{i}.txt"},
            "rerank_score": 0.5,
        }
        for i in range(n)
    ]


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("kein JSON")
        return self._payload


def _nli_payload(*entailments):
    return {
        "model": "mdeberta-v3-xnli",
        "results": [
            {"entailment": e, "neutral": 1 - e - 0.01, "contradiction": 0.01} for e in entailments
        ],
    }


class TestResolveGroundingConfig:
    def test_url_falls_back_to_rerank_url(self):
        settings = resolve_grounding_config(_config(grounding_url=""))
        assert settings["url"] == "http://kiron.test:11434"

    def test_explicit_url_wins(self):
        settings = resolve_grounding_config(_config(grounding_url="http://anders:1/"))
        assert settings["url"] == "http://anders:1"


class TestCheckEntailment:
    def test_success_max_entailment(self):
        with patch.object(
            grounding.httpx,
            "post",
            return_value=FakeResponse(payload=_nli_payload(0.2, 0.85, 0.1)),
        ) as post:
            result = check_entailment(_config(), "Antwort.", _chunks())
        assert result["status"] == "success"
        assert result["max_entailment"] == 0.85
        assert post.call_args.kwargs["json"]["model"] == "mdeberta-v3-xnli"
        assert len(post.call_args.kwargs["json"]["pairs"]) == 3

    def test_http_error_returns_error(self):
        with patch.object(
            grounding.httpx, "post", return_value=FakeResponse(status_code=503, text="gate")
        ):
            result = check_entailment(_config(), "Antwort.", _chunks())
        assert result["status"] == "error"

    def test_transport_error_returns_error(self):
        with patch.object(grounding.httpx, "post", side_effect=httpx.ConnectError("nein")):
            result = check_entailment(_config(), "Antwort.", _chunks())
        assert result["status"] == "error"

    def test_no_url_skips(self):
        cfg = _config(rerank_url="", grounding_url="")
        result = check_entailment(cfg, "Antwort.", _chunks())
        assert result["status"] == "skipped"


class TestGroundedRetryFlow:
    def _make_chain(self, config, answers, results):
        from kiara.generation.chain import RAGChain

        with (
            patch("kiara.generation.chain.HybridRetriever") as MockRetriever,
            patch("kiara.retrieval.reranker.KironReranker") as MockReranker,
        ):
            mock_reranker = MockReranker.return_value
            mock_reranker.enabled = True
            mock_reranker.rerank.return_value = results
            MockRetriever.return_value.retrieve.return_value = results
            backend = MagicMock()
            backend.provider = "ollama"
            backend.generate.side_effect = answers
            chain = RAGChain(config, backend=backend, prompt_template_override=PROMPT_TEMPLATE)
            return chain, backend

    def test_retry_accepted_when_grounded(self):
        chain, backend = self._make_chain(
            _config(), [ABSTAIN_MESSAGE_DEFAULT, "Die Antwort ist 42."], _chunks()
        )
        with patch.object(
            grounding.httpx, "post", return_value=FakeResponse(payload=_nli_payload(0.9, 0.1, 0.1))
        ):
            result = chain.query("Frage?", k=3)
        assert result["answer"] == "Die Antwort ist 42."
        assert result["grounded_retry"]["accepted"] is True
        assert backend.generate.call_count == 2
        # Zweitprompt ohne Abstain-Instruktion
        second_user_msg = backend.generate.call_args_list[1][0][1]
        assert "Erfinde keine Fakten" not in second_user_msg

    def test_retry_rejected_below_threshold_keeps_abstain(self):
        chain, backend = self._make_chain(
            _config(), [ABSTAIN_MESSAGE_DEFAULT, "Die Antwort ist 42."], _chunks()
        )
        with patch.object(
            grounding.httpx, "post", return_value=FakeResponse(payload=_nli_payload(0.2, 0.3, 0.1))
        ):
            result = chain.query("Frage?", k=3)
        assert result["answer"] == ABSTAIN_MESSAGE_DEFAULT
        assert result["grounded_retry"]["accepted"] is False

    def test_retry_rejected_on_nli_error_fail_closed(self):
        chain, _ = self._make_chain(
            _config(), [ABSTAIN_MESSAGE_DEFAULT, "Die Antwort ist 42."], _chunks()
        )
        with patch.object(grounding.httpx, "post", side_effect=httpx.ConnectError("weg")):
            result = chain.query("Frage?", k=3)
        assert result["answer"] == ABSTAIN_MESSAGE_DEFAULT
        assert result["grounded_retry"]["accepted"] is False

    def test_no_retry_when_disabled(self):
        chain, backend = self._make_chain(
            _config(grounding_retry=False), [ABSTAIN_MESSAGE_DEFAULT], _chunks()
        )
        result = chain.query("Frage?", k=3)
        assert result["answer"] == ABSTAIN_MESSAGE_DEFAULT
        assert "grounded_retry" not in result
        assert backend.generate.call_count == 1

    def test_no_retry_on_normal_answer(self):
        chain, backend = self._make_chain(_config(), ["Normale Antwort."], _chunks())
        result = chain.query("Frage?", k=3)
        assert result["answer"] == "Normale Antwort."
        assert backend.generate.call_count == 1

    def test_retry_that_abstains_again_keeps_abstain(self):
        chain, backend = self._make_chain(
            _config(), [ABSTAIN_MESSAGE_DEFAULT, "Ich weiss es nicht."], _chunks()
        )
        result = chain.query("Frage?", k=3)
        assert result["answer"] == ABSTAIN_MESSAGE_DEFAULT
        assert result["grounded_retry"]["retry_abstained"] is True
        assert backend.generate.call_count == 2
