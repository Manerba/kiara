"""Tests fuer die Abstain-Policy Stufe 1+2 (Score-Gate + Prompt-Instruktion)."""

from unittest.mock import MagicMock, patch

from kiara.benchmarks.crag import _MISSING_PATTERNS
from kiara.shared.constants import (
    ABSTAIN_INSTRUCTION_TEXT,
    ABSTAIN_MESSAGE_DEFAULT,
)

PROMPT_TEMPLATE = "Kontext:\n{context}\n\nFrage: {question}"


def _base_config(**retrieval_overrides) -> dict:
    retrieval = {
        "reranking": True,
        "rerank_backend": "kiron",
        "rerank_url": "http://kiron.test:11434",
        "rerank_top_k": 5,
        "abstain_enabled": True,
        "abstain_min_rerank_score": 0.02,
    }
    retrieval.update(retrieval_overrides)
    return {
        "ollama": {"url": "http://localhost:11434", "llm_model": "test"},
        "retrieval": retrieval,
    }


def _make_results(scores) -> list[dict]:
    return [
        {
            "text": f"Chunk {i}",
            "metadata": {"source": f"/doc{i}.txt", "chunk_index": i},
            "score": 0.5,
            **({"rerank_score": s} if s is not None else {}),
        }
        for i, s in enumerate(scores)
    ]


def _make_mock_backend():
    backend = MagicMock()
    backend.provider = "ollama"
    backend.generate.return_value = "LLM-Antwort"
    backend.generate_stream.return_value = iter(["LLM", "-Antwort"])
    return backend


def _make_chain(config, reranker_enabled=True, results=None):
    from kiara.generation.chain import RAGChain

    with (
        patch("kiara.generation.chain.HybridRetriever") as MockRetriever,
        patch("kiara.retrieval.reranker.KironReranker") as MockReranker,
    ):
        mock_reranker = MockReranker.return_value
        mock_reranker.enabled = reranker_enabled
        mock_reranker.rerank.return_value = results if results is not None else []
        MockRetriever.return_value.retrieve.return_value = results or []
        backend = _make_mock_backend()
        chain = RAGChain(config, backend=backend, prompt_template_override=PROMPT_TEMPLATE)
        return chain, backend


class TestAbstainGate:
    def test_low_scores_trigger_abstain_without_llm_call(self):
        chain, backend = _make_chain(_base_config(), results=_make_results([0.01, 0.005, 0.001]))
        result = chain.query("Testfrage?", k=3)
        assert result["abstained"] is True
        assert result["answer"] == ABSTAIN_MESSAGE_DEFAULT
        assert result["sources"] == []
        assert result["context_chunks"] == []
        backend.generate.assert_not_called()

    def test_high_score_passes_through(self):
        chain, backend = _make_chain(_base_config(), results=_make_results([0.8, 0.01]))
        result = chain.query("Testfrage?", k=2)
        assert "abstained" not in result
        assert result["answer"] == "LLM-Antwort"
        backend.generate.assert_called_once()

    def test_disabled_by_default_config(self):
        chain, backend = _make_chain(
            _base_config(abstain_enabled=False), results=_make_results([0.001])
        )
        result = chain.query("Testfrage?", k=1)
        assert result["answer"] == "LLM-Antwort"
        backend.generate.assert_called_once()

    def test_no_abstain_when_reranker_disabled(self):
        chain, backend = _make_chain(
            _base_config(), reranker_enabled=False, results=_make_results([0.001])
        )
        result = chain.query("Testfrage?", k=1)
        assert result["answer"] == "LLM-Antwort"
        backend.generate.assert_called_once()

    def test_no_abstain_without_rerank_scores(self):
        # Reranker degraded: Ergebnisse ohne rerank_score duerfen nicht abstainen.
        chain, backend = _make_chain(_base_config(), results=_make_results([None, None]))
        result = chain.query("Testfrage?", k=2)
        assert result["answer"] == "LLM-Antwort"
        backend.generate.assert_called_once()

    def test_custom_abstain_message(self):
        chain, _ = _make_chain(
            _base_config(abstain_message="Nichts gefunden — keine Information."),
            results=_make_results([0.001]),
        )
        result = chain.query("Testfrage?", k=1)
        assert result["answer"] == "Nichts gefunden — keine Information."

    def test_threshold_boundary_inclusive_pass(self):
        # Score == Schwelle darf NICHT abstainen (>= passiert das Gate).
        chain, backend = _make_chain(_base_config(), results=_make_results([0.02]))
        result = chain.query("Testfrage?", k=1)
        assert result["answer"] == "LLM-Antwort"
        backend.generate.assert_called_once()

    def test_stream_abstains(self):
        chain, backend = _make_chain(_base_config(), results=_make_results([0.001]))
        stream, sources = chain.query_stream("Testfrage?", k=1)
        text = "".join(stream)
        assert text == ABSTAIN_MESSAGE_DEFAULT
        assert sources == []
        backend.generate_stream.assert_not_called()

    def test_stream_context_override_skips_gate(self):
        chain, backend = _make_chain(_base_config(), results=_make_results([0.001]))
        override = _make_results([0.001])
        chain._retriever.load_context_chunks.return_value = override
        stream, sources = chain.query_stream("Testfrage?", k=1, context_override=override)
        text = "".join(stream)
        assert text == "LLM-Antwort"
        assert len(sources) == 1

    def test_abstain_diagnostics_annotated(self):
        chain, _ = _make_chain(_base_config(), results=_make_results([0.001]))
        result = chain.query("Testfrage?", k=1)
        diag = result.get("retrieval_diagnostics")
        if isinstance(diag, dict):
            assert diag.get("abstain", {}).get("abstained") is True
            assert diag["abstain"]["reason"] == "max_rerank_score_below_threshold"


class TestAbstainInstruction:
    def test_instruction_appended_to_prompt(self):
        chain, backend = _make_chain(
            _base_config(abstain_instruction=True), results=_make_results([0.9])
        )
        chain.query("Testfrage?", k=1)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT in user_message

    def test_instruction_absent_by_default(self):
        chain, backend = _make_chain(_base_config(), results=_make_results([0.9]))
        chain.query("Testfrage?", k=1)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT not in user_message


class TestBenchmarkCompatibility:
    def test_abstain_message_matches_missing_patterns(self):
        """Der Abstain-Text MUSS von CRAG/Golden als missing erkannt werden."""
        lowered = ABSTAIN_MESSAGE_DEFAULT.lower()
        assert any(p in lowered for p in _MISSING_PATTERNS)

    def test_instruction_answer_matches_missing_patterns(self):
        """Auch der in der Instruktion vorgegebene Satz muss missing-erkennbar sein."""
        assert any(p in ABSTAIN_INSTRUCTION_TEXT.lower() for p in _MISSING_PATTERNS)


class TestConditionalInstruction:
    def test_instruction_suppressed_on_strong_hit(self):
        chain, backend = _make_chain(
            _base_config(abstain_instruction=True, abstain_instruction_max_score=0.3),
            results=_make_results([0.9, 0.1]),
        )
        chain.query("Testfrage?", k=2)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT not in user_message

    def test_instruction_shown_on_weak_hits(self):
        chain, backend = _make_chain(
            _base_config(abstain_instruction=True, abstain_instruction_max_score=0.3),
            results=_make_results([0.25, 0.1]),
        )
        chain.query("Testfrage?", k=2)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT in user_message

    def test_instruction_shown_without_scores(self):
        # Konservativ: ohne Reranker-Scores gilt das Band als erfuellt.
        chain, backend = _make_chain(
            _base_config(abstain_instruction=True, abstain_instruction_max_score=0.3),
            results=_make_results([None, None]),
        )
        chain.query("Testfrage?", k=2)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT in user_message

    def test_zero_band_means_always(self):
        chain, backend = _make_chain(
            _base_config(abstain_instruction=True, abstain_instruction_max_score=0.0),
            results=_make_results([0.9]),
        )
        chain.query("Testfrage?", k=1)
        user_message = backend.generate.call_args[0][1]
        assert ABSTAIN_INSTRUCTION_TEXT in user_message
