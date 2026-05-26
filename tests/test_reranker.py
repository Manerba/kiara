"""Tests fuer den CrossEncoderReranker (kiara.retrieval.reranker)."""

import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from kiara.retrieval.reranker import CrossEncoderReranker

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_sentence_transformers():
    """Mockt das sentence_transformers-Modul fuer den Eager-Import-Check im
    Reranker-Konstruktor. Damit laufen Tests auch ohne installierte Dependency.

    Negative-Tests (Eager-Check bei fehlender Dependency) ueberschreiben
    diesen Mock per eigenem patch.dict-Block.
    """
    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = MagicMock()
    with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
        yield


@pytest.fixture
def disabled_config():
    """Config mit deaktiviertem Reranking."""
    return {
        "retrieval": {
            "reranking": False,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 5,
        },
    }


@pytest.fixture
def enabled_config():
    """Config mit aktiviertem Reranking (Modell wird nicht geladen)."""
    return {
        "retrieval": {
            "reranking": True,
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "rerank_top_k": 3,
        },
    }


@pytest.fixture
def sample_results():
    """Beispiel-Ergebnisliste vom Retriever."""
    return [
        {"text": "Dokument A", "metadata": {"source": "a.txt"}, "score": 0.9},
        {"text": "Dokument B", "metadata": {"source": "b.txt"}, "score": 0.8},
        {"text": "Dokument C", "metadata": {"source": "c.txt"}, "score": 0.7},
        {"text": "Dokument D", "metadata": {"source": "d.txt"}, "score": 0.6},
        {"text": "Dokument E", "metadata": {"source": "e.txt"}, "score": 0.5},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_reranker_disabled(self, disabled_config, sample_results):
        """Wenn deaktiviert: Ergebnisse unveraendert (nur top_k)."""
        reranker = CrossEncoderReranker(disabled_config)
        assert reranker.enabled is False

        result = reranker.rerank("test query", sample_results, top_k=3)
        # Gibt die ersten 3 Ergebnisse zurueck (unveraendert)
        assert len(result) == 3
        assert result[0]["text"] == "Dokument A"
        assert result[1]["text"] == "Dokument B"
        assert result[2]["text"] == "Dokument C"

    def test_reranker_empty_results(self, disabled_config):
        """Leere Eingabe liefert leere Ausgabe."""
        reranker = CrossEncoderReranker(disabled_config)
        result = reranker.rerank("test query", [])
        assert result == []

    def test_reranker_top_k(self, disabled_config, sample_results):
        """Top-K aus Config wird eingehalten."""
        reranker = CrossEncoderReranker(disabled_config)
        # Config hat rerank_top_k=5, wir uebergeben explizit top_k=2
        result = reranker.rerank("test", sample_results, top_k=2)
        assert len(result) == 2

    def test_reranker_config_top_k(self, disabled_config, sample_results):
        """Top-K aus Config wird als Default verwendet."""
        # disabled_config hat rerank_top_k=5
        reranker = CrossEncoderReranker(disabled_config)
        # Ohne expliziten top_k: nimmt rerank_top_k=5
        result = reranker.rerank("test", sample_results)
        assert len(result) == 5

    def test_reranker_enabled_property(self, enabled_config, disabled_config):
        """Die enabled-Property spiegelt die Config wider."""
        enabled_reranker = CrossEncoderReranker(enabled_config)
        assert enabled_reranker.enabled is True

        disabled_reranker = CrossEncoderReranker(disabled_config)
        assert disabled_reranker.enabled is False


class TestCrossEncoderPredict:
    """Tests fuer rerank() mit gemocktem Cross-Encoder (aktiver predict()-Pfad)."""

    def _make_reranker_with_mock(self, config, scores):
        """Erzeugt Reranker mit gemocktem Cross-Encoder."""
        reranker = CrossEncoderReranker(config)
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array(scores)
        reranker._cross_encoder = mock_encoder
        return reranker, mock_encoder

    def test_predict_correct_sorting(self, enabled_config, sample_results):
        """Korrekte Scores: Ergebnisse werden nach rerank_score sortiert."""
        # Score-Reihenfolge: D(0.9) > A(0.7) > E(0.6) > B(0.3) > C(0.1)
        reranker, mock = self._make_reranker_with_mock(enabled_config, [0.7, 0.3, 0.1, 0.9, 0.6])

        result = reranker.rerank("test query", sample_results, top_k=3)

        assert len(result) == 3
        assert result[0]["text"] == "Dokument D"  # Score 0.9
        assert result[1]["text"] == "Dokument A"  # Score 0.7
        assert result[2]["text"] == "Dokument E"  # Score 0.6
        assert result[0]["rerank_score"] == pytest.approx(0.9)

    def test_predict_pairs_passed_correctly(self, enabled_config, sample_results):
        """Cross-Encoder erhaelt korrekte (query, text) Paare."""
        reranker, mock = self._make_reranker_with_mock(enabled_config, [0.5] * 5)

        reranker.rerank("meine Anfrage", sample_results)

        pairs = mock.predict.call_args[0][0]
        assert len(pairs) == 5
        assert pairs[0] == ("meine Anfrage", "Dokument A")
        assert pairs[4] == ("meine Anfrage", "Dokument E")

    def test_predict_exception_fallback(self, enabled_config, sample_results):
        """predict() Exception: Fallback auf results[:top_k]."""
        reranker = CrossEncoderReranker(enabled_config)
        mock_encoder = MagicMock()
        mock_encoder.predict.side_effect = RuntimeError("CUDA OOM")
        reranker._cross_encoder = mock_encoder

        result = reranker.rerank("test", sample_results, top_k=3)

        assert len(result) == 3
        assert result[0]["text"] == "Dokument A"  # Originalreihenfolge

    def test_predict_score_count_mismatch(self, enabled_config, sample_results):
        """Falsche Anzahl Scores: Fallback auf results[:top_k]."""
        reranker, mock = self._make_reranker_with_mock(
            enabled_config, [0.5, 0.3]
        )  # 2 Scores fuer 5 Results

        result = reranker.rerank("test", sample_results, top_k=3)

        assert len(result) == 3
        assert result[0]["text"] == "Dokument A"  # Originalreihenfolge

    def test_predict_non_scalar_scores_fallback(self, enabled_config, sample_results):
        """2D-Logits-Output (Nx2): Fallback auf results[:top_k] statt Crash."""
        # 2-Klassen-Logits wie z.B. von BERT-Reranker mit num_labels=2
        logits_2d = np.array(
            [
                [0.1, 0.9],
                [0.4, 0.6],
                [0.3, 0.7],
                [0.8, 0.2],
                [0.5, 0.5],
            ]
        )
        reranker, mock = self._make_reranker_with_mock(enabled_config, [0.0] * 5)
        # mock direkt auf 2D-Array setzen (umgeht np.array-Wrap im Helper)
        mock.predict.return_value = logits_2d

        result = reranker.rerank("test", sample_results, top_k=3)

        # Kein Crash, Originalreihenfolge zurueckgegeben
        assert len(result) == 3
        assert result[0]["text"] == "Dokument A"
        assert result[1]["text"] == "Dokument B"
        assert result[2]["text"] == "Dokument C"
        # Caller-Dicts duerfen nicht mit rerank_score mutiert worden sein
        assert "rerank_score" not in sample_results[0]

    def test_predict_top_k_from_config(self, enabled_config, sample_results):
        """top_k aus Config (rerank_top_k=3) wird verwendet."""
        reranker, mock = self._make_reranker_with_mock(enabled_config, [0.1, 0.2, 0.3, 0.4, 0.5])

        result = reranker.rerank("test", sample_results)

        assert len(result) == 3  # rerank_top_k=3 aus enabled_config


class TestEagerImportCheck:
    """Eager-Check im Konstruktor: enabled darf nicht True bleiben, wenn die
    sentence-transformers-Dependency fehlt — sonst aktiviert die Chain
    Query-Expansion + rerank_fetch_multiplier ohne Cross-Encoder-Sortierung.
    """

    def test_disabled_when_sentence_transformers_missing(self, enabled_config):
        """Bei reranking=True UND fehlendem sentence-transformers: enabled=False."""

        # autouse-Fixture mockt sentence_transformers — fuer diesen Test ueberschreiben.
        def _fail_import(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise ImportError(f"No module named '{name}'")
            return _real_import(name, *args, **kwargs)

        _real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("sentence_transformers", None)
            with patch("builtins.__import__", side_effect=_fail_import):
                reranker = CrossEncoderReranker(enabled_config)
                assert reranker.enabled is False

    def test_remains_enabled_when_dependency_available(self, enabled_config):
        """Bei reranking=True UND vorhandenem sentence-transformers: enabled=True
        (autouse-Fixture stellt das Modul bereit)."""
        reranker = CrossEncoderReranker(enabled_config)
        assert reranker.enabled is True

    def test_disabled_config_skips_eager_check(self, disabled_config):
        """Bei reranking=False wird der Eager-Check nicht ausgefuehrt."""
        # Auch ohne sentence_transformers im sys.modules muss disabled-Konstruktor klappen.
        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("sentence_transformers", None)
            reranker = CrossEncoderReranker(disabled_config)
            assert reranker.enabled is False

    def test_disabled_when_import_raises_runtime_error(self, enabled_config):
        """Bei transitivem RuntimeError (z.B. zerschossenes torch-Binary) darf der
        Konstruktor NICHT crashen, sondern muss enabled=False setzen."""

        def _raise_runtime(name, *args, **kwargs):
            if name == "sentence_transformers" or name.startswith("sentence_transformers."):
                raise RuntimeError("dlopen failed: libcudart.so.12: cannot open shared object")
            return _real_import(name, *args, **kwargs)

        _real_import = (
            __builtins__["__import__"]
            if isinstance(__builtins__, dict)
            else __builtins__.__import__
        )

        with patch.dict(sys.modules, {}, clear=False):
            sys.modules.pop("sentence_transformers", None)
            with patch("builtins.__import__", side_effect=_raise_runtime):
                reranker = CrossEncoderReranker(enabled_config)
                assert reranker.enabled is False
