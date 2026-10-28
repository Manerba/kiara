"""Tests fuer kiara.tools.builtin_tools."""

import asyncio
import math
from unittest.mock import MagicMock, patch

import pytest

from kiara.tools.builtin_tools import _SAFE_MATH_NAMES, CalculateTool, SearchDocumentsTool


def _compatibility_error(operation: str = "vector_payload_read"):
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        IndexVectorKind,
        make_index_target,
    )

    target = make_index_target(
        {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        collection_name="kb_parent_guard",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="parent-test-model",
    )
    return IndexCompatibilityError(
        "index_contract_unverified",
        operation,
        target,
        "injected parent-document compatibility failure",
        current_state="unverified",
    )


# ===========================================================================
# CalculateTool — Grundlagen
# ===========================================================================


class TestCalculateTool:
    @pytest.fixture
    def tool(self):
        return CalculateTool()

    def test_definition(self, tool):
        defn = tool.definition()
        assert defn.name == "calculate"
        assert defn.required_permissions == []
        assert "expression" in defn.parameters["properties"]

    def test_simple_arithmetic(self, tool):
        result = asyncio.run(tool.execute({"expression": "2 + 3"}))
        assert result.success is True
        assert result.data["result"] == 5

    def test_sqrt(self, tool):
        result = asyncio.run(tool.execute({"expression": "sqrt(144)"}))
        assert result.success is True
        assert result.data["result"] == 12.0

    def test_power(self, tool):
        result = asyncio.run(tool.execute({"expression": "2**10"}))
        assert result.success is True
        assert result.data["result"] == 1024

    def test_pi(self, tool):
        result = asyncio.run(tool.execute({"expression": "pi"}))
        assert result.success is True
        assert abs(result.data["result"] - 3.14159265) < 0.001

    def test_complex_expression(self, tool):
        result = asyncio.run(tool.execute({"expression": "round(sqrt(2), 4)"}))
        assert result.success is True
        assert result.data["result"] == 1.4142

    def test_display_text(self, tool):
        result = asyncio.run(tool.execute({"expression": "10 / 3"}))
        assert "10 / 3" in result.display_text
        assert "=" in result.display_text

    def test_invalid_expression(self, tool):
        result = asyncio.run(tool.execute({"expression": "invalid+++"}))
        assert result.success is False
        assert result.error is not None

    def test_syntax_error_has_user_message(self, tool):
        result = asyncio.run(tool.execute({"expression": "1+"}))
        assert result.success is False
        assert result.error == "Ungueltiger mathematischer Ausdruck"
        assert "invalid syntax" not in result.display_text

    def test_security_no_builtins(self, tool):
        """eval() darf keine builtins aufrufen koennen."""
        result = asyncio.run(tool.execute({"expression": "__import__('os').system('ls')"}))
        assert result.success is False

    def test_security_no_open(self, tool):
        result = asyncio.run(tool.execute({"expression": "open('/etc/passwd')"}))
        assert result.success is False

    def test_validate_missing_expression(self, tool):
        valid, msg = tool.validate_params({})
        assert valid is False
        assert "expression" in msg

    def test_validate_valid(self, tool):
        valid, msg = tool.validate_params({"expression": "1+1"})
        assert valid is True


# ===========================================================================
# CalculateTool — Alle Whitelist-Funktionen
# ===========================================================================


class TestCalculateToolMathFunctions:
    """Jede Funktion in _SAFE_MATH_NAMES wird einzeln getestet."""

    @pytest.fixture
    def tool(self):
        return CalculateTool()

    def test_whitelist_completeness(self):
        """Die Whitelist enthaelt genau die erwarteten Funktionen."""
        expected = {
            "abs",
            "round",
            "min",
            "max",
            "sum",
            "pow",
            "sqrt",
            "log",
            "log10",
            "log2",
            "sin",
            "cos",
            "tan",
            "pi",
            "e",
            "ceil",
            "floor",
        }
        assert set(_SAFE_MATH_NAMES.keys()) == expected

    def test_log(self, tool):
        result = asyncio.run(tool.execute({"expression": "round(log(e), 4)"}))
        assert result.success is True
        assert result.data["result"] == 1.0

    def test_log10(self, tool):
        result = asyncio.run(tool.execute({"expression": "log10(1000)"}))
        assert result.success is True
        assert result.data["result"] == 3.0

    def test_log2(self, tool):
        result = asyncio.run(tool.execute({"expression": "log2(256)"}))
        assert result.success is True
        assert result.data["result"] == 8.0

    def test_sin(self, tool):
        result = asyncio.run(tool.execute({"expression": "round(sin(0), 4)"}))
        assert result.success is True
        assert result.data["result"] == 0.0

    def test_cos(self, tool):
        result = asyncio.run(tool.execute({"expression": "round(cos(0), 4)"}))
        assert result.success is True
        assert result.data["result"] == 1.0

    def test_tan(self, tool):
        result = asyncio.run(tool.execute({"expression": "round(tan(0), 4)"}))
        assert result.success is True
        assert result.data["result"] == 0.0

    def test_sin_pi_half(self, tool):
        """sin(pi/2) = 1.0"""
        result = asyncio.run(tool.execute({"expression": "round(sin(pi/2), 4)"}))
        assert result.success is True
        assert result.data["result"] == 1.0

    def test_e_constant(self, tool):
        result = asyncio.run(tool.execute({"expression": "e"}))
        assert result.success is True
        assert abs(result.data["result"] - math.e) < 0.0001

    def test_ceil(self, tool):
        result = asyncio.run(tool.execute({"expression": "ceil(4.2)"}))
        assert result.success is True
        assert result.data["result"] == 5

    def test_floor(self, tool):
        result = asyncio.run(tool.execute({"expression": "floor(4.8)"}))
        assert result.success is True
        assert result.data["result"] == 4

    def test_abs_negative(self, tool):
        result = asyncio.run(tool.execute({"expression": "abs(-42)"}))
        assert result.success is True
        assert result.data["result"] == 42

    def test_min_max(self, tool):
        result = asyncio.run(tool.execute({"expression": "min(5, 3, 8)"}))
        assert result.success is True
        assert result.data["result"] == 3

        result = asyncio.run(tool.execute({"expression": "max(5, 3, 8)"}))
        assert result.success is True
        assert result.data["result"] == 8

    def test_sum_list(self, tool):
        """sum([...]) ist im AST-Evaluator nicht erlaubt (List-Node nicht unterstuetzt)."""
        result = asyncio.run(tool.execute({"expression": "sum([1, 2, 3, 4])"}))
        assert result.success is False
        assert "List" in result.error

    def test_pow_function(self, tool):
        result = asyncio.run(tool.execute({"expression": "pow(3, 3)"}))
        assert result.success is True
        assert result.data["result"] == 27

    def test_combined_functions(self, tool):
        """Verschachtelte Whitelist-Funktionen."""
        result = asyncio.run(tool.execute({"expression": "ceil(sqrt(50))"}))
        assert result.success is True
        assert result.data["result"] == 8  # sqrt(50) ≈ 7.07 → ceil = 8

    def test_floor_negative(self, tool):
        result = asyncio.run(tool.execute({"expression": "floor(-2.3)"}))
        assert result.success is True
        assert result.data["result"] == -3


# ===========================================================================
# CalculateTool — Edge-Cases und Security
# ===========================================================================


class TestCalculateToolEdgeCases:
    @pytest.fixture
    def tool(self):
        return CalculateTool()

    def test_division_by_zero(self, tool):
        result = asyncio.run(tool.execute({"expression": "1/0"}))
        assert result.success is False
        assert "division" in result.error.lower() or "zero" in result.error.lower()

    def test_security_no_exec(self, tool):
        """exec() ist nicht in der Whitelist."""
        result = asyncio.run(tool.execute({"expression": "exec('print(1)')"}))
        assert result.success is False

    def test_security_lambda_blocked(self, tool):
        """lambda wird vom AST-Evaluator blockiert (kein eval() mehr)."""
        result = asyncio.run(tool.execute({"expression": "(lambda: 42)()"}))
        assert result.success is False
        # lambda mit Builtin-Zugriff schlaegt ebenfalls fehl:
        result2 = asyncio.run(tool.execute({"expression": "(lambda: __import__('os'))()"}))
        assert result2.success is False

    def test_security_no_list_comprehension_with_import(self, tool):
        """List comprehension mit __import__ muss fehlschlagen."""
        result = asyncio.run(tool.execute({"expression": "[__import__('os') for _ in [1]]"}))
        assert result.success is False

    def test_security_no_getattr(self, tool):
        """getattr ist nicht in der Whitelist."""
        result = asyncio.run(tool.execute({"expression": "getattr(type, '__bases__')"}))
        assert result.success is False

    def test_security_no_eval(self, tool):
        """Verschachteltes eval() muss fehlschlagen."""
        result = asyncio.run(tool.execute({"expression": "eval('1+1')"}))
        assert result.success is False

    def test_security_type_manipulation_blocked(self, tool):
        """Typ-Manipulation wird vom AST-Evaluator blockiert (kein Attribute-Zugriff)."""
        result = asyncio.run(tool.execute({"expression": "().__class__.__bases__[0].__name__"}))
        # AST-Evaluator erlaubt keine Attribute-Zugriffe
        assert result.success is False
        assert "Attribute" in result.error

    def test_very_large_number(self, tool):
        result = asyncio.run(tool.execute({"expression": "2**100"}))
        assert result.success is True
        assert result.data["result"] == 2**100

    def test_float_precision(self, tool):
        result = asyncio.run(tool.execute({"expression": "0.1 + 0.2"}))
        assert result.success is True
        # IEEE 754 Ungenauigkeit
        assert abs(result.data["result"] - 0.3) < 1e-10

    def test_negative_sqrt_raises(self, tool):
        """sqrt(-1) loest ValueError/math domain error aus."""
        result = asyncio.run(tool.execute({"expression": "sqrt(-1)"}))
        assert result.success is False

    def test_empty_expression(self, tool):
        """Leerer String schlaegt bei Validation fehl (minLength=1)."""
        valid, _ = tool.validate_params({"expression": ""})
        assert valid is False

    def test_display_text_shows_result(self, tool):
        """display_text enthaelt Ausdruck = Ergebnis."""
        result = asyncio.run(tool.execute({"expression": "7 * 8"}))
        assert "7 * 8" in result.display_text
        assert "56" in result.display_text

    def test_error_display_text(self, tool):
        """Fehler-display_text enthaelt 'Berechnungsfehler'."""
        result = asyncio.run(tool.execute({"expression": "1/0"}))
        assert "Berechnungsfehler" in result.display_text

    def test_data_contains_expression_and_result(self, tool):
        """data-Dict enthaelt den Originalausdruck."""
        result = asyncio.run(tool.execute({"expression": "3 + 4"}))
        assert result.data["expression"] == "3 + 4"
        assert result.data["result"] == 7


# ===========================================================================
# SearchDocumentsTool — Grundlagen
# ===========================================================================


class TestSearchDocumentsTool:
    @pytest.fixture
    def mock_retriever(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {
                "text": "Python ist eine Programmiersprache.",
                "metadata": {"source": "doc1.txt"},
                "score": 0.95,
            },
            {
                "text": "JavaScript laeuft im Browser.",
                "metadata": {"source": "doc2.txt"},
                "score": 0.85,
            },
        ]
        return retriever

    @pytest.fixture
    def mock_reranker(self):
        reranker = MagicMock()
        reranker.enabled = False
        return reranker

    @pytest.fixture
    def config(self):
        return {"retrieval": {"rerank_fetch_multiplier": 3}}

    @pytest.fixture
    def tool(self, mock_retriever, mock_reranker, config):
        return SearchDocumentsTool(mock_retriever, mock_reranker, config)

    def test_definition(self, tool):
        defn = tool.definition()
        assert defn.name == "search_documents"
        assert defn.required_permissions == []
        assert "query" in defn.parameters["properties"]
        assert "heutige Datum" in defn.description
        assert "WICHTIG: Bei zeitbezogenen Anfragen" not in defn.description

    def test_search_with_results(self, tool, mock_retriever):
        result = asyncio.run(tool.execute({"query": "Python"}))
        assert result.success is True
        assert "sources" in result.data
        assert len(result.data["sources"]) == 2
        mock_retriever.retrieve.assert_called_once()

    def test_search_no_results(self, mock_reranker, config):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        tool = SearchDocumentsTool(retriever, mock_reranker, config)

        result = asyncio.run(tool.execute({"query": "Unbekannt"}))
        assert result.success is True
        assert result.data["results"] == []

    def test_search_with_custom_default_k(self, mock_retriever, mock_reranker, config):
        """default_k im Konstruktor bestimmt die Abrufanzahl."""
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config, default_k=3)
        asyncio.run(tool.execute({"query": "Test"}))
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 3

    def test_source_contains_year_not_in_query_is_ignored(
        self, mock_retriever, mock_reranker, config
    ):
        """Prompt-Datum darf keinen heimlichen source_contains-Jahresfilter erzeugen."""
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config)
        asyncio.run(
            tool.execute(
                {
                    "query": "KIARA-CANARY-742 Wartungsstatus",
                    "source_contains": "2026",
                }
            )
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["filter_source"] is None

    def test_source_contains_year_in_query_is_preserved(
        self, mock_retriever, mock_reranker, config
    ):
        """Explizite Jahresfilter bleiben fuer Dateiname/Pfad-Suchen nutzbar."""
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config)
        asyncio.run(
            tool.execute(
                {
                    "query": "Hetzner Bezahlung 2025",
                    "source_contains": "2025",
                }
            )
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["filter_source"] == "2025"

    def test_aggregate_year_query_respects_ui_k(self, mock_retriever, mock_reranker, config):
        """Auch Jahresfragen duerfen die feste UI-Vorgabe nicht erhoehen."""
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config, default_k=5)
        asyncio.run(
            tool.execute(
                {
                    "query": "Hetzner Bezahlung 2025",
                    "source_contains": "2025",
                }
            )
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 5

    def test_model_k_does_not_override_ui_for_aggregate_query(
        self, mock_retriever, mock_reranker, config
    ):
        tool = SearchDocumentsTool(mock_retriever, mock_reranker, config, default_k=5)
        asyncio.run(
            tool.execute(
                {
                    "query": "Hetzner Bezahlung 2025",
                    "source_contains": "2025",
                    "k": 7,
                }
            )
        )
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 5

    def test_search_default_k_is_20(self, tool, mock_retriever):
        """Ohne expliziten default_k gilt der Agent-Profilstandard."""
        asyncio.run(tool.execute({"query": "Test"}))
        call_args = mock_retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 20

    def test_search_with_reranking(self, mock_retriever, config):
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = [
            {
                "text": "Python ist eine Programmiersprache.",
                "metadata": {"source": "doc1.txt"},
                "score": 0.95,
            },
        ]
        tool = SearchDocumentsTool(mock_retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Python", "k": 1}))
        assert result.success is True
        reranker.rerank.assert_called_once()

    def test_validate_missing_query(self, tool):
        valid, msg = tool.validate_params({})
        assert valid is False

    def test_validate_valid(self, tool):
        valid, msg = tool.validate_params({"query": "Suche"})
        assert valid is True

    def test_sources_in_display(self, tool):
        result = asyncio.run(tool.execute({"query": "Python"}))
        assert "doc1.txt" in result.display_text

    def test_sibling_expansion_with_reranking(self):
        """Bei aktivem Reranking werden Geschwister-Chunks ergaenzt."""
        # 3-Chunk-Email: chunk 0 hat den Betrag, chunk 1 ist Boilerplate
        all_chunks = []
        for i in range(3):
            text = ["Rechnungsbetrag: 5,03 EUR", "Boilerplate", "Footer"][i]
            all_chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "source": "rechnung.eml",
                        "chunk_index": i,
                        "chunk_total": 3,
                    },
                    "score": round(1.0 - i * 0.1, 2),
                }
            )

        retriever = MagicMock()
        retriever.retrieve.return_value = all_chunks

        reranker = MagicMock()
        reranker.enabled = True
        # Reranker waehlt nur den Boilerplate-Chunk
        reranker.rerank.return_value = [all_chunks[1]]

        config = {"retrieval": {"rerank_fetch_multiplier": 3, "max_sibling_chunks": 5}}
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Rechnung", "k": 1}))
        assert result.success is True
        # Durch Sibling-Expansion muessen alle 3 Chunks im Ergebnis sein
        assert len(result.data["sources"]) == 3
        # Der Rechnungsbetrag muss im display_text enthalten sein
        assert "5,03" in result.display_text


# ===========================================================================
# SearchDocumentsTool — Retriever-Verhalten
# ===========================================================================


class TestSearchDocumentsRetrieverBehavior:
    """Tests fuer Retriever-Interaktion, Reranking und Error-Handling."""

    @pytest.fixture
    def config(self):
        return {"retrieval": {"rerank_fetch_multiplier": 3}}

    def test_retriever_exception_aborts_search(self, config):
        """Ein essenzieller Suchausfall muss bis zum Chat-Abbruch propagieren."""
        from kiara.exceptions import RetrievalError

        retriever = MagicMock()
        retriever.retrieve.side_effect = RuntimeError("Connection lost")
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, config)

        with pytest.raises(RetrievalError, match="Connection lost"):
            asyncio.run(tool.execute({"query": "Test"}))

    def test_reranker_multiplier_applied(self, config):
        """fetch_k = default_k * rerank_fetch_multiplier bei aktivem Reranker."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": f"Dokument {i}", "metadata": {"source": f"d{i}.txt"}, "score": 0.9}
            for i in range(9)
        ]
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = retriever.retrieve.return_value[:3]
        tool = SearchDocumentsTool(retriever, reranker, config, default_k=3)

        asyncio.run(tool.execute({"query": "Test"}))
        # default_k=3, multiplier=3 → fetch_k=9
        call_args = retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 9

    def test_aggregate_ui_k_uses_rerank_multiplier(self, config):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": f"Dokument {i}", "metadata": {"source": f"d{i}.txt"}, "score": 0.9}
            for i in range(15)
        ]
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = retriever.retrieve.return_value[:5]
        tool = SearchDocumentsTool(retriever, reranker, config, default_k=5)

        asyncio.run(
            tool.execute(
                {
                    "query": "Hetzner Bezahlung 2025",
                    "source_contains": "2025",
                }
            )
        )

        call_args = retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 15
        reranker.rerank.assert_called_once()
        assert reranker.rerank.call_args.kwargs["top_k"] == 5

    def test_reranker_disabled_no_multiplier(self, config):
        """Deaktivierter Reranker → kein Multiplier, normaler k-Wert."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Doc", "metadata": {"source": "d.txt"}, "score": 0.9},
        ]
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, config, default_k=3)

        asyncio.run(tool.execute({"query": "Test"}))
        call_args = retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 3

    def test_default_k_is_20(self, config):
        """Ohne Chat-Vorgabe wird der Agent-Profilstandard verwendet."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, config)

        asyncio.run(tool.execute({"query": "Test"}))
        call_args = retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 20

    def test_custom_default_k(self, config):
        """default_k aus der UI ist verbindlich."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, config, default_k=20)

        asyncio.run(tool.execute({"query": "Test"}))
        call_args = retriever.retrieve.call_args
        assert call_args.kwargs["k"] == 20

    def test_results_trimmed_to_k_without_reranker(self, config):
        """Ohne Reranker werden Ergebnisse auf default_k geschnitten."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": f"Doc {i}", "metadata": {"source": f"d{i}.txt"}, "score": 0.9 - i * 0.1}
            for i in range(10)
        ]
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, config, default_k=3)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert len(result.data["sources"]) == 3

    def test_null_reranker_no_crash(self, config):
        """reranker=None darf keinen Crash verursachen."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Text", "metadata": {"source": "f.txt"}, "score": 0.8},
        ]
        tool = SearchDocumentsTool(retriever, None, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert result.success is True

    def test_parent_document_guard_error_is_not_returned_as_tool_error(self, config):
        """Parent-Nachladen muss denselben Guardfehler bis zum Orchestrator tragen."""
        candidate = {
            "text": "selected",
            "metadata": {
                "source": "parent.txt",
                "chunk_index": 0,
                "chunk_total": 2,
            },
            "score": 0.9,
        }
        error = _compatibility_error()
        retriever = MagicMock()
        retriever.retrieve.return_value = [candidate]
        retriever.load_chunks_by_source.side_effect = error
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = [candidate]
        tool = SearchDocumentsTool(retriever, reranker, config)

        with patch("kiara.generation.chain.expand_sibling_chunks") as sibling_expand:
            with pytest.raises(type(error)) as caught:
                asyncio.run(tool.execute({"query": "Test", "k": 1}))

        assert caught.value is error
        retriever.load_chunks_by_source.assert_called_once_with("parent.txt", None)
        sibling_expand.assert_not_called()
        assert tool._seen_keys == set()

    def test_parent_document_generic_error_keeps_candidate_fallback(self, config):
        """Nicht-Vertragsfehler beim Parent-Nachladen bleiben best-effort."""
        candidate = {
            "text": "selected",
            "metadata": {
                "source": "parent-control.txt",
                "chunk_index": 0,
                "chunk_total": 2,
                "modified": "2026-09-30T12:00:00+00:00",
                "size_bytes": 100,
            },
            "score": 0.9,
        }
        retriever = MagicMock()
        retriever.retrieve.return_value = [candidate]
        retriever.load_chunks_by_source.side_effect = RuntimeError("ordinary parent read failure")
        reranker = MagicMock()
        reranker.enabled = True
        reranker.rerank.return_value = [candidate]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test", "k": 1}))

        assert result.success is True
        returned = result.data["results"][0]
        reference = returned["metadata"].pop("document_ref")
        assert reference.startswith("doc_")
        assert result.data["results"] == [candidate]
        retriever.load_chunks_by_source.assert_called_once_with("parent-control.txt", None)


# ===========================================================================
# SearchDocumentsTool — Metadaten und Formatierung
# ===========================================================================


class TestSearchDocumentsMetadataAndFormatting:
    """Tests fuer Metadaten-Verarbeitung, Excerpt-Truncation und Quellen-Formatierung."""

    @pytest.fixture
    def config(self):
        return {"retrieval": {"rerank_fetch_multiplier": 3}}

    @pytest.fixture
    def reranker(self):
        r = MagicMock()
        r.enabled = False
        return r

    def test_missing_source_metadata_fallback_to_unbekannt(self, config, reranker):
        """Fehlender 'source'-Key in Metadaten → Fallback 'unbekannt'."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Kein Source-Feld.", "metadata": {}, "score": 0.7},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert result.success is True
        assert result.data["sources"][0]["source"] == "unbekannt"

    def test_excerpt_truncation(self, config, reranker):
        """Langer Text wird auf EXCERPT_LENGTH + '...' gekuerzt."""
        from kiara.shared.constants import EXCERPT_LENGTH

        long_text = "A" * (EXCERPT_LENGTH + 100)
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": long_text, "metadata": {"source": "big.txt"}, "score": 0.9},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        excerpt = result.data["sources"][0]["excerpt"]
        assert excerpt.endswith("...")
        assert len(excerpt) == EXCERPT_LENGTH + 3  # +3 fuer "..."

    def test_short_text_no_truncation(self, config, reranker):
        """Kurzer Text bleibt unveraendert (kein '...')."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Kurz.", "metadata": {"source": "s.txt"}, "score": 0.8},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        excerpt = result.data["sources"][0]["excerpt"]
        assert excerpt == "Kurz."
        assert not excerpt.endswith("...")

    def test_score_format_in_display_text(self, config, reranker):
        """Score wird als Prozent (z.B. 95%) im display_text formatiert."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Inhalt", "metadata": {"source": "doc.txt"}, "score": 0.95},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert "0.950" in result.display_text

    def test_missing_score_defaults_to_zero(self, config, reranker):
        """Fehlender score-Key → Default 0, kein Crash."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Kein Score.", "metadata": {"source": "ns.txt"}},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert result.success is True
        assert result.data["sources"][0]["score"] == 0

    def test_no_results_display_text(self, config, reranker):
        """Leere Ergebnisliste → passender display_text."""
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Nichts"}))
        assert "Keine" in result.display_text

    def test_multiple_results_numbered_in_display(self, config, reranker):
        """Mehrere Ergebnisse werden mit [1], [2] nummeriert."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Erster.", "metadata": {"source": "a.txt"}, "score": 0.9},
            {"text": "Zweiter.", "metadata": {"source": "b.txt"}, "score": 0.8},
            {"text": "Dritter.", "metadata": {"source": "c.txt"}, "score": 0.7},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        assert "[1]" in result.display_text
        assert "[2]" in result.display_text
        assert "[3]" in result.display_text

    def test_display_text_shows_full_chunk(self, config, reranker):
        """display_text enthaelt den vollen Chunk-Text (kein Abschneiden)."""
        long_text = "X" * 500
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": long_text, "metadata": {"source": "big.txt"}, "score": 0.9},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        # Voller Text im display_text, damit der Agent alle Infos sieht
        assert "X" * 500 in result.display_text

    def test_sources_structure(self, config, reranker):
        """Jede Source hat source, score und excerpt Keys."""
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            {"text": "Inhalt", "metadata": {"source": "file.pdf"}, "score": 0.88},
        ]
        tool = SearchDocumentsTool(retriever, reranker, config)

        result = asyncio.run(tool.execute({"query": "Test"}))
        source = result.data["sources"][0]
        assert "source" in source
        assert "score" in source
        assert "excerpt" in source
        assert source["source"] == "file.pdf"
        assert source["score"] == 0.88


class TestSearchDocumentsFilterFallback:
    """Fallback gegen halluzinierte Filter (0 Treffer mit file_type/directory/source_contains)."""

    def _tool_with_filtered_behavior(self):
        """Retriever liefert 0 bei gesetzten Filtern, Treffer ohne Filter."""
        retriever = MagicMock()

        def _retrieve(query, k, filter_dir=None, filter_type=None, **kwargs):
            if filter_dir or filter_type or kwargs.get("filter_source"):
                return []
            return [
                {
                    "text": "Hetzner Rechnung 54,49 EUR",
                    "metadata": {"source": "mail1.txt", "chunk_index": 0},
                    "score": 0.9,
                }
            ]

        retriever.retrieve.side_effect = _retrieve
        reranker = MagicMock()
        reranker.enabled = False
        return SearchDocumentsTool(retriever, reranker, {"retrieval": {}})

    def test_fallback_drops_hallucinated_filters(self):
        tool = self._tool_with_filtered_behavior()
        result = asyncio.run(
            tool.execute(
                {
                    "query": "Hetzner Bezahlung 2025",
                    "file_type": "pdf",
                    "directory": "Finanzen",
                }
            )
        )
        assert result.success is True
        assert len(result.data["results"]) == 1
        assert "wurden" in result.display_text and "ignoriert" in result.display_text
        assert "file_type='pdf'" in result.display_text

    def test_no_fallback_without_filters(self):
        tool = self._tool_with_filtered_behavior()
        result = asyncio.run(tool.execute({"query": "Hetzner Bezahlung 2025"}))
        assert result.success is True
        assert "ignoriert" not in (result.display_text or "")

    def test_fallback_still_empty_reports_no_results(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = []
        reranker = MagicMock()
        reranker.enabled = False
        tool = SearchDocumentsTool(retriever, reranker, {"retrieval": {}})
        result = asyncio.run(
            tool.execute({"query": "Hetzner Bezahlung 2025", "directory": "Finanzen"})
        )
        assert result.success is True
        assert result.data["results"] == []
        assert retriever.retrieve.call_count == 2
