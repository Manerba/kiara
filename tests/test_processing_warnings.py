"""UI-Warnungen: echte Fehlerpfade, Stream, Persistierung und Modell-Isolation."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from kiara.exceptions import RetrievalError
from kiara.generation.agent_chain import AgentChain
from kiara.generation.tool_adapter import BaseToolAdapter, ToolStep
from kiara.retrieval.reranker import CrossEncoderReranker, KironReranker, RerankBackendError
from kiara.shared.processing_warnings import (
    ProcessingWarnings,
    current_processing_warnings,
    processing_warning_scope,
    report_processing_warning,
)
from kiara.tools.builtin_tools import SearchDocumentsTool
from kiara.tools.registry import ToolRegistry
from kiara.web import api_chat


def _config(**overrides):
    return {
        "retrieval": {
            "reranking": True,
            "rerank_backend": "kiron",
            "rerank_url": "http://kiron.test",
            "rerank_fail_mode": "degrade",
            **overrides,
        }
    }


def _hits():
    return [
        {
            "text": "Ein belegter Betrag: 7,45 Euro.",
            "score": 0.9,
            "metadata": {"source": "beleg.txt", "chunk_index": 0},
        }
    ]


def _parse(events):
    return [
        (
            event.splitlines()[0].removeprefix("event: "),
            json.loads(event.splitlines()[1].removeprefix("data: ")),
        )
        for event in events
    ]


@pytest.mark.parametrize(
    ("payload", "reason", "gpu"),
    [
        ({"error": "service_unavailable"}, "http_503", False),
        ({"error": "cuda_oom"}, "cuda_oom", True),
        ({"error": {"code": "cuda_oom"}}, "cuda_oom", True),
        ({"message": "cuda_oom maybe?"}, "http_503", False),
    ],
)
def test_rerank_warning_reason_and_seven_occurrences(payload, reason, gpu):
    collector = ProcessingWarnings()
    reranker = KironReranker(_config())
    with (
        processing_warning_scope(collector),
        patch.object(
            reranker,
            "_post_rerank",
            return_value=httpx.Response(503, json=payload),
        ),
    ):
        for _ in range(7):
            assert reranker.rerank("Belege", _hits()) == _hits()
    (warning,) = collector.snapshot()
    assert warning["count"] == 7
    assert warning["reason"] == reason
    assert ("GPU-Speicher" in warning["message"]) == gpu
    assert current_processing_warnings() == []


def test_success_does_not_erase_previous_failure_or_leak_to_next_answer():
    reranker = KironReranker(_config())
    first, second = ProcessingWarnings(), ProcessingWarnings()
    with processing_warning_scope(first), patch.object(reranker, "_post_rerank") as post:
        post.return_value = httpx.Response(503)
        reranker.rerank("Belege", _hits())
        post.return_value = httpx.Response(200, json={"results": [{"index": 0, "score": 0.8}]})
        reranker.rerank("Belege", _hits())
        assert first.snapshot()[0]["count"] == 1
        with processing_warning_scope(second):
            reranker.rerank("Belege", _hits())
            assert current_processing_warnings() == []
        assert current_processing_warnings() == first.snapshot()
    assert current_processing_warnings() == []


def test_timeout_disabled_and_empty_search():
    collector = ProcessingWarnings()
    with processing_warning_scope(collector):
        KironReranker(_config(reranking=False)).rerank("Belege", _hits())
        rr = KironReranker(_config())
        rr.rerank("Belege", [])
        assert collector.snapshot() == []
        with patch.object(rr, "_post_rerank", side_effect=httpx.ReadTimeout("timed out")):
            rr.rerank("Belege", _hits())
    assert collector.snapshot()[0]["reason"] == "timeout"


@pytest.mark.parametrize("fail_mode", ["error", "degrade"])
def test_local_predict_failure_respects_policy(fail_mode):
    rr = CrossEncoderReranker(_config(reranking=False, rerank_fail_mode=fail_mode))
    rr._enabled = True
    rr._cross_encoder = MagicMock()
    rr._cross_encoder.predict.side_effect = RuntimeError("broken model")
    collector = ProcessingWarnings()
    with processing_warning_scope(collector):
        if fail_mode == "error":
            with pytest.raises(RerankBackendError):
                rr.rerank("Belege", _hits())
            assert collector.snapshot() == []
        else:
            assert rr.rerank("Belege", _hits()) == _hits()
            assert collector.snapshot()[0]["reason"] == "predict_error"


def _agent(*, parallel=False, forced=False, fail_mode="degrade", retrieval_error=False):
    config = _config(rerank_fail_mode=fail_mode)
    retriever = MagicMock()
    retriever._store = None
    retriever.retrieve.side_effect = (
        RetrievalError("index unavailable") if retrieval_error else lambda **kw: _hits()
    )
    rr = KironReranker(config)
    rr._post_rerank = MagicMock(return_value=httpx.Response(503, json={"error": "cuda_oom"}))
    tool = SearchDocumentsTool(retriever, rr, config)
    registry = ToolRegistry()
    registry.register(tool)
    adapter = MagicMock(spec=BaseToolAdapter)
    adapter.deanonymize_params.side_effect = lambda params: dict(params)
    adapter.result_truncate_limit = 24000
    adapter.last_usage = {"input_tokens": 10, "output_tokens": 5, "model": "test"}
    adapter.last_conversation = "Modellkontext"
    adapter.stream_final.return_value = iter(["Antwort aus dem Beleg."])
    if forced:
        first = ToolStep(text="Entwurf", is_final=False)
    else:
        first = ToolStep(
            tool_call=("search_documents", {"query": "Belege Januar"}), tool_use_id="a"
        )
        if parallel:
            first.parallel_calls = [("search_documents", {"query": "Belege Februar"}, "b")]
    adapter.step.side_effect = [first, ToolStep(text="Antwort aus dem Beleg.", is_final=True)]
    chain = AgentChain(config, registry, adapter, system_prompt_override="Quellen auswerten.")
    return chain, adapter, rr


@pytest.mark.parametrize(
    ("parallel", "forced", "count"), [(False, False, 1), (True, False, 2), (False, True, 1)]
)
@pytest.mark.asyncio
async def test_real_agent_threads_stream_and_no_warning_in_model_messages(parallel, forced, count):
    chain, adapter, rr = _agent(parallel=parallel, forced=forced)
    with (
        patch.object(api_chat, "_prepare_agent_context", return_value=(chain, None, None)),
        patch.object(api_chat, "_emit_query_audit"),
        patch.object(api_chat, "db_transaction"),
        patch.object(api_chat, "msg_repo") as messages,
        patch.object(api_chat, "chat_repo"),
    ):
        events = _parse(
            [
                event
                async for event in api_chat._stream_agent(
                    _config(),
                    api_chat.QueryRequest(question="Belege Januar", mode="agent"),
                    42,
                    {"id": 1},
                )
            ]
        )
    assert rr._post_rerank.call_count == count
    warnings = [data["warnings"] for kind, data in events if kind == "processing_warnings"]
    assert warnings[-1][0]["count"] == count
    assert [kind for kind, _ in events].index("processing_warnings") < [
        kind for kind, _ in events
    ].index("chunk")
    assert messages.add_message.call_args.kwargs["warnings"] == warnings[-1]
    assert messages.add_message.call_args.args[2] == "Antwort aus dem Beleg."
    model_calls = (
        repr(adapter.init_conversation.call_args)
        + repr(adapter.add_tool_results.call_args_list)
        + repr(adapter.add_tool_result.call_args_list)
    )
    assert "reranking_unavailable" not in model_calls
    assert "GPU-Speicher" not in model_calls
    assert "cuda_oom" not in model_calls
    assert current_processing_warnings() == []


@pytest.mark.asyncio
async def test_agent_context_exhaustion_is_visible_before_answer_and_persisted():
    chain, adapter, _ = _agent()
    chain._config["chat"] = {
        "agent": {
            "context_budget_chars": 10,
            "context_result_chars": 10,
            "context_read_reserve_chars": 0,
        }
    }
    adapter.prepare_tool_text.side_effect = str
    with (
        patch.object(api_chat, "_prepare_agent_context", return_value=(chain, None, None)),
        patch.object(api_chat, "_emit_query_audit"),
        patch.object(api_chat, "db_transaction"),
        patch.object(api_chat, "msg_repo") as messages,
        patch.object(api_chat, "chat_repo"),
    ):
        events = _parse(
            [
                event
                async for event in api_chat._stream_agent(
                    chain._config,
                    api_chat.QueryRequest(question="Belege", mode="agent"),
                    42,
                    {"id": 1},
                )
            ]
        )
    warnings = [data["warnings"] for kind, data in events if kind == "processing_warnings"][-1]
    context = next(w for w in warnings if w["component"] == "context")
    assert context["reason"] == "budget_exhausted"
    assert context["details"]["budget_chars"] == 10
    assert messages.add_message.call_args.kwargs["warnings"] == warnings
    kinds = [kind for kind, _ in events]
    assert kinds.index("processing_warnings") < kinds.index("chunk")
    assert current_processing_warnings() == []


def test_mcp_query_serializes_model_window_warning_and_isolates_requests():
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from kiara.generation.chain import RAGChain
    from kiara.mcp_server import _create_app

    def run_query(user, question, k):
        chain = object.__new__(RAGChain)
        if question == "gross":
            chain._check_context_overflow("x" * 10000, 100)
        audit = SimpleNamespace(last_entities_anonymized=0, last_prompt="test")
        return "Antwort", [], "local", 1, "test", audit

    with (
        patch("kiara.mcp_server._run_rag_query", side_effect=run_query),
        patch("kiara.mcp_server._log_mcp_query"),
        patch("kiara.web.api_key_auth.validate_api_key", return_value={"id": 1}),
    ):
        app = _create_app({"web": {"mcp_enabled": True, "mcp_retrieval_enabled": False}})
        with TestClient(app) as client:
            for index, question in enumerate(("gross", "klein"), 1):
                response = client.post(
                    "/mcp",
                    headers={
                        "Authorization": "Bearer orag_test",
                        "Accept": "application/json, text/event-stream",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": index,
                        "method": "tools/call",
                        "params": {"name": "query", "arguments": {"question": question}},
                    },
                )
                assert response.status_code == 200
                if "text/event-stream" in response.headers.get("content-type", ""):
                    payload = next(
                        json.loads(line[6:])
                        for line in response.text.splitlines()
                        if line.startswith("data: ")
                    )
                else:
                    payload = response.json()
                result = json.loads(payload["result"]["content"][0]["text"])
                assert result["answer"] == "Antwort"
                if question == "gross":
                    (warning,) = result["warnings"]
                    assert warning["reason"] == "window_estimate_exceeded"
                    assert warning["details"]["num_ctx"] == 100
                else:
                    assert result["warnings"] == []


@pytest.mark.parametrize(("parallel", "forced"), [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("retrieval_error", [False, True])
@pytest.mark.asyncio
async def test_essential_failure_aborts_agent_without_saving_or_model_followup(
    parallel, forced, retrieval_error
):
    chain, adapter, _ = _agent(
        parallel=parallel, forced=forced, fail_mode="error", retrieval_error=retrieval_error
    )
    with (
        patch.object(api_chat, "_prepare_agent_context", return_value=(chain, None, None)),
        patch.object(api_chat, "_emit_query_audit"),
        patch.object(api_chat, "_save_assistant_message") as save,
    ):
        events = _parse(
            [
                event
                async for event in api_chat._stream_agent(
                    _config(),
                    api_chat.QueryRequest(question="Belege Januar", mode="agent"),
                    42,
                    {"id": 1},
                )
            ]
        )
    assert "error" in [kind for kind, _ in events]
    assert "chunk" not in [kind for kind, _ in events]
    assert "processing_warnings" not in [kind for kind, _ in events]
    save.assert_not_called()
    assert adapter.step.call_count == 1
    adapter.add_tool_results.assert_not_called()
    adapter.add_tool_result.assert_not_called()


@pytest.mark.asyncio
async def test_interleaved_streams_and_close_do_not_mix_warnings():
    @api_chat._with_processing_warnings
    async def run(component):
        report_processing_warning(component)
        await asyncio.sleep(0)
        yield api_chat._sse_event("chunk", {"text": component})
        report_processing_warning(component)
        yield api_chat._sse_event("done", {})

    async def collect(component):
        return _parse([item async for item in run(component)])

    left, right = await asyncio.gather(collect("graph"), collect("reranking"))
    for events, component in [(left, "graph"), (right, "reranking")]:
        warnings = [data["warnings"] for kind, data in events if kind == "processing_warnings"]
        assert [items[0]["count"] for items in warnings] == [1, 2]
        assert all(len(items) == 1 and items[0]["component"] == component for items in warnings)
    stream = run("graph")
    await anext(stream)
    await stream.aclose()
    assert current_processing_warnings() == []


def test_persisted_warning_reload_and_history_exclusion(test_db):
    from kiara.repositories.chat import ChatRepository, MessageRepository
    from kiara.web.core_schema_migration import migrate_core_schema

    # Also exercise the upgrade from the old message schema and idempotency.
    test_db.execute("ALTER TABLE chat_messages DROP COLUMN warnings")
    migrate_core_schema(test_db)
    migrate_core_schema(test_db)
    user = test_db.execute(
        "INSERT INTO users (username) VALUES ('warning-test') RETURNING id"
    ).fetchone()
    chat_id = ChatRepository().create(user_id=user["id"], title="Warnung", conn=test_db)
    test_db.commit()
    collector = ProcessingWarnings()
    with processing_warning_scope(collector):
        report_processing_warning("reranking", "http_503")
        api_chat._save_assistant_message(
            chat_id, "Belegter Antworttext.", [{"source": "beleg.txt"}]
        )
    repo = MessageRepository()
    messages = repo.list_by_chat(chat_id)
    assert messages[0]["warnings"] == collector.snapshot()
    restored = api_chat._serialize_chat_messages(messages, [])
    assert restored[0]["warnings"] == collector.snapshot()
    assert restored[0]["content"] == "Belegter Antworttext."
    assert repo.load_history(chat_id) == [{"role": "assistant", "content": "Belegter Antworttext."}]


@pytest.mark.asyncio
async def test_final_rerank_warning_reaches_stream_before_answer():
    chain, adapter, rr = _agent()
    adapter.step.side_effect = [ToolStep(text="Entwurf", is_final=False)]
    chain._force_initial_search = False

    def final(**_kwargs):
        rr.rerank("Frage", _hits())
        yield "Finale Antwort"

    adapter.stream_final.side_effect = final
    with (
        patch.object(api_chat, "_prepare_agent_context", return_value=(chain, None, None)),
        patch.object(api_chat, "_emit_query_audit"),
        patch.object(api_chat, "db_transaction"),
        patch.object(api_chat, "msg_repo") as messages,
        patch.object(api_chat, "chat_repo"),
    ):
        events = _parse(
            [
                event
                async for event in api_chat._stream_agent(
                    _config(),
                    api_chat.QueryRequest(question="Belege Januar"),
                    42,
                    {"id": 1},
                )
            ]
        )
    kinds = [kind for kind, _ in events]
    assert kinds.index("processing_warnings") < kinds.index("chunk")
    assert messages.add_message.call_args.kwargs["warnings"][0]["count"] == 1


@pytest.mark.parametrize("mode", ["standard", "rewriter"])
@pytest.mark.asyncio
async def test_rag_modes_emit_and_save_warnings_without_altering_prompts(mode):
    from kiara.generation.chain import RAGChain
    from kiara.generation.rewriter_chain import RewriterChain

    backend = MagicMock()
    backend.provider = "ollama"
    backend.model = "test"
    backend.last_usage = {}
    backend.generate_stream.return_value = iter(["Belegte Antwort."])
    rewrite_backend = MagicMock()
    rewrite_backend.model = "rewrite-test"
    rewrite_backend.generate.side_effect = RuntimeError("rewrite unavailable")
    retriever = MagicMock()
    retriever.retrieve.side_effect = lambda **kw: _hits()
    config = _config()
    with patch("kiara.generation.chain.HybridRetriever", return_value=retriever):
        if mode == "standard":
            chain = RAGChain(config, backend=backend)
        else:
            with (
                patch(
                    "kiara.backend_selector.get_all_active",
                    return_value=[
                        {
                            "provider": "ollama",
                            "url": "http://ollama.test",
                            "model": "test",
                        }
                    ],
                ),
                patch(
                    "kiara.generation.backends.ollama.OllamaBackend", return_value=rewrite_backend
                ),
            ):
                chain = RewriterChain(config, rag_backend=backend)
    with (
        patch("kiara.retrieval.reranker.httpx.post", return_value=httpx.Response(503)),
        patch.object(api_chat, "_prepare_standard_context", return_value=(chain, None, None)),
        patch("kiara.generation.rewriter_chain.RewriterChain", return_value=chain),
        patch("kiara.generation.cloud_routing._create_chain_backend", return_value=backend),
        patch.object(api_chat, "_build_filter_pair", return_value=(None, [])),
        patch.object(api_chat, "_emit_query_audit"),
        patch.object(api_chat, "db_transaction"),
        patch.object(api_chat, "msg_repo") as messages,
        patch.object(api_chat, "chat_repo"),
    ):
        stream = api_chat._stream_standard if mode == "standard" else api_chat._stream_rewriter
        events = _parse(
            [
                event
                async for event in stream(
                    config,
                    api_chat.QueryRequest(question="Belege Januar", k=3, mode=mode),
                    42,
                    {"id": 1},
                )
            ]
        )
    warnings = [data["warnings"] for kind, data in events if kind == "processing_warnings"]
    expected = {"reranking"} | ({"rewrite"} if mode == "rewriter" else set())
    assert {warning["component"] for warning in warnings[-1]} == expected
    assert messages.add_message.call_args.kwargs["warnings"] == warnings[-1]
    assert messages.add_message.call_args.args[2] == "Belegte Antwort."
    prompt = repr(backend.generate_stream.call_args)
    assert "reranking_unavailable" not in prompt
    assert "Die Relevanzsortierung" not in prompt
    assert "Die Umformulierung" not in prompt


def test_optional_expansion_and_graph_chunk_failures_are_reported():
    from kiara.generation.chain import expand_sibling_chunks
    from kiara.graph.retriever import GraphRetriever
    from kiara.retrieval.query_expansion import QueryExpander

    backend = MagicMock(provider="ollama")
    backend.generate.side_effect = RuntimeError("expander down")
    expander = QueryExpander({"retrieval": {"query_expansion": True}}, backend=backend)
    graph = GraphRetriever({}, MagicMock())
    collector = ProcessingWarnings()
    with processing_warning_scope(collector):
        assert expander.expand("Belege") == ["Belege"]
        hits = _hits()
        hits[0]["metadata"]["chunk_total"] = 2
        assert (
            expand_sibling_chunks(
                hits,
                hits,
                5,
                load_source_chunks=MagicMock(side_effect=RuntimeError("store down")),
            )
            == hits
        )
        with patch(
            "kiara.retrieval.vectorstore.VectorStore", side_effect=RuntimeError("store down")
        ):
            assert graph._load_chunks_from_graph(["a"]) is None
    assert {item["component"] for item in collector.snapshot()} == {
        "query_expansion",
        "document_expansion",
        "graph",
    }


def test_successful_connection_retry_does_not_warn():
    rr = KironReranker(_config())
    collector = ProcessingWarnings()
    with (
        processing_warning_scope(collector),
        patch(
            "kiara.retrieval.reranker.httpx.post",
            side_effect=[
                httpx.ConnectError("temporary"),
                httpx.Response(200, json={"results": [{"index": 0, "score": 0.8}]}),
            ],
        ),
    ):
        assert rr.rerank("Belege", _hits())[0]["rerank_score"] == 0.8
    assert collector.snapshot() == []


def test_shared_context_rerank_does_not_put_warning_in_prompt():
    from kiara.generation.context_selection import prepare_context
    from kiara.retrieval.reranker import create_reranker

    chunks = [
        {"text": f"Beleg {i}", "metadata": {"source": f"beleg-{i}.txt", "chunk_index": 0}}
        for i in range(6)
    ]
    collector = ProcessingWarnings()
    with (
        processing_warning_scope(collector),
        patch(
            "kiara.retrieval.reranker.httpx.post",
            return_value=httpx.Response(503),
        ),
    ):
        prepared = prepare_context(
            chunks, budget=200, query="Belege", reranker=create_reranker(_config())
        )
    assert collector.snapshot()[0]["component"] == "reranking"
    assert "Die Relevanzsortierung" not in prepared.text
    assert "reranking_unavailable" not in prepared.text
