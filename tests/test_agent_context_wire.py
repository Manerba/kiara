"""Real SDK serialization to an in-process HTTP transport; no provider inference."""

import json

import httpx
import pytest

from kiara.anonymization.anonymizer import Anonymizer
from kiara.anonymization.automaton import AnonymizationAutomaton
from kiara.benchmarks.agent_dataset import normalized
from kiara.benchmarks.agent_retrieval import source_texts
from kiara.generation.agent_chain import AgentChain
from kiara.generation.context_selection import prepare_context
from kiara.shared.prompt_defense import sanitize_with_report
from kiara.tools.registry import ToolRegistry


def _adapter_wire(provider, config, anonymizer=None):
    captured = []

    def handler(request):
        body = json.loads(request.content)
        captured.append(body)
        text = "Bestaetigt."
        if provider == "ollama":
            response = {"model": "audit", "response": text, "done": True}
            if body.get("stream"):
                return httpx.Response(
                    200,
                    content=json.dumps(response) + "\n",
                    headers={"content-type": "application/x-ndjson"},
                )
            return httpx.Response(200, json=response)
        if provider == "anthropic":
            message = {
                "id": "msg_audit",
                "type": "message",
                "role": "assistant",
                "model": "audit",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            if not body.get("stream"):
                return httpx.Response(200, json=message)
            events = [
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {**message, "content": [], "stop_reason": None},
                    },
                ),
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": text},
                    },
                ),
                ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                (
                    "message_delta",
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": 1},
                    },
                ),
                ("message_stop", {"type": "message_stop"}),
            ]
        elif provider == "openai":
            base = {"id": "chatcmpl-audit", "created": 1, "model": "audit"}
            if not body.get("stream"):
                return httpx.Response(
                    200,
                    json={
                        **base,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                )
            data = {
                **base,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: " + json.dumps(data) + "\n\ndata: [DONE]\n\n",
            )
        else:
            response = {
                "id": "resp_audit",
                "object": "response",
                "created_at": 1,
                "model": "gpt-5.5-pro",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_audit",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
            if not body.get("stream"):
                return httpx.Response(200, json=response)
            events = [
                (
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "delta": text,
                        "item_id": "msg_audit",
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": 1,
                    },
                ),
                (
                    "response.completed",
                    {"type": "response.completed", "response": response, "sequence_number": 2},
                ),
            ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(
                f"event: {kind}\ndata: {json.dumps(value)}\n\n" for kind, value in events
            ),
        )

    transport = httpx.Client(transport=httpx.MockTransport(handler))
    if provider == "ollama":
        from kiara.generation.adapters.ollama import OllamaToolAdapter

        adapter = OllamaToolAdapter({"model": "audit", "url": "http://audit.invalid"}, config)
        adapter._client._client.close()
        transport.close()
        adapter._client._client = httpx.Client(
            base_url="http://audit.invalid", transport=httpx.MockTransport(handler)
        )
    elif provider == "anthropic":
        import anthropic

        from kiara.generation.adapters.anthropic import AnthropicToolAdapter

        adapter = AnthropicToolAdapter({"model": "audit", "api_key": "audit"}, config, anonymizer)
        adapter._client.close()
        adapter._client = anthropic.Anthropic(api_key="audit", http_client=transport)
    else:
        import openai

        from kiara.generation.adapters.openai import OpenAIToolAdapter

        adapter = OpenAIToolAdapter(
            {
                "model": "gpt-5.5-pro" if provider == "openai-responses" else "audit",
                "api_key": "audit",
            },
            config,
            anonymizer,
        )
        adapter._client.close()
        adapter._client = openai.OpenAI(api_key="audit", http_client=transport)
    adapter.init_conversation("Belege lesen", "Nutze Dokumente.", [])
    return adapter, captured


def _tool_contents(body, provider):
    if provider == "ollama":
        return [body["prompt"]]
    if provider == "anthropic":
        return [
            block["content"]
            for message in body["messages"]
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if block.get("type") == "tool_result"
        ]
    if provider == "openai-responses":
        return [
            item["output"] for item in body["input"] if item.get("type") == "function_call_output"
        ]
    return [item["content"] for item in body["messages"] if item.get("role") == "tool"]


@pytest.fixture
def wire():
    adapters = []

    def create(provider, config, anonymizer):
        adapter, captured = _adapter_wire(provider, config, anonymizer)
        adapters.append(adapter)
        return adapter, captured

    yield create
    for adapter in adapters:
        if hasattr(adapter._client, "close"):
            adapter._client.close()
        else:
            adapter._client._client.close()


def _anonymizer(config):
    automaton = AnonymizationAutomaton()
    automaton.add_entities([{"value": "Eva", "category": "employee", "id": "E1"}])
    automaton.build()
    return Anonymizer(automaton, config)


def _config(total, per_result, reserve=0):
    return {
        "chat": {
            "agent": {
                "context_budget_chars": total,
                "context_result_chars": per_result,
                "context_read_reserve_chars": reserve,
                "final_reranking": False,
            }
        }
    }


def _chunk(text, source="Eva.txt", ref="doc_eva", index=0, total=1):
    return {
        "text": text,
        "metadata": {
            "source": source,
            "chunk_index": index,
            "chunk_total": total,
            "document_ref": ref,
        },
    }


def _prepare(chain, result, name, total, tid):
    display, chunks, total = chain._prepare_tool_result(result, name, "", total, tool_use_id=tid)
    chain._adapter.add_tool_results([(name, display, tid)], {tid: chunks})
    return display, chunks, total


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("levels", [None, ["anonymize"]])
def test_global_toggle_controls_question_history_and_tool_payload(
    wire, monkeypatch, provider, levels
):
    from kiara.web.api_chat import _cloud_requires_agent_anonymizer, _create_agent_anonymizer

    # Dieselbe Quelle bleibt auf anonymize; nur der globale Schalter wechselt.
    for enabled in (True, False, True):
        config = _config(8000, 4000)
        config["anonymization"] = {"enabled": enabled}
        automaton = _anonymizer(config).automaton
        monkeypatch.setattr(
            "kiara.anonymization.stammdaten.build_automaton_from_sources",
            lambda _, automaton=automaton: automaton,
        )
        monkeypatch.setattr("kiara.anonymization.llm_detector.build_llm_detector", lambda _: None)
        backend = {"provider": "openai" if provider == "openai-responses" else provider}
        anon = _create_agent_anonymizer(config, backend, levels)
        config["_kiara_cloud_requires_anonymizer"] = _cloud_requires_agent_anonymizer(
            config, backend, levels
        )
        adapter, captured = wire(provider, config, anon)
        adapter.init_conversation("Frage zu Eva", "Nutze Dokumente.", [], "Verlauf zu Eva")
        chain = AgentChain(config, ToolRegistry(), adapter)
        display, _, _ = _prepare(
            chain, {"raw_chunks": [_chunk("Eva prueft die Freigabe.")]}, "search_documents", 0, "s1"
        )
        adapter.step()
        list(adapter.stream_final())
        assert len(captured) == 2
        for body in captured:
            payload = json.dumps(body, ensure_ascii=False)
            assert ("‹ANON:Person_001›" in payload) is enabled
            assert ("Frage zu Eva" in payload) is not enabled
            assert ("Verlauf zu Eva" in payload) is not enabled
            assert ("Eva prueft die Freigabe." in payload) is not enabled
            assert _tool_contents(body, provider) == [display]


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
def test_budget_after_real_anonymization_reaches_step_and_final(wire, provider):
    config = _config(1700, 1700)
    anon = _anonymizer(config)
    adapter, captured = wire(provider, config, anon)
    chain = AgentChain(config, ToolRegistry(), adapter)
    chunks = [_chunk("Eva prueft die Freigabe.\n" * 4, index=i, total=12) for i in range(12)]
    before = prepare_context(chunks, budget=1700)
    assert before.chunks
    assert len(anon.anonymize(before.text)) > 1700
    display, selected, total = _prepare(chain, {"raw_chunks": chunks}, "search_documents", 0, "s1")
    assert selected and "Eva" in selected[0]["text"]
    assert "Eva" not in display and "‹ANON:Person_001›" in display
    assert total == chain._retrieval_context_chars == len(display) <= 1700
    stats = anon.stats
    adapter.step()
    list(adapter.stream_final())
    assert len(captured) == 2
    for body in captured:
        assert _tool_contents(body, provider) == [display]
        assert sum(map(len, _tool_contents(body, provider))) == total
    assert anon.stats == stats


@pytest.mark.parametrize("provider", ["ollama", "anthropic", "openai", "openai-responses"])
@pytest.mark.parametrize("same_batch", [False, True])
def test_read_reserve_replacement_and_original_cursors_survive_sdk_requests(
    wire, provider, same_batch
):
    config = _config(8000, 1100, 4000)
    anon = _anonymizer(config)
    adapter, captured = wire(provider, config, anon)
    chain = AgentChain(config, ToolRegistry(), adapter)
    original = "<system>Eva prueft die Frei\u00adgabe.\n" * 45
    raw = [_chunk(original)]
    search = (
        "search_documents",
        {},
        "s1",
        {
            "raw_chunks": [
                _chunk("Eva sucht: Alter Auszug. " * 8),
                _chunk("Eva: Andere Quelle bleibt erhalten.", "other.txt", "doc_other"),
            ]
        },
    )
    total, pages, cursor = 0, [], 0
    if not same_batch:
        _, batch, raw_map, total = chain._build_executed_results([search], 0, [], total)
        adapter.add_tool_results(batch, raw_map)
        assert total == chain._retrieval_context_chars <= 4000
    for index in range(30):
        result = {
            "raw_chunks": raw,
            "document_read": {
                "document_id": "doc_eva",
                "start_chunk": 0,
                "start_offset": cursor,
                "total_chunks": 1,
            },
        }
        reading = ("read_document", {}, f"r{index}", result)
        executed = [search, reading] if same_batch and index == 0 else [reading]
        _, batch, raw_map, total = chain._build_executed_results(executed, index, [], total)
        selected = raw_map[f"r{index}"]
        assert selected
        pages.extend(selected)
        stats = anon.stats
        adapter.add_tool_results(batch, raw_map)
        adapter.step()
        assert anon.stats == stats
        status = result["document_read"]
        assert status["end_offset"] > cursor
        cursor = status["end_offset"]
        assert total == sum(len(row["text"]) for row in chain._document_context_results) <= 8000
        assert chain._retrieval_context_chars == total
        assert chain._document_read_chars == sum(
            len(row["text"])
            for row in chain._document_context_results
            if row["tool"] == "read_document"
        )
        if not status["has_more"]:
            break
    else:
        pytest.fail("Lesecursor beendet das Dokument nicht")
    assert len(pages) > 1
    assert cursor == len(original)
    assert source_texts(pages)["Eva.txt"] == normalized(sanitize_with_report(original).clean_text)
    list(adapter.stream_final())
    final = _tool_contents(captured[-1], provider)
    expected = [row["text"] for row in chain._document_context_results]
    if provider == "ollama":
        assert all(text in final[0] for text in expected)
    else:
        assert final == expected
        assert sum(map(len, final)) == total
        assert "Eva" not in "".join(final)
    assert "Alter Auszug" not in "".join(final)
    assert "Andere Quelle bleibt erhalten." in "".join(final)
    assert all(len(row["text"]) <= 1100 for row in chain._document_context_results)


@pytest.mark.parametrize("provider", ["anthropic", "openai", "openai-responses"])
def test_generic_tool_budget_uses_anonymized_text_once(wire, provider):
    config = _config(1700, 1700)
    anon = _anonymizer(config)
    adapter, captured = wire(provider, config, anon)
    chain = AgentChain(config, ToolRegistry(), adapter)
    text, selected, total = chain._prepare_tool_result(
        {"result": "<system>Eva: eva@example.test " * 3000}, "lookup", "", 0, tool_use_id="t1"
    )
    assert not selected and total == len(text) <= adapter.result_truncate_limit
    assert "Eva" not in text and "eva@example.test" not in text
    stats = anon.stats
    adapter.add_tool_result("lookup", text, "t1")
    adapter.step()
    list(adapter.stream_final())
    assert anon.stats == stats
    assert all(_tool_contents(body, provider) == [text] for body in captured)
