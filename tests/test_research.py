"""Vollstaendige Recherche-Laeufe mit kontrollierten Originalen und Modellantworten."""

import copy
import json
from itertools import pairwise

import pytest

from kiara.generation.context_selection import ModelCallBudget
from kiara.generation.research_chain import ResearchChain
from kiara.services.research import ResearchResults
from kiara.shared.processing_warnings import ProcessingWarnings, processing_warning_scope
from kiara.tools.builtin_tools import SearchDocumentsTool
from kiara.tools.document_tools import DocumentReferences, ReadDocumentTool


def document(source, texts):
    return [
        {
            "id": f"{source}:{i}",
            "text": text,
            "score": 1,
            "metadata": {
                "source": source,
                "ds_ids": [1],
                "embedding_model": "fixture",
                "chunk_index": i,
                "chunk_total": len(texts),
                "parser": "standard",
                "modified": "2026-10-21",
                "size_bytes": sum(map(len, texts)),
            },
        }
        for i, text in enumerate(texts)
    ]


class Retriever:
    def __init__(self, rounds, documents):
        self.rounds, self.documents = rounds, documents
        self.queries, self.loads = [], []
        self.changed = False

    def retrieve(self, **kwargs):
        self.queries.append(kwargs)
        return copy.deepcopy(self.rounds[min(len(self.queries) - 1, len(self.rounds) - 1)])

    def validate_context(self, chunks, allowed):
        assert allowed == [1]

    def load_document_range(self, source, low, high, **kwargs):
        self.loads.append((source, low, high))
        data = copy.deepcopy(self.documents[source][low:high])
        if self.changed:
            data[0]["metadata"]["modified"] = "changed"
        return data


class Backend:
    provider, model, _url = "ollama", "fixture-llm", "http://fixture"
    last_usage = {"prompt_tokens": 100, "response_tokens": 20}

    def __init__(self, relevant="Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt."):
        self.messages = []
        self.relevant = relevant
        self.bad_quote = False
        self.reader_error = False
        self.verify = True
        self.hook = None

    def generate(self, system, user, **kwargs):
        self.messages.append((system, user, kwargs))
        if self.hook:
            self.hook(system, user)
        if "Analysiere die Frage" in system:
            return '{"entities":[]}'
        if "Du planst" in system:
            return json.dumps({"query": f"Freigabe Frist Nachweis {len(self.messages)}"})
        if "Lese-Unteragent" in system:
            if self.reader_error:
                return "kein JSON"
            relevant = self.relevant in user
            return json.dumps(
                {
                    "verdict": "relevant" if relevant else "not_relevant",
                    "quotes": ["erfunden" if self.bad_quote else self.relevant] if relevant else [],
                    "open_question": "",
                }
            )
        if "Antwortentwurf" in system:
            return json.dumps({"valid": self.verify})
        ids = __import__("re").findall(r'"id":"(E\d+)"', user)
        return json.dumps(
            {"supported": True, "answer": self.relevant, "evidence_ids": ids[:1], "conflict": False}
        )


@pytest.fixture
def build(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kiara.ingestion.document_publication.assert_documents_published", lambda *a: None
    )

    def make(rounds=None, docs=None, **settings):
        irrelevant = document("handbuch.txt", ["Der Betrieb findet montags statt."])
        target = document(
            "vertrag.txt", ["Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt."]
        )
        rounds = rounds if rounds is not None else [irrelevant, target]
        docs = docs if docs is not None else {"handbuch.txt": irrelevant, "vertrag.txt": target}
        config = {
            "chat": {
                "research": {
                    "max_rounds": 2,
                    "reads_per_round": 3,
                    "lock_dir": str(tmp_path),
                    **settings,
                },
                "agent": {"document_read_max_chunks": 4},
            },
            "retrieval": {"reranking": False},
        }
        retriever = Retriever(rounds, docs)
        refs = DocumentReferences()
        search = SearchDocumentsTool(
            retriever, None, config, [1], default_k=20, document_references=refs
        )
        read = ReadDocumentTool(retriever, config, refs, [1])
        chain = ResearchChain(
            config,
            Backend(),
            search,
            read,
            authorize=lambda: [1],
            question="Welche Frist gilt nach Freigabe?",
            model_window=32768,
        )
        return chain, retriever

    return make


async def run(chain):
    warnings = ProcessingWarnings()
    with processing_warning_scope(warnings):
        events = [e async for e in chain.run()]
    return chain.result, events, warnings.snapshot()


@pytest.mark.asyncio
async def test_complete_late_evidence_separate_contexts_and_reload(build):
    chain, retriever = build()
    result, events, warnings = await run(chain)
    assert result["status"] == "supported"
    assert "17 Tage" in result["answer"]
    assert len(result["searches"]) == 2
    assert result["sources"][0]["chunk_ids"] == ["vertrag.txt:0"]
    assert len(retriever.loads) >= 4  # Leser und zwei Originalpruefungen
    readers = [u for s, u, _ in chain.backend.messages if "Lese-Unteragent" in s]
    assert "montags" in readers[0] and "17 Tage" not in readers[0]
    assert "montags" not in readers[1] and "17 Tage" in readers[1]
    assert all(c["budget"]["fits"] for c in result["model_calls"])
    assert all(c["budget"]["exact_token_check"] is False for c in result["model_calls"])
    assert all(
        kw["max_tokens"] == 2048 and kw["think"] is False for _, _, kw in chain.backend.messages
    )
    assert all(q["filter_source"] is None for q in retriever.queries)
    assert warnings[0]["details"]["status"] == "supported"
    assert [e["phase"] for e in events].count("search") == 2


@pytest.mark.asyncio
async def test_true_negative_is_scoped(build):
    chain, _ = build(rounds=[[]], docs={})
    result, _, _ = await run(chain)
    assert result["status"] == "not_found_in_scope"
    assert not result["sources"]
    assert "geprüften" in result["answer"]


@pytest.mark.asyncio
async def test_conflicting_originals_are_both_visible(build):
    first = document("vertrag.txt", ["Die Frist beträgt 17 Tage."])
    second = document("nachtrag.txt", ["Die Frist beträgt 19 Tage."])
    chain, _ = build(rounds=[first, second], docs={"vertrag.txt": first, "nachtrag.txt": second})
    original = chain.backend.generate

    def generate(system, user, **kwargs):
        if "Lese-Unteragent" in system:
            quote = first[0]["text"] if first[0]["text"] in user else second[0]["text"]
            return json.dumps({"verdict": "relevant", "quotes": [quote], "open_question": ""})
        if "Du pruefst Originalbelege" in system:
            assert first[0]["text"] in user and second[0]["text"] in user
            return json.dumps(
                {
                    "supported": True,
                    "answer": "Die Quellen widersprechen sich: 17 Tage beziehungsweise 19 Tage.",
                    "evidence_ids": ["E1", "E2"],
                    "conflict": True,
                }
            )
        return original(system, user, **kwargs)

    chain.backend.generate = generate
    result, _, _ = await run(chain)
    assert result["status"] == "incomplete"
    assert result["evidence_status"] == "supported"
    assert "conflicting_evidence" in result["limitations"]
    assert len(result["sources"]) == 2


@pytest.mark.asyncio
async def test_number_substring_is_not_a_verified_value(build):
    chain, _ = build()
    original = chain.backend.generate

    def generate(system, user, **kwargs):
        response = original(system, user, **kwargs)
        if "Du pruefst Originalbelege" in system:
            response = response.replace("17 Tage", "7 Tage")
        return response

    chain.backend.generate = generate
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "not_established"
    assert not result["sources"]
    assert not any(c["phase"] == "verifier" for c in result["model_calls"])


@pytest.mark.asyncio
async def test_original_identifier_after_joined_pdf_label(build):
    text = "Kundennummer.N123456\nGültig ab Freigabe."
    target = document("konto.txt", [text])
    chain, _ = build(rounds=[target], docs={"konto.txt": target}, max_rounds=1)
    chain.question = "Welche Kundennummer gilt?"
    chain.backend.relevant = text
    original = chain.backend.generate

    def generate(system, user, **kwargs):
        response = original(system, user, **kwargs)
        if "Du pruefst Originalbelege" in system:
            draft = json.loads(response)
            draft["answer"] = "Die Kundennummer lautet N123456."
            return json.dumps(draft)
        return response

    chain.backend.generate = generate
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "supported"


@pytest.mark.asyncio
async def test_answer_can_use_verified_original_beyond_reader_quote(build):
    target = document(
        "vertrag.txt",
        ["Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt. Die Gebühr beträgt 25 Euro."],
    )
    chain, _ = build(rounds=[target], docs={"vertrag.txt": target}, max_rounds=1)
    original = chain.backend.generate

    def generate(system, user, **kwargs):
        response = original(system, user, **kwargs)
        if "Du pruefst Originalbelege" in system:
            draft = json.loads(response)
            draft["answer"] += " Die Gebühr beträgt 25 Euro."
            return json.dumps(draft)
        return response

    chain.backend.generate = generate
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "supported"
    assert "25 Euro" in result["sources"][0]["original_excerpt"]
    assert "25 Euro" not in result["sources"][0]["excerpt"]
    assert any(c["phase"] == "verifier" for c in result["model_calls"])


@pytest.mark.asyncio
async def test_duplicate_originals_do_not_displace_other_evidence(build):
    text = "Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt."
    first, second = document("vertrag.txt", [text]), document("kopie.txt", [text])
    chain, _ = build(
        rounds=[first + second],
        docs={"vertrag.txt": first, "kopie.txt": second},
        max_rounds=1,
        max_evidence=1,
    )
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "supported"
    assert result["evidence_aliases"] == {"E1": "E2"}
    assert "final_evidence_limit" not in result["limitations"]


@pytest.mark.asyncio
async def test_omitted_positive_original_blocks_unchecked_conflict(build):
    text = "Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt."
    first = document("vertrag.txt", [text])
    second = document("nachtrag.txt", [text + " Nachtrag: Jetzt gelten 19 Tage."])
    chain, _ = build(
        rounds=[first + second],
        docs={"vertrag.txt": first, "nachtrag.txt": second},
        max_rounds=1,
        max_evidence=1,
    )
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "not_established"
    assert "final_evidence_limit" in result["limitations"]
    assert not result["sources"]
    assert not any(c["phase"] == "answer" for c in result["model_calls"])


@pytest.mark.asyncio
async def test_timeout_result_is_immutable_after_worker_finishes(build):
    import asyncio
    import time

    chain, _ = build(max_seconds=1)
    chain.started = time.monotonic() - 0.8
    original = chain.backend.generate

    def slow(*args, **kwargs):
        time.sleep(0.4)
        return original(*args, **kwargs)

    chain.backend.generate = slow
    result, _, _ = await run(chain)
    snapshot = json.dumps(result)
    assert "time_limit" in result["limitations"]
    await asyncio.sleep(0.3)
    assert chain.calls[0].get("error")
    assert json.dumps(result) == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["quote", "parse", "verifier", "version", "acl"])
async def test_fail_closed_evidence(build, kind):
    chain, retriever = build()
    if kind == "quote":
        chain.backend.bad_quote = True
    if kind == "parse":
        chain.backend.reader_error = True
    if kind == "verifier":
        chain.backend.verify = False
    if kind == "version":
        chain.backend.hook = lambda s, u: (
            setattr(retriever, "changed", True) if "17 Tage" in u else None
        )
    if kind == "acl":

        def deny(s, u):
            if "17 Tage" in u:
                chain.authorize = lambda: []

        chain.backend.hook = deny
    result, _, _ = await run(chain)
    assert result["status"] == "incomplete"
    assert result["evidence_status"] == "not_established"
    assert not result["sources"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "setting,value,reason",
    [
        ("max_model_calls", 1, "model_call_limit"),
        ("store_bytes", 10, "store_limit"),
        ("num_ctx", 1024, "question_or_history_exceeds_window"),
    ],
)
async def test_limits_prevent_calls_and_are_visible(build, setting, value, reason):
    chain, _ = build(**{setting: value})
    result, _, warnings = await run(chain)
    assert result["status"] in {"incomplete", "failed"}
    assert reason in result["limitations"]
    assert warnings[0]["details"]["limitations"] == result["limitations"]
    if setting == "max_model_calls":
        assert len(chain.backend.messages) == 1
    if setting == "num_ctx":
        assert not chain.backend.messages


@pytest.mark.asyncio
async def test_long_history_rejected_before_model(build):
    chain, _ = build()
    chain.history = "Vorheriger Inhalt " * 10000
    result, _, _ = await run(chain)
    assert result["status"] == "failed"
    assert not chain.backend.messages


@pytest.mark.asyncio
async def test_continuation_reads_late_section(build):
    target = document(
        "lang.txt",
        [
            "Anfang. " * 150,
            "Mittlerer Teil. " * 150,
            "Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt.",
        ],
    )
    chain, retriever = build(
        rounds=[[target[0]], []], docs={"lang.txt": target}, page_chars=1800, reads_per_round=10
    )
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "supported"
    ranges = [r["range"] for r in result["reads"]]
    assert len(ranges) > 1
    for record in result["reads"]:
        assert record["chunk_ids"] == [
            f"lang.txt:{i}"
            for i in range(record["range"]["start_chunk"], record["range"]["end_chunk"] + 1)
        ]
    for left, right in pairwise(ranges):
        assert (left["next_chunk"], left["next_offset"]) == (
            right["start_chunk"],
            right["start_offset"],
        )
    assert result["coverage"]["complete_within_found_documents"]


@pytest.mark.asyncio
async def test_unread_sections_never_negative(build):
    doc = document(
        "lang.txt",
        ["Vorbemerkung. " * 1000, "Die Frist beträgt 17 Tage, sofern die Freigabe vorliegt."],
    )
    chain, _ = build(
        rounds=[[doc[0]], []], docs={"lang.txt": doc}, page_chars=1000, reads_per_round=1
    )
    result, _, _ = await run(chain)
    assert result["status"] == "incomplete"
    assert result["coverage"]["pending_pages"] > 0
    assert result["evidence_status"] == "not_established"


@pytest.mark.asyncio
async def test_mcp_bounded_pages_exact_reassembly_acl_and_expiry(build):
    chain, _ = build()
    await run(chain)
    cache = ResearchResults()
    cache.add(chain, {"id": 1})
    offset, parts = 0, []
    while offset is not None:
        page = await cache.page(chain.run_id, {"id": 1}, offset, 2000)
        assert len(json.dumps(page, ensure_ascii=False)) <= 2000
        assert page["client_context_checked"] is False
        parts.append(page["content"])
        offset = page["next_offset"]
    assert json.loads("".join(parts))["answer"] == chain.result["answer"]
    with pytest.raises(ValueError):
        await cache.page(chain.run_id, {"id": 2})
    expiry, owner, _ = cache._entries[chain.run_id]
    cache._entries[chain.run_id] = (0, owner, chain)
    with pytest.raises(ValueError):
        await cache.page(chain.run_id, {"id": 1})


def test_token_budget_counts_history_tools_unicode_and_reserves():
    budget = ModelCallBudget(5000, 1000, 500)
    assert budget.inspect("system", "Frage")["fits"]
    assert not budget.inspect("system", "Frage", "工具" * 2000)["fits"]
    assert (
        budget.inspect("", "ö" * 100)["input_tokens"]
        > budget.inspect("", "a" * 100)["input_tokens"]
    )


def test_quote_whitespace_is_recovered_from_original_not_invented():
    from kiara.generation.research_chain import _original_quote

    text = "Der Betrag ist 7,20\u00a0EUR.\nNur nach\nFreigabe."
    assert (
        _original_quote(text, "7,20 EUR. Nur nach Freigabe.")
        == "7,20\u00a0EUR.\nNur nach\nFreigabe."
    )
    with pytest.raises(ValueError):
        _original_quote(text, "Der Betrag ist 7,30 EUR.")
    with pytest.raises(ValueError):
        _original_quote(text, "Ohne Freigabe.")


@pytest.mark.asyncio
async def test_cancellation_stops_followup(build):
    chain, _ = build()
    with processing_warning_scope(ProcessingWarnings()):
        stream = chain.run()
        await anext(stream)
        await stream.aclose()
    assert chain.cancelled.is_set()
    assert len(chain.backend.messages) == 1  # Nur Fragezuordnung, keine Recherche
    assert "cancelled" in chain.result["limitations"]


@pytest.mark.asyncio
async def test_named_entity_without_original_never_becomes_positive(build):
    chain, _ = build()
    chain.question = "Welche Frist nennt Nordwerk?"
    original = chain.backend.generate

    def generate(system, user, **kwargs):
        if "Analysiere die Frage" in system:
            return '{"entities":["Nordwerk"],"field":"Frist"}'
        return original(system, user, **kwargs)

    chain.backend.generate = generate
    result, _, _ = await run(chain)
    assert result["evidence_status"] == "not_established"
    assert not result["sources"]


@pytest.mark.asyncio
async def test_chat_stream_persists_same_status_as_core(build, monkeypatch):
    from kiara.shared.processing_warnings import current_processing_warnings
    from kiara.web import api_chat

    chain, _ = build()
    saved, audited = [], []
    monkeypatch.setattr("kiara.services.research.create_research", lambda *a, **kw: chain)
    monkeypatch.setattr(
        api_chat,
        "_save_successful_assistant_message",
        lambda *a: saved.append((a, current_processing_warnings())),
    )
    monkeypatch.setattr(api_chat, "_log_query_audit", lambda *a, **kw: audited.append(kw))
    request = api_chat.QueryRequest(question=chain.question, mode="research")
    events = [event async for event in api_chat._stream_research({}, request, 42, {"id": 1})]
    payloads = [
        json.loads(e.split("data: ", 1)[1]) for e in events if "event: research_status" in e
    ]
    assert payloads[0]["status"] == chain.result["status"]
    sources = next(json.loads(e.split("data: ", 1)[1]) for e in events if "event: sources" in e)
    assert isinstance(sources, list) and sources[0]["verified_original"]
    assert saved[0][0][1] == chain.result["answer"]
    assert saved[0][1][0]["details"]["status"] == chain.result["status"]
    assert audited[0]["chain_type"] == "research"
    assert audited[0]["correlation_id"] == chain.run_id
    assert "event: done" in events[-1]
    assert api_chat._resolve_template("research", 999)[0] == "research"


def test_authenticated_mcp_tool_full_run(build, monkeypatch):
    from fastapi.testclient import TestClient

    from kiara.mcp_server import _create_app

    chain, _ = build()
    monkeypatch.setattr("kiara.services.research.create_research", lambda *a, **kw: chain)
    monkeypatch.setattr(
        "kiara.web.api_key_auth.validate_api_key", lambda key: {"id": 1, "is_admin": True}
    )
    app = _create_app({"web": {"mcp_enabled": True}})
    audits = []
    monkeypatch.setattr(
        "kiara.services.research.audit_research", lambda user, run: audits.append(run.run_id)
    )
    with TestClient(app) as client:
        headers = {
            "Authorization": "Bearer orag_test_key",
            "Accept": "application/json, text/event-stream",
        }
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "research",
                "arguments": {"question": chain.question, "max_chars": 2000},
            },
        }
        response = client.post("/mcp", json=request, headers=headers)
        assert response.status_code == 200
        assert len(response.text) <= 2000
        envelope = next(
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        )
        result = envelope["result"]["structuredContent"]
        assert result["route"] == "research"
        assert result["status"] == "supported"
        assert audits == [chain.run_id]
        assert result["next_offset"]
        assert len(json.dumps(result, ensure_ascii=False)) <= 2000
        request["params"] = {
            "name": "research_result",
            "arguments": {
                "run_id": result["run_id"],
                "offset": result["next_offset"],
                "max_chars": 2000,
            },
        }
        response = client.post("/mcp", json=request, headers=headers)
        assert response.status_code == 200
        envelope = next(
            json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
        )
        assert envelope["result"]["structuredContent"]["offset"] == result["next_offset"]


def test_content_keyword_operator_is_opt_in():
    from kiara.retrieval.retriever import HybridRetriever, _FilterParams

    retriever = object.__new__(HybridRetriever)
    retriever._es_fuzziness = None
    retriever._config = {"retrieval": {}}
    baseline = retriever._build_es_query_body("Beispiel", 5, _FilterParams())
    assert "operator" not in baseline["query"]["bool"]["must"][0]["multi_match"]
    retriever._config["retrieval"] = {"keyword_content_only": True, "keyword_match_operator": "and"}
    query = retriever._build_es_query_body("Nordwerk Eva Klein", 5, _FilterParams())
    match = query["query"]["bool"]["must"][0]["multi_match"]
    assert match["fields"] == ["text"]
    assert match["operator"] == "and"


@pytest.mark.asyncio
async def test_request_deadline_stops_before_inference(build):
    chain, _ = build()
    chain.started -= chain.settings["max_seconds"] + 1
    result, _, _ = await run(chain)
    assert result["status"] == "failed"
    assert result["limitations"] == ["time_limit"]
    assert not chain.backend.messages


@pytest.mark.asyncio
async def test_model_discovery_obeys_total_deadline(build):
    import time
    from types import SimpleNamespace

    chain, _ = build(max_seconds=1)
    chain.started = time.monotonic() - 0.85
    chain._model_window = None

    def slow_show(model):
        time.sleep(0.3)
        return {"model_info": {"fixture.context_length": 32768}}

    chain.backend._make_client = lambda timeout: SimpleNamespace(show=slow_show)
    result, _, _ = await run(chain)
    assert result["status"] == "failed"
    assert "time_limit" in result["limitations"]
    assert not chain.backend.messages


@pytest.mark.asyncio
async def test_result_page_outlives_execution_deadline(build):
    chain, _ = build()
    await run(chain)
    chain.started -= chain.settings["max_seconds"] + 1
    cache = ResearchResults()
    cache.add(chain, {"id": 1})
    page = await cache.page(chain.run_id, {"id": 1}, max_chars=2000)
    assert page["status"] == "supported"


def test_inference_is_serialized_across_independent_contexts(build):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    first, _ = build()
    second, _ = build()
    active = 0
    maximum = 0
    guard = Lock()

    def generate(*args, **kwargs):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.1)
        with guard:
            active -= 1
        return '{"ok":true}'

    first.backend.generate = second.backend.generate = generate
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(c._invoke, "system", "frage", "test") for c in [first, second]]
        assert all(f.result()["ok"] for f in futures)
    assert maximum == 1
