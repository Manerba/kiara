"""MCP-Ausgaben: Quellenregeln (public/anonymize/blocked), Anonymisierung, Audit, fail-closed."""

import pytest

from kiara import mcp_server
from kiara.services import mcp_anonymization
from kiara.services import retrieval_only as retrieval_only_module
from kiara.services.mcp_anonymization import McpOutputAnonymizer, source_ref
from kiara.services.retrieval_only import RetrievalOnlyError, RetrievalOnlyService

NAME = "Erika Musterfrau"
MAIL = "erika.musterfrau@example.org"
SOURCE = f"/kb/Mail von {NAME}.txt"
ADMIN = {"id": 1, "is_admin": True}


def _config(*, entries=True, policy=True, enabled=True):
    return {
        "web": {
            "mcp_retrieval_enabled": True,
            "mcp_retrieval_allow_raw_chunks": True,
            "mcp_output_cloud_policy": policy,
        },
        "retrieval": {"reranking": False},
        "anonymization": {
            "enabled": enabled,
            "ldap_source": {"enabled": False},
            "erp_source": {"enabled": False},
            "custom_entries": [{"text": NAME, "type": "PERSON"}] if entries else [],
        },
    }


def _chunk(ds_id, index=0, source=SOURCE):
    return {
        "chunk_id": f"c{ds_id}-{index}",
        "text": f"Betreff: Angebot\nVon: {NAME} <{MAIL}>\n---\nHallo zusammen",
        "metadata": {
            "source": source,
            "directory": "/kb",
            "file_type": ".txt",
            "ds_ids": [ds_id],
            "chunk_index": index,
            "chunk_total": 3,
        },
        "score": 0.9,
    }


class FakeRetriever:
    def __init__(self, ds_ids=(1,)):
        self.ds_ids = ds_ids
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return [_chunk(ds_id) for ds_id in self.ds_ids]

    def load_chunks_by_source(self, source, filter_datasource_ids=None):
        self.calls.append(source)
        return [_chunk(1, i, source) for i in range(3)]


class FakeReranker:
    enabled = False


class FakeAudit:
    def create(self, **fields):
        return 1


class FakeDatasourceRepo:
    def get_active(self):
        return [{"id": 1}, {"id": 2}, {"id": 3}]

    def list_active_with_chunks(self):
        return [{"id": 1, "name": f"Postfach {NAME}", "type": "pst"}, {"id": 3, "name": "Gesperrt"}]


class FakePermissionRepo:
    def get_public_datasource_ids(self):
        return [1, 2, 3]


@pytest.fixture
def audits(monkeypatch):
    calls = []
    import kiara.anonymization.audit as audit

    monkeypatch.setattr(audit, "log_anonymization_event", lambda **kw: calls.append(kw))
    return calls


def _service(levels, retriever=None, config=None):
    retriever = retriever or FakeRetriever()
    return RetrievalOnlyService(
        config or _config(),
        retriever_factory=lambda _c, _m: retriever,
        reranker_factory=lambda _c: FakeReranker(),
        permission_repo=FakePermissionRepo(),
        datasource_repo=FakeDatasourceRepo(),
        audit_repo=FakeAudit(),
        embedding_model="test-embed",
        restriction_levels=levels,
    )


def _flat(value) -> str:
    return repr(value)


def test_search_anonymizes_text_source_query_and_writes_audit(audits):
    result = _service({1: "anonymize"}).search_documents(
        ADMIN, {"query": f"Mails von {NAME}", "include_text": True}
    )

    flat = _flat(result)
    assert NAME not in flat
    assert MAIL not in flat
    assert "‹ANON:Person_" in flat and "‹ANON:Email_" in flat
    item = result["results"][0]
    assert item["source_ref"] == source_ref(SOURCE)
    assert result["anonymization"]["applied"] is True
    assert result["anonymization"]["entity_count"] >= 2
    assert len(audits) == 1
    assert audits[0]["route"] == "mcp:search_documents"
    assert audits[0]["correlation_id"] == result["diagnostics"]["correlation_id"]


def test_diagnostics_carry_no_excerpts_when_anonymized(audits):
    class DiagRetriever(FakeRetriever):
        def get_last_retrieval_diagnostics(self):
            return {"paths": {"semantic": {"hits": [{"excerpt": NAME, "source": SOURCE}]}}}

    result = _service({1: "anonymize"}, DiagRetriever()).search_documents(
        ADMIN, {"query": "Angebot"}
    )
    hits = result["diagnostics"]["retrieval_diagnostics"]["paths"]["semantic"]["hits"]
    assert "excerpt" not in hits[0]
    assert NAME not in _flat(result)


def test_public_datasource_stays_plaintext(audits):
    result = _service({1: "public"}).search_documents(
        ADMIN, {"query": "Angebot", "include_text": True}
    )

    assert NAME in result["results"][0]["text"]
    assert result["anonymization"] == {
        "applied": False,
        "entity_count": 0,
        "entity_types": {},
        "raw_items": 1,
    }
    assert audits == []


def test_mixed_sources_anonymize_only_non_public(audits):
    result = _service({1: "public", 2: "anonymize"}, FakeRetriever((1, 2))).search_documents(
        ADMIN, {"query": "Angebot", "include_text": True}
    )

    public, private = result["results"]
    assert NAME in public["text"]
    assert NAME not in private["text"]
    assert result["anonymization"]["raw_items"] == 1


def test_blocked_datasource_is_excluded(audits):
    retriever = FakeRetriever((1, 3))
    service = _service({1: "anonymize", 2: "anonymize", 3: "blocked"}, retriever)

    result = service.search_documents(ADMIN, {"query": "Angebot"})
    assert [r["datasource_ids"] for r in result["results"]] == [[1]]
    assert retriever.calls[0]["filter_datasource_ids"] == [1, 2]

    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(ADMIN, {"query": "Angebot", "datasource_id": 3})
    assert exc.value.code == "datasource_forbidden"

    listed = service.list_datasources(ADMIN)
    assert [d["id"] for d in listed["datasources"]] == [1]
    assert NAME not in _flat(listed)


def test_fail_closed_without_master_data(audits):
    service = _service({1: "anonymize"}, config=_config(entries=False))
    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(ADMIN, {"query": "Angebot"})
    assert exc.value.code == "anonymization_unavailable"
    assert exc.value.http_status == 503


def test_audit_failure_fails_closed(monkeypatch):
    import kiara.anonymization.audit as audit

    def boom(**_kw):
        raise RuntimeError("db weg")

    monkeypatch.setattr(audit, "log_anonymization_event", boom)
    with pytest.raises(RetrievalOnlyError) as exc:
        _service({1: "anonymize"}).search_documents(ADMIN, {"query": "Angebot"})
    assert exc.value.code == "anonymization_audit_failed"


def test_policy_switches(audits):
    off = _service({1: "anonymize", 3: "blocked"}, FakeRetriever((1, 3)), _config(policy=False))
    result = off.search_documents(ADMIN, {"query": "Angebot", "include_text": True})
    assert len(result["results"]) == 2 and NAME in result["results"][0]["text"]

    # anonymization.enabled=false: keine Anonymisierung, blocked bleibt gesperrt
    plain = _service({1: "anonymize", 3: "blocked"}, FakeRetriever((1, 3)), _config(enabled=False))
    result = plain.search_documents(ADMIN, {"query": "Angebot", "include_text": True})
    assert [r["datasource_ids"] for r in result["results"]] == [[1]]
    assert NAME in result["results"][0]["text"]


def test_get_document_context_accepts_source_ref(audits, monkeypatch):
    retriever = FakeRetriever()
    monkeypatch.setattr(
        retrieval_only_module,
        "resolve_source_ref",
        lambda ref, ds_id: SOURCE if ref == source_ref(SOURCE) and ds_id == 1 else None,
    )
    service = _service({1: "anonymize"}, retriever)

    result = service.get_document_context(
        ADMIN, {"source": source_ref(SOURCE), "datasource_id": 1, "chunk_index": 1}
    )
    assert retriever.calls == [SOURCE]
    assert result["source_ref"] == source_ref(SOURCE)
    assert NAME not in _flat(result)
    assert len(result["context_chunks"]) == 3

    with pytest.raises(RetrievalOnlyError) as exc:
        service.get_document_context(
            ADMIN, {"source": "src_" + "0" * 40, "datasource_id": 1, "chunk_index": 1}
        )
    assert exc.value.code == "source_not_found"

    with pytest.raises(RetrievalOnlyError) as exc:
        service.get_document_context(
            ADMIN, {"source": "/kb/‹ANON:Person_001›.txt", "datasource_id": 1, "chunk_index": 1}
        )
    assert exc.value.code == "invalid_request"


def test_query_response_is_anonymized(audits, monkeypatch):
    import kiara.config

    monkeypatch.setattr(kiara.config, "load_config", lambda *a, **k: _config())
    monkeypatch.setattr(
        mcp_anonymization, "datasource_restriction_levels", lambda: {1: "anonymize"}
    )
    response = {
        "answer": f"{NAME} hat geschrieben ({MAIL}).",
        "sources": [{"file": SOURCE, "relevance": 0.9}],
        "route": "local",
        "warnings": [],
    }

    result = mcp_server._anonymize_query_response(ADMIN, "Wer schrieb?", response, "corr-1")

    assert NAME not in _flat(result) and MAIL not in _flat(result)
    assert result["route"] == "local"
    assert result["correlation_id"] == "corr-1"
    assert result["anonymization"]["applied"] is True
    assert audits[0]["route"] == "mcp:query" and audits[0]["correlation_id"] == "corr-1"


def test_query_response_fails_closed(monkeypatch):
    import kiara.config

    monkeypatch.setattr(kiara.config, "load_config", lambda *a, **k: _config(entries=False))
    monkeypatch.setattr(
        mcp_anonymization, "datasource_restriction_levels", lambda: {1: "anonymize"}
    )
    result = mcp_server._anonymize_query_response(
        ADMIN, "Wer?", {"answer": NAME, "sources": [], "route": "local", "warnings": []}, "c"
    )
    assert result["route"] == "error"
    assert NAME not in _flat(result)


def test_research_blocked_while_anonymizing(monkeypatch):
    import kiara.config

    monkeypatch.setattr(kiara.config, "load_config", lambda *a, **k: _config())
    blocked = mcp_server._research_blocked_response()
    assert blocked["status"] == "failed"
    monkeypatch.setattr(kiara.config, "load_config", lambda *a, **k: _config(policy=False))
    assert mcp_server._research_blocked_response() is None


def test_level_for_rules():
    anon = McpOutputAnonymizer(_config(), {1: "public", 2: "anonymize", 3: "blocked"})
    assert anon.level_for([1]) == "public"
    assert anon.level_for([1, 2]) == "anonymize"
    assert anon.level_for([2, 3]) == "blocked"
    assert anon.level_for([99]) == "anonymize"
    assert anon.without_blocked(None) == [1, 2]
    assert anon.needs_anonymization([1]) is False
    assert anon.needs_anonymization(None) is True


def test_source_ref_is_stable_and_validated():
    ref = source_ref(SOURCE)
    assert ref == source_ref(SOURCE) and mcp_anonymization.is_source_ref(ref)
    assert not mcp_anonymization.is_source_ref("src_xyz")
    assert NAME not in ref


class _PhoneRetriever(FakeRetriever):
    TEXT = "Bitte zurückrufen. Telefon: +49 (07126) 92 123 45, Fax folgt."

    def retrieve(self, **kwargs):
        chunk = _chunk(1)
        chunk["text"] = self.TEXT
        return [chunk]

    def get_last_retrieval_diagnostics(self):
        hit = {"excerpt": self.TEXT, "text": self.TEXT, "metadata": {"source": SOURCE}, "score": 1}
        return {"paths": {"semantic": {"hits": [hit]}}, "final_results": [hit]}


def test_anonymize_before_cut_leaves_no_fragment(audits):
    cut_inside_number = _PhoneRetriever.TEXT.index("92") + 2
    result = _service({1: "anonymize"}, _PhoneRetriever()).search_documents(
        ADMIN,
        {"query": "Rückruf", "include_text": True, "max_chars_per_result": cut_inside_number},
    )

    item = result["results"][0]
    for field in ("text", "excerpt"):
        assert "+49" not in item[field] and "07126" not in item[field]
    assert "‹ANON:" not in item["text"].rsplit("›", 1)[-1]  # kein angeschnittener Platzhalter

    full = _service({1: "anonymize"}, _PhoneRetriever()).search_documents(
        ADMIN, {"query": "Rückruf", "include_text": True}
    )
    assert "‹ANON:Telefon_001›" in full["results"][0]["text"]


def test_cut_never_splits_placeholder():
    from kiara.services.retrieval_only import _cut

    text = "Anruf von ‹ANON:Person_001› heute"
    assert _cut(text, 15) == "Anruf von "
    assert _cut(text, len(text)) == text
    assert _cut(text, 28) == "Anruf von ‹ANON:Person_001› "


def test_diagnostic_copies_carry_no_text_even_for_public(audits):
    result = _service({1: "public"}, _PhoneRetriever()).search_documents(
        ADMIN, {"query": "Rückruf", "include_text": True}
    )

    diagnostics = repr(result["diagnostics"])
    assert "Telefon" not in diagnostics and SOURCE not in diagnostics
    assert "+49" in result["results"][0]["text"]  # public bleibt Klartext
