"""Tests fuer Retrieval-only MCP/API-Fassade."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from kiara.services.retrieval_only import RetrievalOnlyError, RetrievalOnlyService


def _config(*, enabled=True, allow_raw=False, include_default=False, max_query_length=10_000):
    return {
        "web": {
            "mcp_retrieval_enabled": enabled,
            "mcp_retrieval_allow_raw_chunks": allow_raw,
            "mcp_retrieval_include_text_default": include_default,
            "mcp_retrieval_max_k": 50,
            "mcp_retrieval_max_query_length": max_query_length,
            "mcp_retrieval_max_chars_per_result": 4000,
            "mcp_retrieval_max_context_window": 5,
            "mcp_retrieval_max_context_chunks": 10,
            "mcp_retrieval_max_chars_total": 20000,
        },
        "retrieval": {
            "reranking": False,
            "rerank_fetch_multiplier": 3,
        },
    }


def _app_config(*, mcp_enabled=False, retrieval_enabled=True, oidc_enabled=True):
    return {
        "web": {
            "mcp_enabled": mcp_enabled,
            "mcp_retrieval_enabled": retrieval_enabled,
            "oidc_retrieval": {
                "enabled": oidc_enabled,
                "issuer": "https://id.opendesk.hirschgasse5.de/realms/opendesk",
                "discovery_url": "https://id.opendesk.hirschgasse5.de/realms/opendesk/.well-known/openid-configuration",
                "jwks_url": "",
                "audience": "kiara-retrieval",
                "client_binding_claim": "azp",
                "client_binding_value": "open-webui",
                "algorithms": ["RS256"],
                "identity_claim": "ad_sid",
                "identity_claim_format": "ad_object_sid_or_base64",
                "required_scopes": [],
                "required_roles": [],
                "clock_skew_seconds": 60,
                "ca_bundle_path": "",
                "allow_admin_full_access": False,
            },
        }
    }


class FakePermissionRepo:
    def __init__(self, public_ids=None):
        self.public_ids = public_ids if public_ids is not None else [1]

    def get_public_datasource_ids(self):
        return self.public_ids


class FakeDatasourceRepo:
    def __init__(self):
        self.rows = [
            {"id": 1, "name": "Public Docs", "type": "local", "chunk_count": 10, "file_count": 2},
            {"id": 2, "name": "Team Docs", "type": "smb", "chunk_count": 20, "file_count": 4},
            {"id": 3, "name": "Secret Docs", "type": "local", "chunk_count": 30, "file_count": 6},
        ]

    def get_active(self):
        return [{"id": row["id"]} for row in self.rows]

    def list_active_with_chunks(self):
        return list(self.rows)


class FakeAuditRepo:
    def __init__(self):
        self.calls = []

    def create(self, **fields):
        self.calls.append(fields)
        return 1


class FakeReranker:
    enabled = False

    def rerank(self, query, results, top_k=None):
        return results[:top_k]


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = []
        self.context_calls = []

    def retrieve(self, **kwargs):
        self.retrieve_calls.append(kwargs)
        return [
            {
                "chunk_id": "chunk-canary",
                "text": "Dokumentinhalt nennt KIARA-CANARY-742.txt, aber die Quelle ist die Q-Datei.",
                "metadata": {
                    "source": "/kb/KIARA-CANARY-742-Q.txt",
                    "directory": "/kb",
                    "file_type": ".txt",
                    "ds_ids": [1],
                    "chunk_index": 0,
                    "chunk_total": 1,
                    "title": "Canary",
                },
                "score": 0.91,
                "rrf_score": 0.034,
                "semantic_rank": 1,
                "semantic_score": 0.91,
                "keyword_rank": 2,
                "keyword_score": 7.0,
                "rrf_rank": 1,
                "rrf_contributions": {"semantic": 0.016, "keyword": 0.018},
                "retriever_paths": ["semantic", "keyword"],
            }
        ]

    def get_last_retrieval_diagnostics(self):
        return {
            "schema": "kiara-retrieval-path-diagnostics",
            "paths": {
                "semantic": {"status": "success", "latency_ms": 3, "hits": []},
                "keyword": {"status": "success", "latency_ms": 4, "hits": []},
                "graph": {"status": "disabled", "latency_ms": 0, "hits": []},
                "colbert": {"status": "disabled", "latency_ms": 0, "hits": []},
            },
            "rrf": {"status": "success", "latency_ms": 1},
            "final_results": [
                {
                    "source": "/kb/KIARA-CANARY-742-Q.txt",
                    "rrf_rank": 1,
                    "rrf_score": 0.034,
                    "excerpt": "Diagnose-Excerpt",
                }
            ],
        }

    def load_chunks_by_source(self, source, filter_datasource_ids=None):
        self.context_calls.append((source, filter_datasource_ids))
        return [
            {
                "text": f"Chunk {idx} " + ("x" * 50),
                "metadata": {
                    "source": source,
                    "directory": "/kb",
                    "file_type": ".txt",
                    "ds_ids": [1],
                    "chunk_index": idx,
                    "chunk_total": 5,
                },
                "score": 0.0,
            }
            for idx in range(5)
        ]


def _service(fake_retriever=None, *, config=None, public_ids=None, audit_repo=None):
    fake_retriever = fake_retriever or FakeRetriever()
    return RetrievalOnlyService(
        config or _config(),
        retriever_factory=lambda _config, _embedding_model: fake_retriever,
        reranker_factory=lambda _config: FakeReranker(),
        permission_repo=FakePermissionRepo(public_ids),
        datasource_repo=FakeDatasourceRepo(),
        audit_repo=audit_repo or FakeAuditRepo(),
        embedding_model="test-embed",
    )


def test_auth_required_fail_closed():
    service = _service()

    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(None, {"query": "test"})

    assert exc.value.code == "auth_required"
    assert exc.value.http_status == 401


def test_list_datasources_filters_user_and_public_datasources():
    service = _service(public_ids=[1])

    result = service.list_datasources({"id": 7, "is_admin": False, "datasource_ids": [2]})

    assert result["ok"] is True
    assert [ds["id"] for ds in result["datasources"]] == [1, 2]


def test_forbidden_datasource_is_rejected_before_retrieval():
    retriever = FakeRetriever()
    service = _service(retriever, public_ids=[1])

    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(
            {"id": 7, "is_admin": False, "datasource_ids": [2]},
            {"query": "test", "datasource_id": 3},
        )

    assert exc.value.code == "datasource_forbidden"
    assert retriever.retrieve_calls == []


def test_limits_reject_large_k_and_too_long_query():
    service = _service(config=_config(max_query_length=5))
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    with pytest.raises(RetrievalOnlyError) as k_exc:
        service.search_documents(user, {"query": "test", "k": 51})
    assert k_exc.value.code == "limit_exceeded"

    with pytest.raises(RetrievalOnlyError) as q_exc:
        service.search_documents(user, {"query": "123456"})
    assert q_exc.value.code == "limit_exceeded"


def test_trivial_numeric_query_is_rejected_before_retrieval():
    retriever = FakeRetriever()
    service = _service(retriever)
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(user, {"query": "2026"})

    assert exc.value.code == "query_too_broad"
    assert retriever.retrieve_calls == []


def test_bare_year_source_filter_is_ignored_unless_query_contains_year():
    retriever = FakeRetriever()
    service = _service(retriever)
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    service.search_documents(user, {"query": "Projektstatus", "source_contains": "2026"})

    assert retriever.retrieve_calls[0]["filter_source"] is None


def test_bare_year_source_filter_is_kept_when_query_contains_year():
    retriever = FakeRetriever()
    service = _service(retriever)
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    service.search_documents(user, {"query": "Projektstatus 2026", "source_contains": "2026"})

    assert retriever.retrieve_calls[0]["filter_source"] == "2026"


def test_raw_chunk_policy_rejects_full_text_when_disabled():
    service = _service(config=_config(allow_raw=False))
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    with pytest.raises(RetrievalOnlyError) as exc:
        service.search_documents(user, {"query": "test", "include_text": True})

    assert exc.value.code == "raw_chunks_disabled"


def test_successful_search_returns_metadata_scores_diagnostics_and_audit():
    audit = FakeAuditRepo()
    service = _service(config=_config(allow_raw=True), audit_repo=audit)
    user = {"id": 1, "is_admin": True, "datasource_ids": []}

    result = service.search_documents(user, {"query": "canary", "include_text": True})

    assert result["ok"] is True
    assert "answer" not in result
    hit = result["results"][0]
    assert hit["source"] == "/kb/KIARA-CANARY-742-Q.txt"
    assert hit["path"] == "/kb/KIARA-CANARY-742-Q.txt"
    assert hit["filename"] == "KIARA-CANARY-742-Q.txt"
    assert hit["metadata"]["title"] == "Canary"
    assert hit["datasource_ids"] == [1]
    assert hit["score_type"] == "rrf_score"
    assert hit["scores"]["semantic_score"] == 0.91
    assert hit["ranks"]["final_context_rank"] == 1
    assert hit["rrf_contributions"] == {"semantic": 0.016, "keyword": 0.018}
    assert "text" in hit

    diag = result["diagnostics"]
    assert diag["path_states"]["semantic"] == "success"
    assert diag["path_states"]["rrf"] == "success"
    assert diag["path_latencies_ms"]["keyword"] == 4
    assert diag["retrieval_diagnostics"]["schema"] == "kiara-retrieval-path-diagnostics"

    assert audit.calls
    assert audit.calls[0]["routed_to"] == "mcp:retrieval"
    assert audit.calls[0]["chain_type"] == "retrieval-only"


def test_get_document_context_enforces_acl_window_and_text_limit():
    retriever = FakeRetriever()
    service = _service(retriever, config=_config(allow_raw=True), public_ids=[1])
    user = {"id": 7, "is_admin": False, "datasource_ids": []}

    result = service.get_document_context(
        user,
        {
            "source": "/kb/KIARA-CANARY-742-Q.txt",
            "datasource_id": 1,
            "chunk_index": 2,
            "window": 1,
            "max_chunks": 3,
            "include_text": True,
            "max_chars_total": 80,
        },
    )

    assert retriever.context_calls == [("/kb/KIARA-CANARY-742-Q.txt", [1])]
    assert [c["chunk_index"] for c in result["context_chunks"]] == [1, 2]
    assert sum(len(c.get("text", "")) for c in result["context_chunks"]) <= 80
    assert result["diagnostics"]["tool"] == "get_document_context"
    coverage = result["content_coverage"]
    assert coverage["content_complete"] is False
    assert coverage["available_chunks"] == 3
    assert coverage["returned_chunks"] == 2
    assert coverage["truncated_chunks"] == 1
    assert coverage["omitted_chunks"] == 1
    assert coverage["caller_context_window_known"] is False
    assert result["warnings"][0]["code"] == "retrieval_content_incomplete"


@pytest.mark.parametrize("include_text", [False, True])
def test_search_warns_when_returned_text_is_cut_and_complete_search_is_clean(include_text):
    service = _service(config=_config(allow_raw=True))
    user = {"id": 7, "is_admin": False, "datasource_ids": []}
    params = {"query": "Beleg", "include_text": include_text, "max_chars_per_result": 12}
    partial = service.search_documents(user, params)
    assert partial["content_coverage"]["scope"] == "selected_search_hits"
    assert partial["content_coverage"]["truncated_chunks"] == 1
    assert partial["warnings"][0]["reason"] == (
        "response_limit" if include_text else "excerpt_only"
    )
    complete = service.search_documents(user, {**params, "max_chars_per_result": 4000})
    assert complete["content_coverage"]["content_complete"] is True
    assert complete["warnings"] == []


def test_context_chunk_cap_is_reported_even_without_text_truncation():
    service = _service(config=_config(allow_raw=True))
    result = service.get_document_context(
        {"id": 7, "is_admin": False, "datasource_ids": []},
        {
            "source": "/kb/KIARA-CANARY-742-Q.txt",
            "datasource_id": 1,
            "chunk_index": 2,
            "window": 2,
            "max_chunks": 1,
            "include_text": True,
        },
    )
    assert result["content_coverage"]["omitted_chunks"] == 4
    assert result["content_coverage"]["truncated_chunks"] == 0
    assert result["warnings"]


try:
    from fastmcp import FastMCP  # noqa: F401

    _has_fastmcp = True
except (ImportError, AttributeError):
    _has_fastmcp = False


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_requires_auth():
    from kiara.mcp_server import _create_app

    app = _create_app(_app_config())
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/retrieval/list_datasources")

    assert resp.status_code == 401


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_rejects_orag_api_key():
    from kiara.mcp_server import _create_app

    app = _create_app(_app_config())
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get(
            "/api/retrieval/list_datasources",
            headers={"Authorization": "Bearer orag_valid1234567890"},
        )

    assert resp.status_code == 401


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_disabled_states_are_403():
    from kiara.mcp_server import _create_app

    app = _create_app(_app_config(retrieval_enabled=False, oidc_enabled=True))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/retrieval/list_datasources")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "retrieval_only_disabled"

    app = _create_app(_app_config(retrieval_enabled=True, oidc_enabled=False))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.get("/api/retrieval/list_datasources")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "oidc_retrieval_disabled"


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_mcp_endpoint_disabled_when_mcp_config_false():
    from kiara.mcp_server import _create_app

    app = _create_app(_app_config(mcp_enabled=False, retrieval_enabled=True))
    with TestClient(app, raise_server_exceptions=False) as client:
        resp = client.post(
            "/mcp",
            headers={"Authorization": "Bearer orag_valid1234567890"},
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
        )

    assert resp.status_code == 403


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_uses_authenticated_user():
    from kiara.mcp_server import _create_app

    user = {
        "id": 1,
        "username": "admin",
        "is_admin": True,
        "datasource_ids": [],
        "group_ids": [],
        "api_key_id": 1,
    }
    with (
        patch("kiara.web.oidc_auth.OIDCRetrievalAuthenticator.authenticate", return_value=user),
        patch(
            "kiara.mcp_server._run_retrieval_list_datasources",
            return_value={"ok": True, "datasources": []},
        ) as mock_list,
    ):
        app = _create_app(_app_config())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/api/retrieval/list_datasources",
                headers={"Authorization": "Bearer jwt.header.payload"},
            )

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    mock_list.assert_called_once()
    assert mock_list.call_args.args[0]["id"] == 1


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_oidc_runtime_unavailable_is_503():
    from kiara.mcp_server import _create_app
    from kiara.web.oidc_auth import OIDCUnavailableError

    exc = OIDCUnavailableError("jwks down")
    exc.reason = "jwks_unavailable"
    with patch("kiara.web.oidc_auth.OIDCRetrievalAuthenticator.authenticate", side_effect=exc):
        app = _create_app(_app_config())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.get(
                "/api/retrieval/list_datasources",
                headers={"Authorization": "Bearer jwt.header.payload"},
            )

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "oidc_retrieval_unavailable"


@pytest.mark.skipif(not _has_fastmcp, reason="fastmcp nicht installiert")
def test_retrieval_rest_endpoint_preserves_structured_retrieval_errors():
    from kiara.mcp_server import _create_app

    user = {
        "id": 1,
        "username": "admin",
        "is_admin": True,
        "datasource_ids": [],
        "group_ids": [],
        "api_key_id": 1,
    }
    with (
        patch("kiara.web.oidc_auth.OIDCRetrievalAuthenticator.authenticate", return_value=user),
        patch(
            "kiara.mcp_server._run_retrieval_search",
            side_effect=RetrievalOnlyError("limit_exceeded", "k zu gross", 400),
        ),
    ):
        app = _create_app(_app_config())
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/retrieval/search_documents",
                headers={"Authorization": "Bearer jwt.header.payload"},
                json={"query": "canary", "k": 999},
            )

    assert resp.status_code == 400
    body = resp.json()
    assert body["ok"] is False
    assert body["route"] == "retrieval-only"
    assert body["error"]["code"] == "limit_exceeded"
