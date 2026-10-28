"""Negativtests fuer S4-Guardfehler an breiten Exception-Grenzen."""

import json
import threading
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _compatibility_error(operation: str = "guarded_consumer"):
    from kiara.retrieval.index_compatibility import (
        IndexCompatibilityError,
        IndexVectorKind,
        make_index_target,
    )

    target = make_index_target(
        {"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        collection_name="kb_broad_except_guard_test",
        vector_kind=IndexVectorKind.DENSE,
        requested_model="broad-except-test",
    )
    return IndexCompatibilityError(
        "index_contract_mismatch",
        operation,
        target,
        "injected broad-except compatibility failure",
        current_state="mismatch",
    )


def _sse_payload(event: str) -> tuple[str, dict]:
    lines = event.strip().splitlines()
    return lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))


def test_run_all_datasources_propagates_guard_and_stops_before_next_datasource():
    from kiara.ingestion.pipeline import run_all_datasources

    error = _compatibility_error("run_all_datasources")
    connector = MagicMock()
    connector.HASH_MODE = "mtime"
    connector.safe_prepare_and_scan.return_value = "/tmp"
    rows = [
        {"id": 1, "name": "first", "type": "local"},
        {"id": 2, "name": "second", "type": "local"},
    ]

    with (
        patch("kiara.connectors.registry.ensure_loaded"),
        patch("kiara.connectors.registry.get", return_value=connector),
        patch(
            "kiara.repositories.datasource.DatasourceRepository.get_active",
            return_value=rows,
        ),
        patch("kiara.ingestion.pipeline.load_connector_config", return_value={}),
        patch("kiara.ingestion.pipeline.build_datasource_overrides", return_value={}),
        patch("kiara.ingestion.pipeline.run_ingestion", side_effect=error) as run,
    ):
        with pytest.raises(type(error)) as raised:
            run_all_datasources({"datasource": {}}, incremental=True)

    assert raised.value is error
    run.assert_called_once()
    connector.safe_prepare_and_scan.assert_called_once()


def test_worker_outer_boundary_preserves_guard_diagnostic_and_finalizes_error():
    from kiara.web.ingestion_worker import IngestionWorker

    error = _compatibility_error("worker_context_load")
    worker = IngestionWorker()
    worker._lock = threading.Lock()
    worker._stop = threading.Event()
    worker._current_job_id = None
    worker._build_check_control = MagicMock(return_value=lambda: None)
    worker._load_job_context = MagicMock(side_effect=error)
    worker._finalize_error = MagicMock()

    worker._execute_job(
        {
            "id": 17,
            "datasource_id": 23,
            "job_type": "incremental",
            "embedding_model": "broad-except-test",
        }
    )

    worker._finalize_error.assert_called_once()
    args = worker._finalize_error.call_args.args
    kwargs = worker._finalize_error.call_args.kwargs
    assert args[:2] == (17, 23)
    assert error.reason_code in args[2]
    stats = kwargs["stats"]
    assert stats["abort_reason"] == args[2]
    assert stats["index_compatibility_error"] == error.to_dict()
    assert stats["error_details"] == [
        {
            "phase": "index_compatibility",
            "reason_code": error.reason_code,
            "operation": error.operation,
            "consumer": "ingestion_worker",
            "diagnostic": error.to_dict(),
        }
    ]
    assert worker._current_job_id is None


def test_late_chunking_activation_does_not_consult_backend_or_model_heuristics():
    from kiara.ingestion.runtime import _should_use_late_chunking

    error = _compatibility_error("late_backend_selection")
    model = "arbitrary-contract-model"
    with patch("kiara.backend_selector.get_embedding_backend_for_model", side_effect=error):
        assert _should_use_late_chunking(
            {"enrichment": {"late_chunking": True}},
            model,
        )


def test_run_full_check_propagates_guard_without_partial_following_checks():
    from kiara.integrity import checks

    error = _compatibility_error("integrity_layer_counts")
    with (
        patch.object(checks, "check_layer_counts", side_effect=error),
        patch.object(checks, "check_ds_coverage") as following_check,
    ):
        with pytest.raises(type(error)) as raised:
            checks.run_full_check({})

    assert raised.value is error
    following_check.assert_not_called()


def test_enrichment_scroll_propagates_guard_instead_of_partial_report():
    from kiara.integrity.checks import check_enrichment_prompt_consistency

    error = _compatibility_error("enrichment_scroll")
    connections = []
    for _ in range(3):
        conn = MagicMock()
        conn.__enter__.return_value = conn
        conn.__exit__.return_value = False
        connections.append(conn)
    connections[0].execute.return_value.fetchall.return_value = [
        {"id": 1, "name": "DS", "ds_enrichment_prompt_id": None}
    ]
    connections[1].execute.return_value.fetchone.return_value = {"content": "prompt"}
    connections[2].execute.return_value.fetchall.return_value = []
    store = MagicMock()
    store.scroll_all.side_effect = error

    with (
        patch("kiara.shared.db.db_query", side_effect=connections),
        patch("kiara.retrieval.vectorstore.VectorStore") as vectorstore,
    ):
        vectorstore.list_collections.return_value = [
            {"name": "kb_test", "model": "broad-except-test", "chunks": 1}
        ]
        vectorstore.return_value = store
        with pytest.raises(type(error)) as raised:
            check_enrichment_prompt_consistency({})

    assert raised.value is error


def test_integrity_task_boundary_returns_machine_readable_guard_error():
    from kiara.admin.services import integrity_service

    error = _compatibility_error("integrity_check")
    with integrity_service._lock:
        integrity_service._task.update(
            {"running": True, "action": "check", "progress": None, "result": None}
        )

    with (
        patch("kiara.config.load_config", return_value={}),
        patch("kiara.integrity.checks.run_full_check", side_effect=error),
    ):
        integrity_service._run_check()

    result = integrity_service.get_task_status()["result"]
    assert result == {
        "status": "error",
        "error": str(error),
        "reason_code": error.reason_code,
        "operation": error.operation,
        "consumer": "integrity_check",
        "index_compatibility_error": error.to_dict(),
    }


def test_integrity_reconstruct_guard_runs_before_first_scan_mutation():
    from kiara.admin.services import integrity_service

    error = _compatibility_error("integrity_metadata_reconstruct")

    @contextmanager
    def failing_guard(*_args, **_kwargs):
        raise error
        yield  # pragma: no cover

    with integrity_service._lock:
        integrity_service._task.update(
            {
                "running": True,
                "action": "repair_reconstruct",
                "progress": None,
                "result": None,
            }
        )

    with (
        patch(
            "kiara.config.load_config",
            return_value={"vectordb": {"qdrant_url": "http://qdrant.test:6333"}},
        ),
        patch.object(
            integrity_service,
            "_load_datasource_paths",
            return_value=[{"id": 1, "name": "DS", "path": "/data/"}],
        ),
        patch("kiara.retrieval.vectorstore.VectorStore.list_collections") as collections,
        patch(
            "kiara.retrieval.index_compatibility.guard_index",
            side_effect=failing_guard,
        ),
        patch.object(integrity_service, "_scan_qdrant_assignments") as scan,
    ):
        collections.return_value = [{"name": "kb_test", "model": "broad-except-test", "chunks": 1}]
        integrity_service._run_repair_reconstruct()

    scan.assert_not_called()
    result = integrity_service.get_task_status()["result"]
    assert result["status"] == "error"
    assert result["reason_code"] == error.reason_code
    assert result["operation"] == error.operation
    assert result["consumer"] == "integrity_repair_reconstruct"
    assert result["index_compatibility_error"] == error.to_dict()


def test_admin_tool_registry_does_not_omit_search_tool_after_guard_error():
    from kiara.web import api_tools

    error = _compatibility_error("admin_search_tool_init")
    with (
        patch.object(api_tools, "_registry", None),
        patch.object(api_tools, "_connection_registry", None),
        patch.object(api_tools, "load_config", return_value={}),
        patch.object(api_tools, "_get_connection_registry", return_value=MagicMock()),
        patch.object(api_tools, "ERPBackend", return_value=MagicMock()),
        patch("kiara.retrieval.retriever.HybridRetriever", side_effect=error),
    ):
        with pytest.raises(type(error)) as raised:
            api_tools._get_registry()
        assert api_tools._registry is None

    assert raised.value is error


def test_cache_helpers_rethrow_guard_instead_of_marking_refresh_optional():
    from kiara.web.api_utils import safe_refresh
    from kiara.web.cache_refresh import _maybe_refresh

    error = _compatibility_error("cache_refresh")

    def guarded_refresh(*_args):
        raise error

    with pytest.raises(type(error)) as safe_raised:
        safe_refresh(guarded_refresh)
    with pytest.raises(type(error)) as throttled_raised:
        _maybe_refresh("s4-broad-except-test", guarded_refresh, interval=0)

    assert safe_raised.value is error
    assert throttled_raised.value is error


def test_mcp_and_http_boundaries_preserve_full_guard_diagnostic():
    from kiara.mcp_server import (
        _mcp_retrieval_error_response,
        _retrieval_http_error_response,
    )

    error = _compatibility_error("mcp_retrieval")
    mcp_payload = _mcp_retrieval_error_response(error)
    http_response = _retrieval_http_error_response(error)
    http_payload = json.loads(http_response.body)

    assert mcp_payload["ok"] is False
    assert mcp_payload["error"] == {
        "code": "index_compatibility_blocked",
        "message": str(error),
        "reason_code": error.reason_code,
        "operation": error.operation,
        "consumer": "mcp_retrieval_only",
        "index_compatibility_error": error.to_dict(),
    }
    assert http_response.status_code == 409
    assert http_payload["error"]["consumer"] == "http_retrieval_only"
    assert http_payload["error"]["index_compatibility_error"] == error.to_dict()


def test_beir_background_boundary_records_structured_guard_error():
    from kiara.web import api_benchmarks

    error = _compatibility_error("beir_retrieval")
    settings = {
        "dataset_dir": "/tmp/beir",
        "datasource_id": 1,
        "embedding_model": "broad-except-test",
        "retriever_name": "hybrid",
        "rrf_k": 60,
        "elasticsearch_fuzziness": "none",
        "k_values": [1, 5],
        "max_queries": 1,
        "run_label": "guard-test",
        "use_filter_dir": True,
        "chunk_overfetch_factor": 3,
        "reranking_enabled": False,
        "rerank_model": None,
        "rerank_batch_size": 8,
        "rerank_candidate_limit": 20,
        "doc_aggregation": False,
        "doc_aggregation_rrf_k": 60,
        "doc_aggregation_bonus": 0.01,
        "doc_aggregation_max_bonus_chunks": 3,
    }
    with api_benchmarks._beir_task_lock:
        api_benchmarks._beir_task.update(
            {"running": True, "status": "running", "result": {"stale": True}, "error": None}
        )

    with patch.object(api_benchmarks, "evaluate_beir_retrieval", side_effect=error):
        api_benchmarks._run_beir_background(settings)

    result = api_benchmarks._beir_task_snapshot()
    assert result["running"] is False
    assert result["status"] == "error"
    assert result["result"] is None
    assert result["reason_code"] == error.reason_code
    assert result["operation"] == error.operation
    assert result["consumer"] == "beir_background"
    assert result["index_compatibility_error"] == error.to_dict()


@pytest.mark.asyncio
async def test_admin_function_tool_rethrows_guard_instead_of_tool_failure():
    from kiara.admin.function import AdminFunction
    from kiara.admin.function_tool import AdminFunctionTool

    error = _compatibility_error("admin_function_tool")
    fn = AsyncMock(side_effect=error)
    func = AdminFunction(
        name="guarded.admin.function",
        category="test",
        description="test",
        help="test",
        examples=[],
        parameters={"type": "object", "properties": {}},
        returns="test",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method=None,
        http_path=None,
        fn=fn,
    )

    with pytest.raises(type(error)) as raised:
        await AdminFunctionTool(func).execute({}, user_id=1)

    assert raised.value is error


@pytest.mark.asyncio
async def test_integrity_start_endpoint_does_not_catch_guard_as_runtime_error():
    from kiara.admin.functions.integrity import integrity_repair_reconstruct

    error = _compatibility_error("integrity_repair_start")
    with (
        patch("kiara.admin.services.integrity_service.is_running", return_value=False),
        patch("kiara.admin.services.integrity_service.start_repair", side_effect=error),
    ):
        with pytest.raises(type(error)) as raised:
            await integrity_repair_reconstruct({"id": 1})

    assert raised.value is error


@pytest.mark.asyncio
async def test_admin_http_boundary_returns_full_guard_diagnostic():
    from kiara.admin.function import AdminFunction
    from kiara.admin.router import _create_endpoint

    error = _compatibility_error("admin_http")
    func = AdminFunction(
        name="guarded.admin.http",
        category="test",
        description="test",
        help="test",
        examples=[],
        parameters={"type": "object", "properties": {}},
        returns="test",
        requires_confirmation=False,
        confirm_template=None,
        pk_param=None,
        changelog_area=None,
        changelog_action=None,
        http_method="GET",
        http_path="/test",
        fn=AsyncMock(side_effect=error),
    )
    request = MagicMock()
    request.query_params.multi_items.return_value = []
    request.path_params = {}
    request.headers = {}

    response = await _create_endpoint(func)(request, user={"id": 1})
    payload = json.loads(response.body)

    assert response.status_code == 409
    assert payload["data"] == {
        "reason_code": error.reason_code,
        "operation": error.operation,
        "consumer": "admin_function:guarded.admin.http",
        "index_compatibility_error": error.to_dict(),
    }


def test_admin_chat_guard_has_structured_error_done_and_no_success_message():
    from kiara.web import api_admin_chat

    error = _compatibility_error("admin_chat_retrieval")
    message_repo = MagicMock()
    chat_repo = MagicMock()
    confirmation = MagicMock()
    confirmation.check_and_handle.return_value = (False, [])

    with (
        patch("kiara.config.load_config", return_value={}),
        patch.object(api_admin_chat, "AdminChatMessageRepository", return_value=message_repo),
        patch.object(api_admin_chat, "AdminChatRepository", return_value=chat_repo),
        patch(
            "kiara.admin.confirmation.ConfirmationManager",
            return_value=confirmation,
        ),
        patch(
            "kiara.generation.cloud_routing._resolve_backend_dict",
            return_value={"provider": "ollama", "model": "test"},
        ),
        patch("kiara.web.api_chat._create_agent_anonymizer", return_value=None),
        patch("kiara.web.api_chat._cloud_requires_agent_anonymizer", return_value=False),
        patch("kiara.generation.admin_agent_chain.AdminAgentChain", side_effect=error),
    ):
        events = list(
            api_admin_chat._stream_admin_query(
                51,
                api_admin_chat.AdminChatQueryRequest(question="Test?", use_history=False),
                {"id": 1},
            )
        )

    parsed = [_sse_payload(event) for event in events]
    error_payload = next(payload for name, payload in parsed if name == "error")
    done_payloads = [payload for name, payload in parsed if name == "done"]
    assert error_payload["reason_code"] == error.reason_code
    assert error_payload["operation"] == error.operation
    assert error_payload["consumer"] == "admin_chat_tool_retrieval"
    assert error_payload["index_compatibility_error"] == error.to_dict()
    assert done_payloads == [{"chat_id": 51, "status": "error"}]
    assert message_repo.add_message.call_count == 2
    assert message_repo.add_message.call_args_list[-1].kwargs == {"metadata": {"type": "error"}}


@pytest.mark.asyncio
async def test_standard_sse_guard_is_structured_and_has_no_success_epilogue():
    from kiara.web.api_chat import QueryRequest, _stream_standard

    error = _compatibility_error("chat_standard_retrieval")
    with (
        patch("kiara.web.api_chat._prepare_standard_context", side_effect=error),
        patch("kiara.web.api_chat._audit_stream_failure") as audit,
        patch("kiara.web.api_chat._save_successful_assistant_message") as save,
    ):
        events = [
            event
            async for event in _stream_standard(
                {},
                QueryRequest(question="Test?"),
                41,
                {"id": 1},
            )
        ]

    assert len(events) == 2
    error_name, error_payload = _sse_payload(events[0])
    done_name, done_payload = _sse_payload(events[1])
    assert error_name == "error"
    assert error_payload == {
        "message": str(error),
        "reason_code": error.reason_code,
        "operation": error.operation,
        "consumer": "chat_standard_retrieval",
        "index_compatibility_error": error.to_dict(),
    }
    assert done_name == "done"
    assert done_payload == {"chat_id": 41, "status": "error"}
    audit.assert_called_once()
    save.assert_not_called()


@pytest.mark.asyncio
async def test_lazy_standard_stream_guard_is_not_degraded_to_generic_llm_error():
    from kiara.web.api_chat import QueryRequest, _stream_standard

    error = _compatibility_error("chat_lazy_stream")

    def guarded_generator():
        raise error
        yield  # pragma: no cover

    chain = MagicMock()
    chain.query_stream.return_value = (guarded_generator(), [])
    chain.last_expansions = []
    chain.context_warning = None
    chain.route_used = "local"
    chain.last_prompt = ""
    chain.last_trace_id = None
    chain.last_anon_question = ""
    chain.last_anon_answer = ""

    with (
        patch(
            "kiara.web.api_chat._prepare_standard_context",
            return_value=(chain, None, None),
        ),
        patch("kiara.web.api_chat._audit_stream_failure") as audit,
        patch("kiara.web.api_chat._save_successful_assistant_message") as save,
    ):
        events = [
            event
            async for event in _stream_standard(
                {},
                QueryRequest(question="Test?"),
                42,
                {"id": 1},
            )
        ]

    parsed = [_sse_payload(event) for event in events]
    error_payload = next(payload for name, payload in parsed if name == "error")
    done_payloads = [payload for name, payload in parsed if name == "done"]
    assert error_payload["reason_code"] == error.reason_code
    assert error_payload["operation"] == error.operation
    assert error_payload["consumer"] == "chat_standard_stream"
    assert error_payload["index_compatibility_error"] == error.to_dict()
    assert done_payloads == [{"chat_id": 42, "status": "error"}]
    audit.assert_called_once()
    save.assert_not_called()
