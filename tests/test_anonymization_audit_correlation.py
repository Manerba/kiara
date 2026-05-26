"""Regression tests for audit_anonymization correlation_id wiring."""

from types import SimpleNamespace
from unittest.mock import patch


def test_log_anonymization_event_persists_correlation_id():
    from kiara.anonymization import audit as audit_module

    with patch.object(audit_module._anon_audit_repo, "create") as mock_create:
        audit_module.log_anonymization_event(
            user_id=1,
            session_id="corr-123",
            question="Wer ist Max Mustermann?",
            entity_count=1,
            entity_types={"PERSON": 1},
            route="cloud",
            cloud_provider="openai",
            cloud_model="gpt-4o",
            correlation_id="corr-123",
        )

    kwargs = mock_create.call_args.kwargs
    assert kwargs["session_id"] == "corr-123"
    assert kwargs["correlation_id"] == "corr-123"


def test_log_anonymization_event_accepts_system_initiated_event():
    from kiara.anonymization import audit as audit_module

    with patch.object(audit_module._anon_audit_repo, "create") as mock_create:
        audit_module.log_anonymization_event(
            user_id=None,
            session_id="corr-system",
            question="System-Enrichment",
            entity_count=0,
            entity_types={},
            route="enrichment:cloud",
            correlation_id="corr-system",
        )

    kwargs = mock_create.call_args.kwargs
    assert kwargs["user_id"] is None
    assert kwargs["session_id"] == "corr-system"
    assert kwargs["correlation_id"] == "corr-system"


def test_chat_cloud_audit_reuses_correlation_id_for_anonymization_event():
    from kiara.web.api_chat import _log_cloud_audit

    with (
        patch("kiara.cloud.audit.AuditLogger") as mock_audit_cls,
        patch("kiara.anonymization.audit.log_anonymization_event") as mock_anon,
    ):
        mock_audit_cls.return_value.log_request.return_value = 42

        _log_cloud_audit(
            config={},
            user_id=1,
            question="Raw question",
            answer_text="Raw answer",
            anon_question="Anon question",
            anon_answer="Anon answer",
            provider="openai",
            model_name="gpt-4o",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            entities_count=2,
            duration_ms=123,
            chain_type="standard",
            pseudonymizer_stats={"total_entities": 2, "entity_types": {"PERSON": 2}},
            correlation_id="corr-chat",
        )

    cloud_kwargs = mock_audit_cls.return_value.log_request.call_args.kwargs
    anon_kwargs = mock_anon.call_args.kwargs
    assert cloud_kwargs["correlation_id"] == "corr-chat"
    assert anon_kwargs["session_id"] == "corr-chat"
    assert anon_kwargs["correlation_id"] == "corr-chat"


def test_mcp_cloud_audit_reuses_correlation_id_for_anonymization_event():
    from kiara.mcp_server import _log_mcp_cloud_audit

    backend = SimpleNamespace(
        provider="openai",
        model="gpt-4o",
        anonymizer_stats={"total_entities": 3, "entity_types": {"PERSON": 3}},
    )
    chain = SimpleNamespace(
        last_usage={"prompt_tokens": 20, "completion_tokens": 8},
        backend=backend,
        last_entities_anonymized=3,
        last_anon_question="Anon question",
        last_anon_answer="Anon answer",
    )

    with (
        patch("kiara.cloud.audit.AuditLogger") as mock_audit_cls,
        patch("kiara.config.load_config", return_value={}),
        patch("kiara.anonymization.audit.log_anonymization_event") as mock_anon,
    ):
        _log_mcp_cloud_audit(
            user={"id": 1},
            question="Raw question",
            answer="Raw answer",
            sources=[{"source": "doc.md"}],
            chain=chain,
            duration_ms=456,
            correlation_id="corr-mcp",
        )

    cloud_kwargs = mock_audit_cls.return_value.log_request.call_args.kwargs
    anon_kwargs = mock_anon.call_args.kwargs
    assert cloud_kwargs["correlation_id"] == "corr-mcp"
    assert anon_kwargs["session_id"] == "corr-mcp"
    assert anon_kwargs["correlation_id"] == "corr-mcp"


def test_audit_schema_migration_adds_correlation_id_column_and_index():
    from kiara.web.schema import _migrate_audit_schema

    calls = []

    class FakeConn:
        def execute(self, sql):
            calls.append(sql)

    _migrate_audit_schema(FakeConn())

    assert any("ALTER COLUMN user_id DROP NOT NULL" in sql for sql in calls)
    assert any("ADD COLUMN IF NOT EXISTS correlation_id" in sql for sql in calls)
    assert any("idx_audit_anon_corr_id" in sql for sql in calls)
