"""Tests fuer kiara.shared.tracing und _pg_span_exporter."""

from unittest.mock import MagicMock, patch

import pytest

from kiara.shared.tracing import (
    _NoOpSpan,
    _NoOpTracer,
    attach_context,
    detach_context,
    get_current_context,
    get_current_trace_id,
    get_trace_id_for_span,
    get_tracer,
)


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    """Isoliert Modul-Globals in kiara.shared.tracing zwischen Tests."""
    from kiara.shared import tracing

    old_initialized = tracing._initialized
    old_otel_available = tracing._otel_available
    old_tracer_instance = tracing._tracer_instance
    old_pipeline_flags = tracing._pipeline_flags.copy()
    tracing._initialized = False
    tracing._otel_available = False
    tracing._tracer_instance = None
    tracing._pipeline_flags.clear()
    try:
        yield
    finally:
        tracing._initialized = old_initialized
        tracing._otel_available = old_otel_available
        tracing._tracer_instance = old_tracer_instance
        tracing._pipeline_flags.clear()
        tracing._pipeline_flags.update(old_pipeline_flags)


class TestNoOp:
    """NoOp-Tracer und NoOp-Span Tests."""

    def test_get_tracer_without_init_returns_noop(self):
        tracer = get_tracer("test")
        assert isinstance(tracer, _NoOpTracer)

    def test_noop_span_as_context_manager(self):
        tracer = _NoOpTracer()
        with tracer.start_as_current_span("test") as span:
            assert isinstance(span, _NoOpSpan)

    def test_noop_span_set_attribute(self):
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # Kein Fehler

    def test_noop_span_record_exception(self):
        span = _NoOpSpan()
        span.record_exception(ValueError("test"))  # Kein Fehler

    def test_noop_span_set_status(self):
        span = _NoOpSpan()
        span.set_status("OK")  # Kein Fehler

    def test_noop_span_add_event(self):
        span = _NoOpSpan()
        span.add_event("test_event", {"key": "val"})  # Kein Fehler

    def test_get_current_trace_id_without_otel(self):
        assert get_current_trace_id() is None

    def test_get_trace_id_for_noop_span_returns_none(self):
        assert get_trace_id_for_span(_NoOpSpan()) is None

    def test_get_current_context_without_otel(self):
        assert get_current_context() is None

    def test_attach_context_none(self):
        assert attach_context(None) is None

    def test_detach_context_none(self):
        detach_context(None)  # Kein Fehler

    def test_noop_tracer_delegates_after_init(self):
        """NoOpTracer delegiert an realen Tracer nach init_tracing()."""
        from kiara.shared import tracing

        tracer = _NoOpTracer()
        # Vor Init: NoOpSpan
        with tracer.start_as_current_span("before") as span:
            assert isinstance(span, _NoOpSpan)
        # Simuliere erfolgreiche Initialisierung
        mock_real_tracer = MagicMock()
        mock_span = MagicMock()
        mock_real_tracer.start_as_current_span.return_value = mock_span
        old_instance = tracing._tracer_instance
        tracing._tracer_instance = mock_real_tracer
        try:
            result = tracer.start_as_current_span("delegated")
            assert result is mock_span
            mock_real_tracer.start_as_current_span.assert_called_once_with("delegated")
        finally:
            tracing._tracer_instance = old_instance


class TestInitTracing:
    """init_tracing() Tests (mit Mocks)."""

    def test_init_disabled(self):
        from kiara.shared import tracing

        tracing._initialized = False
        tracing._otel_available = False
        tracing._tracer_instance = None
        tracing.init_tracing({"tracing": {"enabled": False}})
        assert tracing._tracer_instance is None
        tracer = get_tracer("test")
        assert isinstance(tracer, _NoOpTracer)
        # Reset
        tracing._initialized = False

    def test_init_missing_otel(self):
        from kiara.shared import tracing

        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            tracing.init_tracing({"tracing": {"enabled": True}})
        assert tracing._tracer_instance is None

    def test_init_unsupported_exporter_is_noop(self):
        from kiara.shared import tracing

        tracing._initialized = False
        tracing._otel_available = False
        tracing._tracer_instance = None
        try:
            tracing.init_tracing({"tracing": {"enabled": True, "exporter": "kafka"}})
            assert tracing._tracer_instance is None
            assert tracing._otel_available is False
        finally:
            tracing._initialized = False

    def test_init_idempotent(self):
        from kiara.shared import tracing

        tracing.init_tracing({"tracing": {"enabled": False}})
        first_state = tracing._initialized
        tracing.init_tracing({"tracing": {"enabled": True}})  # Sollte ignoriert werden
        assert tracing._initialized == first_state


class TestSerializeValue:
    """Tests fuer _serialize_value() im PgSpanExporter."""

    def test_none_passthrough(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value(None) is None

    def test_string_passthrough(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value("hello") == "hello"

    def test_int_passthrough(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value(42) == 42

    def test_float_passthrough(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value(3.14) == 3.14

    def test_bool_passthrough(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value(True) is True

    def test_other_types_to_str(self):
        from kiara.shared._pg_span_exporter import _serialize_value

        assert _serialize_value([1, 2, 3]) == "[1, 2, 3]"
        assert _serialize_value({"key": "val"}) == "{'key': 'val'}"


class TestPgSpanExporter:
    """Tests fuer PgSpanExporter.export() (mit Mocks).

    Benoetigt opentelemetry — Tests werden uebersprungen wenn nicht installiert.
    """

    @staticmethod
    def _mock_span():
        mock_span = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.trace_id = 12345
        mock_ctx.span_id = 67890
        mock_span.get_span_context.return_value = mock_ctx
        mock_span.parent = None
        mock_span.attributes = {}
        mock_span.status = None
        mock_span.name = "test_span"
        mock_span.start_time = 1000000000
        mock_span.end_time = 2000000000
        return mock_span

    def test_export_empty_spans(self):
        pytest.importorskip("opentelemetry.sdk.trace.export")
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter()
        result = exporter.export([])
        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.SUCCESS

    def test_export_db_failure(self):
        pytest.importorskip("opentelemetry.sdk.trace.export")
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter(retry_base_delay=0)
        db_mock = MagicMock(side_effect=Exception("DB down"))
        with patch("kiara.shared.db.db_transaction", db_mock):
            result = exporter.export([self._mock_span()])
        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.FAILURE
        assert db_mock.call_count == 3

    def test_export_retries_transient_db_failure(self):
        pytest.importorskip("opentelemetry.sdk.trace.export")
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter(retry_base_delay=0)
        mock_conn = MagicMock()
        mock_tx = MagicMock()
        mock_tx.__enter__.return_value = mock_conn
        mock_tx.__exit__.return_value = False
        db_mock = MagicMock(side_effect=[Exception("DB kurz weg"), mock_tx])

        with patch("kiara.shared.db.db_transaction", db_mock):
            result = exporter.export([self._mock_span()])

        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.SUCCESS
        assert db_mock.call_count == 2
        mock_conn.executemany.assert_called_once()

    def test_export_uses_exponential_backoff_between_retries(self):
        pytest.importorskip("opentelemetry.sdk.trace.export")
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter(max_attempts=4, retry_base_delay=0.5)
        db_mock = MagicMock(side_effect=Exception("DB down"))

        with (
            patch("kiara.shared.db.db_transaction", db_mock),
            patch("kiara.shared._pg_span_exporter.time.sleep") as sleep_mock,
        ):
            result = exporter.export([self._mock_span()])

        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.FAILURE
        assert db_mock.call_count == 4
        assert [call.args[0] for call in sleep_mock.call_args_list] == [0.5, 1.0, 2.0]

    def test_shutdown_noop(self):
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter()
        exporter.shutdown()  # Kein Fehler

    def test_force_flush_returns_true(self):
        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter()
        assert exporter.force_flush() is True


class TestCleanupTraces:
    """Tests fuer _cleanup_traces() in downsampling.py."""

    def test_cleanup_missing_table(self):
        """Fehlende traces-Tabelle → 0 geloescht, kein Fehler."""
        from kiara.monitoring.downsampling import _cleanup_traces

        with patch(
            "kiara.monitoring.downsampling.db_transaction",
            side_effect=Exception('relation "traces" does not exist'),
        ):
            result = _cleanup_traces()
        assert result == 0

    def test_cleanup_uses_config_retention(self):
        """Config-Wert fuer retention_days wird verwendet."""
        from kiara.monitoring.downsampling import _cleanup_traces

        mock_conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 3
        mock_conn.execute.return_value = mock_result
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)
        with patch("kiara.monitoring.downsampling.db_transaction", return_value=mock_conn):
            with patch(
                "kiara.config.load_config", return_value={"tracing": {"retention_days": 14}}
            ):
                result = _cleanup_traces()
        assert result == 3
        call_args = mock_conn.execute.call_args
        assert call_args[0][1] == (14,)

    def test_default_config_contains_tracing_section(self):
        from kiara.config import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["tracing"]["enabled"] is False
        assert DEFAULT_CONFIG["tracing"]["exporter"] == "postgresql"

    # test_migrate_add_trace_id_to_query_log_is_idempotent wurde entfernt:
    # importierte aus kiara.web.db_migrations (Modul existiert nicht mehr).
    # trace_id ist heute Teil der initialen Schema-Erstellung (schema.sql).


class TestPipelineConfig:
    """Per-Pipeline Tracing-Konfiguration (S8)."""

    def test_pipeline_default_enabled(self):
        """Ohne pipelines-Config: alle Pipelines aktiv."""
        from kiara.shared import tracing

        tracing._otel_available = True
        tracing._pipeline_flags.clear()
        assert tracing.is_pipeline_enabled("retrieval") is True

    def test_pipeline_disabled(self):
        """Pipeline explizit deaktiviert."""
        from kiara.shared import tracing

        tracing._otel_available = True
        tracing._pipeline_flags.clear()
        tracing._pipeline_flags["ingestion"] = False
        assert tracing.is_pipeline_enabled("ingestion") is False

    def test_init_tracing_reads_pipeline_flags_from_config(self):
        """init_tracing() uebernimmt tracing.pipelines.* aus der Config."""
        import pytest

        pytest.importorskip("opentelemetry.sdk.trace.export")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
            patch("opentelemetry.sdk.trace.TracerProvider.add_span_processor"),
        ):
            tracing.init_tracing(
                {
                    "tracing": {
                        "enabled": True,
                        "exporter": "postgresql",
                        "pipelines": {
                            "retrieval": False,
                            "agent": True,
                            "ingestion": False,
                        },
                    },
                }
            )

        assert tracing._pipeline_flags == {
            "retrieval": False,
            "agent": True,
            "ingestion": False,
        }

    def test_unknown_pipeline_default_true(self):
        """Unbekannte Pipeline: Default True."""
        from kiara.shared import tracing

        tracing._otel_available = True
        tracing._pipeline_flags.clear()
        assert tracing.is_pipeline_enabled("unknown_pipeline") is True

    def test_pipeline_without_otel_is_false(self):
        """Ohne OTel / enabled=false: Pipeline-Checks liefern False."""
        from kiara.shared import tracing

        tracing._otel_available = False
        assert tracing.is_pipeline_enabled("retrieval") is False

    def test_noop_tracer_with_disabled_pipeline(self):
        """NoOpTracer mit deaktivierter Pipeline gibt NoOpSpan auch wenn Tracer aktiv."""
        from unittest.mock import MagicMock

        from kiara.shared import tracing
        from kiara.shared.tracing import _NoOpSpan, _NoOpTracer

        tracing._pipeline_flags["ingestion"] = False
        old_instance = tracing._tracer_instance
        old_otel = tracing._otel_available
        tracing._tracer_instance = MagicMock()
        tracing._otel_available = True
        try:
            tracer = _NoOpTracer(pipeline="ingestion")
            span = tracer.start_as_current_span("test")
            assert isinstance(span, _NoOpSpan)
            tracing._tracer_instance.start_as_current_span.assert_not_called()
        finally:
            tracing._tracer_instance = old_instance
            tracing._otel_available = old_otel

    def test_get_tracer_with_pipeline(self):
        """get_tracer() mit pipeline-Parameter gibt NoOpTracer mit pipeline."""
        from kiara.shared.tracing import _NoOpTracer, get_tracer

        tracer = get_tracer("test", pipeline="retrieval")
        assert isinstance(tracer, _NoOpTracer)
        assert tracer._pipeline == "retrieval"

    def test_get_tracer_with_pipeline_after_init_still_returns_proxy(self):
        """Pipeline-Filter darf auch bei spaeten/lazy Callern nicht umgangen werden."""
        from unittest.mock import MagicMock

        from kiara.shared import tracing
        from kiara.shared.tracing import _NoOpTracer, get_tracer

        old_instance = tracing._tracer_instance
        tracing._tracer_instance = MagicMock()
        try:
            tracer = get_tracer("test", pipeline="retrieval")
            assert isinstance(tracer, _NoOpTracer)
            assert tracer._pipeline == "retrieval"
        finally:
            tracing._tracer_instance = old_instance


class TestOtlpExporter:
    """OTLP-Exporter-Konfiguration (S9)."""

    def test_exporter_postgresql_default_adds_one_processor(self):
        """Default-Exporter ist postgresql und initialisiert genau einen Processor."""
        import pytest

        pytest.importorskip("opentelemetry.sdk.trace.export")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor"),
            patch("opentelemetry.sdk.trace.TracerProvider.add_span_processor") as add_processor,
        ):
            tracing.init_tracing({"tracing": {"enabled": True}})
        assert add_processor.call_count == 1
        assert tracing._otel_available is True
        assert tracing._tracer_instance is not None

    def test_exporter_otlp_with_endpoint_adds_one_processor(self):
        """OTLP-only mit Endpoint initialisiert genau einen Processor."""
        import pytest

        pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_bsp,
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ) as MockOTLP,
        ):
            tracing.init_tracing(
                {
                    "tracing": {
                        "enabled": True,
                        "exporter": "otlp",
                        "otlp": {"endpoint": "http://localhost:4317", "insecure": False},
                    },
                }
            )
        MockOTLP.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=False,
        )
        assert mock_bsp.call_count == 1
        assert tracing._otel_available is True
        assert tracing._tracer_instance is not None

    def test_exporter_otlp_without_endpoint_adds_no_processor(self):
        """OTLP ohne Endpoint: Warning, kein Crash, NoOp-Tracer."""
        import pytest

        pytest.importorskip("opentelemetry.sdk.trace.export")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_bsp,
            patch("kiara.shared.tracing.logger") as mock_logger,
        ):
            tracing.init_tracing(
                {
                    "tracing": {"enabled": True, "exporter": "otlp", "otlp": {}},
                }
            )
        assert mock_bsp.call_count == 0
        assert mock_logger.warning.called
        assert tracing._tracer_instance is None
        assert tracing._otel_available is False

    def test_exporter_both_adds_two_processors(self):
        """Exporter 'both' registriert PG + OTLP auf demselben Provider."""
        import pytest

        pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        with (
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_bsp,
            patch(
                "opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"
            ) as MockOTLP,
        ):
            tracing.init_tracing(
                {
                    "tracing": {
                        "enabled": True,
                        "exporter": "both",
                        "otlp": {"endpoint": "http://localhost:4317", "insecure": False},
                    },
                }
            )
        MockOTLP.assert_called_once_with(
            endpoint="http://localhost:4317",
            insecure=False,
        )
        assert mock_bsp.call_count == 2
        assert tracing._otel_available is True
        assert tracing._tracer_instance is not None

    def test_exporter_unknown(self):
        """Unbekannter Exporter: Warning, NoOp."""
        from kiara.shared import tracing

        tracing.init_tracing(
            {
                "tracing": {"enabled": True, "exporter": "kafka"},
            }
        )
        assert tracing._tracer_instance is None
        assert tracing._otel_available is False

    def test_exporter_otlp_import_fails_falls_back_to_noop(self):
        """OTLP-only + ImportError → NoOp (kein stummer leerer Provider)."""
        import sys

        import pytest

        pytest.importorskip("opentelemetry.sdk.trace.export")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        otlp_mod = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        with (
            patch.dict(sys.modules, {otlp_mod: None}),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_bsp,
            patch("kiara.shared.tracing.logger") as mock_logger,
        ):
            tracing.init_tracing(
                {
                    "tracing": {
                        "enabled": True,
                        "exporter": "otlp",
                        "otlp": {"endpoint": "http://localhost:4317"},
                    },
                }
            )
        assert mock_bsp.call_count == 0
        assert mock_logger.warning.called
        assert tracing._tracer_instance is None
        assert tracing._otel_available is False

    def test_exporter_both_import_failure_keeps_pg_active(self):
        """Exporter 'both' + OTLP-ImportError laesst PostgreSQL allein aktiv."""
        import sys

        import pytest

        pytest.importorskip("opentelemetry.sdk.trace.export")
        from unittest.mock import MagicMock, patch

        from kiara.shared import tracing

        otlp_mod = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        with (
            patch.dict(sys.modules, {otlp_mod: None}),
            patch("opentelemetry.trace.set_tracer_provider"),
            patch("opentelemetry.trace.get_tracer", return_value=MagicMock()),
            patch("opentelemetry.sdk.trace.export.BatchSpanProcessor") as mock_bsp,
            patch("kiara.shared.tracing.logger") as mock_logger,
        ):
            tracing.init_tracing(
                {
                    "tracing": {
                        "enabled": True,
                        "exporter": "both",
                        "otlp": {"endpoint": "http://localhost:4317"},
                    },
                }
            )
        assert mock_bsp.call_count == 1
        assert mock_logger.warning.called
        assert tracing._otel_available is True
        assert tracing._tracer_instance is not None


class TestTracingConfigDefaults:
    def test_default_config_contains_new_tracing_defaults(self):
        from kiara.config import DEFAULT_CONFIG

        tracing_cfg = DEFAULT_CONFIG["tracing"]
        assert tracing_cfg["enabled"] is False
        assert tracing_cfg["exporter"] == "postgresql"
        assert tracing_cfg["pipelines"] == {
            "retrieval": True,
            "agent": True,
            "ingestion": True,
        }
        assert tracing_cfg["otlp"] == {
            "endpoint": "",
            "insecure": True,
        }
        assert tracing_cfg["chat_debug"] is False

    def test_config_template_contains_new_tracing_comment_block(self):
        from pathlib import Path

        content = Path("config/kiara.yaml").read_text(encoding="utf-8")
        assert "#   pipelines:" in content
        assert '#   exporter: postgresql        # "postgresql", "otlp" oder "both"' in content
        assert "#   otlp:" in content
        assert "#   chat_debug: false" in content


class TestNoOpSpanCoverage:
    """Deckt _NoOpSpan-Methoden und Context-Manager-Protokoll ab."""

    def test_noop_span_context_manager(self):
        """_NoOpSpan funktioniert als Context-Manager und gibt sich selbst zurueck."""
        span = _NoOpSpan()
        with span as s:
            assert s is span

    def test_noop_span_set_attribute_is_noop(self):
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # darf nicht raisen

    def test_noop_span_record_exception_is_noop(self):
        span = _NoOpSpan()
        span.record_exception(RuntimeError("test"))

    def test_noop_span_set_status_is_noop(self):
        span = _NoOpSpan()
        span.set_status("ERROR")

    def test_noop_span_add_event_is_noop(self):
        span = _NoOpSpan()
        span.add_event("test_event", attributes={"k": "v"})

    def test_noop_tracer_start_span_delegates_when_pipeline_enabled(self):
        """start_span (nicht start_as_current_span) delegiert bei aktiver Pipeline."""
        from kiara.shared import tracing

        mock_real_tracer = MagicMock()
        mock_span = MagicMock()
        mock_real_tracer.start_span.return_value = mock_span
        tracing._tracer_instance = mock_real_tracer
        tracing._otel_available = True
        tracing._pipeline_flags["retrieval"] = True
        try:
            proxy = _NoOpTracer(pipeline="retrieval")
            result = proxy.start_span("test_op", kind=1)
            assert result is mock_span
            mock_real_tracer.start_span.assert_called_once_with("test_op", kind=1)
        finally:
            tracing._tracer_instance = None
            tracing._otel_available = False

    def test_noop_tracer_start_span_noop_when_pipeline_disabled(self):
        """start_span gibt NoOpSpan bei deaktivierter Pipeline."""
        from kiara.shared import tracing

        tracing._tracer_instance = MagicMock()
        tracing._otel_available = True
        tracing._pipeline_flags["ingestion"] = False
        try:
            proxy = _NoOpTracer(pipeline="ingestion")
            result = proxy.start_span("ingest_file")
            assert isinstance(result, _NoOpSpan)
            tracing._tracer_instance.start_span.assert_not_called()
        finally:
            tracing._tracer_instance = None
            tracing._otel_available = False


class TestGetTraceIdForSpan:
    """Deckt get_trace_id_for_span() mit NoOp- und Mock-Spans ab."""

    def test_noop_span_returns_none(self):
        """_NoOpSpan hat kein get_span_context — Ergebnis muss None sein."""
        assert get_trace_id_for_span(_NoOpSpan()) is None

    def test_mock_span_with_trace_id(self):
        """Span mit gueltigem trace_id gibt hex-formatierten String zurueck."""
        ctx = MagicMock()
        ctx.trace_id = 0xDEADBEEF
        span = MagicMock()
        span.get_span_context.return_value = ctx
        result = get_trace_id_for_span(span)
        assert result == "000000000000000000000000deadbeef"

    def test_mock_span_with_zero_trace_id(self):
        """trace_id == 0 gilt als ungueltig (OTel INVALID_TRACE_ID)."""
        ctx = MagicMock()
        ctx.trace_id = 0
        span = MagicMock()
        span.get_span_context.return_value = ctx
        assert get_trace_id_for_span(span) is None

    def test_span_raising_exception_returns_none(self):
        """Fehler bei get_span_context → None statt Exception."""
        span = MagicMock()
        span.get_span_context.side_effect = RuntimeError("kaputt")
        assert get_trace_id_for_span(span) is None


class TestContextPropagationNoOtel:
    """Deckt get_current_context/attach/detach bei _otel_available=False ab."""

    def test_get_current_context_without_otel(self):
        from kiara.shared import tracing

        tracing._otel_available = False
        assert get_current_context() is None

    def test_attach_context_with_none(self):
        assert attach_context(None) is None

    def test_attach_context_without_otel(self):
        from kiara.shared import tracing

        tracing._otel_available = False
        assert attach_context("fake_ctx") is None

    def test_detach_context_with_none(self):
        # Darf nicht raisen
        detach_context(None)

    def test_detach_context_without_otel(self):
        from kiara.shared import tracing

        tracing._otel_available = False
        # Darf nicht raisen
        detach_context("fake_token")

    def test_get_current_trace_id_without_otel(self):
        from kiara.shared import tracing

        tracing._otel_available = False
        assert get_current_trace_id() is None
