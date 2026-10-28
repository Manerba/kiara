"""OpenTelemetry-Tracing fuer KIara.

Stellt einen Tracer bereit, der Spans je nach `tracing.exporter`
ueber einen PgSpanExporter, einen OTLP-Exporter oder beide Exporter
parallel ausliefert. Wenn OpenTelemetry nicht installiert oder
Tracing deaktiviert ist, wird ein NoOp-Tracer verwendet
(null Overhead).

Usage:
    from kiara.shared.tracing import get_tracer

    _tracer = get_tracer(__name__)

    with _tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("query", question[:200])
        result = do_work()
        span.set_attribute("result_count", len(result))
"""

import logging

logger = logging.getLogger(__name__)

# Modul-State (einmal gesetzt, dann stabil)
_initialized = False
_otel_available = False
_tracer_instance = None
_pipeline_flags: dict[str, bool] = {}


def init_tracing(config: dict) -> None:
    """Initialisiert OTel-Tracing mit PostgreSQL-, OTLP- oder Kombi-Exporter.

    Idempotent — mehrfacher Aufruf hat keinen Effekt.
    Bei deaktiviertem Tracing, unsupported Exporter, fehlender
    OTel-Dependency oder fehlgeschlagener Exporter-Initialisierung:
    kein Effekt.
    """
    global _initialized, _otel_available, _tracer_instance

    if _initialized:
        return
    _initialized = True

    tracing_cfg = config.get("tracing", {})
    if not tracing_cfg.get("enabled", False):
        logger.debug("Tracing deaktiviert (config)")
        return

    exporter_name = tracing_cfg.get("exporter", "postgresql")

    # Unbekannter Exporter? → Frueh abbrechen, bevor OTel importiert wird.
    # So bleibt das bestehende Verhalten (Warning + NoOp) unveraendert und
    # unnoetige Imports werden vermieden.
    if exporter_name not in ("postgresql", "otlp", "both"):
        logger.warning(
            "Tracing-Exporter %r unbekannt — NoOp-Tracer aktiv. "
            "Erlaubte Werte: postgresql, otlp, both",
            exporter_name,
        )
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        from kiara.shared.constants import TRACING_BATCH_SCHEDULE_MS, TRACING_SERVICE_NAME

        provider = TracerProvider()
        processors = []
        active_exporters = []

        # PostgreSQL-Exporter (Default)
        if exporter_name in ("postgresql", "both"):
            from kiara.shared._pg_span_exporter import PgSpanExporter

            pg_processor = BatchSpanProcessor(
                PgSpanExporter(),
                schedule_delay_millis=TRACING_BATCH_SCHEDULE_MS,
            )
            processors.append(pg_processor)
            active_exporters.append("postgresql")
            logger.info("Tracing-Exporter: PostgreSQL aktiv")

        # OTLP-Exporter (optional)
        if exporter_name in ("otlp", "both"):
            otlp_cfg = tracing_cfg.get("otlp", {})
            endpoint = otlp_cfg.get("endpoint", "")
            if not endpoint:
                logger.warning(
                    "Tracing: OTLP aktiviert aber kein Endpoint konfiguriert "
                    "(tracing.otlp.endpoint) — OTLP-Export deaktiviert"
                )
            else:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    otlp_exporter = OTLPSpanExporter(
                        endpoint=endpoint,
                        insecure=otlp_cfg.get("insecure", True),
                    )
                    otlp_processor = BatchSpanProcessor(
                        otlp_exporter,
                        schedule_delay_millis=TRACING_BATCH_SCHEDULE_MS,
                    )
                    processors.append(otlp_processor)
                    active_exporters.append("otlp")
                    logger.info("Tracing-Exporter: OTLP aktiv (Endpoint: %s)", endpoint)
                except ImportError:
                    logger.warning(
                        "opentelemetry-exporter-otlp nicht installiert — "
                        "OTLP-Export deaktiviert. "
                        "Installation: Extra-Gruppe 'tracing-otlp' "
                        "oder Paket 'opentelemetry-exporter-otlp-proto-grpc>=1.20.0' "
                        "nachinstallieren"
                    )
                except Exception as exc:
                    logger.warning("OTLP-Exporter-Init fehlgeschlagen: %s", exc)

        # Kein Processor erfolgreich? → NoOp statt leerer Provider
        if not processors:
            logger.warning("Tracing: Kein Exporter erfolgreich initialisiert — NoOp-Tracer aktiv")
            return

        # Vorhandenen SDK-TracerProvider erkennen — set_tracer_provider() ist
        # no-op, wenn bereits einer aktiv ist (Auto-Instrumentation, Drittpartei).
        # Ohne Detection laufen Spans am Fremd-Provider, KIara-Exporter bleiben leer.
        existing = trace.get_tracer_provider()
        if isinstance(existing, TracerProvider):
            target = existing
            logger.warning(
                "Tracing: Bereits aktiver SDK-TracerProvider erkannt — "
                "KIara-Span-Processors werden dort registriert"
            )
        else:
            trace.set_tracer_provider(provider)
            target = provider

        for processor in processors:
            target.add_span_processor(processor)

        # atexit-Handler unabhaengig vom Provider-Pfad registrieren: bei einem
        # Fremd-Provider ist dessen Shutdown-Hook nicht garantiert, sonst gehen
        # KIara-Spans beim Prozess-Ende verloren. Direkt auf den Processors
        # operieren — der lokale `provider` hat im Fremdpfad keine Processors.
        import atexit

        def _flush_and_shutdown():
            for proc in processors:
                try:
                    proc.force_flush(timeout_millis=5000)
                except Exception:
                    pass
                try:
                    proc.shutdown()
                except Exception:
                    pass

        atexit.register(_flush_and_shutdown)

        _otel_available = True
        _tracer_instance = trace.get_tracer(TRACING_SERVICE_NAME)

        # Per-Pipeline-Flags (S8)
        pipelines = tracing_cfg.get("pipelines", {})
        _pipeline_flags.clear()
        _pipeline_flags["retrieval"] = pipelines.get("retrieval", True)
        _pipeline_flags["agent"] = pipelines.get("agent", True)
        _pipeline_flags["ingestion"] = pipelines.get("ingestion", True)

        logger.info(
            "OpenTelemetry-Tracing initialisiert (Exporter: %s)", " + ".join(active_exporters)
        )

    except ImportError as exc:
        logger.warning(
            "Tracing aktiviert (config), aber opentelemetry-api/sdk nicht "
            "installiert — NoOp-Tracer aktiv, query_log.trace_id bleibt NULL. "
            "Fix: 'pip install opentelemetry-api opentelemetry-sdk' im venv. "
            "Detail: %s",
            exc,
        )
    except Exception as exc:
        logger.warning("OpenTelemetry-Init fehlgeschlagen: %s", exc)


def get_tracer(name: str = "kiara", pipeline: str | None = None):
    """Gibt einen OTel-Tracer oder NoOp-Fallback zurueck.

    Args:
        name: Tracer-Name (Konvention: __name__). Bleibt fuer API-
              Kompatibilitaet erhalten; die reale OTel-Instanz bleibt wie
              bisher serviceweit.
        pipeline: Optionaler Pipeline-Name ("retrieval", "agent", "ingestion").
                  Wenn gesetzt, wird Tracing fuer diese Pipeline per Config
                  steuerbar (tracing.pipelines.<name>).
    """
    if pipeline is not None:
        # Pipeline-aware Proxy auch NACH init_tracing(), damit spaete/lazy
        # Caller den Config-Filter nicht umgehen.
        return _NoOpTracer(pipeline=pipeline)
    if _tracer_instance is not None:
        return _tracer_instance
    return _NoOpTracer()


def get_current_trace_id() -> str | None:
    """Gibt die aktuelle trace_id als Hex-String zurueck, oder None.

    Fuer die Korrelation mit query_log (AD-7).
    """
    if not _otel_available:
        return None
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx and ctx.trace_id:
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None


def get_trace_id_for_span(span) -> str | None:
    """Extrahiert die trace_id sicher aus einem konkreten Span.

    Funktioniert auch mit `_NoOpSpan` und liefert dann None statt Exception.
    Wird genutzt, wenn die trace_id waehrend eines offenen Root-Spans
    persistiert werden soll (RAGChain / AgentChain).
    """
    try:
        ctx = span.get_span_context()
        if ctx and getattr(ctx, "trace_id", 0):
            return format(ctx.trace_id, "032x")
    except Exception:
        pass
    return None


def get_current_context():
    """Gibt den aktuellen OTel-Context zurueck (fuer Thread-Propagation).

    Returns None wenn OTel nicht verfuegbar.
    """
    if not _otel_available:
        return None
    try:
        from opentelemetry import context as otel_context

        return otel_context.get_current()
    except Exception:
        return None


def attach_context(ctx):
    """Setzt einen OTel-Context im aktuellen Thread. Gibt Token zurueck.

    Returns None wenn ctx None ist oder OTel nicht verfuegbar.
    """
    if ctx is None or not _otel_available:
        return None
    try:
        from opentelemetry import context as otel_context

        return otel_context.attach(ctx)
    except Exception:
        return None


def detach_context(token):
    """Entfernt einen per attach_context() gesetzten Context."""
    if token is None or not _otel_available:
        return
    try:
        from opentelemetry import context as otel_context

        otel_context.detach(token)
    except Exception:
        pass


def is_pipeline_enabled(pipeline: str) -> bool:
    """Prueft ob Tracing fuer eine bestimmte Pipeline aktiviert ist.

    Bei unbekannter Pipeline: True (Default).
    Ohne aktives OTel / bei globalem `enabled: false`: False.
    """
    if not _otel_available:
        return False
    return _pipeline_flags.get(pipeline, True)


def mark_current_span_error(exc: BaseException) -> None:
    """Markiert den aktuell aktiven Span als ERROR und zeichnet die Exception auf.

    Genutzt in Defensive-Pfaden, die Exceptions abfangen und einen leeren
    Default zurueckgeben — ohne diese Markierung wuerde der Span den
    Fehlerpfad nicht erkennen lassen.

    No-op wenn OTel nicht verfuegbar oder kein aktiver Span existiert.
    """
    if not _otel_available:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.trace.status import Status, StatusCode

        span = trace.get_current_span()
        if span is None:
            return
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# NoOp-Fallback (kein Overhead im Hot Path)
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Minimaler Span-Ersatz wenn OTel nicht verfuegbar."""

    def set_attribute(self, key, value):
        pass

    def record_exception(self, exc):
        pass

    def set_status(self, status):
        pass

    def add_event(self, name, attributes=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class _NoOpTracer:
    """Tracer-Ersatz / Proxy.

    Delegiert an den realen Tracer sobald init_tracing() erfolgreich war.
    Bei gesetztem pipeline-Parameter wird zusaetzlich geprueft ob die
    Pipeline in der Config aktiviert ist.
    """

    def __init__(self, pipeline: str | None = None):
        self._pipeline = pipeline

    def start_as_current_span(self, name, **kwargs):
        if _tracer_instance is not None:
            if self._pipeline and not is_pipeline_enabled(self._pipeline):
                return _NoOpSpan()
            return _tracer_instance.start_as_current_span(name, **kwargs)
        return _NoOpSpan()

    def start_span(self, name, **kwargs):
        if _tracer_instance is not None:
            if self._pipeline and not is_pipeline_enabled(self._pipeline):
                return _NoOpSpan()
            return _tracer_instance.start_span(name, **kwargs)
        return _NoOpSpan()
