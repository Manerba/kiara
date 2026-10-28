"""PostgreSQL-Exporter fuer OpenTelemetry-Spans.

Schreibt Spans als Batch in die `traces`-Tabelle. Kein eigenes Cleanup —
Retention wird von monitoring/downsampling.py ueber das taegliche
Admin-App-Downsampling-Fenster (`METRICS_DOWNSAMPLE_HOUR`, default 03xx) erledigt.
"""

import json
import logging
import time

logger = logging.getLogger(__name__)


_MAX_ATTR_LEN = 4096


def _serialize_value(v):
    """Serialisiert OTel-Attributwerte fuer JSONB."""
    if v is None:
        return None
    if isinstance(v, (int, float, bool)):
        return v
    s = v if isinstance(v, str) else str(v)
    if len(s) > _MAX_ATTR_LEN:
        return s[:_MAX_ATTR_LEN] + "...(truncated)"
    return s


class PgSpanExporter:
    """SpanExporter der in PostgreSQL schreibt.

    Implementiert das opentelemetry.sdk.trace.export.SpanExporter Interface.
    """

    def __init__(self, max_attempts: int = 3, retry_base_delay: float = 0.5):
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_delay = max(0.0, float(retry_base_delay))

    def _write_rows(self, rows):
        from kiara.shared.db import db_transaction

        with db_transaction() as conn:
            conn.executemany(
                "INSERT INTO traces "
                "(trace_id, span_id, parent_span_id, name, "
                "start_time_ns, end_time_ns, attributes, status) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
                "ON CONFLICT (span_id) DO NOTHING",
                rows,
            )

    def export(self, spans):
        """Schreibt Spans in die `traces`-Tabelle."""
        from opentelemetry.sdk.trace.export import SpanExportResult

        rows = []
        for span in spans:
            ctx = span.get_span_context()
            parent = span.parent
            parent_id = None
            if parent and hasattr(parent, "span_id"):
                parent_id = format(parent.span_id, "016x")

            attrs = {}
            if span.attributes:
                attrs = {k: _serialize_value(v) for k, v in span.attributes.items()}

            status = "OK"
            if span.status and span.status.status_code:
                status = span.status.status_code.name

            rows.append(
                (
                    format(ctx.trace_id, "032x"),
                    format(ctx.span_id, "016x"),
                    parent_id,
                    span.name,
                    span.start_time,
                    span.end_time or span.start_time,
                    json.dumps(attrs, ensure_ascii=False),
                    status,
                )
            )

        if not rows:
            return SpanExportResult.SUCCESS

        for attempt in range(1, self.max_attempts + 1):
            try:
                self._write_rows(rows)
                return SpanExportResult.SUCCESS
            except Exception as exc:
                if attempt >= self.max_attempts:
                    logger.warning(
                        "Span-Export nach PostgreSQL nach %d Versuch(en) fehlgeschlagen: %s",
                        attempt,
                        exc,
                    )
                    return SpanExportResult.FAILURE
                delay = self.retry_base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Span-Export nach PostgreSQL fehlgeschlagen, Retry %d/%d in %.1fs: %s",
                    attempt + 1,
                    self.max_attempts,
                    delay,
                    exc,
                )
                if delay > 0:
                    time.sleep(delay)

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=None):
        return True
