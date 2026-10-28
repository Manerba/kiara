"""Tests fuer den erweiterten Trace Explorer (Sprint 17.6).

Deckt API-Endpunkte, Schema-Refactoring, Router-Integration,
Serialisierung und Edge Cases ab.
"""

import sys
import types

# ldap3 ist eine optionale Abhaengigkeit und steht in der Test-Umgebung nicht
# zur Verfuegung. Stub-Modul vorab registrieren, damit der transitive Import-Pfad
# admin_app -> api_health -> health.checker -> checks_services -> shared.ldap
# nicht abbricht (benoetigt von TestRouterIntegration).
if "ldap3" not in sys.modules:
    _ldap3_stub = types.ModuleType("ldap3")
    _ldap3_stub.Server = object
    _ldap3_stub.Connection = object
    _ldap3_stub.NONE = None
    _ldap3_stub.SUBTREE = None
    _ldap3_stub.Tls = object
    sys.modules["ldap3"] = _ldap3_stub

import json
import uuid

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kiara.shared.db import get_connection
from kiara.web.api_tracing import _normalize_to_date, _trace_status_from_spans, router
from kiara.web.auth import get_admin_user
from kiara.web.schema import create_tables


def _has_otel():
    try:
        import opentelemetry.sdk.trace.export  # noqa: F401

        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _init_trace_explorer_schema(conn, with_migrations=False):
    """Richtet das vollstaendige Schema inkl. traces-Tabelle ein.

    ``with_migrations`` wird ignoriert — schema.sql enthaelt bereits alle Indizes
    (frueher von _migrate_trace_explorer_indices zugefuegt).
    """
    create_tables(conn)
    conn.commit()


def _insert_span(
    conn,
    *,
    trace_id,
    span_id=None,
    parent_span_id=None,
    name="test_span",
    start_time_ns=1000000000000,
    end_time_ns=2000000000000,
    attributes=None,
    status="OK",
):
    """Fuegt einen Span in die traces-Tabelle ein."""
    span_id = span_id or str(uuid.uuid4())[:16]
    attrs = json.dumps(attributes or {})
    conn.execute(
        "INSERT INTO traces (trace_id, span_id, parent_span_id, name, "
        "start_time_ns, end_time_ns, attributes, status) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
        "ON CONFLICT (span_id) DO NOTHING",
        (trace_id, span_id, parent_span_id, name, start_time_ns, end_time_ns, attrs, status),
    )
    return span_id


@pytest.fixture
def trace_db():
    """Erstellt Schema, liefert Connection, rollt am Ende zurueck."""
    conn = get_connection()
    _init_trace_explorer_schema(conn, with_migrations=True)
    # Andere Trace-Tests committen absichtlich, weil die API ueber separate
    # Connections liest. Jeder Fixture-Aufruf braucht deshalb eine explizit
    # leere Testmenge statt eines wirkungslosen Rollbacks nach ``commit``.
    conn.execute("DELETE FROM traces")
    conn.commit()
    try:
        yield conn
    finally:
        conn.rollback()
        conn.execute("DELETE FROM traces")
        conn.commit()
        conn.close()


@pytest.fixture
def seeded_db(trace_db):
    """DB mit Beispiel-Traces fuer Filter-/Aggregations-Tests."""
    conn = trace_db

    # Trace 1: OK, root=rag_query, 2 Spans
    t1 = "trace_ok_" + uuid.uuid4().hex[:8]
    root1 = _insert_span(
        conn,
        trace_id=t1,
        name="rag_query",
        start_time_ns=1_700_000_000_000_000_000,
        end_time_ns=1_700_000_001_000_000_000,
        status="OK",
        attributes={"model": "llama3.1"},
    )
    _insert_span(
        conn,
        trace_id=t1,
        parent_span_id=root1,
        name="rerank",
        start_time_ns=1_700_000_000_200_000_000,
        end_time_ns=1_700_000_000_500_000_000,
        status="OK",
    )

    # Trace 2: ERROR, root=rag_query, 2 Spans
    t2 = "trace_err_" + uuid.uuid4().hex[:8]
    root2 = _insert_span(
        conn,
        trace_id=t2,
        name="rag_query",
        start_time_ns=1_700_000_002_000_000_000,
        end_time_ns=1_700_000_005_000_000_000,
        status="OK",
    )
    _insert_span(
        conn,
        trace_id=t2,
        parent_span_id=root2,
        name="graph_retrieve",
        start_time_ns=1_700_000_002_100_000_000,
        end_time_ns=1_700_000_003_000_000_000,
        status="ERROR",
    )

    # Trace 3: UNSET, root=hybrid_retrieve, 1 Span
    t3 = "trace_unset_" + uuid.uuid4().hex[:8]
    _insert_span(
        conn,
        trace_id=t3,
        name="hybrid_retrieve",
        start_time_ns=1_700_000_010_000_000_000,
        end_time_ns=1_700_000_010_500_000_000,
        status="UNSET",
        attributes={"final_status": "success"},
    )

    # Trace 4: Agent-Trace mit doppelten Span-Namen (fuer Compare #23)
    t4 = "trace_agent_" + uuid.uuid4().hex[:8]
    root4 = _insert_span(
        conn,
        trace_id=t4,
        name="agent_loop",
        start_time_ns=1_700_000_020_000_000_000,
        end_time_ns=1_700_000_024_000_000_000,
        status="OK",
    )
    _insert_span(
        conn,
        trace_id=t4,
        parent_span_id=root4,
        name="tool_call",
        start_time_ns=1_700_000_020_100_000_000,
        end_time_ns=1_700_000_021_000_000_000,
        status="OK",
    )
    _insert_span(
        conn,
        trace_id=t4,
        parent_span_id=root4,
        name="tool_call",
        start_time_ns=1_700_000_022_000_000_000,
        end_time_ns=1_700_000_023_500_000_000,
        status="OK",
    )

    conn.commit()
    return conn, {"t1": t1, "t2": t2, "t3": t3, "t4": t4}


@pytest.fixture
def client(seeded_db):
    """FastAPI TestClient mit auth-Mock."""
    app = FastAPI()
    app.include_router(router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with TestClient(app) as c:
        yield c, seeded_db[1]  # client + trace_ids


# ---------------------------------------------------------------------------
# Schema-Tests (#45)
# ---------------------------------------------------------------------------


class TestSchema:
    def test_create_tables_creates_traces(self, trace_db):
        """#45: create_tables() legt traces inkl. Basis-Indizes an."""
        conn = trace_db
        row = conn.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_name = 'traces'"
        ).fetchone()
        assert row is not None

        indices = {
            r["indexname"]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'traces'"
            ).fetchall()
        }
        assert "idx_traces_trace_id" in indices
        assert "idx_traces_created" in indices
        assert "idx_traces_name" in indices


# ---------------------------------------------------------------------------
# Index-Existenz (#31-#32): frueher ueber _migrate_trace_explorer_indices,
# jetzt in schema.sql enthalten. Idempotenz-Tests (#33-#34) entfernt —
# die Migration-Funktion existiert nicht mehr.
# ---------------------------------------------------------------------------


class TestIndices:
    def test_status_index_exists(self, trace_db):
        """#31: idx_traces_status existiert im Schema."""
        indices = {
            r["indexname"]
            for r in trace_db.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'traces'"
            ).fetchall()
        }
        assert "idx_traces_status" in indices

    def test_attributes_gin_index_exists(self, trace_db):
        """#32: idx_traces_attributes (GIN) existiert im Schema."""
        indices = {
            r["indexname"]
            for r in trace_db.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'traces'"
            ).fetchall()
        }
        assert "idx_traces_attributes" in indices


# ---------------------------------------------------------------------------
# Router-Integration (#47)
# ---------------------------------------------------------------------------


class TestRouterIntegration:
    def test_tracing_router_in_admin_app(self):
        """#47: admin_app.py include't tracing_router."""
        from kiara.web.admin_app import app as admin_app

        routes = [r.path for r in admin_app.routes if hasattr(r, "path")]
        assert "/api/traces" in routes or any("/api/traces" in str(r) for r in routes)


# ---------------------------------------------------------------------------
# Admin-HTML-Integration (#46)
# ---------------------------------------------------------------------------


class TestAdminHTML:
    def test_admin_html_contains_traces_tab(self):
        """#46: admin.html enthaelt tab_system_traces.js und system-traces."""
        from pathlib import Path

        html = Path("kiara/web/templates/admin.html").read_text()
        assert "tab_system_traces.js" in html
        assert "system-traces" in html
        assert "initSystemTracesTab" in html


# ---------------------------------------------------------------------------
# Filter-Tests (#1-#10)
# ---------------------------------------------------------------------------


class TestListTraces:
    def test_list_all(self, client):
        """Alle Traces auflisten."""
        c, ids = client
        resp = c.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data
        assert "pagination" in data
        assert data["pagination"]["total_items"] >= 3

    def test_filter_status_error(self, client):
        """#1: Nur ERROR-Traces."""
        c, ids = client
        resp = c.get("/api/traces?status=ERROR")
        assert resp.status_code == 200
        data = resp.json()
        for row in data["data"]:
            assert row["status"] == "ERROR"

    def test_filter_status_unset(self, client):
        """#2: Nur UNSET-Traces."""
        c, ids = client
        resp = c.get("/api/traces?status=UNSET")
        assert resp.status_code == 200
        for row in resp.json()["data"]:
            assert row["status"] == "UNSET"

    def test_filter_future_date_empty(self, client):
        """#4: Zukuenftiges from_date → leere Liste."""
        c, _ = client
        resp = c.get("/api/traces?from_date=2099-01-01")
        data = resp.json()
        assert data["pagination"]["total_items"] == 0
        assert data["pagination"]["total_pages"] == 0

    def test_filter_span_name(self, client):
        """#5: Traces mit bestimmtem Span-Namen."""
        c, _ = client
        resp = c.get("/api/traces?span_name=rerank")
        assert resp.status_code == 200
        assert resp.json()["pagination"]["total_items"] >= 1

    def test_filter_span_name_nonexistent(self, client):
        """#6: Nicht-existenter Span-Name → leer."""
        c, _ = client
        resp = c.get("/api/traces?span_name=nonexistent_span_xyz")
        assert resp.json()["pagination"]["total_items"] == 0

    def test_filter_attr_search(self, client):
        """#7: JSONB-Attributsuche."""
        c, _ = client
        resp = c.get("/api/traces?attr_search=model%3Dllama3.1")
        assert resp.status_code == 200
        assert resp.json()["pagination"]["total_items"] >= 1

    def test_filter_attr_search_final_status(self, client):
        """#8: Ingestion-Attributsuche."""
        c, _ = client
        resp = c.get("/api/traces?attr_search=final_status%3Dsuccess")
        assert resp.status_code == 200
        assert resp.json()["pagination"]["total_items"] >= 1

    def test_filter_attr_search_invalid(self, client):
        """#9: attr_search ohne '=' → keine Filterung, kein Fehler."""
        c, _ = client
        resp = c.get("/api/traces?attr_search=invalid")
        assert resp.status_code == 200

    def test_filter_combined(self, client):
        """#10: Kombination aller Filter."""
        c, _ = client
        resp = c.get("/api/traces?name=rag_query&status=ERROR")
        assert resp.status_code == 200
        for row in resp.json()["data"]:
            assert row["status"] == "ERROR"
            assert row["root_span"] == "rag_query"


# ---------------------------------------------------------------------------
# Aggregation-Tests (#11-#18)
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_histogram_empty(self, client):
        """#11: Histogramm ohne passende Traces."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram?from_date=2099-01-01")
        data = resp.json()
        assert data["total"] == 0
        assert data["buckets"] == []

    def test_histogram_with_traces(self, client):
        """#12: Histogramm mit Traces."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 3
        assert len(data["buckets"]) <= 20

    def test_histogram_custom_buckets(self, client):
        """#13: Histogramm mit 5 Buckets."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram?buckets=5")
        data = resp.json()
        assert len(data["buckets"]) <= 5

    def test_histogram_with_filter(self, client):
        """#14: Histogramm mit Name-Filter."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram?name=rag_query")
        assert resp.status_code == 200

    def test_percentiles(self, client):
        """#15: Percentile sind vorhanden."""
        c, _ = client
        data = c.get("/api/traces/stats/duration-histogram").json()
        for key in ["p50_ms", "p95_ms", "p99_ms", "avg_ms"]:
            assert key in data

    def test_span_averages(self, client):
        """#17: Span-Durchschnitte."""
        c, _ = client
        resp = c.get("/api/traces/stats/span-averages")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["spans"]) >= 1
        for s in data["spans"]:
            for key in ["name", "count", "avg_ms", "min_ms", "max_ms", "p95_ms"]:
                assert key in s

    def test_span_averages_empty(self, client):
        """#18: Span-Durchschnitte ohne Traces."""
        c, _ = client
        resp = c.get("/api/traces/stats/span-averages?from_date=2099-01-01")
        assert resp.json()["spans"] == []


# ---------------------------------------------------------------------------
# Vergleich-Tests (#19-#23)
# ---------------------------------------------------------------------------


class TestCompare:
    def test_compare_two_traces(self, client):
        """#19: Vergleich zweier Traces."""
        c, ids = client
        resp = c.get(f"/api/traces/compare?trace_a={ids['t1']}&trace_b={ids['t2']}")
        assert resp.status_code == 200
        data = resp.json()
        assert "trace_a" in data
        assert "trace_b" in data
        assert "diff" in data

    def test_compare_nonexistent(self, client):
        """#20: 404 bei nicht-existentem Trace."""
        c, ids = client
        resp = c.get(f"/api/traces/compare?trace_a={ids['t1']}&trace_b=INVALID_TRACE")
        assert resp.status_code == 404

    def test_compare_different_spans(self, client):
        """#21: Diff bei unterschiedlichen Span-Typen."""
        c, ids = client
        resp = c.get(f"/api/traces/compare?trace_a={ids['t1']}&trace_b={ids['t3']}")
        data = resp.json()
        for d in data["diff"]:
            # Fehlende Spans haben None
            if d["span_name"] == "rerank":
                assert d["b_ms"] is None
            if d["span_name"] == "hybrid_retrieve":
                assert d["a_ms"] is None

    def test_compare_identical(self, client):
        """#22: Vergleich desselben Trace → delta == 0."""
        c, ids = client
        resp = c.get(f"/api/traces/compare?trace_a={ids['t1']}&trace_b={ids['t1']}")
        data = resp.json()
        for d in data["diff"]:
            assert d["delta_ms"] == 0


# ---------------------------------------------------------------------------
# Export-Tests (#24-#30, #63-#65)
# ---------------------------------------------------------------------------


class TestExport:
    def test_csv_export(self, client):
        """#24: CSV-Export."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=csv")
        assert resp.status_code == 200
        assert "text/csv" in resp.headers["content-type"]
        assert "trace_id" in resp.text

    def test_json_export(self, client):
        """#25: JSON-Export."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=json")
        assert resp.status_code == 200
        data = resp.json()
        assert "data" in data

    def test_invalid_format(self, client):
        """#26: Ungueltiges Format → 422."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=xml")
        assert resp.status_code == 422

    def test_single_trace_export(self, client):
        """#27: Einzel-Trace-Export."""
        c, ids = client
        resp = c.get(f"/api/traces/{ids['t1']}/export")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == ids["t1"]
        assert "spans" in data
        disposition = resp.headers["content-disposition"]
        assert disposition == "attachment; filename=trace_export.json"
        assert ids["t1"] not in disposition

    def test_single_trace_export_404(self, client):
        """#28: 404 bei nicht-existentem Trace."""
        c, _ = client
        resp = c.get("/api/traces/NONEXISTENT_TRACE_XYZ/export")
        assert resp.status_code == 404

    def test_csv_export_with_filter(self, client):
        """#29: CSV-Export mit Filtern."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=csv&name=rag_query")
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) >= 2  # Header + mindestens 1 Zeile

    def test_export_with_span_filter(self, client):
        """#30: Export mit span_name-Filter."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=json&span_name=rerank")
        assert resp.status_code == 200

    def test_export_limit_zero(self, client):
        """#63: limit=0 → 422."""
        c, _ = client
        resp = c.get("/api/traces/export?limit=0")
        assert resp.status_code == 422

    def test_export_limit_too_high(self, client):
        """#63: limit=5001 → 422."""
        c, _ = client
        resp = c.get("/api/traces/export?limit=5001")
        assert resp.status_code == 422

    def test_json_export_no_id_field(self, client):
        """#65: JSON-Export enthaelt kein UI-Hilfsfeld 'id'."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=json")
        data = resp.json()
        for row in data["data"]:
            assert "id" not in row


# ---------------------------------------------------------------------------
# Route-Reihenfolge (#35-#37)
# ---------------------------------------------------------------------------


class TestRouteOrder:
    def test_histogram_not_trace_id(self, client):
        """#35: /stats/duration-histogram wird nicht als trace_id interpretiert."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram")
        assert resp.status_code == 200

    def test_compare_not_trace_id(self, client):
        """#36: /compare wird nicht als trace_id interpretiert."""
        c, _ = client
        # Ohne Parameter → 422 (fehlende Pflichtparameter), nicht 404
        resp = c.get("/api/traces/compare")
        assert resp.status_code == 422

    def test_export_not_trace_id(self, client):
        """#37: /export wird nicht als trace_id interpretiert."""
        c, _ = client
        resp = c.get("/api/traces/export")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Serialisierung (#38-#39, #43-#44)
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_list_start_time_iso(self, client):
        """#38: start_time ist ISO-8601-String."""
        c, _ = client
        resp = c.get("/api/traces")
        for row in resp.json()["data"]:
            st = row["start_time"]
            assert isinstance(st, str)
            assert "T" in st  # ISO-Format

    def test_list_duration_is_float(self, client):
        """#43: duration_ms ist float."""
        c, _ = client
        resp = c.get("/api/traces")
        for row in resp.json()["data"]:
            assert isinstance(row["duration_ms"], float)

    def test_single_export_spans_duration_float(self, client):
        """#44: Span-duration_ms im Einzel-Export ist float."""
        c, ids = client
        data = c.get(f"/api/traces/{ids['t1']}/export").json()
        for span in data["spans"]:
            if span["duration_ms"] is not None:
                assert isinstance(span["duration_ms"], float)


# ---------------------------------------------------------------------------
# Edge Cases (#40-#41, #52-#53)
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_attr_search_empty_key(self, client):
        """#40: attr_search mit leerem Key."""
        c, _ = client
        resp = c.get("/api/traces?attr_search=%3Dvalue")
        assert resp.status_code == 200

    def test_attr_search_empty_value(self, client):
        """#41: attr_search mit leerem Value."""
        c, _ = client
        resp = c.get("/api/traces?attr_search=model%3D")
        assert resp.status_code == 200

    def test_normalize_to_date_with_time(self):
        """#52: _normalize_to_date laesst zeitbehaftete Werte unveraendert."""
        assert _normalize_to_date("2026-04-02 12:00:00") == "2026-04-02 12:00:00"
        assert _normalize_to_date("2026-04-02T12:00:00") == "2026-04-02T12:00:00"

    def test_normalize_to_date_date_only(self):
        """#52: _normalize_to_date ergaenzt date-only um 23:59:59."""
        assert _normalize_to_date("2026-04-02") == "2026-04-02 23:59:59"

    def test_normalize_to_date_none(self):
        """_normalize_to_date mit None."""
        assert _normalize_to_date(None) is None

    def test_trace_status_aggregation(self):
        """#53: Aggregierter Status: ERROR > UNSET > OK."""
        assert _trace_status_from_spans([{"status": "OK"}, {"status": "ERROR"}]) == "ERROR"
        assert _trace_status_from_spans([{"status": "OK"}, {"status": "UNSET"}]) == "UNSET"
        assert _trace_status_from_spans([{"status": "OK"}]) == "OK"
        assert _trace_status_from_spans([{}]) == "OK"  # Kein Status → OK


# ---------------------------------------------------------------------------
# Detail-Endpunkt (#54-#55)
# ---------------------------------------------------------------------------


class TestDetail:
    def test_get_trace_detail(self, client):
        """#54: Detail-Endpunkt liefert Root-Metadaten + Spans."""
        c, ids = client
        resp = c.get(f"/api/traces/{ids['t1']}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["trace_id"] == ids["t1"]
        assert data["root_span"] == "rag_query"
        assert isinstance(data["duration_ms"], float)
        assert data["span_count"] == 2
        assert len(data["spans"]) == 2

    def test_get_trace_404(self, client):
        """#55: 404 bei nicht-existentem Trace."""
        c, _ = client
        resp = c.get("/api/traces/NONEXISTENT_TRACE_XYZ")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Pagination + Sortierung (#48, #59)
# ---------------------------------------------------------------------------


class TestPagination:
    def test_list_has_id_and_pagination(self, client):
        """#48: Jede Zeile hat id == trace_id und pagination-Block."""
        c, _ = client
        data = c.get("/api/traces").json()
        assert "pagination" in data
        for row in data["data"]:
            assert row["id"] == row["trace_id"]

    def test_pagination_and_sort(self, client):
        """#59: Pagination + Sortierung."""
        c, _ = client
        resp = c.get("/api/traces?page=1&page_size=2&sort_field=duration_ms&sort_dir=asc")
        data = resp.json()
        assert data["pagination"]["page"] == 1
        assert data["pagination"]["page_size"] == 2
        assert len(data["data"]) <= 2


# ---------------------------------------------------------------------------
# PgSpanExporter Kompatibilitaet (#62)
# ---------------------------------------------------------------------------


class TestExporterCompat:
    @pytest.mark.skipif(
        not _has_otel(),
        reason="opentelemetry SDK nicht installiert",
    )
    def test_pg_span_exporter_against_fresh_schema(self, trace_db):
        """#62: PgSpanExporter.export() gegen fresh create_tables()-Schema."""
        from unittest.mock import MagicMock

        from kiara.shared._pg_span_exporter import PgSpanExporter

        exporter = PgSpanExporter()

        # Mock-Span erstellen
        span = MagicMock()
        ctx = MagicMock()
        ctx.trace_id = int(uuid.uuid4().hex[:32], 16)
        ctx.span_id = int(uuid.uuid4().hex[:16], 16)
        span.get_span_context.return_value = ctx
        span.parent = None
        span.name = "test_exporter_span"
        span.start_time = 1_700_000_000_000_000_000
        span.end_time = 1_700_000_001_000_000_000
        span.attributes = {"test_key": "test_value"}
        span.status = MagicMock()
        span.status.status_code = MagicMock()
        span.status.status_code.name = "OK"

        result = exporter.export([span])

        from opentelemetry.sdk.trace.export import SpanExportResult

        assert result == SpanExportResult.SUCCESS

        # Verifiziere, dass der Span in der DB ist
        row = trace_db.execute(
            "SELECT * FROM traces WHERE span_id = %s",
            (format(ctx.span_id, "016x"),),
        ).fetchone()
        assert row is not None
        assert row["name"] == "test_exporter_span"


# ---------------------------------------------------------------------------
# Histogram-Identische-Dauern (#16)
# ---------------------------------------------------------------------------


class TestHistogramEdge:
    def test_identical_durations(self, client):
        """#16: Histogram mit identischen Dauern → kuenstliche Spreizung."""
        c, _ = client
        # Alle unsere Test-Traces haben unterschiedliche Dauern,
        # aber der Endpunkt sollte trotzdem nicht crashen
        resp = c.get("/api/traces/stats/duration-histogram")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] > 0

    def test_export_over_limit(self, client):
        """#64: Export bei > limit Traces → 400."""
        c, _ = client
        # Wir haben 4 Traces, setze limit=2
        resp = c.get("/api/traces/export?fmt=json&limit=2")
        assert resp.status_code == 400
        assert "begrenzt" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Fehlende Testfaelle (#3, #23, #39, #42, #50, #51)
# ---------------------------------------------------------------------------


class TestDateRangeFilter:
    def test_filter_from_and_to_date(self, client):
        """#3: Zeitraum from_date+to_date schraenkt auf start_time_ns ein."""
        c, _ = client
        # Seed-Traces liegen bei ~2023-11-14 (1_700_000_000 s).
        # Zeitraum umschliesst sie:
        resp = c.get("/api/traces?from_date=2023-11-14&to_date=2023-11-15")
        data = resp.json()
        assert data["pagination"]["total_items"] >= 1
        # Zeitraum ausserhalb:
        resp2 = c.get("/api/traces?from_date=2020-01-01&to_date=2020-01-02")
        assert resp2.json()["pagination"]["total_items"] == 0


class TestCompareDuplicateSpanNames:
    def test_compare_duplicate_span_names(self, client):
        """#23: Diff bei doppelten Span-Namen — summierte Dauern + count."""
        c, ids = client
        # Vergleiche t4 (agent_loop mit 2x tool_call) gegen sich selbst
        resp = c.get(f"/api/traces/compare?trace_a={ids['t4']}&trace_b={ids['t4']}")
        assert resp.status_code == 200
        data = resp.json()

        tool_diff = next(d for d in data["diff"] if d["span_name"] == "tool_call")
        # Beide Seiten haben 2 tool_call-Spans
        assert tool_diff["a_count"] == 2
        assert tool_diff["b_count"] == 2
        # delta == 0 bei identischem Trace
        assert tool_diff["delta_ms"] == 0
        # a_ms sollte die Summe beider tool_call-Dauern sein (900+1500 = 2400 ms)
        assert tool_diff["a_ms"] == tool_diff["b_ms"]
        assert tool_diff["a_ms"] == 2400.0

    def test_compare_agent_vs_simple(self, client):
        """#23 erweitert: Agent-Trace vs. einfacher Trace — fehlende Spans."""
        c, ids = client
        resp = c.get(f"/api/traces/compare?trace_a={ids['t4']}&trace_b={ids['t1']}")
        data = resp.json()

        # t4 hat tool_call + agent_loop, t1 hat rerank + rag_query
        span_names = {d["span_name"] for d in data["diff"]}
        assert "tool_call" in span_names
        assert "rerank" in span_names

        tool_diff = next(d for d in data["diff"] if d["span_name"] == "tool_call")
        assert tool_diff["a_ms"] is not None  # In t4 vorhanden
        assert tool_diff["b_ms"] is None  # In t1 nicht vorhanden


class TestCsvTimestamp:
    def test_csv_start_time_is_readable(self, client):
        """#39: CSV start_time ist lesbarer Timestamp, kein Python-Objekt."""
        c, _ = client
        resp = c.get("/api/traces/export?fmt=csv")
        assert resp.status_code == 200
        lines = resp.text.strip().split("\n")
        assert len(lines) >= 2
        # Zweite Zeile = erster Datensatz
        fields = lines[1].split(";")
        start_time_csv = fields[2]  # trace_id;root_span;start_time;...
        # Darf kein Python-repr sein (z.B. "datetime.datetime(...)")
        assert "datetime" not in start_time_csv
        # Muss ein ISO-8601-Timestamp mit 'T' oder ein lesbares Datum sein
        assert len(start_time_csv) >= 10


class TestHistogramBoundaryClamp:
    def test_bucket_clamping_no_duplicates(self, client):
        """#42: Histogram-Buckets bei Werten an min/max-Grenze — kein Doppelzaehlen."""
        c, _ = client
        resp = c.get("/api/traces/stats/duration-histogram?buckets=5")
        assert resp.status_code == 200
        data = resp.json()

        # Summe aller Bucket-Counts muss exakt total entsprechen
        bucket_sum = sum(b["count"] for b in data["buckets"])
        assert bucket_sum == data["total"], (
            f"Bucket-Summe {bucket_sum} != total {data['total']} — "
            "Overflow/Underflow-Clamping fehlerhaft"
        )

        # Jeder Bucket darf nur einmal vorkommen (keine doppelten Eintraege)
        from_values = [b["from_ms"] for b in data["buckets"]]
        assert len(from_values) == len(set(from_values))


class TestFilterScopeConsistency:
    def test_histogram_uses_same_scope_as_list(self, client):
        """#50: Histogram mit Filtern verwendet denselben Trace-Scope wie Liste."""
        c, _ = client
        # Filtere auf status=ERROR — nur t2 matcht
        list_resp = c.get("/api/traces?status=ERROR")
        list_total = list_resp.json()["pagination"]["total_items"]

        hist_resp = c.get("/api/traces/stats/duration-histogram?status=ERROR")
        hist_total = hist_resp.json()["total"]

        assert list_total == hist_total, (
            f"Liste hat {list_total} Traces, Histogram zaehlt {hist_total}"
        )

    def test_span_averages_scoped_to_filtered_traces(self, client):
        """#51: Span-Averages aggregiert nur ueber Spans der gefilterten Menge."""
        c, _ = client
        # Ohne Filter: Alle Span-Typen sichtbar
        all_resp = c.get("/api/traces/stats/span-averages")
        all_names = {s["name"] for s in all_resp.json()["spans"]}

        # Mit name=hybrid_retrieve: nur Spans von t3
        filtered_resp = c.get("/api/traces/stats/span-averages?name=hybrid_retrieve")
        filtered_names = {s["name"] for s in filtered_resp.json()["spans"]}

        # Gefiltert darf nur hybrid_retrieve enthalten (Root-Span von t3)
        assert "hybrid_retrieve" in filtered_names
        # rerank/graph_retrieve gehoeren zu anderen Traces und duerfen NICHT auftauchen
        assert "rerank" not in filtered_names
        assert "graph_retrieve" not in filtered_names
        # Die ungefilterte Menge muss breiter sein
        assert len(all_names) > len(filtered_names)


class TestTraceLookupHelper:
    def test_returns_none_for_unknown_trace(self, trace_db):
        from kiara.web.api_tracing import _get_trace_by_id

        assert _get_trace_by_id("nonexistent-trace-id") is None

    def test_returns_dict_for_known_trace(self, seeded_db):
        from kiara.web.api_tracing import _get_trace_by_id

        ids = seeded_db[1]
        result = _get_trace_by_id(ids["t1"])
        assert result["trace_id"] == ids["t1"]
        assert result["root_span"] == "rag_query"
        assert result["span_count"] == 2
        assert isinstance(result["spans"], list)


# ---------------------------------------------------------------------------
# Diagnostics (Sprint 4 AP1) — leerer Trace-Explorer
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_client(trace_db):
    """TestClient ohne Seed-Daten fuer Diagnostics-Tests."""
    conn = trace_db
    conn.execute("DELETE FROM traces")
    conn.commit()

    app = FastAPI()
    app.include_router(router)

    async def mock_admin():
        return {"id": 1, "username": "admin", "is_admin": True}

    app.dependency_overrides[get_admin_user] = mock_admin

    with TestClient(app) as c:
        yield c, conn


class TestDiagnosticsEmptyTraces:
    """AP1: Diagnostics bei leerem Trace-Explorer."""

    def test_empty_list_has_diagnostics(self, empty_client):
        """Bei total==0 liefert /api/traces ein diagnostics-Objekt."""
        c, _ = empty_client
        resp = c.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []
        assert data["pagination"]["total_items"] == 0
        assert "diagnostics" in data

    def test_diagnostics_tracing_enabled_field(self, empty_client):
        """diagnostics.tracing_enabled spiegelt Config-Wert wider."""
        c, _ = empty_client
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        # tracing_enabled ist bool oder None (bei Config-Fehler)
        assert "tracing_enabled" in diag
        assert isinstance(diag["tracing_enabled"], (bool, type(None)))

    def test_diagnostics_trace_counts(self, empty_client):
        """diagnostics enthaelt trace_span_count und trace_root_count."""
        c, _ = empty_client
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        assert diag["trace_span_count"] == 0
        assert diag["trace_root_count"] == 0

    def test_diagnostics_query_log_counts(self, empty_client):
        """diagnostics enthaelt query_log_count und query_log_trace_id_count."""
        c, _ = empty_client
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        assert "query_log_count" in diag
        assert "query_log_trace_id_count" in diag
        assert isinstance(diag["query_log_count"], int)
        assert isinstance(diag["query_log_trace_id_count"], int)

    def test_diagnostics_query_log_without_trace_id(self, empty_client):
        """query_log-Rows ohne trace_id: query_log_count > 0, query_log_trace_id_count == 0."""
        c, conn = empty_client
        conn.execute(
            "INSERT INTO query_log (user_id, question_hash, question_excerpt, answer_excerpt) "
            "VALUES (%s, %s, %s, %s)",
            (1, "testhash_diag", "test frage", "test antwort"),
        )
        conn.commit()
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        assert diag["query_log_count"] >= 1
        assert diag["query_log_trace_id_count"] == 0

    def test_no_diagnostics_when_data_present(self, client):
        """Bei vorhandenen Traces kein diagnostics-Feld."""
        c, _ = client
        data = c.get("/api/traces").json()
        assert data["pagination"]["total_items"] > 0
        assert "diagnostics" not in data

    def test_data_and_pagination_always_present(self, empty_client):
        """data und pagination sind immer vorhanden, auch bei Diagnostics."""
        c, _ = empty_client
        data = c.get("/api/traces").json()
        assert "data" in data
        assert "pagination" in data
        assert data["pagination"]["page"] == 1


class TestDiagnosticsDataMinimization:
    """Diagnostics enthaelt nur aggregierte Counts/Flags."""

    def test_no_query_log_content(self, empty_client):
        """diagnostics enthaelt keine query_log-Inhalte."""
        c, conn = empty_client
        conn.execute(
            "INSERT INTO query_log (user_id, question_hash, question_excerpt, answer_excerpt) "
            "VALUES (%s, %s, %s, %s)",
            (1, "testhash_secret", "geheime frage", "geheime antwort"),
        )
        conn.commit()
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        diag_str = json.dumps(diag)
        assert "geheime" not in diag_str
        assert "frage" not in diag_str
        assert "antwort" not in diag_str

    def test_no_trace_id_lists(self, empty_client):
        """diagnostics enthaelt keine Trace-ID-Listen."""
        c, _ = empty_client
        data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        for v in diag.values():
            assert not isinstance(v, list), f"Diagnostics-Wert darf keine Liste sein: {v}"


class TestDiagnosticsConfigError:
    """Fehlerpfad: load_config()-Exception bricht /api/traces nicht."""

    def test_config_error_still_200(self, empty_client):
        """load_config-Exception → HTTP 200 mit diagnostics_error."""
        from unittest.mock import patch

        c, _ = empty_client
        with patch("kiara.web.api_tracing.load_config", side_effect=RuntimeError("config kaputt")):
            resp = c.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["data"] == []
        assert "diagnostics" in data
        assert data["diagnostics"]["tracing_enabled"] is None
        assert "diagnostics_error" in data
        assert "config kaputt" in data["diagnostics_error"]

    def test_config_error_preserves_counts(self, empty_client):
        """Auch bei Config-Fehler werden DB-Counts geliefert."""
        from unittest.mock import patch

        c, _ = empty_client
        with patch("kiara.web.api_tracing.load_config", side_effect=RuntimeError("boom")):
            data = c.get("/api/traces").json()
        diag = data["diagnostics"]
        assert "trace_span_count" in diag
        assert "query_log_count" in diag


class TestDiagnosticsReverseConsumerCompat:
    """Reverse-Consumer: admin/functions/tracing.py Imports bleiben stabil."""

    def test_count_traces_importable(self):
        """_count_traces ist importierbar und callable."""
        from kiara.web.api_tracing import _count_traces

        assert callable(_count_traces)

    def test_query_traces_importable(self):
        """_query_traces ist importierbar und callable."""
        from kiara.web.api_tracing import _query_traces

        assert callable(_query_traces)

    def test_get_trace_by_id_importable(self):
        """_get_trace_by_id ist importierbar."""
        from kiara.web.api_tracing import _get_trace_by_id

        assert callable(_get_trace_by_id)

    def test_build_trace_scope_params_importable(self):
        """_build_trace_scope_params ist importierbar."""
        from kiara.web.api_tracing import _build_trace_scope_params

        assert callable(_build_trace_scope_params)

    def test_sql_constants_importable(self):
        """SQL-Konstanten sind importierbar."""
        from kiara.web.api_tracing import (
            BOUNDS_SQL,
            HISTOGRAM_SQL,
            SPAN_AVERAGES_SQL,
            SPAN_QUERY_SQL,
        )

        assert isinstance(SPAN_QUERY_SQL, str)
        assert isinstance(BOUNDS_SQL, str)
        assert isinstance(HISTOGRAM_SQL, str)
        assert isinstance(SPAN_AVERAGES_SQL, str)

    def test_build_trace_scope_params_admin_call_shape(self):
        """Admin-Consumer rufen ohne span_name/attr_search auf — muss 15-Tuple liefern."""
        from kiara.web.api_tracing import _build_trace_scope_params

        result = _build_trace_scope_params(
            name=None,
            min_duration_ms=None,
            status=None,
            from_date=None,
            to_date=None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 15
        # span_name-Slots (Index 10+11) muessen None sein
        assert result[10] is None
        assert result[11] is None
        # attr_filter_active muss False sein
        assert result[12] is False

    def test_build_trace_scope_params_tuple_length(self):
        """_build_trace_scope_params liefert ein 15-Werte-Tuple (DB-frei)."""
        from kiara.web.api_tracing import _build_trace_scope_params

        result = _build_trace_scope_params(
            name=None,
            min_duration_ms=None,
            status=None,
            from_date=None,
            to_date=None,
            span_name=None,
            attr_search=None,
        )
        assert isinstance(result, tuple)
        assert len(result) == 15

    def test_build_trace_scope_params_with_values(self):
        """_build_trace_scope_params mit konkreten Werten — 15 Positionen stabil."""
        from kiara.web.api_tracing import _build_trace_scope_params

        result = _build_trace_scope_params(
            name="rag_query",
            min_duration_ms=100.0,
            status="OK",
            from_date="2026-01-01",
            to_date="2026-01-31",
            span_name="rerank",
            attr_search="model=llama3.1",
        )
        assert isinstance(result, tuple)
        assert len(result) == 15
        # Positions-Duplikation: name an [0]+[1], min_duration_ms an [2]+[3] usw.
        assert result[0] == "rag_query"
        assert result[1] == "rag_query"
        assert result[10] == "rerank"
        assert result[11] == "rerank"
        # attr_filter_active + key + value
        assert result[12] is True
        assert result[13] == "model"
        assert result[14] == "llama3.1"

    def test_build_trace_scope_params_rejects_large_date_range(self):
        """Explizite Trace-Zeitfenster sind auf 31 Tage begrenzt."""
        from fastapi import HTTPException

        from kiara.web.api_tracing import _build_trace_scope_params

        with pytest.raises(HTTPException) as exc:
            _build_trace_scope_params(
                name=None,
                min_duration_ms=None,
                status=None,
                from_date="2026-01-01",
                to_date="2026-03-01",
                span_name=None,
                attr_search=None,
            )

        assert exc.value.status_code == 400
        assert "maximal 31 Tage" in str(exc.value.detail)

    def test_build_trace_scope_params_rejects_reverse_date_range(self):
        """to_date muss nach from_date liegen."""
        from fastapi import HTTPException

        from kiara.web.api_tracing import _build_trace_scope_params

        with pytest.raises(HTTPException) as exc:
            _build_trace_scope_params(
                name=None,
                min_duration_ms=None,
                status=None,
                from_date="2026-02-01",
                to_date="2026-01-01",
                span_name=None,
                attr_search=None,
            )

        assert exc.value.status_code == 400
        assert "to_date muss nach from_date" in str(exc.value.detail)

    def test_count_traces_call_shape(self, empty_client):
        """_count_traces mit 7 Positions-Argumenten (wie admin/functions/tracing.py)."""
        from kiara.web.api_tracing import _count_traces

        result = _count_traces(None, None, None, None, None, None, None)
        assert isinstance(result, int)
        assert result == 0

    def test_query_traces_call_shape(self, empty_client):
        """_query_traces mit 8 Positions- + kwargs (wie admin/functions/tracing.py)."""
        from kiara.web.api_tracing import _query_traces

        result = _query_traces(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            10,
            offset=0,
            sort_field="start_time",
            sort_dir="desc",
        )
        assert isinstance(result, list)
