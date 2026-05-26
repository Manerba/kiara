"""Tests fuer System-Monitor: Admin-Funktionen (system.metrics, system.journal).

Testet AdminResult-Format, leere/volle Zustande, subprocess-Mocking — 6 Unit-Tests.
"""

import asyncio
from unittest.mock import MagicMock, patch

import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"


def _run(coro):
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


class TestSystemMetrics:
    @patch("kiara.monitoring.collector.get_latest")
    def test_metrics_ok(self, mock_latest):
        """system.metrics liefert AdminResult.ok mit Metriken."""
        mock_latest.return_value = {
            "cpu_percent": 15.0,
            "ram_used_bytes": 4_000_000_000,
            "ram_total_bytes": 8_000_000_000,
            "load_1m": 1.0,
            "load_5m": 0.8,
            "load_15m": 0.5,
            "disk_used_bytes": 50_000_000_000,
            "disk_total_bytes": 100_000_000_000,
        }
        from kiara.admin.functions.monitoring import system_metrics

        result = _run(system_metrics(MOCK_USER))

        assert result.success is True
        assert "15.0%" in result.message
        assert result.data["cpu_percent"] == 15.0

    @patch("kiara.monitoring.collector.get_latest")
    def test_metrics_collector_not_running(self, mock_latest):
        """system.metrics bei leerem Collector → fail."""
        mock_latest.return_value = {}
        from kiara.admin.functions.monitoring import system_metrics

        result = _run(system_metrics(MOCK_USER))

        assert result.success is False
        assert "Collector" in result.message


class TestSystemJournal:
    @patch("kiara.admin.functions.monitoring.subprocess")
    @patch("kiara.admin.functions.services.ALLOWED_SERVICES", ["kiara.service"])
    def test_journal_with_errors(self, mock_subprocess):
        """system.journal liefert Fehlerzeilen aus journalctl."""
        mock_result = MagicMock()
        mock_result.stdout = (
            "2026-03-29T10:12:05+0100 host kiara.service[123]: Connection refused\n"
            "2026-03-29T10:11:00+0100 host kiara.service[123]: Timeout\n"
        )
        mock_subprocess.run.return_value = mock_result

        from kiara.admin.functions.monitoring import system_journal

        result = _run(system_journal(MOCK_USER, max_lines=10))

        assert result.success is True
        assert len(result.data["errors"]) == 2
        # Absteigend sortiert (neueste zuerst)
        assert "10:12" in result.data["errors"][0]

    @patch("kiara.admin.functions.monitoring.subprocess")
    @patch("kiara.admin.functions.services.ALLOWED_SERVICES", ["kiara.service"])
    def test_journal_no_errors(self, mock_subprocess):
        """system.journal ohne Fehler → Erfolgsmeldung."""
        mock_result = MagicMock()
        mock_result.stdout = "-- No entries --\n"
        mock_subprocess.run.return_value = mock_result

        from kiara.admin.functions.monitoring import system_journal

        result = _run(system_journal(MOCK_USER))

        assert result.success is True
        assert "Keine Fehler" in result.message
        assert result.data["errors"] == []

    @patch("kiara.admin.functions.services.ALLOWED_SERVICES", [])
    def test_journal_empty_services(self):
        """system.journal bei leerer Service-Liste → leere Ergebnisse."""
        from kiara.admin.functions.monitoring import system_journal

        result = _run(system_journal(MOCK_USER))

        assert result.success is True
        assert result.data["errors"] == []

    @patch("kiara.admin.functions.monitoring.subprocess")
    @patch("kiara.admin.functions.services.ALLOWED_SERVICES", ["kiara.service"])
    def test_journal_subprocess_timeout(self, mock_subprocess):
        """subprocess-Timeout wird abgefangen, kein Crash."""
        import subprocess as sp

        mock_subprocess.run.side_effect = sp.TimeoutExpired(cmd="journalctl", timeout=10)

        from kiara.admin.functions.monitoring import system_journal

        result = _run(system_journal(MOCK_USER))

        assert result.success is True
        assert result.data["errors"] == []
