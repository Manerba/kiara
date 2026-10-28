"""Tests fuer Sprint 09: Testsuite Admin-Funktionen.

Testet testsuite.run, .status — 5 Unit-Tests.
"""

import asyncio
import subprocess
import time
from unittest.mock import MagicMock, patch


def _run(coro):
    """Fuehrt eine Coroutine synchron aus."""
    return asyncio.run(coro)


MOCK_USER = {"id": 1, "username": "admin"}


# ---------------------------------------------------------------------------
# testsuite.run
# ---------------------------------------------------------------------------


class TestTestsuiteRun:
    @patch("kiara.web.api_testsuite._run_tests")
    @patch("kiara.web.api_testsuite._reset_status")
    @patch("kiara.web.api_testsuite._test_status", {"running": False})
    def test_run_starts(self, mock_reset, mock_run):
        """Test 1: Tests starten wenn nicht laufend."""
        from kiara.admin.functions.testsuite import testsuite_run

        result = _run(testsuite_run(MOCK_USER))

        assert result.success is True
        assert "gestartet" in result.data["message"]
        # Die Produktionsfunktion ruft _test_status.update() direkt auf
        # (innerhalb eines _status_lock-Blocks) und startet einen Thread
        # mit _run_tests als Target — _reset_status wird nicht aufgerufen.

    @patch("kiara.web.api_testsuite._test_status", {"running": True})
    def test_run_already_running(self):
        """Test 2: Tests laufen bereits → fail."""
        from kiara.admin.functions.testsuite import testsuite_run

        result = _run(testsuite_run(MOCK_USER))

        assert result.success is False
        assert "laeuft bereits" in result.message


# ---------------------------------------------------------------------------
# testsuite.status
# ---------------------------------------------------------------------------


class TestTestsuiteStatus:
    @patch(
        "kiara.web.api_testsuite._test_status",
        {
            "running": False,
            "progress": 0,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "current_test": "",
            "duration": 0.0,
            "results": [],
            "error": None,
        },
    )
    def test_status_idle(self):
        """Test 3: Idle-Status."""
        from kiara.admin.functions.testsuite import testsuite_status

        result = _run(testsuite_status(MOCK_USER))

        assert result.success is True
        assert result.data["running"] is False

    @patch(
        "kiara.web.api_testsuite._test_status",
        {
            "running": True,
            "progress": 5,
            "total": 100,
            "passed": 3,
            "failed": 1,
            "errors": 1,
            "skipped": 0,
            "current_test": "tests/test_foo.py::test_bar",
            "duration": 0.0,
            "results": [],
            "error": None,
        },
    )
    def test_status_running(self):
        """Test 4: Laufende Tests zeigen Fortschritt."""
        from kiara.admin.functions.testsuite import testsuite_status

        result = _run(testsuite_status(MOCK_USER))

        assert result.success is True
        assert result.data["running"] is True
        assert result.data["progress"] == 5
        assert result.data["total"] == 100
        assert "5/100" in result.message

    @patch(
        "kiara.web.api_testsuite._test_status",
        {
            "running": False,
            "progress": 100,
            "total": 100,
            "passed": 95,
            "failed": 3,
            "errors": 2,
            "skipped": 0,
            "current_test": "",
            "duration": 42.5,
            "results": [{"name": "test_a", "status": "passed", "details": ""}],
            "error": None,
        },
    )
    def test_status_completed(self):
        """Test 5: Abgeschlossene Tests mit Ergebnissen."""
        from kiara.admin.functions.testsuite import testsuite_status

        result = _run(testsuite_status(MOCK_USER))

        assert result.success is True
        assert "95 bestanden" in result.message
        assert "3 fehlgeschlagen" in result.message
        assert result.data["duration"] == 42.5


# ---------------------------------------------------------------------------
# testsuite backend watchdog
# ---------------------------------------------------------------------------


class TestTestsuiteWatchdog:
    def test_run_tests_times_out_when_stdout_hangs(self):
        """Der Watchdog muss auch ohne neue stdout-Zeile feuern."""
        from kiara.web import api_testsuite as mod

        class HangingStdout:
            def readline(self):
                time.sleep(0.2)
                return ""

        class HangingProc:
            def __init__(self):
                self.stdout = HangingStdout()
                self.returncode = None
                self.killed = False

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout=None):
                if self.killed:
                    return self.returncode
                raise subprocess.TimeoutExpired("pytest", timeout)

        proc = HangingProc()
        collect = MagicMock(stdout="1 test collected in 0.01s\n")

        with (
            patch.object(mod, "_save_results"),
            patch.object(mod, "_TESTSUITE_RUN_TIMEOUT_SECONDS", 0.01),
            patch.object(mod, "_STDOUT_POLL_SECONDS", 0.001),
            patch.object(mod.subprocess, "run", return_value=collect),
            patch.object(mod.subprocess, "Popen", return_value=proc),
        ):
            mod._reset_status()
            mod._run_tests()

        assert proc.killed is True
        assert mod._test_status["running"] is False
        assert "Timeout" in mod._test_status["error"]
