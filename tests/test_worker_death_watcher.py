"""Tests fuer `kiara.web.worker_death_watcher`-Modulkomponenten.

Modul-Tests, kein systemd-E2E. Subprocess-Aufrufe werden gemockt; die
Daemon-Thread- und `signal.pause()`-Logik der `main()`-Funktion ist out
of scope (E2E-Domaene).
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from kiara.web import worker_death_watcher as wdw


@pytest.fixture
def isolated_cursor_dir(tmp_path, monkeypatch):
    """Lenkt _CURSOR_DIR auf einen leeren tmp_path.

    Wirkung: cursor_file.exists() liefert False -> der Watcher startet
    journalctl ohne `--after-cursor`-Argument und beruehrt keinen echten
    Pfad unter /var/lib/kiara/.
    """
    fake_dir = tmp_path / "watcher_cursors"
    fake_dir.mkdir()
    monkeypatch.setattr(wdw, "_CURSOR_DIR", fake_dir)
    return fake_dir


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_calls_under_limit(self):
        rl = wdw._RateLimiter(window=60.0, limit=3)
        assert rl.allow(now=100.0) is True
        assert rl.allow(now=101.0) is True
        assert rl.allow(now=102.0) is True

    def test_denies_when_limit_reached_in_window(self):
        rl = wdw._RateLimiter(window=60.0, limit=3)
        for t in (100.0, 101.0, 102.0):
            rl.allow(t)
        assert rl.allow(now=103.0) is False

    def test_window_slides_and_allows_again(self):
        rl = wdw._RateLimiter(window=60.0, limit=2)
        rl.allow(now=100.0)
        rl.allow(now=110.0)
        assert rl.allow(now=120.0) is False
        # Spaeter als window+oldest -> beide alten Eintraege fallen raus
        assert rl.allow(now=170.0) is True

    def test_constants_match_documented_values(self):
        """Der RateLimiter wird mit den Modul-Konstanten initialisiert.

        Die Konstanten sind Teil des Operator-Vertrags (Doku in der Modul-
        Docstring + Service-Unit). Aenderung muss bewusst erfolgen.
        """
        assert wdw._GLOBAL_RATE_WINDOW == 60.0
        assert wdw._GLOBAL_RATE_LIMIT == 6
        assert wdw._PER_PID_DEDUP_SECONDS == 30.0


# ---------------------------------------------------------------------------
# _is_service_stopping
# ---------------------------------------------------------------------------


class TestIsServiceStopping:
    @pytest.mark.parametrize(
        "state,expected",
        [
            ("active", False),
            ("inactive", True),
            ("deactivating", True),
            ("activating", True),
            ("failed", True),
        ],
    )
    def test_state_mapping(self, state, expected):
        with patch("kiara.web.worker_death_watcher.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout=f"{state}\n", returncode=0)
            assert wdw._is_service_stopping("kiara.service") is expected

    def test_returns_false_when_systemctl_missing(self):
        with patch("kiara.web.worker_death_watcher.subprocess.run", side_effect=FileNotFoundError):
            assert wdw._is_service_stopping("kiara.service") is False

    def test_returns_false_when_systemctl_times_out(self):
        with patch(
            "kiara.web.worker_death_watcher.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="systemctl", timeout=5),
        ):
            assert wdw._is_service_stopping("kiara.service") is False

    def test_returns_false_when_oserror(self):
        with patch("kiara.web.worker_death_watcher.subprocess.run", side_effect=OSError("boom")):
            assert wdw._is_service_stopping("kiara.service") is False


# ---------------------------------------------------------------------------
# _DIED_RE
# ---------------------------------------------------------------------------


class TestDiedRegex:
    @pytest.mark.parametrize(
        "line,pid",
        [
            ("INFO:     Child process [12345] died", "12345"),
            ("Child process [1] died", "1"),
            (
                "2026-04-30T12:00:00+0200 host service[42]: INFO:     Child process [4242] died",
                "4242",
            ),
        ],
    )
    def test_matches_real_uvicorn_lines(self, line, pid):
        match = wdw._DIED_RE.search(line)
        assert match is not None
        assert match.group(1) == pid

    @pytest.mark.parametrize(
        "line",
        [
            "INFO:     Child process started",
            "Child process [12345] running",
            "child process [12345] died",  # case-sensitive: "Child" mit grossem C
            "Child process [abc] died",
            "Child process died",
        ],
    )
    def test_non_match(self, line):
        assert wdw._DIED_RE.search(line) is None


# ---------------------------------------------------------------------------
# _trigger_log_command
# ---------------------------------------------------------------------------


class TestTriggerLogCommand:
    def test_invokes_kiara_log_worker_death(self):
        with patch("kiara.web.worker_death_watcher.subprocess.run") as run:
            run.return_value = SimpleNamespace(stdout="", returncode=0)
            wdw._trigger_log_command("kiara.service", "4242")
        run.assert_called_once()
        args = run.call_args.args[0]
        assert args == [
            "/opt/kiara/venv/bin/kiara",
            "log-worker-death",
            "4242",
            "kiara.service",
        ]
        # capture-output, text und timeout sollten gesetzt sein
        kwargs = run.call_args.kwargs
        assert kwargs.get("capture_output") is True
        assert kwargs.get("text") is True
        assert kwargs.get("timeout") == 20

    def test_swallows_timeout_expired(self):
        with patch(
            "kiara.web.worker_death_watcher.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="kiara", timeout=20),
        ):
            # Kein Re-raise erwartet
            wdw._trigger_log_command("kiara.service", "1")

    def test_swallows_file_not_found(self):
        with patch("kiara.web.worker_death_watcher.subprocess.run", side_effect=FileNotFoundError):
            wdw._trigger_log_command("kiara.service", "1")

    def test_swallows_oserror(self):
        with patch(
            "kiara.web.worker_death_watcher.subprocess.run", side_effect=OSError("kein Speicher")
        ):
            wdw._trigger_log_command("kiara.service", "1")


# ---------------------------------------------------------------------------
# Per-PID-Dedup in _watch_service
# ---------------------------------------------------------------------------


def _stdout_iter(lines: list[str]):
    """Hilfsiterator fuer proc.stdout — endet nach allen Lines (kein Block)."""
    return iter(lines)


def _journal_lines(messages: list[str]) -> list[str]:
    """Wickelt Plain-Text-Messages in journalctl --output=json-Lines.

    Audit-Run-2 #2: Watcher erwartet JSON mit MESSAGE und __CURSOR-Feld.
    """
    import json

    out = []
    for idx, msg in enumerate(messages):
        out.append(
            json.dumps(
                {
                    "MESSAGE": msg.rstrip("\n"),
                    "__CURSOR": f"s=test;i={idx};b=test;m=test;t=test",
                    "__REALTIME_TIMESTAMP": str(1700000000000000 + idx),
                }
            )
            + "\n"
        )
    return out


def _journal_line(message: str, cursor: str | None) -> str:
    payload = {
        "MESSAGE": message.rstrip("\n"),
        "__REALTIME_TIMESTAMP": "1700000000000000",
    }
    if cursor is not None:
        payload["__CURSOR"] = cursor
    return json.dumps(payload) + "\n"


def _build_popen_mock(lines: list[str]):
    """Liefert einen Popen-Mock, dessen stdout `lines` durchiteriert."""
    proc = MagicMock()
    proc.stdout = _stdout_iter(lines)
    proc.stderr = MagicMock()
    proc.stderr.read.return_value = ""
    proc.pid = 999
    proc.returncode = 0
    proc.poll.return_value = 0
    proc.terminate.return_value = None
    proc.wait.return_value = 0
    return proc


def _watch_once(*args, **kwargs):
    """Test-Wrapper: ruft _watch_service mit max_iterations=1 auf, mockt sleep.

    Audit-Run-2 #3: Watcher hat einen Restart-Loop, ohne Iterations-Limit
    laufen Tests endlos. time.sleep wird auch gemockt, damit Tests schnell
    bleiben.
    """
    with patch("kiara.web.worker_death_watcher.time.sleep", return_value=None):
        wdw._watch_service(*args, max_iterations=1, **kwargs)


def _monotonic_sequence(*values: float):
    """side_effect-Helfer fuer time.monotonic — liefert die Werte und bleibt
    dann auf dem letzten Wert stehen, statt StopIteration zu werfen.

    `_watch_service` ruft monotonic mehrmals fuer loop_started_at / uptime
    zusaetzlich zu den Aufrufen in `_handle_line`. Tests, die nur die
    Match-Zeitpunkte vorgeben wollen, bekommen mit dieser Funktion einen
    stabilen Default fuer die uebrigen Aufrufe.
    """
    iterator = iter(values)
    last = [values[-1] if values else 100.0]

    def _next():
        try:
            v = next(iterator)
            last[0] = v
            return v
        except StopIteration:
            return last[0]

    return _next


class TestWatchServiceDedup:
    """Per-PID-Dedup verhindert Doppel-Trigger fuer dieselbe PID innerhalb 30s."""

    def test_two_died_lines_with_same_pid_within_window_trigger_once(
        self,
        isolated_cursor_dir,
    ):
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [12345] died",
                    "INFO:     Child process [12345] died",
                ]
            )
        )
        # Beide Lines werden zum gleichen monotonic-Tick verarbeitet.
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(100.0, 100.0, 100.5),
            ),
        ):
            _watch_once("kiara.service")

        assert trig.call_count == 1
        trig.assert_called_with("kiara.service", "12345")

    def test_same_pid_after_dedup_window_triggers_again(self, isolated_cursor_dir):
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [555] died",
                    "INFO:     Child process [555] died",
                ]
            )
        )
        # Erster Tick: 100.0, zweiter Tick: 131.0 (ueber 30s)
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(100.0, 100.0, 131.0),
            ),
        ):
            _watch_once("kiara.service")

        assert trig.call_count == 2

    def test_different_pids_both_trigger_in_same_window(self, isolated_cursor_dir):
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [111] died",
                    "INFO:     Child process [222] died",
                ]
            )
        )
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(100.0, 100.0, 100.1),
            ),
        ):
            _watch_once("kiara.service")

        assert trig.call_count == 2
        triggered_pids = {call.args[1] for call in trig.call_args_list}
        assert triggered_pids == {"111", "222"}

    def test_service_stopping_suppresses_trigger(self, isolated_cursor_dir):
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [777] died",
                ]
            )
        )
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=True),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch("kiara.web.worker_death_watcher.time.monotonic", return_value=100.0),
        ):
            _watch_once("kiara.service")

        trig.assert_not_called()

    def test_non_match_lines_are_skipped_silently(self, isolated_cursor_dir):
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "Some unrelated line",
                    "Child process started",
                    "INFO:     Child process [42] died",
                ]
            )
        )
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch("kiara.web.worker_death_watcher.time.monotonic", return_value=100.0),
        ):
            _watch_once("kiara.service")

        trig.assert_called_once_with("kiara.service", "42")

    def test_rate_limit_drops_excess_triggers_per_window(self, isolated_cursor_dir):
        # 7 verschiedene PIDs, alle in derselben 60s-Window -> Limit ist 6.
        proc = _build_popen_mock(
            _journal_lines([f"INFO:     Child process [{p}] died" for p in range(1, 8)])
        )
        # 7 monotonic-Werte, alle innerhalb von 60s
        ticks = [100.0 + i for i in range(7)]
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(*ticks),
            ),
        ):
            _watch_once("kiara.service")

        # Limit 6: 7. Trigger wird verworfen.
        assert trig.call_count == 6


class TestWatchServiceFailureRecovery:
    """Trigger-Fail-Pfade durch _watch_service.

    Ein fehlgeschlagener log-worker-death-Aufruf wird als Pending-State
    persistiert. Der Watcher liest journald erst weiter, wenn dieser Pending-
    State erfolgreich geloggt wurde oder das Retry-Limit erreicht ist.
    """

    def test_trigger_fail_aborts_stream_before_later_cursor_write(
        self,
        isolated_cursor_dir,
    ):
        """Nach TRIGGERED_FAIL wird ein Pending-Death geschrieben.

        Eine spaetere NO_MATCH-Line im selben journalctl-Stream darf den
        Cursor nicht weiterschreiben, solange das Death-Event nicht geloggt
        wurde.
        """
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        proc = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [666] died",  # Triggert Fail
                    "INFO:     irgendein Health-Check 200",  # NO_MATCH
                ]
            )
        )
        # _trigger_log_command returnt False -> TRIGGERED_FAIL
        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch("kiara.web.worker_death_watcher._trigger_log_command", return_value=False),
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(100.0, 100.0, 100.5, 100.6),
            ),
        ):
            _watch_once("kiara.service")

        assert not cursor_file.exists()
        pending = json.loads(pending_file.read_text())
        assert pending["service"] == "kiara.service"
        assert pending["pid"] == "666"
        assert pending["cursor"] == "s=test;i=0;b=test;m=test;t=test"
        assert pending["attempts"] == 1

    def test_pending_retry_success_writes_cursor_and_clears_pending(
        self,
        isolated_cursor_dir,
    ):
        """Ein erfolgreiches Pending-Retry setzt den Cursor hinter das Death-Event."""
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        assert wdw._write_pending_death_atomic(
            pending_file,
            wdw._PendingWorkerDeath(
                service="kiara.service",
                pid="666",
                message="INFO:     Child process [666] died",
                cursor="s=test;i=42;b=test;m=test;t=test",
                attempts=2,
            ),
        )
        proc = _build_popen_mock([])

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc) as popen,
            patch("kiara.web.worker_death_watcher._trigger_log_command", return_value=True) as trig,
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
            patch("kiara.web.worker_death_watcher.time.monotonic", return_value=100.0),
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_called_once_with("kiara.service", "666")
        assert cursor_file.read_text() == "s=test;i=42;b=test;m=test;t=test"
        assert not pending_file.exists()
        args = popen.call_args.args[0]
        assert "--after-cursor" in args
        assert "s=test;i=42;b=test;m=test;t=test" in args
        assert "--lines=0" not in args

    def test_pending_retry_failure_blocks_journal_and_increments_attempts(
        self,
        isolated_cursor_dir,
    ):
        """Solange Pending nicht geloggt ist, darf journald nicht weiterlaufen."""
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        assert wdw._write_pending_death_atomic(
            pending_file,
            wdw._PendingWorkerDeath(
                service="kiara.service",
                pid="777",
                message="INFO:     Child process [777] died",
                cursor="s=test;i=7;b=test;m=test;t=test",
                attempts=1,
            ),
        )

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen") as popen,
            patch(
                "kiara.web.worker_death_watcher._trigger_log_command", return_value=False
            ) as trig,
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_called_once_with("kiara.service", "777")
        popen.assert_not_called()
        pending = json.loads(pending_file.read_text())
        assert pending["attempts"] == 2
        assert not cursor_file.exists()

    def test_pending_retry_success_keeps_logged_pending_when_cursor_write_fails(
        self,
        isolated_cursor_dir,
    ):
        """Cursor-Write ist Teil des Pending-Commit-Protokolls."""
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        assert wdw._write_pending_death_atomic(
            pending_file,
            wdw._PendingWorkerDeath(
                service="kiara.service",
                pid="778",
                message="INFO:     Child process [778] died",
                cursor="s=test;i=8;b=test;m=test;t=test",
                attempts=1,
            ),
        )

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen") as popen,
            patch("kiara.web.worker_death_watcher._trigger_log_command", return_value=True) as trig,
            patch("kiara.web.worker_death_watcher._write_cursor_atomic", return_value=False),
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_called_once_with("kiara.service", "778")
        popen.assert_not_called()
        assert not cursor_file.exists()
        pending = json.loads(pending_file.read_text())
        assert pending["logged"] is True
        assert pending["cursor"] == "s=test;i=8;b=test;m=test;t=test"

    def test_pending_without_cursor_terminates_without_retry_or_journal(
        self,
        isolated_cursor_dir,
    ):
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        pending_file.write_text(
            json.dumps(
                {
                    "version": 1,
                    "service": "kiara.service",
                    "pid": "779",
                    "message": "INFO:     Child process [779] died",
                    "attempts": 1,
                }
            )
        )

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen") as popen,
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_not_called()
        popen.assert_not_called()

    def test_corrupt_pending_terminates_without_retry_or_journal(
        self,
        isolated_cursor_dir,
    ):
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        pending_file.write_text("{not-json")

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen") as popen,
            patch("kiara.web.worker_death_watcher._trigger_log_command") as trig,
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_not_called()
        popen.assert_not_called()

    def test_live_fail_then_pending_retry_success_advances_failed_cursor(
        self,
        isolated_cursor_dir,
    ):
        """Deckt den frischen No-Cursor-Fall ab: Fail geht nicht verloren."""
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        failed_stream = _build_popen_mock(
            _journal_lines(
                [
                    "INFO:     Child process [888] died",
                    "INFO:     irgendein Health-Check 200",
                ]
            )
        )
        resumed_stream = _build_popen_mock([])

        with (
            patch(
                "kiara.web.worker_death_watcher.subprocess.Popen",
                side_effect=[failed_stream, resumed_stream],
            ) as popen,
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch(
                "kiara.web.worker_death_watcher._trigger_log_command", side_effect=[False, True]
            ) as trig,
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(100.0, 100.0, 100.5, 101.0),
            ),
        ):
            wdw._watch_service("kiara.service", max_iterations=2)

        assert [call.args for call in trig.call_args_list] == [
            ("kiara.service", "888"),
            ("kiara.service", "888"),
        ]
        assert cursor_file.read_text() == "s=test;i=0;b=test;m=test;t=test"
        assert not pending_file.exists()
        resumed_args = popen.call_args_list[1].args[0]
        assert "--after-cursor" in resumed_args
        assert "s=test;i=0;b=test;m=test;t=test" in resumed_args

    def test_live_fail_without_cursor_terminates_without_pending(
        self,
        isolated_cursor_dir,
    ):
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        pending_file = wdw._pending_file_for(cursor_file)
        proc = _build_popen_mock(
            [
                _journal_line("INFO:     Child process [889] died", cursor=None),
            ]
        )

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch(
                "kiara.web.worker_death_watcher._trigger_log_command", return_value=False
            ) as trig,
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
            patch("kiara.web.worker_death_watcher.time.monotonic", return_value=100.0),
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        trig.assert_called_once_with("kiara.service", "889")
        assert not cursor_file.exists()
        assert not pending_file.exists()

    def test_corrupt_cursor_file_terminates_without_journalctl(
        self,
        isolated_cursor_dir,
    ):
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        cursor_file.write_text("definitely-not-a-cursor")

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen") as popen,
            patch("kiara.web.worker_death_watcher.time.sleep") as sleep,
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        popen.assert_not_called()
        sleep.assert_not_called()

    def test_journalctl_cursor_seek_failure_terminates_thread(
        self,
        isolated_cursor_dir,
    ):
        cursor_file = isolated_cursor_dir / "kiara.service.cursor"
        cursor_file.write_text("s=test;i=1;b=test;m=test;t=test")
        proc = _build_popen_mock([])
        proc.returncode = 1
        proc.poll.return_value = 1
        proc.stderr.read.return_value = "Failed to seek to cursor: Invalid argument"

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", return_value=proc),
            patch("kiara.web.worker_death_watcher.time.sleep") as sleep,
            patch("kiara.web.worker_death_watcher.time.monotonic", return_value=100.0),
        ):
            wdw._watch_service("kiara.service", max_iterations=1)

        sleep.assert_not_called()

    def test_consecutive_fail_limit_terminates_thread_via_return(
        self,
        isolated_cursor_dir,
    ):
        """Nach _CONSECUTIVE_FAIL_LIMIT failed Triggers muss
        _watch_service per `return` enden, damit main()/systemd den Thread
        bemerkt und neu startet.
        """

        def fake_popen(*_args, **_kwargs):
            return _build_popen_mock(
                _journal_lines(
                    [
                        "INFO:     Child process [1000] died",
                    ]
                )
            )

        with (
            patch("kiara.web.worker_death_watcher.subprocess.Popen", side_effect=fake_popen),
            patch("kiara.web.worker_death_watcher._is_service_stopping", return_value=False),
            patch(
                "kiara.web.worker_death_watcher._trigger_log_command", return_value=False
            ) as trig,
            patch("kiara.web.worker_death_watcher.time.sleep", return_value=None),
            patch(
                "kiara.web.worker_death_watcher.time.monotonic",
                side_effect=_monotonic_sequence(*[100.0 + i for i in range(50)]),
            ),
        ):
            # max_iterations sehr hoch: wenn das Limit nicht greift, wuerde
            # der Test nach 99 Iterationen nicht mehr returnen — wir wollen
            # explizit pruefen dass _watch_service VOR Limit-Erreichung
            # selbststaendig returnt.
            wdw._watch_service("kiara.service", max_iterations=99)

        # _CONSECUTIVE_FAIL_LIMIT (5) Trigger-Calls bevor return: ein Live-
        # Fail plus Pending-Retries bis zum Limit.
        assert trig.call_count == wdw._CONSECUTIVE_FAIL_LIMIT, (
            f"Erwartet {wdw._CONSECUTIVE_FAIL_LIMIT} Trigger-Aufrufe vor "
            f"Thread-Terminierung, bekommen: {trig.call_count}"
        )


class TestTrackedServices:
    def test_all_kiara_services_listed(self):
        """Watcher kennt alle 5 KIara-Service-Units."""
        expected = {
            "kiara.service",
            "kiara-admin.service",
            "kiara-admin-chat.service",
            "kiara-worker.service",
            "kiara-mcp.service",
        }
        assert set(wdw._TRACKED_SERVICES) == expected
