"""Tests fuer den `kiara log-worker-death`-CLI-Befehl.

Der Befehl wird vom `kiara-worker-death-watcher.service` aufgerufen, sobald
der uvicorn-Master "Child process [PID] died" loggt. Er sammelt einen
Diagnose-Snapshot (dmesg-Tail, CIFS-Mount-Status, cgroup-Memory, journal)
und persistiert ein `WORKER_DIED`-Event ueber den security-Logger.

Tests verifizieren ohne echte Subprocess-/Filesystem-Zugriffe:
- Log-Format und kiara_details-Schluesselsatz.
- CIFS-Heuristik (`tcon failed` -> `cifs_unreachable_correlation`).
- cgroup-Reads sind fail-safe (`FileNotFoundError`-Fall).
- Input-Validation lehnt ungueltige PIDs/Service-Namen ab.
"""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Helper: Subprocess-Dispatcher + open()-Mock (cgroup-Files)
# ---------------------------------------------------------------------------


def _subprocess_dispatch(dmesg_tail: str = "", mount_cifs: str = "", journal_tail: str = ""):
    def fake_run(args, **kwargs):
        if not args:
            return SimpleNamespace(stdout="", returncode=0)
        if args[0] == "dmesg":
            return SimpleNamespace(stdout=dmesg_tail, returncode=0)
        if args[0] == "mount":
            return SimpleNamespace(stdout=mount_cifs, returncode=0)
        if args[0] == "journalctl":
            return SimpleNamespace(stdout=journal_tail, returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    return fake_run


def _cgroup_open_factory(
    memory_current="123456789", memory_peak="987654321", oom_kill_line="oom_kill 0\n", missing=False
):
    """Liefert `open()`-Side-Effect, der cgroup-Pfade je nach Suffix mockt.

    Wenn `missing=True`, wirft jeder Open-Aufruf auf cgroup-Pfade
    `FileNotFoundError`. Andere Pfade (sollte es keine geben) raisen ebenfalls.
    """
    builtin_open = open  # echter open() fuer Nicht-cgroup-Pfade

    def _open(path, *a, **kw):
        path_str = str(path)
        if "/cgroup/" in path_str:
            if missing:
                raise FileNotFoundError(path_str)
            if path_str.endswith("memory.current"):
                return StringIO(memory_current)
            if path_str.endswith("memory.peak"):
                return StringIO(memory_peak)
            if path_str.endswith("memory.events"):
                return StringIO(oom_kill_line)
            raise FileNotFoundError(path_str)
        return builtin_open(path, *a, **kw)

    return _open


def _invoke(
    *cli_args,
    dispatch=None,
    open_factory=None,
    log_error_side_effect=None,
    setup_side_effect=None,
):
    from kiara.cli import main

    fake_run = dispatch if dispatch else _subprocess_dispatch()
    open_side = open_factory if open_factory else _cgroup_open_factory()
    security_log = MagicMock()
    if log_error_side_effect is not None:
        security_log.error.side_effect = log_error_side_effect
    if setup_side_effect is not None:
        setup_logger_patch = patch(
            "kiara.web.logging_config.setup_security_logger",
            side_effect=setup_side_effect,
        )
    else:
        setup_logger_patch = patch(
            "kiara.web.logging_config.setup_security_logger",
            return_value=security_log,
        )

    with (
        patch("subprocess.run", side_effect=fake_run),
        patch("builtins.open", side_effect=open_side),
        patch("kiara.shared.db.init_db_backend"),
        setup_logger_patch as setup_logger,
        patch(
            "kiara.config.load_config",
            return_value={
                "datasource": {"path": "/tmp"},
                "web": {"host": "0.0.0.0"},
            },
        ),
    ):
        runner = CliRunner()
        result = runner.invoke(main, ["log-worker-death", *cli_args])
    if result.exit_code != 2:
        setup_logger.assert_called_once_with(strict_db=True)
    return result, security_log


# ---------------------------------------------------------------------------
# Schema-Verify
# ---------------------------------------------------------------------------


def test_logs_worker_died_with_expected_format_and_kiara_details():
    """WORKER_DIED-Format + alle erwarteten kiara_details-Keys."""
    result, security_log = _invoke(
        "12345",
        "kiara.service",
        dispatch=_subprocess_dispatch(
            dmesg_tail="Apr 30 12:00 kernel: just stuff",
            mount_cifs="//host/share on /mnt/x type cifs (rw)",
            journal_tail="some journal output",
        ),
    )

    assert result.exit_code == 0, result.output
    security_log.error.assert_called_once()
    args, kwargs = security_log.error.call_args
    assert args[0] == "WORKER_DIED service=%s pid=%s reason=%s"
    assert args[1:] == ("kiara.service", "12345", "unknown")

    extra = kwargs["extra"]["kiara_details"]
    expected_keys = {
        "service",
        "dmesg_tail",
        "cifs_mounts",
        "memory_current",
        "memory_peak",
        "cgroup_oom_kill",
        "journal_tail",
    }
    assert expected_keys.issubset(extra.keys())
    assert extra["service"] == "kiara.service"
    assert extra["memory_current"] == "123456789"
    assert extra["memory_peak"] == "987654321"
    assert extra["cgroup_oom_kill"] == "0"
    assert "//host/share" in extra["cifs_mounts"]


def test_journal_tail_is_truncated_to_3000_chars():
    """journal_tail wird auf 3000 Zeichen geschnitten."""
    long_journal = "j" * 7000
    _, security_log = _invoke(
        "1234",
        "kiara.service",
        dispatch=_subprocess_dispatch(journal_tail=long_journal),
    )
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    assert len(extra["journal_tail"]) == 3000


# ---------------------------------------------------------------------------
# CIFS-Heuristik
# ---------------------------------------------------------------------------


def test_cifs_heuristic_sets_reason_to_cifs_unreachable_correlation():
    """`tcon failed` in dmesg → reason=cifs_unreachable_correlation."""
    cifs_dmesg = (
        "Apr 30 12:00 kernel: CIFS: VFS: \\\\srv\\share Send error in "
        "SessSetup = -126\nApr 30 12:00 kernel: CIFS: VFS: reconnect tcon "
        "failed rc = -2"
    )
    _, security_log = _invoke(
        "999",
        "kiara-worker.service",
        dispatch=_subprocess_dispatch(dmesg_tail=cifs_dmesg),
    )
    args, kwargs = security_log.error.call_args
    assert args[3] == "cifs_unreachable_correlation"
    assert "tcon failed" in kwargs["extra"]["kiara_details"]["dmesg_tail"]


def test_unknown_reason_when_no_cifs_indicator_in_dmesg():
    """Ohne `tcon failed` → reason=unknown."""
    _, security_log = _invoke(
        "999",
        "kiara.service",
        dispatch=_subprocess_dispatch(dmesg_tail="Apr 30 nothing special"),
    )
    assert security_log.error.call_args.args[3] == "unknown"


# ---------------------------------------------------------------------------
# cgroup-Reads (FileNotFoundError fail-safe)
# ---------------------------------------------------------------------------


def test_cgroup_reads_are_fail_safe_when_files_missing():
    """Wenn cgroup-Files fehlen: keine Exception, Felder bleiben leer/None."""
    result, security_log = _invoke(
        "1234",
        "kiara.service",
        open_factory=_cgroup_open_factory(missing=True),
    )

    assert result.exit_code == 0, result.output
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    # Filter im CLI: empty-string -> None im extra-dict.
    assert extra["memory_current"] is None
    assert extra["memory_peak"] is None
    assert extra["cgroup_oom_kill"] is None


def test_cgroup_oom_kill_line_extracted_when_present():
    """memory.events: 'oom_kill 5' → cgroup_oom_kill='5'."""
    _, security_log = _invoke(
        "1234",
        "kiara.service",
        open_factory=_cgroup_open_factory(oom_kill_line="low 0\noom_kill 5\n"),
    )
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    assert extra["cgroup_oom_kill"] == "5"


def test_db_logging_failure_returns_nonzero():
    """Strict-DB-Pfad: kein rc=0 wenn system_events nicht geschrieben wurde."""
    result, security_log = _invoke(
        "1234",
        "kiara.service",
        log_error_side_effect=RuntimeError("db down"),
    )

    assert result.exit_code == 1
    assert "WORKER_DIED konnte nicht persistiert werden" in result.output
    security_log.error.assert_called_once()


def test_logger_setup_failure_returns_nonzero():
    """Strict-DB-Pfad: kein rc=0 wenn der DB-Handler nicht initialisiert."""
    result, security_log = _invoke(
        "1234",
        "kiara.service",
        setup_side_effect=RuntimeError("handler missing"),
    )

    assert result.exit_code == 1
    assert "WORKER_DIED-Logger konnte nicht initialisiert werden" in result.output
    security_log.error.assert_not_called()


# ---------------------------------------------------------------------------
# Input-Validation
# ---------------------------------------------------------------------------


def test_rejects_non_numeric_pid():
    """Nicht-numerische PID → exit-code 2."""
    result, security_log = _invoke("abc", "kiara.service")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_rejects_pid_longer_than_12_chars():
    """PID > 12 Zeichen → exit-code 2 (Schutz vor PID-Spoof-Spam)."""
    result, security_log = _invoke("1234567890123", "kiara.service")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_rejects_service_without_dot_service_suffix():
    """Service ohne '.service'-Suffix → exit-code 2."""
    result, security_log = _invoke("1234", "kiara")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_rejects_service_with_metacharacter():
    """Service-Name mit Sonderzeichen → exit-code 2."""
    result, security_log = _invoke("1234", "ki;rm-rf.service")
    assert result.exit_code == 2
    security_log.error.assert_not_called()
