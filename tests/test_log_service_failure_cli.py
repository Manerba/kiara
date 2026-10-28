"""Tests fuer den `kiara log-service-failure`-CLI-Befehl.

Der Befehl wird von systemd via `OnFailure=kiara-failure-logger@%n.service`
getriggert, sammelt Diagnose-Daten (`systemctl show`, `journalctl`, `dmesg`)
und persistiert ein `SERVICE_DIED`-Event ueber den security-Logger.

Tests verifizieren ohne echte Subprocess-Aufrufe oder DB-Zugriffe:
- Die Log-Message hat das vereinbarte Format (event_type + KV-Felder).
- `extra={"kiara_details": {...}}` enthaelt die strukturierten Multi-Line-Felder.
- Die OOM-Heuristik schaltet `reason` korrekt um.
- Service-Name-Validation lehnt ungueltige/gefaehrliche Inputs ab.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner


def _setup_dispatch(
    systemctl_show: str = "Result=signal\nExecMainStatus=11\nExecMainCode=killed\nMainPID=4242",
    journal_tail: str = "Apr 30 12:00 kiara[4242]: panic",
    dmesg_tail: str = "",
):
    """Erzeugt einen `subprocess.run`-Side-Effect, der je nach argv mockt.

    Liefert einen FakeRun, der `systemctl show`, `journalctl` und `dmesg --ctime`
    nach Argv unterscheidet — alle anderen Aufrufe geben leeren Output zurueck.
    """

    def fake_run(args, **kwargs):
        if not args:
            return SimpleNamespace(stdout="", returncode=0)
        if args[0] == "systemctl" and "show" in args:
            return SimpleNamespace(stdout=systemctl_show, returncode=0)
        if args[0] == "journalctl":
            return SimpleNamespace(stdout=journal_tail, returncode=0)
        if args[0] == "dmesg":
            return SimpleNamespace(stdout=dmesg_tail, returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    return fake_run


def _invoke_cli(
    *cli_args,
    dmesg_path="/usr/bin/dmesg",
    log_error_side_effect=None,
    setup_side_effect=None,
    **dispatch_kwargs,
):
    """Ruft `kiara log-service-failure` mit gemockten externen Aufrufen.

    Liefert (CliResult, security_log_mock) zurueck. Patches:
      - subprocess.run via _setup_dispatch
      - shutil.which → simuliert dmesg-Verfuegbarkeit
      - init_db_backend → no-op (DB darf nicht angefasst werden)
      - setup_security_logger → liefert MagicMock (so faengt der Test alle .error-Calls)
    """
    from kiara.cli import main

    fake_run = _setup_dispatch(**dispatch_kwargs)
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
        patch("shutil.which", return_value=dmesg_path),
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
        result = runner.invoke(main, ["log-service-failure", *cli_args])
    if result.exit_code != 2:
        setup_logger.assert_called_once_with(strict_db=True)
    return result, security_log


# ---------------------------------------------------------------------------
# Schema-Verify
# ---------------------------------------------------------------------------


def test_logs_service_died_with_expected_format_and_kiara_details():
    """SERVICE_DIED-Format + kiara_details-Keys vorhanden."""
    result, security_log = _invoke_cli(
        "kiara.service",
        systemctl_show="Result=signal\nExecMainStatus=11\nMainPID=4242",
        journal_tail="2026-04-30 12:00 panic full trace",
        dmesg_tail="",
    )

    assert result.exit_code == 0, result.output
    security_log.error.assert_called_once()
    args, kwargs = security_log.error.call_args
    fmt = args[0]
    assert fmt == "SERVICE_DIED service=%s reason=%s exit_code=%s pid=%s"
    assert args[1:] == ("kiara.service", "signal", "11", "4242")

    extra = kwargs["extra"]["kiara_details"]
    assert "journal_tail" in extra
    assert "dmesg_oom" in extra
    assert "panic full trace" in extra["journal_tail"]


def test_journal_tail_is_truncated_to_4000_chars():
    """journal_tail-Tail wird auf 4000 Zeichen begrenzt."""
    huge = "x" * 9000
    _, security_log = _invoke_cli(
        "kiara.service",
        journal_tail=huge,
    )
    extra = security_log.error.call_args.kwargs["extra"]["kiara_details"]
    assert len(extra["journal_tail"]) == 4000


# ---------------------------------------------------------------------------
# OOM-Heuristik
# ---------------------------------------------------------------------------


def test_oom_heuristic_sets_reason_to_oom_killed_when_dmesg_mentions_service():
    """Wenn dmesg den Service-Namen oder die PID nennt: reason=oom_killed."""
    dmesg = "Apr 30 12:00:01 host kernel: Out of memory: Killed process 4242 (kiara) total-vm:1.2G"
    _, security_log = _invoke_cli(
        "kiara.service",
        systemctl_show="Result=signal\nExecMainStatus=137\nMainPID=4242",
        dmesg_tail=dmesg,
    )
    args, kwargs = security_log.error.call_args
    assert args[2] == "oom_killed"
    assert "Killed process 4242" in kwargs["extra"]["kiara_details"]["dmesg_oom"]


def test_oom_heuristic_falls_back_to_systemctl_result_without_oom_match():
    """Ohne OOM-Hinweis: reason = systemctl Result-Wert."""
    _, security_log = _invoke_cli(
        "kiara.service",
        systemctl_show="Result=core-dump\nExecMainStatus=139\nMainPID=4242",
        dmesg_tail="",  # kein dmesg-Hinweis
    )
    args, kwargs = security_log.error.call_args
    assert args[2] == "core-dump"
    # dmesg_oom ist None (leer-string filter -> None) im extra-dict
    assert kwargs["extra"]["kiara_details"]["dmesg_oom"] is None


def test_oom_heuristic_ignores_unrelated_dmesg_lines():
    """OOM-Lines in dmesg, die NICHT auf den Service zeigen, kippen reason nicht."""
    dmesg = "Apr 30 12:00 host kernel: Killed process 9999 (some-other)"
    _, security_log = _invoke_cli(
        "kiara.service",
        systemctl_show="Result=success\nExecMainStatus=0\nMainPID=4242",
        dmesg_tail=dmesg,
    )
    args, _ = security_log.error.call_args
    # 9999 ist nicht 4242 und der Service-Name (kiara) kommt im dmesg-Text vor
    # → bei diesem Test ist "kiara" nicht im dmesg-text; sonst wuerde es matchen.
    # Daher: reason bleibt der systemctl-Wert.
    assert args[2] == "success"


def test_db_logging_failure_returns_nonzero():
    """Strict-DB-Pfad: kein rc=0 wenn system_events nicht geschrieben wurde."""
    result, security_log = _invoke_cli(
        "kiara.service",
        log_error_side_effect=RuntimeError("db down"),
    )

    assert result.exit_code == 1
    assert "SERVICE_DIED konnte nicht persistiert werden" in result.output
    security_log.error.assert_called_once()


def test_logger_setup_failure_returns_nonzero():
    """Strict-DB-Pfad: kein rc=0 wenn der DB-Handler nicht initialisiert."""
    result, security_log = _invoke_cli(
        "kiara.service",
        setup_side_effect=RuntimeError("handler missing"),
    )

    assert result.exit_code == 1
    assert "SERVICE_DIED-Logger konnte nicht initialisiert werden" in result.output
    security_log.error.assert_not_called()


# ---------------------------------------------------------------------------
# Input-Validation
# ---------------------------------------------------------------------------


def test_rejects_service_name_without_dot_service_suffix():
    """Service ohne '.service'-Suffix → exit-code 2."""
    result, security_log = _invoke_cli("kiara")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_rejects_service_name_with_metacharacter():
    """Service-Name mit Sonderzeichen ('a;b.service') → exit-code 2."""
    result, security_log = _invoke_cli("a;b.service")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_rejects_empty_service_name():
    """Leerer Service-Name → exit-code 2 (Click sieht keinen Wert ohne argument)."""
    # Ein leerer String als Argument muss explizit uebergeben werden, sonst
    # wuerde click ueber missing-argument abbrechen (exit-code 2 sowieso).
    result, security_log = _invoke_cli("")
    assert result.exit_code == 2
    security_log.error.assert_not_called()


def test_strips_carriage_return_and_newline_from_service_name():
    """\\r und \\n im Service-Namen werden vor der Validierung entfernt.

    Trotzdem muss .service-Suffix erhalten bleiben — der safe_name prueft
    danach mit isalnum() und akzeptiert.
    """
    result, security_log = _invoke_cli("kiara\r.service")
    # 'kiara.service' ohne carriage-return ist ein valider Name → exit 0
    assert result.exit_code == 0, result.output
    security_log.error.assert_called_once()
    assert security_log.error.call_args.args[1] == "kiara.service"
