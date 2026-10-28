"""Tests fuer `kiara.web.faulthandler_setup.enable_faulthandler`.

Der Setup aktiviert Pythons faulthandler (Stack-Dumps bei SIGSEGV/SIGABRT
nach stderr) und registriert SIGUSR1 als On-Demand-Trigger fuer Live-
Debugging hangender Worker.

Wichtige Garantien:
- Idempotent (Doppelaufruf ist sicher).
- Fail-safe: Faulthandler-Errors duerfen den Worker nicht killen.
- SIGUSR1 wird auf Linux registriert; auf Plattformen ohne SIGUSR1
  (Windows in der Theorie) wird der Aufruf uebersprungen.
"""

import signal
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_double_invocation_is_idempotent():
    """enable_faulthandler() darf mehrfach aufgerufen werden, ohne zu crashen.

    `faulthandler.enable` selbst ist idempotent (einmal an = an), und das
    `register`-Pattern ueberschreibt den vorherigen Handler. Der Test
    verifiziert: zweiter Aufruf wirft nicht und ruft enable+register erneut.
    """
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.register = MagicMock()
        fh.enable = MagicMock()
        enable_faulthandler()
        enable_faulthandler()
        assert fh.enable.call_count == 2
        # SIGUSR1 ist auf Linux verfuegbar -> register wurde 2x aufgerufen
        if hasattr(signal, "SIGUSR1"):
            assert fh.register.call_count == 2


def test_swallows_runtime_error_from_faulthandler_enable():
    """Wenn faulthandler.enable() RuntimeError wirft, kein Re-raise."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable.side_effect = RuntimeError("kaputt")
        # Kein Re-raise erwartet
        enable_faulthandler()
        # Wenn enable failt, soll register nicht aufgerufen werden (early return)
        fh.register.assert_not_called()


def test_swallows_attribute_error_from_faulthandler_enable():
    """AttributeError beim enable() wird geschluckt (z.B. unsupported sys.stderr)."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable.side_effect = AttributeError("no fileno")
        enable_faulthandler()
        fh.register.assert_not_called()


def test_swallows_value_error_from_faulthandler_enable():
    """ValueError beim enable() wird geschluckt."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable.side_effect = ValueError("invalid stream")
        enable_faulthandler()
        fh.register.assert_not_called()


@pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="SIGUSR1 nicht auf dieser Plattform verfuegbar"
)
def test_sigusr1_is_registered_on_linux():
    """Auf Linux wird faulthandler.register fuer SIGUSR1 aufgerufen."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable = MagicMock()
        fh.register = MagicMock()
        enable_faulthandler()

    fh.register.assert_called_once()
    args, kwargs = fh.register.call_args
    # Erstes Positional-Arg muss SIGUSR1 sein
    assert args[0] == signal.SIGUSR1
    # all_threads + chain=False (Doku in der Modul-Docstring)
    assert kwargs.get("all_threads") is True
    assert kwargs.get("chain") is False


def test_sigusr1_register_failure_is_swallowed():
    """register() darf RuntimeError werfen ohne den Setup zu killen."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable = MagicMock()
        fh.register.side_effect = RuntimeError("signal already in use")
        # Kein Re-raise
        enable_faulthandler()


def test_register_uses_stderr_stream():
    """Register-Call muss `file=sys.stderr` verwenden — Doku-Garantie."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 nicht verfuegbar")

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable = MagicMock()
        fh.register = MagicMock()
        enable_faulthandler()

    kwargs = fh.register.call_args.kwargs
    assert kwargs.get("file") is sys.stderr


def test_enable_uses_stderr_and_all_threads():
    """faulthandler.enable() muss mit file=stderr + all_threads=True aufgerufen werden."""
    from kiara.web.faulthandler_setup import enable_faulthandler

    with patch("kiara.web.faulthandler_setup.faulthandler") as fh:
        fh.enable = MagicMock()
        fh.register = MagicMock()
        enable_faulthandler()

    kwargs = fh.enable.call_args.kwargs
    assert kwargs.get("file") is sys.stderr
    assert kwargs.get("all_threads") is True
