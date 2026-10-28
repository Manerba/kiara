"""Faulthandler-Aktivierung fuer KIara-Service-Workers.

Bei fatalen Signalen (SIGSEGV, SIGFPE, SIGABRT, SIGBUS, SIGILL) dumpt der
Python-Faulthandler einen vollstaendigen Stack-Trace aller Threads nach
stderr. Bei uvicorn-Sub-Workers landet stderr in journald unter der
Service-Unit — der kiara-worker-death-watcher.service kann den Trace dann
in den WORKER_DIED-Event-Kontext aufnehmen.

Ohne diesen Hook geht der Stack-Trace eines abrupt sterbenden Workers
verloren (RotatingFileHandler-Buffer noch nicht geflusht, kein Python-
Logger mehr aktiv). Mit dem Hook bleibt mindestens der C-Frame plus die
Python-Frames erhalten, was die Diagnose-Luecke zwischen "Worker tot" und
"Crash-Ursache" deutlich verkleinert.

SIGUSR1 wird als On-Demand-Trigger registriert: `kill -USR1 <pid>` dumpt
den Stack ohne den Worker zu beenden — nuetzlich fuer Live-Debugging
hangender Worker (z.B. CIFS-D-State).
"""

from __future__ import annotations

import faulthandler
import signal
import sys


def enable_faulthandler() -> None:
    """Aktiviert faulthandler fail-safe (idempotent).

    Doppelaufruf ist OK — faulthandler.enable() ist idempotent, das
    register-Pattern ueberschreibt vorherige Handler.
    """
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except (RuntimeError, AttributeError, ValueError):
        return

    if hasattr(signal, "SIGUSR1"):
        try:
            faulthandler.register(
                signal.SIGUSR1,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
        except (RuntimeError, AttributeError, ValueError):
            pass
