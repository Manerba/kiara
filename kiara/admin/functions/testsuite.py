"""Admin-Funktionen fuer die Selbsttest-Suite."""

import logging

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult

logger = logging.getLogger(__name__)


@admin_function(
    name="testsuite.run",
    category="testsuite",
    description="Selbsttest-Suite starten",
    help="Startet die pytest-Testsuite im Hintergrund. "
    "Status und Ergebnisse mit testsuite.status abfragen.",
    examples=[
        "Starte Selbsttest",
        "Tests ausführen",
        "Testsuite starten",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Bestätigung dass Tests gestartet wurden",
    raw_response=True,
    http=("POST", "/api/admin/tests/run"),
)
async def testsuite_run(user: dict) -> AdminResult:
    import threading

    from kiara.web.api_testsuite import _run_tests, _status_lock, _test_status

    with _status_lock:
        if _test_status["running"]:
            return AdminResult.fail("Selbsttest laeuft bereits")
        _test_status.update(
            {
                "running": True,
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
            }
        )

    threading.Thread(target=_run_tests, daemon=True).start()
    return AdminResult.ok(
        "Selbsttest gestartet",
        data={"message": "Selbsttest gestartet"},
    )


@admin_function(
    name="testsuite.status",
    category="testsuite",
    description="Test-Status und Ergebnisse abfragen",
    help="Zeigt den aktuellen Status der Testsuite inklusive Fortschritt, "
    "bestandene/fehlgeschlagene Tests und Detail-Ergebnisse.",
    examples=[
        "Wie ist der Test-Status?",
        "Sind die Tests fertig?",
        "Test-Ergebnisse anzeigen",
    ],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Test-Status mit Fortschritt und Ergebnissen",
    raw_response=True,
    http=("GET", "/api/admin/tests/status"),
)
async def testsuite_status(user: dict) -> AdminResult:
    from kiara.web.api_testsuite import _status_lock, _test_status

    with _status_lock:
        status = dict(_test_status)
        status["results"] = [dict(r) for r in status.get("results", [])]
    if status["running"]:
        msg = f"Tests laufen: {status['progress']}/{status['total']}"
    elif status["error"]:
        msg = f"Tests fehlgeschlagen: {status['error']}"
    else:
        msg = (
            f"Tests abgeschlossen: {status['passed']} bestanden, {status['failed']} fehlgeschlagen"
        )

    return AdminResult.ok(msg, data=status)
