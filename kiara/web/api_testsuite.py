"""KIara Admin - Selbsttest API (pytest-Suite ausfuehren)"""

import json
import logging
import queue
import re
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter

logger = logging.getLogger(__name__)

testsuite_router = APIRouter(prefix="/api/admin/tests", tags=["testsuite"])

_test_status = {
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
}
_status_lock = threading.Lock()

_PYTEST_BIN = "/opt/kiara/venv/bin/pytest"
_PROJECT_DIR = "/opt/kiara"
_TESTS_DIR = _PROJECT_DIR + "/tests"
_RESULTS_FILE = Path("/var/lib/kiara/selftest_results.json")

# Pattern: "tests/test_foo.py::TestClass::test_bar PASSED     [  7%]"
_RESULT_RE = re.compile(r"^(tests/\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s")
# Pattern: letzte Zeile von --collect-only -q, z.B. "1563 tests collected in 9.61s"
_COLLECT_RE = re.compile(r"(\d+)\s+tests?\s+collected")
_TESTSUITE_RUN_TIMEOUT_SECONDS = 600
_STDOUT_POLL_SECONDS = 0.2


def _save_results():
    """Ergebnis als JSON in _RESULTS_FILE schreiben (ueberschreibt vorheriges)."""
    try:
        _RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        snapshot = {k: v for k, v in _test_status.items() if k != "current_test"}
        snapshot["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _RESULTS_FILE.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning("Selftest-Ergebnis konnte nicht gespeichert werden: %s", exc)


def _load_results():
    """Letztes Ergebnis aus _RESULTS_FILE laden (falls vorhanden)."""
    try:
        if _RESULTS_FILE.is_file():
            data = json.loads(_RESULTS_FILE.read_text())
            for key in (
                "progress",
                "total",
                "passed",
                "failed",
                "errors",
                "skipped",
                "duration",
                "results",
                "error",
            ):
                if key in data:
                    _test_status[key] = data[key]
    except Exception as exc:
        logger.warning("Selftest-Ergebnis konnte nicht geladen werden: %s", exc)


_load_results()


def _reset_status():
    with _status_lock:
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


def _run_tests():
    try:
        # Phase 1: Tests zaehlen
        collect = subprocess.run(
            [_PYTEST_BIN, "--collect-only", "-q", "--continue-on-collection-errors", _TESTS_DIR],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=_PROJECT_DIR,
        )
        for line in collect.stdout.splitlines():
            m = _COLLECT_RE.search(line)
            if m:
                with _status_lock:
                    _test_status["total"] = int(m.group(1))
                break

        # Phase 2: Tests ausfuehren
        start = time.monotonic()
        # KIARA_SKIP_DB_LOG_HANDLER=1 deaktiviert den System-Event-DB-Handler
        # im Test-Prozess. Hintergrund: Tests, die security_log.*-Events
        # erzeugen (z.B. Login-/Admin-Chat-Tests), wuerden sonst synchrone
        # DB-Transactions im Test-Prozess triggern und koennen in Deadlocks
        # laufen, weil init_db_backend() im pytest-Subprozess nicht garantiert
        # initialisiert ist.
        env = {
            **__import__("os").environ,
            "COLUMNS": "500",
            "KIARA_SKIP_DB_LOG_HANDLER": "1",
        }
        proc = subprocess.Popen(
            [
                _PYTEST_BIN,
                _TESTS_DIR,
                "-v",
                "--tb=short",
                "--no-header",
                "--continue-on-collection-errors",
                "--timeout=30",
                "--timeout-method=signal",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=_PROJECT_DIR,
            env=env,
        )

        output_queue: queue.Queue[str | None] = queue.Queue()

        def _read_stdout() -> None:
            try:
                if proc.stdout is None:
                    return
                for raw_line in iter(proc.stdout.readline, ""):
                    output_queue.put(raw_line)
            finally:
                output_queue.put(None)

        stdout_reader = threading.Thread(target=_read_stdout, daemon=True)
        stdout_reader.start()

        tail_buffer = []  # letzte Zeilen fuer Fehlerdiagnose
        in_summary = False  # True wenn in "short test summary info"
        in_failures = False  # True wenn in "FAILURES" / "ERRORS" Block
        current_block_name = ""  # Testname des aktuellen Traceback-Blocks
        current_block_lines = []  # Zeilen des aktuellen Blocks
        traceback_blocks = {}  # test_path -> traceback-Text
        summary_details = {}  # test_path -> Einzeiler aus Summary

        while True:
            if time.monotonic() - start > _TESTSUITE_RUN_TIMEOUT_SECONDS:
                proc.kill()
                proc.wait()
                with _status_lock:
                    _test_status["error"] = (
                        f"Testsuite nach {_TESTSUITE_RUN_TIMEOUT_SECONDS:g}s abgebrochen (Timeout)"
                    )
                    _test_status["duration"] = round(time.monotonic() - start, 1)
                    _test_status["running"] = False
                _save_results()
                return

            try:
                queued_line = output_queue.get(timeout=_STDOUT_POLL_SECONDS)
            except queue.Empty:
                if proc.poll() is not None and not stdout_reader.is_alive():
                    break
                continue

            if queued_line is None:
                break

            line = queued_line.rstrip("\n")
            tail_buffer.append(line)
            if len(tail_buffer) > 30:
                tail_buffer.pop(0)

            # --- Phase-Erkennung ---
            if "= short test summary info =" in line:
                in_summary = True
                # Letzten offenen Traceback-Block abschliessen
                if current_block_name and current_block_lines:
                    traceback_blocks[current_block_name] = "\n".join(
                        current_block_lines,
                    )[:3000]
                in_failures = False
                continue
            if "= FAILURES =" in line or "= ERRORS =" in line:
                in_failures = True
                continue

            # --- Short Test Summary parsen ---
            if in_summary:
                # Format: "FAILED tests/test_foo.py::Class::test - Error msg"
                for prefix in ("FAILED ", "ERROR "):
                    if line.startswith(prefix):
                        rest = line[len(prefix) :]
                        if " - " in rest:
                            tname, detail = rest.split(" - ", 1)
                        else:
                            tname, detail = rest, ""
                        summary_details[tname.strip()] = detail.strip()
                        break
                continue

            # --- Traceback-Bloecke sammeln ---
            if in_failures:
                # Block-Separator: "_____ TestClass.test_name _____"
                if line.startswith("___") and line.endswith("___"):
                    # Vorherigen Block speichern
                    if current_block_name and current_block_lines:
                        traceback_blocks[current_block_name] = "\n".join(
                            current_block_lines,
                        )[:3000]
                    current_block_name = line.strip("_ ").strip()
                    current_block_lines = []
                else:
                    current_block_lines.append(line)
                continue

            # --- Test-Ergebnis-Zeilen parsen ---
            m = _RESULT_RE.match(line)
            if not m:
                continue

            name, status = m.group(1), m.group(2)
            with _status_lock:
                _test_status["current_test"] = name
                _test_status["progress"] += 1

                if status == "PASSED" or status == "XPASS":
                    _test_status["passed"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "passed", "details": ""},
                    )
                elif status == "FAILED":
                    _test_status["failed"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "failed", "details": ""},
                    )
                elif status == "ERROR":
                    _test_status["errors"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "error", "details": ""},
                    )
                elif status == "SKIPPED" or status == "XFAIL":
                    _test_status["skipped"] += 1
                    _test_status["results"].append(
                        {"name": name, "status": "skipped", "details": ""},
                    )

        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            with _status_lock:
                _test_status["error"] = (
                    f"Testsuite nach {_TESTSUITE_RUN_TIMEOUT_SECONDS:g}s abgebrochen (Timeout)"
                )
                _test_status["duration"] = round(time.monotonic() - start, 1)
                _test_status["running"] = False
            _save_results()
            return

        # Details den Ergebnissen zuordnen
        with _status_lock:
            _assign_failure_details(traceback_blocks, summary_details)

            # Pytest-Abbruch erkennen: returncode != 0 und kein Test ausgefuehrt
            if rc != 0 and _test_status["progress"] == 0:
                error_lines = [
                    ln
                    for ln in tail_buffer
                    if ln.strip() and not ln.startswith("=") and "warning" not in ln.lower()
                ]
                detail = "\n".join(error_lines[-10:]) if error_lines else ""
                _test_status["error"] = (
                    f"pytest abgebrochen (Exit-Code {rc}), kein Test ausgefuehrt.\n{detail}"
                ).strip()

            _test_status["duration"] = round(time.monotonic() - start, 1)

    except FileNotFoundError:
        with _status_lock:
            _test_status["error"] = f"pytest nicht gefunden: {_PYTEST_BIN}"
    except subprocess.TimeoutExpired:
        with _status_lock:
            _test_status["error"] = "Testsammlung Timeout (>60s)"
    except Exception as exc:
        with _status_lock:
            _test_status["error"] = str(exc)
    finally:
        with _status_lock:
            _test_status["running"] = False
            _test_status["current_test"] = ""
        _save_results()


def _assign_failure_details(traceback_blocks: dict, summary_details: dict):
    """Ordnet gesammelte Traceback-Bloecke und Summary-Zeilen den Ergebnissen zu."""
    failed_results = [
        r
        for r in _test_status["results"]
        if r["status"] in ("failed", "error") and not r["details"]
    ]
    # Block-Keys nutzen "." zwischen Klasse und Methode,
    # Ergebnis-Namen nutzen "::" — normalisieren fuer Matching
    norm_blocks = {}
    for bk, bv in traceback_blocks.items():
        norm_blocks[bk.replace(".", "::")] = bv

    for result in failed_results:
        name = result["name"]  # "tests/test_foo.py::Class::test"
        short = name.replace("tests/", "")  # "test_foo.py::Class::test"
        # Nur Class::method fuer Block-Matching
        parts = short.split("::", 1)
        suffix = parts[1] if len(parts) > 1 else short

        # 1. Versuch: Traceback-Block (detailliert)
        for block_key, block_text in norm_blocks.items():
            if suffix in block_key or block_key in suffix:
                result["details"] = block_text
                break

        # 2. Fallback: Summary-Einzeiler
        if not result["details"]:
            detail = summary_details.get(name, "")
            if not detail:
                for skey, sval in summary_details.items():
                    if short in skey or skey.endswith(short):
                        detail = sval
                        break
            result["details"] = detail


# --- Migriert zu Admin-Registry: testsuite.run (Sprint 09) ---
# @testsuite_router.post("/run")
# async def run_tests(user=Depends(get_admin_user)):
#     if _test_status["running"]:
#         raise HTTPException(status_code=409, detail="Selbsttest laeuft bereits")
#
#     _reset_status()
#     threading.Thread(target=_run_tests, daemon=True).start()
#     return {"message": "Selbsttest gestartet"}


# --- Migriert zu Admin-Registry: testsuite.status (Sprint 09) ---
# @testsuite_router.get("/status")
# async def test_status(user=Depends(get_admin_user)):
#     return dict(_test_status)
