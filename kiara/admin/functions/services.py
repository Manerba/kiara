"""Service-Management-Funktionen fuer den Admin-Chat.

Sprint 07: status, list, logs, restart fuer KIara systemd-Services.
Nur Services aus ALLOWED_SERVICES koennen gesteuert werden (Whitelist).
"""

import logging
import re
import subprocess

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.config import DEFAULT_TIMEOUTS

logger = logging.getLogger(__name__)

_OUTPUT_LIMIT = 10_000
_JOURNAL_ABSOLUTE_SINCE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?$")
_JOURNAL_RELATIVE_SINCE_RE = re.compile(
    r"^\d{1,4}\s+"
    r"(?:second|seconds|minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)"
    r"\s+ago$",
    re.IGNORECASE,
)
_JOURNAL_NAMED_SINCE = {"now", "today", "yesterday"}

ALLOWED_SERVICES = [
    "kiara.service",
    "kiara-admin.service",
    "kiara-worker.service",
    "kiara-mcp.service",
    "kiara-admin-chat.service",
]


def _validate_service(service: str) -> tuple[str, str | None]:
    """Validiert und normalisiert Service-Name gegen Whitelist.

    Returns:
        Tuple (normalized_name, error). Bei Erfolg: ("kiara.service", None).
        Bei Fehler: (original_input, "Service '...' ist nicht erlaubt...").
    """
    import re as _re

    if not _re.match(r"^[a-zA-Z0-9_.\-]+$", service):
        return service, "Service-Name enthaelt unerlaubte Zeichen"
    if not service.endswith(".service"):
        service = f"{service}.service"
    if service not in ALLOWED_SERVICES:
        return (
            service,
            f"Service '{service}' ist nicht erlaubt. Erlaubt: {', '.join(ALLOWED_SERVICES)}",
        )
    return service, None


def _validate_journal_since(since: str) -> bool:
    """Erlaubt nur explizit unterstuetzte journalctl --since-Formate."""
    value = since.strip()
    if not value:
        return True
    return (
        value.lower() in _JOURNAL_NAMED_SINCE
        or _JOURNAL_ABSOLUTE_SINCE_RE.match(value) is not None
        or _JOURNAL_RELATIVE_SINCE_RE.match(value) is not None
    )


# ---------------------------------------------------------------------------
# services.status
# ---------------------------------------------------------------------------


@admin_function(
    name="services.status",
    category="services",
    description="Zeigt den Status eines systemd-Service",
    help="Prueft ob ein KIara-Service laeuft und zeigt Details wie PID, Uptime und Memory.",
    examples=["Status kiara-worker", "Laeuft der Admin-Service?"],
    parameters={
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "Service-Name (z.B. 'kiara-worker' oder 'kiara-worker.service')",
                "minLength": 1,
            },
        },
        "required": ["service"],
    },
    returns="Service-Status (active/inactive/failed), PID, Uptime",
)
async def services_status(user: dict, service: str) -> AdminResult:
    service, error = _validate_service(service)
    if error:
        return AdminResult.fail(error)

    try:
        result = subprocess.run(
            ["systemctl", "--no-pager", "status", service],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_quick"],
        )
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"Timeout bei Status-Abfrage fuer {service}")

    output = result.stdout[:_OUTPUT_LIMIT]

    # Status parsen
    import re

    status = "unknown"
    active_match = re.search(r"Active:\s+(\S+)", output)
    if active_match:
        status = active_match.group(1)

    pid = None
    pid_match = re.search(r"Main PID:\s+(\d+)", output)
    if pid_match:
        pid = int(pid_match.group(1))

    memory = None
    mem_match = re.search(r"Memory:\s+(\S+)", output)
    if mem_match:
        memory = mem_match.group(1)

    uptime = None
    since_match = re.search(r"Active:.*?since\s+(.+?);\s+(.+?)$", output, re.MULTILINE)
    if since_match:
        uptime = since_match.group(2).strip()

    data = {"service": service, "status": status}
    if pid:
        data["pid"] = pid
    if uptime:
        data["uptime"] = uptime
    if memory:
        data["memory"] = memory

    msg = f"{service}: {status}"
    if uptime:
        msg += f" (seit {uptime})"
    return AdminResult.ok(msg, data)


# ---------------------------------------------------------------------------
# services.list
# ---------------------------------------------------------------------------


@admin_function(
    name="services.list",
    category="services",
    description="Zeigt den Status aller KIara-Services",
    help="Prueft alle erlaubten Services und zeigt eine Uebersicht.",
    examples=["Status aller Services", "Welche Services laufen?"],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Liste aller KIara-Services mit Status",
    http=("GET", "/api/monitoring/services"),
)
async def services_list(user: dict, **kwargs) -> AdminResult:
    services_result = []
    for svc in ALLOWED_SERVICES:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", svc],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUTS["subprocess_quick"],
            )
            status = result.stdout.strip()
        except subprocess.TimeoutExpired:
            status = "timeout"
        services_result.append({"name": svc, "status": status})

    active = sum(1 for s in services_result if s["status"] == "active")
    msg = f"{active}/{len(services_result)} Services aktiv"
    return AdminResult.ok(msg, {"services": services_result})


# ---------------------------------------------------------------------------
# services.logs
# ---------------------------------------------------------------------------


@admin_function(
    name="services.logs",
    category="services",
    description="Liest die Logs eines KIara-Service",
    help="Zeigt die letzten Log-Zeilen eines Service via journalctl.",
    examples=["Logs von kiara-worker", "Letzte 100 Zeilen kiara.service"],
    parameters={
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Service-Name", "minLength": 1},
            "lines": {
                "type": "integer",
                "description": "Anzahl Zeilen",
                "default": 50,
                "minimum": 1,
                "maximum": 500,
            },
            "since": {
                "type": "string",
                "description": "Zeitfilter, z.B. '1 hour ago' oder '2026-03-27 10:00:00'",
            },
        },
        "required": ["service"],
    },
    returns="Log-Ausgabe als Text",
)
async def services_logs(user: dict, service: str, lines: int = 50, since: str = "") -> AdminResult:
    service, error = _validate_service(service)
    if error:
        return AdminResult.fail(error)

    # Argument-Reihenfolge MUSS mit sudoers uebereinstimmen:
    # sudo journalctl --no-pager --output=short-iso -u {service} [-n {lines}] [--since ...]
    # sudoers: /usr/bin/journalctl --no-pager --output=short-iso -u {service} *
    cmd = [
        "sudo",
        "journalctl",
        "--no-pager",
        "--output=short-iso",
        "-u",
        service,
        "-n",
        str(lines),
    ]
    if since:
        since = since.strip()
    if since:
        if not _validate_journal_since(since):
            return AdminResult.fail("Ungueltiger --since Zeitfilter")
        cmd.extend(["--since", since])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_service_logs"],
        )
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"Timeout bei Log-Abfrage fuer {service}")

    output = result.stdout[:_OUTPUT_LIMIT]
    if result.returncode != 0 and not output:
        stderr = result.stderr.strip()[:500]
        # Fallback: Wenn sudoers die Argumente nicht erlaubt, ohne -n/-since versuchen
        if "not allowed" in stderr.lower() or "a]password is required" in stderr.lower():
            logger.warning("sudo journalctl mit -n/%s fehlgeschlagen, Fallback ohne -n", since)
            fallback_cmd = [
                "sudo",
                "journalctl",
                "--no-pager",
                "--output=short-iso",
                "-u",
                service,
            ]
            try:
                result = subprocess.run(
                    fallback_cmd,
                    capture_output=True,
                    text=True,
                    timeout=DEFAULT_TIMEOUTS["subprocess_service_logs"],
                )
                output = result.stdout[:_OUTPUT_LIMIT]
            except subprocess.TimeoutExpired:
                return AdminResult.fail(f"Timeout bei Log-Abfrage fuer {service}")
        if result.returncode != 0 and not output:
            return AdminResult.fail(f"Fehler bei Log-Abfrage: {stderr}")

    msg = f"Logs fuer {service} ({lines} Zeilen)"
    return AdminResult.ok(msg, {"service": service, "lines": lines, "output": output})


# ---------------------------------------------------------------------------
# services.restart
# ---------------------------------------------------------------------------


@admin_function(
    name="services.restart",
    category="services",
    description="Startet einen KIara-Service neu",
    help="Fuehrt systemctl restart aus. Erfordert Bestaetigung vor der Ausfuehrung.",
    examples=["Starte kiara.service neu", "Restart kiara-worker"],
    parameters={
        "type": "object",
        "properties": {
            "service": {"type": "string", "description": "Service-Name", "minLength": 1},
        },
        "required": ["service"],
    },
    returns="Bestaetigung des Neustarts",
    requires_confirmation=True,
    confirm_template="service {service} neustarten",
    changelog_area="services",
    changelog_action="restart",
    pk_param="service",
    http=("POST", "/api/monitoring/services/{service}/restart"),
)
async def services_restart(user: dict, service: str, **kwargs) -> AdminResult:
    service, error = _validate_service(service)
    if error:
        return AdminResult.fail(error)

    if service == "kiara-admin-chat.service":
        return AdminResult.fail(
            "Der Admin-Chat-Service kann sich nicht selbst neustarten. "
            "Bitte manuell: systemctl restart kiara-admin-chat.service"
        )

    try:
        result = subprocess.run(
            ["sudo", "systemctl", "restart", service],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_restart"],
        )
    except subprocess.TimeoutExpired:
        return AdminResult.fail(f"Timeout bei Restart von {service}")

    if result.returncode != 0:
        return AdminResult.fail(f"Restart fehlgeschlagen: {result.stderr.strip()[:500]}")

    # Status nach Restart pruefen
    try:
        status_result = subprocess.run(
            ["systemctl", "is-active", service],
            capture_output=True,
            text=True,
            timeout=DEFAULT_TIMEOUTS["subprocess_quick"],
        )
        new_status = status_result.stdout.strip()
    except subprocess.TimeoutExpired:
        new_status = "unbekannt"

    msg = f"Service {service} wurde neugestartet. Status: {new_status}"
    return AdminResult.ok(msg, {"service": service, "status": new_status})
