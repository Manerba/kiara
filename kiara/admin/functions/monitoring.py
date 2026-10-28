"""Admin-Funktionen fuer System-Monitoring (Admin-Chat + HTTP-Routen)."""

import subprocess

from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult


@admin_function(
    name="system.storage",
    description="Speicherbedarf der Datenbanken und Suchindizes",
    category="system",
    help="Zeigt PostgreSQL inklusive größter Tabellen, Qdrant inklusive aller Collections "
    "und Elasticsearch. Messwerte werden im Hintergrund jede Minute aktualisiert.",
    examples=["Wie groß sind die Datenbanken?", "Welche Collection belegt am meisten Speicher?"],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="Speicherübersicht mit Gesamtgröße, Tabellen und Collections",
    raw_response=True,
    http=("GET", "/api/monitoring/storage"),
)
async def system_storage(user: dict) -> AdminResult:
    from kiara.monitoring.storage import get_storage_sizes

    return AdminResult.ok("Speicherbedarf der Datenbanken und Indizes", get_storage_sizes())


def _fmt_bytes(n: int) -> str:
    """Formatiert Bytes als menschenlesbare Groesse (KB/MB/GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


@admin_function(
    name="system.metrics",
    description="Aktuelle Systemmetriken abrufen (CPU, RAM, Swap, Disk, Load, Netzwerk)",
    category="system",
    help="Gibt die aktuellen Systemmetriken des Servers zurueck.",
    examples=["Wie ist die CPU-Auslastung?", "Systemstatus", "Wie viel RAM ist belegt?"],
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
    returns="Aktuelle Systemmetriken (CPU, RAM, Swap, Disk, Load, Netzwerk)",
    http=("GET", "/api/monitoring/current"),
)
async def system_metrics(user: dict, **kwargs) -> AdminResult:
    """Gibt den letzten Metrik-Snapshot zurueck."""
    from kiara.monitoring.collector import _PSUTIL_OK, get_latest

    data = get_latest()
    if not data:
        if not _PSUTIL_OK:
            return AdminResult.fail(
                "Collector deaktiviert: psutil nicht verfuegbar. "
                "Bitte installieren: pip install psutil"
            )
        return AdminResult.fail("Collector laeuft noch nicht.")
    return AdminResult.ok(
        f"CPU: {data['cpu_percent']}%, RAM: {_fmt_bytes(data['ram_used_bytes'])}/{_fmt_bytes(data['ram_total_bytes'])}, "
        f"Load: {data['load_1m']}/{data['load_5m']}/{data['load_15m']}, "
        f"Disk: {_fmt_bytes(data['disk_used_bytes'])}/{_fmt_bytes(data['disk_total_bytes'])}",
        data,
    )


@admin_function(
    name="system.journal",
    description="Letzte Fehler aus dem systemd-Journal aller KIara-Dienste",
    category="system",
    help="Zeigt die letzten Fehlereintraege (priority=err) aus journalctl fuer alle KIara-Services.",
    examples=["Zeig mir die letzten Systemfehler", "Journal-Fehler", "Gibt es Fehler in den Logs?"],
    parameters={
        "type": "object",
        "properties": {
            "max_lines": {
                "type": "integer",
                "description": "Maximale Anzahl Fehlerzeilen (Standard: 50)",
                "default": 50,
            },
        },
        "required": [],
    },
    returns="Liste der letzten systemd-Fehler aller KIara-Dienste",
    http=("GET", "/api/monitoring/journal"),
)
async def system_journal(user: dict, max_lines: int = 50, **kwargs) -> AdminResult:
    """Letzte Fehler aus dem systemd-Journal aller KIara-Dienste."""
    from kiara.admin.functions.services import ALLOWED_SERVICES

    if not ALLOWED_SERVICES:
        return AdminResult.ok("Keine Services konfiguriert.", {"errors": []})

    errors = []
    lines_per_svc = max(1, max_lines // len(ALLOWED_SERVICES))
    for svc in ALLOWED_SERVICES:
        try:
            result = subprocess.run(
                [
                    "sudo",
                    "journalctl",
                    "--no-pager",
                    "--output=short-iso",
                    "-u",
                    svc,
                    "-p",
                    "err",
                    "-n",
                    str(lines_per_svc),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            for line in result.stdout.strip().splitlines():
                if line.strip() and not line.startswith("-- "):
                    errors.append(line)
        except Exception:
            pass

    errors.sort(reverse=True)
    errors = errors[:max_lines]

    if not errors:
        return AdminResult.ok("Keine Fehler im Journal.", {"errors": []})

    # Chat-freundliche Zusammenfassung
    summary = "\n".join(errors[:10])
    suffix = f"\n... (+{len(errors) - 10} weitere)" if len(errors) > 10 else ""
    return AdminResult.ok(
        f"{len(errors)} Fehler gefunden:\n```\n{summary}{suffix}\n```",
        {"errors": errors},
    )


@admin_function(
    name="health.cleanup_history",
    description="Alte Health-Log-Eintraege loeschen",
    category="system",
    help="Loescht Health-Log-Eintraege die aelter als die angegebene Anzahl Tage sind. "
    "Reduziert Datenbankwachstum (~1440 Eintraege/Tag bei 60s-Intervall).",
    examples=[
        "Health-Logs bereinigen",
        "Loesche Health-Logs aelter als 7 Tage",
    ],
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type": "integer",
                "description": "Logs aelter als X Tage loeschen (Default: 30)",
                "default": 30,
            },
        },
        "required": [],
    },
    returns="Anzahl geloeschter Health-Log-Eintraege",
    requires_confirmation=True,
    confirm_template="health-logs älter als {days} tage löschen",
    changelog_area="system",
    changelog_action="delete",
    http=("DELETE", "/api/monitoring/health-log/history"),
)
async def health_cleanup_history(user: dict, days: int = 30, **kwargs) -> AdminResult:
    """Loescht alte Health-Log-Eintraege."""
    from kiara.repositories.health import HealthLogRepository

    repo = HealthLogRepository()
    count = repo.cleanup_old(days=days)
    return AdminResult.ok(
        f"{count} Health-Log-Eintraege geloescht",
        data={"deleted": count, "message": f"{count} Health-Logs geloescht"},
    )
