from kiara.admin.function import admin_function
from kiara.admin.result import AdminResult
from kiara.config import load_config
from kiara.web.api_dashboard import get_dashboard_kpis


@admin_function(
    name="dashboard.status",
    category="dashboard",
    description="Vollständiger Systemstatus mit KPIs",
    help="Zeigt eine Übersicht aller KPIs: KI-Backends (aktiv/total), "
    "Datenquellen (aktiv/total, nach Typ), VektorDB (Chunks, Größe), "
    "Jobs (laufend, pending).",
    examples=[
        "Zeig mir den Systemstatus",
        "Wie viele Datenquellen sind aktiv?",
        "Dashboard-Übersicht",
    ],
    parameters={"type": "object", "properties": {}, "required": []},
    returns="KPI-Dashboard mit Backends, Datenquellen, VektorDB und Jobs",
    raw_response=True,
    http=("GET", "/api/admin/dashboard"),
)
async def dashboard_status(user: dict) -> AdminResult:
    config = load_config()
    data = get_dashboard_kpis(config)
    return AdminResult.ok("Dashboard-KPIs", data=data)
