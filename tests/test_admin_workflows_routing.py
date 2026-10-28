"""Regressionen fuer Admin-Workflow-Routing."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_admin_workflow_templates_route_uses_admin_function_first():
    """Admin-App darf nicht den User-Workflow-Router mit get_chat_user matchen."""
    from kiara.admin.functions import workflows  # noqa: F401
    from kiara.admin.registry import admin_registry

    func = admin_registry.get("workflows.templates.list")
    assert func is not None
    assert func.http_method == "GET"
    assert func.http_path == "/api/workflows/templates"

    admin_app_src = (ROOT / "kiara" / "web" / "admin_app.py").read_text(encoding="utf-8")
    assert "from kiara.web.api_workflows import router as workflows_router" not in admin_app_src
    assert "app.include_router(workflows_router" not in admin_app_src
    assert admin_app_src.index("app.include_router(admin_workflows_router)") < admin_app_src.index(
        "generate_admin_routes(app, admin_registry)"
    )


def test_admin_workflow_download_route_is_admin_only_file_route():
    """FileResponse-Download bleibt im Admin-Prozess explizit verfuegbar."""
    from kiara.web.api_admin_workflows import router

    routes = [
        route
        for route in router.routes
        if getattr(route, "path", None) == "/api/workflows/exports/{filename}/download"
        and "GET" in (getattr(route, "methods", set()) or set())
    ]
    assert routes
    assert routes[0].endpoint.__name__ == "download_export"
