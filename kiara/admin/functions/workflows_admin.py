from dataclasses import asdict

from kiara.admin.function import admin_function
from kiara.admin.registry import admin_registry
from kiara.admin.result import AdminResult
from kiara.admin.workflows import workflow_registry


@admin_function(
    name="workflow.list",
    category="workflows_admin",
    description="Verfuegbare Guided Workflows auflisten",
    help="Listet alle fuer den Admin-Chat verfuegbaren Guided Workflows auf.",
    examples=["Welche Setup-Workflows gibt es?"],
    parameters={
        "type": "object",
        "properties": {
            "category": {"type": "string", "description": "Optionaler Kategorie-Filter"}
        },
        "required": [],
    },
    returns="Liste der Workflows mit Label und Beschreibung",
)
async def workflow_list(user: dict, category: str = "") -> AdminResult:
    workflows = (
        workflow_registry.list_by_category(category) if category else workflow_registry.list_all()
    )
    return AdminResult.ok(
        f"{len(workflows)} Workflows verfuegbar",
        data={
            "workflows": [
                {
                    "name": wf.name,
                    "label": wf.label,
                    "description": wf.description,
                    "category": wf.category,
                    "can_auto_execute": wf.can_auto_execute,
                }
                for wf in workflows
            ]
        },
    )


@admin_function(
    name="workflow.help",
    category="workflows_admin",
    description="Details zu einem Guided Workflow",
    help="Zeigt alle Schritte, benoetigten Funktionen und Resume-Hinweise.",
    examples=["Erklaere den Workflow setup.datasource"],
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Workflow-Name"}},
        "required": ["name"],
    },
    returns="Workflow-Detailansicht",
)
async def workflow_help(user: dict, name: str) -> AdminResult:
    info = workflow_registry.get_help(name)
    if not info:
        return AdminResult.fail(f"Workflow '{name}' nicht gefunden")
    return AdminResult.ok(f"Workflow '{name}'", data=info)


@admin_function(
    name="workflow.start",
    category="workflows_admin",
    description="Workflow-Definition fuer den naechsten Schritt laden",
    help="Liefert den Einstiegspunkt eines Guided Workflows. Fuehrt selbst keine "
    "Admin-Funktion aus; der Agent ruft die im Workflow referenzierten "
    "Admin-Funktionen separat auf.",
    examples=["Starte den Workflow setup.llm_backend"],
    parameters={
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Workflow-Name"}},
        "required": ["name"],
    },
    returns="Workflow-Definition mit Einstiegsschritt",
)
async def workflow_start(user: dict, name: str) -> AdminResult:
    wf = workflow_registry.get(name)
    if not wf:
        return AdminResult.fail(f"Workflow '{name}' nicht gefunden")
    return AdminResult.ok(
        f"Workflow '{name}' gestartet",
        data={
            "name": wf.name,
            "label": wf.label,
            "stateless": True,
            "can_auto_execute": wf.can_auto_execute,
            "entry_step_id": wf.steps[0].id if wf.steps else "",
            "steps": [asdict(step) for step in wf.steps],
            "required_functions": wf.required_functions,
            "resume_checks": wf.resume_checks,
            "notes": [
                "workflow.start fuehrt selbst keine Aktion aus",
                "Rufe die in step.action referenzierten Admin-Funktionen direkt auf",
                "Wenn step.workflow_ref gesetzt ist, lade zuerst workflow.start fuer diesen Teilworkflow",
                "Verwende step.fixed_params und step.param_mapping fuer den konkreten Admin-Function-Call",
                "Reiche Rueckgabewerte wie backend_id, ds_id und job_id an Folgeschritte weiter",
            ],
        },
    )


missing_functions = workflow_registry.validate_against_admin_registry(admin_registry)
missing_refs = workflow_registry.validate_workflow_refs()
binding_errors = workflow_registry.validate_action_bindings(admin_registry)
if missing_functions or missing_refs or binding_errors:
    raise RuntimeError(
        "Guided Workflow Registry ungueltig: "
        f"functions={missing_functions}, refs={missing_refs}, bindings={binding_errors}"
    )
