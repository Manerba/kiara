"""Tests fuer Sprint 11b — Guided Workflow Registry und workflow.* Admin-Funktionen.

14 Tests: Registry-Validierung, workflow.list/help/start, Workflow-Struktur.
"""

import asyncio

import pytest

# DB-Backend initialisieren (vor allen kiara-Imports)
import kiara.shared.db as _db

_db._dsn = "postgresql://kiara:kiara@localhost:5432/kiara_test"

# Admin-Registry laden, damit workflows_admin.py Validierung durchlaeuft
from kiara.admin.registry import load_admin_functions
from kiara.admin.workflows import workflow_registry
from kiara.admin.workflows.base import WorkflowDefinition, WorkflowStep
from kiara.admin.workflows.registry import WorkflowRegistry
from kiara.admin.workflows.setup_ingestion import SETUP_INGESTION
from kiara.admin.workflows.setup_llm_backend import SETUP_LLM_BACKEND

load_admin_functions()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def user():
    return {"id": 1}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# 1. workflow.list listet alle vier Workflows
# ---------------------------------------------------------------------------


def test_workflow_list_returns_all_four(user):
    from kiara.admin.functions.workflows_admin import workflow_list

    result = _run(workflow_list(user))
    assert result.success
    names = [w["name"] for w in result.data["workflows"]]
    assert "setup.llm_backend" in names
    assert "setup.datasource" in names
    assert "setup.ingestion" in names
    assert "first_setup" in names
    assert len(names) == 4


# ---------------------------------------------------------------------------
# 2. workflow.help liefert Schritte und required_functions
# ---------------------------------------------------------------------------


def test_workflow_help_returns_steps(user):
    from kiara.admin.functions.workflows_admin import workflow_help

    result = _run(workflow_help(user, name="setup.datasource"))
    assert result.success
    assert "steps" in result.data
    assert "required_functions" in result.data
    assert "datasource.create" in result.data["required_functions"]


# ---------------------------------------------------------------------------
# 3. workflow.start liefert entry_step_id und can_auto_execute=False
# ---------------------------------------------------------------------------


def test_workflow_start_returns_structure(user):
    from kiara.admin.functions.workflows_admin import workflow_start

    result = _run(workflow_start(user, name="setup.ingestion"))
    assert result.success
    assert result.data["entry_step_id"] == "list_datasources"
    assert result.data["can_auto_execute"] is False
    assert "resume_checks" in result.data


# ---------------------------------------------------------------------------
# 4. Registry-Validierung meldet keine fehlenden Admin-Funktionen
# ---------------------------------------------------------------------------


def test_validation_no_missing_functions():
    from kiara.admin.registry import admin_registry

    missing = workflow_registry.validate_against_admin_registry(admin_registry)
    assert missing == [], f"Fehlende Funktionen: {missing}"


# ---------------------------------------------------------------------------
# 5. load_admin_functions registriert workflow.*
# ---------------------------------------------------------------------------


def test_admin_registry_has_workflow_functions():
    from kiara.admin.registry import admin_registry

    assert admin_registry.get("workflow.list") is not None
    assert admin_registry.get("workflow.help") is not None
    assert admin_registry.get("workflow.start") is not None


# ---------------------------------------------------------------------------
# 6. Unbekannter Workflow -> AdminResult.fail
# ---------------------------------------------------------------------------


def test_unknown_workflow_fails(user):
    from kiara.admin.functions.workflows_admin import workflow_start

    result = _run(workflow_start(user, name="nonexistent"))
    assert not result.success
    assert "nicht gefunden" in result.message


# ---------------------------------------------------------------------------
# 7. workflow.start fuehrt keine Admin-Funktion aus
# ---------------------------------------------------------------------------


def test_workflow_start_does_not_execute(user):
    from kiara.admin.functions.workflows_admin import workflow_start

    result = _run(workflow_start(user, name="setup.llm_backend"))
    assert result.success
    assert result.data["stateless"] is True
    notes = result.data["notes"]
    assert any("fuehrt selbst keine Aktion aus" in n for n in notes)


# ---------------------------------------------------------------------------
# 8. first_setup liefert workflow_ref-Schritte
# ---------------------------------------------------------------------------


def test_first_setup_has_workflow_refs(user):
    from kiara.admin.functions.workflows_admin import workflow_start

    result = _run(workflow_start(user, name="first_setup"))
    assert result.success
    steps = result.data["steps"]
    refs = [s["workflow_ref"] for s in steps if s["workflow_ref"]]
    assert "setup.llm_backend" in refs
    assert "setup.datasource" in refs
    assert "setup.ingestion" in refs


# ---------------------------------------------------------------------------
# 9. workflow.list(category="setup") filtert korrekt
# ---------------------------------------------------------------------------


def test_workflow_list_category_filter(user):
    from kiara.admin.functions.workflows_admin import workflow_list

    result = _run(workflow_list(user, category="setup"))
    assert result.success
    assert len(result.data["workflows"]) == 4
    for wf in result.data["workflows"]:
        assert wf["category"] == "setup"


# ---------------------------------------------------------------------------
# 10. workflow.start serialisiert fixed_params, param_mapping, next_step_ids
# ---------------------------------------------------------------------------


def test_workflow_start_serializes_action_details(user):
    from kiara.admin.functions.workflows_admin import workflow_start

    result = _run(workflow_start(user, name="setup.datasource"))
    steps = {s["id"]: s for s in result.data["steps"]}
    create_step = steps["create_datasource"]
    assert "connector_config" in create_step["param_mapping"]
    assert create_step["param_mapping"]["connector_config"] == "config"
    assert "done" in create_step["next_step_ids"]


# ---------------------------------------------------------------------------
# 11. validate_action_bindings erkennt Fehler
# ---------------------------------------------------------------------------


def test_validate_action_bindings_detects_errors():
    reg = WorkflowRegistry()
    bad_wf = WorkflowDefinition(
        name="test.bad",
        label="Bad",
        description="Test",
        category="test",
        help_text="Test",
        steps=[
            WorkflowStep(
                id="bad_step",
                label="Bad",
                description="Bad",
                action="nonexistent.action",
            ),
        ],
        required_functions=[],
        resume_checks=[],
    )
    reg.register(bad_wf)
    from kiara.admin.registry import admin_registry

    errors = reg.validate_action_bindings(admin_registry)
    assert len(errors) >= 1
    assert "nonexistent.action" in errors[0]


# ---------------------------------------------------------------------------
# 12. validate_workflow_refs erkennt fehlende Referenzen
# ---------------------------------------------------------------------------


def test_validate_workflow_refs_detects_missing():
    reg = WorkflowRegistry()
    bad_wf = WorkflowDefinition(
        name="test.refs",
        label="Refs",
        description="Test",
        category="test",
        help_text="Test",
        steps=[
            WorkflowStep(
                id="ref_step",
                label="Ref",
                description="Ref",
                workflow_ref="nonexistent",
            ),
        ],
        required_functions=[],
        resume_checks=[],
    )
    reg.register(bad_wf)
    missing = reg.validate_workflow_refs()
    assert missing == ["nonexistent"]


# ---------------------------------------------------------------------------
# 13. setup.ingestion Verzweigungslogik: conditions
# ---------------------------------------------------------------------------


def test_setup_ingestion_branching():
    steps = {s.id: s for s in SETUP_INGESTION.steps}
    assert steps["start_incremental"].condition == "mode == 'incremental'"
    assert steps["start_full"].condition == "mode == 'full'"
    plan_step = steps["collect_ingestion_plan"]
    assert "start_incremental" in plan_step.next_step_ids
    assert "start_full" in plan_step.next_step_ids
    assert "collect_schedule" in plan_step.next_step_ids


# ---------------------------------------------------------------------------
# 14. setup.llm_backend Datenflusskette: output_keys -> param_mapping
# ---------------------------------------------------------------------------


def test_setup_llm_backend_data_flow():
    steps = {s.id: s for s in SETUP_LLM_BACKEND.steps}
    assert steps["create_local_backend"].output_keys == ["backend_id"]
    assert steps["create_cloud_backend"].output_keys == ["backend_id"]
    assert steps["test_backend"].param_mapping == {"backend_id": "backend_id"}
