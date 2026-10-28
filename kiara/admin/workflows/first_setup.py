from kiara.admin.workflows.base import WorkflowDefinition, WorkflowStep

FIRST_SETUP = WorkflowDefinition(
    name="first_setup",
    label="Ersteinrichtung",
    description="Meta-Workflow fuer die Reihenfolge Backend -> Datenquelle -> Ingestion.",
    category="setup",
    help_text=(
        "Leitet nacheinander durch Backend-, Datenquellen- und Ingestion-"
        "Setup. Der Workflow bleibt stateless und fuehrt selbst nichts aus."
    ),
    steps=[
        WorkflowStep(
            id="setup_backend",
            label="KI-Backend einrichten",
            description="Lade zuerst den Teilworkflow `setup.llm_backend`.",
            workflow_ref="setup.llm_backend",
            next_step_ids=["setup_datasource"],
        ),
        WorkflowStep(
            id="setup_datasource",
            label="Datenquelle einrichten",
            description="Lade danach den Teilworkflow `setup.datasource`.",
            workflow_ref="setup.datasource",
            next_step_ids=["setup_ingestion"],
        ),
        WorkflowStep(
            id="setup_ingestion",
            label="Indexierung einrichten",
            description="Lade abschliessend den Teilworkflow `setup.ingestion`.",
            workflow_ref="setup.ingestion",
            next_step_ids=["done"],
        ),
        WorkflowStep(
            id="done",
            label="Abgeschlossen",
            description="Die Ersteinrichtung ist logisch vollstaendig durchlaufen.",
        ),
    ],
    required_functions=["ki.backends.list", "datasource.list", "jobs.list"],
    resume_checks=["ki.backends.list", "datasource.list", "jobs.list"],
    can_auto_execute=False,
)
