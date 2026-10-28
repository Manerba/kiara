from kiara.admin.workflows.base import WorkflowDefinition, WorkflowStep

SETUP_LLM_BACKEND = WorkflowDefinition(
    name="setup.llm_backend",
    label="KI-Backend einrichten",
    description="Gefuehrtes Setup fuer das erste LLM-Backend.",
    category="setup",
    help_text=(
        "Fuehrt durch Auswahl, Anlage und Test eines lokalen oder Cloud-"
        "LLM-Backends. Nutze danach die bestehenden `ki.*`-Funktionen fuer "
        "die eigentliche Ausfuehrung."
    ),
    steps=[
        WorkflowStep(
            id="choose_deployment",
            label="Deployment-Typ waehlen",
            description="Frage, ob ein lokales oder Cloud-Backend eingerichtet werden soll.",
            required_inputs=["deployment_type"],
            next_step_ids=["collect_local_details", "collect_cloud_details"],
        ),
        WorkflowStep(
            id="collect_local_details",
            label="Lokale Backend-Daten sammeln",
            description="Sammle URL und Modell fuer ein Ollama-LLM-Backend.",
            required_inputs=["url", "model"],
            optional_inputs=["name"],
            next_step_ids=["create_local_backend"],
            condition="deployment_type == 'local'",
        ),
        WorkflowStep(
            id="collect_cloud_details",
            label="Cloud-Backend-Daten sammeln",
            description="Sammle Provider, Modell und API-Key fuer ein Cloud-Backend.",
            required_inputs=["provider", "model", "api_key"],
            optional_inputs=["name", "base_url"],
            next_step_ids=["create_cloud_backend"],
            condition="deployment_type == 'cloud'",
        ),
        WorkflowStep(
            id="create_local_backend",
            label="Lokales Backend anlegen",
            description="Rufe `ki.backends.create` fuer ein lokales Ollama-LLM-Backend auf.",
            action="ki.backends.create",
            required_inputs=["url", "model"],
            optional_inputs=["name"],
            fixed_params={"category": "llm", "provider": "ollama"},
            param_mapping={"name": "name", "url": "url", "model": "model"},
            output_keys=["backend_id"],
            next_step_ids=["test_backend"],
        ),
        WorkflowStep(
            id="create_cloud_backend",
            label="Cloud-Backend anlegen",
            description="Rufe `ki.backends.create` fuer ein Cloud-LLM-Backend auf.",
            action="ki.backends.create",
            required_inputs=["provider", "model", "api_key"],
            optional_inputs=["name", "base_url"],
            fixed_params={"category": "cloud"},
            param_mapping={
                "provider": "provider",
                "name": "name",
                "model": "model",
                "api_key": "api_key",
                "base_url": "base_url",
            },
            output_keys=["backend_id"],
            next_step_ids=["test_backend"],
        ),
        WorkflowStep(
            id="test_backend",
            label="Backend testen",
            description="Teste das gerade angelegte Backend ueber `ki.backends.test` mit `backend_id`.",
            action="ki.backends.test",
            required_inputs=["backend_id"],
            param_mapping={"backend_id": "backend_id"},
            next_step_ids=["done"],
        ),
        WorkflowStep(
            id="done",
            label="Abgeschlossen",
            description="Das LLM-Backend ist angelegt und getestet.",
        ),
    ],
    required_functions=[
        "ki.backends.categories",
        "ki.backends.create",
        "ki.backends.test",
        "ki.backends.list",
    ],
    resume_checks=["ki.backends.list"],
    can_auto_execute=False,
)
