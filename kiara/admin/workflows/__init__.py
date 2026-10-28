from kiara.admin.workflows.first_setup import FIRST_SETUP
from kiara.admin.workflows.registry import WorkflowRegistry
from kiara.admin.workflows.setup_datasource import SETUP_DATASOURCE
from kiara.admin.workflows.setup_ingestion import SETUP_INGESTION
from kiara.admin.workflows.setup_llm_backend import SETUP_LLM_BACKEND

workflow_registry = WorkflowRegistry()
for workflow in (
    SETUP_LLM_BACKEND,
    SETUP_DATASOURCE,
    SETUP_INGESTION,
    FIRST_SETUP,
):
    workflow_registry.register(workflow)
