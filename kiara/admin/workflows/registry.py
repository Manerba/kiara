from dataclasses import asdict

from kiara.admin.workflows.base import WorkflowDefinition


class WorkflowRegistry:
    def __init__(self) -> None:
        self._workflows: dict[str, WorkflowDefinition] = {}

    def register(self, workflow: WorkflowDefinition) -> None:
        self._workflows[workflow.name] = workflow

    def get(self, name: str) -> WorkflowDefinition | None:
        return self._workflows.get(name)

    def list_all(self) -> list[WorkflowDefinition]:
        return list(self._workflows.values())

    def list_by_category(self, category: str) -> list[WorkflowDefinition]:
        return [wf for wf in self._workflows.values() if wf.category == category]

    def get_help(self, name: str) -> dict | None:
        wf = self.get(name)
        if not wf:
            return None
        return {
            "name": wf.name,
            "label": wf.label,
            "description": wf.description,
            "help_text": wf.help_text,
            "required_functions": wf.required_functions,
            "resume_checks": wf.resume_checks,
            "steps": [asdict(step) for step in wf.steps],
        }

    def validate_against_admin_registry(self, admin_registry) -> list[str]:
        missing = sorted(
            {
                fn_name
                for wf in self._workflows.values()
                for fn_name in (wf.required_functions + wf.resume_checks)
                if admin_registry.get(fn_name) is None
            }
        )
        return missing

    def validate_workflow_refs(self) -> list[str]:
        missing = sorted(
            {
                step.workflow_ref
                for wf in self._workflows.values()
                for step in wf.steps
                if step.workflow_ref and step.workflow_ref not in self._workflows
            }
        )
        return missing

    def validate_action_bindings(self, admin_registry) -> list[str]:
        errors: list[str] = []
        for wf in self._workflows.values():
            for step in wf.steps:
                if not step.action:
                    continue

                func = admin_registry.get(step.action)
                if func is None:
                    errors.append(
                        f"{wf.name}:{step.id}: action '{step.action}' nicht in admin_registry"
                    )
                    continue

                properties = func.parameters.get("properties", {})
                required = set(func.parameters.get("required", []))
                provided_targets = set(step.fixed_params) | set(step.param_mapping)
                available_sources = set(step.required_inputs) | set(step.optional_inputs)

                missing_required = sorted(required - provided_targets)
                unknown_targets = sorted(provided_targets - set(properties))
                unknown_sources = sorted(
                    source_name
                    for source_name in step.param_mapping.values()
                    if source_name not in available_sources
                )

                if missing_required:
                    errors.append(
                        f"{wf.name}:{step.id}: missing required target params {missing_required}"
                    )
                if unknown_targets:
                    errors.append(f"{wf.name}:{step.id}: unknown target params {unknown_targets}")
                if unknown_sources:
                    errors.append(
                        f"{wf.name}:{step.id}: unknown workflow context sources {unknown_sources}"
                    )
        return errors
