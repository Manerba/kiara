from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class WorkflowStep:
    id: str
    label: str
    description: str
    action: str | None = None
    required_inputs: list[str] = field(default_factory=list)
    optional_inputs: list[str] = field(default_factory=list)
    fixed_params: dict[str, str | int | bool] = field(default_factory=dict)
    param_mapping: dict[str, str] = field(default_factory=dict)
    output_keys: list[str] = field(default_factory=list)
    next_step_ids: list[str] = field(default_factory=list)
    workflow_ref: str | None = None
    condition: str | None = None


@dataclass(frozen=True, slots=True)
class WorkflowDefinition:
    name: str
    label: str
    description: str
    category: str
    help_text: str
    steps: list[WorkflowStep]
    required_functions: list[str]
    resume_checks: list[str]
    can_auto_execute: bool = False
