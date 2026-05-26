from pathlib import Path

from tools.generate_requirements import PYPROJECT, build_requirements

ROOT = Path(__file__).resolve().parent.parent


def test_generated_requirements_include_external_monorepo_dependencies():
    requirements = build_requirements(PYPROJECT, ("exchange", "mcp"))

    assert "psycopg2-binary" in requirements
    assert "kiara-db" not in requirements


def test_runtime_lockfile_pins_cpu_torch_stack_and_test_tools():
    lockfile = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in lockfile
    assert "torch==" in lockfile
    assert "torchvision==" in lockfile
    assert "pytest-timeout==" in lockfile
    assert "pip-tools==" not in lockfile


def test_dev_lockfile_includes_dev_tools():
    lockfile = (ROOT / "requirements-dev.lock").read_text(encoding="utf-8")

    assert "pip-tools==" in lockfile
    assert "pre-commit==" in lockfile
    assert "ruff==" in lockfile
