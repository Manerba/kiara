#!/usr/bin/env python3
"""Generate requirements.txt from pyproject.toml files.

KIara keeps pyproject.toml as the source of truth. requirements.txt remains a
generated deploy artifact because Debian/postinst and deploy.sh install from it.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
DEFAULT_OUTPUT = PROJECT_ROOT / "requirements.txt"
RUNTIME_EXTRAS = ("exchange", "mcp")


def _dependency_key(requirement: str) -> str:
    """Best-effort package key for duplicate detection without extra deps."""
    name = re.split(r"[<>=!~; ]", requirement, maxsplit=1)[0]
    return name.split("[", maxsplit=1)[0].lower().replace("_", "-")


def _dedupe(requirements: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for requirement in requirements:
        key = _dependency_key(requirement)
        if key in seen:
            continue
        seen.add(key)
        result.append(requirement)
    return result


def _read_project(pyproject_path: Path) -> dict:
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    return data["project"]


def _local_package_pyprojects(project_root: Path) -> list[Path]:
    package_root = project_root / "packages"
    if not package_root.exists():
        return []
    return sorted(package_root.glob("*/pyproject.toml"))


def _local_package_dependencies(project_root: Path) -> list[str]:
    package_projects = [_read_project(path) for path in _local_package_pyprojects(project_root)]
    local_project_names = {
        project["name"].lower().replace("_", "-")
        for project in package_projects
        if project.get("name")
    }

    dependencies: list[str] = []
    for project in package_projects:
        for dependency in project.get("dependencies", []):
            if _dependency_key(dependency) in local_project_names:
                continue
            dependencies.append(dependency)
    return _dedupe(dependencies)


def build_requirements(pyproject_path: Path, extras: tuple[str, ...]) -> str:
    project = _read_project(pyproject_path)
    requirements = list(project.get("dependencies", []))
    optional_dependencies = project.get("optional-dependencies", {})
    requirement_keys = {_dependency_key(requirement) for requirement in requirements}
    local_requirements = [
        requirement
        for requirement in _local_package_dependencies(pyproject_path.resolve().parent)
        if _dependency_key(requirement) not in requirement_keys
    ]

    extra_requirements: list[str] = []
    for extra in extras:
        if extra not in optional_dependencies:
            raise KeyError(f"Optional dependency group not found: {extra}")
        extra_requirements.extend(optional_dependencies[extra])

    lines = [
        "# Generated from pyproject.toml by tools/generate_requirements.py.",
        "# Do not edit manually. Run: make requirements",
        f"# Included runtime extras: {', '.join(extras) if extras else 'none'}",
        "",
        *_dedupe(requirements),
    ]
    if local_requirements:
        lines.extend(
            [
                "",
                "# Monorepo package runtime dependencies",
                *_dedupe(local_requirements),
            ]
        )
    if extra_requirements:
        lines.extend(
            [
                "",
                "# Runtime extras included for Debian/deploy compatibility",
                *_dedupe(extra_requirements),
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if output is stale")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--extra",
        action="append",
        dest="extras",
        help="optional dependency group to include; defaults to runtime extras",
    )
    args = parser.parse_args(argv)

    extras = tuple(args.extras) if args.extras is not None else RUNTIME_EXTRAS
    generated = build_requirements(PYPROJECT, extras)

    if args.check:
        current = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if current != generated:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile=str(args.output),
                tofile="generated",
            )
            sys.stderr.writelines(diff)
            return 1
        return 0

    args.output.write_text(generated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
