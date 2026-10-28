#!/usr/bin/env python3
"""Generate pinned dependency lockfiles from pyproject.toml files."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from generate_requirements import PYPROJECT, build_requirements

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@dataclass(frozen=True)
class LockSpec:
    output: Path
    extras: tuple[str, ...]
    label: str


LOCK_SPECS = (
    LockSpec(
        output=PROJECT_ROOT / "requirements.lock",
        extras=("exchange", "mcp", "test"),
        label="deployed runtime plus test tools",
    ),
    LockSpec(
        output=PROJECT_ROOT / "requirements-dev.lock",
        extras=("exchange", "mcp", "test", "dev"),
        label="local development and CI",
    ),
)


def _input_text(spec: LockSpec) -> str:
    return build_requirements(PYPROJECT, spec.extras)


def _source_hash(spec: LockSpec) -> str:
    return hashlib.sha256(_input_text(spec).encode("utf-8")).hexdigest()


def _header(spec: LockSpec) -> str:
    return "\n".join(
        [
            "# Generated from pyproject.toml files by tools/generate_lockfiles.py.",
            "# Do not edit manually. Run: make requirements-lock",
            f"# Included extras: {', '.join(spec.extras)}",
            f"# Scope: {spec.label}",
            f"# Source-SHA256: {_source_hash(spec)}",
            "",
        ]
    )


def _compile_lock(spec: LockSpec, *, upgrade: bool) -> str:
    with tempfile.TemporaryDirectory(prefix="kiara-lock-") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / f"{spec.output.stem}.in"
        output_path = tmp_path / spec.output.name
        input_path.write_text(_input_text(spec), encoding="utf-8")
        if spec.output.exists() and not upgrade:
            output_path.write_text(spec.output.read_text(encoding="utf-8"), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--resolver=backtracking",
            "--strip-extras",
            "--no-header",
            "--no-annotate",
            "--quiet",
            "--extra-index-url",
            PYTORCH_CPU_INDEX,
            "--output-file",
            str(output_path),
            str(input_path),
        ]
        if upgrade:
            cmd.insert(4, "--upgrade")

        try:
            subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        except ModuleNotFoundError:
            raise SystemExit("pip-tools fehlt. Installiere Dev-Tools mit: make dev") from None
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc

        return _header(spec) + output_path.read_text(encoding="utf-8").lstrip()


def _fast_check(spec: LockSpec) -> int:
    if not spec.output.exists():
        sys.stderr.write(f"Lockfile fehlt: {spec.output}\n")
        return 1

    current = spec.output.read_text(encoding="utf-8")
    expected_header = _header(spec)
    if current.startswith(expected_header):
        return 0

    expected_lines = expected_header.splitlines(keepends=True)
    current_lines = current.splitlines(keepends=True)[: len(expected_lines)]
    diff = difflib.unified_diff(
        current_lines,
        expected_lines,
        fromfile=str(spec.output),
        tofile="expected-header",
    )
    sys.stderr.writelines(diff)
    return 1


def _write_or_check(spec: LockSpec, *, check: bool, upgrade: bool) -> int:
    generated = _compile_lock(spec, upgrade=upgrade)
    if check:
        current = spec.output.read_text(encoding="utf-8") if spec.output.exists() else ""
        if current != generated:
            diff = difflib.unified_diff(
                current.splitlines(keepends=True),
                generated.splitlines(keepends=True),
                fromfile=str(spec.output),
                tofile="generated",
            )
            sys.stderr.writelines(diff)
            return 1
        return 0

    spec.output.write_text(generated, encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if any lockfile is stale")
    parser.add_argument(
        "--resolve-check",
        action="store_true",
        help="re-resolve lockfiles during check instead of only validating source hashes",
    )
    parser.add_argument(
        "--upgrade", action="store_true", help="refresh pins to newest allowed versions"
    )
    args = parser.parse_args(argv)

    failed = 0
    for spec in LOCK_SPECS:
        if args.check and not args.resolve_check:
            failed |= _fast_check(spec)
        else:
            failed |= _write_or_check(spec, check=args.check, upgrade=args.upgrade)
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
