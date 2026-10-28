"""Smoke-Test: Alle internen Packages unter packages/ sind installiert.

Verhindert den Fall, dass ein Package angelegt und in loader.py/pipeline.py
referenziert wird, aber nie via pip install installiert wurde — was zu
kaskadierenden ModuleNotFoundError und hunderten Collection-Errors fuehrt.
"""

import importlib
from pathlib import Path

import pytest

# Alle pyproject.toml unter packages/ einsammeln
_PACKAGES_DIR = Path(__file__).resolve().parent.parent / "packages"


def _discover_packages():
    """Liest Package-Name aus jeder packages/*/pyproject.toml."""
    results = []
    if not _PACKAGES_DIR.is_dir():
        return results
    for toml_path in sorted(_PACKAGES_DIR.glob("*/pyproject.toml")):
        # Package-Name = Verzeichnisname, Import-Name = mit Unterstrichen
        dir_name = toml_path.parent.name
        import_name = dir_name.replace("-", "_")
        results.append((dir_name, import_name))
    return results


@pytest.mark.parametrize(
    "pkg_name,import_name", _discover_packages(), ids=[p[0] for p in _discover_packages()]
)
def test_package_importable(pkg_name, import_name):
    """Jedes interne Package muss importierbar sein."""
    try:
        mod = importlib.import_module(import_name)
    except ModuleNotFoundError:
        pytest.fail(
            f"Package '{pkg_name}' ist nicht installiert. "
            f"Ausfuehren: pip install -e packages/{pkg_name}/"
        )
    assert mod is not None
