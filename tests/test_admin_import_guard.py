"""Tests: Lazy-Import-Guard fuer Admin-Funktionen.

Stellt sicher, dass:
1. Externe Packages (ldap3, httpx, qdrant_client) korrekt installiert sind
2. Alle Admin-Funktionsmodule ohne Fehler geladen werden koennen
3. Lazy Imports in Admin-Funktionen zur Laufzeit aufloesbar sind

Verhindert ImportError-Fehler wie den ldap3-Ausfall (2026-04-05).
"""

import ast
import importlib
import pathlib
import sys

import pytest

# Pfade
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
_ADMIN_FUNCTIONS_DIR = _PROJECT_ROOT / "kiara" / "admin" / "functions"
_REGISTRY_PATH = _PROJECT_ROOT / "kiara" / "admin" / "registry.py"

_STDLIB = getattr(sys, "stdlib_module_names", set())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_admin_modules() -> list[str]:
    """Extrahiert die Modul-Liste aus load_admin_functions() per AST.

    Liest registry.py und findet die _modules-Liste in load_admin_functions(),
    damit neue Module automatisch mitgetestet werden.
    """
    source = _REGISTRY_PATH.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "load_admin_functions":
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and target.id == "_modules":
                            return ast.literal_eval(child.value)
    return []


def _collect_lazy_imports() -> list[str]:
    """Scannt Admin-Funktionsdateien per AST, sammelt Imports aus Funktionsbodies.

    Findet alle 'from X import Y' und 'import X' Statements innerhalb von
    (async) def Bodies — also Imports die erst zur Laufzeit ausgefuehrt werden.
    Stdlib-Module werden uebersprungen.
    """
    seen = set()
    for py_file in sorted(_ADMIN_FUNCTIONS_DIR.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        tree = ast.parse(py_file.read_text(), filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.ImportFrom) and child.module:
                    top = child.module.split(".")[0]
                    if top not in _STDLIB:
                        seen.add(child.module)
                elif isinstance(child, ast.Import):
                    for alias in child.names:
                        top = alias.name.split(".")[0]
                        if top not in _STDLIB:
                            seen.add(alias.name)
    return sorted(seen)


_ADMIN_MODULES = _get_admin_modules()
_LAZY_MODULES = _collect_lazy_imports()


# ---------------------------------------------------------------------------
# 1. Externe Packages — Smoke-Test
# ---------------------------------------------------------------------------


class TestExternalPackages:
    """Prueft, dass externe Packages korrekt installiert und importierbar sind.

    Diese Packages werden von Admin-Funktionen lazy importiert.
    Ein kaputter oder fehlender Import fuehrt zu 'Speichern fehlgeschlagen'
    ohne hilfreiche Fehlermeldung.
    """

    @pytest.mark.parametrize(
        "package,expected_names",
        [
            ("ldap3", ["Server", "Connection", "NONE", "SUBTREE", "Tls"]),
            ("httpx", ["AsyncClient", "Client"]),
            ("qdrant_client", ["QdrantClient"]),
        ],
    )
    def test_external_package_importable(self, package, expected_names):
        """Externes Package muss importierbar sein und erwartete Namen exportieren."""
        mod = importlib.import_module(package)
        for name in expected_names:
            assert hasattr(mod, name), (
                f"{package}.{name} nicht gefunden — Package kaputt oder falsche Version?"
            )


# ---------------------------------------------------------------------------
# 2. Admin-Funktionsmodule — Lade-Test
# ---------------------------------------------------------------------------


class TestAdminModuleLoading:
    """Prueft, dass alle Admin-Funktionsmodule aus der Registry importierbar sind.

    Die Modul-Liste wird automatisch aus load_admin_functions() in registry.py
    extrahiert, damit neue Module automatisch mitgetestet werden.
    """

    @pytest.mark.parametrize("module_path", _ADMIN_MODULES)
    def test_admin_module_importable(self, module_path):
        """Jedes Admin-Funktionsmodul muss ohne ImportError ladbar sein.

        RuntimeError (z.B. Registry-Validierung in workflows_admin) wird
        toleriert — das sind Business-Logic-Fehler, keine fehlenden Packages.
        """
        try:
            importlib.import_module(module_path)
        except ImportError:
            raise  # Fehlende Packages → Test soll fehlschlagen
        except RuntimeError:
            pass  # Registry-Validierung etc. → kein Import-Problem


# ---------------------------------------------------------------------------
# 3. Lazy Imports — Aufloesung
# ---------------------------------------------------------------------------


class TestLazyImportResolution:
    """Prueft, dass alle lazy-importierten Module aufloesbar sind.

    Auto-Discovery per AST: Scannt alle Dateien in admin/functions/ nach
    Import-Statements innerhalb von Funktionsbodies. Jedes gefundene Modul
    muss importierbar sein.
    """

    # Optionale Cloud-Packages — nicht in jeder Umgebung installiert
    _OPTIONAL_TOPS = {"anthropic", "openai"}

    @pytest.mark.parametrize("module_path", _LAZY_MODULES)
    def test_lazy_import_resolvable(self, module_path):
        """Jedes lazy-importierte Modul muss importierbar sein."""
        top = module_path.split(".")[0]
        if top in self._OPTIONAL_TOPS:
            pytest.importorskip(top)
        importlib.import_module(module_path)
