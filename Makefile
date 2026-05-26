# Runtime commands use VENV; lint/pre-commit tools use DEV_VENV so deploy
# rebuilds do not have to carry development-only dependencies.
VENV ?= /opt/kiara/venv
DEV_VENV ?= .venv
BOOTSTRAP_PYTHON := python3
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
KIARA := $(VENV)/bin/kiara
DEV_PYTHON := $(DEV_VENV)/bin/python
DEV_PIP := $(DEV_VENV)/bin/pip
RUFF := $(DEV_VENV)/bin/ruff
PRE_COMMIT := $(DEV_VENV)/bin/pre-commit
RUFF_PATHS := kiara kiara_auth tests packages/kiara-anonymizer packages/kiara-db packages/kiara-document-loaders packages/kiara-ingestion-core integrity_cli_tests tools
COMPILE_CACHE_DIR := .cache/compileall
SMOKE_TESTS := tests/test_existing_modules.py tests/test_admin_registry_consistency.py tests/test_backend_selector.py tests/test_chain_reranker.py tests/test_config_ddu.py tests/test_ddu_client.py
PYTORCH_CPU_INDEX := https://download.pytorch.org/whl/cpu
LOCK_TOOL_SPEC := pip-tools>=7.4,<8.0

.PHONY: venv dev-venv lock-tools requirements requirements-check requirements-lock requirements-lock-check install dev lint lint-fix lint-report format format-check pre-commit-install pre-commit-run deps-check compile-check test-collect smoke-test quality-gate test index serve deb clean publish publish-internal iso

## Create virtual environment
venv:
	$(BOOTSTRAP_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

## Create local development virtual environment
dev-venv:
	$(BOOTSTRAP_PYTHON) -m venv $(DEV_VENV)
	$(DEV_PIP) install --upgrade pip

## Install dependencies and package
install: venv
	$(BOOTSTRAP_PYTHON) tools/generate_requirements.py --check
	test -f requirements.lock
	$(PIP) install torch torchvision --index-url $(PYTORCH_CPU_INDEX) --upgrade
	$(PIP) install -r requirements.lock
	$(PIP) install --no-deps -e packages/kiara-anonymizer
	$(PIP) install --no-deps -e packages/kiara-db
	$(PIP) install --no-deps -e packages/kiara-ingestion-core
	$(PIP) install --no-deps -e packages/kiara-document-loaders
	$(PIP) install --no-deps -e .

## Install with dev dependencies
dev: dev-venv
	$(BOOTSTRAP_PYTHON) tools/generate_requirements.py --check
	test -f requirements-dev.lock
	$(DEV_PIP) install torch torchvision --index-url $(PYTORCH_CPU_INDEX) --upgrade
	$(DEV_PIP) install -r requirements-dev.lock
	$(DEV_PIP) install --no-deps -e packages/kiara-anonymizer
	$(DEV_PIP) install --no-deps -e packages/kiara-db
	$(DEV_PIP) install --no-deps -e packages/kiara-ingestion-core
	$(DEV_PIP) install --no-deps -e packages/kiara-document-loaders
	$(DEV_PIP) install --no-deps -e .

## Install lockfile generation tool into the local development venv
lock-tools:
	test -x $(DEV_PYTHON) || $(MAKE) dev-venv
	$(DEV_PYTHON) -c "import piptools" 2>/dev/null || $(DEV_PIP) install "$(LOCK_TOOL_SPEC)"

## Run ruff lint checks on the configured project paths
lint:
	$(RUFF) check $(RUFF_PATHS)

## Run ruff and apply safe automatic lint fixes
lint-fix:
	$(RUFF) check --fix $(RUFF_PATHS)

## Show the current ruff baseline without failing the build
lint-report:
	$(RUFF) check --statistics --exit-zero $(RUFF_PATHS)

## Format Python files with ruff
format:
	$(RUFF) format $(RUFF_PATHS)

## Verify Python formatting with ruff
format-check:
	$(RUFF) format --check $(RUFF_PATHS)

## Install prepared pre-commit hooks locally
pre-commit-install:
	@hooks_path="$$(git config --get core.hooksPath 2>/dev/null || true)"; \
	if [ "$$hooks_path" = "$$(pwd)/.git/hooks" ] || [ "$$hooks_path" = ".git/hooks" ]; then \
		git config --unset-all core.hooksPath; \
	fi
	$(PRE_COMMIT) install

## Run prepared pre-commit hooks over all files
pre-commit-run:
	$(PRE_COMMIT) run --all-files

## Regenerate deploy requirements from pyproject.toml
requirements:
	$(BOOTSTRAP_PYTHON) tools/generate_requirements.py

## Verify requirements.txt matches pyproject.toml
requirements-check:
	$(BOOTSTRAP_PYTHON) tools/generate_requirements.py --check

## Regenerate pinned lockfiles from pyproject.toml files
requirements-lock: requirements lock-tools
	$(DEV_PYTHON) tools/generate_lockfiles.py

## Verify pinned lockfiles match pyproject.toml files
requirements-lock-check: requirements-check lock-tools
	$(DEV_PYTHON) tools/generate_lockfiles.py --check

## Check installed dependency consistency
deps-check: requirements-check
	$(PYTHON) -m pip check

## Compile Python sources without writing pycache files into the source tree
compile-check:
	mkdir -p $(COMPILE_CACHE_DIR)
	PYTHONPYCACHEPREFIX=$(COMPILE_CACHE_DIR) $(PYTHON) -m compileall -q $(RUFF_PATHS)

## Verify pytest can collect the configured test suite
test-collect: deps-check
	$(PYTHON) -m pytest --collect-only -q

## Run fast smoke tests for central contracts used by the CI quality gate
smoke-test:
	$(PYTHON) -m pytest -q $(SMOKE_TESTS)

## Run the minimal CI quality gate
quality-gate:
	$(MAKE) requirements-lock-check
	$(MAKE) deps-check
	$(MAKE) compile-check
	$(MAKE) lint
	$(MAKE) format-check
	$(PYTHON) -m pytest --collect-only -q
	$(MAKE) smoke-test
	$(MAKE) pre-commit-run

## Run tests
test: test-collect
	$(PYTHON) -m pytest -v

## Full indexing
index:
	$(KIARA) index

## Incremental indexing
index-incremental:
	$(KIARA) index --incremental

## Start web UI
serve:
	$(KIARA) serve

## Show status
status:
	$(KIARA) status

## Build .deb package
deb: clean
	dpkg-buildpackage -us -uc -b

## Install default config
config:
	mkdir -p /etc/kiara
	cp -n config/kiara.yaml /etc/kiara/kiara.yaml


## Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf packages/*/build packages/*/dist packages/*/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf debian/kiara debian/.debhelper debian/files debian/*.substvars debian/*.debhelper.log
