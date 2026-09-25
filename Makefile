SHELL := /bin/bash

LIBRARY_JSON := griptape-nodes-library.json
EXEC_TEST_VENV := .venv-test-exec

.PHONY: version/get
version/get: ## Get version.
	@jq -r '.metadata.library_version' $(LIBRARY_JSON)

.PHONY: version/set
version/set: ## Set version. Usage: make version/set v=1.2.3
	@jq --arg v "$(v)" '.metadata.library_version = $$v' $(LIBRARY_JSON) > $(LIBRARY_JSON).tmp
	@mv $(LIBRARY_JSON).tmp $(LIBRARY_JSON)
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/patch
version/patch: ## Bump patch version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$${minor}.$$((patch + 1))"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' $(LIBRARY_JSON) > $(LIBRARY_JSON).tmp; \
	mv $(LIBRARY_JSON).tmp $(LIBRARY_JSON); \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/minor
version/minor: ## Bump minor version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$${major}.$$((minor + 1)).0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' $(LIBRARY_JSON) > $(LIBRARY_JSON).tmp; \
	mv $(LIBRARY_JSON).tmp $(LIBRARY_JSON); \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/major
version/major: ## Bump major version.
	@CURRENT=$$($(MAKE) --no-print-directory version/get); \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	NEW_VERSION="$$((major + 1)).0.0"; \
	jq --arg v "$$NEW_VERSION" '.metadata.library_version = $$v' $(LIBRARY_JSON) > $(LIBRARY_JSON).tmp; \
	mv $(LIBRARY_JSON).tmp $(LIBRARY_JSON); \
	echo "Bumped to $$NEW_VERSION"
	@$(MAKE) --no-print-directory version/commit

.PHONY: version/commit
version/commit: ## Commit version.
	@git add $(LIBRARY_JSON)
	@git commit -m "chore: bump v$$($(MAKE) --no-print-directory version/get)"

.PHONY: version/publish
version/publish: ## Create and push git tags.
	@git fetch --tags --force
	@VERSION=$$($(MAKE) --no-print-directory version/get); \
	git tag "v$$VERSION"; \
	git tag stable -f; \
	git push origin "v$$VERSION"; \
	git push -f origin stable

.PHONY: deps/sync
deps/sync: ## Sync pip_dependencies and pip_dependencies_exec in the library JSON from pyproject.toml.
	@uv run python scripts/sync_dependencies.py

.PHONY: install
install: ## Install all dependencies.
	@make install/all

.PHONY: install/core
install/core: deps/sync ## Install core dependencies.
	@uv sync

.PHONY: install/all
install/all: deps/sync ## Install all dependencies.
	@# No `--all-extras`: that installs the `exec` extra into .venv, which the engine splices onto the
	@# orchestrator's sys.path. A heavy import would then succeed for a developer and fail for a user.
	@uv sync --all-groups

.PHONY: install/dev
install/dev: ## Install dev dependencies.
	@uv sync --group dev

.PHONY: lint
lint: ## Lint project.
	@uv run ruff check --fix

.PHONY: format
format: ## Format project.
	@uv run ruff format

.PHONY: fix
fix: ## Fix project.
	@make format
	@uv run ruff check --fix --unsafe-fixes

# `test/unit` is a prerequisite because CI runs `make check` and nothing else, so a suite outside it
# gates nothing -- and these tests pin invariants whose violations are silent.
.PHONY: check
check: check/format check/lint check/types check/json check/worker-safe check/edit-time-imports test/unit ## Run all checks.

.PHONY: check/format
check/format:
	@uv run ruff format --check

.PHONY: check/lint
check/lint:
	@uv run ruff check .

.PHONY: check/types
check/types:
	@# Type-checked against the execution environment, because that is where torch and diffusers are.
	@# The default venv is the edit-time one, and pyright cannot resolve what is deliberately not there.
	@UV_PROJECT_ENVIRONMENT=$(EXEC_TEST_VENV) uv sync --extra exec --all-groups
	@UV_PROJECT_ENVIRONMENT=$(EXEC_TEST_VENV) uv run pyright .

.PHONY: check/json
check/json: ## Validate JSON files.
	@echo "Checking JSON files..."
	@find . -name "*.json" -type f \
		! -path "./.venv/*" \
		! -path "./node_modules/*" \
		-exec sh -c 'jq empty "{}" > /dev/null 2>&1 || (echo "Invalid JSON: {}" && exit 1)' \;

.PHONY: check/worker-safe
check/worker-safe: ## Fail if node code reaches an engine manager it cannot have in a worker.
	@uv run python scripts/check_worker_safe_managers.py

.PHONY: check/edit-time-imports
check/edit-time-imports: ## Fail if building the node classes reaches an execution-set package.
	@# Deliberately the edit-time venv: reaching a heavy package has to fail here the way it would on a
	@# real orchestrator, which is the one environment where the packages are absent.
	@uv run python scripts/check_edit_time_imports.py

.PHONY: test
test: ## Run all tests.
	@uv run pytest tests

.PHONY: test/unit
test/unit: ## Run unit tests (everything except workflow tests).
	@uv run pytest tests --ignore=tests/workflows

.PHONY: test/exec
test/exec: ## Run the tests that need the execution environment (diffusers, torch).
	@# A separate venv on purpose: .venv is the edit-time environment the engine splices onto the
	@# orchestrator, and fattening it would hide exactly the bugs that separation exists to expose.
	@UV_PROJECT_ENVIRONMENT=$(EXEC_TEST_VENV) uv sync --extra exec --all-groups
	@UV_PROJECT_ENVIRONMENT=$(EXEC_TEST_VENV) uv run pytest tests --ignore=tests/workflows

.PHONY: test/workflows
test/workflows: ## Run workflow tests.
	@uv run pytest -s tests/workflows

.DEFAULT_GOAL := help
.PHONY: help
help: ## Print Makefile help text.
	@# Matches targets with a comment in the format <target>: ## <comment>
	@# then formats help output using these values.
	@grep -E '^[a-zA-Z_\/-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| awk 'BEGIN {FS = ":.*?## "}; \
		{printf "\033[36m%-12s\033[0m%s\n", $$1, $$2}'
