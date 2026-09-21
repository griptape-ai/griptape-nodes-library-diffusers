import json
import logging
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.retained_mode.managers.config_manager import USER_CONFIG_PATH
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY
from griptape_nodes.utils.dict_utils import get_dot_value

from griptape_nodes.exe_types.param_components.huggingface.huggingface_utils import list_repo_revisions_in_cache
from tests.workflows.dependencies import ExtractionBlocker, WorkflowDependencies, extract_workflow_dependencies

logger = logging.getLogger(__name__)

LIBRARY_ROOT = Path(__file__).parents[1]
WORKFLOW_TEMPLATES_DIR = LIBRARY_ROOT / "workflows" / "templates"

# Set by tests/workflows/conftest.py's pytest_collection_modifyitems once it knows whether any
# workflow-parametrized tests were actually collected this session (before preflight deselection).
# tests/conftest.py reads it to decide whether preflight reporting is relevant to this invocation.
WORKFLOW_TESTS_PRESENT_KEY: pytest.StashKey[bool] = pytest.StashKey()

_PREFLIGHT_DATA: dict[str, Any] | None = None


def _discover_workflow_templates() -> list[str]:
    # Only actual templates should become test cases.
    templates = [
        f.name
        for f in WORKFLOW_TEMPLATES_DIR.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]
    return sorted(templates)


def _read_library_name_from_manifest(manifest_path: Path) -> str | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read library manifest: %s", manifest_path)
        return None

    library_name = manifest.get("name")
    if not isinstance(library_name, str):
        logger.warning("Library manifest missing string 'name': %s", manifest_path)
        return None

    return library_name


def _discover_installed_library_manifests() -> dict[str, Path]:
    """Find libraries the griptape_nodes engine has registered, keyed by display name."""
    manifests_by_name: dict[str, Path] = {}
    if not USER_CONFIG_PATH.is_file():
        return manifests_by_name

    try:
        user_config = json.loads(USER_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read engine config: %s", USER_CONFIG_PATH)
        return manifests_by_name

    registered_paths = get_dot_value(user_config, LIBRARIES_TO_REGISTER_KEY, default=[])
    for manifest_path_string in registered_paths:
        manifest_path = Path(manifest_path_string)
        if not manifest_path.is_file():
            continue

        library_name = _read_library_name_from_manifest(manifest_path)
        if library_name is None:
            continue
        manifests_by_name.setdefault(library_name, manifest_path)

    return manifests_by_name


def _build_preflight_data() -> dict[str, Any]:
    # Build once per pytest session: discover templates -> extract canonical dependencies once ->
    # resolve what is locally cached -> compute missing/runnable sets.
    discovered_workflows = _discover_workflow_templates()

    extraction_results: dict[str, WorkflowDependencies] = {}
    for workflow_name in discovered_workflows:
        workflow_path = WORKFLOW_TEMPLATES_DIR / workflow_name
        extraction_results[workflow_name] = extract_workflow_dependencies(workflow_path)

    workflow_required_libraries: dict[str, tuple[str, ...]] = {
        workflow_name: result.library_names for workflow_name, result in extraction_results.items()
    }

    installed_library_manifests = _discover_installed_library_manifests()

    workflow_required_repos: dict[str, tuple[str, ...]] = {}
    workflow_extraction_blockers: dict[str, tuple[ExtractionBlocker, ...]] = {}
    workflows_with_no_repos: list[str] = []
    for workflow_name, result in extraction_results.items():
        required_repos = tuple(dependency.repo_id for dependency in result.model_dependencies)
        workflow_required_repos[workflow_name] = required_repos
        if not result.conclusive:
            workflow_extraction_blockers[workflow_name] = result.blockers
            continue
        if not required_repos:
            workflows_with_no_repos.append(workflow_name)

    unique_required_repos = {repo for repos in workflow_required_repos.values() for repo in repos}
    repo_available: dict[str, bool] = {}
    for repo_id in sorted(unique_required_repos):
        repo_available[repo_id] = bool(list_repo_revisions_in_cache(repo_id))

    missing_by_workflow: dict[str, tuple[str, ...]] = {}
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = {}
    for workflow_name in discovered_workflows:
        required = workflow_required_repos.get(workflow_name, ())
        missing = tuple(repo for repo in required if not repo_available.get(repo, False))
        missing_by_workflow[workflow_name] = missing

        required_libraries = workflow_required_libraries.get(workflow_name, ())
        missing_libraries = tuple(
            library_name for library_name in required_libraries if library_name not in installed_library_manifests
        )
        missing_libraries_by_workflow[workflow_name] = missing_libraries

    skipped_workflows = sorted(
        [
            workflow_name
            for workflow_name in discovered_workflows
            if missing_by_workflow[workflow_name] or missing_libraries_by_workflow[workflow_name]
        ]
    )
    runnable_workflows = sorted(
        [
            workflow_name
            for workflow_name in discovered_workflows
            if not missing_by_workflow[workflow_name] and not missing_libraries_by_workflow[workflow_name]
        ]
    )

    return {
        "discovered_workflows": discovered_workflows,
        "workflow_required_libraries": workflow_required_libraries,
        "installed_library_manifests": installed_library_manifests,
        "workflow_required_repos": workflow_required_repos,
        "workflow_extraction_blockers": workflow_extraction_blockers,
        "workflows_with_no_repos": sorted(workflows_with_no_repos),
        "repo_available": repo_available,
        "missing_by_workflow": missing_by_workflow,
        "missing_libraries_by_workflow": missing_libraries_by_workflow,
        "skipped_workflows": skipped_workflows,
        "runnable_workflows": runnable_workflows,
    }


def get_preflight_data() -> dict[str, Any]:
    """Compute (once per session) and return the workflow preflight data."""
    global _PREFLIGHT_DATA
    if _PREFLIGHT_DATA is None:
        _PREFLIGHT_DATA = _build_preflight_data()
    return _PREFLIGHT_DATA
