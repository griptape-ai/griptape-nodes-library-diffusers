import ast
import json
import logging
import os
import pickle
import re
import shutil
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY
from griptape_nodes.utils import install_file_url_support
from modular_diffusion_nodes_library.utils.huggingface_utils import list_repo_revisions_in_cache

logger = logging.getLogger(__name__)

# Install file:// URL support for httpx/requests in tests.
install_file_url_support()

LIBRARY_ROOT = Path(__file__).parents[2]
WORKFLOW_TEMPLATES_DIR = LIBRARY_ROOT / "workflows" / "templates"
WORKFLOW_RUNS_DIR = LIBRARY_ROOT / ".pytest-workflow-runs"
WORKFLOW_ASSETS_DIR = LIBRARY_ROOT / "workflows" / "assets"
LEGACY_ASSETS_ALIAS = Path("/workspace/workflows/assets")
LIBRARY_MANIFEST_FILENAMES = ("griptape-nodes-library.json", "griptape_nodes_library.json")

REPO_ID_PATTERN = re.compile(r"\b[\w.-]+/[\w.-]+\b")
REPO_PARAMETERS = {"model", "controlnet_model", "upsampler_model"}

# Intentionally excludes text encoders to avoid overskipping for now.
IGNORED_PARAMETERS = {"text_encoder", "text_encoder_2"}

_PREFLIGHT_DATA: dict[str, Any] | None = None

load_dotenv()


def _discover_workflow_templates() -> list[str]:
    # Only actual templates should become test cases.
    templates = [
        f.name
        for f in WORKFLOW_TEMPLATES_DIR.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]
    return sorted(templates)


def _extract_repo_strings(value: Any) -> set[str]:
    # We only treat exact repo-id shaped strings as model requirements.
    repos: set[str] = set()
    if isinstance(value, str):
        if REPO_ID_PATTERN.fullmatch(value):
            repos.add(value)
        return repos
    if isinstance(value, list):
        for element in value:
            repos.update(_extract_repo_strings(element))
    return repos


def _extract_repos_from_call(call: ast.Call) -> tuple[str | None, set[str]]:
    # Templates are code-generated, but model selections are usually expressed through
    # AddParameterToNodeRequest(...). We extract those deterministic selections here.
    if not isinstance(call.func, ast.Name):
        return None, set()
    if call.func.id != "AddParameterToNodeRequest":
        return None, set()

    parameter_name: str | None = None
    repos: set[str] = set()

    for keyword in call.keywords:
        if keyword.arg == "parameter_name" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                parameter_name = keyword.value.value
        if keyword.arg == "default_value" and isinstance(keyword.value, ast.Constant):
            repos.update(_extract_repo_strings(keyword.value.value))
        if keyword.arg == "ui_options" and isinstance(keyword.value, ast.Dict):
            for key_node, value_node in zip(keyword.value.keys, keyword.value.values, strict=False):
                if not isinstance(key_node, ast.Constant):
                    continue
                if key_node.value != "simple_dropdown":
                    continue
                if isinstance(value_node, ast.List):
                    list_values = [
                        elt.value for elt in value_node.elts if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
                    ]
                    repos.update(_extract_repo_strings(list_values))

    return parameter_name, repos


def _extract_required_libraries_from_template(workflow_path: Path) -> tuple[str, ...]:
    """Extract library display names required by CreateNodeRequest calls."""
    try:
        source = workflow_path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to read workflow template for library preflight: %s", workflow_path)
        return tuple()

    try:
        module = ast.parse(source)
    except SyntaxError:
        logger.exception("Failed to parse workflow template for library preflight: %s", workflow_path)
        return tuple()

    required_libraries: set[str] = set()
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "CreateNodeRequest":
            continue
        for keyword in node.keywords:
            if keyword.arg != "specific_library_name":
                continue
            if not isinstance(keyword.value, ast.Constant):
                continue
            if not isinstance(keyword.value.value, str):
                continue
            required_libraries.add(keyword.value.value)

    return tuple(sorted(required_libraries))


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
    """Find installed library manifests in this libraries workspace by display name."""
    manifests_by_name: dict[str, Path] = {}
    libraries_root = LIBRARY_ROOT.parent

    for candidate in libraries_root.iterdir():
        if not candidate.is_dir():
            continue

        for manifest_filename in LIBRARY_MANIFEST_FILENAMES:
            manifest_path = candidate / manifest_filename
            if not manifest_path.is_file():
                continue

            library_name = _read_library_name_from_manifest(manifest_path)
            if library_name is None:
                continue
            manifests_by_name.setdefault(library_name, manifest_path)

    return manifests_by_name


def _extract_pickled_string_table(module: ast.Module) -> dict[str, str]:
    """Map top_level_unique_values_dict keys to pickled string payloads when possible."""
    pickled_strings: dict[str, str] = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id != "top_level_unique_values_dict":
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        for key_node, value_node in zip(node.value.keys, node.value.values, strict=False):
            if not isinstance(key_node, ast.Constant):
                continue
            if not isinstance(key_node.value, str):
                continue
            if not isinstance(value_node, ast.Call):
                continue
            if not isinstance(value_node.func, ast.Attribute):
                continue
            if not isinstance(value_node.func.value, ast.Name):
                continue
            if value_node.func.value.id != "pickle" or value_node.func.attr != "loads":
                continue
            if len(value_node.args) != 1:
                continue

            payload_node = value_node.args[0]
            if not isinstance(payload_node, ast.Constant):
                continue
            if not isinstance(payload_node.value, bytes):
                continue

            try:
                unpickled = pickle.loads(payload_node.value)
            except Exception:
                continue
            if isinstance(unpickled, str):
                pickled_strings[key_node.value] = unpickled

    return pickled_strings


def _extract_repos_from_set_parameter_call(call: ast.Call, pickled_strings: dict[str, str]) -> set[str]:
    """Extract repos from SetParameterValueRequest calls for repo-bearing parameters."""
    if not isinstance(call.func, ast.Name):
        return set()
    if call.func.id != "SetParameterValueRequest":
        return set()

    parameter_name: str | None = None
    value_key: str | None = None
    repos: set[str] = set()

    for keyword in call.keywords:
        if keyword.arg == "parameter_name" and isinstance(keyword.value, ast.Constant):
            if isinstance(keyword.value.value, str):
                parameter_name = keyword.value.value
        if keyword.arg != "value":
            continue
        if isinstance(keyword.value, ast.Constant):
            repos.update(_extract_repo_strings(keyword.value.value))
            continue
        if not isinstance(keyword.value, ast.Subscript):
            continue
        if not isinstance(keyword.value.value, ast.Name):
            continue
        if keyword.value.value.id != "top_level_unique_values_dict":
            continue
        key_node = keyword.value.slice
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            value_key = key_node.value

    if parameter_name in IGNORED_PARAMETERS:
        return set()
    if parameter_name not in REPO_PARAMETERS:
        return set()
    if value_key is None:
        return repos

    resolved_value = pickled_strings.get(value_key)
    if resolved_value is None:
        return repos

    repos.update(_extract_repo_strings(resolved_value))
    return repos


def _extract_repos_from_template(workflow_path: Path) -> tuple[str, ...]:
    """Extract required repos from one workflow template.

    The extractor intentionally focuses on explicit node-parameter selections used by
    templates (default values and dropdown values). This keeps preflight predictable
    and avoids broad heuristics that can over-match unrelated strings.
    """
    try:
        source = workflow_path.read_text(encoding="utf-8")
    except OSError:
        logger.exception("Failed to read workflow template for preflight: %s", workflow_path)
        return tuple()

    repos: set[str] = set()

    try:
        module = ast.parse(source)
    except SyntaxError:
        logger.exception("Failed to parse workflow template for preflight: %s", workflow_path)
        return tuple()

    pickled_strings = _extract_pickled_string_table(module)

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue

        repos.update(_extract_repos_from_set_parameter_call(node, pickled_strings))

        parameter_name, extracted_repos = _extract_repos_from_call(node)
        if parameter_name is None:
            continue
        if parameter_name in IGNORED_PARAMETERS:
            continue
        if parameter_name not in REPO_PARAMETERS:
            continue
        # Only keep repos from parameters that represent model-bearing selections.
        repos.update(extracted_repos)

    return tuple(sorted(repos))


def _build_preflight_data() -> dict[str, Any]:
    # Build once per pytest session: discover templates -> derive required repos ->
    # resolve what is locally cached -> compute missing/runnable sets.
    discovered_workflows = _discover_workflow_templates()

    workflow_required_libraries: dict[str, tuple[str, ...]] = {}
    for workflow_name in discovered_workflows:
        workflow_path = WORKFLOW_TEMPLATES_DIR / workflow_name
        workflow_required_libraries[workflow_name] = _extract_required_libraries_from_template(workflow_path)

    installed_library_manifests = _discover_installed_library_manifests()

    workflow_required_repos: dict[str, tuple[str, ...]] = {}
    workflows_with_no_repos: list[str] = []
    for workflow_name in discovered_workflows:
        workflow_path = WORKFLOW_TEMPLATES_DIR / workflow_name
        extracted = _extract_repos_from_template(workflow_path)
        workflow_required_repos[workflow_name] = extracted
        if not extracted:
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
        "workflows_with_no_repos": sorted(workflows_with_no_repos),
        "repo_available": repo_available,
        "missing_by_workflow": missing_by_workflow,
        "missing_libraries_by_workflow": missing_libraries_by_workflow,
        "skipped_workflows": skipped_workflows,
        "runnable_workflows": runnable_workflows,
    }


def _get_preflight_data() -> dict[str, Any]:
    global _PREFLIGHT_DATA
    if _PREFLIGHT_DATA is None:
        _PREFLIGHT_DATA = _build_preflight_data()
    return _PREFLIGHT_DATA


def _get_test_workflow_name(item: pytest.Item) -> str:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return item.name

    workflow_param = callspec.params.get("workflow_path")
    if workflow_param is None:
        return item.name

    return Path(str(workflow_param)).stem


def _build_run_directory_name(item: pytest.Item) -> str:
    workflow_name = _get_test_workflow_name(item)
    workflow_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", workflow_name).strip("-")
    if not workflow_slug:
        workflow_slug = "workflow"
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{workflow_slug}"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Keep per-test workflow run folders instead of deleting them after each test.",
    )
    parser.addoption(
        "--preflight-strict",
        action="store_true",
        default=False,
        help="Fail test collection if any workflow is missing required cached model repos.",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # Strict mode fails fast if any workflow would be deselected by preflight.
    if not session.config.getoption("--preflight-strict"):
        return

    preflight_data = _get_preflight_data()
    skipped_workflows: list[str] = preflight_data["skipped_workflows"]
    if not skipped_workflows:
        return

    missing_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_by_workflow"]
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_libraries_by_workflow"]
    missing_repo_lines = []
    missing_library_lines = []
    for workflow_name in skipped_workflows:
        missing_repos = missing_by_workflow.get(workflow_name, ())
        if missing_repos:
            missing_repo_lines.append(f"  - {workflow_name}: {', '.join(missing_repos)}")

        missing_libraries = missing_libraries_by_workflow.get(workflow_name, ())
        if missing_libraries:
            missing_library_lines.append(f"  - {workflow_name}: {', '.join(missing_libraries)}")

    details_sections: list[str] = []
    if missing_repo_lines:
        details_sections.append("Missing cached model repos:\n" + "\n".join(missing_repo_lines))
    if missing_library_lines:
        details_sections.append("Missing installed node libraries:\n" + "\n".join(missing_library_lines))

    details = "\n\n".join(details_sections)
    raise pytest.UsageError(
        "Workflow preflight strict mode failed.\n"
        f"{details}\n"
        "Disable strict mode to run only cached workflows (default deselection behavior)."
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Deselect (not skip) tests whose required repos are missing so they disappear
    # from the active run set entirely.
    preflight_data = _get_preflight_data()
    missing_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_by_workflow"]
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_libraries_by_workflow"]
    kept_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            kept_items.append(item)
            continue
        if "workflow_path" not in callspec.params:
            kept_items.append(item)
            continue

        workflow_path = Path(str(callspec.params["workflow_path"]))
        workflow_name = workflow_path.name
        missing_repos = missing_by_workflow.get(workflow_name, ())
        missing_libraries = missing_libraries_by_workflow.get(workflow_name, ())
        if not missing_repos and not missing_libraries:
            kept_items.append(item)
            continue

        # Keep reasoning in preflight data/reporting; collection here just filters.
        deselected_items.append(item)

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = kept_items


def pytest_report_header(config: pytest.Config) -> list[str]:
    # Make preflight behavior visible in test output so deselection is transparent.
    preflight_data = _get_preflight_data()
    discovered = preflight_data["discovered_workflows"]
    runnable = preflight_data["runnable_workflows"]
    skipped = preflight_data["skipped_workflows"]
    zero_repo = preflight_data["workflows_with_no_repos"]
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_libraries_by_workflow"]

    header = [
        "Workflow preflight:",
        f"  discovered={len(discovered)} runnable={len(runnable)} skipped={len(skipped)}",
    ]

    cleanup_mode = "preserve (--no-cleanup)" if config.getoption("--no-cleanup") else "auto-delete"
    header.append(f"  run-folder cleanup mode: {cleanup_mode}")
    preflight_mode = "strict" if config.getoption("--preflight-strict") else "deselect-missing"
    header.append(f"  preflight mode: {preflight_mode}")

    if skipped:
        header.append(f"  skipped workflows: {', '.join(skipped)}")

    workflows_missing_libraries = [name for name, missing in missing_libraries_by_workflow.items() if missing]
    if workflows_missing_libraries:
        header.append(
            "  workflows missing libraries: "
            f"{', '.join(sorted(workflows_missing_libraries))}"
        )

    if zero_repo:
        header.append(
            "  workflows with no extracted repos: "
            f"{', '.join(zero_repo)}"
        )

    return header


@pytest.fixture(scope="session")
def griptape_nodes() -> GriptapeNodes:
    """Initialize GriptapeNodes before tests and clean up afterwards."""
    return GriptapeNodes()


@pytest.fixture(scope="session", autouse=True)
def legacy_assets_path_bridge() -> None:
    """Provide compatibility for templates using `{project_dir}/../workflows/assets`.

    Some generated templates reference assets through a legacy macro expansion
    path that resolves to `/workspace/workflows/assets` in local test runs.
    Creating a symlink bridge keeps templates immutable while making those
    references resolvable during tests.
    """
    if LEGACY_ASSETS_ALIAS.exists():
        return

    try:
        LEGACY_ASSETS_ALIAS.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_ASSETS_ALIAS.symlink_to(WORKFLOW_ASSETS_DIR, target_is_directory=True)
    except OSError:
        logger.warning(
            "Could not create legacy assets alias '%s' -> '%s'. Templates using project_dir asset macros may fail.",
            LEGACY_ASSETS_ALIAS,
            WORKFLOW_ASSETS_DIR,
        )


@pytest.fixture(autouse=True)
def workflow_run_workspace(
    request: pytest.FixtureRequest, griptape_nodes: GriptapeNodes
) -> Generator[None, None, None]:
    """Use a unique workspace path for each test run and clean it up by default."""
    WORKFLOW_RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_directory = WORKFLOW_RUNS_DIR / _build_run_directory_name(request.node)
    run_directory.mkdir(parents=True, exist_ok=False)

    config_manager = griptape_nodes.ConfigManager()
    previous_workspace_path = config_manager.workspace_path
    previous_workspace_env = os.environ.get("GTN_CONFIG_WORKSPACE_DIRECTORY")

    # Route workflow-generated files to a run-specific workspace.
    os.environ["GTN_CONFIG_WORKSPACE_DIRECTORY"] = str(run_directory)
    config_manager.workspace_path = run_directory

    yield

    config_manager.workspace_path = previous_workspace_path
    if previous_workspace_env is None:
        os.environ.pop("GTN_CONFIG_WORKSPACE_DIRECTORY", None)
    else:
        os.environ["GTN_CONFIG_WORKSPACE_DIRECTORY"] = previous_workspace_env

    if request.config.getoption("--no-cleanup"):
        logger.info("Preserved workflow test run directory: %s", run_directory)
        return

    shutil.rmtree(run_directory, ignore_errors=True)

    try:
        WORKFLOW_RUNS_DIR.rmdir()
    except OSError:
        # Keep the parent folder if another test is still using it.
        pass


@pytest_asyncio.fixture(scope="session")
async def workflow_executor() -> AsyncGenerator[LocalWorkflowExecutor, Any]:
    """Create and manage a single LocalWorkflowExecutor for all tests."""
    async with LocalWorkflowExecutor() as executor:
        yield executor


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_library(griptape_nodes: GriptapeNodes) -> AsyncGenerator[None, Any]:
    """Set up this library for testing and restore original state afterwards."""
    config_manager = griptape_nodes.ConfigManager()

    # Save the original libraries state.
    original_libraries = config_manager.get_config_value(key=LIBRARIES_TO_REGISTER_KEY, default=[])

    preflight_data = _get_preflight_data()
    required_library_names = {
        library_name
        for library_names in preflight_data["workflow_required_libraries"].values()
        for library_name in library_names
    }
    installed_library_manifests: dict[str, Path] = preflight_data["installed_library_manifests"]

    library_paths_to_register: list[str] = [str(LIBRARY_ROOT / "griptape-nodes-library.json")]
    for library_name in sorted(required_library_names):
        manifest_path = installed_library_manifests.get(library_name)
        if manifest_path is None:
            continue
        manifest_path_string = str(manifest_path)
        if manifest_path_string in library_paths_to_register:
            continue
        library_paths_to_register.append(manifest_path_string)

    # Set discovered workflow-required libraries for testing.
    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=library_paths_to_register,
    )

    yield

    # Restore original libraries state.
    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=original_libraries,
    )


@pytest_asyncio.fixture(autouse=True)
async def clear_state_before_each_test(griptape_nodes: GriptapeNodes) -> AsyncGenerator[None, Any]:
    """Clear all object state before each test to ensure clean starting conditions."""
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)

    griptape_nodes.ConfigManager()._set_log_level("DEBUG")

    yield

    # Clean up after test.
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)