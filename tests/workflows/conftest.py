import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import AsyncGenerator, Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY
from griptape_nodes.utils import install_file_url_support

from tests.preflight import WORKFLOW_TESTS_PRESENT_KEY, get_preflight_data
from tests.workflows.workflow_configs import ParamOverride, WorkflowConfig

logger = logging.getLogger(__name__)

# Install file:// URL support for httpx/requests in tests.
install_file_url_support()

LIBRARY_ROOT = Path(__file__).parents[2]
WORKFLOW_ASSETS_DIR = LIBRARY_ROOT / "workflows" / "assets"
LEGACY_ASSETS_ALIAS = Path("/workspace/workflows/assets")

load_dotenv()


class ConfigurableWorkflowExecutor(LocalWorkflowExecutor):
    """Executor that applies a config's parameter overrides after load, before the flow runs.

    Overrides arrive per run via the `parameter_overrides` kwarg that `arun` forwards into
    `aprepare_workflow_for_run`, so a single session-scoped instance can drive every config.
    """

    async def aprepare_workflow_for_run(self, flow_input: Any, **kwargs: Any) -> str:
        overrides: tuple[ParamOverride, ...] = kwargs.pop("parameter_overrides", ())
        flow_name = await super().aprepare_workflow_for_run(flow_input, **kwargs)
        for override in overrides:
            # initial_setup=False so the builder node rebuilds dependent params (pipeline_type, model).
            result = await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name=override.parameter_name,
                    node_name=override.node_name,
                    value=override.value,
                    initial_setup=False,
                )
            )
            if result.failed():
                msg = (
                    f"Attempted to override parameter '{override.parameter_name}' on node "
                    f"'{override.node_name}' with value {override.value!r}. Failed with result: {result}."
                )
                raise RuntimeError(msg)
        return flow_name


def _link_directory(link: Path, target: Path) -> None:
    """Point `link` at `target` without duplicating its contents.

    Tries a real symlink first. On Windows without Developer Mode/symlink
    privilege, falls back to an NTFS junction (`mklink /J`), which requires no
    special privilege and, unlike copying, stays in sync with `target` and
    doesn't duplicate large binary assets on disk.
    """
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as e:
        if os.name == "nt" and e.winerror == 1314:  # WinError: A required privilege is not held
            subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                check=True,
                capture_output=True,
                text=True,
            )
        else:
            raise


def _unlink_directory(link: Path) -> None:
    """Remove a directory link created by `_link_directory` without touching its target.

    Handles both real symlinks and NTFS junctions. `Path.is_symlink()` doesn't
    recognize junctions (they're a distinct reparse point type), so junctions
    are removed via `rmdir`, which detaches the reparse point without
    recursing into the target. Falls back to `rmtree` for plain directories
    (e.g. ones populated by a pre-existing copy).
    """
    if link.is_symlink():
        link.unlink(missing_ok=True)
        return
    try:
        link.rmdir()
    except OSError:
        shutil.rmtree(link, ignore_errors=True)


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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Deselect (not skip) tests whose required repos are missing so they disappear
    # from the active run set entirely.
    preflight_data = get_preflight_data()
    missing_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_by_workflow"]
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_libraries_by_workflow"]
    missing_loras_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_loras_by_workflow"]
    repo_available: dict[str, bool] = preflight_data["repo_available"]
    kept_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    workflow_tests_present = False

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            kept_items.append(item)
            continue
        if "workflow_path" not in callspec.params:
            kept_items.append(item)
            continue

        workflow_tests_present = True
        workflow_path = Path(str(callspec.params["workflow_path"]))
        workflow_name = workflow_path.name
        missing_libraries = missing_libraries_by_workflow.get(workflow_name, ())
        missing_loras = missing_loras_by_workflow.get(workflow_name, ())

        # A config that declares its own repos is checked against those; otherwise fall back to
        # the repos statically extracted from the template as-shipped.
        config_param = callspec.params.get("config")
        if isinstance(config_param, WorkflowConfig) and config_param.repos:
            missing_repos = tuple(repo for repo in config_param.repos if not repo_available.get(repo, False))
        else:
            missing_repos = missing_by_workflow.get(workflow_name, ())

        if not missing_repos and not missing_libraries and not missing_loras:
            kept_items.append(item)
            continue

        # Keep reasoning in preflight data/reporting; collection here just filters.
        deselected_items.append(item)

    config.stash[WORKFLOW_TESTS_PRESENT_KEY] = workflow_tests_present

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = kept_items


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


@pytest.fixture(scope="session")
def workflow_runs_root(request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    """Base directory holding per-test workflow run folders for this session.

    Defaults to a fresh OS temp directory; pass --workflow-runs-dir to use a
    specific location instead. A `workflows` symlink is created here so templates
    using the `{workspace_dir}/../workflows/assets/...}` macro keep resolving
    correctly regardless of where the run folder actually lives.
    """
    configured_dir = request.config.getoption("--workflow-runs-dir")
    if configured_dir:
        root = Path(configured_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        owns_root = False
    else:
        root = Path(tempfile.mkdtemp(prefix="griptape-workflow-runs-"))
        owns_root = True

    workflows_link = root / "workflows"
    if not workflows_link.exists():
        _link_directory(workflows_link, LIBRARY_ROOT / "workflows")

    yield root

    if request.config.getoption("--no-cleanup"):
        logger.info("Preserved workflow runs root: %s", root)
        return

    _unlink_directory(workflows_link)
    if owns_root:
        shutil.rmtree(root, ignore_errors=True)
    else:
        try:
            root.rmdir()
        except OSError:
            # Keep a user-specified directory if it still has other content.
            pass


@pytest.fixture(autouse=True)
def workflow_run_workspace(
    request: pytest.FixtureRequest, griptape_nodes: GriptapeNodes
) -> Generator[None, None, None]:
    """Use a unique workspace path for each workflow-execution test and clean it up by default.

    Note: this only affects `{workspace_dir}`-relative saves (e.g. failure snapshots). Node-generated
    output resolves via the `{workflow_dir}` macro, which is keyed off the running workflow's
    WorkflowRegistry entry rather than the configured workspace, so this fixture does not isolate
    node output into the run directory.
    """
    callspec = getattr(request.node, "callspec", None)
    if callspec is None or "workflow_path" not in callspec.params:
        yield
        return

    workflow_runs_root: Path = request.getfixturevalue("workflow_runs_root")
    run_directory = workflow_runs_root / _build_run_directory_name(request.node)
    run_directory.mkdir(parents=True, exist_ok=False)

    # Some templates reference assets via `{workspace_dir}/libraries/<repo>/workflows/assets/...`,
    # assuming `{workspace_dir}` is the outer GriptapeNodes workspace. Bridge that convention too.
    assets_link = run_directory / "libraries" / LIBRARY_ROOT.name / "workflows" / "assets"
    assets_link.parent.mkdir(parents=True, exist_ok=True)
    _link_directory(assets_link, WORKFLOW_ASSETS_DIR)

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

    # Detach the assets link before removing the tree: `shutil.rmtree` can't
    # tell a junction apart from a real directory, so it would happily recurse
    # into it and delete the real assets it points at.
    _unlink_directory(assets_link)
    shutil.rmtree(run_directory, ignore_errors=True)


@pytest_asyncio.fixture(scope="session")
async def workflow_executor() -> AsyncGenerator[ConfigurableWorkflowExecutor, Any]:
    """Create and manage a single ConfigurableWorkflowExecutor for all tests."""
    async with ConfigurableWorkflowExecutor() as executor:
        yield executor


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_library(griptape_nodes: GriptapeNodes) -> AsyncGenerator[None, Any]:
    """Set up this library for testing and restore original state afterwards."""
    config_manager = griptape_nodes.ConfigManager()

    # Save the original libraries state.
    original_libraries = config_manager.get_config_value(key=LIBRARIES_TO_REGISTER_KEY, default=[])

    preflight_data = get_preflight_data()
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
