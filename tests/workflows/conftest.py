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
from griptape_nodes.retained_mode.engine import Engine
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, DeleteConnectionRequest
from griptape_nodes.retained_mode.events.node_events import DeleteNodeRequest
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.settings import LIBRARIES_TO_REGISTER_KEY
from griptape_nodes.utils import install_file_url_support

from tests.preflight import (
    WORKFLOW_SELECTED_KEYS_KEY,
    WORKFLOW_TESTS_PRESENT_KEY,
    compute_config_skip_reasons,
    get_preflight_data,
)
from tests.workflows.workflow_configs import (
    ConnectOverride,
    DeleteNodeOverride,
    DisconnectOverride,
    ParamOverride,
    WorkflowConfig,
    WorkflowOverride,
)

logger = logging.getLogger(__name__)

# Install file:// URL support for httpx/requests in tests.
install_file_url_support()

LIBRARY_ROOT = Path(__file__).parents[2]
WORKFLOW_ASSETS_DIR = LIBRARY_ROOT / "workflows" / "assets"
AUTO_RESIZE_CONFIG_KEY = "modular_diffusion_library.enable_auto_resize"

load_dotenv()


class ConfigurableWorkflowExecutor(LocalWorkflowExecutor):
    """Executor that applies a config's parameter overrides after load, before the flow runs.

    Overrides arrive per run via the `parameter_overrides` kwarg that `arun` forwards into
    `aprepare_workflow_for_run`, so a single session-scoped instance can drive every config.
    """

    async def aprepare_workflow_for_run(self, flow_input: Any, **kwargs: Any) -> str:
        overrides: tuple[WorkflowOverride, ...] = kwargs.pop("parameter_overrides", ())
        flow_name = await super().aprepare_workflow_for_run(flow_input, **kwargs)
        for override in overrides:
            await self._apply_override(override)
        return flow_name

    @staticmethod
    async def _apply_override(override: WorkflowOverride) -> None:
        match override:
            case ParamOverride():
                # initial_setup=False so the builder node rebuilds dependent params (pipeline_type, model).
                request = SetParameterValueRequest(
                    parameter_name=override.parameter_name,
                    node_name=override.node_name,
                    value=override.value,
                    initial_setup=False,
                )
                description = (
                    f"override parameter '{override.parameter_name}' on node "
                    f"'{override.node_name}' with value {override.value!r}"
                )
            case DisconnectOverride():
                request = DeleteConnectionRequest(
                    source_node_name=override.source_node_name,
                    source_parameter_name=override.source_parameter_name,
                    target_node_name=override.target_node_name,
                    target_parameter_name=override.target_parameter_name,
                )
                description = (
                    f"disconnect '{override.source_node_name}.{override.source_parameter_name}' from "
                    f"'{override.target_node_name}.{override.target_parameter_name}'"
                )
            case ConnectOverride():
                request = CreateConnectionRequest(
                    source_node_name=override.source_node_name,
                    source_parameter_name=override.source_parameter_name,
                    target_node_name=override.target_node_name,
                    target_parameter_name=override.target_parameter_name,
                    initial_setup=False,
                )
                description = (
                    f"connect '{override.source_node_name}.{override.source_parameter_name}' to "
                    f"'{override.target_node_name}.{override.target_parameter_name}'"
                )
            case DeleteNodeOverride():
                request = DeleteNodeRequest(node_name=override.node_name)
                description = f"delete node '{override.node_name}'"
            case _:
                msg = f"Unknown workflow override type: {type(override).__name__}"
                raise TypeError(msg)

        result = await GriptapeNodes.ahandle_request(request)
        if result.failed():
            msg = f"Attempted to {description}. Failed with result: {result}."
            raise RuntimeError(msg)


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


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # Deselect (not skip) tests with missing repos/libraries/LoRAs/paths. Runs trylast (after -k/-m)
    # and records which (workflow, config) keys survived, so the preflight report can scope to them.
    preflight_data = get_preflight_data()
    kept_items: list[pytest.Item] = []
    deselected_items: list[pytest.Item] = []
    workflow_tests_present = False
    selected_keys: set[tuple[str, str]] = set()

    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec is None:
            kept_items.append(item)
            continue
        if "workflow_path" not in callspec.params:
            kept_items.append(item)
            continue

        workflow_tests_present = True
        workflow_file_name = Path(str(callspec.params["workflow_path"])).name
        config_param = callspec.params.get("config")
        config_obj = config_param if isinstance(config_param, WorkflowConfig) else None
        config_id = config_obj.config_id if config_obj is not None else "default"
        selected_keys.add((workflow_file_name, config_id))
        reasons = compute_config_skip_reasons(workflow_file_name, config_obj, preflight_data)

        # Keep detailed reasoning in the preflight report; collection here just filters.
        if reasons.is_skipped:
            deselected_items.append(item)
        else:
            kept_items.append(item)

    config.stash[WORKFLOW_TESTS_PRESENT_KEY] = workflow_tests_present
    config.stash[WORKFLOW_SELECTED_KEYS_KEY] = selected_keys

    if deselected_items:
        config.hook.pytest_deselected(items=deselected_items)
    items[:] = kept_items


@pytest.fixture(scope="session")
def griptape_nodes() -> Engine:
    """Initialize GriptapeNodes before tests and clean up afterwards."""
    return GriptapeNodes()


@pytest.fixture(scope="session")
def workflow_runs_root(request: pytest.FixtureRequest) -> Generator[Path, None, None]:
    """Base directory holding per-test workflow run folders for this session.

    Defaults to a fresh OS temp directory; pass --workflow-runs-dir to use a
    specific location instead.
    """
    configured_dir = request.config.getoption("--workflow-runs-dir")
    if configured_dir:
        root = Path(configured_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        owns_root = False
    else:
        root = Path(tempfile.mkdtemp(prefix="griptape-workflow-runs-"))
        owns_root = True

    yield root

    if request.config.getoption("--no-cleanup"):
        logger.info("Preserved workflow runs root: %s", root)
        return

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
async def workflow_executor(setup_test_library: None) -> AsyncGenerator[ConfigurableWorkflowExecutor, Any]:
    """Create and manage a single ConfigurableWorkflowExecutor for all tests."""
    async with ConfigurableWorkflowExecutor() as executor:
        yield executor


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_test_library(griptape_nodes: Engine) -> AsyncGenerator[None, Any]:
    """Set up this library for testing and restore original state afterwards."""
    config_manager = griptape_nodes.ConfigManager()

    # Save the original libraries state.
    original_libraries = config_manager.get_config_value(key=LIBRARIES_TO_REGISTER_KEY, default=[])

    manifest_paths = [
        LIBRARY_ROOT / "griptape-nodes-library.json",
        LIBRARY_ROOT.parent / "griptape-nodes-library-standard" / "griptape_nodes_library.json",
    ]
    missing_paths = [str(path) for path in manifest_paths if not path.is_file()]
    if missing_paths:
        pytest.fail(f"Workflow tests require diffusers and sibling standard library manifests: {missing_paths}")

    config_manager.set_config_value(
        key=LIBRARIES_TO_REGISTER_KEY,
        value=[str(path) for path in manifest_paths],
    )
    try:
        yield
    finally:
        config_manager.set_config_value(
            key=LIBRARIES_TO_REGISTER_KEY,
            value=original_libraries,
        )


@pytest_asyncio.fixture(autouse=True)
async def clear_state_before_each_test(griptape_nodes: Engine) -> AsyncGenerator[None, Any]:
    """Clear all object state before each test to ensure clean starting conditions."""
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)

    griptape_nodes.ConfigManager()._set_log_level("DEBUG")

    yield

    # Clean up after test.
    clear_request = ClearAllObjectStateRequest(i_know_what_im_doing=True)
    await griptape_nodes.ahandle_request(clear_request)


@pytest.fixture(autouse=True)
def workflow_config_settings(
    request: pytest.FixtureRequest, griptape_nodes: GriptapeNodes
) -> Generator[None, None, None]:
    callspec = getattr(request.node, "callspec", None)
    if callspec is None:
        yield
        return

    config_param = callspec.params.get("config")
    config_obj = config_param if isinstance(config_param, WorkflowConfig) else None
    if config_obj is None or config_obj.enable_auto_resize is None:
        yield
        return

    config_manager = griptape_nodes.ConfigManager()
    previous_auto_resize = config_manager.get_config_value(AUTO_RESIZE_CONFIG_KEY)
    config_manager.set_config_value(AUTO_RESIZE_CONFIG_KEY, config_obj.enable_auto_resize)

    yield

    config_manager.set_config_value(AUTO_RESIZE_CONFIG_KEY, previous_auto_resize)
