from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.param_components.huggingface.huggingface_utils import list_repo_revisions_in_cache

from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros
from tests.workflows.dependencies import (
    COMPONENT_FOLDER_NODE_TYPES,
    ExtractionBlocker,
    PathRequirement,
    WorkflowDependencies,
    extract_workflow_dependencies,
)
from tests.workflows.workflow_configs import WorkflowConfig, config_repos

LIBRARY_ROOT = Path(__file__).parents[1]
WORKFLOW_TEMPLATES_DIR = LIBRARY_ROOT / "workflows" / "templates"
WORKFLOW_ASSETS_DIR = LIBRARY_ROOT / "workflows" / "assets"
LORA_ASSETS_DIR = WORKFLOW_ASSETS_DIR / "lora"

# Set by tests/workflows/conftest.py's pytest_collection_modifyitems once it knows whether any
# workflow-parametrized tests were actually collected this session (before preflight deselection).
# tests/conftest.py reads it to decide whether preflight reporting is relevant to this invocation.
WORKFLOW_TESTS_PRESENT_KEY: pytest.StashKey[bool] = pytest.StashKey()

# Set by the same hook (which runs trylast, after -k/-m selection): the (workflow_file_name, config_id)
# keys that survived keyword/marker filtering, so tests/conftest.py can scope its preflight report to
# exactly the tests this invocation targets instead of the whole matrix.
WORKFLOW_SELECTED_KEYS_KEY: pytest.StashKey[set[tuple[str, str]]] = pytest.StashKey()

_PREFLIGHT_DATA: dict[str, Any] | None = None

# Keep the LoRA validation narrow: only a small set of model-file extensions are treated as
# legitimate LoRA assets, and they must live under the repository-local workflow assets folder.
SUPPORTED_LORA_EXTENSIONS = {".safetensors", ".sft", ".pt", ".bin", ".json", ".lora"}


def _discover_workflow_templates() -> list[str]:
    # Only actual templates should become test cases.
    templates = [
        f.name
        for f in WORKFLOW_TEMPLATES_DIR.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__")
    ]
    return sorted(templates)


def _resolve_workflow_path_locally(path_string: str) -> list[Path]:
    raw_path = path_string.strip()
    if not raw_path:
        return []

    # Expand the project/workflow macros before checking the path. These are the values used by
    # generated workflow templates when they point at repo-local assets.
    normalized = raw_path.replace("{project_dir}", str(LIBRARY_ROOT)).replace("{workspace_dir}", str(LIBRARY_ROOT))
    normalized = normalized.replace("{workflow_dir}", str(LIBRARY_ROOT / "workflows"))
    normalized = normalized.replace("\\", "/")

    # Repo-relative paths like '../workflows/assets/lora/foo.safetensors' are valid in the workflow
    # context, but they should resolve back inside this library root instead of wandering elsewhere.
    repo_root_prefix = str(LIBRARY_ROOT).replace("\\", "/")
    while normalized.startswith(f"{repo_root_prefix}/../"):
        normalized = f"{repo_root_prefix}/" + normalized[len(f"{repo_root_prefix}/../") :]
    if normalized.startswith("../"):
        normalized = normalized[3:]
    elif normalized.startswith("./"):
        normalized = normalized[2:]

    # Templates also write {workspace_dir} assuming it points at the outer GriptapeNodes workspace
    # (which contains this repo at 'libraries/<repo-name>'), not at this library's own root. Collapse
    # that redundant segment back to the library root so both conventions resolve to the same place.
    redundant_prefix = f"{repo_root_prefix}/libraries/{LIBRARY_ROOT.name}/"
    if normalized.startswith(redundant_prefix):
        normalized = f"{repo_root_prefix}/" + normalized[len(redundant_prefix) :]

    # Try both the fully resolved library-root path and the raw relative form so templates using
    # either style are accepted when they still end up inside the local LoRA assets directory.
    candidates: list[Path] = []
    candidate = Path(normalized)
    try:
        candidate = candidate.expanduser()
    except (OSError, RuntimeError):
        pass

    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(LIBRARY_ROOT / candidate)
        candidates.append(candidate)

    seen: set[Path] = set()
    resolved_paths: list[Path] = []
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=False)
        except (OSError, RuntimeError):
            continue
        if not resolved.exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        resolved_paths.append(resolved)

    return resolved_paths


def _resolve_workflow_path(path_string: str) -> Path | None:
    """Resolve workflow paths locally without depending on the user's engine workspace config."""
    raw_path = path_string.strip()
    if not raw_path:
        return None

    expanded_path = expand_path_macros(raw_path)
    if expanded_path != raw_path:
        try:
            expanded_candidate = Path(expanded_path).expanduser().resolve(strict=False)
            if expanded_candidate.exists():
                return expanded_candidate
        except (OSError, RuntimeError):
            pass

    return next(iter(_resolve_workflow_path_locally(raw_path)), None)


def _resolve_lora_path(path_string: str) -> Path | None:
    """Resolve a LoRA path only when it points to a supported file under the local LoRA assets."""
    lora_assets_root = LORA_ASSETS_DIR.resolve(strict=False)
    local_resolved = _resolve_workflow_path_locally(path_string)
    resolved: Path | None = None
    for candidate in local_resolved:
        try:
            candidate.relative_to(lora_assets_root)
        except ValueError:
            continue
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in SUPPORTED_LORA_EXTENSIONS:
            continue
        resolved = candidate
        break

    return resolved


def _is_lora_path_available(path_string: str) -> bool:
    # A LoRA only counts as available when the path is valid, in-bounds, and readable.
    path = _resolve_lora_path(path_string)
    if path is None:
        return False

    try:
        if path.suffix.lower() in {".safetensors", ".sft"}:
            import safetensors

            with safetensors.safe_open(str(path), framework="pt"):
                pass
    except Exception:
        return False
    return True


def _is_file_path_available(path_string: str) -> bool:
    resolved = _resolve_workflow_path(path_string)
    return resolved is not None and resolved.is_file()


def _is_component_folder_available(path_string: str) -> bool:
    # A diffusers component folder holds a config file: config.json, tokenizer_config.json (tokenizer),
    # or scheduler_config.json (scheduler).
    resolved = _resolve_workflow_path(path_string)
    if resolved is None or not resolved.is_dir():
        return False
    return (
        (resolved / "config.json").is_file()
        or (resolved / "tokenizer_config.json").is_file()
        or (resolved / "scheduler_config.json").is_file()
    )


def _is_folder_available(path_string: str) -> bool:
    # Non-component folders only need to exist on disk.
    resolved = _resolve_workflow_path(path_string)
    return resolved is not None and resolved.is_dir()


def _is_folder_requirement_available(folder: PathRequirement) -> bool:
    # Component/scheduler folders get the diffusers config-file check; any other folder just needs to exist.
    if folder.node_type in COMPONENT_FOLDER_NODE_TYPES:
        return _is_component_folder_available(folder.path)
    return _is_folder_available(folder.path)


def _is_path_requirement_available(requirement: PathRequirement) -> bool:
    # Dispatch by picker kind: 'file' must be a file, 'folder' uses the node-type-aware folder check,
    # and 'both' is satisfied by either.
    if requirement.kind == "file":
        return _is_file_path_available(requirement.path)
    if requirement.kind == "folder":
        return _is_folder_requirement_available(requirement)
    return _is_file_path_available(requirement.path) or _is_folder_requirement_available(requirement)


def _build_preflight_data() -> dict[str, Any]:
    # Build once per pytest session: discover templates -> extract canonical dependencies once ->
    # resolve what is locally cached -> compute missing/runnable sets.
    # The LoRA check is intentionally part of the same gating pass as model repo checks.
    discovered_workflows = _discover_workflow_templates()

    extraction_results: dict[str, WorkflowDependencies] = {}
    for workflow_name in discovered_workflows:
        workflow_path = WORKFLOW_TEMPLATES_DIR / workflow_name
        extraction_results[workflow_name] = extract_workflow_dependencies(workflow_path)

    workflow_required_loras: dict[str, tuple[str, ...]] = {
        workflow_name: result.lora_file_paths for workflow_name, result in extraction_results.items()
    }
    workflow_required_paths: dict[str, tuple[PathRequirement, ...]] = {
        workflow_name: result.path_requirements for workflow_name, result in extraction_results.items()
    }

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

    # Config-declared repos are checked alongside statically-extracted ones so the collection
    # hook can deselect a config whose model is not cached.
    unique_required_repos = {repo for repos in workflow_required_repos.values() for repo in repos} | config_repos()
    repo_available: dict[str, bool] = {}
    for repo_id in sorted(unique_required_repos):
        repo_available[repo_id] = bool(list_repo_revisions_in_cache(repo_id))

    # Keep the workflow-level checks readable: one bucket each for repos, LoRAs, and
    # file/folder paths. A workflow is skipped if any bucket is non-empty.
    missing_by_workflow: dict[str, tuple[str, ...]] = {}
    missing_loras_by_workflow: dict[str, tuple[str, ...]] = {}
    missing_paths_by_workflow: dict[str, tuple[str, ...]] = {}
    for workflow_name in discovered_workflows:
        required = workflow_required_repos.get(workflow_name, ())
        missing = tuple(repo for repo in required if not repo_available.get(repo, False))
        missing_by_workflow[workflow_name] = missing

        required_loras = workflow_required_loras.get(workflow_name, ())
        missing_loras = tuple(path for path in required_loras if not _is_lora_path_available(path))
        missing_loras_by_workflow[workflow_name] = missing_loras

        required_paths = workflow_required_paths.get(workflow_name, ())
        missing_paths = tuple(req.path for req in required_paths if not _is_path_requirement_available(req))
        missing_paths_by_workflow[workflow_name] = missing_paths

    skipped_workflows = sorted(
        [
            workflow_name
            for workflow_name in discovered_workflows
            if missing_by_workflow[workflow_name]
            or missing_loras_by_workflow[workflow_name]
            or missing_paths_by_workflow[workflow_name]
        ]
    )
    runnable_workflows = sorted(
        [
            workflow_name
            for workflow_name in discovered_workflows
            if not missing_by_workflow[workflow_name]
            and not missing_loras_by_workflow[workflow_name]
            and not missing_paths_by_workflow[workflow_name]
        ]
    )

    return {
        "discovered_workflows": discovered_workflows,
        "workflow_required_loras": workflow_required_loras,
        "workflow_required_paths": workflow_required_paths,
        "workflow_required_repos": workflow_required_repos,
        "workflow_extraction_blockers": workflow_extraction_blockers,
        "workflows_with_no_repos": sorted(workflows_with_no_repos),
        "repo_available": repo_available,
        "missing_by_workflow": missing_by_workflow,
        "missing_loras_by_workflow": missing_loras_by_workflow,
        "missing_paths_by_workflow": missing_paths_by_workflow,
        "skipped_workflows": skipped_workflows,
        "runnable_workflows": runnable_workflows,
    }


def get_preflight_data() -> dict[str, Any]:
    """Compute (once per session) and return the workflow preflight data."""
    global _PREFLIGHT_DATA
    if _PREFLIGHT_DATA is None:
        _PREFLIGHT_DATA = _build_preflight_data()
    return _PREFLIGHT_DATA


@dataclass(frozen=True)
class WorkflowSkipReasons:
    """Missing requirements that cause one (workflow template, config) test to be deselected."""

    missing_repos: tuple[str, ...] = ()
    missing_loras: tuple[str, ...] = ()
    missing_paths: tuple[str, ...] = ()

    @property
    def is_skipped(self) -> bool:
        return bool(self.missing_repos or self.missing_loras or self.missing_paths)


def compute_config_skip_reasons(
    workflow_file_name: str, config: WorkflowConfig | None, preflight_data: dict[str, Any]
) -> WorkflowSkipReasons:
    """Resolve missing requirements for one (template, config) pair.

    Repos check the config's own declared repos if set, else the template's extracted repos.
    LoRAs/paths are template-level and apply to every config.
    """
    repo_available: dict[str, bool] = preflight_data["repo_available"]
    if config is not None and config.repos:
        missing_repos = tuple(repo for repo in config.repos if not repo_available.get(repo, False))
    else:
        missing_repos = preflight_data["missing_by_workflow"].get(workflow_file_name, ())

    missing_loras = preflight_data["missing_loras_by_workflow"].get(workflow_file_name, ())
    missing_paths = preflight_data["missing_paths_by_workflow"].get(workflow_file_name, ())
    return WorkflowSkipReasons(
        missing_repos=missing_repos,
        missing_loras=missing_loras,
        missing_paths=missing_paths,
    )
