from pathlib import Path

import pytest

from tests.preflight import (
    WORKFLOW_SELECTED_KEYS_KEY,
    WORKFLOW_TESTS_PRESENT_KEY,
    WorkflowSkipReasons,
    compute_config_skip_reasons,
    get_preflight_data,
)
from tests.workflows.dependencies import ExtractionBlocker
from tests.workflows.workflow_configs import WorkflowConfig, workflow_test_params


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
        help="Fail test collection if any workflow is missing required cached model repos, LoRAs, or local paths.",
    )
    parser.addoption(
        "--collect-workflow-config-only",
        action="store_true",
        default=False,
        help="Show the selected workflow/config hierarchy in collection output without dependency details.",
    )
    parser.addoption(
        "--workflow-runs-dir",
        default=None,
        help="Base directory for per-test workflow run folders (default: a fresh OS temp directory, "
        "removed after the session unless --no-cleanup is set).",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # Strict mode fails fast if any workflow/config pair would be deselected by preflight. It walks
    # the same per-config logic collection uses, so a config whose matrix repo is uncached fails here
    # even when the template's own shipped model is present.
    # It prints the exact repo/LoRA/path gaps that caused the deselection so the issue is easy to
    # diagnose before the workflow test run starts.
    if not session.config.getoption("--preflight-strict"):
        return

    preflight_data = get_preflight_data()

    missing_repo_lines: list[str] = []
    missing_lora_lines: list[str] = []
    missing_path_lines: list[str] = []
    for param in workflow_test_params():
        workflow_path, config_obj = param.values
        if not isinstance(config_obj, WorkflowConfig):
            continue

        workflow_file_name = Path(str(workflow_path)).name
        reasons = compute_config_skip_reasons(workflow_file_name, config_obj, preflight_data)
        if not reasons.is_skipped:
            continue

        label = f"  - {workflow_file_name}[{config_obj.config_id}]"
        if reasons.missing_repos:
            missing_repo_lines.append(f"{label}: {', '.join(reasons.missing_repos)}")
        if reasons.missing_loras:
            missing_lora_lines.append(f"{label}: {', '.join(reasons.missing_loras)}")
        if reasons.missing_paths:
            missing_path_lines.append(f"{label}: {', '.join(reasons.missing_paths)}")

    details_sections: list[str] = []
    if missing_repo_lines:
        details_sections.append("Missing cached model repos:\n" + "\n".join(missing_repo_lines))
    if missing_lora_lines:
        details_sections.append("Missing or unreadable LoRA files:\n" + "\n".join(missing_lora_lines))
    if missing_path_lines:
        details_sections.append("Missing local files or folders:\n" + "\n".join(missing_path_lines))

    if not details_sections:
        return

    details = "\n\n".join(details_sections)
    raise pytest.UsageError(
        "Workflow preflight strict mode failed.\n"
        f"{details}\n"
        "Disable strict mode to run only workflows with available models and assets (default deselection behavior)."
    )


def _render_tree(root_label: str, children: list[tuple[str, list]]) -> list[str]:
    lines = [root_label]

    def render_children(nodes: list[tuple[str, list]], prefix: str) -> None:
        for index, (label, grandchildren) in enumerate(nodes):
            is_last = index == len(nodes) - 1
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{label}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            render_children(grandchildren, child_prefix)

    render_children(children, "")
    return lines


def pytest_report_collectionfinish(config: pytest.Config) -> list[str]:
    # Only relevant when this invocation actually collected workflow tests (set by
    # tests/workflows/conftest.py's pytest_collection_modifyitems, which runs beforehand).
    if not config.stash.get(WORKFLOW_TESTS_PRESENT_KEY, default=False):
        return []

    # Make the preflight decisions visible in the collection output so the user can see exactly why
    # workflows were skipped without having to inspect internal data structures manually.
    preflight_data = get_preflight_data()
    zero_repo = preflight_data["workflows_with_no_repos"]
    extraction_blockers_by_workflow: dict[str, tuple[ExtractionBlocker, ...]] = preflight_data[
        "workflow_extraction_blockers"
    ]

    # Scope the report to (workflow, config) keys that survived -k/-m selection.
    selected_keys = config.stash.get(WORKFLOW_SELECTED_KEYS_KEY, default=None)
    selected_workflows = {workflow_name for workflow_name, _ in selected_keys} if selected_keys is not None else None

    # Per-config, not whole-template, so a mixed template (e.g. FirstAndLastFrameImage2Video)
    # only lists the configs actually deselected.
    skipped_configs_by_workflow: dict[str, list[tuple[str, WorkflowSkipReasons]]] = {}
    config_status_by_workflow: dict[str, list[tuple[str, WorkflowSkipReasons]]] = {}
    discovered = 0
    runnable = 0
    discovered_workflows: set[str] = set()
    for param in workflow_test_params():
        workflow_path, config_obj = param.values
        if not isinstance(config_obj, WorkflowConfig):
            continue
        workflow_file_name = Path(str(workflow_path)).name
        if selected_keys is not None and (workflow_file_name, config_obj.config_id) not in selected_keys:
            continue
        discovered += 1
        discovered_workflows.add(workflow_file_name)
        reasons = compute_config_skip_reasons(workflow_file_name, config_obj, preflight_data)
        config_status_by_workflow.setdefault(workflow_file_name, []).append((config_obj.config_id, reasons))
        if reasons.is_skipped:
            skipped_configs_by_workflow.setdefault(workflow_file_name, []).append((config_obj.config_id, reasons))
        else:
            runnable += 1
    skipped = discovered - runnable
    flow_count = len(discovered_workflows)

    tw = config.get_terminal_writer()

    if config.getoption("--no-cleanup"):
        cleanup_detail = "per-test workflow run folders are kept after the test (--no-cleanup is set)"
    else:
        cleanup_detail = "per-test workflow run folders are deleted after the test (pass --no-cleanup to keep them)"

    if config.getoption("--preflight-strict"):
        preflight_detail = (
            "the run fails immediately if any workflow is missing required model repos, LoRAs, or local paths"
        )
    else:
        preflight_detail = (
            "workflows missing required model repos, LoRAs, or local paths are skipped, not failed "
            "(pass --preflight-strict to fail the run instead)"
        )

    runnable_count = tw.markup(f"runnable={runnable}", green=True, bold=True)
    skipped_count = tw.markup(f"skipped={skipped}", red=True, bold=True) if skipped else f"skipped={skipped}"

    children: list[tuple[str, list]] = [
        (f"discovered={discovered} {runnable_count} {skipped_count} flows={flow_count} ", []),
        (cleanup_detail, []),
        (preflight_detail, []),
    ]

    show_config_hierarchy = config.getoption("--collect-workflow-config-only")
    if show_config_hierarchy:
        workflow_nodes: list[tuple[str, list]] = []
        for workflow_name in sorted(config_status_by_workflow):
            config_nodes: list[tuple[str, list]] = []
            for config_id, reasons in config_status_by_workflow[workflow_name]:
                if reasons.is_skipped:
                    config_nodes.append((tw.markup(f"{config_id} (skipped)", yellow=True), []))
                else:
                    config_nodes.append((tw.markup(f"{config_id} (runnable)", green=True), []))
            workflow_nodes.append((workflow_name, config_nodes))
        children.append((f"config hierarchy ({discovered} configs)", workflow_nodes))

    # Report each skipped test config (and each extraction-blocked template) with its reasons, so
    # users can see the reason directly in test output instead of inspecting preflight data by hand.
    blocked_workflows = set(extraction_blockers_by_workflow)
    if selected_workflows is not None:
        blocked_workflows &= selected_workflows
    workflows_needing_detail = sorted(set(skipped_configs_by_workflow) | blocked_workflows)
    if workflows_needing_detail and not show_config_hierarchy:
        workflow_nodes: list[tuple[str, list]] = []
        for workflow_name in workflows_needing_detail:
            config_nodes: list[tuple[str, list]] = []
            for config_id, reasons in skipped_configs_by_workflow.get(workflow_name, ()):
                reason_nodes: list[tuple[str, list]] = []
                if reasons.missing_repos:
                    reason_nodes.append(
                        (tw.markup(f"missing model repos: {', '.join(reasons.missing_repos)}", yellow=True), [])
                    )
                if reasons.missing_loras:
                    reason_nodes.append(
                        (tw.markup(f"missing LoRAs: {', '.join(reasons.missing_loras)}", yellow=True), [])
                    )
                if reasons.missing_paths:
                    reason_nodes.append(
                        (tw.markup(f"missing files/folders: {', '.join(reasons.missing_paths)}", yellow=True), [])
                    )
                config_nodes.append((tw.markup(config_id, yellow=True), reason_nodes))
            blockers = extraction_blockers_by_workflow.get(workflow_name, ())
            if blockers:
                blocker_detail = "; ".join(blocker.detail for blocker in blockers)
                config_nodes.append((tw.markup(f"extraction blocked: {blocker_detail}", red=True), []))
            workflow_nodes.append((tw.markup(workflow_name, red=True), config_nodes))
        children.append((tw.markup(f"skipped configs ({skipped})", red=True, bold=True), workflow_nodes))

    if zero_repo:
        scoped_zero_repo = (
            zero_repo if selected_workflows is None else [w for w in zero_repo if w in selected_workflows]
        )
        if scoped_zero_repo:
            children.append(
                (tw.markup(f"workflows with no extracted repos: {', '.join(scoped_zero_repo)}", yellow=True), [])
            )

    return _render_tree("Workflow preflight", children)
