import pytest

from tests.preflight import WORKFLOW_TESTS_PRESENT_KEY, get_preflight_data
from tests.workflows.dependencies import ExtractionBlocker


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
    parser.addoption(
        "--workflow-runs-dir",
        default=None,
        help="Base directory for per-test workflow run folders (default: a fresh OS temp directory, "
        "removed after the session unless --no-cleanup is set).",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # Strict mode fails fast if any workflow would be deselected by preflight.
    if not session.config.getoption("--preflight-strict"):
        return

    preflight_data = get_preflight_data()
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

    # Make preflight behavior visible in test output so deselection is transparent.
    preflight_data = get_preflight_data()
    discovered = preflight_data["discovered_workflows"]
    runnable = preflight_data["runnable_workflows"]
    skipped = preflight_data["skipped_workflows"]
    zero_repo = preflight_data["workflows_with_no_repos"]
    missing_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_by_workflow"]
    missing_libraries_by_workflow: dict[str, tuple[str, ...]] = preflight_data["missing_libraries_by_workflow"]
    extraction_blockers_by_workflow: dict[str, tuple[ExtractionBlocker, ...]] = preflight_data[
        "workflow_extraction_blockers"
    ]

    tw = config.get_terminal_writer()

    if config.getoption("--no-cleanup"):
        cleanup_detail = "per-test workflow run folders are kept after the test (--no-cleanup is set)"
    else:
        cleanup_detail = "per-test workflow run folders are deleted after the test (pass --no-cleanup to keep them)"

    if config.getoption("--preflight-strict"):
        preflight_detail = "the run fails immediately if any workflow is missing required repos/libraries"
    else:
        preflight_detail = (
            "workflows missing required repos/libraries are skipped, not failed "
            "(pass --preflight-strict to fail the run instead)"
        )

    runnable_count = tw.markup(f"runnable={len(runnable)}", green=True, bold=True)
    skipped_count = tw.markup(f"skipped={len(skipped)}", red=True, bold=True) if skipped else f"skipped={len(skipped)}"

    children: list[tuple[str, list]] = [
        (f"discovered={len(discovered)} {runnable_count} {skipped_count}", []),
        (cleanup_detail, []),
        (preflight_detail, []),
    ]

    # Report exactly why each skipped (or extraction-blocked) workflow is blocked, so users can
    # see the reason directly in test output instead of having to inspect preflight data by hand.
    workflows_needing_detail = sorted(set(skipped) | set(extraction_blockers_by_workflow))
    if workflows_needing_detail:
        workflow_nodes: list[tuple[str, list]] = []
        for workflow_name in workflows_needing_detail:
            reasons: list[tuple[str, list]] = []
            missing_repos = missing_by_workflow.get(workflow_name, ())
            if missing_repos:
                reasons.append((tw.markup(f"missing repos: {', '.join(missing_repos)}", yellow=True), []))
            missing_libraries = missing_libraries_by_workflow.get(workflow_name, ())
            if missing_libraries:
                reasons.append((tw.markup(f"missing libraries: {', '.join(missing_libraries)}", yellow=True), []))
            blockers = extraction_blockers_by_workflow.get(workflow_name, ())
            if blockers:
                blocker_detail = "; ".join(blocker.detail for blocker in blockers)
                reasons.append((tw.markup(f"extraction blocked: {blocker_detail}", red=True), []))
            workflow_nodes.append((tw.markup(workflow_name, red=True), reasons))
        children.append(
            (tw.markup(f"skipped workflows ({len(workflows_needing_detail)})", red=True, bold=True), workflow_nodes)
        )

    if zero_repo:
        children.append((tw.markup(f"workflows with no extracted repos: {', '.join(zero_repo)}", yellow=True), []))

    return _render_tree("Workflow preflight", children)
