from pathlib import Path

import nox

nox.options.sessions = ["workflow_collect", "workflow_tests"]

REPO_ROOT = Path(__file__).parent.resolve()
WORKFLOW_TEST_FILE = REPO_ROOT / "tests" / "workflows" / "test_workflows.py"
WORKFLOW_TEMPLATE_DIR = REPO_ROOT / "workflows" / "templates"


def _run_pytest(session: nox.Session, *pytest_args: str, success_codes: tuple[int, ...] = (0,)) -> None:
    session.run("uv", "run", "pytest", "-q", "-ra", *pytest_args, success_codes=list(success_codes))


def _resolve_template_path(session: nox.Session, template_arg: str) -> Path:
    template = Path(template_arg)
    if template.is_absolute():
        candidate = template
    else:
        candidate = WORKFLOW_TEMPLATE_DIR / template
        if not candidate.suffix:
            candidate = candidate.with_suffix(".py")

    if not candidate.exists():
        session.error(
            f"Template '{template_arg}' does not exist. Expected file under {WORKFLOW_TEMPLATE_DIR} or an absolute path."
        )

    return candidate.resolve()


@nox.session(venv_backend="none")
def workflow_collect(session: nox.Session) -> None:
    """Collect workflow tests with preflight deselection and show summary."""
    _run_pytest(session, "tests/workflows", "--collect-only", *session.posargs, success_codes=(0, 5))


@nox.session(venv_backend="none")
def workflow_collect_config(session: nox.Session) -> None:
    """Collect workflow tests and show the selected workflow/config hierarchy."""
    _run_pytest(
        session,
        "tests/workflows",
        "--collect-only",
        "--collect-workflow-config-only",
        *session.posargs,
        success_codes=(0, 5),
    )


@nox.session(venv_backend="none")
def workflow_tests(session: nox.Session) -> None:
    """Run workflow tests with preflight deselection."""
    _run_pytest(session, "tests/workflows", *session.posargs)


@nox.session(venv_backend="none")
def workflow_tests_strict(session: nox.Session) -> None:
    """Run workflow tests in strict preflight mode (fail if cache is incomplete)."""
    _run_pytest(session, "tests/workflows", "--preflight-strict", *session.posargs)


@nox.session(venv_backend="none")
def workflow_single(session: nox.Session) -> None:
    """Run one workflow template by file name or path.

    Usage:
        nox -s workflow_single -- Text2Image.py
        nox -s workflow_single -- workflows/templates/Text2Image.py --no-cleanup
    """
    if not session.posargs:
        session.error("Provide a workflow template name or path, for example: nox -s workflow_single -- Text2Image.py")

    template_arg, *pytest_args = session.posargs
    template_path = _resolve_template_path(session, template_arg)
    node_id = f"{WORKFLOW_TEST_FILE}::test_workflow_runs[{template_path}]"

    _run_pytest(session, node_id, *pytest_args)
