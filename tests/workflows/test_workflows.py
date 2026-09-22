import logging
from pathlib import Path

import pytest
import pytest_asyncio  # type: ignore[reportMissingImports]
from dotenv import load_dotenv
from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor

LIBRARY_ROOT = Path(__file__).parents[2]
IGNORED_WORKFLOW_NAMES = {
    "WanAnimate.py",
    "WanReplace.py",
}


def get_workflows() -> list[str]:
    """Get all workflow templates for this library."""
    workflows_dir = LIBRARY_ROOT / "workflows" / "templates"
    return [
        str(f)
        for f in workflows_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("__") and f.name not in IGNORED_WORKFLOW_NAMES
    ]


# TODO: https://github.com/griptape-ai/griptape-nodes-library-advanced-media/issues/4
#       Workflows in this library perform CUDA checks that fail on standard CI runners.
# TODO: Re-enable WAN workflows (WanAnimate.py, WanReplace.py) once the framework
#       resolves parameter values from connections before validation. Currently:
#       - CreateConnectionRequest creates graph edges (connections exist)
#       - But get_parameter_value() still returns None at validation time
#       - Validator runs before source nodes execute, finds None, throws ValueError
#       Fix requires one of:
#       1. Auto-resolve parameter values from connected sources before validation
#       2. Change validator to check connection existence instead of value presence
#       3. Explicitly set input_latent values via SetParameterValueRequest in templates
@pytest.mark.parametrize("workflow_path", get_workflows())
@pytest.mark.asyncio
async def test_workflow_runs(workflow_path: str, workflow_executor: LocalWorkflowExecutor) -> None:
    """Simple test to check if the workflow runs without errors."""
    await workflow_executor.arun(workflow_name="main", flow_input={}, workflow_path=workflow_path)
