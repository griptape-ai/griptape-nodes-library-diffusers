from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.workflows.workflow_configs import (
    WORKFLOW_CONFIGS,
    WorkflowConfig,
    get_workflow_paths,
    workflow_test_params,
)

if TYPE_CHECKING:
    from tests.workflows.conftest import ConfigurableWorkflowExecutor

def test_workflow_configs_reference_existing_templates() -> None:
    """Every WORKFLOW_CONFIGS key must match a template stem, or its configs are silently dropped."""
    template_stems = {Path(path).stem for path in get_workflow_paths()}
    unknown = sorted(set(WORKFLOW_CONFIGS) - template_stems)
    assert not unknown, f"WORKFLOW_CONFIGS references templates that do not exist: {unknown}"


# TODO: https://github.com/griptape-ai/griptape-nodes-library-advanced-media/issues/4
#       Workflows in this library perform CUDA checks that fail on standard CI runners.
@pytest.mark.parametrize(("workflow_path", "config"), workflow_test_params())
@pytest.mark.asyncio
async def test_workflow_runs(
    workflow_path: str, config: WorkflowConfig, workflow_executor: ConfigurableWorkflowExecutor
) -> None:
    """Run a workflow template under a given configuration, applying its parameter overrides."""
    await workflow_executor.arun(
        workflow_name="main",
        flow_input={},
        workflow_path=workflow_path,
        parameter_overrides=config.overrides,
    )
