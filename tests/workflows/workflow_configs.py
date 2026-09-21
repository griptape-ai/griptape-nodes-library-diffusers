"""Declarative, in-code test matrix for workflow templates.

Each workflow template maps to one or more `WorkflowConfig`s. A config swaps the model
(and any other node parameters) by declaring `ParamOverride`s that are applied
programmatically after the template loads but before it runs.

This module holds data and helpers for test parametrization and the preflight repo-availability 
check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _pytest.mark.structures import ParameterSet

LIBRARY_ROOT = Path(__file__).parents[2]
WORKFLOW_TEMPLATES_DIR = LIBRARY_ROOT / "workflows" / "templates"

# Node name of the builder nodes
BUILDER_NODE_NAME = "Modular Diffusion Pipeline Builder"


@dataclass(frozen=True)
class ParamOverride:
    """A single node-parameter override applied after load, before run."""

    node_name: str
    parameter_name: str
    value: Any


@dataclass(frozen=True)
class WorkflowConfig:
    """One test configuration for a workflow template.

    `repos` declares the HuggingFace repos this config needs cached; it drives per-config
    deselection when a model is not available locally. 
    `overrides` is an ordered tuple: the builder node requires provider -> pipeline_type -> 
    model ordering because each rebuilds the next parameter's options.
    """

    config_id: str
    repos: tuple[str, ...] = ()
    overrides: tuple[ParamOverride, ...] = field(default_factory=tuple)


def builder_overrides(
    provider: str, pipeline_type: str, repo: str, *, node_name: str = BUILDER_NODE_NAME
) -> tuple[ParamOverride, ...]:
    """Build the ordered provider/pipeline_type/model overrides for a pipeline builder node."""
    return (
        ParamOverride(node_name, "provider", provider),
        ParamOverride(node_name, "pipeline_type", pipeline_type),
        ParamOverride(node_name, "model", repo),
    )


# Keyed by workflow template filename stem. Templates absent from this map fall back to a
# single default run that exercises the values baked into the template as-shipped.
WORKFLOW_CONFIGS: dict[str, tuple[WorkflowConfig, ...]] = {
    "Text2Image": (
        WorkflowConfig(config_id="z-image", repos=("Tongyi-MAI/Z-Image-Turbo",)),
        WorkflowConfig(
            config_id="flux1-schnell",
            repos=("black-forest-labs/FLUX.1-schnell",),
            overrides=builder_overrides("Flux", "FluxPipeline", "black-forest-labs/FLUX.1-schnell"),
        ),
    ),
    "Image2Image": (
        WorkflowConfig(config_id="z-image", repos=("Tongyi-MAI/Z-Image-Turbo",)),
        WorkflowConfig(
            config_id="flux1-schnell",
            repos=("black-forest-labs/FLUX.1-schnell",),
            overrides=builder_overrides("Flux", "FluxPipeline", "black-forest-labs/FLUX.1-schnell"),
        ),
    ),
    "Inpainting": (
        WorkflowConfig(config_id="flux2-klein-4b", repos=("black-forest-labs/FLUX.2-klein-4B",)),
        WorkflowConfig(
            config_id="flux1-fill-dev",
            repos=("black-forest-labs/FLUX.1-Fill-dev",),
            overrides=builder_overrides("Flux", "FluxFillPipeline", "black-forest-labs/FLUX.1-Fill-dev"),
        ),
    ),
}


def get_workflow_paths() -> list[str]:
    """Discover all workflow template files for this library."""
    return [
        str(path)
        for path in WORKFLOW_TEMPLATES_DIR.iterdir()
        if path.is_file() and path.suffix == ".py" and not path.name.startswith("__")
    ]


def config_repos() -> set[str]:
    """Every repo declared across all configs, for preflight availability checks."""
    return {repo for configs in WORKFLOW_CONFIGS.values() for config in configs for repo in config.repos}


def workflow_test_params() -> list[ParameterSet]:
    """Pair each workflow template with each of its configs as a pytest parameter set."""
    params: list[ParameterSet] = []
    for workflow_path in sorted(get_workflow_paths()):
        stem = Path(workflow_path).stem
        configs = WORKFLOW_CONFIGS.get(stem, (WorkflowConfig(config_id="default"),))
        for config in configs:
            params.append(pytest.param(workflow_path, config, id=f"{stem}-{config.config_id}"))
    return params
