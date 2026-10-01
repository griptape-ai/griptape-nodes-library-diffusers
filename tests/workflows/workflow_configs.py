"""Declarative, in-code test matrix for workflow templates.

Each template maps to one or more `WorkflowConfig`s, which override the model/params via
`ParamOverride`s applied after load. Also backs the preflight repo-availability check.
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
# Nodes shared by the first/last-frame i2v template.
CONDITIONING_NODE_NAME = "Media Generation Conditioning"
GENERATE_NODE_NAME = "Generate Media Latents (Modular Diffusion Pipeline)"


@dataclass(frozen=True)
class ParamOverride:
    """A single node-parameter override applied after load, before run."""

    node_name: str
    parameter_name: str
    value: Any


@dataclass(frozen=True)
class ConnectOverride:
    """Create a connection between two node parameters after load, before run."""

    source_node_name: str
    source_parameter_name: str
    target_node_name: str
    target_parameter_name: str


@dataclass(frozen=True)
class DisconnectOverride:
    """Delete a connection between two node parameters after load, before run."""

    source_node_name: str
    source_parameter_name: str
    target_node_name: str
    target_parameter_name: str


# Any operation the executor applies after load; order within a config's tuple is preserved.
WorkflowOverride = ParamOverride | ConnectOverride | DisconnectOverride


@dataclass(frozen=True)
class WorkflowConfig:
    """One test configuration for a workflow template.

    `repos` drives per-config deselection when not cached. `overrides` order matters: builder
    params rebuild each other's options (provider -> pipeline_type -> model), and connect/disconnect
    ops must bracket those rebuilds correctly.
    `enable_auto_resize` overrides the library setting for configs that need dimension snapping.
    """

    config_id: str
    repos: tuple[str, ...] = ()
    overrides: tuple[WorkflowOverride, ...] = field(default_factory=tuple)
    enable_auto_resize: bool | None = None


def builder_overrides(
    provider: str, pipeline_type: str, repo: str, *, node_name: str = BUILDER_NODE_NAME
) -> tuple[ParamOverride, ...]:
    """Build the ordered provider/pipeline_type/model overrides for a pipeline builder node."""
    return (
        ParamOverride(node_name, "provider", provider),
        ParamOverride(node_name, "pipeline_type", pipeline_type),
        ParamOverride(node_name, "model", repo),
    )


def first_last_conditioning_overrides(
    provider: str, pipeline_type: str, repo: str, *, conditioning_param: str
) -> tuple[WorkflowOverride, ...]:
    """Retarget the first/last i2v template from WAN to another model.

    Switching the builder rebuilds the generate node's runtime params, which can rename the
    conditioning input (e.g. LTX2's `media_conditions`) and drops the shipped connection. So:
    drop the builder->conditioning pipeline link first, retarget the builder, then re-wire the
    conditioning output to the new model's conditioning input.
    """
    return (
        DisconnectOverride(
            source_node_name=BUILDER_NODE_NAME,
            source_parameter_name="pipeline",
            target_node_name=CONDITIONING_NODE_NAME,
            target_parameter_name="pipeline",
        ),
        *builder_overrides(provider, pipeline_type, repo),
        ConnectOverride(
            source_node_name=CONDITIONING_NODE_NAME,
            source_parameter_name="conditioning",
            target_node_name=GENERATE_NODE_NAME,
            target_parameter_name=conditioning_param,
        ),
    )


# Keyed by template filename stem; templates not listed get one default as-shipped run.
WORKFLOW_CONFIGS: dict[str, tuple[WorkflowConfig, ...]] = {
    # Pipelines registered in driver_factory._DRIVER_REGISTRY.
    "Text2Image": (
        WorkflowConfig(config_id="z-image", repos=("Tongyi-MAI/Z-Image-Turbo",)),
        WorkflowConfig(
            config_id="flux1-schnell",
            repos=("black-forest-labs/FLUX.1-schnell",),
            overrides=builder_overrides("Flux", "FluxPipeline", "black-forest-labs/FLUX.1-schnell"),
        ),
        WorkflowConfig(
            config_id="flux2-dev",
            repos=("diffusers/FLUX.2-dev-bnb-4bit",),
            overrides=builder_overrides("Flux2", "Flux2Pipeline", "diffusers/FLUX.2-dev-bnb-4bit"),
        ),
        WorkflowConfig(
            config_id="qwen-image",
            repos=("Qwen/Qwen-Image",),
            overrides=builder_overrides("Qwen", "QwenImagePipeline", "Qwen/Qwen-Image"),
        ),
        WorkflowConfig(
            config_id="sdxl-base",
            repos=("stabilityai/stable-diffusion-xl-base-1.0",),
            overrides=builder_overrides(
                "Stable Diffusion", "StableDiffusionXLPipeline", "stabilityai/stable-diffusion-xl-base-1.0"
            ),
        ),
        WorkflowConfig(
            config_id="sd3.5-large-turbo",
            repos=("stabilityai/stable-diffusion-3.5-large-turbo",),
            overrides=builder_overrides(
                "Stable Diffusion 3", "StableDiffusion3Pipeline", "stabilityai/stable-diffusion-3.5-large-turbo"
            ),
        ),
        WorkflowConfig(
            config_id="wan2.1-t2v-1.3b",
            repos=("Wan-AI/Wan2.1-T2V-1.3B-Diffusers",),
            overrides=builder_overrides("WAN", "WanPipeline", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"),
        ),
        WorkflowConfig(
            config_id="ltx-0.9.7-distilled",
            repos=("Lightricks/LTX-Video-0.9.7-distilled",),
            overrides=builder_overrides("LTX", "LTXPipeline", "Lightricks/LTX-Video-0.9.7-distilled"),
        ),
        WorkflowConfig(
            config_id="ltx2-distilled",
            repos=("dg845/LTX-2.3-Distilled-Diffusers",),
            overrides=builder_overrides("LTX2", "LTX2Pipeline", "dg845/LTX-2.3-Distilled-Diffusers"),
        ),
        WorkflowConfig(
            config_id="hunyuanvideo1.5-480p-distilled",
            repos=("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",),
            overrides=builder_overrides(
                "HunyuanVideo 1.5",
                "HunyuanVideo15Pipeline",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
            ),
        ),
        WorkflowConfig(
            config_id="minimax-h3",
            repos=("MiniMaxAI/MiniMax-H3",),
            overrides=builder_overrides("MiniMax-H3", "MiniMaxH3ModularPipeline", "MiniMaxAI/MiniMax-H3"),
            enable_auto_resize=True,
        ),
    ),
    # Image-only pipelines (img2img here means image-to-image, no text-to-video drivers).
    "Image2Image": (
        WorkflowConfig(config_id="z-image", repos=("Tongyi-MAI/Z-Image-Turbo",)),
        WorkflowConfig(
            config_id="flux1-schnell",
            repos=("black-forest-labs/FLUX.1-schnell",),
            overrides=builder_overrides("Flux", "FluxPipeline", "black-forest-labs/FLUX.1-schnell"),
        ),
        WorkflowConfig(
            config_id="flux2-dev",
            repos=("diffusers/FLUX.2-dev-bnb-4bit",),
            overrides=builder_overrides("Flux2", "Flux2Pipeline", "diffusers/FLUX.2-dev-bnb-4bit"),
            enable_auto_resize=True,
        ),
        WorkflowConfig(
            config_id="qwen-image",
            repos=("Qwen/Qwen-Image",),
            overrides=builder_overrides("Qwen", "QwenImagePipeline", "Qwen/Qwen-Image"),
        ),
        WorkflowConfig(
            config_id="sdxl-base",
            repos=("stabilityai/stable-diffusion-xl-base-1.0",),
            overrides=builder_overrides(
                "Stable Diffusion", "StableDiffusionXLPipeline", "stabilityai/stable-diffusion-xl-base-1.0"
            ),
        ),
        WorkflowConfig(
            config_id="sd3.5-large-turbo",
            repos=("stabilityai/stable-diffusion-3.5-large-turbo",),
            overrides=builder_overrides(
                "Stable Diffusion 3", "StableDiffusion3Pipeline", "stabilityai/stable-diffusion-3.5-large-turbo"
            ),
        ),
    ),
    # Inpainting checkpoints (Fill/Klein) plus base pipelines that support it via VaeMaskEncodeNode.
    "Inpainting": (
        WorkflowConfig(config_id="flux2-klein-4b", repos=("black-forest-labs/FLUX.2-klein-4B",)),
        WorkflowConfig(
            config_id="flux1-fill-dev",
            repos=("black-forest-labs/FLUX.1-Fill-dev",),
            overrides=builder_overrides("Flux", "FluxFillPipeline", "black-forest-labs/FLUX.1-Fill-dev"),
        ),
        WorkflowConfig(
            config_id="flux1-schnell",
            repos=("black-forest-labs/FLUX.1-schnell",),
            overrides=builder_overrides("Flux", "FluxPipeline", "black-forest-labs/FLUX.1-schnell"),
        ),
        WorkflowConfig(
            config_id="qwen-image",
            repos=("Qwen/Qwen-Image",),
            overrides=builder_overrides("Qwen", "QwenImagePipeline", "Qwen/Qwen-Image"),
        ),
        WorkflowConfig(
            config_id="sdxl-base",
            repos=("stabilityai/stable-diffusion-xl-base-1.0",),
            overrides=builder_overrides(
                "Stable Diffusion", "StableDiffusionXLPipeline", "stabilityai/stable-diffusion-xl-base-1.0"
            ),
        ),
        WorkflowConfig(
            config_id="sd3.5-large-turbo",
            repos=("stabilityai/stable-diffusion-3.5-large-turbo",),
            overrides=builder_overrides(
                "Stable Diffusion 3", "StableDiffusion3Pipeline", "stabilityai/stable-diffusion-3.5-large-turbo"
            ),
        ),
        WorkflowConfig(
            config_id="z-image",
            repos=("Tongyi-MAI/Z-Image-Turbo",),
            overrides=builder_overrides("Z-Image", "ZImagePipeline", "Tongyi-MAI/Z-Image-Turbo"),
        ),
    ),
    # WAN i2v is the shipped default. LTX/LTX2/MiniMax-H3/HunyuanVideo 1.5 configs are specified here.
    "FirstAndLastFrameImage2Video": (
        WorkflowConfig(
            config_id="wan2.2-i2v-a14b",
            repos=("Wan-AI/Wan2.2-I2V-A14B-Diffusers",),
        ),
        WorkflowConfig(
            config_id="ltx-0.9.7-distilled",
            repos=("Lightricks/LTX-Video-0.9.7-distilled",),
            overrides=first_last_conditioning_overrides(
                "LTX", "LTXPipeline", "Lightricks/LTX-Video-0.9.7-distilled", conditioning_param="media_conditions_0"
            ),
            enable_auto_resize=True,
        ),
        WorkflowConfig(
            config_id="ltx2-distilled",
            repos=("dg845/LTX-2.3-Distilled-Diffusers",),
            overrides=first_last_conditioning_overrides(
                "LTX2", "LTX2Pipeline", "dg845/LTX-2.3-Distilled-Diffusers", conditioning_param="media_conditions_0"
            ),
            enable_auto_resize=True,
        ),
        WorkflowConfig(
            config_id="minimax-h3",
            repos=("MiniMaxAI/MiniMax-H3",),
            overrides=(
                *first_last_conditioning_overrides(
                    "MiniMax-H3",
                    "MiniMaxH3ModularPipeline",
                    "MiniMaxAI/MiniMax-H3",
                    conditioning_param="conditioning_images",
                ),
                ParamOverride("Create Noise Latents", "num_frames", 124),
            ),
            enable_auto_resize=True,
        ),
        WorkflowConfig(
            config_id="hunyuanvideo1.5-480p-i2v-distilled",
            repos=("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",),
            overrides=(
                *first_last_conditioning_overrides(
                    "HunyuanVideo 1.5",
                    "HunyuanVideo15ImageToVideoPipeline",
                    "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
                    conditioning_param="conditioning_images",
                ),
                ParamOverride("Media Generation Conditioning", "image_preset", "First frame"),
                ConnectOverride(
                    source_node_name="Resize Image",
                    source_parameter_name="output",
                    target_node_name=CONDITIONING_NODE_NAME,
                    target_parameter_name="image_0",
                ),
            ),
            enable_auto_resize=True,
        ),
    ),
    # Edit-pipeline templates (Kontext, Klein, QwenImageEdit) are locked to one pipeline_type each,
    # so they get a single documenting entry rather than the provider variety above.
    "FluxKontextImageEdit": (
        WorkflowConfig(
            config_id="flux1-kontext-dev", repos=("black-forest-labs/FLUX.1-Kontext-dev", "google/t5-v1_1-xxl")
        ),
    ),
    "FluxKontextInpainting": (
        WorkflowConfig(
            config_id="flux1-kontext-dev", repos=("black-forest-labs/FLUX.1-Kontext-dev", "google/t5-v1_1-xxl")
        ),
    ),
    "Flux2KleinGuidedInpainting": (
        WorkflowConfig(config_id="flux2-klein-4b", repos=("black-forest-labs/FLUX.2-klein-4B",)),
    ),
    "Multi-ViewPromptBatcher": (WorkflowConfig(config_id="qwen-image-edit", repos=("Qwen/Qwen-Image-Edit",)),),
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
