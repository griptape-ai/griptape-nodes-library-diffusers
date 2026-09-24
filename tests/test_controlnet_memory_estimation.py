"""Regression tests for ControlNet memory estimation (Area 3 of
memoryEstimationDocs/spikes/memory_estimation_lora_controlnet_plan.md).

Before this change, `pipe.controlnet` fell into `_classify_role()`'s generic "other"
bucket on the post-load path (weight bytes counted by accident, activation bytes
always 0, plus a spurious "Unrecognized component" warning), and the pre-load path
never read `ControlNetDiffusionPipelineArtifact.controlnet_models` at all. Tests below
use real (tiny-dimensioned) diffusers ControlNet classes built on the meta device
(`accelerate.init_empty_weights()`) rather than mocks, since the whole point of this
feature is exact weight-byte accounting from real config shapes.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
from accelerate import init_empty_weights
from diffusers.models.controlnets.controlnet import ControlNetModel
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel, FluxMultiControlNetModel

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.memory_estimation import pipeline_memory_estimator as pme
from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import ComponentConfigNotCachedError
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    _classify_role,
    _controlnet_slot_estimates,
    estimate_pipeline_memory,
)

_TINY_FLUX_CONTROLNET_CONFIG = {
    "patch_size": 1,
    "in_channels": 4,
    "num_layers": 1,
    "num_single_layers": 1,
    "attention_head_dim": 8,
    "num_attention_heads": 2,
    "joint_attention_dim": 16,
    "pooled_projection_dim": 8,
    "guidance_embeds": False,
}

_TINY_SDXL_CONTROLNET_CONFIG = {
    "block_out_channels": [32, 64],
    "down_block_types": ["DownBlock2D", "CrossAttnDownBlock2D"],
    "layers_per_block": 1,
    "cross_attention_dim": 32,
    "transformer_layers_per_block": 1,
}


def _tiny_flux_controlnet() -> FluxControlNetModel:
    with init_empty_weights():
        return FluxControlNetModel.from_config(dict(_TINY_FLUX_CONTROLNET_CONFIG)).to(dtype=torch.bfloat16)


def _tiny_sdxl_controlnet() -> ControlNetModel:
    with init_empty_weights():
        return ControlNetModel.from_config(dict(_TINY_SDXL_CONTROLNET_CONFIG)).to(dtype=torch.bfloat16)


def _flux_latent() -> LatentArtifact:
    return LatentArtifact.from_torch(torch.zeros(1, 16, 64, 64), source_shape=(1, 3, 512, 512))


def test_classify_role_recognizes_controlnet() -> None:
    assert _classify_role("controlnet") == "controlnet"


def test_single_controlnet_is_estimated_with_weights_and_activations() -> None:
    pipe = SimpleNamespace(components={"controlnet": _tiny_flux_controlnet()})

    estimate = estimate_pipeline_memory(pipe, _flux_latent(), {}, "FluxPipeline")

    assert len(estimate.components) == 1
    component = estimate.components[0]
    assert component.component_name == "controlnet"
    assert component.role == "controlnet"
    assert component.weight_bytes > 0
    assert component.activation_bytes > 0


def test_controlnet_no_longer_emits_unrecognized_component_warning() -> None:
    """Regression guard for the _classify_role() fix: a plain 'controlnet' component
    used to land in the generic 'other' bucket and emit this exact warning text."""
    pipe = SimpleNamespace(components={"controlnet": _tiny_flux_controlnet()})

    estimate = estimate_pipeline_memory(pipe, _flux_latent(), {}, "FluxPipeline")

    assert not any("Unrecognized component" in warning for warning in estimate.warnings)


def test_stacked_controlnets_appear_as_separate_components_with_linear_weight_scaling() -> None:
    net_a = _tiny_flux_controlnet()
    net_b = _tiny_flux_controlnet()
    multi = FluxMultiControlNetModel([net_a, net_b])
    pipe = SimpleNamespace(components={"controlnet": multi})

    estimate = estimate_pipeline_memory(pipe, _flux_latent(), {}, "FluxPipeline")

    assert [c.component_name for c in estimate.components] == ["controlnet_0", "controlnet_1"]
    assert estimate.components[0].weight_bytes == estimate.components[1].weight_bytes
    assert estimate.components[0].weight_bytes > 0


def test_sdxl_controlnet_activation_uses_unet_sdpa_family_and_its_own_config() -> None:
    pipe = SimpleNamespace(components={"controlnet": _tiny_sdxl_controlnet()})
    latent = LatentArtifact.from_torch(torch.zeros(1, 4, 64, 64), source_shape=(1, 3, 512, 512))

    estimate = estimate_pipeline_memory(pipe, latent, {}, "StableDiffusionXLPipeline")

    component = estimate.components[0]
    assert component.role == "controlnet"
    assert component.activation_bytes > 0
    assert component.is_estimated
    assert "UNet-SDPA" in component.warning


def test_controlnet_with_unregistered_pipeline_falls_back_to_weights_only() -> None:
    pipe = SimpleNamespace(components={"controlnet": _tiny_flux_controlnet()})

    estimate = estimate_pipeline_memory(pipe, _flux_latent(), {}, "SomeUnregisteredPipeline")

    component = estimate.components[0]
    assert component.weight_bytes > 0
    assert component.activation_bytes == 0
    assert "No memory-estimation family registered" in component.warning


def _controlnet_artifact(repo_ids: list[str]) -> ControlNetDiffusionPipelineArtifact:
    base = DiffusionPipelineArtifact(pipeline_name="FluxPipeline", config_hash="basehash")
    return ControlNetDiffusionPipelineArtifact(base_artifact=base, controlnet_models=repo_ids, config_hash="cnhash")


def test_preload_controlnet_slots_are_estimated_from_each_repos_own_config() -> None:
    artifact = _controlnet_artifact(["org/cn-a", "org/cn-b"])
    latent = _flux_latent()

    with patch.object(
        pme,
        "resolve_base_component_config",
        side_effect=lambda repo_id, component, revision=None: dict(_TINY_FLUX_CONTROLNET_CONFIG),
    ):
        estimates = _controlnet_slot_estimates(artifact, "FluxPipeline", latent, "None", False)

    assert [e.component_name for e in estimates] == ["controlnet_0", "controlnet_1"]
    assert all(e.weight_bytes > 0 and e.activation_bytes > 0 for e in estimates)


def test_preload_controlnet_slot_reports_warning_when_config_not_cached() -> None:
    artifact = _controlnet_artifact(["org/cn-a"])
    latent = _flux_latent()

    def _raise_not_cached(repo_id: str, component: str, revision: str | None = None) -> dict:
        msg = f"config for '{repo_id}' is not in the local HuggingFace cache."
        raise ComponentConfigNotCachedError(msg)

    with patch.object(pme, "resolve_base_component_config", side_effect=_raise_not_cached):
        estimates = _controlnet_slot_estimates(artifact, "FluxPipeline", latent, "None", False)

    assert len(estimates) == 1
    assert estimates[0].weight_bytes == 0
    assert estimates[0].is_estimated
    assert "not in the local HuggingFace cache" in estimates[0].warning


def test_preload_controlnet_slot_warns_when_pipeline_has_no_registered_controlnet_class() -> None:
    artifact = _controlnet_artifact(["org/cn-a", "org/cn-b"])
    latent = _flux_latent()

    estimates = _controlnet_slot_estimates(artifact, "SomeUnregisteredPipeline", latent, "None", False)

    assert len(estimates) == 2  # noqa: PLR2004
    assert all(e.weight_bytes == 0 and e.is_estimated for e in estimates)
    assert all("no ControlNet model class is registered" in e.warning for e in estimates)
