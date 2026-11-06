from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("accelerate", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("safetensors", reason="Uses execution dependencies; run in the execution environment.")

import torch  # noqa: E402
from diffusers.models.controlnets.controlnet_flux import FluxControlNetModel  # noqa: E402
from transformers import PreTrainedModel  # noqa: E402

from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import (  # noqa: E402
    build_component_on_meta_device,
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


def test_diffusers_component_is_built_on_meta_device() -> None:
    component = build_component_on_meta_device(FluxControlNetModel, _TINY_FLUX_CONTROLNET_CONFIG, torch.float32)

    parameters = list(component.parameters())
    assert parameters
    assert all(parameter.device.type == "meta" for parameter in parameters)


def test_generic_pretrained_model_annotation_uses_auto_model() -> None:
    component = torch.nn.Linear(2, 2)
    config = {"model_type": "qwen3", "hidden_size": 2}

    with patch(
        "modular_diffusion_nodes_library.memory_estimation.meta_device_builder.AutoModel.from_config",
        return_value=component,
    ) as from_config:
        result = build_component_on_meta_device(PreTrainedModel, config, torch.bfloat16)

    from_config.assert_called_once()
    assert result is component
    assert next(result.parameters()).dtype == torch.bfloat16
