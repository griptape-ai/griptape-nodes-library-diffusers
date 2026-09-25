from __future__ import annotations

from unittest.mock import patch

import torch
from transformers import PreTrainedModel

from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import build_component_on_meta_device


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
