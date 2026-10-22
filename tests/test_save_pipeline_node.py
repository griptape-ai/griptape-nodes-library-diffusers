from __future__ import annotations

from unittest.mock import MagicMock, patch

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.save_pipeline_node import SaveMode, SavePipelineNode


def test_validation_error_is_reported_in_save_pipeline_status() -> None:
    collision = ValueError(
        "Runtime LoRA filename collision\n\n"
        "Full Pipeline saves runtime LoRAs in one 'loras/' folder. Rename or copy one source file to a unique "
        "filename, update its Load LoRA node, and retry.\n\n"
        "Destination:\n  loras/lora.safetensors\n\n"
        "Source files:\n  1. /models/first/lora.safetensors\n  2. /models/second/lora.safetensors"
    )
    pipeline = MagicMock()
    node = MagicMock()
    node.get_parameter_value.side_effect = {
        "pipeline": pipeline,
        "save_mode": SaveMode.FULL_PIPELINE,
        "file_path": "/tmp/baked",
    }.get

    with (
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.normalize_diffusion_pipeline_value",
            return_value=pipeline,
        ),
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.plan_runtime_lora_copies",
            side_effect=collision,
        ),
    ):
        errors = SavePipelineNode.validate_before_node_run(node)

    assert errors == [collision]
    node._set_status_results.assert_called_once_with(
        was_successful=False,
        result_details=str(collision),
    )


def test_config_only_validation_requires_yaml_destination() -> None:
    node = MagicMock()
    node.name = "Save Pipeline"
    node.get_parameter_value.side_effect = {
        "pipeline": DiffusionPipelineArtifact(pipeline_name="TestPipeline"),
        "save_mode": SaveMode.CONFIG_ONLY,
        "file_path": "/tmp/config.json",
    }.get

    errors = SavePipelineNode.validate_before_node_run(node)

    assert errors is not None
    assert [str(error) for error in errors] == [
        "Parameter 'file_path' on node 'Save Pipeline' must point to a .yaml file."
    ]
