from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.save_pipeline_node import SaveMode, SavePipelineNode


def test_initial_setup_restores_full_pipeline_picker_without_resetting_destination() -> None:
    node = SavePipelineNode(name="Save Pipeline")
    restored_path = "{project_dir}/saved-bake"

    node.set_parameter_value("file_path", restored_path, initial_setup=True)
    node.set_parameter_value("save_mode", SaveMode.FULL_PIPELINE, initial_setup=True)

    parameter = node.get_parameter_by_name("file_path")
    assert parameter is not None
    picker = next(iter(parameter.find_elements_by_type(FileSystemPicker)))
    assert node.get_parameter_value("file_path") == restored_path
    assert parameter.ui_options["display_name"] == "Output Folder"
    assert picker.allow_files is False
    assert picker.allow_directories is True
    assert picker.file_types == []


def test_interactive_save_mode_change_resets_destination() -> None:
    node = SavePipelineNode(name="Save Pipeline")
    node.set_parameter_value("file_path", "{project_dir}/custom.yaml")

    node.set_parameter_value("save_mode", SaveMode.FULL_PIPELINE)

    assert node.get_parameter_value("file_path") == "{project_dir}/baked-pipeline"


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


def test_process_uses_request_backed_workspace_path(tmp_path: Path) -> None:
    destination = tmp_path / "saved-pipeline.yaml"
    pipeline = DiffusionPipelineArtifact(pipeline_name="TestPipeline")
    node = MagicMock()
    node.name = "Save Pipeline"
    node.get_parameter_value.side_effect = {
        "pipeline": pipeline,
        "save_mode": SaveMode.CONFIG_ONLY,
        "file_path": "saved-pipeline.yaml",
    }.get
    node._run_with_status.side_effect = lambda operation, **_kwargs: operation()

    with (
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.normalize_diffusion_pipeline_value",
            return_value=pipeline,
        ),
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.get_workspace_path",
            return_value=tmp_path,
        ) as get_workspace_path_mock,
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.resolve_workspace_path",
            return_value=destination,
        ),
        patch(
            "modular_diffusion_nodes_library.nodes.save_pipeline_node.serialize_pipeline_artifact",
            return_value="pipeline recipe\n",
        ),
        patch.object(
            GriptapeNodes,
            "ConfigManager",
            side_effect=AssertionError("worker execution must not access ConfigManager"),
        ) as config_manager_mock,
    ):
        SavePipelineNode.process(node)

    get_workspace_path_mock.assert_called_once_with()
    config_manager_mock.assert_not_called()
    assert destination.read_text(encoding="utf-8") == "pipeline recipe\n"
