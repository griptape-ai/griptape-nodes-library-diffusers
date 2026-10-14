from pathlib import Path
from unittest.mock import Mock

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    deserialize_pipeline_artifact,
    serialize_pipeline_artifact,
)
from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode


def test_load_pipeline_exposes_optimization_parameters_without_quantization() -> None:
    node = LoadPipelineNode(name="Load Pipeline")

    assert node.huggingface_pipeline_params.parameter_names == [
        "memory_optimization_strategy",
        "attention_slicing",
        "vae_slicing",
        "vae_tiling",
        "transformer_layerwise_casting",
        "cpu_offload_strategy",
    ]
    assert node.get_parameter_by_name("memory_optimization_strategy") is not None
    assert node.get_parameter_by_name("attention_slicing") is not None
    assert node.get_parameter_by_name("vae_slicing") is not None
    assert node.get_parameter_by_name("vae_tiling") is not None
    assert node.get_parameter_by_name("transformer_layerwise_casting") is not None
    assert node.get_parameter_by_name("cpu_offload_strategy") is not None
    assert node.get_parameter_by_name("quantization_mode") is None


def test_load_pipeline_automatic_mode_hides_manual_optimization_parameters() -> None:
    node = LoadPipelineNode(name="Load Pipeline")

    node.set_parameter_value("memory_optimization_strategy", "Automatic")

    for parameter_name in node.huggingface_pipeline_params.parameter_names[1:]:
        parameter = node.get_parameter_by_name(parameter_name)
        assert parameter is not None
        assert parameter.ui_options["hide"] is True

    node.set_parameter_value("memory_optimization_strategy", "Manual")

    for parameter_name in node.huggingface_pipeline_params.parameter_names[1:]:
        parameter = node.get_parameter_by_name(parameter_name)
        assert parameter is not None
        assert parameter.ui_options["hide"] is False


def test_load_pipeline_sets_optimization_parameters_from_pipeline() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": True,
            "vae_slicing": True,
            "vae_tiling": True,
            "transformer_layerwise_casting": True,
            "cpu_offload_strategy": "Sequential",
            "quantization_mode": "int8",
        },
    )

    node._set_optimization_parameters_from_pipeline(pipeline)

    assert node.huggingface_pipeline_params.get_hf_pipeline_parameters() == {
        "memory_optimization_strategy": "Manual",
        "attention_slicing": True,
        "vae_slicing": True,
        "vae_tiling": True,
        "transformer_layerwise_casting": True,
        "cpu_offload_strategy": "Sequential",
    }


def test_load_pipeline_parameter_change_publishes_effective_pipeline() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
            "transformer_layerwise_casting": False,
            "cpu_offload_strategy": "None",
            "quantization_mode": "int8",
        },
    )
    node._source_pipeline = source_pipeline
    node._publish_pipeline = Mock()

    node.set_parameter_value("vae_slicing", True)

    node._publish_pipeline.assert_called_once()
    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs == {
        "memory_optimization_strategy": "Manual",
        "attention_slicing": False,
        "vae_slicing": True,
        "vae_tiling": False,
        "transformer_layerwise_casting": False,
        "cpu_offload_strategy": "None",
        "quantization_mode": "int8",
    }
    assert effective_pipeline.config_hash != source_pipeline.config_hash
    assert source_pipeline.optimization_kwargs["vae_slicing"] is False


def test_load_pipeline_clears_output_when_path_preparation_fails() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
    )
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._set_optimization_parameters_from_pipeline = Mock(side_effect=ValueError("invalid optimization"))
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()

    node._publish_pipeline.assert_called_once_with(None)
    assert node._source_pipeline is None


def test_load_pipeline_clears_output_when_process_loading_fails() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    node._source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="PreviousPipeline",
        config_hash="previous-hash",
    )
    node._load_pipeline_from_path = Mock(side_effect=ValueError("invalid pipeline source"))
    node._publish_pipeline = Mock()
    node._handle_failure_exception = Mock()

    node.process()

    node._publish_pipeline.assert_called_once_with(None)
    assert node._source_pipeline is None


def test_load_pipeline_path_selection_sets_parameters_and_publishes_pipeline() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": True,
            "vae_slicing": True,
            "vae_tiling": False,
            "transformer_layerwise_casting": False,
            "cpu_offload_strategy": "Model",
            "quantization_mode": "int8",
        },
    )
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()

    assert node.get_parameter_value("attention_slicing") is True
    assert node.get_parameter_value("vae_slicing") is True
    assert node.get_parameter_value("cpu_offload_strategy") == "Model"
    node._publish_pipeline.assert_called_once_with(source_pipeline)


def test_load_pipeline_process_preserves_initial_setup_parameters() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
            "transformer_layerwise_casting": False,
            "cpu_offload_strategy": "None",
        },
    )
    node.set_parameter_value("vae_slicing", True, initial_setup=True)
    node.set_parameter_value("cpu_offload_strategy", "Sequential", initial_setup=True)
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node.process()

    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["vae_slicing"] is True
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "Sequential"


def test_load_pipeline_baked_settings_populate_parameters(tmp_path: Path) -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    baked_pipeline = DiffusionPipelineArtifact.from_baked_config_dict(
        {
            "pipeline_name": "TestPipeline",
            "builder_module": "test.module",
            "builder_class_name": "TestParameters",
            "optimization": {
                "memory_optimization_strategy": "Manual",
                "attention_slicing": False,
                "vae_slicing": True,
                "vae_tiling": True,
                "transformer_layerwise_casting": False,
                "cpu_offload_strategy": "Model",
                "quantization_mode": "None",
            },
        },
        baked_path=tmp_path,
        config_hash="baked-hash",
    )
    node._load_pipeline_from_path = Mock(return_value=baked_pipeline)
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()

    assert node.get_parameter_value("vae_slicing") is True
    assert node.get_parameter_value("vae_tiling") is True
    assert node.get_parameter_value("cpu_offload_strategy") == "Model"


def test_load_pipeline_initial_setup_restores_automatic_mode_visibility() -> None:
    node = LoadPipelineNode(name="Load Pipeline")

    node.set_parameter_value("memory_optimization_strategy", "Automatic", initial_setup=True)

    for parameter_name in node.huggingface_pipeline_params.parameter_names[1:]:
        parameter = node.get_parameter_by_name(parameter_name)
        assert parameter is not None
        assert parameter.ui_options["hide"] is True


def test_load_pipeline_edited_settings_round_trip_through_recipe() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
            "transformer_layerwise_casting": False,
            "cpu_offload_strategy": "None",
            "quantization_mode": "int8",
        },
    )
    node._source_pipeline = source_pipeline
    node._publish_pipeline = Mock()

    node.set_parameter_value("vae_slicing", True)

    effective_pipeline = node._publish_pipeline.call_args.args[0]
    restored_pipeline = deserialize_pipeline_artifact(serialize_pipeline_artifact(effective_pipeline))
    assert restored_pipeline.optimization_kwargs["vae_slicing"] is True
    assert restored_pipeline.optimization_kwargs["quantization_mode"] == "int8"
