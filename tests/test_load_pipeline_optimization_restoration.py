from unittest.mock import Mock

import pytest

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode


def _pipeline_with_optimization(
    *,
    name: str = "TestPipeline",
    config_hash: str = "source-hash",
    **overrides: bool | str,
) -> DiffusionPipelineArtifact:
    optimization_kwargs: dict[str, bool | str] = {
        "memory_optimization_strategy": "Manual",
        "attention_slicing": False,
        "vae_slicing": False,
        "vae_tiling": False,
        "transformer_layerwise_casting": False,
        "cpu_offload_strategy": "None",
    }
    optimization_kwargs.update(overrides)
    return DiffusionPipelineArtifact(
        pipeline_name=name,
        config_hash=config_hash,
        optimization_kwargs=optimization_kwargs,
    )


def test_missing_recipe_preserves_restored_overrides() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    node.set_parameter_value(
        "_optimization_overrides",
        {"cpu_offload_strategy": "Sequential"},
        initial_setup=True,
    )
    node._load_pipeline_from_path = Mock(side_effect=FileNotFoundError("missing recipe"))
    node._publish_pipeline = Mock()
    node._handle_failure_exception = Mock()

    node.process()

    assert node.get_parameter_value("_optimization_overrides") == {
        "cpu_offload_strategy": "Sequential",
    }
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is False
    assert node._source_pipeline is None
    node._publish_pipeline.assert_called_once_with(None)


def test_recovers_after_restored_recipe_becomes_available() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(cpu_offload_strategy="Sequential")
    node.set_parameter_value(
        "_optimization_overrides",
        {"cpu_offload_strategy": "Sequential"},
        initial_setup=True,
    )
    node._load_pipeline_from_path = Mock(side_effect=[FileNotFoundError("missing recipe"), source_pipeline])
    node._publish_pipeline = Mock()
    node._handle_failure_exception = Mock()

    node.process()
    node._publish_pipeline.reset_mock()
    node.process()

    assert node.get_parameter_value("_optimization_overrides") == {}
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is True
    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "Sequential"


@pytest.mark.parametrize(
    "restored_overrides",
    [
        None,
        {"unknown_setting": True},
        {"cpu_offload_strategy": "invalid"},
        {"vae_slicing": "true"},
    ],
)
def test_reports_and_ignores_invalid_restored_overrides(restored_overrides: object) -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    node._set_status_results = Mock()

    node.set_parameter_value("_optimization_overrides", restored_overrides, initial_setup=True)

    assert node.get_parameter_value("_optimization_overrides") == {}
    node._set_status_results.assert_called_once()
    assert node._set_status_results.call_args.kwargs["was_successful"] is False
    assert (
        "Could not restore Load Pipeline optimization overrides"
        in node._set_status_results.call_args.kwargs["result_details"]
    )


def test_restored_recipe_values_do_not_become_overrides() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(
        attention_slicing=True,
        vae_tiling=True,
        cpu_offload_strategy="Model",
        quantization_mode="int8",
    )
    node.set_parameter_value("vae_slicing", True, initial_setup=True)
    node.set_parameter_value("cpu_offload_strategy", "Sequential", initial_setup=True)
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node.process()

    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert node.get_parameter_value("attention_slicing") is True
    assert node.get_parameter_value("vae_tiling") is True
    assert effective_pipeline.optimization_kwargs["vae_slicing"] is False
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "Model"
    assert effective_pipeline.optimization_kwargs["attention_slicing"] is True
    assert effective_pipeline.optimization_kwargs["vae_tiling"] is True
    assert effective_pipeline.optimization_kwargs["quantization_mode"] == "int8"


def test_restores_explicit_override_equal_to_parameter_default() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(cpu_offload_strategy="Sequential")
    node.set_parameter_value(
        "_optimization_overrides",
        {"cpu_offload_strategy": "None"},
        initial_setup=True,
    )
    node.set_parameter_value("cpu_offload_strategy", "None", initial_setup=True)
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node.process()

    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert node.get_parameter_value("cpu_offload_strategy") == "None"
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "None"


def test_late_override_restore_republishes_effective_pipeline() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(cpu_offload_strategy="Sequential")
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()
    node._publish_from_committed_path()
    node._publish_pipeline.reset_mock()

    node.set_parameter_value(
        "_optimization_overrides",
        {"cpu_offload_strategy": "None"},
        initial_setup=True,
    )

    assert node.get_parameter_value("cpu_offload_strategy") == "None"
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is False
    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "None"


def test_late_override_restore_keeps_valid_state_when_publish_fails() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(cpu_offload_strategy="Sequential")
    node._source_pipeline = source_pipeline
    node._publish_pipeline = Mock(side_effect=RuntimeError("publish rejected"))
    node._set_status_results = Mock()

    node.set_parameter_value(
        "_optimization_overrides",
        {"cpu_offload_strategy": "None"},
        initial_setup=True,
    )

    assert node.get_parameter_value("_optimization_overrides") == {
        "cpu_offload_strategy": "None",
    }
    assert node.get_parameter_value("cpu_offload_strategy") == "None"
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is False
    node._set_status_results.assert_called_once()
    assert node._set_status_results.call_args.kwargs["was_successful"] is False


def test_process_preserves_preload_user_edit() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(
        attention_slicing=True,
        vae_tiling=True,
        cpu_offload_strategy="Sequential",
    )
    node.set_parameter_value("vae_slicing", True)
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node.process()

    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["vae_slicing"] is True
    assert effective_pipeline.optimization_kwargs["attention_slicing"] is True
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "Sequential"


def test_preload_edit_equal_to_default_remains_override() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    source_pipeline = _pipeline_with_optimization(cpu_offload_strategy="Sequential")
    node.set_parameter_value("cpu_offload_strategy", "None")
    node._load_pipeline_from_path = Mock(return_value=source_pipeline)
    node._publish_pipeline = Mock()

    node.process()

    assert node.get_parameter_value("_optimization_overrides") == {
        "cpu_offload_strategy": "None",
    }
    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "None"


def test_new_recipe_replaces_recipe_controlled_parameters() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    first_pipeline = _pipeline_with_optimization(
        name="FirstPipeline",
        config_hash="first-hash",
        attention_slicing=True,
        vae_slicing=True,
        cpu_offload_strategy="Sequential",
    )
    second_pipeline = _pipeline_with_optimization(
        name="SecondPipeline",
        config_hash="second-hash",
        vae_tiling=True,
        transformer_layerwise_casting=True,
        cpu_offload_strategy="Model",
    )
    node._load_pipeline_from_path = Mock(side_effect=[first_pipeline, second_pipeline])
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()
    node._publish_from_committed_path()

    assert node.get_parameter_value("attention_slicing") is False
    assert node.get_parameter_value("vae_slicing") is False
    assert node.get_parameter_value("vae_tiling") is True
    assert node.get_parameter_value("transformer_layerwise_casting") is True
    assert node.get_parameter_value("cpu_offload_strategy") == "Model"


def test_new_recipe_preserves_only_user_edited_parameters() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    first_pipeline = _pipeline_with_optimization(name="FirstPipeline", config_hash="first-hash")
    second_pipeline = _pipeline_with_optimization(
        name="SecondPipeline",
        config_hash="second-hash",
        attention_slicing=True,
        vae_tiling=True,
        transformer_layerwise_casting=True,
        cpu_offload_strategy="Model",
    )
    node._load_pipeline_from_path = Mock(side_effect=[first_pipeline, second_pipeline])
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()
    node.set_parameter_value("vae_slicing", True)
    node._publish_from_committed_path()

    assert node.get_parameter_value("vae_slicing") is True
    assert node.get_parameter_value("attention_slicing") is True
    assert node.get_parameter_value("vae_tiling") is True
    assert node.get_parameter_value("transformer_layerwise_casting") is True
    assert node.get_parameter_value("cpu_offload_strategy") == "Model"


def test_new_recipe_clears_matching_user_override() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    first_pipeline = _pipeline_with_optimization(
        name="FirstPipeline",
        config_hash="first-hash",
        cpu_offload_strategy="Sequential",
    )
    second_pipeline = _pipeline_with_optimization(
        name="SecondPipeline",
        config_hash="second-hash",
        cpu_offload_strategy="None",
    )
    node._load_pipeline_from_path = Mock(side_effect=[first_pipeline, second_pipeline])
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()
    node.set_parameter_value("cpu_offload_strategy", "None")
    node._publish_from_committed_path()

    assert node.get_parameter_value("_optimization_overrides") == {}
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is True
    effective_pipeline = node._publish_pipeline.call_args.args[0]
    assert effective_pipeline.optimization_kwargs["cpu_offload_strategy"] == "None"


def test_recipe_publish_failure_keeps_reconciled_settings() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    first_pipeline = _pipeline_with_optimization(
        name="FirstPipeline",
        config_hash="first-hash",
        cpu_offload_strategy="Sequential",
    )
    second_pipeline = _pipeline_with_optimization(
        name="SecondPipeline",
        config_hash="second-hash",
        cpu_offload_strategy="None",
    )
    node._load_pipeline_from_path = Mock(side_effect=[first_pipeline, second_pipeline])
    node._publish_pipeline = Mock()

    node._publish_from_committed_path()
    node.set_parameter_value("cpu_offload_strategy", "None")
    node._publish_pipeline.reset_mock()
    node._publish_pipeline.side_effect = [RuntimeError("publish rejected"), None]

    node._publish_from_committed_path()

    assert node.get_parameter_value("_optimization_overrides") == {}
    assert node.get_parameter_value("cpu_offload_strategy") == "None"
    reset_button = node.get_parameter_by_name("_optimization_overrides")
    assert reset_button is not None
    assert reset_button.ui_options["hide"] is True
    assert node._source_pipeline is None
    assert node._publish_pipeline.call_args_list[-1].args == (None,)


def test_initial_setup_restores_automatic_mode_visibility() -> None:
    node = LoadPipelineNode(name="Load Pipeline")

    node.set_parameter_value("memory_optimization_strategy", "Automatic", initial_setup=True)

    for parameter_name in node.huggingface_pipeline_params.parameter_names[1:]:
        parameter = node.get_parameter_by_name(parameter_name)
        assert parameter is not None
        assert parameter.ui_options["hide"] is True
