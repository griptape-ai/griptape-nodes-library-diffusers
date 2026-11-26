from __future__ import annotations

from modular_diffusion_nodes_library.nodes.load_component_node import LoadComponent


def _node() -> LoadComponent:
    return LoadComponent(name="Load Component", metadata={"library": "Diffusion"})


def test_changing_the_file_types_tells_the_editor() -> None:
    node = _node()
    node.emit_parameter_changes()

    node.set_parameter_value("component", "Text Encoder", initial_setup=True)

    # The picker only narrows in the editor if the parameter is queued for an update. Setting the trait
    # without writing the new options through `ui_options` left it unqueued, so the old types stayed.
    assert "file_path" in [parameter.name for parameter in node._tracked_parameters]  # noqa: SLF001
    file_path = node.get_parameter_by_name("file_path")
    assert file_path is not None
    assert file_path.ui_options["fileSystemPicker"]["fileTypes"] == [".gguf"]


def test_config_is_hidden_for_a_text_encoder_and_shown_for_other_components() -> None:
    node = _node()

    node.set_parameter_value("component", "Text Encoder", initial_setup=True)
    node._apply_source_type_visibility("Single File")  # noqa: SLF001
    config_source = node.get_parameter_by_name("config_source")
    assert config_source is not None
    assert config_source.ui_options["hide"]

    node.set_parameter_value("component", "VAE", initial_setup=True)
    node._apply_source_type_visibility("Single File")  # noqa: SLF001
    assert not config_source.ui_options["hide"]


def test_text_encoder_artifact_ignores_a_config_left_over_from_another_component() -> None:
    node = _node()
    node.set_parameter_value("component", "Text Encoder", initial_setup=True)
    node.set_parameter_value("file_path", "/models/text_encoder-Q4_K_M.gguf", initial_setup=True)
    node.set_parameter_value("config_source", "/models/transformer/config.json", initial_setup=True)

    artifact = node._build_single_file_artifact("text_encoder")  # noqa: SLF001

    assert artifact is not None
    assert artifact.config_source is None


def test_picking_text_encoder_in_the_editor_hides_config() -> None:
    node = _node()
    config_source = node.get_parameter_by_name("config_source")
    assert config_source is not None
    assert not config_source.ui_options["hide"]

    node.set_parameter_value("component", "Text Encoder")

    assert config_source.ui_options["hide"]

    node.set_parameter_value("component", "VAE")

    assert not config_source.ui_options["hide"]
