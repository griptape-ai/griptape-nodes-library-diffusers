from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import _write_baked_config
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import serialize_pipeline_artifact
from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode


def _load_from_path(path: Path) -> tuple[MagicMock, DiffusionPipelineArtifact]:
    node = MagicMock()
    node.name = "Load Pipeline"
    node.get_parameter_value.return_value = str(path)
    with (
        patch(
            "modular_diffusion_nodes_library.nodes.load_pipeline_node.get_workspace_path",
            return_value=path.parent,
        ),
        patch(
            "modular_diffusion_nodes_library.nodes.load_pipeline_node.GriptapeNodes.ConfigManager",
            side_effect=AssertionError("worker execution must not access ConfigManager"),
        ),
        patch(
            "modular_diffusion_nodes_library.nodes.load_pipeline_node.resolve_workspace_path",
            return_value=path,
        ),
    ):
        pipeline = LoadPipelineNode._load_pipeline_from_path(node)
    return node, pipeline


def test_file_picker_accepts_only_yaml_files() -> None:
    node = LoadPipelineNode(name="Load Pipeline")
    parameter = node.get_parameter_by_name("file_path")

    assert parameter is not None
    picker = next(iter(parameter.find_elements_by_type(FileSystemPicker)))
    assert picker.allow_files is True
    assert picker.allow_directories is False
    assert picker.file_types == [".yaml"]


def test_loads_config_only_recipe_from_custom_yaml_filename(tmp_path: Path) -> None:
    recipe_path = tmp_path / "custom-flux-recipe.yaml"
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="config-hash",
        builder_module="test.module",
        builder_class_name="TestBuilder",
    )
    recipe_path.write_text(serialize_pipeline_artifact(artifact), encoding="utf-8")

    node, loaded = _load_from_path(recipe_path)

    assert loaded.to_recipe_dict() == artifact.to_recipe_dict()
    assert loaded.config_hash is not None
    assert node._loaded_from_baked_dir is False


@pytest.mark.parametrize("index_filename", ["model_index.json", "modular_model_index.json"])
def test_loads_baked_manifest_through_containing_folder(tmp_path: Path, index_filename: str) -> None:
    baked_folder = tmp_path / "baked"
    baked_folder.mkdir()
    (baked_folder / index_filename).write_text("{}\n", encoding="utf-8")
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        builder_module="test.module",
        builder_class_name="TestBuilder",
    )
    _write_baked_config(baked_folder, artifact, "float32")

    node, loaded = _load_from_path(baked_folder / "baked-config.yaml")

    assert loaded.build_data["_baked_path"] == str(baked_folder.resolve())
    assert node._loaded_from_baked_dir is True
    assert LoadPipelineNode._success_status_details(node, loaded).startswith("Loaded baked pipeline\n\n")


def test_rebaked_manifest_changes_loaded_cache_key(tmp_path: Path) -> None:
    baked_folder = tmp_path / "baked"
    baked_folder.mkdir()
    (baked_folder / "model_index.json").write_text("{}\n", encoding="utf-8")
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        builder_module="test.module",
        builder_class_name="TestBuilder",
    )
    _write_baked_config(baked_folder, artifact, "float32")
    manifest_path = baked_folder / "baked-config.yaml"

    _, first = _load_from_path(manifest_path)
    manifest_path.write_text(manifest_path.read_text(encoding="utf-8") + "rebake_marker: true\n", encoding="utf-8")
    _, second = _load_from_path(manifest_path)

    assert first.config_hash != second.config_hash


@pytest.mark.parametrize("filename", ["config.json", "config.yml"])
def test_rejects_non_yaml_file_extensions(tmp_path: Path, filename: str) -> None:
    source = tmp_path / filename
    source.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"must point to a \.yaml file"):
        _load_from_path(source)


def test_rejects_folder_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"must point to a \.yaml file"):
        _load_from_path(tmp_path)
