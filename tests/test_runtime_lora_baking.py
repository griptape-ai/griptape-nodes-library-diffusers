from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from modular_diffusion_nodes_library.artifact_utils import pipeline_baking
from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import (
    make_baked_artifact,
    plan_runtime_lora_copies,
    save_baked_pipeline,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import format_baked_pipeline_summary
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec


class _FakePipe:
    components: dict[str, Any] = {}
    dtype = "float32"

    def save_pretrained(self, folder: str | Path) -> None:
        destination = Path(folder)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model_index.json").write_text('{"_class_name": "TestPipeline"}\n', encoding="utf-8")


def _base_artifact() -> DiffusionPipelineArtifact:
    return DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestBuilder",
    )


def _write_generated_bake(
    folder: Path,
    *,
    marker_name: str = "old.txt",
    modular: bool = False,
) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    index_name = "modular_model_index.json" if modular else "model_index.json"
    (folder / index_name).write_text('{"_class_name": "TestPipeline"}\n', encoding="utf-8")
    (folder / "baked-config.yaml").write_text("pipeline_name: TestPipeline\n", encoding="utf-8")
    marker = folder / marker_name
    marker.write_text("old", encoding="utf-8")
    return marker


def _temporary_bake_dirs(destination: Path) -> list[Path]:
    return list(destination.parent.glob(f"{destination.name}.tmp-bake-*"))


def _is_temporary_bake_path(path: Path, destination: Path) -> bool:
    temporary_root = path.parent
    return (
        path.name == "bake"
        and temporary_root.parent == destination.parent
        and temporary_root.name.startswith(f"{destination.name}.tmp-bake-")
    )


def _artifact_with_runtime_loras(
    loras: list[tuple[Path, float]],
    *,
    fused_loras: dict[str, float] | None = None,
    build_data: dict[str, Any] | None = None,
) -> DiffusionPipelineArtifact:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestBuilder",
        build_data=build_data,
        loras=fused_loras,
    )
    specs = {str(path): LoraSpec(path=str(path), weight=weight) for path, weight in loras}
    return artifact.with_additional_runtime_adapter_steps([LoraPipelineRuntimeAdapterStep(specs)])


@pytest.mark.parametrize("modular", [False, True])
def test_full_pipeline_replaces_existing_bake_and_removes_backup(tmp_path: Path, modular: bool) -> None:
    destination = tmp_path / "baked"
    _write_generated_bake(destination, modular=modular)

    save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert (destination / "model_index.json").is_file()
    assert not (destination / "old.txt").exists()
    assert not destination.with_name("baked.previous-bake").exists()
    assert _temporary_bake_dirs(destination) == []


@pytest.mark.parametrize("index_name", [None, "model_index.json", "modular_model_index.json"])
def test_full_pipeline_refuses_unrelated_nonempty_destination(tmp_path: Path, index_name: str | None) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()
    marker = destination / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")
    if index_name is not None:
        (destination / index_name).write_text('{"_class_name": "UnrelatedModel"}\n', encoding="utf-8")
    pipe = _FakePipe()
    pipe.save_pretrained = Mock(wraps=pipe.save_pretrained)

    with pytest.raises(FileExistsError, match="nonempty and is not a generated Full Pipeline bake"):
        save_baked_pipeline(pipe, _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "keep"
    pipe.save_pretrained.assert_not_called()
    assert not destination.with_name("baked.previous-bake").exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_allows_empty_destination(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()

    save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert (destination / "model_index.json").is_file()
    assert (destination / "baked-config.yaml").is_file()
    assert not destination.with_name("baked.previous-bake").exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_refuses_symlink_destination_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")
    destination = tmp_path / "baked"
    destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(NotADirectoryError, match="not a regular directory"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert destination.is_symlink()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_preserves_unrelated_legacy_temp_directory(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    unrelated_temp = destination.with_name("baked.tmp-bake")
    unrelated_temp.mkdir()
    marker = unrelated_temp / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")

    save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert (destination / "model_index.json").is_file()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_revalidates_destination_before_installing(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    marker = destination / "unrelated.txt"
    pipe = _FakePipe()
    real_save = pipe.save_pretrained

    def save_and_create_destination(folder: str | Path) -> None:
        real_save(folder)
        destination.mkdir()
        marker.write_text("keep", encoding="utf-8")

    pipe.save_pretrained = Mock(side_effect=save_and_create_destination)

    with pytest.raises(FileExistsError, match="nonempty and is not a generated Full Pipeline bake"):
        save_baked_pipeline(pipe, _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not destination.with_name("baked.previous-bake").exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_revalidates_destination_after_moving_to_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "baked"
    marker = _write_generated_bake(destination)
    backup = destination.with_name("baked.previous-bake")
    real_replace = pipeline_baking.os.replace

    def invalidate_after_move(source: str | Path, target: str | Path) -> None:
        real_replace(source, target)
        if Path(source) == destination and Path(target) == backup:
            (backup / "baked-config.yaml").unlink()

    monkeypatch.setattr(pipeline_baking.os, "replace", invalidate_after_move)

    with pytest.raises(FileExistsError, match="previous destination"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "old"
    assert not backup.exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_restores_existing_bake_when_installation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "baked"
    marker = _write_generated_bake(destination)
    real_replace = pipeline_baking.os.replace

    def fail_install(source: str | Path, target: str | Path) -> None:
        if _is_temporary_bake_path(Path(source), destination) and Path(target) == destination:
            raise OSError("installation failed")
        real_replace(source, target)

    monkeypatch.setattr(pipeline_baking.os, "replace", fail_install)

    with pytest.raises(OSError, match="installation failed"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "old"
    assert not destination.with_name("baked.previous-bake").exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_retains_backup_when_installation_and_restore_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "baked"
    _write_generated_bake(destination)
    backup = destination.with_name("baked.previous-bake")
    real_replace = pipeline_baking.os.replace

    def fail_install_and_restore(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if _is_temporary_bake_path(source_path, destination) and target_path == destination:
            raise OSError("installation failed")
        if source_path == backup and target_path == destination:
            raise OSError("restore failed")
        real_replace(source, target)

    monkeypatch.setattr(pipeline_baking.os, "replace", fail_install_and_restore)

    with pytest.raises(RuntimeError, match="previous bake remains"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert not destination.exists()
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_refuses_existing_destination_and_recovery_backup(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()
    (destination / "current.txt").write_text("current", encoding="utf-8")
    backup = destination.with_name("baked.previous-bake")
    backup.mkdir()
    (backup / "previous.txt").write_text("previous", encoding="utf-8")

    with pytest.raises(FileExistsError, match="Preserving both directories"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert (destination / "current.txt").read_text(encoding="utf-8") == "current"
    assert (backup / "previous.txt").read_text(encoding="utf-8") == "previous"
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_recovers_orphaned_previous_bake_before_validating_new_save(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    backup = destination.with_name("baked.previous-bake")
    marker = _write_generated_bake(backup, marker_name="previous.txt")
    marker.write_text("previous", encoding="utf-8")
    invalid_artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data_error="invalid build data",
    )

    with pytest.raises(ValueError, match="invalid build data"):
        save_baked_pipeline(_FakePipe(), invalid_artifact, destination)

    assert (destination / marker.name).read_text(encoding="utf-8") == "previous"
    assert not backup.exists()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_refuses_unrecognized_orphaned_recovery_directory(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    backup = destination.with_name("baked.previous-bake")
    backup.mkdir()
    marker = backup / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")
    pipe = _FakePipe()
    pipe.save_pretrained = Mock(wraps=pipe.save_pretrained)

    with pytest.raises(FileExistsError, match="recovery path"):
        save_baked_pipeline(pipe, _base_artifact(), destination)

    assert not destination.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    pipe.save_pretrained.assert_not_called()
    assert _temporary_bake_dirs(destination) == []


def test_full_pipeline_copies_runtime_loras_and_rebuilds_adapter_step(tmp_path: Path) -> None:
    style = tmp_path / "sources" / "style.safetensors"
    hdr = tmp_path / "sources" / "hdr.safetensors"
    style.parent.mkdir()
    style.write_bytes(b"style-lora")
    hdr.write_bytes(b"hdr-lora")
    artifact = _artifact_with_runtime_loras(
        [(style, 0.75), (hdr, 1.25)],
        fused_loras={str(tmp_path / "fused.safetensors"): 0.5},
    )
    destination = tmp_path / "baked"

    save_baked_pipeline(_FakePipe(), artifact, destination)

    config = yaml.safe_load((destination / "baked-config.yaml").read_text(encoding="utf-8"))
    assert config["runtime_loras"] == [
        {"path": "loras/style.safetensors", "weight": 0.75},
        {"path": "loras/hdr.safetensors", "weight": 1.25},
    ]
    assert (destination / "loras" / style.name).read_bytes() == style.read_bytes()
    assert (destination / "loras" / hdr.name).read_bytes() == hdr.read_bytes()

    loaded = make_baked_artifact(destination)

    assert loaded.loras == {}
    assert loaded.runtime_loras() == {
        str((destination / "loras" / style.name).resolve()): LoraSpec(
            path=str((destination / "loras" / style.name).resolve()),
            weight=0.75,
        ),
        str((destination / "loras" / hdr.name).resolve()): LoraSpec(
            path=str((destination / "loras" / hdr.name).resolve()),
            weight=1.25,
        ),
    }
    (step,) = loaded.runtime_adapter_steps()
    assert isinstance(step, LoraPipelineRuntimeAdapterStep)

    summary = format_baked_pipeline_summary(loaded)
    assert "Runtime LoRAs" in summary
    assert "Path: loras/style.safetensors" in summary
    assert "Weight: 0.75" in summary
    assert "Path: loras/hdr.safetensors" in summary
    assert "Weight: 1.25" in summary


def test_full_pipeline_load_names_missing_runtime_lora_and_adapter(tmp_path: Path) -> None:
    source = tmp_path / "style.safetensors"
    source.write_bytes(b"style-lora")
    destination = tmp_path / "baked"
    save_baked_pipeline(_FakePipe(), _artifact_with_runtime_loras([(source, 0.8)]), destination)
    (destination / "loras" / source.name).unlink()

    with pytest.raises(
        FileNotFoundError,
        match=r"Runtime LoRA adapter 'style_[0-9a-f]{8}'.*'loras/style\.safetensors' is missing",
    ):
        make_baked_artifact(destination)


def test_runtime_lora_copy_plan_rejects_missing_source(tmp_path: Path) -> None:
    missing = tmp_path / "missing.safetensors"
    artifact = _artifact_with_runtime_loras([(missing, 1.0)])

    with pytest.raises(
        FileNotFoundError,
        match=r"Runtime LoRA adapter 'missing_[0-9a-f]{8}'.*file is missing",
    ):
        plan_runtime_lora_copies(artifact)


def test_runtime_lora_copy_plan_rejects_duplicate_destination_names(tmp_path: Path) -> None:
    first = tmp_path / "first" / "style.safetensors"
    second = tmp_path / "second" / "style.safetensors"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    artifact = _artifact_with_runtime_loras([(first, 0.5), (second, 1.0)])

    with pytest.raises(ValueError) as exc_info:
        plan_runtime_lora_copies(artifact)

    assert str(exc_info.value) == (
        "Runtime LoRA filename collision\n\n"
        "Full Pipeline saves runtime LoRAs in one 'loras/' folder. Rename or copy one source file to a unique "
        "filename, update its Load LoRA node, and retry.\n\n"
        "Destination:\n  loras/style.safetensors\n\n"
        f"Source files:\n  1. {first.resolve()}\n  2. {second.resolve()}"
    )


def test_full_pipeline_preserves_symlink_filename(tmp_path: Path) -> None:
    source = tmp_path / "models" / "original.safetensors"
    source.parent.mkdir()
    source.write_bytes(b"style-lora")
    alias = tmp_path / "named-style.safetensors"
    alias.symlink_to(source)
    destination = tmp_path / "baked"

    save_baked_pipeline(_FakePipe(), _artifact_with_runtime_loras([(alias, 1.0)]), destination)

    config = yaml.safe_load((destination / "baked-config.yaml").read_text(encoding="utf-8"))
    assert config["runtime_loras"] == [{"path": "loras/named-style.safetensors", "weight": 1.0}]
    assert (destination / "loras" / alias.name).read_bytes() == source.read_bytes()


def test_baked_runtime_lora_path_must_be_relative(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    source = tmp_path / "style.safetensors"
    source.write_bytes(b"style-lora")
    save_baked_pipeline(_FakePipe(), _artifact_with_runtime_loras([(source, 1.0)]), destination)
    config_path = destination / "baked-config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["runtime_loras"][0]["path"] = str(source)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be relative"):
        make_baked_artifact(destination)


def test_runtime_lora_is_allowed_with_gguf_denoiser(tmp_path: Path) -> None:
    lora = tmp_path / "style.safetensors"
    lora.write_bytes(b"style-lora")
    gguf = tmp_path / "transformer.gguf"
    gguf.write_bytes(b"GGUF")
    override = ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path=str(gguf),
    )
    artifact = _artifact_with_runtime_loras(
        [(lora, 0.8)],
        build_data={"_component_overrides": {"transformer": override}},
    )

    assert plan_runtime_lora_copies(artifact)[0].source_path == lora.resolve()
