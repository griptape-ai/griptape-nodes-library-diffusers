from __future__ import annotations

from pathlib import Path
from typing import Any

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


def test_full_pipeline_replaces_existing_bake_and_removes_backup(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")

    save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert (destination / "model_index.json").is_file()
    assert not (destination / "old.txt").exists()
    assert not destination.with_name("baked.previous-bake").exists()
    assert not destination.with_name("baked.tmp-bake").exists()


def test_full_pipeline_restores_existing_bake_when_installation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()
    marker = destination / "old.txt"
    marker.write_text("old", encoding="utf-8")
    tmp_bake = destination.with_name("baked.tmp-bake")
    real_replace = pipeline_baking.os.replace

    def fail_install(source: str | Path, target: str | Path) -> None:
        if Path(source) == tmp_bake and Path(target) == destination:
            raise OSError("installation failed")
        real_replace(source, target)

    monkeypatch.setattr(pipeline_baking.os, "replace", fail_install)

    with pytest.raises(OSError, match="installation failed"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert marker.read_text(encoding="utf-8") == "old"
    assert not destination.with_name("baked.previous-bake").exists()
    assert not tmp_bake.exists()


def test_full_pipeline_retains_backup_when_installation_and_restore_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "baked"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    tmp_bake = destination.with_name("baked.tmp-bake")
    backup = destination.with_name("baked.previous-bake")
    real_replace = pipeline_baking.os.replace

    def fail_install_and_restore(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if source_path == tmp_bake and target_path == destination:
            raise OSError("installation failed")
        if source_path == backup and target_path == destination:
            raise OSError("restore failed")
        real_replace(source, target)

    monkeypatch.setattr(pipeline_baking.os, "replace", fail_install_and_restore)

    with pytest.raises(RuntimeError, match="previous bake remains"):
        save_baked_pipeline(_FakePipe(), _base_artifact(), destination)

    assert not destination.exists()
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"
    assert not tmp_bake.exists()


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
    assert not destination.with_name("baked.tmp-bake").exists()


def test_full_pipeline_recovers_orphaned_previous_bake_before_validating_new_save(tmp_path: Path) -> None:
    destination = tmp_path / "baked"
    backup = destination.with_name("baked.previous-bake")
    backup.mkdir()
    marker = backup / "previous.txt"
    marker.write_text("previous", encoding="utf-8")
    invalid_artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data_error="invalid build data",
    )

    with pytest.raises(ValueError, match="invalid build data"):
        save_baked_pipeline(_FakePipe(), invalid_artifact, destination)

    assert (destination / marker.name).read_text(encoding="utf-8") == "previous"
    assert not backup.exists()
    assert not destination.with_name("baked.tmp-bake").exists()


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
