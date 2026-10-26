"""Bake copies GGUF; load injects that module into from_pretrained."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from diffusers import Flux2Transformer2DModel  # type: ignore[reportMissingImports]
from torch import nn
from transformers import AutoConfig, CLIPTextConfig  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.baked_file_overrides import (
    component_refuses_official_save,
    is_gguf_quantizer_component,
    is_torchao_quantizer_component,
    skip_official_save_slots,
)
from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import make_baked_artifact, save_baked_pipeline
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    QuantizationStatus,
    collect_quantization_status,
)
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import TextEncoderComponentArtifact
from modular_diffusion_nodes_library.component_loading.config_resolver import resolve_config_path
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)


class _MarkerSlot(nn.Module):
    def save_pretrained(self, folder: str | Path, **kwargs: Any) -> None:
        dest = Path(folder)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "saved.txt").write_text("ok\n", encoding="utf-8")


class _RefusingSlot:
    def __init__(self) -> None:
        self.hf_quantizer = MagicMock()
        self.hf_quantizer.is_serializable.return_value = False
        self.save_calls = 0

    def save_pretrained(self, folder: str | Path, **kwargs: Any) -> None:
        self.save_calls += 1
        raise ValueError("The model is quantized with GGUFQuantizer and is not serializable")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}


class _PropertyQuantizer:
    @property
    def is_serializable(self) -> bool:
        return False


class _PropertyRefusingSlot:
    """Diffusers GGUFQuantizer exposes is_serializable as a property, not a method."""

    def __init__(self) -> None:
        self.hf_quantizer = _PropertyQuantizer()
        self.save_calls = 0

    def save_pretrained(self, folder: str | Path, **kwargs: Any) -> None:
        self.save_calls += 1
        raise ValueError("The model is quantized with GGUFQuantizer and is not serializable")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}


class _GgufQuantizer:
    @property
    def is_serializable(self) -> bool:
        return False


class _GgufQuantizerSlot:
    def __init__(self) -> None:
        self.hf_quantizer = _GgufQuantizer()
        self.save_calls = 0

    def save_pretrained(self, folder: str | Path, **kwargs: Any) -> None:
        self.save_calls += 1
        raise ValueError("The model is quantized with GGUFQuantizer and is not serializable")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}


class _TorchAoQuantizer:
    def is_serializable(self) -> bool:
        return True


class _TorchAoQuantizerSlot:
    def __init__(self) -> None:
        self.hf_quantizer = _TorchAoQuantizer()
        self.save_calls = 0

    def save_pretrained(self, folder: str | Path, **kwargs: Any) -> None:
        self.save_calls += 1
        raise ValueError("Unsupported tensor type: AffineQuantizedTensor")

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {}


class _IndexLastPipe:
    def __init__(self, components: dict[str, Any]) -> None:
        self.components = components
        for name, component in components.items():
            setattr(self, name, component)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def save_pretrained(self, save_directory: str | Path) -> None:
        folder = Path(save_directory)
        folder.mkdir(parents=True, exist_ok=True)
        for name, component in self.components.items():
            component.save_pretrained(folder / name)
        self.save_config(folder)
        self.finished_official_save = True

    def save_config(self, save_directory: str | Path) -> None:
        folder = Path(save_directory)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "model_index.json").write_text(
            json.dumps(
                {"_class_name": "FluxPipeline", "transformer": ["diffusers", "FluxTransformer2DModel"]}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )


def _gguf_override(path: Path, slot: str = "transformer") -> ModelComponentArtifact:
    return ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component=slot,
        torch_dtype="bfloat16",
        file_path=str(path),
        config_source=str(path.parent),
    )


def _write_dummy_gguf(folder: Path, name: str = "weights.gguf") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(b"GGUF-dummy\n")
    (folder / "config.json").write_text("{}\n", encoding="utf-8")
    return path


def test_bake_noops_gguf_save_copies_file_and_records_override(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    refusing = _RefusingSlot()
    pipe = _IndexLastPipe({"transformer": refusing, "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": _gguf_override(src)}},
    )
    dest = tmp_path / "baked"
    save_baked_pipeline(pipe, artifact, dest)

    assert refusing.save_calls == 0
    copied = dest / "transformer" / src.name
    assert copied.is_file()
    assert copied.read_bytes() == src.read_bytes()
    assert (dest / "transformer" / "config.json").is_file()
    assert (dest / "vae" / "saved.txt").is_file()
    assert not any(dest.joinpath("transformer").glob("*.safetensors"))

    config = json.loads((dest / "baked-config.json").read_text(encoding="utf-8"))
    recorded = config["file_overrides"]["transformer"]
    assert recorded["file_path"] == f"transformer/{src.name}"
    assert recorded["source_type"] == "single_file"


def test_bake_round_trips_gguf_text_encoder_override(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src", "text-encoder.gguf")
    override = TextEncoderComponentArtifact(
        load_id="text-encoder-gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(src.parent),
    )
    pipe = _IndexLastPipe({"text_encoder": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"text_encoder": override}},
    )
    dest = tmp_path / "baked"

    save_baked_pipeline(pipe, artifact, dest)

    manifest = json.loads((dest / "baked-config.json").read_text(encoding="utf-8"))
    recorded = manifest["file_overrides"]["text_encoder"]
    loaded = make_baked_artifact(dest)
    loaded_override = loaded.build_data["_component_overrides"]["text_encoder"]

    assert recorded["artifact_type"] == "TextEncoderComponentArtifact"
    assert recorded["file_path"] == "text_encoder/text-encoder.gguf"
    assert recorded["config_source"] == "text_encoder/config.json"
    assert isinstance(loaded_override, TextEncoderComponentArtifact)
    assert Path(loaded_override.file_path).resolve() == (dest / recorded["file_path"]).resolve()
    assert Path(loaded_override.config_source).resolve() == (dest / recorded["config_source"]).resolve()


def test_bake_preserves_explicit_text_encoder_config_filename(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src", "text-encoder.gguf")
    custom_config = src.parent / "text_encoder.json"
    custom_config.write_text('{"marker": "explicit"}\n', encoding="utf-8")
    override = TextEncoderComponentArtifact(
        load_id="text-encoder-gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(custom_config),
    )
    pipe = _IndexLastPipe({"text_encoder": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"text_encoder": override}},
    )
    dest = tmp_path / "baked"

    save_baked_pipeline(pipe, artifact, dest)

    slot_dir = dest / "text_encoder"
    manifest = json.loads((dest / "baked-config.json").read_text(encoding="utf-8"))
    recorded = manifest["file_overrides"]["text_encoder"]
    loaded = make_baked_artifact(dest)
    loaded_override = loaded.build_data["_component_overrides"]["text_encoder"]

    assert json.loads((slot_dir / "text_encoder.json").read_text(encoding="utf-8")) == {"marker": "explicit"}
    assert not (slot_dir / "config.json").exists()
    assert recorded["config_source"] == "text_encoder/text_encoder.json"
    assert isinstance(loaded_override, TextEncoderComponentArtifact)
    assert Path(loaded_override.config_source).resolve() == (slot_dir / "text_encoder.json").resolve()


def test_text_encoder_single_file_loads_explicit_config_before_transformers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _write_dummy_gguf(tmp_path / "src", "text-encoder.gguf")
    custom_config = src.parent / "text_encoder.json"
    CLIPTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
    ).to_json_file(custom_config)
    captured: dict[str, Any] = {}
    sentinel = object()

    class _TextEncoder:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            captured["path"] = path
            captured["kwargs"] = kwargs
            return sentinel

    monkeypatch.setattr(
        "modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact.get_component_class",
        lambda pipeline_cls, slot: _TextEncoder,
    )
    override = TextEncoderComponentArtifact(
        load_id="text-encoder-gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(custom_config),
    )

    loaded = override._materialize_single_file(pipeline_cls=object, effective_slot="text_encoder")

    assert loaded is sentinel
    assert captured["path"] == str(src.parent)
    assert captured["kwargs"]["gguf_file"] == src.name
    assert isinstance(captured["kwargs"]["config"], CLIPTextConfig)
    assert captured["kwargs"]["config"].hidden_size == 32


def test_text_encoder_fallback_loads_explicit_config_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _write_dummy_gguf(tmp_path / "src", "text-encoder.gguf")
    custom_config = src.parent / "text_encoder.json"
    CLIPTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_hidden_layers=1,
    ).to_json_file(custom_config)
    captured: dict[str, Any] = {}
    checkpoint_weight = object()
    aligned_weight = object()
    model = MagicMock()
    model.state_dict.return_value = {"weight": object()}
    model.to.return_value = model

    class _TextEncoder:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> object:
            raise ValueError("GGUF architecture is not supported yet")

        @classmethod
        def from_config(cls, loaded_config: object) -> object:
            captured["builder_config"] = loaded_config
            return model

    real_load_config = AutoConfig.from_pretrained

    def _load_config(config_source: str, **kwargs: Any) -> object:
        captured["config_source"] = config_source
        captured["config_kwargs"] = kwargs
        return real_load_config(config_source, **kwargs)

    def _load_checkpoint(path: str) -> dict[str, object]:
        captured["checkpoint_path"] = path
        return {"checkpoint_weight": checkpoint_weight}

    def _align_state(
        state: dict[str, object],
        target: set[str],
    ) -> dict[str, object]:
        captured["alignment_state"] = state
        captured["alignment_target"] = target
        return {"weight": aligned_weight}

    monkeypatch.setattr(
        "modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact.get_component_class",
        lambda pipeline_cls, slot: _TextEncoder,
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", _load_config)
    monkeypatch.setattr(
        "diffusers.models.model_loading_utils.load_gguf_checkpoint",
        _load_checkpoint,
    )
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.artifact_utils.packed_quant_io.align_state_dict_to_module_keys",
        _align_state,
    )
    override = TextEncoderComponentArtifact(
        load_id="text-encoder-gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(custom_config),
    )

    loaded = override._materialize_single_file(pipeline_cls=object, effective_slot="text_encoder")

    assert loaded is model
    assert captured["config_source"] == str(custom_config)
    assert captured["config_kwargs"] == {"local_files_only": True}
    assert isinstance(captured["builder_config"], CLIPTextConfig)
    assert captured["builder_config"].hidden_size == 32
    assert captured["checkpoint_path"] == str(src)
    assert captured["alignment_state"] == {"checkpoint_weight": checkpoint_weight}
    assert captured["alignment_target"] == {"weight"}
    model.load_state_dict.assert_called_once_with({"weight": aligned_weight}, strict=True)
    model.to.assert_called_once_with(torch.bfloat16)


def test_bake_rejects_missing_explicit_config_instead_of_using_sibling(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    override = ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(src.parent / "missing.json"),
    )
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": override}},
    )
    dest = tmp_path / "baked"

    with pytest.raises(FileNotFoundError, match="explicit config source.*missing\\.json.*does not exist"):
        save_baked_pipeline(pipe, artifact, dest)

    assert not dest.exists()


def test_bake_rejects_explicit_config_directory_without_component_config(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    override = ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(config_dir),
    )
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": override}},
    )
    dest = tmp_path / "baked"

    with pytest.raises(FileNotFoundError, match="explicit config directory.*does not contain 'config\\.json'"):
        save_baked_pipeline(pipe, artifact, dest)

    assert not dest.exists()


def test_bake_preserves_hf_repo_config_source_without_copying_sibling(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    override = ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source="org/model",
    )
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": override}},
    )
    dest = tmp_path / "baked"

    save_baked_pipeline(pipe, artifact, dest)

    manifest = json.loads((dest / "baked-config.json").read_text(encoding="utf-8"))
    recorded = manifest["file_overrides"]["transformer"]
    assert recorded["config_source"] == "org/model"
    assert not (dest / "transformer" / "config.json").exists()


def _bake_with_explicit_config_file(tmp_path: Path) -> Path:
    """Bake a GGUF transformer whose config_source is a file not named config.json."""
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()
    src = weights_dir / "weights.gguf"
    src.write_bytes(b"GGUF-dummy\n")
    (weights_dir / "config.json").write_text('{"marker": "sibling"}\n', encoding="utf-8")
    custom_config = weights_dir / "transformer.json"
    custom_config.write_text('{"marker": 1}\n', encoding="utf-8")
    override = ModelComponentArtifact(
        load_id="gguf",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="transformer",
        torch_dtype="bfloat16",
        file_path=str(src),
        config_source=str(custom_config),
    )
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": override}},
    )
    dest = tmp_path / "baked"
    save_baked_pipeline(pipe, artifact, dest)
    return dest


def test_bake_preserves_explicit_config_filename(tmp_path: Path) -> None:
    dest = _bake_with_explicit_config_file(tmp_path)

    baked_config = dest / "transformer" / "transformer.json"
    assert json.loads(baked_config.read_text(encoding="utf-8")) == {"marker": 1}
    assert not (dest / "transformer" / "config.json").exists()
    recorded = json.loads((dest / "baked-config.json").read_text(encoding="utf-8"))["file_overrides"]["transformer"]
    assert recorded["config_source"] == "transformer/transformer.json"


def test_baked_explicit_config_file_resolves_on_load(tmp_path: Path) -> None:
    dest = _bake_with_explicit_config_file(tmp_path)

    loaded = make_baked_artifact(dest)
    override = loaded.build_data["_component_overrides"]["transformer"]
    config_path = resolve_config_path(
        "",
        Flux2Transformer2DModel,
        override.config_source,
        pipeline_slot="transformer",
        artifact_component="transformer",
    )
    assert config_path.resolve() == (dest / "transformer" / "transformer.json").resolve()
    assert json.loads(config_path.read_text(encoding="utf-8")) == {"marker": 1}


def test_make_baked_artifact_resolves_gguf_override_and_status(tmp_path: Path) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": _gguf_override(src)}},
    )
    dest = tmp_path / "baked"
    save_baked_pipeline(pipe, artifact, dest)

    loaded = make_baked_artifact(dest)
    override = loaded.build_data["_component_overrides"]["transformer"]
    assert Path(override.file_path).resolve() == (dest / "transformer" / src.name).resolve()
    assert override.is_quantized
    assert collect_quantization_status(loaded) == QuantizationStatus(slots={"transformer": "GGUF"})


def test_build_pipeline_from_baked_passes_gguf_module_to_from_pretrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = _write_dummy_gguf(tmp_path / "src")
    pipe = _IndexLastPipe({"transformer": _RefusingSlot(), "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"transformer": _gguf_override(src)}},
    )
    dest = tmp_path / "baked"
    save_baked_pipeline(pipe, artifact, dest)
    loaded = make_baked_artifact(dest)
    sentinel = object()
    captured: dict[str, Any] = {}

    class DummyPipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> MagicMock:
            captured["path"] = path
            captured.update(kwargs)
            mock = MagicMock()
            mock.components = {}
            return mock

    monkeypatch.setattr(ModelComponentArtifact, "materialize", lambda self, **kwargs: sentinel)
    ModularDiffusionPipelineTypePipelineParameters._build_pipeline_from_baked(
        {**loaded.build_data, "_pipeline_cls": DummyPipeline},
        pipeline_cls=DummyPipeline,
    )
    assert captured["transformer"] is sentinel
    assert Path(captured["path"]) == dest


def test_bake_raises_on_refuse_save_that_is_not_gguf(tmp_path: Path) -> None:
    refusing = _RefusingSlot()
    pipe = _IndexLastPipe({"text_encoder": refusing, "unet": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionPipeline",
        builder_module="mod",
        builder_class_name="Cls",
    )
    dest = tmp_path / "baked"
    with pytest.raises(ValueError, match="not serializable and is not GGUF"):
        save_baked_pipeline(pipe, artifact, dest)
    assert refusing.save_calls == 0
    assert not dest.exists()


def test_component_refuses_official_save_reads_property() -> None:
    assert component_refuses_official_save(_PropertyRefusingSlot())
    assert component_refuses_official_save(_RefusingSlot())
    assert not component_refuses_official_save(_MarkerSlot())


def test_bake_raises_on_property_refuse_save_that_is_not_gguf(tmp_path: Path) -> None:
    refusing = _PropertyRefusingSlot()
    pipe = _IndexLastPipe({"transformer": refusing, "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
    )
    dest = tmp_path / "baked"
    with pytest.raises(ValueError, match="not serializable and is not GGUF"):
        save_baked_pipeline(pipe, artifact, dest)
    assert refusing.save_calls == 0
    assert not dest.exists()


def test_skip_official_save_slots_is_gguf_only() -> None:
    pipe = _IndexLastPipe(
        {
            "transformer": _GgufQuantizerSlot(),
            "text_encoder": _RefusingSlot(),
            "vae": _MarkerSlot(),
        }
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
    )
    assert skip_official_save_slots(pipe, artifact) == {"transformer"}
    assert is_gguf_quantizer_component(pipe.components["transformer"])
    assert not is_gguf_quantizer_component(pipe.components["text_encoder"])

    src = Path("/tmp/transformer.gguf")
    artifact_override = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        build_data={"_component_overrides": {"unet": _gguf_override(src, slot="unet")}},
    )
    assert skip_official_save_slots(_IndexLastPipe({"unet": _MarkerSlot()}), artifact_override) == {"unet"}


def test_bake_raises_on_torchao_before_official_save(tmp_path: Path) -> None:
    torchao_slot = _TorchAoQuantizerSlot()
    marker = _MarkerSlot()
    pipe = _IndexLastPipe({"text_encoder": torchao_slot, "vae": marker})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionPipeline",
        builder_module="mod",
        builder_class_name="Cls",
    )
    dest = tmp_path / "baked"
    assert is_torchao_quantizer_component(torchao_slot)
    assert not component_refuses_official_save(torchao_slot)
    assert skip_official_save_slots(pipe, artifact) == set()
    with pytest.raises(ValueError, match="torchao"):
        save_baked_pipeline(pipe, artifact, dest)
    assert torchao_slot.save_calls == 0
    assert not dest.exists()


def test_bake_raises_on_gguf_quantizer_without_file(tmp_path: Path) -> None:
    gguf_slot = _GgufQuantizerSlot()
    pipe = _IndexLastPipe({"transformer": gguf_slot, "vae": _MarkerSlot()})
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module="mod",
        builder_class_name="Cls",
    )
    dest = tmp_path / "baked"
    with pytest.raises(ValueError, match="GGUF and no file"):
        save_baked_pipeline(pipe, artifact, dest)
    assert gguf_slot.save_calls == 0
    assert not dest.exists()
