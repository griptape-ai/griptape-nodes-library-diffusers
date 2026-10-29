from __future__ import annotations

import socket
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from diffusers import ConfigMixin, DiffusionPipeline, ModelMixin  # type: ignore[reportMissingImports]
from diffusers.configuration_utils import register_to_config  # type: ignore[reportMissingImports]
from optimum.quanto import freeze, qint8, quantize  # type: ignore[reportMissingImports]
from torch import nn

from modular_diffusion_nodes_library.artifact_utils.packed_quant_io import write_packed_quant_maps
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_build_steps import LoadPipelineStep
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters


class _TinyModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, width: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width, bias=False)


class _TinyPipeline(DiffusionPipeline):
    last_load_kwargs: dict[str, Any] | None = None

    def __init__(self, model: _TinyModel) -> None:
        super().__init__()
        self.register_modules(model=model)

    @classmethod
    def from_pretrained(cls, *args: Any, **kwargs: Any) -> _TinyPipeline:
        cls.last_load_kwargs = dict(kwargs)
        return super().from_pretrained(*args, **kwargs)


def _baked_build_data(path: Path, *, requires_device_map: bool) -> dict[str, Any]:
    return {
        "_baked_path": str(path),
        "_baked_dtype": "float32",
        "_baked_requires_device_map": requires_device_map,
    }


def _disable_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_connection(*args: Any, **kwargs: Any) -> None:
        pytest.fail("The local baked-pipeline test attempted a network connection.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(socket.socket, "connect", reject_connection)


def test_baked_artifact_injects_device_map_requirement_only_into_load_step(tmp_path: Path) -> None:
    artifact = DiffusionPipelineArtifact.from_baked_config_dict(
        {
            "pipeline_name": "WanPipeline",
            "builder_module": "test.module",
            "builder_class_name": "WanParameters",
            "requires_device_map": True,
        },
        baked_path=tmp_path,
        config_hash="baked-hash",
    )

    assert artifact.requires_device_map is True
    assert "_baked_requires_device_map" not in artifact.build_data
    load_step = artifact._load_and_fuse_steps()[0]
    assert isinstance(load_step, LoadPipelineStep)
    assert load_step._build_data["_baked_requires_device_map"] is True


def test_baked_artifact_requires_device_map_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Baked pipeline requires_device_map flag must be a boolean"):
        DiffusionPipelineArtifact.from_baked_config_dict(
            {
                "pipeline_name": "WanPipeline",
                "builder_module": "test.module",
                "builder_class_name": "WanParameters",
            },
            baked_path=tmp_path,
            config_hash="baked-hash",
        )


@pytest.mark.parametrize(
    ("requires_device_map", "expected_device_map"),
    [(False, None), (True, "balanced")],
)
def test_standard_baked_loader_applies_device_map_policy(
    tmp_path: Path,
    requires_device_map: bool,
    expected_device_map: str | None,
) -> None:
    captured: dict[str, Any] = {}
    pipe = SimpleNamespace(components={})

    class RecordingPipeline:
        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> Any:
            captured["path"] = path
            captured.update(kwargs)
            return pipe

    built = ModularDiffusionPipelineTypePipelineParameters._build_pipeline_from_baked(
        _baked_build_data(tmp_path, requires_device_map=requires_device_map),
        pipeline_cls=RecordingPipeline,
    )

    assert built is pipe
    assert captured["path"] == str(tmp_path)
    assert captured["local_files_only"] is True
    if expected_device_map is None:
        assert "device_map" not in captured
    else:
        assert captured["device_map"] == expected_device_map


def test_minimax_baked_loader_keeps_components_manager_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, Any] = {}
    pipe = object()

    def build_from_source(
        cls: type[MiniMaxH3PipelineParameters],
        source: str,
        *,
        revision: str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Any:
        captured.update(source=source, revision=revision, dtype=dtype)
        return pipe

    monkeypatch.setattr(MiniMaxH3PipelineParameters, "_build_pipeline_from_source", classmethod(build_from_source))

    built = MiniMaxH3PipelineParameters._build_pipeline_from_baked(
        _baked_build_data(tmp_path, requires_device_map=True),
        pipeline_cls=_TinyPipeline,
    )

    assert built is pipe
    assert captured == {
        "source": str(tmp_path),
        "revision": None,
        "dtype": torch.float32,
    }


def test_standard_baked_loader_loads_tiny_local_pipeline_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "tiny-pipeline"
    _TinyPipeline(_TinyModel()).save_pretrained(folder, safe_serialization=True)
    _TinyPipeline.last_load_kwargs = None
    _disable_network(monkeypatch)

    built = ModularDiffusionPipelineTypePipelineParameters._build_pipeline_from_baked(
        _baked_build_data(folder, requires_device_map=True),
        pipeline_cls=_TinyPipeline,
    )

    assert isinstance(built, _TinyPipeline)
    assert _TinyPipeline.last_load_kwargs is not None
    assert _TinyPipeline.last_load_kwargs["device_map"] == "balanced"
    assert _TinyPipeline.last_load_kwargs["local_files_only"] is True
    assert sum(path.stat().st_size for path in folder.rglob("*") if path.is_file()) < 4096


def test_standard_baked_loader_restores_tiny_packed_quanto_pipeline_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "tiny-packed-pipeline"
    model = _TinyModel()
    quantize(model, weights=qint8)
    freeze(model)
    source = _TinyPipeline(model)
    source.save_pretrained(folder, safe_serialization=True)
    assert write_packed_quant_maps(source, folder) == ["model"]
    _TinyPipeline.last_load_kwargs = None
    _disable_network(monkeypatch)

    built = ModularDiffusionPipelineTypePipelineParameters._build_pipeline_from_baked(
        _baked_build_data(folder, requires_device_map=True),
        pipeline_cls=_TinyPipeline,
    )

    assert isinstance(built, _TinyPipeline)
    assert _TinyPipeline.last_load_kwargs is not None
    assert _TinyPipeline.last_load_kwargs["device_map"] == "balanced"
    assert any(key.endswith("weight._data") for key in built.model.state_dict())
