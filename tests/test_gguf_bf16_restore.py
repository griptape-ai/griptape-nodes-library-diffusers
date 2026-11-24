"""Exercise BF16 repair through real GGUF files and Diffusers' actual GGUF quantiser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("gguf")
pytest.importorskip("numpy")
pytest.importorskip("accelerate")
pytest.importorskip("diffusers")

import gguf  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from accelerate import init_empty_weights  # noqa: E402
from diffusers.models.model_loading_utils import load_gguf_checkpoint  # noqa: E402
from diffusers.models.modeling_utils import ModelMixin, get_parameter_dtype  # noqa: E402
from diffusers.quantizers.gguf.gguf_quantizer import GGUFQuantizer  # noqa: E402
from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor  # noqa: E402
from diffusers.quantizers.quantization_config import GGUFQuantizationConfig  # noqa: E402

from modular_diffusion_nodes_library.component_loading.gguf_bf16 import restore_bf16_gguf_tensors  # noqa: E402

NORM_WEIGHT = torch.arange(1, 9, dtype=torch.float32).bfloat16()
QUANTISED_BIAS = torch.tensor([0.5, -1.0, 2.0, 3.25]).bfloat16()
PROJECTION_WEIGHT = torch.linspace(-1, 1, 32).reshape(4, 8).bfloat16()
PROJECTION_BIAS = torch.tensor([0.25, 0.0, -0.75, 1.5]).bfloat16()


class TinyModel(ModelMixin):
    def __init__(self) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(8)
        self.quantised = torch.nn.Linear(64, 4)
        self.projection = torch.nn.Linear(8, 4)


@pytest.fixture(scope="module")
def gguf_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("gguf") / "tiny.gguf"
    writer = gguf.GGUFWriter(str(path), arch="test")
    for name, values in (
        ("norm.weight", NORM_WEIGHT),
        ("quantised.bias", QUANTISED_BIAS),
        ("projection.weight", PROJECTION_WEIGHT),
        ("projection.bias", PROJECTION_BIAS),
    ):
        writer.add_tensor(
            name,
            values.view(torch.int16).numpy().view(np.uint8),
            raw_dtype=gguf.GGMLQuantizationType.BF16,
        )
    weight = np.random.default_rng(0).standard_normal((4, 64)).astype(np.float32)
    writer.add_tensor(
        "quantised.weight",
        gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q4_0),
        raw_dtype=gguf.GGMLQuantizationType.Q4_0,
    )
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def _load(gguf_file: Path) -> Any:
    state = load_gguf_checkpoint(str(gguf_file))
    quantiser = GGUFQuantizer(GGUFQuantizationConfig(compute_dtype=torch.bfloat16))
    with init_empty_weights():
        model = TinyModel()
    quantiser._process_model_before_weight_loading(model, device_map=None, state_dict=state)
    for name, value in state.items():
        quantiser.create_quantized_param(model, value, name, torch.device("cpu"), state_dict=state)
    return model


def test_real_gguf_reproduces_h3_byte_input_failure(gguf_file: Path) -> None:
    model = _load(gguf_file)
    dtype = get_parameter_dtype(model.projection)
    assert dtype == torch.uint8
    inputs = torch.ones((2, 8), dtype=torch.bfloat16).to(dtype)

    with pytest.raises(RuntimeError, match="same dtype"):
        model.projection(inputs)


def test_restore_preserves_file_values_and_quantised_weights(gguf_file: Path) -> None:
    model = _load(gguf_file)
    quantised_weight = model.quantised.weight

    assert restore_bf16_gguf_tensors(model) == 4
    assert torch.equal(model.norm.weight, NORM_WEIGHT)
    assert torch.equal(model.quantised.bias, QUANTISED_BIAS)
    assert torch.equal(model.projection.weight, PROJECTION_WEIGHT)
    assert torch.equal(model.projection.bias, PROJECTION_BIAS)
    assert type(model.projection) is torch.nn.Linear
    assert get_parameter_dtype(model.projection) == torch.bfloat16
    assert isinstance(model.quantised, GGUFLinear)
    assert isinstance(model.quantised.weight, GGUFParameter)
    assert model.quantised.weight is quantised_weight
    assert restore_bf16_gguf_tensors(model) == 0


def test_restored_projection_norm_and_quantised_linear_run_on_cpu(gguf_file: Path) -> None:
    model = _load(gguf_file)
    restore_bf16_gguf_tensors(model)
    inputs = torch.randn((2, 8), dtype=torch.bfloat16)
    wide_inputs = torch.randn((2, 64), dtype=torch.bfloat16)
    weight = dequantize_gguf_tensor(model.quantised.weight).bfloat16()

    assert torch.equal(model.norm(inputs), torch.nn.functional.rms_norm(inputs, (8,), NORM_WEIGHT))
    assert torch.equal(
        model.projection(inputs.to(get_parameter_dtype(model.projection))),
        torch.nn.functional.linear(inputs, PROJECTION_WEIGHT, PROJECTION_BIAS),
    )
    assert torch.allclose(
        model.quantised.forward_native(wide_inputs),
        torch.nn.functional.linear(wide_inputs, weight, QUANTISED_BIAS),
        atol=1e-2,
    )


def test_plain_bf16_model_is_untouched() -> None:
    model = TinyModel()
    torch.nn.Module.to(model, dtype=torch.bfloat16)
    weight = model.projection.weight

    assert restore_bf16_gguf_tensors(model) == 0
    assert model.projection.weight is weight


@pytest.mark.parametrize("suffix", ["gguf", "safetensors"])
def test_single_file_materialisation_repairs_only_gguf(
    monkeypatch: pytest.MonkeyPatch, gguf_file: Path, suffix: str
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts

    model = TinyModel()
    if suffix == "gguf":
        model = _load(gguf_file)

    class Component:
        @classmethod
        def from_single_file(cls, **kwargs: Any) -> Any:
            return model

    restored: list[Any] = []

    def restore(component: Any) -> int:
        restored.append(component)
        return restore_bf16_gguf_tensors(component)

    monkeypatch.setattr(artifacts, "get_component_class", lambda *args: Component)
    monkeypatch.setattr(artifacts, "get_loadable_component_class", lambda *args: Component)
    monkeypatch.setattr(artifacts, "loadable_class_name", lambda component_cls: "Component")
    monkeypatch.setattr(artifacts, "resolve_config_path", lambda *args, **kwargs: Path("/virtual/config"))
    monkeypatch.setattr(artifacts, "restore_bf16_gguf_tensors", restore)
    monkeypatch.setattr("diffusers.loaders.single_file_utils.load_single_file_checkpoint", lambda path: {})
    monkeypatch.setattr("diffusers.loaders.single_file_utils.infer_diffusers_model_type", lambda weights: "minimax-h3")
    artifact = artifacts.ModelComponentArtifact(
        load_id=f"test-{suffix}",
        source_type=artifacts.ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path=f"/virtual/model.{suffix}",
    )

    assert artifact.materialize(pipeline_cls=Component, slot="transformer") is model
    if suffix == "gguf":
        assert restored == [model]
        assert get_parameter_dtype(model.projection) == torch.bfloat16
    else:
        assert restored == []
