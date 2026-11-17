"""Focused tests for Qwen Image 2.1 driver contracts."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402
from diffusers.modular_pipelines.qwenimage21.decoders import unpack_latents  # noqa: E402
from diffusers.modular_pipelines.qwenimage21.inputs import pack_latents  # noqa: E402
from griptape.artifacts import ImageArtifact  # noqa: E402
from PIL import Image  # noqa: E402

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (  # noqa: E402
    GeneratorState,
    ImageMedia,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.qwen_image21 import (  # noqa: E402
    QwenImage21LatentPipelineDriver,
)
from modular_diffusion_nodes_library.nodes.vae_encoder import VaeEncodeNode  # noqa: E402
from modular_diffusion_nodes_library.runtime_parameters.qwen_image21_runtime_parameters import (  # noqa: E402
    QwenImage21PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import resolve_conditioning_image  # noqa: E402


def test_qwen_image21_packing_round_trips_public_image_latents() -> None:
    public_latents = torch.randn(2, 64, 4, 6)

    packed = pack_latents(public_latents.unsqueeze(2))
    unpacked = unpack_latents(packed, height=64, width=96, vae_scale_factor=16)

    assert packed.shape == (2, 24, 64)
    assert torch.equal(unpacked.squeeze(2), public_latents)


def test_turbo_modular_pipeline_receives_checkpoint_sigma_schedule() -> None:
    sigmas = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.0]
    driver = QwenImage21LatentPipelineDriver(cast(Any, SimpleNamespace(config=SimpleNamespace(sample_sigmas=sigmas))))

    modular_pipe = driver._create_modular_pipe()

    assert type(modular_pipe).__name__ == "QwenImage21TurboModularPipeline"
    assert modular_pipe.config["sample_sigmas"] == sigmas


def test_conditioning_images_can_preserve_alpha_without_changing_default_rgb_behavior() -> None:
    rgba_image = Image.new("RGBA", (8, 8), (12, 34, 56, 78))

    assert resolve_conditioning_image(rgba_image).mode == "RGB"
    preserved = resolve_conditioning_image(rgba_image, preserve_alpha=True)
    assert preserved.mode == "RGBA"
    assert preserved.getpixel((0, 0)) == (12, 34, 56, 78)


def test_vae_encode_path_preserves_qwen_rgba_image_and_source_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    image = Image.new("RGBA", (32, 32), (12, 34, 56, 78))
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_artifact = ImageArtifact(value=image_bytes.getvalue(), format="png", width=32, height=32)

    node = cast(Any, VaeEncodeNode.__new__(VaeEncodeNode))
    node.get_parameter_value = lambda _name: image_artifact
    node.pipe_params = SimpleNamespace(get_pipeline_class=lambda: "QwenImage21Pipeline")
    input_image = node.get_input_image()
    assert input_image.mode == "RGBA"

    image_tensor = VaeImageProcessor(
        vae_scale_factor=16,
        vae_latent_channels=64,
        do_convert_rgb=False,
    ).preprocess(input_image)
    source_shape = tuple(image_tensor.shape)
    assert source_shape == (1, 4, 32, 32)

    driver = QwenImage21LatentPipelineDriver(cast(Any, SimpleNamespace(config=SimpleNamespace(sample_sigmas=None))))
    processed_input_images: list[Image.Image] = []

    def call_block(_block: Any, **kwargs: Any) -> dict[str, Any]:
        processed_input_images.extend(kwargs["resized_image"])
        return {"image_latents": [torch.zeros((1, 64, 1, 2, 2))]}

    monkeypatch.setattr(driver, "_call_block", call_block)
    latent = driver.encode_media(ImageMedia(image=input_image, source_shape=source_shape), GeneratorState.from_seed(7))

    assert processed_input_images[0].mode == "RGBA"
    assert latent.source_shape == source_shape


@pytest.mark.parametrize("output_type", ["pil", "np"])
def test_qwen_decode_path_preserves_rgba_output(output_type: str, monkeypatch: pytest.MonkeyPatch) -> None:
    driver = QwenImage21LatentPipelineDriver(cast(Any, SimpleNamespace(config=SimpleNamespace(sample_sigmas=None))))
    driver_for_test = cast(Any, driver)
    driver_for_test._modular_pipe = SimpleNamespace(blocks=SimpleNamespace(sub_blocks={"decode": object()}))
    driver_for_test._get_device_and_type = lambda: (torch.device("cpu"), torch.float32)

    def call_block(_block: Any, **kwargs: Any) -> dict[str, Any]:
        if kwargs["output_type"] == "np":
            return {"images": np.zeros((1, 32, 32, 4), dtype=np.float32)}
        return {"images": [Image.new("RGBA", (32, 32))]}

    monkeypatch.setattr(driver, "_call_block", call_block)
    latent = cast(Any, SimpleNamespace(to_torch=lambda **_kwargs: torch.zeros((1, 64, 2, 2))))

    decoded = driver.decode_latent(latent, output_type=cast(Any, output_type))

    if output_type == "np":
        assert isinstance(decoded.media, np.ndarray)
        assert decoded.media.shape == (32, 32, 4)
    else:
        assert isinstance(decoded.media, Image.Image)
        assert decoded.media.mode == "RGBA"


class _RuntimeParameterNode:
    def __init__(self) -> None:
        self.parameters: dict[str, Any] = {}
        self.parameter_values: dict[str, Any] = {}

    def add_parameter(self, parameter: Any) -> None:
        self.parameters[parameter.name] = parameter
        self.parameter_values.setdefault(parameter.name, parameter.default_value)

    def get_parameter_by_name(self, name: str) -> Any:
        return self.parameters.get(name)

    def get_parameter_value(self, name: str) -> Any:
        return self.parameter_values.get(name)

    def set_parameter_value(self, name: str, value: Any, *, emit_change: bool = True) -> None:  # noqa: ARG002
        self.parameter_values[name] = value

    def remove_parameter_element_by_name(self, name: str) -> None:
        self.parameters.pop(name, None)


def _runtime_parameters_for(repo_id: str, node: _RuntimeParameterNode) -> Any:
    runtime_parameters = cast(Any, QwenImage21PipelineRuntimeParameters.__new__(QwenImage21PipelineRuntimeParameters))
    runtime_parameters._pipeline_metadata = {"build_data": {"base_repo_id": repo_id}}
    runtime_parameters._node = node
    runtime_parameters._seed_parameter = SimpleNamespace(
        add_input_parameters=lambda: None,
        remove_input_parameters=lambda: None,
        preprocess=lambda: None,
    )
    runtime_parameters._reference_images = SimpleNamespace(
        add_input_parameters=lambda: None,
        remove_input_parameters=lambda: None,
        get_pipe_kwargs=lambda: {},
    )
    return runtime_parameters


def test_base_turbo_switch_resets_stale_steps_and_restores_base_default() -> None:
    node = _RuntimeParameterNode()
    base = _runtime_parameters_for("Qwen/Qwen-Image-2.1", node)
    base.add_input_parameters()
    assert node.get_parameter_value("num_inference_steps") == 40

    # An untouched base default is cleared as the parameter is removed.
    base.remove_input_parameters()
    assert "num_inference_steps" not in node.parameter_values

    turbo = _runtime_parameters_for("Qwen/Qwen-Image-2.1-Turbo", node)
    turbo.add_input_parameters()
    assert node.get_parameter_value("num_inference_steps") == 8
    turbo.preprocess()
    turbo.remove_input_parameters()

    base_again = _runtime_parameters_for("Qwen/Qwen-Image-2.1", node)
    base_again.add_input_parameters()
    assert node.get_parameter_value("num_inference_steps") == 40

    # A user-edited base value can survive the switch, but Turbo must replace it with its fixed schedule length.
    node.set_parameter_value("num_inference_steps", 20)
    base_again.remove_input_parameters()
    turbo_again = _runtime_parameters_for("Qwen/Qwen-Image-2.1-Turbo", node)
    turbo_again.add_input_parameters()
    assert node.get_parameter_value("num_inference_steps") == 20
    turbo_again.preprocess()
    assert node.get_parameter_value("num_inference_steps") == 8
    turbo_again.remove_input_parameters()

    base_restored = _runtime_parameters_for("Qwen/Qwen-Image-2.1", node)
    base_restored.add_input_parameters()
    assert node.get_parameter_value("num_inference_steps") == 40
