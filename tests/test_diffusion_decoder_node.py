from pathlib import Path
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.nodes.diffusion_decoder_node import DiffusionDecoderNode
from modular_diffusion_nodes_library.parameters import diffusion_decoder_parameters as decoder_parameters_module
from modular_diffusion_nodes_library.parameters.diffusion_decoder_parameters import (
    DiffusionDecoderSpec,
    LTX25DiffusionDecoderParameters,
)


def test_ltx25_decoder_spec_declares_video_and_fps() -> None:
    assert LTX25DiffusionDecoderParameters.spec == DiffusionDecoderSpec(media_type="video", fps=24)


def test_decoder_dynamic_parameters_are_declared_by_parameter_class() -> None:
    assert LTX25DiffusionDecoderParameters._DYNAMIC_PARAMETERS == {
        "diffusion_decoder_model",
        "diffusion_decoder_model_download",
        "decoder_num_inference_steps",
        "decoder_use_tiling",
        "output_image",
        "output_video",
        "raw_media",
        "fps",
        "raw_output",
    }


def test_ltx25_decode_loads_selected_repo_with_download_support(monkeypatch: pytest.MonkeyPatch) -> None:
    params = LTX25DiffusionDecoderParameters.__new__(LTX25DiffusionDecoderParameters)
    params._node = MagicMock()
    params._node.get_parameter_value.side_effect = lambda name: {
        "decoder_num_inference_steps": 2,
        "decoder_use_tiling": False,
    }[name]
    params._model_repo_parameter = MagicMock()
    params._model_repo_parameter.validate_before_node_run.return_value = None
    params._model_repo_parameter.get_repo_revision.return_value = ("Lightricks/LTX-2.5-Diffusers", "revision")

    decoder = MagicMock()
    decoder.latents_mean = torch.zeros(4)
    decoder.latents_std = torch.ones(4)
    decoder.config.scaling_factor = 1.0
    decoder.decode.return_value = (
        torch.stack(
            [
                torch.full((2, 4, 4), 0.5),
                torch.full((2, 4, 4), 1.5),
                torch.full((2, 4, 4), -0.5),
            ]
        ).unsqueeze(0),
    )
    monkeypatch.setattr(decoder_parameters_module, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(
        decoder_parameters_module.LTX2VideoDiffusionDecoderModel,
        "from_pretrained",
        MagicMock(return_value=decoder),
    )
    monkeypatch.setattr(decoder_parameters_module, "cleanup_memory_caches", lambda: None)

    decoded = params.decode(
        MagicMock(to_torch=MagicMock(return_value=torch.zeros((1, 4, 2, 4, 4)))), seed=3, output_type="np"
    )

    assert decoded.shape == (1, 2, 4, 4, 3)
    assert decoder.to.call_args_list == [((), {"device": "cpu"}), (("cpu",), {})]
    decoder.decode.assert_called_once_with(
        ANY,
        generator=ANY,
        num_inference_steps=2,
        return_dict=False,
    )
    assert decoder.decode.call_args.args[0].shape == (1, 4, 2, 4, 4)
    assert decoder.decode.call_args.kwargs["num_inference_steps"] == 2
    assert isinstance(decoder.decode.call_args.kwargs["generator"], torch.Generator)
    assert decoded.dtype == np.float32
    assert decoded[0, 0, 0, 0, 0] == pytest.approx(0.75)
    assert decoded[0, 0, 0, 0, 1] == pytest.approx(1.25)
    assert decoded[0, 0, 0, 0, 2] == pytest.approx(0.25)

    pil_output = params.decode(
        MagicMock(to_torch=MagicMock(return_value=torch.zeros((1, 4, 2, 4, 4)))), seed=3, output_type="pil"
    )
    assert isinstance(pil_output[0][0], Image.Image)
    np.testing.assert_array_equal(np.asarray(pil_output[0][0])[0, 0], [191, 255, 64])

    decoder_parameters_module.LTX2VideoDiffusionDecoderModel.from_pretrained.assert_called_with(
        "Lightricks/LTX-2.5-Diffusers",
        subfolder="diffusion_decoder",
        revision="revision",
        torch_dtype=decoder_parameters_module.torch.bfloat16,
    )
    params._model_repo_parameter.get_repo_revision.assert_called_with()


def test_node_publishes_video_from_decoded_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.name = "Diffusion Decoder"
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="video", fps=24)
    node.decoder_params.decode.return_value = [Image.new("RGB", (4, 4)) for _ in range(2)]
    node.decoder_params.validate_before_node_run.return_value = None
    latent = LatentArtifact.from_torch(torch.zeros((1, 4, 2, 4, 4)), source_shape=(9, 32, 32, 3))
    node.get_parameter_value = lambda name: {"latent_tensor": latent, "seed": 5, "fps": 30, "raw_output": False}[name]
    node.parameter_output_values = {}
    node.set_parameter_value = MagicMock()

    encoded = {}

    def encode(frames: np.ndarray, fps: int, path: str) -> None:
        encoded.update(frames=frames, fps=fps, path=path)
        Path(path).write_bytes(b"video")

    monkeypatch.setattr("modular_diffusion_nodes_library.nodes.diffusion_decoder_node.encode_video", encode)
    saved = MagicMock(location="workspace://video.mp4")
    destination = MagicMock()
    destination.write_bytes.side_effect = lambda data: saved
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.nodes.diffusion_decoder_node.ProjectFileDestination.from_situation",
        lambda **kwargs: destination,
    )

    node._decode()

    assert encoded["frames"].shape == (2, 4, 4, 3)
    assert encoded["frames"].dtype == np.uint8
    assert encoded["fps"] == 30
    node.decoder_params.decode.assert_called_once_with(latent, seed=5, output_type="pil")
    node.set_parameter_value.assert_called_once_with("output_video", node.parameter_output_values["output_video"])
    assert node.parameter_output_values["output_video"].value == "workspace://video.mp4"


def test_node_switches_to_image_output_without_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="image")
    node._current_media_type = "video"
    node._current_raw_output = False
    node.get_parameter_value = lambda name: False
    removed = []
    added = []
    monkeypatch.setattr(node, "remove_parameter_element_by_name", removed.append, raising=False)
    monkeypatch.setattr(node, "add_parameter", added.append, raising=False)

    node._update_output_parameters()

    assert removed == ["output_video", "fps"]
    assert [parameter.name for parameter in added] == ["output_image"]
    assert node._current_media_type == "image"


def test_node_switches_to_video_output_with_spec_fps(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="video", fps=24)
    node._current_media_type = "image"
    node._current_raw_output = False
    node.get_parameter_value = lambda name: False
    removed = []
    added = []
    monkeypatch.setattr(node, "remove_parameter_element_by_name", removed.append, raising=False)
    monkeypatch.setattr(node, "add_parameter", added.append, raising=False)

    node._update_output_parameters()

    assert removed == ["output_image"]
    assert [parameter.name for parameter in added] == ["fps", "output_video"]
    assert added[0].default_value == 24
    assert node._current_media_type == "video"


def test_node_publishes_image_from_decoded_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.name = "Diffusion Decoder"
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="image")
    node.decoder_params.decode.return_value = [Image.new("RGB", (4, 4))]
    latent = LatentArtifact.from_torch(torch.zeros((1, 4, 4, 4)), source_shape=(4, 4, 3))
    node.get_parameter_value = lambda name: {"latent_tensor": latent, "seed": 0, "raw_output": False}[name]
    node.parameter_output_values = {}
    node.set_parameter_value = MagicMock()
    artifact = object()
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.nodes.diffusion_decoder_node.pil_to_image_artifact",
        lambda image: artifact,
    )

    node._decode()

    node.set_parameter_value.assert_called_once_with("output_image", artifact)
    assert node.parameter_output_values["output_image"] is artifact
    node.decoder_params.decode.assert_called_once_with(latent, seed=0, output_type="pil")


def test_node_raw_mode_preserves_float_media_and_clips_display_output(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.name = "Diffusion Decoder"
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="image")
    decoded = np.array([[[[-0.5, 0.4, 1.5], [0.2, 0.5, 2.0]]]], dtype=np.float32)
    node.decoder_params.decode.return_value = decoded
    latent = LatentArtifact.from_torch(torch.zeros((1, 4, 4, 4)), source_shape=(1, 2, 3))
    node.get_parameter_value = lambda name: {"latent_tensor": latent, "seed": 4, "raw_output": True}[name]
    node.parameter_output_values = {}
    node.set_parameter_value = MagicMock()
    images = []
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.nodes.diffusion_decoder_node.pil_to_image_artifact",
        lambda image: images.append(image.copy()) or "image-artifact",
    )

    node._decode()

    raw = node.parameter_output_values["raw_media"]
    assert isinstance(raw, FloatMediaArtifact)
    np.testing.assert_array_equal(raw.to_numpy(), decoded[0])
    assert raw.dtype == np.float32
    assert raw.to_numpy().min() == -0.5
    assert raw.to_numpy().max() == 2.0
    np.testing.assert_array_equal(np.asarray(images[0])[0, 0], [0, 102, 255])
    node.decoder_params.decode.assert_called_once_with(latent, seed=4, output_type="np")


def test_node_raw_video_output_keeps_fps_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.name = "Diffusion Decoder"
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="video", fps=24)
    node.decoder_params.decode.return_value = np.full((1, 2, 4, 4, 3), 0.5, dtype=np.float32)
    latent = LatentArtifact.from_torch(torch.zeros((1, 4, 2, 4, 4)), source_shape=(2, 32, 32, 3))
    node.get_parameter_value = lambda name: {"latent_tensor": latent, "seed": 4, "fps": 30, "raw_output": True}[name]
    node.parameter_output_values = {}
    node.set_parameter_value = MagicMock()
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.nodes.diffusion_decoder_node.encode_video",
        lambda frames, fps, path: Path(path).write_bytes(b"video"),
    )
    saved = MagicMock(location="workspace://video.mp4")
    destination = MagicMock()
    destination.write_bytes.return_value = saved
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.nodes.diffusion_decoder_node.ProjectFileDestination.from_situation",
        lambda **kwargs: destination,
    )

    node._decode()

    raw = node.parameter_output_values["raw_media"]
    assert isinstance(raw, FloatMediaArtifact)
    assert raw.meta["fps"] == 30
    assert raw.shape == (1, 2, 4, 4, 3)


def test_node_rejects_non_video_latents() -> None:
    node = DiffusionDecoderNode.__new__(DiffusionDecoderNode)
    node.decoder_params = MagicMock()
    node.decoder_params.spec = DiffusionDecoderSpec(media_type="video", fps=24)
    node.get_parameter_value = lambda name: {
        "latent_tensor": LatentArtifact.from_torch(torch.zeros((1, 4, 8, 8)), source_shape=(64, 64, 3)),
        "fps": 24,
    }[name]

    errors = node.validate_before_node_run()

    assert errors is not None
    assert "5-D video latent" in str(errors[0])
