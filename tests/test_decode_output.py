import inspect

import numpy as np
import PIL.Image

from modular_diffusion_nodes_library.latent_pipeline_drivers._base_driver_forwardable_signature import (
    FORWARDABLE_METHOD_POSITIONAL,
    validate_forwardable_signature,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import DecodeResult


def test_driver_decode_signatures_follow_output_type_contract() -> None:
    classes = set(_DRIVER_REGISTRY.values())
    assert classes
    assert FORWARDABLE_METHOD_POSITIONAL["decode_latent"] == ("latent", "output_type")

    for driver_class in classes:
        method = driver_class.decode_latent
        validate_forwardable_signature(driver_class.__name__, "decode_latent", method)
        parameters = list(inspect.signature(method).parameters.values())[1:]
        assert [parameter.name for parameter in parameters] == ["latent", "output_type"]
        assert parameters[1].default == "pil"


def test_image_decode_output_selects_numpy_batch_item() -> None:
    batch = np.zeros((1, 12, 16, 3), dtype=np.float32)

    image = LatentPipelineDriver._get_decoded_media({"images": batch}, "images", "np", is_video=False)

    assert isinstance(image, DecodeResult)
    assert isinstance(image.media, np.ndarray)
    assert image.media.shape == (12, 16, 3)


def test_video_decode_output_selects_numpy_batch_item() -> None:
    batch = np.zeros((1, 5, 12, 16, 3), dtype=np.float32)

    frames = LatentPipelineDriver._get_decoded_media({"videos": batch}, "videos", "np", is_video=True)

    assert isinstance(frames, DecodeResult)
    assert isinstance(frames.media, np.ndarray)
    assert frames.media.shape == (5, 12, 16, 3)


def test_pil_decode_output_preserves_image_and_video_batch_items() -> None:
    image = PIL.Image.new("RGB", (16, 12))
    frames = [PIL.Image.new("RGB", (16, 12)) for _ in range(5)]

    decoded_image = LatentPipelineDriver._get_decoded_media({"images": [image]}, "images", "pil", is_video=False)
    decoded_video = LatentPipelineDriver._get_decoded_media({"videos": [frames]}, "videos", "pil", is_video=True)

    assert decoded_image.media is image
    assert decoded_video.media is frames
