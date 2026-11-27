"""`MaskedLatentArtifact` bundles a latent with a mask and has to behave on both sides of a process boundary.

The node seeds its `output_latent` with an empty placeholder that the engine encodes for a worker, while a
real artifact holds the latent in-process and must refuse to be sent. The mask tensor is what masked
denoising reads, so its shape and values are pinned here as well.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from griptape_nodes.serialization.values import ValueEncodeError, decode_value, dump_json, encode_value
from PIL import Image as PILImage

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.masked_latent_artifact import MaskedLatentArtifact


def _over_the_wire(value: Any) -> Any:
    """`value` as a consumer in another process receives it, encoded exactly as the engine does."""
    return decode_value(json.loads(dump_json(encode_value(value))))


def _mask() -> PILImage.Image:
    mask = PILImage.new("L", (8, 8), 0)
    for x in range(4, 8):
        for y in range(8):
            mask.putpixel((x, y), 255)
    return mask


def _latent(shape: tuple[int, ...]) -> LatentArtifact:
    """A latent that carries its shape in metadata only, so anything reading its tensor would fail."""
    return LatentArtifact(shape=shape, dtype="torch.float32", source_shape=shape, meta={"producer": "encode"})


def test_the_placeholder_survives_the_trip() -> None:
    received = _over_the_wire(MaskedLatentArtifact())

    assert isinstance(received, MaskedLatentArtifact)
    assert received.mask_image is None
    assert received.source_latent is None


def test_a_mask_without_a_latent_survives_the_trip() -> None:
    received = _over_the_wire(MaskedLatentArtifact(mask_image=_mask(), meta={"producer": "set-mask"}))

    assert received.mask_image.mode == "L"
    assert received.mask_image.size == (8, 8)
    assert received.mask_image.tobytes() == _mask().tobytes()
    assert received.metadata == {"producer": "set-mask"}


def test_the_latent_is_refused_rather_than_dropped() -> None:
    """Sending the artifact with its latent stripped would fail later, far from the cause."""
    artifact = MaskedLatentArtifact(mask_image=_mask(), source_latent=_latent((1, 4, 4, 4)))

    with pytest.raises(ValueEncodeError, match="source latent is held in the process that encoded it"):
        encode_value(artifact)


def test_metadata_is_copied_from_the_latent_not_shared() -> None:
    latent = _latent((1, 4, 4, 4))

    artifact = MaskedLatentArtifact(mask_image=_mask(), source_latent=latent)
    artifact.meta["added"] = True

    assert artifact.metadata["producer"] == "encode"
    assert "added" not in latent.meta


def test_it_is_truthy_only_with_a_latent() -> None:
    assert not MaskedLatentArtifact(mask_image=_mask())
    assert MaskedLatentArtifact(mask_image=_mask(), source_latent=_latent((1, 4, 4, 4)))


def test_the_image_mask_tensor_matches_the_latent_shape() -> None:
    torch = pytest.importorskip("torch", reason="Reads real torch tensors; run `make test/exec`.")
    artifact = MaskedLatentArtifact(mask_image=_mask(), source_latent=_latent((2, 4, 4, 4)))

    tensor = artifact.to_mask_tensor()

    assert tuple(tensor.shape) == (2, 4, 4, 4)
    assert tensor.dtype == torch.float32
    # Antialiased bilinear softens the mask's hard edge across the middle columns.
    assert torch.all(tensor[..., 0] == 0)
    assert torch.all(tensor[..., -1] == 1)
    assert torch.all(tensor[..., 1:] >= tensor[..., :-1])
    assert torch.all(tensor[..., 1] > 0)
    assert torch.all(tensor[..., 2] < 1)


def test_the_video_mask_tensor_repeats_over_frames() -> None:
    torch = pytest.importorskip("torch", reason="Reads real torch tensors; run `make test/exec`.")
    artifact = MaskedLatentArtifact(mask_image=_mask(), source_latent=_latent((1, 4, 3, 4, 4)))

    tensor = artifact.to_mask_tensor()

    assert tuple(tensor.shape) == (1, 4, 3, 4, 4)
    assert torch.equal(tensor[:, :, 0], tensor[:, :, 2])
    assert torch.all(tensor[..., -1] == 1)
    assert torch.all(tensor[..., 2] > 0.5)


def test_a_latent_that_is_neither_image_nor_video_is_refused() -> None:
    pytest.importorskip("torch", reason="Reads real torch tensors; run `make test/exec`.")
    artifact = MaskedLatentArtifact(mask_image=_mask(), source_latent=_latent((4, 4, 4)))

    with pytest.raises(ValueError, match="must be a 4-D image or 5-D video tensor"):
        artifact.to_mask_tensor()
