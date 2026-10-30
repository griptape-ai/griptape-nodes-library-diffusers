"""An inpaint mask artifact without latents must survive the trip between the orchestrator and a worker.

The encode node seeds its `latents` output with a placeholder artifact so downstream nodes can react to
its type, and the engine encodes every parameter value it sends to a worker, that placeholder included.
Without a codec the engine falls back to griptape's `to_dict`, which raises on the class's
`TYPE_CHECKING`-only annotations, and the node cannot run in a worker at all. These tests drive the
engine's own encoding instead of trusting it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.retained_mode.managers.node_manager import NodeManager
from griptape_nodes.serialization.values import ValueEncodeError, decode_value, dump_json, encode_value
from PIL import Image as PILImage

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.nodes.vae_mask_encoder import VaeMaskEncodeNode


def _over_the_wire(value: Any) -> Any:
    """`value` as a consumer in another process receives it, encoded exactly as the engine does."""
    return decode_value(json.loads(dump_json(encode_value(value))))


def _mask() -> PILImage.Image:
    mask = PILImage.new("L", (4, 2), 0)
    mask.putpixel((1, 0), 255)
    mask.putpixel((3, 1), 255)
    return mask


def test_the_encode_node_can_be_sent_to_a_worker() -> None:
    node = VaeMaskEncodeNode(name="encode")

    assert NodeManager._unencodable_value_names(node._list_rows.values_for_dispatch()) == []


def test_the_mask_survives() -> None:
    received = _over_the_wire(InpaintMaskArtifact(mask_image=_mask()))

    assert isinstance(received, InpaintMaskArtifact)
    assert received.mask_image.mode == "L"
    assert received.mask_image.size == (4, 2)
    assert received.mask_image.tobytes() == _mask().tobytes()


def test_every_other_field_survives() -> None:
    original = InpaintMaskArtifact(
        mask_image=_mask(),
        source_image=ImageUrlArtifact(value="http://localhost:8124/static/source.png"),
        source_shape=(1, 3, 64, 64),
        strength=0.6,
        meta={"producer": "encode"},
    )

    received = _over_the_wire(original)

    assert isinstance(received.source_image_artifact, ImageUrlArtifact)
    assert received.source_image_artifact.value == "http://localhost:8124/static/source.png"
    assert received.source_shape == (1, 3, 64, 64)
    assert received.strength == 0.6
    assert received.metadata == {"producer": "encode"}


@pytest.mark.parametrize("latent_field", ["source_latent", "masked_latent"])
def test_latents_are_refused_rather_than_dropped(latent_field: str) -> None:
    """Sending an artifact with its latents stripped would fail later, far from the cause."""
    latent: Any = SimpleNamespace(shape=(1, 4, 8, 8))
    artifact = InpaintMaskArtifact(mask_image=_mask(), **{latent_field: latent})

    with pytest.raises(ValueEncodeError, match="latents are held in the process that encoded them"):
        encode_value(artifact)
