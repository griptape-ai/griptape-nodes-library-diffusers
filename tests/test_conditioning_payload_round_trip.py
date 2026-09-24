"""A media-gen conditioning payload must survive the trip between the orchestrator and a worker.

The payload is URL artifacts, an enum and scalars, so it travels as data rather than being held in the
process that built it. It was declared `serializable=False`, which handed every consumer -- including the
editor -- a reference instead, and the consuming node's validation raised rather than validating.

Two things break it on the default encoding path, both silently. The engine encodes event payloads with
`json.dumps(default=str)`, which replaces a type it cannot unstructure with that type's `str()`. And
cattrs cannot tell `ImageUrlArtifact` from `VideoUrlArtifact` in the `artifact` union, so a payload that
did encode came back with its artifact left as a bare dict. These tests pin the encoding instead of
trusting it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.retained_mode.events.event_converter import safe_unstructure

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


def _over_the_wire(value: Any) -> Any:
    """`value` as a consumer in another process receives it, encoded exactly as the engine does."""
    payload = {"parameter_output_values": {"conditioning": value}}
    encoded = json.dumps(safe_unstructure(payload), default=str)
    return json.loads(encoded)["parameter_output_values"]["conditioning"]


def _round_trip(value: Any) -> list[MediaGenConditioningPayload] | None:
    return normalize_to_payloads(_over_the_wire(value))


def _image_payload() -> MediaGenConditioningPayload:
    """Two entries, one symbolic frame index and one concrete, which is the flexible image config."""
    return MediaGenConditioningPayload(
        mode=ConditioningMode.IMAGE,
        entries=(
            ConditioningInputValue(
                artifact=ImageUrlArtifact(value="http://localhost:8124/static/first.png"),
                frame_index="first",
                strength=0.75,
                kind="image",
            ),
            ConditioningInputValue(
                artifact=ImageUrlArtifact(value="http://localhost:8124/static/last.png"),
                frame_index=23,
                strength=1.0,
                kind="image",
            ),
        ),
    )


def _video_payload() -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(
        mode=ConditioningMode.VIDEO,
        entries=(
            ConditioningInputValue(
                artifact=VideoUrlArtifact(value="http://localhost:8124/static/clip.mp4"),
                frame_index=0,
                strength=1.0,
                kind="video",
            ),
        ),
    )


def test_the_payload_does_not_arrive_as_a_string() -> None:
    """What `json.dumps(default=str)` does to a type it cannot unstructure, and the original bug."""
    assert not isinstance(_over_the_wire(_image_payload()), str)


def test_every_field_survives() -> None:
    original = _image_payload()

    assert _round_trip(original) == [original]


def test_the_artifact_arrives_as_its_own_class_rather_than_a_dict() -> None:
    """`kind` is what tells the two apart; the union alone leaves cattrs nothing to disambiguate on."""
    original = _video_payload()

    restored = _round_trip(original)

    assert restored is not None
    assert isinstance(restored[0].entries[0].artifact, VideoUrlArtifact)
    assert restored[0].entries[0].artifact.value == original.entries[0].artifact.value


def test_a_merged_list_of_payloads_survives() -> None:
    """Several conditioning nodes can feed one consumer, which receives them as a list."""
    originals = [_image_payload(), _video_payload()]

    assert _round_trip(originals) == originals


def test_an_unknown_kind_is_refused_rather_than_guessed() -> None:
    entry = {"kind": "audio", "artifact": {}, "frame_index": 0, "strength": 1.0}

    with pytest.raises(ValueError, match="name an artifact class"):
        ConditioningInputValue._cattrs_structure(entry)  # noqa: SLF001 - the inverse of the module's own encoding


def test_a_genuinely_wrong_type_is_still_rejected() -> None:
    """Accepting the wire form must not turn the normalizer into something that accepts anything."""
    with pytest.raises(ValueError, match="MediaGenConditioningPayload is required"):
        normalize_to_payloads(7)
