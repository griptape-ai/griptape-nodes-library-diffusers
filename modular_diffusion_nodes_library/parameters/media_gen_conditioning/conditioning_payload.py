"""Typed payload for media-gen conditioning.

Written by the conditioning node, consumed by drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


@dataclass(frozen=True)
class ConditioningInputValue:
    """Resolved value of one `ConditioningInput`; `kind` discriminates image vs video."""

    artifact: ImageUrlArtifact | VideoUrlArtifact
    frame_index: int | str  # int, or a `FramePosition` value (str)
    strength: float
    kind: Literal["image", "video"]


@dataclass(frozen=True)
class MediaGenConditioningPayload:
    """Typed conditioning payload; `entries` is always a tuple (video mode = 1 entry)."""

    mode: ConditioningMode
    entries: tuple[ConditioningInputValue, ...]

    @classmethod
    def empty(cls, mode: ConditioningMode) -> MediaGenConditioningPayload:
        return cls(mode=mode, entries=())


def normalize_to_payloads(value: Any) -> list[MediaGenConditioningPayload] | None:
    """Normalize the `media_gen_conditioning` kwarg into a list of typed payloads.

    Accepts `None`, a single payload, or a list (when upstream nodes merge).
    Returns `None` only when `value` is `None`.
    """
    if value is None:
        return None
    items = value if isinstance(value, list) else [value]
    payloads: list[MediaGenConditioningPayload] = []
    for item in items:
        if not isinstance(item, MediaGenConditioningPayload):
            msg = (
                f"Attempted to normalize media_gen_conditioning value. "
                f"Failed with item type '{type(item).__name__}' because a "
                f"MediaGenConditioningPayload is required."
            )
            raise ValueError(msg)
        payloads.append(item)
    return payloads
