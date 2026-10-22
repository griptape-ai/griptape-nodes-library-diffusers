"""Typed payload for media-gen conditioning.

Written by the conditioning node, consumed by drivers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.retained_mode.events.event_converter import register_polymorphic_dataclass

from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

#: Marks a dict as an encoded payload, so a consumer can tell one that crossed a process boundary from
#: one built here. Present in the wire form only.
_PAYLOAD_TAG = "__media_gen_conditioning__"

#: The artifact class each `kind` names. `kind` is the union's discriminator, so rebuilding never has to
#: guess between two artifact dicts.
_ARTIFACT_CLASSES: dict[str, type[ImageUrlArtifact] | type[VideoUrlArtifact]] = {
    "image": ImageUrlArtifact,
    "video": VideoUrlArtifact,
}


@dataclass(frozen=True)
class ConditioningInputValue:
    """Resolved value of one `ConditioningInput`; `kind` discriminates image vs video."""

    artifact: ImageUrlArtifact | VideoUrlArtifact
    frame_index: int | str  # int, or a `FramePosition` value (str)
    strength: float
    kind: Literal["image", "video"]

    def _cattrs_unstructure(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact.to_dict(),
            "frame_index": self.frame_index,
            "strength": self.strength,
            "kind": self.kind,
        }

    @classmethod
    def _cattrs_structure(cls, data: dict[str, Any], _type: Any = None) -> ConditioningInputValue:
        kind = data["kind"]
        artifact_cls = _ARTIFACT_CLASSES.get(kind)
        if artifact_cls is None:
            msg = (
                f"Attempted to rebuild a conditioning entry. "
                f"Failed with kind='{kind}' because only {sorted(_ARTIFACT_CLASSES)} name an artifact class."
            )
            raise ValueError(msg)
        return cls(
            artifact=artifact_cls.from_dict(data["artifact"]),
            frame_index=data["frame_index"],
            strength=data["strength"],
            kind=kind,
        )


@dataclass(frozen=True)
class MediaGenConditioningPayload:
    """Typed conditioning payload; `entries` is always a tuple (video mode = 1 entry)."""

    mode: ConditioningMode
    entries: tuple[ConditioningInputValue, ...]

    @classmethod
    def empty(cls, mode: ConditioningMode) -> MediaGenConditioningPayload:
        return cls(mode=mode, entries=())

    # --- Crossing a process boundary ------------------------------------------------------------
    #
    # Every field is a URL artifact, an enum or a scalar, so the payload travels as data rather than
    # being held in the process that built it. Two things stop it travelling on the default path: the
    # engine encodes event payloads with `json.dumps(default=str)`, which replaces a type it cannot
    # unstructure with its `str()`, and cattrs cannot tell `ImageUrlArtifact` from `VideoUrlArtifact` in
    # the `artifact` union. `_cattrs_unstructure` and `_cattrs_structure` are the engine's seam for a
    # class that knows its own wire form.

    def _cattrs_unstructure(self) -> dict[str, Any]:
        return {
            _PAYLOAD_TAG: True,
            "mode": self.mode.value,
            "entries": [entry._cattrs_unstructure() for entry in self.entries],  # noqa: SLF001 - same module
        }

    @classmethod
    def _cattrs_structure(cls, data: dict[str, Any], _type: Any = None) -> MediaGenConditioningPayload:
        return cls(
            mode=ConditioningMode(data["mode"]),
            entries=tuple(
                ConditioningInputValue._cattrs_structure(entry)  # noqa: SLF001 - same module
                for entry in data["entries"]
            ),
        )


def is_encoded_conditioning_payload(value: Any) -> bool:
    """Whether `value` is a payload that crossed a process boundary and needs rebuilding."""
    return isinstance(value, dict) and _PAYLOAD_TAG in value


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
        # A payload that crossed a process boundary arrives as its wire form. Rebuilding it here means
        # every consumer gets a real payload without knowing a boundary was involved.
        payload = (
            MediaGenConditioningPayload._cattrs_structure(item)  # noqa: SLF001 - same module
            if is_encoded_conditioning_payload(item)
            else item
        )
        if not isinstance(payload, MediaGenConditioningPayload):
            msg = (
                f"Attempted to normalize media_gen_conditioning value. "
                f"Failed with item type '{type(payload).__name__}' because a "
                f"MediaGenConditioningPayload is required."
            )
            raise ValueError(msg)
        payloads.append(payload)
    return payloads


# `include_subclasses` reads `__subclasses__` when it is called, so registration belongs at the bottom of
# the module, where every class it has to cover already exists.
register_polymorphic_dataclass(MediaGenConditioningPayload)
