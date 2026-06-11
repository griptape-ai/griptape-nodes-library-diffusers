"""Shared types for the latent pipeline driver surface.

Leaf module — must not import anything from ``base_driver`` or any concrete
driver. ``base_driver`` re-exports these names so existing call sites that
import from there continue to work.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch  # type: ignore[reportMissingImports]
from PIL.Image import Image

TextEncodings = dict[str, Any]
DecodeResult = Image | list[Image] | np.ndarray

#: Well-known key under which drivers store their namespaced sub-bag on
#: ``LatentArtifact.meta``. The driver's class name is stored at this key, and
#: the namespaced sub-bag of driver-specific values lives at ``meta[<class name>]``.
META_DRIVER_KEY = "LatentPipelineDriver"

#: Key inside the driver-namespaced sub-bag under which the post-call
#: :class:`GeneratorState` is stamped, so downstream same-driver calls can
#: continue the same RNG stream instead of restarting from the UI seed.
GENERATOR_STATE_META_KEY = "generator_state"


def read_driver_meta(artifact: Any, key: str, default: Any = None) -> Any:
    """Return ``artifact.meta[<driver_namespace>][key]`` or ``default``."""
    if artifact is None:
        return default
    meta = getattr(artifact, "meta", None) or {}
    driver_name = meta.get(META_DRIVER_KEY)
    if not driver_name:
        return default
    sub = meta.get(driver_name)
    if not isinstance(sub, dict):
        return default
    return sub.get(key, default)


@dataclass(frozen=True)
class ImageMedia:
    image: Image | torch.Tensor
    source_shape: tuple[int, ...]


@dataclass(frozen=True)
class VideoMedia:
    frames: list[Image]
    source_shape: tuple[int, ...]


@dataclass(frozen=True)
class GeneratorState:
    """Frozen snapshot of a ``torch.Generator`` that round-trips through ``LatentArtifact.meta``.

    Construction is always via the factories so the public surface only ever sees
    ``GeneratorState`` instances (never raw seeds or generators).
    """

    state: torch.Tensor
    device: str

    @classmethod
    def from_seed(cls, seed: int, device: str = "cpu") -> "GeneratorState":
        gen = torch.Generator(device=device).manual_seed(int(seed))
        return cls(state=gen.get_state(), device=device)

    @classmethod
    def from_generator(cls, gen: torch.Generator) -> "GeneratorState":
        return cls(state=gen.get_state(), device=str(gen.device))

    @classmethod
    def from_artifact(cls, artifact: Any) -> "GeneratorState | None":
        """Return the GeneratorState stamped onto ``artifact.meta``, or None.

        Looks up the driver-namespaced sub-bag indicated by
        :data:`META_DRIVER_KEY` and returns the value at
        :data:`GENERATOR_STATE_META_KEY` if it is a :class:`GeneratorState`.
        Accepts ``None`` and any object exposing a ``meta`` dict.
        """
        if artifact is None:
            return None
        meta = getattr(artifact, "meta", None) or {}
        driver_name = meta.get(META_DRIVER_KEY)
        if not driver_name:
            return None
        sub = meta.get(driver_name)
        if not isinstance(sub, dict):
            return None
        state = sub.get(GENERATOR_STATE_META_KEY)
        if isinstance(state, cls):
            return state
        return None

    def to_generator(self) -> torch.Generator:
        gen = torch.Generator(device=self.device)
        gen.set_state(self.state)
        return gen

    def as_meta(self) -> dict[str, Any]:
        return {GENERATOR_STATE_META_KEY: self}
