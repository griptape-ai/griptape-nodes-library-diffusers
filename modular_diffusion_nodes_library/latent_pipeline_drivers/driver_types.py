"""Shared types for the latent pipeline driver surface.

Leaf module — must not import anything from ``base_driver`` or any concrete
driver.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch  # type: ignore[reportMissingImports]
from PIL.Image import Image

TextEncodings = dict[str, Any]
DecodeResult = Image | list[Image] | np.ndarray

#: Fixed key under which drivers store their namespaced sub-bag on
#: ``LatentArtifact.meta``. The driver's class name is stored at this key, and
#: the namespaced sub-bag of driver-specific values lives at ``meta[<class name>]``.
META_DRIVER_KEY = "LatentPipelineDriver"

#: Key inside the driver-namespaced sub-bag under which the post-call
#: :class:`GeneratorState` is stamped, so downstream same-driver calls can
#: continue the same RNG stream instead of restarting from the UI seed.
GENERATOR_STATE_META_KEY = "generator_state"


def read_driver_meta(artifact: Any, key: str, required_driver_name: str, default: Any = None) -> Any:
    """Return ``artifact.meta[<driver_namespace>][key]`` or ``default``."""
    if artifact is None:
        return default
    meta = getattr(artifact, "meta", None) or {}
    driver_name = meta.get(META_DRIVER_KEY)
    if not driver_name or driver_name != required_driver_name:
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
class MaskMedia:
    """L-mode PIL mask paired with its ``source_shape``.
    White (255) marks pixels to inpaint, black (0) marks pixels to keep.
    """

    mask: Image
    source_shape: tuple[int, ...]


@dataclass(frozen=True)
class GeneratorState:
    """Frozen snapshot of a ``torch.Generator``"""

    state: torch.Tensor
    device: str

    @classmethod
    def from_seed(cls, seed: int, device: str = "cpu") -> "GeneratorState":
        """Defaults to cpu because it is more portable across devices."""
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
