from __future__ import annotations

import pickle
from typing import Any, NoReturn

import numpy as np
from griptape.artifacts.base_artifact import BaseArtifact


class FloatMediaArtifact(BaseArtifact):
    """In-process float RGB media decoded from a diffusion latent.

    Supported arrays are a single image ``(H, W, 3)``, a frame sequence
    ``(F, H, W, 3)``, or a single-batch video ``(1, F, H, W, 3)``. The final
    dimension is RGB; values retain the decoder's float output values.
    """

    def __init__(
        self,
        array: np.ndarray,
        *,
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError(f"Expected a NumPy array, got {type(array).__name__}.")
        if array.ndim not in (3, 4, 5) or array.shape[-1] != 3:
            raise ValueError(f"Expected RGB image/video with rank 3, 4, or 5, got shape {array.shape}.")
        if array.ndim == 5 and array.shape[0] != 1:
            raise ValueError(f"Expected a single video batch, got batch size {array.shape[0]}.")
        if array.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
            raise TypeError(f"Expected float16 or float32 media, got {array.dtype}.")

        owned_array = np.array(array, copy=True, order="C")
        owned_array.setflags(write=False)
        summary = {} if meta is None else meta.copy()
        summary.update(
            {
                "shape": list(owned_array.shape),
                "dtype": str(owned_array.dtype),
                "min": float(owned_array.min()),
                "max": float(owned_array.max()),
                "mean": float(owned_array.mean()),
            }
        )
        super().__init__(value=None, meta=summary, **kwargs)
        self._array = owned_array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    @property
    def dtype(self) -> np.dtype:
        return self._array.dtype

    @property
    def ndim(self) -> int:
        return self._array.ndim

    @property
    def num_frames(self) -> int:
        if self._array.ndim == 3:
            return 1
        if self._array.ndim == 5:
            return self._array.shape[1]
        return self._array.shape[0]

    def to_numpy(self) -> np.ndarray:
        """Return the immutable RGB media array without copying it."""
        return self._array

    def to_frames(self) -> np.ndarray:
        """Return frames in ``(F, H, W, 3)`` order without copying when possible."""
        if self._array.ndim == 3:
            frames = self._array[np.newaxis]
        elif self._array.ndim == 5:
            frames = self._array[0]
        else:
            frames = self._array
        return frames

    def to_text(self) -> str:
        return f"FloatMediaArtifact(shape={self.shape}, dtype={self.dtype})"

    def to_dict(self) -> dict[str, Any]:  # type: ignore[reportIncompatibleMethodOverride]
        return {
            "type": "FloatMediaArtifact",
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "num_frames": self.num_frames,
            "min": self.meta["min"],
            "max": self.meta["max"],
            "mean": self.meta["mean"],
        }

    def __reduce__(self) -> NoReturn:
        raise pickle.PicklingError("FloatMediaArtifact is an in-process-only type and should not be serialized.")

    def __bool__(self) -> bool:
        return True
