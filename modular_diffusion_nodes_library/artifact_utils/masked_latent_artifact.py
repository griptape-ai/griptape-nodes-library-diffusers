from __future__ import annotations

import pickle
from typing import TYPE_CHECKING, Any

from griptape.artifacts.blob_artifact import BlobArtifact
from griptape_nodes.serialization.values import ValueEncodeError, register_value_codec

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from PIL.Image import Image


@register_value_codec
class MaskedLatentArtifact(BlobArtifact):
    """In-process bundle of an encoded source latent and its mask image.

    Carries a latent and mask between nodes within a single process. This type is
    deliberately unserializable — any Parameter holding a MaskedLatentArtifact
    must be marked serializable=False.
    """

    def __init__(
        self,
        mask_image: Image | None = None,
        source_latent: LatentArtifact | None = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if source_latent is not None:
            artifact_meta = dict(source_latent.meta)
        else:
            artifact_meta = {}
        if meta is not None:
            artifact_meta.update(meta)
        super().__init__(value=b"", meta=artifact_meta)
        self._mask_image = mask_image
        self._source_latent = source_latent

    @property
    def mime_type(self) -> str:
        return "application/octet-stream"

    @property
    def mask_image(self) -> Image | None:
        return self._mask_image

    @property
    def source_latent(self) -> LatentArtifact | None:
        return self._source_latent

    @property
    def source_shape(self) -> tuple[int, ...]:
        if self._source_latent is None:
            return ()
        return self._source_latent.source_shape

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.meta or {})

    def to_torch(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self._source_latent is None:
            raise ValueError("MaskedLatentArtifact has no source_latent to convert to torch.")
        return self._source_latent.to_torch(device=device, dtype=dtype)

    def to_mask_tensor(self, device: str | torch.device | None = None) -> torch.Tensor:
        """Return the grayscale mask resized and expanded to the encoded source latent shape."""
        import numpy as np
        import torch  # type: ignore[reportMissingImports]
        import torch.nn.functional as functional  # type: ignore[reportMissingImports]

        if self._source_latent is None:
            raise ValueError("MaskedLatentArtifact has no source_latent to shape its mask tensor.")
        if self._mask_image is None:
            raise ValueError("MaskedLatentArtifact has no mask image to convert to a tensor.")

        # The shape is read from the artifact's metadata so the latent itself is never moved to `device`.
        latent_shape = self._source_latent.shape
        if len(latent_shape) not in (4, 5):
            raise ValueError(
                f"MaskedLatentArtifact source latent must be a 4-D image or 5-D video tensor, got shape {latent_shape}."
            )

        mask_array = np.array(self._mask_image.convert("L"), dtype=np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array)[None, None].to(device=device)
        mask_tensor = functional.interpolate(
            mask_tensor, size=latent_shape[-2:], mode="bilinear", align_corners=False, antialias=True
        )
        if len(latent_shape) == 5:
            return mask_tensor.unsqueeze(2).expand(latent_shape[0], latent_shape[1], latent_shape[2], -1, -1)
        return mask_tensor.expand(latent_shape[0], latent_shape[1], -1, -1)

    def to_state(self) -> dict[str, Any]:
        if self._source_latent is not None:
            msg = (
                f"Attempted to send {self!r} to another process. Failed because its source latent is held in "
                "the process that encoded it and cannot be sent as data."
            )
            raise ValueEncodeError(msg)

        mask_state = None
        if self._mask_image is not None:
            mask_state = {
                "mode": self._mask_image.mode,
                "size": list(self._mask_image.size),
                "bytes": self._mask_image.tobytes(),
            }
        return {
            "mask": mask_state,
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MaskedLatentArtifact:
        mask_state = state["mask"]
        mask_image = None
        if mask_state is not None:
            from PIL import Image as PILImage

            mask_image = PILImage.frombytes(mask_state["mode"], tuple(mask_state["size"]), mask_state["bytes"])
        return cls(
            mask_image=mask_image,
            meta=state["meta"],
        )

    def __bool__(self) -> bool:
        # The blob value is always empty, so truthiness follows the source latent like LatentArtifact does.
        return self._source_latent is not None

    def __reduce__(self) -> Any:
        raise pickle.PicklingError("MaskedLatentArtifact is an in-process-only type and should not be serialized.")

    def __repr__(self) -> str:
        mask_size = None
        if self._mask_image is not None:
            mask_size = self._mask_image.size
        latent_shape = None
        if self._source_latent is not None:
            latent_shape = self._source_latent.shape
        return f"MaskedLatentArtifact(mask_size={mask_size}, source_latent={latent_shape})"
