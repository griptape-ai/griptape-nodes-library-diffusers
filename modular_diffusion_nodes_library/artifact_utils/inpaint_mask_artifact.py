"""Artifact that bundles required by inpainting pipelines.

Some pipelines (SDXL) require both an encoded image and masked
image; others (Z-Image) need only the masked image.

This artifact carries:
- the mask PIL image,
- a reference to the original source image artifact,
- the source image latent,
- the masked-image latent,
- the inpaint denoising ``strength``.

This artifact is intentionally NOT serializable as it holds large runtime
data (PIL images, tensors).
"""

from __future__ import annotations

from typing import Any

import torch  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.blob_artifact import BlobArtifact
from PIL.Image import Image

from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

SourceImageInput = ImageArtifact | ImageUrlArtifact | None


class InpaintMaskArtifact(BlobArtifact):
    """Bundles a mask image, source/masked latents and denoising strength."""

    _mask_image: Image
    _source_image_artifact: SourceImageInput
    _source_latent: torch.Tensor | None
    _masked_latent: torch.Tensor | None
    _source_shape: tuple[int, ...]
    _strength: float

    def __init__(
        self,
        mask_image: Image,
        source_image: SourceImageInput = None,
        source_latent: torch.Tensor | None = None,
        masked_latent: torch.Tensor | None = None,
        source_shape: tuple[int, ...] = (),
        strength: float = 1.0,
        meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(value=b"", meta=meta or {})
        self._mask_image = mask_image
        self._source_image_artifact = source_image
        self._source_latent = source_latent
        self._masked_latent = masked_latent
        self._source_shape = tuple(source_shape)
        self._strength = float(strength)

    @property
    def mime_type(self) -> str:
        return "application/octet-stream"

    @property
    def mask_image(self) -> Image:
        """Full-resolution mask PIL image (L mode, white = inpaint region)."""
        return self._mask_image

    @property
    def source_image_artifact(self) -> SourceImageInput:
        """Original source image artifact (URL or in-memory), or None if not stored."""
        return self._source_image_artifact

    def source_image_pil(self) -> Image | None:
        """Lazily materialize the source image as a PIL Image (RGB)."""
        artifact = self._source_image_artifact
        if artifact is None:
            return None
        if isinstance(artifact, ImageUrlArtifact):
            artifact = load_image_from_url_artifact(artifact)
        return image_artifact_to_pil(artifact).convert("RGB")

    @property
    def source_latent(self) -> torch.Tensor | None:
        """VAE-encoded source-image latent, or None if not stored."""
        return self._source_latent

    @property
    def masked_latent(self) -> torch.Tensor | None:
        """VAE-encoded masked image latent, or None if not stored."""
        return self._masked_latent

    @property
    def strength(self) -> float:
        """Inpaint denoising strength (0.0-1.0)."""
        return self._strength

    @property
    def metadata(self) -> dict[str, Any]:
        """Read-only view of the artifact's metadata.
        Carries arbitrary provenance from upstream producers.
        """
        return dict(self.meta or {})

    # ------------------------------------------------------------------
    # LatentArtifact duck-typing — lets this artifact stand in as input_latent.
    # ------------------------------------------------------------------

    @property
    def source_shape(self) -> tuple[int, ...]:
        return self._source_shape

    def to_torch(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if self._source_latent is None:
            raise ValueError("InpaintMaskArtifact has no source_latent to convert to torch.")
        tensor = self._source_latent
        if device is None and dtype is None:
            return tensor
        return tensor.to(device=device, dtype=dtype)

    def __repr__(self) -> str:
        size_str = f"{self._mask_image.size[0]}x{self._mask_image.size[1]}" if self._mask_image else "None"
        src_str = "x".join(str(d) for d in self._source_latent.shape) if self._source_latent is not None else "none"
        masked_str = "x".join(str(d) for d in self._masked_latent.shape) if self._masked_latent is not None else "none"
        return (
            f"InpaintMaskArtifact(mask={size_str}, source_latent={src_str}, "
            f"masked_latent={masked_str}, strength={self._strength})"
        )
