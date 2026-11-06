"""Artifact that bundles required by inpainting pipelines.

Some pipelines (SDXL) require both an encoded image and masked
image; others (Z-Image) need only the masked image.

This artifact carries:
- the mask PIL image,
- a reference to the original source image artifact,
- the source image latent,
- the masked-image latent,
- the inpaint denoising ``strength``.

The latents are large runtime data and never leave the process that encoded
them. An artifact without latents, such as the typed placeholder the encode
node seeds its output with, travels between processes as data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.artifacts.blob_artifact import BlobArtifact
from griptape_nodes.serialization.values import ValueEncodeError, register_value_codec

from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from PIL.Image import Image

SourceImageInput = ImageArtifact | ImageUrlArtifact | None


@register_value_codec
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

    # ------------------------------------------------------------------
    # Crossing a process boundary.
    #
    # The engine encodes every parameter value it sends to a worker, including the
    # placeholder on the encode node's output. `register_value_codec` makes `to_state`
    # and `from_state` that codec, in place of griptape's `to_dict`.
    # ------------------------------------------------------------------

    def to_state(self) -> dict[str, Any]:
        if self._source_latent is not None or self._masked_latent is not None:
            msg = (
                f"Attempted to send {self!r} to another process. Failed because its latents are held in "
                "the process that encoded them and cannot be sent as data."
            )
            raise ValueEncodeError(msg)

        return {
            "mask_mode": self._mask_image.mode,
            "mask_size": list(self._mask_image.size),
            "mask_bytes": self._mask_image.tobytes(),
            "source_image": self._source_image_artifact,
            "source_shape": list(self._source_shape),
            "strength": self._strength,
            "meta": self.meta,
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> InpaintMaskArtifact:
        from PIL import Image as PILImage  # type: ignore[reportMissingImports]

        mask_image = PILImage.frombytes(state["mask_mode"], tuple(state["mask_size"]), state["mask_bytes"])
        return cls(
            mask_image=mask_image,
            source_image=state["source_image"],
            source_shape=tuple(state["source_shape"]),
            strength=state["strength"],
            meta=state["meta"],
        )

    def __repr__(self) -> str:
        size_str = f"{self._mask_image.size[0]}x{self._mask_image.size[1]}" if self._mask_image else "None"
        src_str = "x".join(str(d) for d in self._source_latent.shape) if self._source_latent is not None else "none"
        masked_str = "x".join(str(d) for d in self._masked_latent.shape) if self._masked_latent is not None else "none"
        return (
            f"InpaintMaskArtifact(mask={size_str}, source_latent={src_str}, "
            f"masked_latent={masked_str}, strength={self._strength})"
        )
