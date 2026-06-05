from __future__ import annotations

import pickle
from typing import Any

import torch  # type: ignore[reportMissingImports]
from griptape.artifacts.base_artifact import BaseArtifact


class LatentArtifact(BaseArtifact):
    """In-process latent tensor wrapper.

    Wraps a torch.Tensor for transport between nodes within a single process.
    This type is deliberately unserializable — any Parameter holding a
    LatentArtifact must be marked serializable=False.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: str,
        source_shape: tuple[int, ...],
        local_tensor: torch.Tensor | None = None,
        meta: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        merged_meta = {} if meta is None else meta.copy()
        merged_meta["shape"] = list(shape)
        merged_meta["dtype"] = dtype
        merged_meta["source_shape"] = list(source_shape)

        super().__init__(value=None, meta=merged_meta, **kwargs)

        self._local_tensor = local_tensor

    @property
    def shape(self) -> tuple[int, ...]:
        shape = self.meta.get("shape", [])
        return tuple(int(dim) for dim in shape)

    @property
    def dtype(self) -> str:
        return str(self.meta.get("dtype", "unknown"))

    @property
    def source_shape(self) -> tuple[int, ...]:
        """Shape of the original image before VAE encoding, e.g. (H, W, C) or (C, H, W)."""
        raw = self.meta.get("source_shape", [])
        return tuple(int(dim) for dim in raw)

    @classmethod
    def from_torch(
        cls,
        tensor: torch.Tensor,
        *,
        source_shape: tuple[int, ...],
        meta: dict[str, Any] | None = None,
    ) -> LatentArtifact:
        """Create a LatentArtifact from a torch.Tensor.

        Args:
            tensor: The latent tensor to store.
            source_shape: Shape of the source image prior to VAE encoding, e.g. ``(N, C, H, W)``.
                Persisted in ``meta`` and available via the ``source_shape`` property so downstream
                nodes can recover original spatial dimensions.
            meta: Optional dict of additional metadata to attach to the artifact.
                Keys ``shape``, ``dtype``, ``device``, and ``is_sparse`` are reserved and
                will be overwritten.
        """
        merged_meta = {} if meta is None else meta.copy()
        merged_meta["device"] = str(tensor.device)
        merged_meta["is_sparse"] = bool(tensor.is_sparse)

        local_tensor = tensor.detach()

        return cls(
            shape=tuple(int(dim) for dim in local_tensor.shape),
            dtype=str(local_tensor.dtype),
            source_shape=source_shape,
            local_tensor=local_tensor,
            meta=merged_meta,
        )

    def to_torch(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        local_tensor = getattr(self, "_local_tensor", None)
        if local_tensor is None:
            return torch.Tensor()

        if device is None and dtype is None:
            return local_tensor

        return local_tensor.to(device=device, dtype=dtype)

    def to_text(self) -> str:
        return repr(self)

    def to_dict(self) -> dict[str, Any]:
        tensor = self.to_torch().detach().cpu()
        if tensor.numel() == 0:
            return {
                "type": "LatentArtifact",
                "shape": list(self.shape),
                "dtype": self.dtype,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
                "sample": [],
            }
        flattened = tensor.flatten()
        sample = flattened[: min(8, flattened.numel())].tolist()
        minimum = tensor.min().item()
        maximum = tensor.max().item()
        mean = tensor.mean().item()
        std = tensor.std().item() if tensor.numel() > 1 else 0.0

        return {
            "type": "LatentArtifact",
            "shape": list(self.shape),
            "dtype": self.dtype,
            "min": minimum,
            "max": maximum,
            "mean": mean,
            "std": std,
            "sample": sample,
        }

    def __reduce__(self) -> None:
        raise pickle.PicklingError("LatentArtifact is an in-process-only type and should not be serialized.")

    def __bool__(self) -> bool:
        return getattr(self, "_local_tensor", None) is not None

    def __add__(self, other: LatentArtifact) -> LatentArtifact:
        return self._elementwise_binary_op(other, operation_name="add", operation=lambda left, right: left + right)

    def __sub__(self, other: LatentArtifact) -> LatentArtifact:
        return self._elementwise_binary_op(other, operation_name="subtract", operation=lambda left, right: left - right)

    def __mul__(self, other: LatentArtifact) -> LatentArtifact:
        return self._elementwise_binary_op(other, operation_name="multiply", operation=lambda left, right: left * right)

    def _elementwise_binary_op(
        self,
        other: LatentArtifact,
        *,
        operation_name: str,
        operation: Any,
    ) -> LatentArtifact:
        if not isinstance(other, LatentArtifact):
            msg = (
                f"Attempted to {operation_name} latent artifacts. Failed with other='{type(other).__name__}' "
                f"because both operands must be LatentArtifact values."
            )
            raise TypeError(msg)

        left = self.to_torch()
        right = other.to_torch(device=left.device, dtype=left.dtype)

        if left.shape != right.shape:
            msg = (
                f"Attempted to {operation_name} latent artifacts. Failed with left shape {tuple(left.shape)} and "
                f"right shape {tuple(right.shape)} because both tensors must have the same shape."
            )
            raise ValueError(msg)

        combined_meta = {} if other.meta is None else other.meta.copy()
        combined_meta.update(self.meta)

        result = operation(left, right)
        return LatentArtifact.from_torch(result, source_shape=self.source_shape, meta=combined_meta)
