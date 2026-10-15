"""Estimate VAE activation memory by tracing encode and decode on meta tensors.

The tracer builds a weight-free meta-device VAE and tracks live tensor storage
through each public operation, excluding persistent parameters and buffers.
Encode and decode are measured separately; results are cached by VAE class,
config, input shapes, element size, and supported runtime options.
"""

from __future__ import annotations

import json
import threading
import weakref
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch  # type: ignore[reportMissingImports]
from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
from diffusers.models.autoencoders.autoencoder_kl_hunyuanvideo15 import HunyuanVideo15AttnBlock
from torch.utils._python_dispatch import TorchDispatchMode

from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import build_component_on_meta_device

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact


_TRACE_CACHE_MAXSIZE = 128
_MAX_TRACE_OPERATIONS = 1_000_000
_VAE_ESTIMATE_WARNINGS = {
    "AutoencoderKLMiniMaxH3": "VAE activation memory estimate omits backend-dependent attention workspaces; treat it as approximate."
}
_HUNYUAN_MASK_PATCH_LOCK = threading.RLock()


class VaeMetaTraceError(ValueError):
    """Raised when a VAE encode/decode path cannot be traced on meta tensors."""


@dataclass(frozen=True)
class VaeActivationTrace:
    """Peak live tensor-storage bytes for independent encode and decode passes."""

    encode_bytes: int
    decode_bytes: int

    @property
    def estimated_bytes(self) -> int:
        """Return the larger peak from encode and decode."""
        return max(self.encode_bytes, self.decode_bytes)


class _LivenessTracker(TorchDispatchMode):
    def __init__(self, excluded_storage_ids: set[int]) -> None:
        """Track live tensor storage except persistent model storage."""
        super().__init__()
        self._excluded_storage_ids = excluded_storage_ids
        self._tensor_storages: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int]] = {}
        self._storage_sizes: dict[int, int] = {}
        self._storage_refs: defaultdict[int, int] = defaultdict(int)
        self._live_bytes = 0
        self.peak_bytes = 0
        self._operation_count = 0

    def register(self, tensor: torch.Tensor) -> None:
        """Start tracking a tensor used as the trace input."""
        self._track_tensor(tensor)

    def _track_tensor(self, tensor: torch.Tensor) -> None:
        """Count a tensor's storage once while any view remains live."""
        tensor_id = id(tensor)
        existing = self._tensor_storages.get(tensor_id)
        if existing is not None and existing[0]() is tensor:
            return

        try:
            storage = tensor.untyped_storage()
            storage_id = storage._cdata
            storage_bytes = storage.nbytes()
        except (RuntimeError, NotImplementedError) as error:
            msg = f"Cannot inspect live storage for meta tensor '{type(tensor).__name__}': {error}"
            raise VaeMetaTraceError(msg) from error

        if storage_id in self._excluded_storage_ids:
            return

        self._tensor_storages[tensor_id] = (weakref.ref(tensor), storage_id)
        if self._storage_refs[storage_id] == 0:
            self._storage_sizes[storage_id] = storage_bytes
            self._live_bytes += storage_bytes
            self.peak_bytes = max(self.peak_bytes, self._live_bytes)
        self._storage_refs[storage_id] += 1
        weakref.finalize(tensor, self._release_tensor, tensor_id, storage_id)

    def _release_tensor(self, tensor_id: int, storage_id: int) -> None:
        """Remove storage from the live total when its last tensor is released."""
        existing = self._tensor_storages.get(tensor_id)
        if existing is None or existing[1] != storage_id:
            return
        self._tensor_storages.pop(tensor_id, None)
        self._storage_refs[storage_id] -= 1
        if self._storage_refs[storage_id] == 0:
            self._live_bytes -= self._storage_sizes.pop(storage_id)
            self._storage_refs.pop(storage_id, None)

    def _visit(self, value: Any) -> None:
        """Track tensors nested in operation arguments or outputs."""
        if isinstance(value, torch.Tensor):
            self._track_tensor(value)
        elif isinstance(value, dict):
            for child in value.values():
                self._visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self._visit(child)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """Run a tensor operation and update the live-storage peak."""
        self._operation_count += 1
        if self._operation_count > _MAX_TRACE_OPERATIONS:
            msg = f"Meta trace exceeded {_MAX_TRACE_OPERATIONS} dispatched operations."
            raise VaeMetaTraceError(msg)
        self._visit(args)
        self._visit(kwargs or {})
        output = func(*args, **(kwargs or {}))
        self._visit(output)
        return output


def _freeze(value: Any) -> Any:
    """Convert runtime options into a hashable, type-preserving cache key."""
    if isinstance(value, dict):
        items = ((_freeze(key), _freeze(child)) for key, child in value.items())
        return ("dict", tuple(sorted(items, key=repr)))
    if isinstance(value, tuple):
        return ("tuple", tuple(_freeze(child) for child in value))
    if isinstance(value, list):
        return ("list", tuple(_freeze(child) for child in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_freeze(child) for child in value), key=repr)))
    if isinstance(value, (str, int, float, bool, type(None))):
        return ("scalar", type(value).__name__, value)
    return ("repr", repr(value))


def _thaw(value: Any) -> Any:
    """Restore a frozen runtime option before applying it to a meta VAE."""
    kind, contents, *extra = value
    if kind == "dict":
        return {_thaw(key): _thaw(child) for key, child in contents}
    if kind == "tuple":
        return tuple(_thaw(child) for child in contents)
    if kind == "list":
        return [_thaw(child) for child in contents]
    if kind == "set":
        return {_thaw(child) for child in contents}
    if kind == "scalar":
        return extra[0]
    if kind == "repr":
        return contents
    return value


def _config_dict(vae: torch.nn.Module) -> dict[str, Any]:
    """Return the VAE config as a plain dictionary."""
    config = getattr(vae, "config", None)
    if config is None:
        msg = f"Cannot meta-trace VAE '{type(vae).__name__}': it exposes no config."
        raise VaeMetaTraceError(msg)
    if hasattr(config, "to_dict"):
        return config.to_dict()
    try:
        return dict(config)
    except (TypeError, ValueError) as error:
        msg = f"Cannot meta-trace VAE '{type(vae).__name__}': its config cannot be converted to a dict."
        raise VaeMetaTraceError(msg) from error


def _option_key(vae: torch.nn.Module) -> tuple[Any, ...]:
    """Return hashable keys for runtime options copied to the meta VAE."""
    values = []
    for name, value in vars(vae).items():
        if name.startswith(("use_", "tile_")) and isinstance(
            value, (str, int, float, bool, type(None), tuple, list)
        ):
            values.append((name, _freeze(value)))
    return tuple(sorted(values))


def _construct_meta_vae(
    vae_class: type[torch.nn.Module], config_json: str, element_size: int, options: tuple[Any, ...]
) -> torch.nn.Module:
    """Build a weight-free meta VAE and apply cached runtime options."""
    dtype = torch.bfloat16 if element_size <= 2 else torch.float32
    try:
        with init_empty_weights():
            meta_vae = build_component_on_meta_device(vae_class, json.loads(config_json), dtype)
        torch.nn.Module.to(meta_vae, device="meta", dtype=dtype)
    except Exception as error:
        msg = f"Cannot construct config-only meta VAE '{vae_class.__name__}': {error}"
        raise VaeMetaTraceError(msg) from error

    for name, value in options:
        setattr(meta_vae, name, _thaw(value))
    return meta_vae


@contextmanager
def _fast_hunyuan_meta_mask(vae: torch.nn.Module) -> Iterator[None]:
    """Skip Hunyuan mask-value construction while preserving its meta shape."""
    if type(vae).__name__ != "AutoencoderKLHunyuanVideo15":
        yield
        return

    with _HUNYUAN_MASK_PATCH_LOCK:
        attentions = [module for module in vae.modules() if isinstance(module, HunyuanVideo15AttnBlock)]
        original_methods = [
            (attention, vars(attention).get("prepare_causal_attention_mask")) for attention in attentions
        ]

        def meta_mask(n_frame, n_hw, dtype, device, batch_size=None):
            """Create the causal mask shape without filling its values."""
            seq_len = n_frame * n_hw
            mask = torch.empty((seq_len, seq_len), dtype=dtype, device=device)
            if batch_size is not None:
                mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
            return mask

        for attention, _original in original_methods:
            attention.prepare_causal_attention_mask = meta_mask
        try:
            yield
        finally:
            for attention, original in original_methods:
                if original is None:
                    del attention.prepare_causal_attention_mask
                else:
                    attention.prepare_causal_attention_mask = original


def _trace_pass(
    vae: torch.nn.Module,
    operation: str,
    input_shape: tuple[int, ...],
    element_size: int,
) -> int:
    """Trace one VAE pass and return its peak live activation bytes."""
    if len(input_shape) not in (4, 5) or any(dimension <= 0 for dimension in input_shape):
        msg = f"Cannot meta-trace VAE {operation}: invalid tensor shape {input_shape}."
        raise VaeMetaTraceError(msg)

    dtype = torch.bfloat16 if element_size <= 2 else torch.float32
    input_tensor = torch.empty(input_shape, device="meta", dtype=dtype)
    excluded_storage_ids = {tensor.untyped_storage()._cdata for tensor in (*vae.parameters(), *vae.buffers())}
    tracker = _LivenessTracker(excluded_storage_ids)
    tracker.register(input_tensor)

    try:
        with torch.inference_mode(), _fast_hunyuan_meta_mask(vae), tracker:
            vae_operation = getattr(vae, operation, None)
            if not callable(vae_operation):
                msg = f"VAE '{type(vae).__name__}' has no callable {operation} method."
                raise VaeMetaTraceError(msg)
            if operation == "encode":
                output = vae_operation(input_tensor, return_dict=True)
                posterior = getattr(output, "latent_dist", None)
                if posterior is not None:
                    output = posterior.sample()
            else:
                output = vae_operation(input_tensor, return_dict=True)
        peak_bytes = tracker.peak_bytes
        del output
        return peak_bytes
    except Exception as error:
        msg = f"Meta trace failed for VAE '{type(vae).__name__}' {operation} shape {input_shape}: {error}"
        raise VaeMetaTraceError(msg) from error
    finally:
        del tracker, input_tensor


@lru_cache(maxsize=_TRACE_CACHE_MAXSIZE)
def _trace_cached(
    vae_class: type[torch.nn.Module],
    config_json: str,
    source_shape: tuple[int, ...],
    latent_shape: tuple[int, ...],
    element_size: int,
    options: tuple[Any, ...],
) -> VaeActivationTrace:
    """Cache structural VAE traces; simultaneous first misses may trace twice."""
    meta_vae = _construct_meta_vae(vae_class, config_json, element_size, options)
    try:
        return VaeActivationTrace(
            encode_bytes=_trace_pass(meta_vae, "encode", source_shape, element_size),
            decode_bytes=_trace_pass(meta_vae, "decode", latent_shape, element_size),
        )
    except VaeMetaTraceError:
        raise
    except Exception as error:
        msg = f"Meta trace failed for VAE '{vae_class.__name__}': {error}"
        raise VaeMetaTraceError(msg) from error


def trace_vae_activation_memory(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    element_size: int,
) -> VaeActivationTrace:
    """Trace public encode/decode paths on meta tensors and cache exact-shape results."""
    config_json = json.dumps(_config_dict(vae), sort_keys=True, default=repr)
    return _trace_cached(
        type(vae),
        config_json,
        latent.source_shape,
        latent.shape,
        element_size,
        _option_key(vae),
    )


def vae_estimate_warning(vae: torch.nn.Module) -> str | None:
    """Return the class-specific warning for a VAE memory estimate, if any."""
    return _VAE_ESTIMATE_WARNINGS.get(type(vae).__name__)


def estimate_vae_activation_bytes(
    vae: torch.nn.Module,
    latent: LatentArtifact,
    element_size: int,
) -> int:
    """Return the larger traced peak from VAE encode or decode."""
    return trace_vae_activation_memory(vae, latent, element_size).estimated_bytes


__all__ = [
    "VaeActivationTrace",
    "VaeMetaTraceError",
    "estimate_vae_activation_bytes",
    "trace_vae_activation_memory",
    "vae_estimate_warning",
]
