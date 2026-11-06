"""Make VAE meta traces fast by replaying repeated encoder/decoder forwards.

Diffusers' `vae.encode`/`vae.decode` split big inputs into tiles (and video into
frame chunks), calling `vae.encoder(...)` / `vae.decoder(...)` once per piece.
Nearly every call sees the same input shape, so tracing each one is wasted work.

How `reuse_tiled_forwards` works:
1. `encoder.forward` / `decoder.forward` are wrapped by `_memoized_forward`.
2. First call for an input `_signature`: run the real forward under the tracker,
   then `_record` its peak bytes and a `_TensorView` description of every output
   tensor and every in-place-updated cache list (Wan/Qwen `feat_cache`).
3. Later calls with the same signature: skip the forward and `_replay` the record.
   Replay allocates fresh meta storage of the recorded sizes, so the tracker still
   counts outputs and caches exactly as a real run would.
4. `blend_v/h/t` are limited to one overlap row. They blend in place row by row,
   so one row produces the same temporary allocations as all of them.

Wrapper code (slicing, stitching, cache bookkeeping) still runs for every tile.
Very large grids are handled separately by `vae_trace._trace_pass`, which traces
4- and 5-tile samples and extrapolates. Backend workspaces (e.g. attention
scratch space) are never captured.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from modular_diffusion_nodes_library.memory_estimation.vae_trace_common import LivenessTracker, patched_attribute

_TRACE_LOCK = threading.RLock()
_SUPPORTED_VAES = {
    "AutoencoderKL",
    "AutoencoderKLFlux2",
    "AutoencoderKLHunyuanVideo15",
    "AutoencoderKLLTXVideo",
    "AutoencoderKLLTX2Video",
    "AutoencoderKLMiniMaxH3",
    "AutoencoderKLQwenImage",
    "AutoencoderKLWan",
}


@dataclass(frozen=True)
class _TensorView:
    """A recorded tensor without its storage: which storage it viewed, and how.

    `storage` indexes the replay's base list: input storages first (in first-seen
    order), then storages the forward allocated (`_ForwardTrace.storages`).
    A real meta tensor can't be kept instead because the tracker would count it live forever.
    """

    storage: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int
    dtype: torch.dtype


@dataclass
class _ForwardTrace:
    """Everything needed to fake a repeat call to one encoder/decoder forward."""

    peak_bytes: int
    storages: list[tuple[int, torch.dtype]]
    result: Any
    # Exact layouts of inputs whose storage the result aliases; replay restores views at recorded strides.
    pinned_inputs: dict[int, tuple[Any, ...]]


@dataclass
class _ForwardPhasePeaks:
    """Largest storage observed in wrapper code and inside a core forward."""

    loop_peak_bytes: int = 0
    forward_peak_bytes: int = 0


def _signature(value: Any) -> Any:
    """Return a hashable cache key for forward arguments."""
    if isinstance(value, torch.Tensor):
        dense = []
        expected_stride = 1
        for size, stride in zip(reversed(value.shape), reversed(value.stride()), strict=True):
            dense.append(size == 1 or stride == expected_stride)
            expected_stride *= size
        return ("layout", tuple(value.shape), tuple(dense), value.dtype)
    if isinstance(value, dict):
        return tuple((key, _signature(child)) for key, child in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(_signature(child) for child in value)
    return value


def _leaves[T](value: Any, leaf_type: type[T]) -> Iterator[T]:
    """Yield matching values nested in containers, in traversal order."""
    if isinstance(value, leaf_type):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _leaves(child, leaf_type)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _leaves(child, leaf_type)


def _map_leaves(value: Any, leaf_type: type, transform: Callable[[Any], Any]) -> Any:
    """Apply `transform` to every `leaf_type` value, rebuilding dict/list/tuple containers around it."""
    if isinstance(value, leaf_type):
        return transform(value)
    if isinstance(value, dict):
        return {key: _map_leaves(child, leaf_type, transform) for key, child in value.items()}
    if isinstance(value, list):
        return [_map_leaves(child, leaf_type, transform) for child in value]
    if isinstance(value, tuple):
        return tuple(_map_leaves(child, leaf_type, transform) for child in value)
    return value


def _view_of(described: _TensorView, bases: list[torch.Tensor]) -> torch.Tensor:
    """Recreate a recorded tensor as a view over the storage it originally used."""
    base = bases[described.storage]
    if base.dtype != described.dtype:
        base = base.view(described.dtype)
    return base.as_strided(described.shape, described.stride, described.offset)


def _record(
    output: Any,
    states: list[list[Any]],
    input_refs: list[tuple[int, weakref.ref]],
    input_layouts: list[tuple[Any, ...]],
    peak_bytes: int,
) -> _ForwardTrace:
    """Describe a finished forward's output and cache lists so `_replay` can recreate them."""
    # Number input storages 0..N-1 in first-seen order; new storages are numbered from N.
    external_ids: dict[int, int] = {}
    live_ids = set()
    for storage_id, ref in input_refs:
        external_ids.setdefault(storage_id, len(external_ids))
        if ref() is not None:
            live_ids.add(storage_id)
    # A dead input's storage id may have been reused by a new allocation, so only live ones are external.
    storage_ids = {storage_id: index for storage_id, index in external_ids.items() if storage_id in live_ids}
    storages: list[tuple[int, torch.dtype]] = []

    def describe(tensor: torch.Tensor) -> _TensorView:
        storage = tensor.untyped_storage()
        storage_id = storage._cdata
        if storage_id not in storage_ids:
            storage_ids[storage_id] = len(external_ids) + len(storages)
            storages.append((storage.nbytes(), tensor.dtype))
        return _TensorView(
            storage_ids[storage_id],
            tuple(tensor.shape),
            tuple(tensor.stride()),
            int(tensor.storage_offset()),
            tensor.dtype,
        )

    result = _map_leaves((output, [list(state) for state in states]), torch.Tensor, describe)
    # Results that alias an input (views, in-place ops) are only valid for that exact input layout.
    aliased = {view.storage for view in _leaves(result, _TensorView) if view.storage < len(external_ids)}
    pinned_inputs = {
        position: input_layouts[position]
        for position, (storage_id, _ref) in enumerate(input_refs)
        if external_ids[storage_id] in aliased
    }
    return _ForwardTrace(peak_bytes, storages, result, pinned_inputs)


def _replay(trace: _ForwardTrace, inputs: Any, states: list[list[Any]], tracker: LivenessTracker) -> Any:
    """Recreate a recorded forward's output and cache updates without running it."""
    with tracker.paused():
        # Same numbering as `_record`: input storages first, then fresh storage per allocation.
        inputs_by_storage: dict[int, torch.Tensor] = {}
        for tensor in _leaves(inputs, torch.Tensor):
            inputs_by_storage.setdefault(tensor.untyped_storage()._cdata, tensor)
        bases = list(inputs_by_storage.values())
        for size, dtype in trace.storages:
            bases.append(torch.empty((size // dtype.itemsize,), device="meta", dtype=dtype))
        output, restored_states = _map_leaves(trace.result, _TensorView, partial(_view_of, bases=bases))
        del bases
        for state, restored in zip(states, restored_states, strict=True):
            state[:] = restored
        # Count new storage only after replaced cache entries are released.
        tracker.visit((output, restored_states))
    return output


def _memoized_forward(
    forward: Callable[..., Any],
    tracker: LivenessTracker,
    phase_peaks: _ForwardPhasePeaks,
    traces: dict[Any, _ForwardTrace],
) -> Callable[..., Any]:
    """Wrap an encoder/decoder forward: trace the first call per `_signature`, replay the rest."""

    def run(*args: Any, **kwargs: Any) -> Any:
        key = (_signature(args), _signature(kwargs))
        inputs = (args, kwargs)
        input_tensors = list(_leaves(inputs, torch.Tensor))
        input_layouts = [
            (tuple(tensor.shape), tuple(tensor.stride()), int(tensor.storage_offset())) for tensor in input_tensors
        ]
        input_refs = [(tensor.untyped_storage()._cdata, weakref.ref(tensor)) for tensor in input_tensors]
        # Strong refs would keep replaced cache entries alive past the forward and inflate the peak.
        del input_tensors
        # List arguments (Wan/Qwen feat_cache, feat_idx) are mutated in place; replay must mutate them too.
        states = [value for value in (*args, *kwargs.values()) if isinstance(value, list)]
        entry_bytes = tracker.live_bytes
        phase_peaks.loop_peak_bytes = max(phase_peaks.loop_peak_bytes, tracker.peak_bytes)

        trace = traces.get(key)
        # The key ignores exact strides, so a result aliasing an input is reusable only at the recorded layout.
        if trace is not None and any(
            input_layouts[position] != layout for position, layout in trace.pinned_inputs.items()
        ):
            trace = None
        if trace is None:
            # Run for real under the outer tracker so the recorded delta matches a full trace.
            tracker.peak_bytes = entry_bytes
            output = forward(*args, **kwargs)
            trace = _record(output, states, input_refs, input_layouts, tracker.peak_bytes - entry_bytes)
            traces[key] = trace
            phase_peaks.forward_peak_bytes = max(phase_peaks.forward_peak_bytes, tracker.peak_bytes)
        else:
            output = _replay(trace, inputs, states, tracker)
            phase_peaks.forward_peak_bytes = max(phase_peaks.forward_peak_bytes, entry_bytes + trace.peak_bytes)

        # Start the next wrapper phase from what is live now.
        tracker.peak_bytes = tracker.live_bytes
        return output

    return run


def _single_blend_iteration(blend: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]:
    """Blend a single overlap row, which is enough to measure its peak."""

    def run(first: torch.Tensor, second: torch.Tensor, extent: int) -> torch.Tensor:
        if extent <= 0:
            return second
        return blend(first, second, 1)

    return run


def spatial_tile_strides(vae: Any, operation: str) -> tuple[int, int] | None:
    """Return the wrapper's (height, width) tile stride in the operation's input units, or None if untiled."""
    name = type(vae).__name__
    if name not in _SUPPORTED_VAES or not getattr(vae, "use_tiling", False):
        return None

    encode = operation == "encode"
    match name:
        case "AutoencoderKL" | "AutoencoderKLFlux2":
            if encode:
                size = vae.tile_sample_min_size
            else:
                size = vae.tile_latent_min_size
            stride = int(size * (1 - vae.tile_overlap_factor))
            stride_h, stride_w = stride, stride
        case "AutoencoderKLHunyuanVideo15":
            if encode:
                sizes = (vae.tile_sample_min_height, vae.tile_sample_min_width)
            else:
                sizes = (vae.tile_latent_min_height, vae.tile_latent_min_width)
            overlap = 1 - vae.tile_overlap_factor
            stride_h, stride_w = int(sizes[0] * overlap), int(sizes[1] * overlap)
        case "AutoencoderKLLTXVideo" | "AutoencoderKLLTX2Video" | "AutoencoderKLQwenImage" | "AutoencoderKLWan":
            stride_h, stride_w = vae.tile_sample_stride_height, vae.tile_sample_stride_width
            if encode and name == "AutoencoderKLWan":
                # Wan 2.2 tiles the patchified input; LTX patchifies inside the encoder instead.
                patch_size = vae.config.get("patch_size") or 1
                stride_h, stride_w = stride_h * patch_size, stride_w * patch_size
            if not encode:
                stride_h //= vae.spatial_compression_ratio
                stride_w //= vae.spatial_compression_ratio
        case "AutoencoderKLMiniMaxH3":
            stride_h = vae.tile_sample_min_height - vae.tile_sample_min_overlap_height
            stride_w = vae.tile_sample_min_width - vae.tile_sample_min_overlap_width
            if not encode:
                stride_h //= vae.spatial_compression_ratio
                stride_w //= vae.spatial_compression_ratio
        case _:
            msg = f"Attempted to read tile strides. Failed for VAE '{name}' because it has no stride mapping."
            raise ValueError(msg)
    if stride_h <= 0 or stride_w <= 0:
        return None
    return (int(stride_h), int(stride_w))


@contextmanager
def reuse_tiled_forwards(
    vae: Any,
    tracker: LivenessTracker,
    *,
    enabled: bool = True,
    shared_traces: dict[str, dict[Any, _ForwardTrace]] | None = None,
) -> Iterator[_ForwardPhasePeaks]:
    """Temporarily patch a meta VAE so repeated encoder/decoder forwards are replayed."""
    phase_peaks = _ForwardPhasePeaks()
    if not enabled or type(vae).__name__ not in _SUPPORTED_VAES:
        yield phase_peaks
        return
    if shared_traces is None:
        shared_traces = {}
    with _TRACE_LOCK, ExitStack() as patches:
        for name in ("encoder", "decoder"):
            module = getattr(vae, name)
            memoized = _memoized_forward(module.forward, tracker, phase_peaks, shared_traces.setdefault(name, {}))
            patches.enter_context(patched_attribute(module, "forward", memoized))
        for name in ("blend_v", "blend_h", "blend_t"):
            if hasattr(vae, name):
                patches.enter_context(patched_attribute(vae, name, _single_blend_iteration(getattr(vae, name))))
        yield phase_peaks
