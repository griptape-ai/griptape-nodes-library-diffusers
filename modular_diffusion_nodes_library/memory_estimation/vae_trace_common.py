"""Shared utilities for meta-device VAE activation tracing."""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch  # type: ignore[reportMissingImports]
from torch.utils._python_dispatch import TorchDispatchMode

_MAX_TRACE_OPERATIONS = 1_000_000


class VaeMetaTraceError(ValueError):
    """Raised when a VAE encode/decode path cannot be traced on meta tensors."""


@contextmanager
def patched_attribute(target: Any, name: str, value: Any) -> Iterator[None]:
    """Set an instance attribute for the block, then restore the previous instance value or remove it."""
    original = vars(target).get(name)
    setattr(target, name, value)
    try:
        yield
    finally:
        if original is None:
            delattr(target, name)
        else:
            setattr(target, name, original)


class LivenessTracker(TorchDispatchMode):
    def __init__(self, module: torch.nn.Module) -> None:
        """Track live tensor storage except the module's parameters and buffers."""
        super().__init__()
        self._excluded_storage_ids = {
            tensor.untyped_storage()._cdata for tensor in (*module.parameters(), *module.buffers())
        }
        self._tensor_storages: dict[int, tuple[weakref.ReferenceType[torch.Tensor], int]] = {}
        self._storage_sizes: dict[int, int] = {}
        self._storage_refs: defaultdict[int, int] = defaultdict(int)
        self.live_bytes = 0
        self.peak_bytes = 0
        self._operation_count = 0

    def register(self, tensor: torch.Tensor) -> None:
        """Start tracking a tensor used as the trace input."""
        self._track_tensor(tensor)

    @contextmanager
    def paused(self) -> Iterator[None]:
        """Stop intercepting tensor operations inside the block."""
        self.__exit__(None, None, None)
        try:
            yield
        finally:
            self.__enter__()

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
            self.live_bytes += storage_bytes
            self.peak_bytes = max(self.peak_bytes, self.live_bytes)
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
            self.live_bytes -= self._storage_sizes.pop(storage_id)
            self._storage_refs.pop(storage_id, None)

    def visit(self, value: Any) -> None:
        """Track tensors nested in operation arguments or outputs."""
        if isinstance(value, torch.Tensor):
            self._track_tensor(value)
        elif isinstance(value, dict):
            for child in value.values():
                self.visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                self.visit(child)

    def __torch_dispatch__(self, func, _types, args=(), kwargs=None):
        """Run a tensor operation and update the live-storage peak."""
        self._operation_count += 1
        if self._operation_count > _MAX_TRACE_OPERATIONS:
            msg = f"Meta trace exceeded {_MAX_TRACE_OPERATIONS} dispatched operations."
            raise VaeMetaTraceError(msg)
        self.visit(args)
        self.visit(kwargs or {})
        output = func(*args, **(kwargs or {}))
        self.visit(output)
        return output
