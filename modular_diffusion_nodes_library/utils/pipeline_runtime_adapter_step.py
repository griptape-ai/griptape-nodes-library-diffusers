"""Per-generation pipeline transformations.

`ActivationStep` is the abstract base for context-managed transformations
applied around a single generation (e.g. activating LoRA adapters,
attaching ControlNets). Implementations enter to activate, exit to
deactivate, and expose a `metadata` dict describing what they do.

Not thread-safe — assumes serial generation. Add per-pipeline locks when
parallel generation lands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

#: Every `KIND` that has been declared, so `rebuild_runtime_adapter_step` can invert `metadata`.
_STEP_KINDS: dict[str, type[PipelineRuntimeAdapterStep]] = {}


class PipelineRuntimeAdapterStep(ABC):
    """Context-managed pipeline transformation applied per generation.

    Implementations enter to activate, exit to deactivate. The yielded pipe
    is what the caller should use for the generation.

    Implementations must also expose a `metadata` dict describing what they
    do (e.g. which adapters they activate).

    `metadata` is the wire form: an artifact carrying steps travels between
    processes, so `_metadata` must record everything `_from_metadata` needs to
    rebuild the step, and both sides must stay in step.
    """

    KIND: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = cls.__dict__.get("KIND")
        if kind is not None:
            _STEP_KINDS[kind] = cls

    @property
    def metadata(self) -> dict[str, Any]:
        return {"kind": self.KIND, **self._metadata()}

    @abstractmethod
    def _metadata(self) -> dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def _from_metadata(cls, data: dict[str, Any]) -> PipelineRuntimeAdapterStep:
        """Rebuild a step from what `_metadata` recorded."""

    @abstractmethod
    @contextmanager
    def activate(self, pipe: Any, *, node_name: str | None = None) -> Iterator[Any]: ...


def rebuild_runtime_adapter_step(data: dict[str, Any]) -> PipelineRuntimeAdapterStep:
    """Invert `PipelineRuntimeAdapterStep.metadata`.

    Raises:
        ValueError: if `data` names a kind this process has no implementation for, which means an
            artifact arrived from a process running a different version of this library.
    """
    kind = data.get("kind")
    step_cls = _STEP_KINDS.get(kind) if isinstance(kind, str) else None
    if step_cls is None:
        msg = (
            f"Attempted to rebuild a pipeline runtime adapter step. Failed with kind='{kind}' because "
            f"no step class declares it. Known kinds: {sorted(_STEP_KINDS)}."
        )
        raise ValueError(msg)
    return step_cls._from_metadata(data)  # noqa: SLF001 - the inverse of this class's own `metadata`
