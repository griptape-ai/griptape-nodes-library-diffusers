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


class PipelineRuntimeAdapterStep(ABC):
    """Context-managed pipeline transformation applied per generation.

    Implementations enter to activate, exit to deactivate. The yielded pipe
    is what the caller should use for the generation.

    Implementations must also expose a `metadata` dict describing what they
    do (e.g. which adapters they activate).
    """

    KIND: ClassVar[str]

    @property
    def metadata(self) -> dict[str, Any]:
        return {"kind": self.KIND, **self._metadata()}

    @abstractmethod
    def _metadata(self) -> dict[str, Any]: ...

    @abstractmethod
    @contextmanager
    def activate(self, pipe: Any, *, node_name: str | None = None) -> Iterator[Any]: ...
