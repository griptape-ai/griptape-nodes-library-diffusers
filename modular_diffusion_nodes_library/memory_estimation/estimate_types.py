"""Return types for pipeline memory estimation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from modular_diffusion_nodes_library.memory_estimation.lora_memory import LoraAdapterMemoryEstimate


class ComponentRole(StrEnum):
    """Semantic category used to select a component's memory estimation logic."""

    TEXT_ENCODER = "text_encoder"
    DENOISER = "denoiser"
    VAE = "vae"
    CONTROLNET = "controlnet"
    OTHER = "other"


_BYTES_PER_GB = 1024**3


def _bytes_to_gb(num_bytes: int) -> float:
    """Convert a byte count to GB, rounded to a sane display precision."""
    return round(num_bytes / _BYTES_PER_GB, 4)


@dataclass(frozen=True)
class ComponentMemoryEstimate:
    """Memory estimate for a single resident pipeline component (e.g. a transformer or a VAE)."""

    component_name: str
    role: ComponentRole
    weight_bytes: int
    activation_bytes: int
    total_bytes: int
    is_estimated: bool = False
    warning: str | None = None
    tooltip: str | None = None

    @classmethod
    def create(
        cls,
        component_name: str,
        role: ComponentRole | str,
        weight_bytes: int,
        activation_bytes: int,
        *,
        is_estimated: bool = False,
        warning: str | None = None,
        tooltip: str | None = None,
    ) -> ComponentMemoryEstimate:
        return cls(
            component_name=component_name,
            role=ComponentRole(role),
            weight_bytes=weight_bytes,
            activation_bytes=activation_bytes,
            total_bytes=weight_bytes + activation_bytes,
            is_estimated=is_estimated,
            warning=warning,
            tooltip=tooltip,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_name": self.component_name,
            "role": self.role.value,
            "weight_gb": _bytes_to_gb(self.weight_bytes),
            "activation_gb": _bytes_to_gb(self.activation_bytes),
            "total_gb": _bytes_to_gb(self.total_bytes),
            "is_estimated": self.is_estimated,
            "warning": self.warning,
            "tooltip": self.tooltip,
        }


@dataclass(frozen=True)
class PipelineMemoryEstimate:
    """Per-component memory estimate for an entire pipeline, loaded or not.

    Attributes:
        pipeline_name: Name of the pipeline the estimate was computed for.
        offload_mode: CPU offload topology used for the peak calculation: "model", "sequential",
            or None when no offload is applied. None is also used for "Automatic" strategies,
            which show the no-offload upper bound.
        components: Per-component estimates for everything resident in the pipeline. When LoRA adapters are
            present, their weight bytes are already added into the primary denoiser's entry, so this list is the
            source for totals.
        estimated_peak_bytes: Estimated peak VRAM in bytes, combining weight topology and activations.
        warnings: Human-readable notes about estimates that are approximate or could not be computed.
        lora_adapters: Itemized unfused LoRA adapters (name, path, weight bytes) for display and reporting. Their
            bytes are already counted in components, so do not add them to the total. Empty when none are applied.
    """

    pipeline_name: str
    offload_mode: str | None
    components: list[ComponentMemoryEstimate]
    estimated_peak_bytes: int
    warnings: list[str] = field(default_factory=list)
    lora_adapters: list[LoraAdapterMemoryEstimate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_name": self.pipeline_name,
            "offload_mode": self.offload_mode,
            "components": [component.to_dict() for component in self.components],
            "estimated_peak_gb": _bytes_to_gb(self.estimated_peak_bytes),
            "warnings": list(self.warnings),
            "lora_adapters": [adapter.to_dict() for adapter in self.lora_adapters],
        }
