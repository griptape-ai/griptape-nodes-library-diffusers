"""Memory metadata for unfused runtime LoRA adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import safetensors  # type: ignore[reportMissingImports]
from griptape_nodes.files.path_utils import canonicalize_for_identity, canonicalize_for_io

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep

_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
}


@dataclass(frozen=True)
class LoraAdapterMemoryEstimate:
    """Header-derived memory information for one distinct runtime adapter file."""

    adapter_name: str
    path: str
    strength: float
    weight_bytes: int
    is_estimated: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "path": self.path,
            "strength": self.strength,
            "weight_bytes": self.weight_bytes,
            "is_estimated": self.is_estimated,
            "warning": self.warning,
        }


def _read_safetensors_weight_bytes(path: str) -> tuple[int, str | None]:
    try:
        io_path = canonicalize_for_io(path)
        with safetensors.safe_open(io_path, framework="pt") as adapter_file:  # type: ignore[reportAttributeAccessIssue]
            weight_bytes = 0
            for tensor_name in adapter_file.keys():
                tensor_slice = adapter_file.get_slice(tensor_name)
                dtype = tensor_slice.get_dtype()
                bytes_per_element = _SAFETENSORS_DTYPE_BYTES.get(dtype)
                if bytes_per_element is None:
                    return 0, f"Unsupported safetensors dtype '{dtype}' in LoRA file '{path}'."
                weight_bytes += math.prod(tensor_slice.get_shape()) * bytes_per_element
    except (OSError, ValueError, KeyError, TypeError, safetensors.SafetensorError) as error:
        return 0, f"Attempted to read LoRA weights from '{path}'. Failed because {error}."
    return weight_bytes, None


def _adapter_name(path: str, index: int, used_names: set[str]) -> str:
    stem = Path(path).stem or "adapter"
    candidate = stem
    if candidate in used_names:
        candidate = f"{stem}_{index}"
    used_names.add(candidate)
    return candidate


def collect_runtime_lora_memory(artifact: DiffusionPipelineArtifact) -> list[LoraAdapterMemoryEstimate]:
    """Collect distinct unfused runtime LoRAs attached to an artifact."""
    estimates: list[LoraAdapterMemoryEstimate] = []
    seen_paths: set[str] = set()
    used_names: set[str] = set()

    for step in artifact.runtime_adapter_steps():
        if not isinstance(step, LoraPipelineRuntimeAdapterStep):
            continue
        for raw_path, raw_lora in step.metadata.get("loras", {}).items():
            if not isinstance(raw_lora, dict):
                continue
            path = str(raw_lora.get("path") or raw_path)
            try:
                identity_path = canonicalize_for_identity(path)
            except (OSError, ValueError):
                identity_path = str(Path(path).expanduser().absolute())
            if identity_path in seen_paths:
                continue
            seen_paths.add(identity_path)

            try:
                strength = float(raw_lora.get("weight", 1.0))
            except (TypeError, ValueError):
                strength = 1.0
            weight_bytes, warning = _read_safetensors_weight_bytes(path)
            estimates.append(
                LoraAdapterMemoryEstimate(
                    adapter_name=_adapter_name(path, len(estimates), used_names),
                    path=path,
                    strength=strength,
                    weight_bytes=weight_bytes,
                    is_estimated=warning is not None,
                    warning=warning,
                )
            )

    return estimates
