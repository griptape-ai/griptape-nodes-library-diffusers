import hashlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import safetensors  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec
from modular_diffusion_nodes_library.utils.pipeline_runtime_adapter_step import PipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


def _to_adapter_name(model_path: str) -> str:
    """Returns a unique name for an adapter given its model path.

    Combines the file stem (for human-readable token matching) with an 8-character path hash.
    Uses resolve() so symlinks and their targets produce the same name.
    """
    resolved = str(Path(model_path).resolve())
    stem = Path(model_path).stem.replace(".", "-")
    short_hash = hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{short_hash}"


def _load_missing_lora_adapters(pipe: Any, lora_by_name: dict[str, dict[str, Any]]) -> None:
    loras_to_load = dict(lora_by_name)
    existing_adapter_names = {name for names in pipe.get_list_adapters().values() for name in names}
    for name in existing_adapter_names:
        if name in loras_to_load:
            # Don't reload existing loras.
            loras_to_load.pop(name)

    for item in loras_to_load.values():
        lora_path = item["path"]
        msg = f"Loading lora weights: {lora_path}"
        logger.info(msg)
        state_dict = safetensors.torch.load_file(lora_path)  # type: ignore[reportAttributeAccessIssue]
        pipe.load_lora_weights(state_dict, adapter_name=item["name"])
        _stash_lora_metadata(pipe, adapter_name=item["name"], lora_path=lora_path)


def _stash_lora_metadata(pipe: Any, *, adapter_name: str, lora_path: str) -> None:
    """Capture safetensors header metadata onto the pipe so downstream code can read it."""
    try:
        with safetensors.safe_open(lora_path, framework="pt") as f:  # type: ignore[reportAttributeAccessIssue]
            metadata = dict(f.metadata() or {})
    except (OSError, ValueError) as err:
        logger.warning("Failed to read safetensors metadata from %s: %s", lora_path, err)
        metadata = {}

    if not hasattr(pipe, "_gtn_lora_metadata"):
        pipe._gtn_lora_metadata = {}
    pipe._gtn_lora_metadata[adapter_name] = {"path": lora_path, "metadata": metadata}


def configure_loras_on_pipeline(pipe: Any, loras: dict[str, float], *, fuse_loras: bool = True) -> None:
    if not loras:
        return

    lora_by_name: dict[str, dict[str, Any]] = {}
    adapter_names: list[str] = []
    adapter_weights: list[float] = []

    for lora_path, raw_weight in loras.items():
        adapter_name = _to_adapter_name(lora_path)
        weight = float(raw_weight)
        lora_by_name[adapter_name] = {
            "name": adapter_name,
            "path": lora_path,
            "weight": weight,
        }
        adapter_names.append(adapter_name)
        adapter_weights.append(weight)

    _load_missing_lora_adapters(pipe, lora_by_name)

    msg = f"Using adapter_names with weights:\n{adapter_names=}\n{adapter_weights=}"
    logger.info(msg)
    pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)

    if not fuse_loras:
        logger.info("Configured LoRA adapters without fusing into base model.")
        return

    logger.info("Fusing lora weights with diffusion model.")
    pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    pipe.unload_lora_weights()


def release_lora_adapters(pipe: Any, adapter_names: list[str]) -> None:
    """Deactivate + delete adapters and drop stashed metadata to release VRAM."""
    if not adapter_names:
        return

    try:
        pipe.delete_adapters(adapter_names)
    except Exception as err:  # noqa: BLE001 — delete_adapters can raise model-specific errors
        logger.warning(
            "Attempted to delete LoRA adapters. Failed with adapter_names=%s because of %r.",
            adapter_names,
            err,
        )

    if hasattr(pipe, "_gtn_lora_metadata"):
        for name in adapter_names:
            pipe._gtn_lora_metadata.pop(name, None)


class LoraPipelineRuntimeAdapterStep(PipelineRuntimeAdapterStep):
    """Activate a set of non-fused LoRA adapters for the duration of a generation.

    Loads any missing adapters onto the pipeline, calls `set_adapters` with the
    requested weights, yields the pipe, then on exit deactivates the adapters,
    deletes them from the transformer to free VRAM, and drops stashed metadata.
    """

    KIND: ClassVar[str] = "lora_pipeline_runtime_adapter"

    def __init__(self, loras: dict[str, LoraSpec]) -> None:
        self._loras = dict(loras)

    def _metadata(self) -> dict[str, Any]:
        return {
            "loras": {key: {"path": spec.path, "weight": float(spec.weight)} for key, spec in self._loras.items()},
        }

    @contextmanager
    def activate(self, pipe: Any, *, node_name: str | None = None) -> Iterator[Any]:
        if not self._loras:
            yield pipe
            return

        loaded_adapter_names: list[str] = []
        try:
            adapter_names, adapter_weights = self._build_adapter_lists()
            lora_by_name = self._build_load_payload()
            loaded_adapter_names = adapter_names
            cleanup_memory_caches()
            _load_missing_lora_adapters(pipe, lora_by_name)
            pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)
            yield pipe
        except Exception as err:
            release_lora_adapters(pipe, loaded_adapter_names)
            cleanup_memory_caches()
            prefix = f"{node_name}: " if node_name else ""
            raise RuntimeError(
                f"{prefix}Attempted to activate LoRAs. "
                f"Failed with adapter_names={loaded_adapter_names} because of {err!r}."
            ) from err
        else:
            release_lora_adapters(pipe, loaded_adapter_names)
            cleanup_memory_caches()

    def _build_adapter_lists(self) -> tuple[list[str], list[float]]:
        names: list[str] = []
        weights: list[float] = []
        for spec in self._loras.values():
            names.append(_to_adapter_name(spec.path))
            weights.append(float(spec.weight))
        return names, weights

    def _build_load_payload(self) -> dict[str, dict[str, Any]]:
        payload: dict[str, dict[str, Any]] = {}
        for spec in self._loras.values():
            name = _to_adapter_name(spec.path)
            payload[name] = {"name": name, "path": spec.path, "weight": float(spec.weight)}
        return payload
