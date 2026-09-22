"""Persist and reload post-hoc optimum.quanto packed weights.

Save uses the standard Diffusers/HF slot folder (`save_pretrained` safetensors
with `weight._data` / `weight._scale`). Load wraps the rebuilt module with
`optimum.quanto.requantize` so those keys are consumed. This is not a second
Absmax pass.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger("modular_diffusers_nodes_library")

QUANT_MAP_FILENAME = "quantization_map.json"


def _state_has_packed_weights(state: dict[str, Any]) -> bool:
    return any(key.endswith("weight._data") for key in state)


def _module_has_packed_weights(component: Any) -> bool:
    if not hasattr(component, "state_dict"):
        return False
    return _state_has_packed_weights(component.state_dict())


def component_quantization_map(component: Any) -> dict[str, dict[str, str]]:
    try:
        from optimum.quanto import quantization_map
    except ImportError:
        return {}
    if not isinstance(component, nn.Module):
        return {}
    return quantization_map(component)


def write_quantization_map(folder: Path, qmap: dict[str, dict[str, str]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / QUANT_MAP_FILENAME).write_text(json.dumps(qmap, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_quantization_map(folder: Path) -> dict[str, dict[str, str]]:
    path = folder / QUANT_MAP_FILENAME
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        msg = f"Attempted to read {path}. Failed because the file is not a JSON object."
        raise ValueError(msg)
    return data


def write_packed_quant_maps(pipe: Any, folder: Path) -> list[str]:
    """Write a map under each component slot that actually has QModules."""
    folder = Path(folder)
    written: list[str] = []
    for name, component in _iter_components(pipe):
        qmap = component_quantization_map(component)
        if not qmap or not _module_has_packed_weights(component):
            continue
        write_quantization_map(folder / name, qmap)
        written.append(name)
    return sorted(written)


def _iter_components(pipe: Any) -> list[tuple[str, Any]]:
    components = getattr(pipe, "components", None)
    if isinstance(components, dict) and components:
        return [(name, component) for name, component in components.items() if component is not None]
    return []


def load_slot_state_dict(folder: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    folder = Path(folder)
    files = sorted(path for path in folder.glob("*.safetensors") if path.is_file())
    if not files:
        msg = f"Attempted to load packed weights from {folder}. Failed because no .safetensors files were found."
        raise FileNotFoundError(msg)
    state: dict[str, torch.Tensor] = {}
    for path in files:
        state.update(load_file(path))
    return state


def _align_state_dict_to_qmap(
    state: dict[str, torch.Tensor],
    qmap: dict[str, dict[str, str]],
) -> dict[str, torch.Tensor]:
    """Map checkpoint key prefixes onto in-memory module names.

    Transformers `save_pretrained` still writes CLIP-L keys under `text_model.*`
    even when the loaded module names are `encoder.*`. quanto `requantize` then
    does `load_state_dict(..., strict=False)` and silently drops the packed
    tensors, leaving QLinear layers on randomly initialized floats.
    """
    if not qmap or not state:
        return state
    target_keys = {f"{name}.weight._data" for name in qmap}
    return align_state_dict_to_module_keys(state, target_keys)


def align_state_dict_to_module_keys(
    state: dict[str, torch.Tensor],
    target_keys: set[str],
) -> dict[str, torch.Tensor]:
    """Strip or keep a uniform checkpoint prefix so keys match an in-memory module."""
    if not state or not target_keys:
        return state
    if target_keys <= set(state):
        return state
    sample = next(iter(target_keys))
    prefixes = {key[: -len(sample)] for key in state if key.endswith(sample) and len(key) > len(sample)}
    if len(prefixes) == 1:
        prefix = prefixes.pop()
        aligned = {(key[len(prefix) :] if key.startswith(prefix) else key): value for key, value in state.items()}
        if target_keys <= set(aligned):
            logger.info("Stripping checkpoint key prefix %r to match module names", prefix)
            return aligned
    msg = (
        "Attempted to align a GGUF state dict to module keys. Failed because the "
        f"checkpoint keys do not match the module (sample module key {sample!r})."
    )
    raise ValueError(msg)


def _assert_qmodules_frozen(component: nn.Module, qmap: dict[str, dict[str, str]], folder: Path) -> None:
    from optimum.quanto import QModuleMixin

    frozen = sum(
        1 for module in component.modules() if isinstance(module, QModuleMixin) and hasattr(module.weight, "_data")
    )
    if frozen != len(qmap):
        msg = (
            f"Attempted to requantize {type(component).__name__} from {folder}. "
            f"Failed because {frozen} frozen QModules were restored but the map lists {len(qmap)}."
        )
        raise RuntimeError(msg)


def _cast_unpacked_floats(model: nn.Module, dtype: torch.dtype) -> None:
    for module in model.modules():
        for name, param in list(module.named_parameters(recurse=False)):
            if hasattr(param, "_data"):
                continue
            if param.dtype.is_floating_point and param.dtype != dtype:
                setattr(module, name, nn.Parameter(param.data.to(dtype)))
        for name, buf in list(module.named_buffers(recurse=False)):
            if buf.dtype.is_floating_point and buf.dtype != dtype:
                module.register_buffer(name, buf.to(dtype), persistent=True)


def requantize_component(
    component: nn.Module,
    folder: Path,
    *,
    unpacked_dtype: torch.dtype | None = None,
) -> None:
    from optimum.quanto import requantize

    folder = Path(folder)
    qmap = read_quantization_map(folder)
    if not qmap:
        msg = f"Attempted to requantize {type(component).__name__} from {folder}. Failed because {QUANT_MAP_FILENAME} is missing."
        raise FileNotFoundError(msg)
    state = _align_state_dict_to_qmap(load_slot_state_dict(folder), qmap)
    # Not a second quantisation pass: requantize restores QModules from saved weight._data + map.
    requantize(component, state, qmap, device=torch.device("cpu"))
    _assert_qmodules_frozen(component, qmap, folder)
    if unpacked_dtype is not None:
        _cast_unpacked_floats(component, unpacked_dtype)


def apply_packed_quant_from_folder(
    pipe: Any,
    folder: Path,
    *,
    unpacked_dtype: torch.dtype | None = None,
) -> list[str]:
    """Requantize every slot that has a map. Mutates `pipe` components in place."""
    folder = Path(folder)
    applied: list[str] = []
    for name, component in _iter_components(pipe):
        if not isinstance(component, nn.Module):
            continue
        slot_dir = folder / name
        if not (slot_dir / QUANT_MAP_FILENAME).is_file():
            continue
        try:
            state = load_slot_state_dict(slot_dir)
        except FileNotFoundError:
            continue
        if not _state_has_packed_weights(state):
            logger.warning(
                "Skipping requantize for %s; %s has a map but no packed weight._data keys.",
                name,
                slot_dir,
            )
            continue
        logger.info("Reloading packed quanto weights for %s from %s", name, slot_dir)
        requantize_component(component, slot_dir, unpacked_dtype=unpacked_dtype)
        applied.append(name)
    return sorted(applied)
