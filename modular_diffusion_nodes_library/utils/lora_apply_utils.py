import hashlib
import logging
from pathlib import Path
from typing import Any

import safetensors  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")


def _to_adapter_name(model_path: str) -> str:
    """Returns a unique name for an adapter given its model path."""
    # Use resolve() here (not absolute()) so that symlinks and their targets
    # hash to the same adapter name, preventing the same file from being
    # loaded twice when referenced via different paths.
    resolved = str(Path(model_path).resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()


def configure_loras_on_pipeline(pipe: Any, loras: dict[str, float]) -> None:
    if not loras:
        return

    lora_by_name = {
        _to_adapter_name(k): {"name": _to_adapter_name(k), "path": k, "weight": float(v)} for k, v in loras.items()
    }

    loras_to_load = dict(lora_by_name)
    existing_adapter_names = {name for names in pipe.get_list_adapters().values() for name in names}
    for name in existing_adapter_names:
        if name in loras_to_load:
            # Don't reload existing loras.
            loras_to_load.pop(name)

    # Load the loras.
    for item in loras_to_load.values():
        lora_path = item["path"]
        msg = f"Loading lora weights: {lora_path}"
        logger.info(msg)
        state_dict = safetensors.torch.load_file(lora_path)  # type: ignore[reportAttributeAccessIssue]
        pipe.load_lora_weights(state_dict, adapter_name=item["name"])

    # Use them with given weights.
    adapter_names = [v["name"] for v in lora_by_name.values()]
    adapter_weights = [v["weight"] for v in lora_by_name.values()]
    msg = f"Using adapter_names with weights:\n{adapter_names=}\n{adapter_weights=}"
    logger.info(msg)
    pipe.set_adapters(adapter_names=adapter_names, adapter_weights=adapter_weights)

    logger.info("Fusing lora weights with diffusion model.")
    pipe.fuse_lora(adapter_names=adapter_names, lora_scale=1.0)
    pipe.unload_lora_weights()
