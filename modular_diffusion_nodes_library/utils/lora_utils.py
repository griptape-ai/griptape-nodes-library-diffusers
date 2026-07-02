import hashlib
import logging
from pathlib import Path
from typing import Any

import safetensors  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec, normalize_loras

logger = logging.getLogger("modular_diffusers_nodes_library")


class LorasParameter:
    def __init__(self, node: BaseNode):
        self._node = node
        self._loras_parameter_name = "loras"

    def add_input_parameters(self) -> None:
        loras_param = ParameterList(
            name="loras",
            input_types=["loras", "dict"],
            default_value=[],
            type="loras",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            tooltip=(
                "LoRAs to fuse permanently into the pipeline weights. Fused LoRAs become part of "
                "the model's cached identity \u2014 changing them rebuilds the pipeline. Use this for "
                "production-stable style/character LoRAs. For dynamic adapters (IC-LoRA, "
                "distillation, slider, swap-per-generation), use the LoRA Pipeline node instead, "
                "which activates adapters without modifying the cache."
            ),
        )
        loras_param.set_badge(
            variant="help",
            title="Fused vs. activation LoRAs",
            message=(
                "***Fused (this input)*** — LoRA weights are merged into the pipeline on load and become "
                "part of its cache identity. This means:\n\n"
                "- Changing or removing a LoRA evicts the pipeline from cache and triggers a full rebuild.\n"
                "- Best for a fixed, production-stable style or character LoRA that never changes between runs.\n\n"
                "***Activation (LoRA Pipeline node)*** — adapters are applied dynamically around each generation "
                "call and released afterward. The base pipeline is never modified, so:\n\n"
                "- Swapping LoRAs between runs does ***not*** cause a rebuild.\n"
                "- Multiple branches can share the same cached pipeline (e.g. one with LoRAs, one without).\n\n"
                "Prefer activation for IC-LoRAs, distillation/acceleration LoRAs, slider LoRAs, "
                "or any workflow that swaps adapters between runs."
            ),
        )
        self._node.add_parameter(loras_param)

    def to_adapter_name(self, model_path: str) -> str:
        """Returns a unique name for an adapter given its model path."""
        # Use resolve() here (not absolute()) so that symlinks and their targets
        # hash to the same adapter name, preventing the same file from being
        # loaded twice when referenced via different paths.
        resolved = str(Path(model_path).resolve())
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()

    def get_lora_specs(self) -> dict[str, LoraSpec]:
        loras_list = self._node.get_parameter_value(self._loras_parameter_name) or []
        return normalize_loras(loras_list)

    def get_loras(self) -> dict[str, float]:
        return {path: spec.weight for path, spec in self.get_lora_specs().items()}

    def configure_loras(self, pipe: Any) -> None:
        loras = self.get_loras()

        if not loras:
            return

        lora_by_name = {
            self.to_adapter_name(k): {"name": self.to_adapter_name(k), "path": k, "weight": float(v)}
            for k, v in loras.items()
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
