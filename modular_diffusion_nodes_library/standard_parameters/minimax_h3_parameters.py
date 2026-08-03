import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers import ComponentsManager, ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # type: ignore[reportMissingImports]
    MiniMaxH3ModularPipeline,
)
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

logger = logging.getLogger("modular_diffusers_nodes_library")

# The transformer is 61.7 GB in bfloat16 and the Qwen3-VL conditioner another 62.1 GB, so nothing
# fits alongside the accelerator's own working set. The margin is the model card's recommendation
# for a single 80 GB card.
AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN = "12GB"


class MiniMaxH3PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=["MiniMaxAI/MiniMax-H3"],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    @property
    def pipeline_class(self) -> type:
        return MiniMaxH3ModularPipeline

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        return {
            "repo_id": repo_id,
            "revision": revision,
        }

    def requires_device_map(self) -> bool:
        # Used here as an opt-out from post-hoc pipeline optimization, not literally to request an
        # accelerate device_map. MiniMax-H3 is a ModularPipeline, on which
        # `optimize_diffusion_pipeline` would call `.to(device)` (a guaranteed OOM at ~124 GB) while
        # both `enable_*_cpu_offload` calls are hasattr-gated and silently skipped. Placement is
        # instead owned by `build_pipeline_from_build_data` via the ComponentsManager below.
        return True

    def is_prequantized(self) -> bool:
        # Suppresses quantization and layerwise casting, which fire before the requires_device_map
        # short-circuit. Neither is safe to apply on top of the ComponentsManager offload hooks.
        return True

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> ModularPipeline:
        # `from_pretrained` resolves the component specs but loads no weights; `load_components`
        # fetches them. Only the `t2va` / `fl2va` half is touched, never `transformer_ref/`.
        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            build_data["repo_id"],
            revision=build_data["revision"],
            components_manager=manager,
        )
        pipe.load_components(dtype=torch.bfloat16)
        manager.enable_auto_cpu_offload(
            device=get_best_device(),
            memory_reserve_margin=AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN,
        )
        return pipe
