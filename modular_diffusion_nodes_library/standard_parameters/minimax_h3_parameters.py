from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.component_loading.modular_pipeline_loader import (
    ComponentLoadContext,
    ModularPipelineLoadOptions,
)
from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.parameters.diffusers_modular_pipeline_parameters import (
    DiffusersModularPipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

logger = logging.getLogger("modular_diffusers_nodes_library")

# The transformer is 61.7 GB in bfloat16 and the Qwen3-VL conditioner another 62.1 GB, so nothing
# fits alongside the accelerator's own working set. The margin is the model card's recommendation
# for a single 80 GB card.
AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN = "12GB"


class MiniMaxH3PipelineParameters(DiffusersModularPipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.minimax_h3.modular_pipeline:MiniMaxH3ModularPipeline"
    _component_slots: ClassVar[list[str]] = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "audio_vae",
        "audio_scheduler",
    ]
    supported_workflows: ClassVar[tuple[str, ...]] = ("t2va", "fl2va")
    text_conditioning_target_dim_key = "text_dim"
    audio_latent_target_dim_key = "audio_in_channels"

    @classmethod
    def text_conditioning_width(cls, resolve_config) -> int | None:
        return cls._extract_text_conditioning_width(resolve_config("text_encoder"))

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """MiniMax-H3 still needs its base repository when components are overridden."""
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = ConnectableHuggingFaceRepoParameter(
            node,
            repo_ids=["MiniMaxAI/MiniMax-H3"],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        return self.build_data(repo_id, revision)

    @classmethod
    def build_data(cls, repo_id: str, revision: str) -> dict[str, Any]:
        return {"base_repo_id": repo_id, "revision": revision}

    @classmethod
    def get_modular_load_options(cls, build_data: dict[str, Any]) -> ModularPipelineLoadOptions:
        import torch  # type: ignore[reportMissingImports]

        def load_processor(context: ComponentLoadContext) -> Any:
            if not isinstance(context.source, str):
                raise ValueError(
                    f"Attempted to load the MiniMax processor. Failed with invalid source {context.source!r}."
                )
            return cls._load_processor(context.source, context.revision)

        return ModularPipelineLoadOptions(
            workflows=cls.supported_workflows,
            dtype=torch.bfloat16,
            component_loaders={"processor": load_processor},
            device=get_best_device(),
            memory_reserve_margin=AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN,
        )

    @staticmethod
    def _load_processor(source: str, revision: str | None) -> Any:
        """Resolve the processor's local folder so Transformers can load it offline."""
        from huggingface_hub import try_to_load_from_cache
        from transformers import Qwen3VLProcessor

        local_source = Path(source) / "processor"
        if local_source.is_dir():
            folder = local_source
        else:
            cached = try_to_load_from_cache(source, "processor/tokenizer_config.json", revision=revision)
            if not isinstance(cached, str):
                raise FileNotFoundError(
                    f"Attempted to load the MiniMax-H3 processor from '{source}' at revision '{revision}'. "
                    "Failed because processor/tokenizer_config.json is not in the local Hugging Face cache."
                )
            folder = Path(cached).parent
        return Qwen3VLProcessor.from_pretrained(folder, local_files_only=True)
