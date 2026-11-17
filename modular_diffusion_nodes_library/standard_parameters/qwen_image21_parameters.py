from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

if TYPE_CHECKING:
    import diffusers  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")


class QwenImage21PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls_path = "diffusers:QwenImage21Pipeline"
    _component_slots: ClassVar[list[str]] = ["transformer", "vae", "text_encoder", "scheduler"]
    text_conditioning_target_dim_key = "context_in_dim"
    latent_packing_ratio = 1

    @classmethod
    def text_conditioning_width(cls, resolve_config) -> int | None:
        return cls._extract_text_conditioning_width(resolve_config("text_encoder"))

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """The pipeline requires a processor, which is not an allowed component override slot."""
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = ConnectableHuggingFaceRepoParameter(
            node,
            repo_ids=["Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1-Turbo"],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = self._model_repo_parameter.validate_before_node_run()
        if errors:
            return errors
        return None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._resolve_repo(self._model_repo_parameter)
        return self.build_data(base_repo_id, base_revision)

    @classmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.QwenImage21Pipeline:  # type: ignore[reportAttributeAccessIssue]
        import torch  # type: ignore[reportMissingImports]

        return cls.pipeline_cls().from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["base_repo_id"],
            revision=build_data["base_revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **overrides,
        )
