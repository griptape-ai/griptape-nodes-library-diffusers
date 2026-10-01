from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

if TYPE_CHECKING:
    import diffusers  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")


class WanAnimatePipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls_path = "diffusers:WanAnimatePipeline"
    _component_slots: ClassVar[list[str]] = [
        "transformer",
        "vae",
        "text_encoder",
        "tokenizer",
        "scheduler",
    ]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Wan-AI/Wan2.2-Animate-14B-Diffusers",
            ],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        # WanAnimatePipeline requires image_processor and image_encoder components
        # that are not in ALLOWED_COMPONENT_SLOTS, so it cannot be built from
        # component overrides alone.
        return False

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        return {
            "base_repo_id": repo_id,
            "revision": revision,
        }

    def requires_device_map(self) -> bool:
        # WAN Animate models are 14B parameters (~28 GB in bfloat16). Using device_map
        # lets accelerate place each layer directly on the right device during loading,
        # avoiding the CPU-RAM spike entirely.
        return True

    @classmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.WanAnimatePipeline:  # type: ignore[reportAttributeAccessIssue]
        import diffusers  # type: ignore[reportMissingImports]
        import torch  # type: ignore[reportMissingImports]

        repo_id = build_data["base_repo_id"]
        return diffusers.WanAnimatePipeline.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=repo_id,
            revision=build_data["revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="balanced",
            **overrides,
        )
