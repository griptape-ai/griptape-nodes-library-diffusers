import logging
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class HunyuanVideo15PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls = diffusers.HunyuanVideo15Pipeline  # type: ignore[reportAttributeAccessIssue]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
                "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
            ],
            parameter_name="model",
            list_all_models=list_all_models,
        )

    @classmethod
    def _build_pipeline_from_overrides_only(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.HunyuanVideo15Pipeline:  # type: ignore[reportAttributeAccessIssue]
        # HunyuanVideo15Pipeline requires a `guider` component that is not in
        # ALLOWED_COMPONENT_SLOTS. Instantiate one with the T2V checkpoint's
        # default settings so the pipeline can still be built from component
        # overrides alone.
        guider = diffusers.ClassifierFreeGuidance(  # type: ignore[reportAttributeAccessIssue]
            guidance_scale=6.0,
            guidance_rescale=0.0,
            use_original_formulation=False,
            start=0.0,
            stop=1.0,
            enabled=True,
        )
        return diffusers.HunyuanVideo15Pipeline(  # type: ignore[reportAttributeAccessIssue]
            **overrides,
            guider=guider,
        )

    @classmethod
    def get_auto_supplied_components(cls) -> set[str]:
        # We construct `guider` ourselves in _build_pipeline_from_overrides_only,
        return {"guider"}

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
            "repo_id": repo_id,
            "revision": revision,
        }

    def requires_device_map(self) -> bool:
        # HunyuanVideo 1.5 includes a Qwen2.5-VL-7B text encoder plus the video
        # transformer — total weights exceed 20 GB in bfloat16. Using device_map
        # lets accelerate stream each layer directly to the right device during
        # loading so we never materialise the full model on CPU RAM.
        return True

    @classmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.HunyuanVideo15Pipeline:  # type: ignore[reportAttributeAccessIssue]
        return diffusers.HunyuanVideo15Pipeline.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["repo_id"],
            revision=build_data["revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="balanced",
            **overrides,
        )
