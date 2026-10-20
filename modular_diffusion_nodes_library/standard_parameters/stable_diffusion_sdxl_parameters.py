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


class StableDiffusionXLPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls = diffusers.StableDiffusionXLImg2ImgPipeline  # type: ignore[reportAttributeAccessIssue]
    text_conditioning_target_dim_key = "cross_attention_dim"

    @classmethod
    def text_conditioning_width(cls, resolve_config) -> int | None:
        text_encoder_width = cls._extract_text_conditioning_width(resolve_config("text_encoder"))
        text_encoder_2_width = cls._extract_text_conditioning_width(resolve_config("text_encoder_2"))
        if text_encoder_width is None or text_encoder_2_width is None:
            return None
        return text_encoder_width + text_encoder_2_width

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._huggingface_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "stabilityai/stable-diffusion-xl-base-1.0",
                "stabilityai/stable-diffusion-xl-refiner-1.0",
                "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            ],
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._huggingface_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._huggingface_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    @property
    def pipeline_name(self) -> str:
        return "StableDiffusionXLPipeline"

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = self._huggingface_repo_parameter.validate_before_node_run()
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._resolve_repo(self._huggingface_repo_parameter)
        return {
            "base_repo_id": repo_id,
            "revision": revision,
        }

    @classmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.StableDiffusionXLImg2ImgPipeline:  # type: ignore[reportAttributeAccessIssue]
        repo_id = build_data["base_repo_id"]
        return cls._pipeline_cls.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=repo_id,
            revision=build_data["revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **overrides,
        )
