# Copied from diffusers_nodes_library/common/parameters/diffusion/qwen/edit_parameters.py
import copy
import logging
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
import transformers  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from diffusers_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from diffusers_nodes_library.parameters.scheduler_parameters import SchedulerParameters

logger = logging.getLogger("modular_diffusers_nodes_library")


class QwenEditPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Qwen/Qwen-Image-Edit",
                "Qwen/Qwen-Image-Edit-2509",
                "Qwen/Qwen-Image-Edit-2511",
            ],
            parameter_name="model",
            list_all_models=list_all_models,
        )

        self._text_encoder_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Qwen/Qwen2.5-VL-7B-Instruct",
            ],
            parameter_name="text_encoder",
            list_all_models=list_all_models,
        )

        self._scheduler_parameters = SchedulerParameters(
            node,
            scheduler_types=[diffusers.FlowMatchEulerDiscreteScheduler],  # type: ignore[reportAttributeAccessIssue]
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._text_encoder_repo_parameter.add_input_parameters()
        self._scheduler_parameters.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()
        self._text_encoder_repo_parameter.remove_input_parameters()
        self._scheduler_parameters.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
            "text_encoder": self._node.get_parameter_value("text_encoder"),
            **self._scheduler_parameters.get_config_kwargs(),
        }

    @property
    def pipeline_class(self) -> type:
        return diffusers.QwenImageEditPipeline  # type: ignore[reportAttributeAccessIssue]

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        text_encoder_errors = self._text_encoder_repo_parameter.validate_before_node_run()
        if text_encoder_errors:
            errors.extend(text_encoder_errors)

        scheduler_errors = self._scheduler_parameters.validate_before_node_run()
        if scheduler_errors:
            errors.extend(scheduler_errors)

        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._model_repo_parameter.get_repo_revision()
        text_encoder_repo_id, text_encoder_revision = self._text_encoder_repo_parameter.get_repo_revision()

        scheduler_type = self._scheduler_parameters.get_scheduler_class().__name__
        if not hasattr(diffusers, scheduler_type):
            msg = f"Unknown scheduler type '{scheduler_type}'; not found in diffusers module."
            raise ValueError(msg)

        return {
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
            "text_encoder_repo_id": text_encoder_repo_id,
            "text_encoder_revision": text_encoder_revision,
            "scheduler_type": scheduler_type,
            "scheduler_config": copy.deepcopy(self._scheduler_parameters.get_scheduler_config()),
        }

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> diffusers.QwenImageEditPipeline:  # type: ignore[reportAttributeAccessIssue]
        scheduler_class = getattr(diffusers, build_data["scheduler_type"])

        text_encoder = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=build_data["text_encoder_repo_id"],
            revision=build_data["text_encoder_revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )

        return diffusers.QwenImageEditPipeline.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["base_repo_id"],
            revision=build_data["base_revision"],
            text_encoder=text_encoder,
            scheduler=scheduler_class.from_config(build_data["scheduler_config"]),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
