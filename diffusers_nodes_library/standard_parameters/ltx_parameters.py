import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers import LTXPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from diffusers_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


LTX_REPO_IDS = [
    "Lightricks/LTX-Video-0.9.8-13B-distilled",
    "Lightricks/LTX-Video-0.9.7-distilled",
    "Lightricks/LTX-Video-0.9.7-dev",
    "Lightricks/LTX-Video",
]


class LTXPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=LTX_REPO_IDS,
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
        return LTXPipeline

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._model_repo_parameter.get_repo_revision()

        return {
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
        }

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> DiffusionPipeline | Any | None:
        pipeline = LTXPipeline.from_pretrained(
            build_data["base_repo_id"],
            revision=build_data["base_revision"],
            torch_dtype=torch.bfloat16,
        )
        pipeline.vae.use_framewise_decoding = True
        return pipeline
