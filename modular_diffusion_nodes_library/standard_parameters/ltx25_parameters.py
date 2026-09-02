import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers.models.autoencoders.ltx2_diffusion_decoder import (  # type: ignore[reportMissingImports]
    LTX2VideoDiffusionDecoderModel,
)
from diffusers.models.transformers.transformer_ltx2 import (  # type: ignore[reportMissingImports]
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx2.pipeline_ltx2 import LTX2Pipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("diffusers_nodes_library")

_LTX25_REPO_ID = "Lightricks/LTX-2.5-Diffusers"


class _LTX25PipelineParametersBase(ModularDiffusionPipelineTypePipelineParameters):
    """Shared plumbing for the two LTX-2.5 pipeline_type entries.

    Both entries build an `LTX2Pipeline` from the same repo, differing only in which
    transformer subfolder is loaded and whether `is_distilled` runtime behavior is used.
    Decoding always uses the diffusion decoder (`diffusion_decoder` subfolder), attached to
    the built pipe as a plain attribute; see `LTX2PipelineDriver.decode_latent`.
    """

    _pipeline_cls = LTX2Pipeline
    _transformer_subfolder: str
    _is_distilled: bool

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        "We don't have support for overriding vocoder and audio vae, so we don't support building from overrides only."
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):  # noqa: ARG002
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[_LTX25_REPO_ID],
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

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)

        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._resolve_repo(self._model_repo_parameter)

        return {
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
            "is_distilled": self._is_distilled,
            "transformer_subfolder": self._transformer_subfolder,
        }

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> LTX2Pipeline:
        base_repo_id = build_data["base_repo_id"]
        base_revision = build_data["base_revision"]

        overrides.setdefault(
            "transformer",
            LTX2VideoTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path=base_repo_id,
                subfolder=build_data["transformer_subfolder"],
                revision=base_revision,
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            ),
        )

        pipe = LTX2Pipeline.from_pretrained(
            pretrained_model_name_or_path=base_repo_id,
            revision=base_revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **overrides,
        )
        diffusion_decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
            pretrained_model_name_or_path=base_repo_id,
            subfolder="diffusion_decoder",
            revision=base_revision,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        # register_modules (not a plain attribute) so pipe.to(device), CPU offload, and
        # quantization treat diffusion_decoder like any other pipeline component.
        pipe.register_modules(diffusion_decoder=diffusion_decoder)
        return pipe


class LTX25DistilledPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer"
    _is_distilled = True


class LTX25FullPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer_full"
    _is_distilled = False
