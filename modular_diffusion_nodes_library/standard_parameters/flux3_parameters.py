from __future__ import annotations

import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.component_loading.modular_pipeline_loader import (
    ComponentLoadContext,
    ModularPipelineLoadOptions,
)
from modular_diffusion_nodes_library.misc.bfl_fp8 import require_fp8r_support
from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
    ConnectableHuggingFaceRepoVariantParameter,
)
from modular_diffusion_nodes_library.parameters.diffusers_modular_pipeline_parameters import (
    DiffusersModularPipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

logger = logging.getLogger("modular_diffusers_nodes_library")
PROMPT_UPSAMPLER_NONE = "Disabled"
PROMPT_UPSAMPLER_REPO_ID = "black-forest-labs/flux-3-image-klein-pup"
PROMPT_UPSAMPLER_COMPONENTS = ("prompt_upsampler", "prompt_upsampler_processor")
# Quantized transformer weights live beside the root bf16 weights, one folder per variant.
VARIANTS_ROOT = "variants"
TRANSFORMER_VARIANT_FOLDERS = {"fp8r": f"{VARIANTS_ROOT}/fp8r", "nvfp4": f"{VARIANTS_ROOT}/nvfp4"}
TRANSFORMER_WEIGHTS_FILENAME = "model.safetensors"


class Flux3ModelRepoParameter(ConnectableHuggingFaceRepoVariantParameter):
    def get_repo_revision(self) -> tuple[str, str]:
        if not self._has_transformer_override():
            return super().get_repo_revision()
        return self.resolve_base_repo_revision()

    def _has_transformer_override(self) -> bool:
        if self._node.get_parameter_value("component_transformer") is not None:
            return True
        return self._node._param_has_incoming_connection("component_transformer")


class OptionalPromptUpsamplerRepoParameter(ConnectableHuggingFaceRepoParameter):
    def get_choices(self) -> list[str]:
        return [*super().get_choices(), PROMPT_UPSAMPLER_NONE]

    def repo_id_for_choice(self, choice: str) -> str | None:
        if choice == PROMPT_UPSAMPLER_NONE:
            return None
        return super().repo_id_for_choice(choice)

    def validate_before_node_run(self) -> list[Exception] | None:
        if self._node.get_parameter_value(self._parameter_name) == PROMPT_UPSAMPLER_NONE:
            return None
        return super().validate_before_node_run()


class _Flux3ModularPipelineParameters(DiffusersModularPipelineParameters):
    _component_slots: ClassVar[list[str]] = ["transformer", "vae", "text_encoder", "tokenizer"]
    supported_workflows: ClassVar[tuple[str, ...]] = ("text2image", "image_conditioned")
    latent_packing_ratio = 4
    text_conditioning_target_dim_key = "joint_attention_dim"
    text_encoder_hidden_state_indices: ClassVar[tuple[int, ...]] = (4, 8, 12, 16, 20, 24, 28, 32)
    _repo_id: ClassVar[str]

    @classmethod
    def text_conditioning_width(cls, resolve_config) -> int | None:
        hidden = cls._extract_text_conditioning_width(resolve_config("text_encoder"))
        if hidden is None:
            return None
        return hidden * len(cls.text_encoder_hidden_state_indices)

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = Flux3ModelRepoParameter(
            node,
            repo_id=self._repo_id,
            variant_folders=TRANSFORMER_VARIANT_FOLDERS,
            parameter_name="model",
            weights_filename=TRANSFORMER_WEIGHTS_FILENAME,
            list_all_models=list_all_models,
        )
        self._prompt_upsampler_repo_parameter = OptionalPromptUpsamplerRepoParameter(
            node,
            repo_ids=[PROMPT_UPSAMPLER_REPO_ID],
            parameter_name="prompt_upsampler_model",
            list_all_models=list_all_models,
        )

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        return False

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()
        self._prompt_upsampler_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._prompt_upsampler_repo_parameter.remove_input_parameters()
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
            "prompt_upsampler_model": self._node.get_parameter_value("prompt_upsampler_model"),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = self._model_repo_parameter.validate_before_node_run() or []
        upsampler_errors = self._prompt_upsampler_repo_parameter.validate_before_node_run()
        if upsampler_errors:
            errors.extend(upsampler_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        self._resolve_repo(self._model_repo_parameter)
        repo_id, variant, revision = self._model_repo_parameter.get_repo_variant_revision()
        variant_weights_path = None
        if variant:
            variant_weights_path = str(self._model_repo_parameter.variant_weights_path(variant, revision))
        build_data: dict[str, Any] = {
            "base_repo_id": repo_id,
            "variant": variant,
            "variant_weights_path": variant_weights_path,
            "revision": revision,
            "prompt_upsampler_repo_id": None,
            "prompt_upsampler_revision": None,
        }
        if self._node.get_parameter_value("prompt_upsampler_model") != PROMPT_UPSAMPLER_NONE:
            upsampler_repo_id, upsampler_revision = self._resolve_repo(self._prompt_upsampler_repo_parameter)
            build_data["prompt_upsampler_repo_id"] = upsampler_repo_id
            build_data["prompt_upsampler_revision"] = upsampler_revision
        return build_data

    def is_prequantized(self) -> bool:
        variant = self._model_repo_parameter.get_selected_variant()
        if not variant:
            return False
        return not self._model_repo_parameter._has_transformer_override()

    @classmethod
    def get_modular_load_options(cls, build_data: dict[str, Any]) -> ModularPipelineLoadOptions:
        import torch  # type: ignore[reportMissingImports]

        device = get_best_device()
        upsampler_selected = build_data.get("prompt_upsampler_repo_id") is not None
        component_options: dict[str, dict[str, Any]] = {
            "vae": {"dtype": torch.float32},
            "tokenizer": {"fix_mistral_regex": True},
        }
        if upsampler_selected:
            for name in PROMPT_UPSAMPLER_COMPONENTS:
                component_options[name] = {
                    "pretrained_model_name_or_path": build_data["prompt_upsampler_repo_id"],
                    "subfolder": "",
                    "revision": build_data.get("prompt_upsampler_revision"),
                }

        def prepare(pipe: Any) -> None:
            if not upsampler_selected:
                pipe._blocks.sub_blocks.pop("prompt_upsampler", None)
            else:
                missing = sorted(set(PROMPT_UPSAMPLER_COMPONENTS) - set(pipe.components))
                if missing:
                    raise ValueError(
                        f"Attempted to load the {cls.pipeline_name()} prompt upsampler. Failed because "
                        f"the pinned Diffusers component specs do not include {missing}."
                    )

        def validate_variant(_context: ComponentLoadContext) -> None:
            require_fp8r_support(device)

        def load_variant(context: ComponentLoadContext) -> Any:
            from diffusers.models.transformers.transformer_flux3 import Flux3Transformer2DModel
            from diffusers.quantizers.quantization_config import BFLQuantizationConfig

            return Flux3Transformer2DModel.from_single_file(
                build_data["variant_weights_path"],
                config=context.source,
                config_revision=context.revision,
                subfolder=context.subfolder,
                quantization_config=BFLQuantizationConfig(compute_dtype=context.options["dtype"]),
                dtype=context.options["dtype"],
                local_files_only=True,
            )

        component_validators = {}
        if build_data.get("variant") == "fp8r":
            component_validators["transformer"] = validate_variant
        component_loaders = {}
        if build_data.get("variant"):
            component_loaders["transformer"] = load_variant
        return ModularPipelineLoadOptions(
            workflows=cls.supported_workflows,
            dtype=torch.bfloat16,
            component_options=component_options,
            component_validators=component_validators,
            component_loaders=component_loaders,
            prepare_pipeline=prepare,
            device=device,
            memory_reserve_margin="12GB",
        )


class Flux3PipelineParameters(_Flux3ModularPipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.flux3.modular_pipeline:Flux3ModularPipeline"
    _repo_id = "black-forest-labs/flux-3-image-klein-base"


class Flux3DistilledPipelineParameters(_Flux3ModularPipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.flux3.modular_pipeline:Flux3DistilledModularPipeline"
    _repo_id = "black-forest-labs/flux-3-image-klein"
