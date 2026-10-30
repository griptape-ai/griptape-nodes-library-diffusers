from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, cast

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.misc.bfl_fp8 import require_fp8r_support
from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
    ConnectableHuggingFaceRepoVariantParameter,
)
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

if TYPE_CHECKING:
    from diffusers import ModularPipeline  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")
PROMPT_UPSAMPLER_NONE = "Disabled"
PROMPT_UPSAMPLER_REPO_ID = "black-forest-labs/flux-3-image-klein-pup"
PROMPT_UPSAMPLER_COMPONENTS = ("prompt_upsampler", "prompt_upsampler_processor")
# Quantized transformer weights live beside the root bf16 weights, one folder per variant.
VARIANTS_ROOT = "variants"
TRANSFORMER_VARIANT_FOLDERS = {"fp8r": f"{VARIANTS_ROOT}/fp8r", "nvfp4": f"{VARIANTS_ROOT}/nvfp4"}
TRANSFORMER_WEIGHTS_FILENAME = "model.safetensors"


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


class _Flux3ModularPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _component_slots: ClassVar[list[str]] = []
    _repo_id: ClassVar[str]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = ConnectableHuggingFaceRepoVariantParameter(
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
            "repo_id": repo_id,
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

    def requires_device_map(self) -> bool:
        # Used here as an opt-out from post-hoc pipeline optimization, not literally to request an
        # accelerate device_map. Flux3 is a ModularPipeline, on which
        # `optimize_diffusion_pipeline` would call `.to(device)` (a guaranteed OOM at ~124 GB) while
        # both `enable_*_cpu_offload` calls are hasattr-gated and silently skipped. Placement is
        # instead owned by `build_pipeline_from_build_data` via the ComponentsManager below.
        return True

    def is_prequantized(self) -> bool:
        # Suppresses quantization and layerwise casting, which fire before the requires_device_map
        # short-circuit. Neither is safe to apply on top of the ComponentsManager offload hooks.
        return True

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> ModularPipeline:
        import torch  # type: ignore[reportMissingImports]
        from diffusers import (  # type: ignore[reportMissingImports]
            BFLQuantizationConfig,  # type: ignore[reportMissingImports]
            ComponentsManager,  # type: ignore[reportMissingImports]
            Flux3Transformer2DModel,  # type: ignore[reportMissingImports]
            ModularPipeline,  # type: ignore[reportMissingImports]
        )

        if build_data["variant"] == "fp8r":
            require_fp8r_support(get_best_device())

        manager = ComponentsManager()
        pipe = cast(
            ModularPipeline,
            ModularPipeline.from_pretrained(
                build_data["repo_id"],
                revision=build_data["revision"],
                components_manager=manager,
                local_files_only=True,
            ),
        )
        component_names = cls._components_to_load(pipe, build_data)
        load_kwargs: dict[str, Any] = {
            "dtype": {"vae": torch.float32, "default": torch.bfloat16},
            "local_files_only": True,
            "fix_mistral_regex": {"tokenizer": True},
        }

        variant = build_data["variant"]
        upsampler_repo_id = build_data["prompt_upsampler_repo_id"]
        if upsampler_repo_id is None:
            pipe.blocks.sub_blocks.pop("prompt_upsampler", None)
        else:
            upsampler_revision = build_data["prompt_upsampler_revision"]
            load_kwargs["pretrained_model_name_or_path"] = dict.fromkeys(PROMPT_UPSAMPLER_COMPONENTS, upsampler_repo_id)
            load_kwargs["subfolder"] = dict.fromkeys(PROMPT_UPSAMPLER_COMPONENTS, "")
            load_kwargs["revision"] = dict.fromkeys(PROMPT_UPSAMPLER_COMPONENTS, upsampler_revision)

        pipe.load_components(names=component_names, **load_kwargs)
        if variant:
            transformer = Flux3Transformer2DModel.from_single_file(
                build_data["variant_weights_path"],
                config=build_data["repo_id"],
                config_revision=build_data["revision"],
                subfolder="transformer",
                quantization_config=BFLQuantizationConfig(compute_dtype=torch.bfloat16),
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.update_components(transformer=transformer)
        manager.enable_auto_cpu_offload(device=get_best_device(), memory_reserve_margin="12GB")
        return pipe

    @classmethod
    def _components_to_load(cls, pipe: ModularPipeline, build_data: dict[str, Any]) -> list[str]:
        component_names = cls._unloaded_component_names(pipe)

        upsampler_repo_id = build_data["prompt_upsampler_repo_id"]
        if upsampler_repo_id is not None:
            missing_components = [name for name in PROMPT_UPSAMPLER_COMPONENTS if name not in component_names]
            if missing_components:
                raise ValueError(
                    f"Attempted to load FLUX.3 prompt upsampler from '{upsampler_repo_id}'. Failed because "
                    f"the pinned Diffusers component specs do not include {missing_components}."
                )

        excluded: set[str] = set()
        if build_data["variant"]:
            # The quantized transformer is read from a BFL-native single file, so the default
            # transformer weights are never loaded.
            excluded.add("transformer")
        if upsampler_repo_id is None:
            excluded.update(PROMPT_UPSAMPLER_COMPONENTS)

        return [name for name in component_names if name not in excluded]

    @staticmethod
    def _unloaded_component_names(pipe: ModularPipeline) -> list[str]:
        names = []
        for name in pipe.components:
            spec = pipe.get_component_spec(name)
            if spec.default_creation_method != "from_pretrained":
                continue
            if spec.pretrained_model_name_or_path is None:
                continue
            if getattr(pipe, name) is not None:
                continue
            names.append(name)
        return names


class Flux3PipelineParameters(_Flux3ModularPipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.flux3.modular_pipeline:Flux3ModularPipeline"
    _repo_id = "black-forest-labs/flux-3-image-klein-base"


class Flux3DistilledPipelineParameters(_Flux3ModularPipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.flux3.modular_pipeline:Flux3DistilledModularPipeline"
    _repo_id = "black-forest-labs/flux-3-image-klein"
