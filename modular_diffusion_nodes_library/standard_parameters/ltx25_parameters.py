import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from diffusers import (  # type: ignore[reportMissingImports]
    ComponentsManager,
    FlowMatchEulerDiscreteScheduler,
    ModularPipeline,
)
from diffusers.modular_pipelines.ltx2.modular_pipeline import LTX25ModularPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

logger = logging.getLogger("diffusers_nodes_library")

_LTX25_REPO_ID = "Lightricks/LTX-2.5-Diffusers"

# The transformer alone is tens of GB in bfloat16 plus the Gemma text encoder and connectors; this
# margin is the size the reference example (`ltx25_SFT.py`) reserves for a single accelerator card.
AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN = "32GB"


class _LTX25PipelineParametersBase(ModularDiffusionPipelineTypePipelineParameters):
    """Shared plumbing for the two LTX-2.5 pipeline_type entries.

    Both entries build an `LTX25ModularPipeline` from the same gated repo and decode through the
    diffusion decoder (`LTX25AutoBlocks`'s own default) — the HF-authored `ltx25_SFT.py` reference
    script's swap to plain convolutional VAE decode was an example of what's possible, not a
    requirement tied to the Full (SFT) transformer. They differ in which transformer subfolder is
    loaded and, per that same reference script, in scheduler shifting config:

    - Distilled: default scheduler config.
    - Full (SFT): dynamic shifting re-enabled (`use_dynamic_shifting=True, shift_terminal=0.1`) — the
      distilled checkpoint's shipped `scheduler/` config has shifting turned off.
    """

    _pipeline_cls = LTX25ModularPipeline
    _transformer_subfolder: str
    _is_distilled: bool

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        "LTX25ModularPipeline has no component-override-only construction path."
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

    def requires_device_map(self) -> bool:
        # Opt-out from post-hoc pipeline optimization (`.to(device)`/`enable_*_cpu_offload`), not a
        # literal accelerate device_map request. Placement is owned entirely by `_build_pipeline_from_repo`
        # via the `ComponentsManager` below. Matches `MiniMaxH3PipelineParameters`'s identical use.
        return True

    def is_prequantized(self) -> bool:
        # Suppresses quantization and layerwise casting, both of which fire before the
        # `requires_device_map` short-circuit and are unsafe atop the `ComponentsManager` offload hooks.
        return True

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> ModularPipeline:  # noqa: ARG003
        base_repo_id = build_data["base_repo_id"]
        base_revision = build_data["base_revision"]

        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            base_repo_id,
            revision=base_revision,
            components_manager=manager,
        )

        pipe.load_components(
            dtype=torch.bfloat16,
            subfolder={"transformer": build_data["transformer_subfolder"]},
        )

        if not build_data["is_distilled"]:
            # Re-enable the dynamic shifting the distilled `scheduler/` config turns off.
            # `audio_scheduler` is deep-copied from `scheduler` at denoise time, so this one update
            # covers both.
            pipe.update_components(
                scheduler=FlowMatchEulerDiscreteScheduler.from_config(
                    pipe.scheduler.config, use_dynamic_shifting=True, shift_terminal=0.1
                )
            )

        manager.enable_auto_cpu_offload(
            device=get_best_device(),
            memory_reserve_margin=AUTO_CPU_OFFLOAD_MEMORY_RESERVE_MARGIN,
        )
        return pipe  # type: ignore[reportReturnType]


class LTX25DistilledPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer"
    _is_distilled = True


class LTX25FullPipelineParameters(_LTX25PipelineParametersBase):
    _transformer_subfolder = "transformer_full"
    _is_distilled = False
