from __future__ import annotations

from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    FlexibleImageConfig,
    MediaGenConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


class QwenImage21PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    _TURBO_NUM_INFERENCE_STEPS: ClassVar[int] = 8

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(min_count=0, max_count=10, expose_strength=False, expose_frame_index=False),
    )

    def __init__(self, node: BaseNode, pipeline_metadata: dict[str, Any]):
        super().__init__(node)
        self._pipeline_metadata = pipeline_metadata
        self._reference_images = MediaGenConditioningRuntimeParameter(
            node,
            param_name="reference_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="Reference images for Qwen Image 2.1. Connect a Reference Media node.",
            badge_title="Reference images",
            badge_message=(
                "Connect a **Reference Media** node to provide up to **10 reference images** for editing or "
                "image-conditioned generation. Only image-mode payloads are accepted. You can also connect an "
                "image directly."
            ),
        )

    @property
    def _is_turbo(self) -> bool:
        build_data = self._pipeline_metadata.get("build_data", {})
        return build_data.get("is_turbo", False)

    def preprocess(self) -> None:
        if self._is_turbo:
            self._node.set_parameter_value("num_inference_steps", self._TURBO_NUM_INFERENCE_STEPS, emit_change=False)
        super().preprocess()

    def add_input_parameters(self) -> None:
        self._add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=self._TURBO_NUM_INFERENCE_STEPS if self._is_turbo else 40,
                type="int",
                hide=self._is_turbo,
                settable=not self._is_turbo,
                allow_input=not self._is_turbo,
                tooltip="The number of denoising steps. The base checkpoint recommends 40; Turbo uses its saved 8-step sigma schedule.",
            )
        )
        if self._is_turbo:
            self._node.set_parameter_value("num_inference_steps", self._TURBO_NUM_INFERENCE_STEPS, emit_change=False)
        self._seed_parameter.add_input_parameters()

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide image generation.",
            )
        )
        self._reference_images.add_input_parameters()
        if self._is_turbo:
            return

        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide image generation.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="true_cfg_scale",
                default_value=1.0,
                type="float",
                tooltip="True classifier-free guidance scale. Set above 1.0 and provide a negative_prompt to enable.",
            )
        )

    def _remove_input_parameters(self) -> None:
        self._reference_images.remove_input_parameters()
        self._clear_untouched_default("true_cfg_scale")
        for parameter_name in ("prompt", "negative_prompt", "true_cfg_scale"):
            self._node.remove_parameter_element_by_name(parameter_name)

    def _get_pipe_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "prompt": self._node.get_parameter_value("prompt"),
            **self._reference_images.get_pipe_kwargs(),
        }
        if not self._is_turbo:
            negative_prompt = self._node.get_parameter_value("negative_prompt")
            if negative_prompt:
                kwargs["negative_prompt"] = negative_prompt
            else:
                kwargs["negative_prompt"] = None
            kwargs["true_cfg_scale"] = self._node.get_parameter_value("true_cfg_scale")
        return kwargs

    def validate_in_execution_environment(self) -> list[Exception] | None:
        return self._reference_images.validate_in_execution_environment()
