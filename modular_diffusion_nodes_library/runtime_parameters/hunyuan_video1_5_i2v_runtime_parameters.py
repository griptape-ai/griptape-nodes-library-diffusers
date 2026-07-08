import logging
from typing import ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    PRESET_FIRST,
    MediaGenConditioningConfig,
    PresetCatalogImageConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

logger = logging.getLogger("diffusers_nodes_library")


class HunyuanVideo15ImageToVideoPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    # HunyuanVideo I2V conditions only on the first frame (no last-frame support).
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=PresetCatalogImageConfig(presets=(PRESET_FIRST,), expose_strength=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="conditioning_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="First-frame conditioning image for HunyuanVideo I2V, from a Media Gen Conditioning node.",
            badge_title="First-frame conditioning image",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the start frame "
                "for image-to-video generation. Only **image**-mode payloads are accepted.\n\n"
                "**Tip:** You can also connect an image directly — without a Media Gen Conditioning node — "
                "for single first-frame conditioning (frame index **0**)."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide the video generation from the input image.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide the video generation.",
            )
        )
        self._media_gen_conditioning_param.add_input_parameters()
        guidance_scale_param = Parameter(
            name="guidance_scale",
            default_value=7.5,
            type="float",
            tooltip=(
                "Controls how strongly the model follows the text prompt. "
                "Higher values produce videos that more closely match the prompt, usually at the expense of quality. "
                "Default 7.5 is sourced from the ClassifierFreeGuidance component default."
            ),
        )
        guidance_scale_param.set_badge(
            variant="help",
            title="Guidance scale",
            message="For recommended values, reset the node to restore the model author's defaults.",
        )
        self._node.add_parameter(guidance_scale_param)

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("guidance_scale")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_before_node_run()
        if conditioning_errors:
            errors.extend(conditioning_errors)
        return errors or None
