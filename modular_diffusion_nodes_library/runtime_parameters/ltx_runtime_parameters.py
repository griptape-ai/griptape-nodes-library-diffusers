import logging
from typing import ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    PRESET_FIRST,
    PRESET_FIRST_LAST,
    PRESET_FIRST_MIDDLE_LAST,
    FlexibleImageConfig,
    HybridImageConfig,
    MediaGenConditioningConfig,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.conditioning_runtime_parameter import (
    MediaGenConditioningRuntimeParameter,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

logger = logging.getLogger("diffusers_nodes_library")


class LTXPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=HybridImageConfig(
            presets=(PRESET_FIRST_MIDDLE_LAST, PRESET_FIRST_LAST, PRESET_FIRST),
            flexible=FlexibleImageConfig(),
            default_choice=PRESET_FIRST_MIDDLE_LAST.display_name,
        ),
        video=VideoConditioningConfig(),
        default_mode=ConditioningMode.IMAGE,
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="media_conditions",
            multiple=True,
            badge_title="Frame conditions",
            badge_message=(
                "Connect one or more **Media Gen Conditioning** nodes here to insert conditioning frames at "
                "chosen positions of the generated video. Accepts both **image** and **video** payloads. "
                "Each payload sets a frame index (`first`, `last`, or a keyframe index) and a "
                "strength in `[0, 1]` \u2014 `1.0` keeps the condition fully clean, intermediate values "
                "mix it with noise. First-frame conditions overwrite the corresponding tokens; "
                "non-first-frame conditions are appended as keyframe tokens.\n\n"
                "**Tip:** You can also connect an image or video directly — without a Media Gen Conditioning node — "
                "for quick single-item conditioning at frame index **0** and strength **1.0**."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide the video generation.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide the video generation. Ignored when guidance_scale < 1.",
            )
        )
        guidance_scale_param = Parameter(
            name="guidance_scale",
            default_value=5.0,
            type="float",
            tooltip="Higher guidance scale encourages the model to generate videos more closely linked to the text prompt, usually at the expense of lower quality. Guidance is enabled by setting guidance_scale > 1.",
        )
        self._media_gen_conditioning_param.add_input_parameters()
        guidance_scale_param.set_badge(
            variant="help",
            title="Guidance scale",
            message=("For recommended values, reset the node to restore the model author's defaults."),
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
