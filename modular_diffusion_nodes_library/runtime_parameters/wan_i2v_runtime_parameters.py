import logging
from typing import ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    PRESET_FIRST,
    PRESET_FIRST_LAST,
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


class WanImageToVideoPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=PresetCatalogImageConfig(presets=(PRESET_FIRST_LAST, PRESET_FIRST), expose_strength=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="conditioning_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="First/last conditioning images for WAN i2v, from a Media Gen Conditioning node.",
            badge_title="First/last conditioning images",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the start (and optionally end) "
                "frame for image-to-video generation. Only **image**-mode payloads are accepted; "
                "**video** payloads are rejected.\n\n"
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
