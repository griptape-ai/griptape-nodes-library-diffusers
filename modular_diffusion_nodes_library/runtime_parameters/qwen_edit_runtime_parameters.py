import logging
from typing import ClassVar

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

logger = logging.getLogger("diffusers_nodes_library")


class QwenEditPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(
            min_count=1,
            max_count=8,
            expose_strength=False,
            expose_frame_index=False,
        ),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._image_references = MediaGenConditioningRuntimeParameter(
            node,
            param_name="image_references",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="Image(s) to edit. Connect a Media Gen Conditioning node for multi-image support, or connect an image directly.",
            badge_title="Reference images",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply one or more reference images for editing. "
                "Only **image**-mode payloads are accepted; **video** payloads are not allowed.\n\n"
                "**Tip:** You can also connect an image directly — without a Media Gen Conditioning node — "
                "for single-image conditioning."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._image_references.add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide the image editing.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide the image editing.",
            )
        )
        true_cfg_param = Parameter(
            name="true_cfg_scale",
            default_value=1.0,
            type="float",
            tooltip="True classifier-free guidance scale. Set above 1.0 and provide a negative_prompt to enable. Higher values push output further from the negative prompt.",
        )
        true_cfg_param.set_badge(
            variant="help",
            title="Using negative prompts",
            message=(
                "This model doesn't support negative prompts out of the box. "
                "This setting adds that ability, but at a cost: the model runs twice per step, "
                "so generation takes roughly twice as long.\n\n"
                "- ***1.0*** — off (default, faster)\n"
                "- ***2.0–4.0*** — on; only has an effect when you also fill in `negative_prompt`\n\n"
                "For most use cases, adjusting `guidance_scale` is enough and has no speed penalty."
            ),
        )
        self._node.add_parameter(true_cfg_param)
        guidance_scale_param = Parameter(
            name="guidance_scale",
            default_value=4.0,
            type="float",
            tooltip="Higher guidance_scale encourages a model to generate images more aligned with prompt at the expense of lower image quality.",
        )
        guidance_scale_param.set_badge(
            variant="help",
            title="Guidance scale",
            message=("For recommended values, reset the node to restore the model author's defaults."),
        )
        self._node.add_parameter(guidance_scale_param)

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("true_cfg_scale")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._image_references.remove_input_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        if self._node.get_parameter_value("image_references") is None:
            return [ValueError("image_references must be connected to use Qwen Edit.")]
        return self._image_references.validate_before_node_run()

    def _get_pipe_kwargs(self) -> dict:
        base = {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "true_cfg_scale": self._node.get_parameter_value("true_cfg_scale"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
        }
        base.update(self._image_references.get_pipe_kwargs())
        return base
