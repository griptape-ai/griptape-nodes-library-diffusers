# Copied from diffusers_nodes_library.common.parameters.diffusion.qwen.edit_runtime_parameters
import logging

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from PIL.Image import Image

from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import (
    image_artifact_to_pil,
)

logger = logging.getLogger("diffusers_nodes_library")


class QwenEditPipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Image to be edited.",
                ui_options={"hide_property": True},
            )
        )
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
        self._node.remove_parameter_element_by_name("image")

    def validate_before_node_run(self) -> list[Exception] | None:
        image = self._node.get_parameter_value("image")
        if image is None:
            return [ValueError("Image must be connected to use Qwen Edit")]
        return None

    def get_image_pil(self) -> Image:
        input_image_artifact = self._node.get_parameter_value("image")
        if isinstance(input_image_artifact, ImageUrlArtifact):
            input_image_artifact = load_image_from_url_artifact(input_image_artifact)
        input_image_pil = image_artifact_to_pil(input_image_artifact)
        return input_image_pil.convert("RGB")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "true_cfg_scale": self._node.get_parameter_value("true_cfg_scale"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "image": self.get_image_pil(),
        }
