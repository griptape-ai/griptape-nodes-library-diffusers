# Copied from diffusers_nodes_library.common.parameters.diffusion.flux2.runtime_parameters
import logging

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)

logger = logging.getLogger("diffusers_nodes_library")


class Flux2PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide the image generation.",
            )
        )
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
        self._node.remove_parameter_element_by_name("guidance_scale")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
        }
