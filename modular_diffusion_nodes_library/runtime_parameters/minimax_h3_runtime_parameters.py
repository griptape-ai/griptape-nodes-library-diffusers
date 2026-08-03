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


class MiniMaxH3PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """Runtime surface for MiniMax-H3.

    Deliberately narrow. The checkpoint is guidance-distilled, so there is no ``guidance_scale`` and
    no ``negative_prompt``. Frame count and canvas are not exposed either: they come from the input
    latent, like every other video pipeline in this library.
    """

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=PresetCatalogImageConfig(presets=(PRESET_FIRST_LAST, PRESET_FIRST), expose_strength=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="conditioning_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="First/last keyframes for keyframe-to-video generation, from a Media Gen Conditioning node.",
            badge_title="First/last keyframes",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the frame the video starts "
                "from and/or the frame it ends on. Leave it unconnected for text-only "
                "video-and-audio generation. Only **image**-mode payloads are accepted.\n\n"
                "**Note:** the generated canvas comes from the input latent, not from the keyframe — "
                "set the dimensions on the **Create Noise Latents** node, matching your keyframe's "
                "aspect ratio if you want the framing preserved."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt to guide generation of the video and its soundtrack.",
            )
        )
        self._media_gen_conditioning_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_before_node_run()
        if conditioning_errors:
            errors.extend(conditioning_errors)
        return errors or None
