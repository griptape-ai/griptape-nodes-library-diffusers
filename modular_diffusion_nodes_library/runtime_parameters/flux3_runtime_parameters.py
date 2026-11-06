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
from modular_diffusion_nodes_library.utils.flux3_layout import _LAYOUT_CANVAS


def _make_prompt_parameter() -> Parameter:
    prompt_param = Parameter(
        name="prompt", default_value="", type="str", tooltip="The prompt to guide image generation."
    )
    prompt_param.set_badge(
        variant="help",
        title="Layout control",
        message=(
            "Add a JSON list of regions after the caption. The caption mentions each region as `<id>`.\n\n"
            "**Text-to-Image** regions have an `id`, a `bbox`, and a `desc`.\n\n"
            "**Image Edits** (reference images connected) each region has `id`, `from`, `src_bbox`, `tgt_bbox`, and `desc`:\n"
            "- `from`: `ref_image_N` for an item in reference N (counted from 0, in the order references are "
            "supplied), or `target` for newly created content.\n"
            "- `src_bbox`: where the item sits in its reference image. Null for `target`.\n"
            "- `tgt_bbox`: where the item goes in the output. Null for removed items.\n"
            "- Moved or resized items have both boxes. Added items have only `tgt_bbox`.\n\n"
            f"> Boxes are written as `[y_min, x_min, y_max, x_max]`, y first, on a 0–{_LAYOUT_CANVAS} scale. "
            "Because the scale is normalized, the same box works at any image size."
        ),
    )
    return prompt_param


class Flux3PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 50
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(min_count=1, max_count=10, expose_strength=False, expose_frame_index=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="reference_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="Reference images for FLUX.3 image editing. Connect a Media Gen Conditioning node.",
            badge_title="Reference images",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply up to **10 reference images** for editing. "
                "Only **image**-mode payloads are accepted; **video** payloads are not allowed.\n\n"
                "**Tip:** You can also connect an image directly for single-image conditioning."
            ),
        )

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(_make_prompt_parameter())
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt not to guide generation. An empty prompt enables classifier-free guidance without negative text.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=3.5,
                type="float",
                tooltip="Higher guidance follows the prompt more closely. FLUX.3 recommends 3.5.",
            )
        )
        self._media_gen_conditioning_param.add_input_parameters()

    def add_input_parameters(self) -> None:
        self._add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=self.DEFAULT_NUM_INFERENCE_STEPS,
                type="int",
                tooltip="The number of denoising steps.",
            )
        )
        self._seed_parameter.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        for parameter_name in ("prompt", "negative_prompt", "guidance_scale"):
            self._node.remove_parameter_element_by_name(parameter_name)

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }

    def validate_in_execution_environment(self) -> list[Exception] | None:
        errors = super().validate_in_execution_environment() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_in_execution_environment()
        if conditioning_errors:
            errors.extend(conditioning_errors)
        return errors or None


class Flux3DistilledPipelineRuntimeParameters(Flux3PipelineRuntimeParameters):
    DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 16

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(_make_prompt_parameter())
        self._media_gen_conditioning_param.add_input_parameters()

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")

    def _get_pipe_kwargs(self) -> dict:
        return {
            "prompt": self._node.get_parameter_value("prompt"),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }
