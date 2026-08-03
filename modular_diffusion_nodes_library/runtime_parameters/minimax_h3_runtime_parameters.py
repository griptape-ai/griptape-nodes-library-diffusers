import logging
from typing import ClassVar

from diffusers.modular_pipelines.minimax_h3.packing import (  # type: ignore[reportMissingImports]
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MIN_DURATION,
    align_num_frames,
)
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

# `num_frames` is snapped up to the next `17 * n + 5` the video VAE can decode, and the resulting
# duration must land in the 5-15 s window MiniMax-H3 generates. That makes 108 the smallest
# requestable count (-> 124 frames, 5.167 s) and 345 the largest (14.375 s): 346 would snap to 362,
# i.e. 15.083 s, which upstream rejects rather than silently stretching.
MIN_NUM_FRAMES = 108
MAX_NUM_FRAMES = 345
DEFAULT_NUM_FRAMES = 124


class MiniMaxH3PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=PresetCatalogImageConfig(presets=(PRESET_FIRST_LAST, PRESET_FIRST), expose_strength=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)
        self._media_gen_conditioning_param = MediaGenConditioningRuntimeParameter(
            node,
            param_name="conditioning_images",
            accepted_modes=(ConditioningMode.IMAGE,),
            tooltip="First/last keyframes for MiniMax-H3 `fl2va`, from a Media Gen Conditioning node.",
            badge_title="First/last keyframes",
            badge_message=(
                "Connect a **Media Gen Conditioning** node here to supply the frame the video starts "
                "from (`image`) and/or the frame it ends on (`last_image`). Leave it unconnected for "
                "text-only generation (`t2va`). Only **image**-mode payloads are accepted.\n\n"
                "**Note:** the canvas follows the **first** keyframe's aspect ratio, so `height` and "
                "`width` are derived from it unless you set them explicitly."
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
        num_frames_param = Parameter(
            name="num_frames",
            default_value=DEFAULT_NUM_FRAMES,
            type="int",
            tooltip=(
                f"Number of frames to generate, at the fixed {MINIMAX_H3_FPS} fps. Snapped up to the "
                f"next 17 * n + 5 the video VAE can decode; the resulting duration must stay between "
                f"{MINIMAX_H3_MIN_DURATION:g} and {MINIMAX_H3_MAX_DURATION:g} seconds."
            ),
            ui_options={"min": MIN_NUM_FRAMES, "max": MAX_NUM_FRAMES},
        )
        num_frames_param.set_badge(
            variant="help",
            title="Frame count is snapped",
            message=(
                "MiniMax-H3 only decodes frame counts of the form **17 x n + 5**, so your value is "
                "rounded **up** to the next one: 124, 141, 158, 175, 192, 209, 226, 243, 260, 277, "
                "294, 311, 328, 345.\n\n"
                "The Create Noise Latents node's frame count wins if the two disagree, because the "
                "latent's own shape is what gets denoised."
            ),
        )
        self._node.add_parameter(num_frames_param)
        self._node.add_parameter(
            Parameter(
                name="height",
                default_value=0,
                type="int",
                tooltip=(
                    f"Height of the generated video in pixels, a multiple of {MINIMAX_H3_CANVAS_MULTIPLE}. "
                    "Leave at 0 to use MiniMax-H3's own canvas for the aspect ratio of the first "
                    "keyframe, or 16:9 without one."
                ),
            )
        )
        self._node.add_parameter(
            Parameter(
                name="width",
                default_value=0,
                type="int",
                tooltip=(
                    f"Width of the generated video in pixels, a multiple of {MINIMAX_H3_CANVAS_MULTIPLE}. "
                    "Leave at 0 to use MiniMax-H3's own canvas for the aspect ratio of the first "
                    "keyframe, or 16:9 without one."
                ),
            )
        )

    def _remove_input_parameters(self) -> None:
        self._media_gen_conditioning_param.remove_input_parameters()
        self._node.remove_parameter_element_by_name("prompt")
        self._node.remove_parameter_element_by_name("num_frames")
        self._node.remove_parameter_element_by_name("height")
        self._node.remove_parameter_element_by_name("width")

    def _get_pipe_kwargs(self) -> dict:
        # 0 means "let MiniMax-H3 resolve its own canvas", which the blocks express as None.
        height = self._node.get_parameter_value("height")
        width = self._node.get_parameter_value("width")
        kwargs = {
            "prompt": self._node.get_parameter_value("prompt"),
            "num_frames": int(self._node.get_parameter_value("num_frames")),
            **self._media_gen_conditioning_param.get_pipe_kwargs(),
        }
        if height:
            kwargs["height"] = int(height)
        if width:
            kwargs["width"] = int(width)
        return kwargs

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = super().validate_before_node_run() or []
        conditioning_errors = self._media_gen_conditioning_param.validate_before_node_run()
        if conditioning_errors:
            errors.extend(conditioning_errors)

        # Validate here rather than at denoise time so a bad canvas or duration surfaces before the
        # ~124 GB load, not after it.
        num_frames = int(self._node.get_parameter_value("num_frames"))
        if num_frames < 1:
            errors.append(ValueError(f"'num_frames' must be positive, got {num_frames}."))
        else:
            duration = align_num_frames(num_frames) / MINIMAX_H3_FPS
            if not MINIMAX_H3_MIN_DURATION <= duration <= MINIMAX_H3_MAX_DURATION:
                errors.append(
                    ValueError(
                        f"Attempted to configure MiniMax-H3 generation. Failed with "
                        f"num_frames={num_frames} because it snaps up to {align_num_frames(num_frames)} "
                        f"frames ({duration:.3f} s), outside the {MINIMAX_H3_MIN_DURATION:g}-"
                        f"{MINIMAX_H3_MAX_DURATION:g} s window MiniMax-H3 generates. Use "
                        f"{MIN_NUM_FRAMES}-{MAX_NUM_FRAMES}."
                    )
                )

        for name in ("height", "width"):
            value = int(self._node.get_parameter_value(name))
            if value and value % MINIMAX_H3_CANVAS_MULTIPLE:
                errors.append(
                    ValueError(
                        f"Attempted to configure MiniMax-H3 generation. Failed with {name}={value} "
                        f"because it must be a multiple of {MINIMAX_H3_CANVAS_MULTIPLE}."
                    )
                )

        return errors or None
