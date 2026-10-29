import logging
from typing import ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    MediaGenConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    resolve_conditioning_image,
    resolve_conditioning_video,
)

logger = logging.getLogger("diffusers_nodes_library")

#: Wan VAE temporal compression. Segment lengths of the form 4N + 1 decode from whole latent frames.
_VAE_TEMPORAL_FACTOR = 4


class WanAnimate2PipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """Runtime surface for WAN Animate 2.

    There are no height, width or frame-count parameters: the input latent owns the geometry, and the
    driver stretches the reference image and the driving video to it.
    """

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = None

    #: Defaults from the diffusers blocks (`num_inference_steps`) and the guider's config (`guidance_scale`).
    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 40
    _DEFAULT_GUIDANCE_SCALE: ClassVar[float] = 3.0

    def __init__(self, node: BaseNode):
        super().__init__(node)

    def add_input_parameters(self) -> None:
        # Overridden only to change the num_inference_steps default, which the base hard-codes.
        self._add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=self._DEFAULT_NUM_INFERENCE_STEPS,
                type="int",
                tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality video at the expense of slower inference.",
            )
        )
        self._seed_parameter.add_input_parameters()

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt to guide video generation, e.g. a description of the character and the scene.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide video generation. Ignored when guidance_scale is 1.0.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="prompt_ref",
                default_value="",
                type="str",
                tooltip="Optional override for the reference prompt that describes the driving video to the model. Leave empty to use the model's built-in reference prompt.",
                hide_property=True,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=self._DEFAULT_GUIDANCE_SCALE,
                type="float",
                tooltip="Classifier-Free Guidance scale. Higher values steer generation more strongly toward the prompt. 1.0 disables guidance and runs only one model pass per step.",
            )
        )

        image_param = Parameter(
            name="image",
            default_value=None,
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageUrlArtifact",
            tooltip="A photo of the character to animate. Connect a Load Image node here.",
            hide_property=True,
        )
        image_param.set_badge(
            variant="help",
            title="Character reference image",
            message=(
                "A **photo of the character** you want to animate. The character's appearance — clothing, "
                "face, hair — comes from this image; the motion comes from the **driving video**.\n\n"
                "**Note:** the image is stretched to the input latent's width and height. Set the "
                "dimensions on the **Create Noise Latents** node to match the image's aspect ratio, or the "
                "character will look squashed or stretched."
            ),
        )
        self._node.add_parameter(image_param)

        driving_video_param = Parameter(
            name="driving_video",
            default_value=None,
            input_types=["VideoUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="A video of a person moving. The character copies this motion. Connect a Load Video node here.",
            hide_property=True,
        )
        driving_video_param.set_badge(
            variant="help",
            title="Driving video",
            message=(
                "Ordinary footage of a person moving — a dance, a walk, a performance. Unlike WAN Animate, "
                "no pose or face video is needed.\n\n"
                "**Length:** the output has the input latent's frame count. The driving video must have at "
                "least that many frames (at 24 fps, after the optional **driving_video_fps** resample); "
                "extra frames at the end are ignored.\n\n"
                "Frames are stretched to the input latent's width and height."
            ),
        )
        self._node.add_parameter(driving_video_param)

        self._node.add_parameter(
            Parameter(
                name="driving_video_fps",
                default_value=0.0,
                type="float",
                tooltip="The frame rate the driving video was recorded at. When set, its frames are resampled to the model's 24 fps so the motion keeps its real speed. 0 uses the frames as they are.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="segment_frame_length",
                default_value=81,
                type="int",
                tooltip="Frames generated per segment. Longer videos are generated in overlapping segments of this length. Must be 4N + 1.",
                hide_property=True,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="prev_segment_conditioning_frames",
                default_value=1,
                type="int",
                tooltip="Frames from the end of one segment that condition the next, for motion continuity. Recommended 1 or 5. Must be 4N + 1.",
                hide_property=True,
            )
        )

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("prev_segment_conditioning_frames")
        self._node.remove_parameter_element_by_name("segment_frame_length")
        self._node.remove_parameter_element_by_name("driving_video_fps")
        self._node.remove_parameter_element_by_name("driving_video")
        self._node.remove_parameter_element_by_name("image")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("prompt_ref")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("prompt")

    def _get_pipe_kwargs(self) -> dict:
        pipe_kwargs = {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "image": resolve_conditioning_image(self._node.get_parameter_value("image")),
            "driving_video": resolve_conditioning_video(self._node.get_parameter_value("driving_video")),
            "driving_video_fps": self._node.get_parameter_value("driving_video_fps"),
            "segment_frame_length": self._node.get_parameter_value("segment_frame_length"),
            "prev_segment_conditioning_frames": self._node.get_parameter_value("prev_segment_conditioning_frames"),
        }
        # Only an explicit override is passed: when it is absent, the text-encoder block falls back to the
        # reference prompt the model was trained with. An empty string would replace that with "".
        prompt_ref = self._node.get_parameter_value("prompt_ref")
        if prompt_ref:
            pipe_kwargs["prompt_ref"] = prompt_ref
        return pipe_kwargs

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = super().validate_before_node_run() or []

        if self._node.get_parameter_value("image") is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'image' (character reference) is required but not connected."
                )
            )

        if self._node.get_parameter_value("driving_video") is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'driving_video' is required but not connected."
                )
            )

        segment_frame_length = int(self._node.get_parameter_value("segment_frame_length"))
        prev_segment_conditioning_frames = int(self._node.get_parameter_value("prev_segment_conditioning_frames"))

        if segment_frame_length % _VAE_TEMPORAL_FACTOR != 1:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    f"Failed because segment_frame_length={segment_frame_length} must be of the form 4N + 1 "
                    "(e.g. 81)."
                )
            )

        if prev_segment_conditioning_frames % _VAE_TEMPORAL_FACTOR != 1:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    f"Failed because prev_segment_conditioning_frames={prev_segment_conditioning_frames} must be "
                    "of the form 4N + 1 (e.g. 1 or 5)."
                )
            )

        if prev_segment_conditioning_frames >= segment_frame_length:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    f"Failed because prev_segment_conditioning_frames={prev_segment_conditioning_frames} must be "
                    f"less than segment_frame_length={segment_frame_length}."
                )
            )

        return errors or None


class WanAnimate2DistilledPipelineRuntimeParameters(WanAnimate2PipelineRuntimeParameters):
    """Runtime surface for the distilled WAN Animate 2 checkpoint: few steps, no classifier-free guidance."""

    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 10
    _DEFAULT_GUIDANCE_SCALE: ClassVar[float] = 1.0
