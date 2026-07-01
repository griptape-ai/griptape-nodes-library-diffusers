import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    MediaGenConditioningConfig,
    VideoConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    resolve_conditioning_image,
    resolve_conditioning_video,
)

logger = logging.getLogger("diffusers_nodes_library")


class WanAnimatePipelineRuntimeParameters(DiffusionPipelineRuntimeParameters):
    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        video=VideoConditioningConfig(expose_strength=False, expose_frame_index=False),
    )

    def __init__(self, node: BaseNode):
        super().__init__(node)

    def _add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts to guide video generation.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="negative_prompt",
                default_value="",
                type="str",
                tooltip="The prompt or prompts not to guide video generation. Ignored when guidance_scale is less than 1.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="guidance_scale",
                default_value=1.0,
                type="float",
                tooltip="Classifier-Free Guidance scale. By default CFG is not used in Wan Animate inference (guidance_scale=1.0). Higher values steer generation more strongly toward the prompt, usually at the expense of visual quality.",
            )
        )
        self._node.add_parameter(
            Parameter(
                name="mode",
                default_value="animate",
                type="str",
                tooltip="Generation mode. 'animate' drives a character with pose and face videos. 'replace' additionally composites the animated character into a background scene using background_video and mask_video.",
                traits={Options(choices=["animate", "replace"])},
            )
        )
        self._node.add_parameter(
            Parameter(
                name="prev_segment_conditioning_frames",
                default_value=1,
                type="int",
                tooltip="Number of frames from the previous segment used for temporal continuity guidance. Recommended to be 1 or 5. Should generally be 4N + 1 for non-negative integer N.",
                hide_property=True,
            )
        )
        image_param = Parameter(
            name="image",
            default_value=None,
            input_types=["ImageArtifact", "ImageUrlArtifact"],
            type="ImageUrlArtifact",
            tooltip="A photo of the character you want to animate. Connect a Load Image node here.",
            hide_property=True,
        )
        image_param.set_badge(
            variant="help",
            title="Character reference image",
            message=(
                "This is a **photo of the character** you want to animate — a person, a fictional character, "
                "or any subject whose body and face you want to bring to life.\n\n"
                "**What to connect:** Use a **Load Image** node to load a photo from your computer or a URL.\n\n"
                "**Tips for best results:**\n"
                "- Use a clear, well-lit photo where the character is fully visible\n"
                "- Front-facing or three-quarter views work better than side profiles\n"
                "- The full body (or at least the upper body) should be visible\n"
                "- Avoid blurry or heavily cropped images\n\n"
                "The character's appearance — clothing, face, hair — comes entirely from this image. "
                "Their body movement will follow the **pose video**, and their facial expressions will follow the **face video**."
            ),
        )
        self._node.add_parameter(image_param)

        pose_param = Parameter(
            name="pose_video",
            default_value=None,
            input_types=["VideoUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="Stick-figure skeleton video that drives the character's body movement. Generate it from a source video using the OpenPose Video Detection node.",
            hide_property=True,
        )
        pose_param.set_badge(
            variant="help",
            title="Pose keypoint video",
            message=(
                "This is **not** the original source footage. It is a computer-generated **stick-figure skeleton video** "
                "where colored dots and lines represent body joints and limbs (shoulders, elbows, wrists, hips, knees, etc.).\n\n"
                "**Why it looks like this:** The model reads body positions from this skeleton visualization, not from "
                "real video pixels. The stick-figure format tells the model exactly where each body part is in every frame.\n\n"
                "**How to create it:**\n"
                "1. Start with any video showing a person moving — a dance, a walk, a performance\n"
                "2. Connect a **Load Video** node → **OpenPose Video Detection** node\n"
                "3. The OpenPose node outputs a skeleton video — connect that here\n\n"
                "**The character in your reference image will copy the body movements from this skeleton video.** "
                "The longer the pose video, the longer the generated animation."
            ),
        )
        self._node.add_parameter(pose_param)

        face_param = Parameter(
            name="face_video",
            default_value=None,
            input_types=["VideoUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="Close-up face video that drives the character's facial expressions. Should be cropped tightly to the face from a source video.",
            hide_property=True,
        )
        face_param.set_badge(
            variant="help",
            title="Face feature video",
            message=(
                "This is a **close-up video of a face** that drives the facial expressions of your animated character. "
                "Unlike the pose video, this is real footage — but it must be **tightly cropped to just the face region**.\n\n"
                "The model's internal motion encoder reads the face pixels frame-by-frame and extracts the expression, "
                "lip movements, and eye motion. Your reference character's face will mimic these movements.\n\n"
                "**How to create it:**\n"
                "1. Take your source video (the same one you used for the pose video, or a different one)\n"
                "2. Crop it so only the face fills the frame — no background, no shoulders if possible\n"
                "3. Connect a **Load Video** node here with the cropped face video\n\n"
                "**Tips:**\n"
                "- The face should be well-lit and clearly visible in every frame\n"
                "- Exaggerated expressions transfer more effectively than subtle ones\n"
                "- The video is automatically resized internally — exact dimensions are not critical"
            ),
        )
        self._node.add_parameter(face_param)

        background_param = Parameter(
            name="background_video",
            default_value=None,
            input_types=["VideoArtifact", "VideoUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="Background scene video for 'replace' mode. The animated character will be composited into this video. Connect a Load Video node here.",
            hide=True,
            hide_property=True,
        )
        background_param.set_badge(
            variant="help",
            title="Background video (replace mode)",
            message=(
                "This is the **scene or location** where your animated character will appear. Think of it as the "
                "film set or backdrop — any regular video of a place, room, street, or environment.\n\n"
                "**What to connect:** Use a **Load Video** node to load any video file.\n\n"
                "**How it works:** In 'replace' mode, the model composites the animated character *into* this video. "
                "The **mask video** (below) controls exactly where in each frame the character appears — "
                "white areas in the mask are replaced by the animated character; black areas keep the original background.\n\n"
                "**Tips:**\n"
                "- The background video length should match or exceed the pose video length\n"
                "- The background and the pose video should have compatible framing (e.g. both at the same scale)\n"
                "- The background does not need to have a person in it — it can be an empty scene\n\n"
                "⚠️ Only visible when **mode** is set to **'replace'**."
            ),
        )
        self._node.add_parameter(background_param)

        mask_video_param = Parameter(
            name="mask_video",
            default_value=None,
            input_types=["VideoArtifact", "VideoUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="Per-frame binary mask for 'replace' mode. Black (0) = preserve background; white (255) = generate from prompt.",
            hide=True,
            hide_property=True,
        )
        mask_video_param.set_badge(
            variant="help",
            title="Mask video (replace mode)",
            message=(
                "A grayscale video used as a per-frame mask in **replace** mode.\n\n"
                "**Black (0)** = preserve the background video content at that frame.\n"
                "**White (255)** = generate from the prompt at that frame.\n\n"
                "⚠️ Only visible when **mode** is set to **'replace'**."
            ),
        )
        self._node.add_parameter(mask_video_param)

        if self._node.get_parameter_value("mode") == "replace":
            self._node.show_parameter_by_name("background_video")
            self._node.show_parameter_by_name("mask_video")

    def _remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("mask_video")
        self._node.remove_parameter_element_by_name("background_video")
        self._node.remove_parameter_element_by_name("face_video")
        self._node.remove_parameter_element_by_name("pose_video")
        self._node.remove_parameter_element_by_name("image")
        self._node.remove_parameter_element_by_name("prev_segment_conditioning_frames")
        self._node.remove_parameter_element_by_name("mode")
        self._node.remove_parameter_element_by_name("guidance_scale")
        self._node.remove_parameter_element_by_name("negative_prompt")
        self._node.remove_parameter_element_by_name("prompt")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter.name == "mode":
            is_replace = value == "replace"
            if is_replace:
                self._node.show_parameter_by_name("background_video")
                self._node.show_parameter_by_name("mask_video")
            else:
                self._node.hide_parameter_by_name("background_video")
                self._node.hide_parameter_by_name("mask_video")

    def _get_pipe_kwargs(self) -> dict:
        pipe_kwargs: dict[str, Any] = {
            "prompt": self._node.get_parameter_value("prompt"),
            "negative_prompt": self._node.get_parameter_value("negative_prompt"),
            "guidance_scale": self._node.get_parameter_value("guidance_scale"),
            "mode": self._node.get_parameter_value("mode"),
            "prev_segment_conditioning_frames": self._node.get_parameter_value("prev_segment_conditioning_frames"),
            "image": resolve_conditioning_image(self._node.get_parameter_value("image")),
            "pose_video": resolve_conditioning_video(self._node.get_parameter_value("pose_video")),
            "face_video": resolve_conditioning_video(self._node.get_parameter_value("face_video")),
        }
        background_video = self._node.get_parameter_value("background_video")
        if background_video is not None:
            pipe_kwargs["background_video"] = resolve_conditioning_video(background_video)
        mask_video = self._node.get_parameter_value("mask_video")
        if mask_video is not None:
            pipe_kwargs["mask_video"] = resolve_conditioning_video(mask_video)
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

        if self._node.get_parameter_value("pose_video") is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'pose_video' is required but not connected."
                )
            )

        if self._node.get_parameter_value("face_video") is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'face_video' is required but not connected."
                )
            )

        input_latent = self._node.get_parameter_value("input_latent")
        pose_video_value = self._node.get_parameter_value("pose_video")
        if input_latent is not None and pose_video_value is not None:
            source_shape = getattr(input_latent, "source_shape", None)
            if source_shape is not None and len(source_shape) >= 3:
                pose_frames = resolve_conditioning_video(pose_video_value)
                if pose_frames is not None and len(pose_frames) != source_shape[-3]:
                    errors.append(
                        ValueError(
                            f"Attempted to validate '{self._node.name}'. "
                            f"Failed because input latent num_frames={source_shape[-3]} does not match "
                            f"pose_video length={len(pose_frames)}. "
                            "Ensure the Create Noise Latent node frame count matches the pose video frame count."
                        )
                    )

        mode = self._node.get_parameter_value("mode")
        background_video = self._node.get_parameter_value("background_video")
        mask_video = self._node.get_parameter_value("mask_video")

        if mode == "replace" and background_video is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'background_video' is required when mode is 'replace'. "
                    "Connect a background video or switch mode to 'animate'."
                )
            )

        if mode == "replace" and mask_video is None:
            errors.append(
                ValueError(
                    f"Attempted to validate '{self._node.name}'. "
                    "Failed because 'mask_video' is required when mode is 'replace'. "
                    "Connect a mask video or switch mode to 'animate'."
                )
            )

        return errors or None
