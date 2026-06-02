import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

logger = logging.getLogger("diffusers_nodes_library")

MODE_IMAGE = "image"
MODE_VIDEO = "video"
MODE_CHOICES = [MODE_IMAGE, MODE_VIDEO]

DEFAULT_STRENGTH = 1.0
MIN_IMAGES = 0
MAX_IMAGES = 8


class MediaGenConditioningNode(ControlNode):
    """Conditioning node for media generation pipelines.

    Supports two modes:
    - image: one or more conditioning images, each with a strength value.
    - video: a single conditioning video with a strength value.

    The mode can be toggled via the 'mode' dropdown. In image mode the number of
    image inputs is controlled by the 'num_images' property.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="conditioning",
                default_value={},
                output_type="dict",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="media generation conditioning output.",
                serializable=False,
            )
        )

        self.add_parameter(
            Parameter(
                name="mode",
                default_value=MODE_IMAGE,
                type="str",
                traits={Options(choices=MODE_CHOICES)},
                tooltip="Select conditioning input type: image or video.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

        self._add_num_images_param()

    def add_parameter(self, param: Parameter) -> None:
        if not self.does_name_exist(param.name):
            super().add_parameter(param)

    # ------------------------------------------------------------------
    # Parameter construction helpers
    # ------------------------------------------------------------------

    def _add_num_images_param(self) -> None:
        if self.get_parameter_by_name("num_images") is not None:
            return
        self.add_parameter(
            Parameter(
                name="num_images",
                default_value=0,
                type="int",
                tooltip="Number of conditioning images.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"slider": {"min_val": MIN_IMAGES, "max_val": MAX_IMAGES}, "step": 1},
            )
        )

    def _add_image_params_for_index(self, index: int) -> None:
        if self.get_parameter_by_name(f"image_{index}") is None:
            self.add_parameter(
                Parameter(
                    name=f"image_{index}",
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    type="ImageUrlArtifact",
                    tooltip=f"Conditioning image {index + 1}.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )
        if self.get_parameter_by_name(f"image_{index}_frame_index") is None:
            self.add_parameter(
                Parameter(
                    name=f"image_{index}_frame_index",
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip=f"Frame index in the output video where conditioning image {index + 1} is applied.",
                    user_defined=True,
                )
            )
        if self.get_parameter_by_name(f"image_{index}_strength") is None:
            self.add_parameter(
                Parameter(
                    name=f"image_{index}_strength",
                    default_value=DEFAULT_STRENGTH,
                    input_types=["float"],
                    type="float",
                    tooltip=f"Strength for conditioning image {index + 1}.",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    user_defined=True,
                )
            )

    def _remove_image_params_for_index(self, index: int) -> None:
        self._remove_param(f"image_{index}")
        self._remove_param(f"image_{index}_frame_index")
        self._remove_param(f"image_{index}_strength")

    def _add_video_params(self) -> None:
        if self.get_parameter_by_name("video") is None:
            self.add_parameter(
                Parameter(
                    name="video",
                    input_types=["VideoArtifact", "VideoUrlArtifact"],
                    type="VideoUrlArtifact",
                    tooltip="Conditioning video.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )
        if self.get_parameter_by_name("frame_index") is None:
            self.add_parameter(
                Parameter(
                    name="frame_index",
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip="Frame index in the output video where conditioning video is applied.",
                    user_defined=True,
                )
            )
        if self.get_parameter_by_name("video_strength") is None:
            self.add_parameter(
                Parameter(
                    name="video_strength",
                    default_value=DEFAULT_STRENGTH,
                    input_types=["float"],
                    type="float",
                    tooltip="Strength for the conditioning video.",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    user_defined=True,
                )
            )

    def _remove_video_params(self) -> None:
        self._remove_param("video")
        self._remove_param("frame_index")
        self._remove_param("video_strength")

    def _remove_param(self, name: str) -> None:
        if self.get_element_by_name_and_type(name):
            GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(parameter_name=name, node_name=self.name))

    # ------------------------------------------------------------------
    # Mode / num_images change logic
    # ------------------------------------------------------------------

    def _apply_mode_change(self, new_mode: str, num_images: int) -> None:
        if new_mode == MODE_VIDEO:
            for i in range(num_images):
                self._remove_image_params_for_index(i)
            if self.get_parameter_by_name("num_images") is not None:
                self.hide_parameter_by_name("num_images")
            self._add_video_params()
        else:
            self._remove_video_params()
            if self.get_parameter_by_name("num_images") is not None:
                self.show_parameter_by_name("num_images")
            for i in range(num_images):
                self._add_image_params_for_index(i)

    def _apply_num_images_change(self, new_num: int, current_num: int) -> None:
        new_num = max(MIN_IMAGES, min(MAX_IMAGES, new_num))
        if self._current_mode() != MODE_VIDEO:
            if new_num > current_num:
                for i in range(current_num, new_num):
                    self._add_image_params_for_index(i)
            else:
                for i in range(new_num, current_num):
                    self._remove_image_params_for_index(i)

    # ------------------------------------------------------------------
    # Value-set override
    # ------------------------------------------------------------------

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return

        current_mode = self._current_mode()
        current_num = self._current_num_images()

        if param_name == "num_images":
            value = max(MIN_IMAGES, min(MAX_IMAGES, int(value)))

        did_mode_change = param_name == "mode" and current_mode != value
        did_num_images_change = param_name == "num_images" and current_num != value

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if did_mode_change:
            self._apply_mode_change(str(value), current_num)
        elif did_num_images_change:
            self._apply_num_images_change(int(value), current_num)

    # ------------------------------------------------------------------
    # Validation & execution
    # ------------------------------------------------------------------

    def _current_mode(self) -> str:
        return str(self.get_parameter_value("mode") or MODE_IMAGE)

    def _current_num_images(self) -> int:
        value = self.get_parameter_value("num_images")
        if value is None:
            return 0
        return max(MIN_IMAGES, min(MAX_IMAGES, int(value)))

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        if self._current_mode() == MODE_VIDEO:
            if self.get_parameter_value("video") is None:
                errors.append(ValueError("Missing required 'video' conditioning input."))
        else:
            for i in range(self._current_num_images()):
                if self.get_parameter_value(f"image_{i}") is None:
                    errors.append(ValueError(f"Missing required 'image_{i}' conditioning input."))
        return errors or None

    def process(self) -> None:
        conditioning = {"media_gen_conditioning": self._build_conditioning()}
        self.set_parameter_value("conditioning", conditioning)
        self.parameter_output_values["conditioning"] = conditioning

    def _build_conditioning(self) -> dict:
        if self._current_mode() == MODE_VIDEO:
            video_artifact = self._get_video_artifact()
            frame_index_value = self.get_parameter_value("frame_index")
            return {
                "mode": "video",
                "video": video_artifact,
                "frame_index": int(frame_index_value) if frame_index_value is not None else 0,
                "strength": float(self.get_parameter_value("video_strength")),
            }

        images = []
        for i in range(self._current_num_images()):
            frame_index_value = self.get_parameter_value(f"image_{i}_frame_index")
            images.append(
                {
                    "image": self._get_image_artifact(f"image_{i}"),
                    "frame_index": int(frame_index_value) if frame_index_value is not None else i,
                    "strength": float(self.get_parameter_value(f"image_{i}_strength")),
                }
            )
        return {
            "mode": "image",
            "images": images,
        }

    def _get_image_artifact(self, param_name: str) -> ImageUrlArtifact:
        image_artifact = self.get_parameter_value(param_name)
        if image_artifact is None:
            msg = f"Attempted to load image. Failed with data {param_name}=None because input image was missing."
            raise ValueError(msg)
        return image_artifact

    def _get_video_artifact(self) -> VideoUrlArtifact:
        video_artifact = self.get_parameter_value("video")
        if video_artifact is None:
            msg = "Attempted to build video conditioning. Failed with data video=None because input video was missing."
            raise ValueError(msg)

        return video_artifact
