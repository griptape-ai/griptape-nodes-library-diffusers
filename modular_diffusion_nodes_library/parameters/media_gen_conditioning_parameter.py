"""Parameter component for media-generation conditioning inputs.

Owns the conditioning surface (`mode`, `num_images`, `image_N*`, `video*`,
`conditioning` output) on a host `BaseNode`. The host node delegates the
`set_parameter_value` override, `validate_before_node_run`, and `process`
payload construction to this component.

The component is config-driven through a tagged union of dataclasses. Today
only `ImageOrVideoConfig` + `ImageConditioningConfig` + `VideoConditioningConfig`
is exercised by `MediaGenConditioningNode`; the unused variants
(`ImageOnlyConfig`, `VideoOnlyConfig`) are defined so Stage-2 driver authors
can adopt them without touching the dispatch shape.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode, MediaGenConditioningKey

logger = logging.getLogger("diffusers_nodes_library")


# ----------------------------------------------------------------------
# Per-modality configuration dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ImageConditioningConfig:
    """Image conditioning: 0..max_count user-added image slots.

    Each slot gets `image_N`, `image_N_frame_index`, `image_N_strength`
    parameters. The active slot count is driven by a `num_images` parameter
    the user sets on the node.
    """

    min_count: int = 0
    max_count: int = 8
    default_strength: float = 1.0
    expose_strength: bool = True
    expose_frame_index: bool = True

    def __post_init__(self) -> None:
        if self.min_count < 0:
            msg = f"ImageConditioningConfig: min_count must be >= 0, got {self.min_count}."
            raise ValueError(msg)
        if self.max_count < 1:
            msg = f"ImageConditioningConfig: max_count must be >= 1, got {self.max_count}."
            raise ValueError(msg)
        if self.max_count < self.min_count:
            msg = f"ImageConditioningConfig: max_count ({self.max_count}) must be >= min_count ({self.min_count})."
            raise ValueError(msg)
        if not 0.0 <= self.default_strength <= 1.0:
            msg = f"ImageConditioningConfig: default_strength must be in [0.0, 1.0], got {self.default_strength}."
            raise ValueError(msg)


@dataclass(frozen=True)
class VideoConditioningConfig:
    """Video conditioning: single video slot with strength + frame index."""

    default_strength: float = 1.0
    expose_strength: bool = True
    expose_frame_index: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.default_strength <= 1.0:
            msg = f"VideoConditioningConfig: default_strength must be in [0.0, 1.0], got {self.default_strength}."
            raise ValueError(msg)


# ----------------------------------------------------------------------
# Top-level conditioning config (tagged union by Python type)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class ImageOnlyConfig:
    """Image conditioning only — no `mode` dropdown, no video parameters."""

    image: ImageConditioningConfig = field(default_factory=ImageConditioningConfig)


@dataclass(frozen=True)
class VideoOnlyConfig:
    """Video conditioning only — no `mode` dropdown, no image parameters."""

    video: VideoConditioningConfig = field(default_factory=VideoConditioningConfig)


@dataclass(frozen=True)
class ImageOrVideoConfig:
    """Image OR video — user picks via a `mode` dropdown."""

    image: ImageConditioningConfig = field(default_factory=ImageConditioningConfig)
    video: VideoConditioningConfig = field(default_factory=VideoConditioningConfig)
    default_mode: ConditioningMode = ConditioningMode.IMAGE


MediaGenConditioningConfig = ImageOnlyConfig | VideoOnlyConfig | ImageOrVideoConfig


# ----------------------------------------------------------------------
# Parameter component
# ----------------------------------------------------------------------


# Parameter names — the UI-facing names the user sees on the node and that
# saved workflows reference. These are independent of the payload dict keys
# in ``MediaGenConditioningKey``; some happen to share a string value (e.g.
# both _PARAM_VIDEO and MediaGenConditioningKey.VIDEO are ``"video"``) but
# they are semantically distinct contracts. Renaming any of these breaks
# every saved workflow that references them.
_PARAM_CONDITIONING = "conditioning"
_PARAM_MODE = "mode"
_PARAM_NUM_IMAGES = "num_images"
_PARAM_VIDEO = "video"
_PARAM_VIDEO_FRAME_INDEX = "frame_index"
_PARAM_VIDEO_STRENGTH = "video_strength"


def _image_param_name(index: int) -> str:
    return f"image_{index}"


def _image_frame_index_param_name(index: int) -> str:
    return f"image_{index}_frame_index"


def _image_strength_param_name(index: int) -> str:
    return f"image_{index}_strength"


class MediaGenConditioningParameter:
    """Owns the conditioning parameter surface on a host BaseNode.

    The host node:
      1. Constructs the component in `__init__` and calls
         `add_output_parameters()` then `add_input_parameters()`.
      2. Overrides `set_parameter_value` to capture the parameter's current
         value BEFORE calling `super().set_parameter_value(...)`, then calls
         `on_parameter_value_change(name, old_value, new_value, ...)` AFTER
         super has written the new value.
      3. Delegates `validate_before_node_run` to the component.
      4. In `process`, writes
         `self.parameter_output_values["conditioning"] = component.build_conditioning_payload()`.

    The component holds no mutable state of its own. All "current value"
    reads go through the node's `get_parameter_value`. This guarantees the
    component can never desync from the node.
    """

    def __init__(self, node: BaseNode, config: MediaGenConditioningConfig) -> None:
        self._node = node
        self._config = config

    # ------------------------------------------------------------------
    # Public hook surface (called by the host node)
    # ------------------------------------------------------------------

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name=_PARAM_CONDITIONING,
                default_value={},
                output_type="dict",
                type="dict",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Media generation conditioning output.",
                serializable=False,
            )
        )

    def remove_output_parameters(self) -> None:
        self._node.remove_parameter_element_by_name(_PARAM_CONDITIONING)

    def add_input_parameters(self) -> None:
        match self._config:
            case ImageOrVideoConfig():
                self._add_mode_param()
                self._add_num_images_param()
                if self._config.default_mode is ConditioningMode.VIDEO:
                    self._node.hide_parameter_by_name(_PARAM_NUM_IMAGES)
                    self._add_video_params()
                else:
                    for i in range(self._image_cfg.min_count):
                        self._add_image_params_for_index(i)
            case ImageOnlyConfig():
                self._add_num_images_param()
                for i in range(self._image_cfg.min_count):
                    self._add_image_params_for_index(i)
            case VideoOnlyConfig():
                self._add_video_params()

    def remove_input_parameters(self) -> None:
        # Wipe every parameter we could possibly have created, regardless of
        # current mode. Safe because removal is a no-op for missing params.
        match self._config:
            case ImageOrVideoConfig():
                for i in range(self._config.image.max_count):
                    self._remove_image_params_for_index(i)
                self._node.remove_parameter_element_by_name(_PARAM_NUM_IMAGES)
                self._remove_video_params()
                self._node.remove_parameter_element_by_name(_PARAM_MODE)
            case ImageOnlyConfig():
                for i in range(self._config.image.max_count):
                    self._remove_image_params_for_index(i)
                self._node.remove_parameter_element_by_name(_PARAM_NUM_IMAGES)
            case VideoOnlyConfig():
                self._remove_video_params()

    def on_parameter_value_change(
        self,
        param_name: str,
        old_value: Any,
        new_value: Any,
        *,
        initial_setup: bool,
    ) -> None:
        """React to a parameter value change after the node has stored it.

        Called by the host node AFTER `super().set_parameter_value(...)`. The
        host is responsible for capturing `old_value` via
        `get_parameter_value(param_name)` BEFORE delegating to super, so the
        diff against `new_value` is honest.

        No-ops when:
          * `initial_setup=True` (workflow load — rebuilding params during
            value restoration corrupts the restoration order).
          * The changed parameter does not affect the conditioning surface.
          * `old_value == new_value` after any per-param normalization.
        """
        if initial_setup:
            return

        if param_name == _PARAM_MODE:
            self._handle_mode_change(old_value, new_value)
        elif param_name == _PARAM_NUM_IMAGES:
            self._handle_num_images_change(old_value, new_value)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        active_mode = self._active_mode()
        if active_mode is ConditioningMode.VIDEO:
            if self._node.get_parameter_value(_PARAM_VIDEO) is None:
                msg = f"{self._node.name}: missing required '{_PARAM_VIDEO}' conditioning input."
                errors.append(ValueError(msg))
        else:
            for i in range(self._active_num_images()):
                param_name = _image_param_name(i)
                if self._node.get_parameter_value(param_name) is None:
                    msg = f"{self._node.name}: missing required '{param_name}' conditioning input."
                    errors.append(ValueError(msg))
        return errors or None

    def build_conditioning_payload(self) -> dict:
        """Build the dict that should be written to `parameter_output_values[conditioning]`."""
        return {MediaGenConditioningKey.OUTPUT: self._build_payload()}

    # ------------------------------------------------------------------
    # Private — parameter construction helpers
    # ------------------------------------------------------------------

    def _add_mode_param(self) -> None:
        if not isinstance(self._config, ImageOrVideoConfig):
            return
        if self._node.get_parameter_by_name(_PARAM_MODE) is not None:
            return
        self._node.add_parameter(
            Parameter(
                name=_PARAM_MODE,
                default_value=self._config.default_mode.value,
                type="str",
                traits={Options(choices=[m.value for m in ConditioningMode])},
                tooltip="Select conditioning input type: image or video.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )

    def _add_num_images_param(self) -> None:
        if self._node.get_parameter_by_name(_PARAM_NUM_IMAGES) is not None:
            return
        image_cfg = self._image_cfg
        self._node.add_parameter(
            Parameter(
                name=_PARAM_NUM_IMAGES,
                default_value=image_cfg.min_count,
                type="int",
                tooltip="Number of conditioning images.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={
                    "slider": {"min_val": image_cfg.min_count, "max_val": image_cfg.max_count},
                    "step": 1,
                },
            )
        )

    def _add_image_params_for_index(self, index: int) -> None:
        image_cfg = self._image_cfg
        if self._node.get_parameter_by_name(_image_param_name(index)) is None:
            self._node.add_parameter(
                Parameter(
                    name=_image_param_name(index),
                    input_types=["ImageArtifact", "ImageUrlArtifact"],
                    type="ImageUrlArtifact",
                    tooltip=f"Conditioning image {index + 1}.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )
        if (
            image_cfg.expose_frame_index
            and self._node.get_parameter_by_name(_image_frame_index_param_name(index)) is None
        ):
            self._node.add_parameter(
                Parameter(
                    name=_image_frame_index_param_name(index),
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip=(f"Frame index in the output video where conditioning image {index + 1} is applied."),
                    user_defined=True,
                )
            )
        if image_cfg.expose_strength and self._node.get_parameter_by_name(_image_strength_param_name(index)) is None:
            self._node.add_parameter(
                Parameter(
                    name=_image_strength_param_name(index),
                    default_value=image_cfg.default_strength,
                    input_types=["float"],
                    type="float",
                    tooltip=f"Strength for conditioning image {index + 1}.",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    user_defined=True,
                )
            )

    def _remove_image_params_for_index(self, index: int) -> None:
        # These are user_defined params that may be wired to upstream nodes —
        # use the request path so the engine also tears down any connections.
        self._remove_dynamic_param(_image_param_name(index))
        self._remove_dynamic_param(_image_frame_index_param_name(index))
        self._remove_dynamic_param(_image_strength_param_name(index))

    def _add_video_params(self) -> None:
        video_cfg = self._video_cfg
        if self._node.get_parameter_by_name(_PARAM_VIDEO) is None:
            self._node.add_parameter(
                Parameter(
                    name=_PARAM_VIDEO,
                    input_types=["VideoArtifact", "VideoUrlArtifact"],
                    type="VideoUrlArtifact",
                    tooltip="Conditioning video.",
                    allowed_modes={ParameterMode.INPUT},
                    user_defined=True,
                )
            )
        if video_cfg.expose_frame_index and self._node.get_parameter_by_name(_PARAM_VIDEO_FRAME_INDEX) is None:
            self._node.add_parameter(
                Parameter(
                    name=_PARAM_VIDEO_FRAME_INDEX,
                    default_value=0,
                    input_types=["int"],
                    type="int",
                    tooltip="Frame index in the output video where conditioning video is applied.",
                    user_defined=True,
                )
            )
        if video_cfg.expose_strength and self._node.get_parameter_by_name(_PARAM_VIDEO_STRENGTH) is None:
            self._node.add_parameter(
                Parameter(
                    name=_PARAM_VIDEO_STRENGTH,
                    default_value=video_cfg.default_strength,
                    input_types=["float"],
                    type="float",
                    tooltip="Strength for the conditioning video.",
                    ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
                    user_defined=True,
                )
            )

    def _remove_video_params(self) -> None:
        # These are user_defined params that may be wired to upstream nodes —
        # use the request path so the engine also tears down any connections.
        self._remove_dynamic_param(_PARAM_VIDEO)
        self._remove_dynamic_param(_PARAM_VIDEO_FRAME_INDEX)
        self._remove_dynamic_param(_PARAM_VIDEO_STRENGTH)

    def _remove_dynamic_param(self, name: str) -> None:
        """Remove a user_defined param and any connections attached to it.

        ``remove_parameter_element_by_name`` only detaches the parameter from
        the node's UI tree; it does NOT delete connections. For any param the
        user may have wired up (`image_N`, `image_N_*`, `video`, `frame_index`,
        `video_strength`), we must go through ``RemoveParameterFromNodeRequest``
        so the node manager also issues ``DeleteConnectionRequest`` for each
        incoming / outgoing connection.
        """
        if self._node.get_element_by_name_and_type(name) is None:
            return
        GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(parameter_name=name, node_name=self._node.name))

    # ------------------------------------------------------------------
    # Private — value-change handlers
    # ------------------------------------------------------------------

    def _handle_mode_change(self, old_value: Any, new_value: Any) -> None:
        new_mode = self._normalize_mode(new_value)
        if old_value is None:
            old_mode = self._config.default_mode
        else:
            old_mode = self._normalize_mode(old_value)
        if new_mode is old_mode:
            return
        # super() only wrote `mode`; `num_images` is still pre-write, so this
        # tells us how many image_N slots exist right now.
        prior_num_images = self._active_num_images()
        self._apply_mode_change(new_mode, prior_num_images)

    def _handle_num_images_change(self, old_value: Any, new_value: Any) -> None:
        clamped_new = self._clamp_num_images(new_value)
        if old_value is None:
            clamped_old = self._image_cfg.min_count
        else:
            clamped_old = self._clamp_num_images(old_value)
        if clamped_new == clamped_old:
            return
        # Skip image-slot rebuilds while in video mode (ImageOrVideoConfig only).
        if self._active_mode() is ConditioningMode.VIDEO:
            return
        self._apply_num_images_change(clamped_new, clamped_old)

    def _apply_mode_change(
        self,
        new_mode: ConditioningMode,
        prior_num_images: int,
    ) -> None:
        if new_mode is ConditioningMode.VIDEO:
            for i in range(prior_num_images):
                self._remove_image_params_for_index(i)
            self._node.hide_parameter_by_name(_PARAM_NUM_IMAGES)
            self._add_video_params()
        else:
            self._remove_video_params()
            self._node.show_parameter_by_name(_PARAM_NUM_IMAGES)
            for i in range(self._active_num_images()):
                self._add_image_params_for_index(i)

    def _apply_num_images_change(
        self,
        new_num: int,
        current_num: int,
    ) -> None:
        if new_num > current_num:
            for i in range(current_num, new_num):
                self._add_image_params_for_index(i)
        else:
            for i in range(new_num, current_num):
                self._remove_image_params_for_index(i)

    # ------------------------------------------------------------------
    # Private — state readers (always read from the node)
    # ------------------------------------------------------------------

    def _active_mode(self) -> ConditioningMode:
        """Return the active conditioning mode regardless of which config variant is active."""
        match self._config:
            case ImageOrVideoConfig():
                value = self._node.get_parameter_value(_PARAM_MODE)
                if value is None:
                    return self._config.default_mode
                return self._normalize_mode(value)
            case ImageOnlyConfig():
                return ConditioningMode.IMAGE
            case VideoOnlyConfig():
                return ConditioningMode.VIDEO

    def _active_num_images(self) -> int:
        if not isinstance(self._config, ImageOrVideoConfig | ImageOnlyConfig):
            return 0
        value = self._node.get_parameter_value(_PARAM_NUM_IMAGES)
        if value is None:
            return self._image_cfg.min_count
        return self._clamp_num_images(value)

    # ------------------------------------------------------------------
    # Private — payload builders
    # ------------------------------------------------------------------

    def _build_payload(self) -> dict:
        if self._active_mode() is ConditioningMode.VIDEO:
            return self._build_video_payload()
        return self._build_image_payload()

    def _build_video_payload(self) -> dict:
        video_artifact = self._require_video_artifact()
        frame_index_value = self._node.get_parameter_value(_PARAM_VIDEO_FRAME_INDEX)
        if frame_index_value is None:
            frame_index_value = 0
        strength_value = self._node.get_parameter_value(_PARAM_VIDEO_STRENGTH)
        if strength_value is None:
            strength_value = self._video_cfg.default_strength
        return {
            MediaGenConditioningKey.MODE: ConditioningMode.VIDEO.value,
            MediaGenConditioningKey.VIDEO: video_artifact,
            MediaGenConditioningKey.FRAME_INDEX: int(frame_index_value),
            MediaGenConditioningKey.STRENGTH: float(strength_value),
        }

    def _build_image_payload_at(self, index: int) -> dict:
        image_artifact = self._require_image_artifact(_image_param_name(index))
        frame_index_value = self._node.get_parameter_value(_image_frame_index_param_name(index))
        if frame_index_value is None:
            frame_index_value = 0
        strength_value = self._node.get_parameter_value(_image_strength_param_name(index))
        if strength_value is None:
            strength_value = self._image_cfg.default_strength
        return {
            MediaGenConditioningKey.IMAGE: image_artifact,
            MediaGenConditioningKey.FRAME_INDEX: int(frame_index_value),
            MediaGenConditioningKey.STRENGTH: float(strength_value),
        }

    def _build_image_payload(self) -> dict:
        images = [self._build_image_payload_at(i) for i in range(self._active_num_images())]
        return {
            MediaGenConditioningKey.MODE: ConditioningMode.IMAGE.value,
            MediaGenConditioningKey.IMAGES: images,
        }

    def _require_image_artifact(self, param_name: str) -> ImageUrlArtifact:
        image_artifact = self._node.get_parameter_value(param_name)
        if image_artifact is None:
            msg = (
                f"{self._node.name}: attempted to load image. "
                f"Failed because input image was missing."
            )
            raise ValueError(msg)
        return image_artifact

    def _require_video_artifact(self) -> VideoUrlArtifact:
        video_artifact = self._node.get_parameter_value(_PARAM_VIDEO)
        if video_artifact is None:
            msg = (
                f"{self._node.name}: attempted to build video conditioning. "
                f"Failed because input video was missing."
            )
            raise ValueError(msg)
        return video_artifact

    # ------------------------------------------------------------------
    # Private — small helpers
    # ------------------------------------------------------------------

    @property
    def _image_cfg(self) -> ImageConditioningConfig:
        if not isinstance(self._config, ImageOrVideoConfig | ImageOnlyConfig):
            msg = (
                f"{self._node.name}: image config requested for config without image support: "
                f"{type(self._config).__name__}."
            )
            raise RuntimeError(msg)
        return self._config.image

    @property
    def _video_cfg(self) -> VideoConditioningConfig:
        if not isinstance(self._config, ImageOrVideoConfig | VideoOnlyConfig):
            msg = (
                f"{self._node.name}: video config requested for config without video support: "
                f"{type(self._config).__name__}."
            )
            raise RuntimeError(msg)
        return self._config.video

    def _clamp_num_images(self, value: Any) -> int:
        image_cfg = self._image_cfg
        return max(image_cfg.min_count, min(image_cfg.max_count, int(value)))

    @staticmethod
    def _normalize_mode(value: Any) -> ConditioningMode:
        """Normalize a stored `mode` value into a `ConditioningMode`.

        Accepts both raw strings and ``ConditioningMode`` members. 
        Unknown strings fall back to ``IMAGE``.
        """
        if isinstance(value, ConditioningMode):
            return value
        try:
            return ConditioningMode(str(value))
        except ValueError:
            return ConditioningMode.IMAGE
