"""Declarative layout for the media-gen conditioning parameter surface.

Configs (`FlexibleImageConfig`, `PresetCatalogImageConfig`, `HybridImageConfig`,
`VideoConditioningConfig`, and the top-level `MediaGenConditioningConfig`)
produce a `Layout` of `ControlParam`/`ConditioningInput` via
`derive_layout(control_values)`. Pure data \u2014 the composer in
`node_layout_composer.py` is what applies a layout to the host node.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    FramePosition,
)

# ----------------------------------------------------------------------
# Control-parameter names (single source of truth)
# ----------------------------------------------------------------------

PARAM_MODE = "mode"
PARAM_NUM_IMAGES = "num_images"
PARAM_IMAGE_PRESET = "image_preset"
PARAM_VIDEO = "video"
PARAM_VIDEO_FRAME_INDEX = "frame_index"
PARAM_VIDEO_STRENGTH = "video_strength"

CONTROL_PARAM_NAMES = frozenset({PARAM_MODE, PARAM_NUM_IMAGES, PARAM_IMAGE_PRESET})


# ----------------------------------------------------------------------
# Derived types (what configs produce)
# ----------------------------------------------------------------------


class ControlKind(StrEnum):
    """How the composer should materialize a `ControlParam`."""

    OPTION = "Option"
    SLIDER = "Slider"


@dataclass(frozen=True)
class ControlParam:
    """A non-input parameter that drives layout structure."""

    name: str
    kind: ControlKind
    default: Any
    display_name: str
    tooltip: str
    choices: tuple[str, ...] = ()
    slider_min: int = 0
    slider_max: int = 0
    hide: bool = False

    @classmethod
    def mode_toggle(
        cls,
        default_mode: ConditioningMode,
        *,
        allowed_modes: tuple[ConditioningMode, ...],
        hide: bool = False,
    ) -> ControlParam:
        fallback_mode = allowed_modes[0]
        default_value = default_mode.value if default_mode in allowed_modes else fallback_mode.value
        return cls(
            name=PARAM_MODE,
            kind=ControlKind.OPTION,
            default=default_value,
            display_name="Conditioning Mode",
            tooltip="Select conditioning input type: image or video.",
            choices=tuple(mode.value for mode in allowed_modes),
            hide=hide,
        )

    @classmethod
    def num_images_slider(cls, min_count: int, max_count: int, *, hide: bool = False) -> ControlParam:
        return cls(
            name=PARAM_NUM_IMAGES,
            kind=ControlKind.SLIDER,
            default=min_count,
            display_name="Number of Images",
            tooltip="Number of conditioning images.",
            slider_min=min_count,
            slider_max=max_count,
            hide=hide,
        )

    @classmethod
    def preset_dropdown(
        cls,
        choices: tuple[str, ...],
        default: str,
        *,
        tooltip: str | None = None,
        hide: bool = False,
    ) -> ControlParam:
        return cls(
            name=PARAM_IMAGE_PRESET,
            kind=ControlKind.OPTION,
            default=default,
            display_name="Preset",
            tooltip=tooltip or "Select which preset arrangement of conditioning images to use.",
            choices=choices,
            hide=hide,
        )


@dataclass(frozen=True)
class ConditioningInput:
    """One conditioning input the user provides (one image or one video)."""

    index: int
    kind: Literal["image", "video"]
    label: str
    group_label: str
    fixed_position: FramePosition | None
    expose_strength: bool
    expose_frame_index: bool
    default_strength: float
    image_tooltip: str
    strength_tooltip: str
    frame_index_tooltip: str

    @property
    def key(self) -> str:
        """Stable diff identity used by the composer across layout transitions."""
        if self.kind == "video":
            return "video"
        if self.fixed_position is not None:
            return self.fixed_position.value
        return f"extra_{self.index}"

    @property
    def media_param(self) -> str:
        """Name of the image/video Parameter on the host node."""
        if self.kind == "video":
            return PARAM_VIDEO
        return f"image_{self.index}"

    @property
    def strength_param(self) -> str:
        if self.kind == "video":
            return PARAM_VIDEO_STRENGTH
        return f"image_{self.index}_strength"

    @property
    def frame_index_param(self) -> str:
        if self.kind == "video":
            return PARAM_VIDEO_FRAME_INDEX
        return f"image_{self.index}_frame_index"

    @property
    def group_param(self) -> str | None:
        """Wrapping `ParameterGroup` name, or None for video (ungrouped)."""
        if self.kind == "video":
            return None
        return f"image_{self.index}_group"

    @classmethod
    def flexible(cls, index: int, config: FlexibleImageConfig) -> ConditioningInput:
        return cls(
            index=index,
            kind="image",
            label="Image",
            group_label=f"Image {index + 1}",
            fixed_position=None,
            expose_strength=config.expose_strength,
            expose_frame_index=config.expose_frame_index,
            default_strength=config.default_strength,
            image_tooltip=f"Conditioning image {index + 1}.",
            strength_tooltip=f"Strength for conditioning image {index + 1}.",
            frame_index_tooltip=f"Frame index in the output video where conditioning image {index + 1} is applied.",
        )

    @classmethod
    def preset(
        cls,
        index: int,
        position: FramePosition,
        *,
        expose_strength: bool,
        default_strength: float,
    ) -> ConditioningInput:
        label = position.value.capitalize()
        return cls(
            index=index,
            kind="image",
            label=label,
            group_label=f"Image {index + 1} \u2014 {label}",
            fixed_position=position,
            expose_strength=expose_strength,
            expose_frame_index=False,
            default_strength=default_strength,
            image_tooltip=f"Conditioning image for the {label.lower()} input.",
            strength_tooltip=f"Strength for the {label.lower()} conditioning image.",
            frame_index_tooltip="",
        )

    @classmethod
    def video(cls, config: VideoConditioningConfig) -> ConditioningInput:
        return cls(
            index=0,
            kind="video",
            label="Video",
            group_label="Video",
            fixed_position=None,
            expose_strength=config.expose_strength,
            expose_frame_index=config.expose_frame_index,
            default_strength=config.default_strength,
            image_tooltip="Conditioning video.",
            strength_tooltip="Strength for the conditioning video.",
            frame_index_tooltip="Frame index in the output video where conditioning video is applied.",
        )


@dataclass(frozen=True)
class Layout:
    """Snapshot of the conditioning parameter surface."""

    control_params: tuple[ControlParam, ...]
    cond_inputs: tuple[ConditioningInput, ...]

    @property
    def all_param_names(self) -> tuple[str, ...]:
        """Render order: controls, then per input (group, media, frame_index, strength)."""
        names: list[str] = [c.name for c in self.control_params]
        for ci in self.cond_inputs:
            if ci.group_param is not None:
                names.append(ci.group_param)
            names.append(ci.media_param)
            if ci.expose_frame_index:
                names.append(ci.frame_index_param)
            if ci.expose_strength:
                names.append(ci.strength_param)
        return tuple(names)


EMPTY_LAYOUT = Layout(control_params=(), cond_inputs=())


# ----------------------------------------------------------------------
# Configs (declared by the library author)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class FlexibleImageConfig:
    """0..max_count user-added image inputs; count driven by `num_images`."""

    min_count: int = 0
    max_count: int = 8
    default_strength: float = 1.0
    expose_strength: bool = True
    expose_frame_index: bool = True

    def __post_init__(self) -> None:
        if self.min_count < 0:
            raise ValueError(f"min_count must be >= 0, got {self.min_count}.")
        if self.max_count < 1:
            raise ValueError(f"max_count must be >= 1, got {self.max_count}.")
        if self.max_count < self.min_count:
            raise ValueError(f"max_count ({self.max_count}) must be >= min_count ({self.min_count}).")
        if not 0.0 <= self.default_strength <= 1.0:
            raise ValueError(f"default_strength must be in [0.0, 1.0], got {self.default_strength}.")

    def derive_layout(self, control_values: Mapping[str, Any]) -> Layout:
        raw = control_values.get(PARAM_NUM_IMAGES)
        count = self.min_count if raw is None else max(self.min_count, min(self.max_count, int(raw)))
        controls = (
            ControlParam.preset_dropdown(("Custom",), "Custom", hide=True),
            ControlParam.num_images_slider(self.min_count, self.max_count),
        )
        cond_inputs = tuple(ConditioningInput.flexible(i, self) for i in range(count))
        return Layout(control_params=controls, cond_inputs=cond_inputs)


@dataclass(frozen=True)
class ImagePreset:
    """Named preset: ordered tuple of frame positions."""

    id: str
    display_name: str
    positions: tuple[FramePosition, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id must be non-empty.")
        if not self.positions:
            raise ValueError(f"ImagePreset {self.id!r}: positions tuple must be non-empty.")
        if len(self.positions) != len(set(self.positions)):
            raise ValueError(f"ImagePreset {self.id!r}: duplicate positions in {list(self.positions)}.")


PRESET_FIRST = ImagePreset("first", "First frame", (FramePosition.FIRST,))
PRESET_FIRST_LAST = ImagePreset("first_last", "First + Last", (FramePosition.FIRST, FramePosition.LAST))
PRESET_FIRST_MIDDLE_LAST = ImagePreset(
    "first_middle_last",
    "First + Middle + Last",
    (FramePosition.FIRST, FramePosition.MIDDLE, FramePosition.LAST),
)


@dataclass(frozen=True)
class PresetCatalogImageConfig:
    """Fixed catalog of named presets; positions locked, no per-input frame_index."""

    presets: tuple[ImagePreset, ...]
    expose_strength: bool = True
    default_strength: float = 1.0

    def __post_init__(self) -> None:
        if not self.presets:
            raise ValueError("presets tuple must be non-empty.")
        ids = [p.id for p in self.presets]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate preset ids in {ids}.")
        if not 0.0 <= self.default_strength <= 1.0:
            raise ValueError(f"default_strength must be in [0.0, 1.0], got {self.default_strength}.")

    def derive_layout(self, control_values: Mapping[str, Any]) -> Layout:
        preset = self._resolve_preset(control_values.get(PARAM_IMAGE_PRESET))
        choices = tuple(p.display_name for p in self.presets)
        preset_count = len(preset.positions)
        hide_presets = len(self.presets) <= 1
        controls = (
            ControlParam.preset_dropdown(choices, preset.display_name, hide=hide_presets),
            ControlParam.num_images_slider(preset_count, preset_count, hide=True),
        )
        cond_inputs = tuple(
            ConditioningInput.preset(
                i, position, expose_strength=self.expose_strength, default_strength=self.default_strength
            )
            for i, position in enumerate(preset.positions)
        )
        return Layout(control_params=controls, cond_inputs=cond_inputs)

    def _resolve_preset(self, value: Any) -> ImagePreset:
        if value is None:
            return self.presets[0]
        for preset in self.presets:
            if preset.display_name == value:
                return preset
        return self.presets[0]


@dataclass(frozen=True)
class HybridImageConfig:
    """Preset catalog plus a flexible fallback choice in a single dropdown."""

    presets: tuple[ImagePreset, ...]
    flexible: FlexibleImageConfig = field(default_factory=FlexibleImageConfig)
    flexible_choice_label: str = "Custom"
    default_choice: str | None = None

    def __post_init__(self) -> None:
        if not self.presets:
            raise ValueError("presets tuple must be non-empty (use FlexibleImageConfig directly otherwise).")
        ids = [p.id for p in self.presets]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate preset ids in {ids}.")
        if not self.flexible_choice_label:
            raise ValueError("flexible_choice_label must be non-empty.")
        names = [p.display_name for p in self.presets]
        if self.flexible_choice_label in names:
            raise ValueError(f"flexible_choice_label {self.flexible_choice_label!r} collides with a preset name.")
        valid = [self.flexible_choice_label, *names]
        if self.default_choice is not None and self.default_choice not in valid:
            raise ValueError(f"default_choice {self.default_choice!r} not in {valid}.")

    def derive_layout(self, control_values: Mapping[str, Any]) -> Layout:
        choices = (self.flexible_choice_label, *(p.display_name for p in self.presets))
        default = self.default_choice or self.flexible_choice_label
        raw_choice = control_values.get(PARAM_IMAGE_PRESET)
        active_choice = raw_choice if raw_choice in choices else default

        tooltip = (
            "Pick a named preset arrangement of conditioning images, "
            f"or {self.flexible_choice_label!r} for arbitrary frame indices."
        )
        dropdown = ControlParam.preset_dropdown(choices, default, tooltip=tooltip)

        if active_choice == self.flexible_choice_label:
            flexible_layout = self.flexible.derive_layout(control_values)
            remaining_controls = tuple(
                control for control in flexible_layout.control_params if control.name != PARAM_IMAGE_PRESET
            )
            return Layout(
                control_params=(dropdown, *remaining_controls),
                cond_inputs=flexible_layout.cond_inputs,
            )

        matched = next((p for p in self.presets if p.display_name == active_choice), self.presets[0])
        preset_count = len(matched.positions)
        cond_inputs = tuple(
            ConditioningInput.preset(
                i,
                position,
                expose_strength=self.flexible.expose_strength,
                default_strength=self.flexible.default_strength,
            )
            for i, position in enumerate(matched.positions)
        )
        controls = (
            dropdown,
            ControlParam.num_images_slider(preset_count, preset_count, hide=True),
        )
        return Layout(control_params=controls, cond_inputs=cond_inputs)


ImageConditioningConfig = FlexibleImageConfig | PresetCatalogImageConfig | HybridImageConfig


@dataclass(frozen=True)
class VideoConditioningConfig:
    """Single video input with optional strength + frame index."""

    default_strength: float = 1.0
    expose_strength: bool = True
    expose_frame_index: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.default_strength <= 1.0:
            raise ValueError(f"default_strength must be in [0.0, 1.0], got {self.default_strength}.")


@dataclass(frozen=True)
class MediaGenConditioningConfig:
    """Top-level conditioning surface for one runtime-parameters subclass.

      * `image` only → image input(s), no mode toggle.
      * `video` only → single video input, no mode toggle.
      * both → mode toggle; initial side from `default_mode`.

    At least one of `image`/`video` must be set.
    """

    image: ImageConditioningConfig | None = None
    video: VideoConditioningConfig | None = None
    default_mode: ConditioningMode = ConditioningMode.IMAGE

    def __post_init__(self) -> None:
        if self.image is None and self.video is None:
            raise ValueError("MediaGenConditioningConfig requires at least one of `image` or `video`.")

    def derive_layout(self, control_values: Mapping[str, Any]) -> Layout:
        image_config = self.image
        video_config = self.video

        # __post_init__ guarantees at least one side is configured.
        if image_config is None or video_config is None:
            if image_config is not None:
                allowed_modes = (ConditioningMode.IMAGE,)
            elif video_config is not None:
                allowed_modes = (ConditioningMode.VIDEO,)
            else:
                msg = "MediaGenConditioningConfig requires at least one of `image` or `video`."
                raise ValueError(msg)
        else:
            allowed_modes = (ConditioningMode.IMAGE, ConditioningMode.VIDEO)

        toggle = ControlParam.mode_toggle(self.default_mode, allowed_modes=allowed_modes)
        mode = self._resolve_mode(control_values.get(PARAM_MODE))
        if mode not in allowed_modes:
            mode = allowed_modes[0]

        if mode is ConditioningMode.VIDEO:
            if video_config is None:
                msg = "Video configuration is required when mode resolves to video."
                raise ValueError(msg)
            return Layout(control_params=(toggle,), cond_inputs=(ConditioningInput.video(video_config),))

        if image_config is None:
            msg = "Image configuration is required when mode resolves to image."
            raise ValueError(msg)

        image_layout = image_config.derive_layout(control_values)
        return Layout(
            control_params=(toggle, *image_layout.control_params),
            cond_inputs=image_layout.cond_inputs,
        )

    def _resolve_mode(self, value: Any) -> ConditioningMode:
        if isinstance(value, ConditioningMode):
            return value
        if value is None:
            return self.default_mode
        try:
            return ConditioningMode(str(value))
        except ValueError:
            return self.default_mode
