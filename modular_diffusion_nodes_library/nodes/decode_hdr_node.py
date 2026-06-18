import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import cv2  # type: ignore[reportMissingImports]
import numpy as np
from diffusers.pipelines.ltx2.export_utils import encode_hdr_tensor_to_mp4  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from PIL import Image

from modular_diffusion_nodes_library.nodes.vae_decoder import VaeDecodeNode
from modular_diffusion_nodes_library.utils.hdr_video_utils import ExrFrameWriteEvent, encode_linear_hdr_exr_sequence
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros
from modular_diffusion_nodes_library.utils.pillow_utils import pil_to_image_artifact

logger = logging.getLogger(__name__)

ToneMapFn = Callable[[np.ndarray], np.ndarray]
DEFAULT_TONE_MAPPING = "aces_filmic"
TONE_MAPPING_CHOICES = ["clip", "reinhard", "aces_filmic", "cv2_reinhard", "cv2_mantiuk"]


class ExrOutputTarget(NamedTuple):
    """Resolved EXR output destination."""

    directory: str
    stem: str


class DecodeHdrNode(VaeDecodeNode):
    """Decode an HDR or standard VAE latent with optional tone mapping and EXR export.

    Inherits pipeline management, latent validation, dynamic image/video output switching,
    and video publishing from VaeDecodeNode. For video pipelines that produce linear HDR
    frames (np.ndarray output), tone mapping is applied before encoding to MP4 and the
    raw EXR sequence can optionally be saved to disk. Standard image and video pipelines
    behave identically to VaeDecodeNode.
    """

    def _additional_parameters(self) -> None:
        self.add_parameter(
            Parameter(
                name="exr_output_folder",
                type="str",
                tooltip="File or folder path for saving the raw linear HDR EXR sequence. Leave empty to skip saving EXRs.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
                traits={
                    FileSystemPicker(
                        allow_files=True,
                        allow_directories=True,
                        multiple=False,
                        file_types=[".exr"],
                        initial_path=str(GriptapeNodes.ConfigManager().workspace_path),
                    )
                },
            )
        )
        self.add_parameter(
            Parameter(
                name="save_exr_as_half_float",
                default_value=True,
                type="bool",
                tooltip="Save EXR as float16 — 2.5x smaller files with negligible quality loss",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="tone_mapping",
                default_value=DEFAULT_TONE_MAPPING,
                type="str",
                tooltip="Tone mapping function applied to linear HDR frames before encoding to SDR MP4.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
                traits={Options(choices=TONE_MAPPING_CHOICES)},
            )
        )
        self.progress_bar_component = ProgressBarComponent(self)
        self.progress_bar_component.add_property_parameters()
        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()

    def _tail_parameter_names(self) -> list[str]:
        return ["progress", "logs"]

    def _decode(self) -> None:
        self.log_params.clear_logs()
        self.progress_bar_component.reset()
        super()._decode()

    def _encode_video_output(self, output: Any, dest_path: Path, fps: int) -> None:
        if not isinstance(output, np.ndarray):
            super()._encode_video_output(output, dest_path, fps)
            return

        frames = output[0]
        self._save_exr_if_requested(frames)
        tone_fn = self._get_hdr_tone_mapping_fn(self.get_parameter_value("tone_mapping"))
        encode_hdr_tensor_to_mp4(frames, str(dest_path), frame_rate=fps, tone_mapping_fn=tone_fn)

    def _handle_image_output(self, output: Any) -> None:
        if not isinstance(output, np.ndarray):
            super()._handle_image_output(output)
            return

        frame = output[0] if output.ndim == 4 else output  # (H, W, 3)
        self._save_exr_if_requested(frame[np.newaxis])  # wrap as (1, H, W, 3)

        tone_fn = self._get_hdr_tone_mapping_fn(self.get_parameter_value("tone_mapping"))
        linear = tone_fn(frame)
        srgb = self._apply_srgb_oetf(linear)
        pil_image = Image.fromarray((srgb * 255 + 0.5).clip(0, 255).astype(np.uint8), mode="RGB")
        image_artifact = pil_to_image_artifact(pil_image)
        self.set_parameter_value("output_image", image_artifact)
        self.parameter_output_values["output_image"] = image_artifact

    def _save_exr_if_requested(self, frames: np.ndarray) -> None:
        exr_path_value = self.get_parameter_value("exr_output_folder") or ""
        if not exr_path_value:
            return

        target = self._resolve_exr_output_path(expand_path_macros(exr_path_value))
        save_as_half_float = bool(self.get_parameter_value("save_exr_as_half_float"))
        total_frames = len(frames)
        self.progress_bar_component.initialize(total_frames)
        precision_name = "float16" if save_as_half_float else "float32"
        self.log_params.append_to_logs(
            f"Saving HDR EXR sequence ({total_frames} frame(s), {precision_name}) to "
            f"'{target.directory}' with stem '{target.stem}'...\n"
        )

        def on_frame_saved(event: ExrFrameWriteEvent) -> None:
            self.progress_bar_component.increment()
            frame_mb = DecodeHdrNode._bytes_to_mb(event.frame_bytes)
            total_mb = DecodeHdrNode._bytes_to_mb(event.total_bytes_written)
            self.log_params.append_to_logs(
                f"[{event.current_frame}/{event.total_frames}] Saved {event.output_path} "
                f"({frame_mb:.2f} MB, cumulative {total_mb:.2f} MB)\n"
            )

        encode_linear_hdr_exr_sequence(
            frames,
            target.directory,
            stem=target.stem,
            save_as_half_float=save_as_half_float,
            progress_callback=on_frame_saved,
        )
        logger.info("DecodeHdrNode: HDR EXR sequence saved to: %s (stem=%s)", target.directory, target.stem)

    @staticmethod
    def _resolve_exr_output_path(path_value: str) -> ExrOutputTarget:
        selected_path = Path(path_value).expanduser()

        if path_value.endswith(("/", "\\")):
            return ExrOutputTarget(str(selected_path), "frame")

        if selected_path.suffix.lower() == ".exr":
            return ExrOutputTarget(str(selected_path.parent), selected_path.stem or "frame")

        return ExrOutputTarget(str(selected_path), "frame")

    @staticmethod
    def _apply_srgb_oetf(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)

    @staticmethod
    def _bytes_to_mb(value_bytes: int) -> float:
        return value_bytes / (1024.0 * 1024.0)

    @staticmethod
    def _get_hdr_tone_mapping_fn(name: str | None) -> ToneMapFn:
        tone_name = name or DEFAULT_TONE_MAPPING
        if tone_name == "reinhard":
            return lambda x: x / (1.0 + x)
        if tone_name == "clip":
            return lambda x: np.clip(x, 0.0, 1.0)
        if tone_name == "cv2_reinhard":
            tonemapper = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0, light_adapt=0.0, color_adapt=0.0)
            return DecodeHdrNode._wrap_cv2_tonemapper(tonemapper)
        if tone_name == "cv2_mantiuk":
            tonemapper = cv2.createTonemapMantiuk(gamma=1.0, scale=0.7, saturation=1.0)
            return DecodeHdrNode._wrap_cv2_tonemapper(tonemapper)

        # Default: aces_filmic
        def aces_filmic(x: np.ndarray) -> np.ndarray:
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

        return aces_filmic

    @staticmethod
    def _wrap_cv2_tonemapper(tonemapper: Any) -> ToneMapFn:
        """Wrap an OpenCV tonemapper so its RGB↔BGR colorspace expectation is hidden from callers."""

        def apply(rgb: np.ndarray) -> np.ndarray:
            bgr = rgb[..., ::-1]
            tone_bgr = tonemapper.process(bgr.astype(np.float32))
            tone_rgb = tone_bgr[..., ::-1]
            return np.clip(tone_rgb, 0.0, 1.0)

        return apply
