import logging
from collections.abc import Callable
from typing import Any

import cv2  # type: ignore[reportMissingImports]
import numpy as np
from diffusers.pipelines.ltx2.export_utils import encode_hdr_tensor_to_mp4  # type: ignore[reportMissingImports]
from diffusers.utils.export_utils import encode_video  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import DecodeOutputType
from modular_diffusion_nodes_library.nodes.vae_decoder import VaeDecodeNode
from modular_diffusion_nodes_library.utils.pillow_utils import pil_to_image_artifact

logger = logging.getLogger("modular_diffusers_nodes_library")

ToneMapFn = Callable[[np.ndarray], np.ndarray]
DEFAULT_TONE_MAPPING = "aces_filmic"
TONE_MAPPING_CHOICES = ["clip", "reinhard", "aces_filmic", "cv2_reinhard", "cv2_mantiuk"]


class DecodeHdrNode(VaeDecodeNode):
    """Decode float media, expose the unmodified array, and publish an SDR preview."""

    def _additional_parameters(self) -> None:
        self.add_parameter(
            Parameter(
                name="raw_media",
                type="FloatMediaArtifact",
                output_type="FloatMediaArtifact",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Unmodified float RGB media decoded from the selected pipeline.",
                serializable=False,
                hide_property=True,
            )
        )
        tone_mapping_param = Parameter(
            name="tone_mapping",
            default_value=DEFAULT_TONE_MAPPING,
            type="str",
            tooltip="Tone mapping function applied to decoded float frames before encoding to the SDR preview.",
            allowed_modes={ParameterMode.PROPERTY},
            user_defined=True,
            traits={Options(choices=TONE_MAPPING_CHOICES)},
        )
        tone_mapping_param.set_badge(
            variant="help",
            title="Tone mapping options",
            message=(
                "Select an algorithm to compress HDR media for viewing on standard screens:\n\n"
                "- ***clip*** — Discards values outside the standard 0-1 range\n"
                "- ***reinhard*** — Provides smooth, natural-looking compression\n"
                "- ***aces_filmic*** — Offers cinematic contrast with film-like highlights (default)\n"
                "- ***cv2_reinhard*** / ***cv2_mantiuk*** — OpenCV alternatives similar to Reinhard"
            ),
        )
        self.add_parameter(tone_mapping_param)

    def _get_decode_output_type(self) -> DecodeOutputType:
        return "np"

    def _encode_video_output(
        self,
        output: Any,
        dest_path: Any,
        fps: int,
        *,
        audio: Any = None,
        audio_sample_rate: int | None = None,
    ) -> None:
        if not isinstance(output, np.ndarray):
            super()._encode_video_output(output, dest_path, fps, audio=audio, audio_sample_rate=audio_sample_rate)
            return

        frames = output[0] if output.ndim == 5 else output
        self.parameter_output_values["raw_media"] = FloatMediaArtifact(frames)
        tone_fn = self._get_hdr_tone_mapping_fn(self.get_parameter_value("tone_mapping"))
        if audio is not None and audio_sample_rate is not None:
            tone_mapped = np.clip(tone_fn(frames), 0.0, 1.0)
            srgb_frames = self._apply_srgb_oetf(tone_mapped)
            audio_output = audio[0] if getattr(audio, "ndim", 0) == 3 else audio
            encode_video(
                srgb_frames,
                fps,
                str(dest_path),
                audio=audio_output,
                audio_sample_rate=audio_sample_rate,
            )
            return
        encode_hdr_tensor_to_mp4(frames, str(dest_path), frame_rate=fps, tone_mapping_fn=tone_fn)

    def _handle_image_output(self, output: Any) -> None:
        if not isinstance(output, np.ndarray):
            super()._handle_image_output(output)
            return

        frame = output[0] if output.ndim == 4 else output
        self.parameter_output_values["raw_media"] = FloatMediaArtifact(frame)
        tone_fn = self._get_hdr_tone_mapping_fn(self.get_parameter_value("tone_mapping"))
        linear = np.clip(tone_fn(frame), 0.0, 1.0)
        srgb = self._apply_srgb_oetf(linear)
        pil_image = Image.fromarray((srgb * 255 + 0.5).clip(0, 255).astype(np.uint8), mode="RGB")
        image_artifact = pil_to_image_artifact(pil_image)
        self.set_parameter_value("output_image", image_artifact)
        self.parameter_output_values["output_image"] = image_artifact

    @staticmethod
    def _apply_srgb_oetf(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)

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
