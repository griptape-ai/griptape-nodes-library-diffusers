from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import DecodeOutputType
from modular_diffusion_nodes_library.nodes.vae_decoder import VaeDecodeNode
from modular_diffusion_nodes_library.utils.hdr_video_utils import linear_to_srgb
from modular_diffusion_nodes_library.utils.pillow_utils import pil_to_image_artifact

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger("modular_diffusers_nodes_library")

# `type` rather than a plain assignment: the right-hand side is evaluated only when something
# introspects the alias, so naming a tone-map function does not import numpy.
type ToneMapFn = Callable[[np.ndarray], np.ndarray]
DEFAULT_TONE_MAPPING = "aces_filmic"
TONE_MAPPING_CHOICES = ["None", "clip", "reinhard", "aces_filmic", "cv2_reinhard", "cv2_mantiuk"]


class DecodeHdrNode(VaeDecodeNode):
    """Decode float media, expose the unmodified array, and publish an SDR preview."""

    def _additional_parameters(self) -> None:
        self.add_parameter(
            Parameter(
                name="raw_media",
                type="FloatMediaArtifact",
                output_type="FloatMediaArtifact",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip=(
                    "Unmodified floating-point RGB output from the decoder's NumPy path. "
                    "Range and color-space interpretation depend on the model pipeline."
                ),
                serializable=False,
                hide_property=True,
            )
        )
        tone_mapping_param = Parameter(
            name="tone_mapping",
            default_value=DEFAULT_TONE_MAPPING,
            type="str",
            tooltip=(
                "Tone mapping assumes decoded values are linear; this may not be true and depends on the model pipeline. "
                "Choose None to skip tone mapping."
            ),
            allowed_modes={ParameterMode.PROPERTY},
            user_defined=True,
            traits={Options(choices=TONE_MAPPING_CHOICES)},
        )
        tone_mapping_param.set_badge(
            variant="help",
            title="Tone mapping options",
            message=(
                "Tone mapping operators assume the decoded input is in linear color space. "
                "That may not be true; the output depends on the model pipeline.\n\n"
                "- ***None*** — Skip tone mapping and display the clipped decoder output.\n"
                "- ***clip*** — Discard values outside the standard 0-1 range\n"
                "- ***reinhard*** — Provide smooth compression\n"
                "- ***aces_filmic*** — Provide film-like highlight compression (default)\n"
                "- ***cv2_reinhard*** / ***cv2_mantiuk*** — Use OpenCV alternatives"
            ),
        )
        self.add_parameter(tone_mapping_param)

    def _get_decode_output_type(self) -> DecodeOutputType:
        return "np"

    def _encode_video_output(
        self,
        output: Any,
        dest_path: Path,
        fps: int,
        *,
        audio: Any = None,
        audio_sample_rate: int | None = None,
    ) -> None:
        import numpy as np
        from diffusers.pipelines.ltx2.export_utils import encode_hdr_tensor_to_mp4  # type: ignore[reportMissingImports]
        from diffusers.utils.export_utils import encode_video  # type: ignore[reportMissingImports]

        if not isinstance(output, np.ndarray):
            super()._encode_video_output(output, dest_path, fps, audio=audio, audio_sample_rate=audio_sample_rate)
            return

        frames = output[0] if output.ndim == 5 else output
        self.parameter_output_values["raw_media"] = FloatMediaArtifact(frames)
        tone_mapping = self.get_parameter_value("tone_mapping") or DEFAULT_TONE_MAPPING
        audio_output = None
        if audio is not None and audio_sample_rate is not None:
            audio_output = audio[0] if getattr(audio, "ndim", 0) == 3 else audio
        if audio_output is not None or tone_mapping == "None":
            # encode_video takes display-range floats in [0, 1] and muxes the soundtrack, which the HDR encoder cannot.
            encode_video(
                self._to_display_range(frames, tone_mapping),
                fps,
                str(dest_path),
                audio=audio_output,
                audio_sample_rate=audio_sample_rate,
            )
            return
        tone_fn = self._get_hdr_tone_mapping_fn(tone_mapping)
        encode_hdr_tensor_to_mp4(frames, str(dest_path), frame_rate=fps, tone_mapping_fn=tone_fn)

    def _handle_image_output(self, output: Any) -> None:
        import numpy as np
        from PIL import Image

        if not isinstance(output, np.ndarray):
            super()._handle_image_output(output)
            return

        frame = output[0] if output.ndim == 4 else output
        self.parameter_output_values["raw_media"] = FloatMediaArtifact(frame)
        tone_mapping = self.get_parameter_value("tone_mapping") or DEFAULT_TONE_MAPPING
        display = self._to_display_range(frame, tone_mapping)
        pil_image = Image.fromarray((display * 255 + 0.5).clip(0, 255).astype(np.uint8), mode="RGB")
        image_artifact = pil_to_image_artifact(pil_image)
        self.set_parameter_value("output_image", image_artifact)
        self.parameter_output_values["output_image"] = image_artifact

    def _to_display_range(self, values: np.ndarray, tone_mapping: str) -> np.ndarray:
        import numpy as np

        tone_fn = self._get_hdr_tone_mapping_fn(tone_mapping)
        if values.ndim == 4:
            # OpenCV tonemappers only accept a single (H, W, 3) frame.
            tone_mapped = np.stack([tone_fn(frame) for frame in values])
        else:
            tone_mapped = tone_fn(values)
        display = np.clip(tone_mapped, 0.0, 1.0)
        if tone_mapping != "None":
            display = linear_to_srgb(display)
        return display

    @staticmethod
    def _get_hdr_tone_mapping_fn(name: str | None) -> ToneMapFn:
        import numpy as np

        tone_name = name or DEFAULT_TONE_MAPPING
        if tone_name == "None":
            return lambda x: x
        if tone_name == "reinhard":
            return lambda x: x / (1.0 + x)
        if tone_name == "clip":
            return lambda x: np.clip(x, 0.0, 1.0)
        if tone_name == "cv2_reinhard":
            import cv2  # type: ignore[reportMissingImports]

            tonemapper = cv2.createTonemapReinhard(gamma=1.0, intensity=0.0, light_adapt=0.0, color_adapt=0.0)
            return DecodeHdrNode._wrap_cv2_tonemapper(tonemapper)
        if tone_name == "cv2_mantiuk":
            import cv2  # type: ignore[reportMissingImports]

            tonemapper = cv2.createTonemapMantiuk(gamma=1.0, scale=0.7, saturation=1.0)
            return DecodeHdrNode._wrap_cv2_tonemapper(tonemapper)

        def aces_filmic(x: np.ndarray) -> np.ndarray:
            a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
            return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

        return aces_filmic

    @staticmethod
    def _wrap_cv2_tonemapper(tonemapper: Any) -> ToneMapFn:
        """Wrap an OpenCV tonemapper so its RGB↔BGR colorspace expectation is hidden from callers."""
        import numpy as np

        def apply(rgb: np.ndarray) -> np.ndarray:
            bgr = rgb[..., ::-1]
            tone_bgr = tonemapper.process(bgr.astype(np.float32))
            tone_rgb = tone_bgr[..., ::-1]
            return np.clip(tone_rgb, 0.0, 1.0)

        return apply
