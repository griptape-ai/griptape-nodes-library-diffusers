import logging
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from diffusers.utils.export_utils import encode_video  # type: ignore[reportMissingImports]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.diffusion_decoder_parameters import (
    DIFFUSION_DECODER_TYPE_MAP,
    BaseDiffusionDecoderParameters,
    create_diffusion_decoder_parameters,
)
from modular_diffusion_nodes_library.utils.pillow_utils import pil_to_image_artifact

logger = logging.getLogger("modular_diffusers_nodes_library")


class DiffusionDecoderNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    _COMMON_PARAMETERS = {"decoder_type", "latent_tensor", "seed", "enable_raw_output", "Status"}

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        self.decoder_params: BaseDiffusionDecoderParameters | None = None
        self._current_media_type: str | None = None
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="decoder_type",
                default_value="LTX-2.5",
                type="str",
                traits={Options(choices=list(DIFFUSION_DECODER_TYPE_MAP.keys()))},
                tooltip="Select a diffusion decoder model family.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.decoder_params = create_diffusion_decoder_parameters(self.get_parameter_value("decoder_type"), self)
        self.decoder_params.add_parameters()

        self.add_parameter(
            Parameter(
                name="latent_tensor",
                input_types=["LatentArtifact"],
                type="LatentArtifact",
                tooltip="Video or image latent tensor to decode with the selected diffusion decoder.",
                allowed_modes={ParameterMode.INPUT},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="seed",
                default_value=0,
                type="int",
                tooltip="Seed used by the diffusion decoder when it samples output noise; use -1 to let the pipeline choose the generator.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="enable_raw_output",
                default_value=False,
                type="bool",
                tooltip="Expose the unbounded float decoder output as raw media alongside a clipped display output.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self._update_output_parameters()
        self._initializing = False
        self._create_status_parameters()
        self._reorder_parameters()

    def add_parameter(self, param: Parameter) -> None:
        if not self._initializing and (
            self.decoder_params is None or param.name not in self.decoder_params._DYNAMIC_PARAMETERS
        ):
            return
        if param.name in {"output_image", "output_video", "raw_media", "enable_raw_output", "fps"}:
            param.user_defined = True
        if not self.does_name_exist(param.name):
            super().add_parameter(param)

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        if self.get_element_by_name_and_type(element_name):
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(parameter_name=element_name, node_name=self.name)
            )

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

        if param_name == "decoder_type":
            current_decoder_type = self.get_parameter_value("decoder_type")
            decoder_type_changed = current_decoder_type != value
        else:
            decoder_type_changed = False

        if param_name == "enable_raw_output":
            raw_output_changed = self.get_parameter_value("enable_raw_output") != value
        else:
            raw_output_changed = False

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if raw_output_changed:
            self._update_output_parameters()
            self._reorder_parameters()

        if decoder_type_changed:
            if self.decoder_params is not None:
                self.decoder_params.remove_parameters()
            self.decoder_params = create_diffusion_decoder_parameters(value, self)
            self.decoder_params.add_parameters()
            self._update_output_parameters()
            self._reorder_parameters()

    def _update_output_parameters(self) -> None:
        if self.decoder_params is None:
            return

        spec = self.decoder_params.spec
        raw_output = bool(self.get_parameter_value("enable_raw_output"))
        media_type_changed = spec.media_type != self._current_media_type
        raw_output_changed = raw_output != getattr(self, "_current_raw_output", None)
        if not media_type_changed and not raw_output_changed:
            return

        if media_type_changed:
            if self._current_media_type == "video":
                self.remove_parameter_element_by_name("output_video")
                self.remove_parameter_element_by_name("fps")
            elif self._current_media_type == "image":
                self.remove_parameter_element_by_name("output_image")

            if spec.media_type == "video":
                self.add_parameter(
                    Parameter(
                        name="fps",
                        default_value=spec.fps,
                        type="int",
                        tooltip="Frames per second for the decoded video.",
                        allowed_modes={ParameterMode.PROPERTY},
                        user_defined=True,
                        ui_options={"min": 1, "max": 120},
                    )
                )
                self.add_parameter(
                    Parameter(
                        name="output_video",
                        output_type="VideoUrlArtifact",
                        tooltip="Decoded video from the latent tensor.",
                        allowed_modes={ParameterMode.OUTPUT},
                        user_defined=True,
                        serializable=False,
                    )
                )
            else:
                self.add_parameter(
                    Parameter(
                        name="output_image",
                        output_type="ImageUrlArtifact",
                        tooltip="Decoded image from the latent tensor.",
                        allowed_modes={ParameterMode.OUTPUT},
                        user_defined=True,
                        serializable=False,
                    )
                )
            self._current_media_type = spec.media_type

        if raw_output:
            if raw_output_changed or not self.get_element_by_name_and_type("raw_media"):
                self.add_parameter(
                    Parameter(
                        name="raw_media",
                        type="FloatMediaArtifact",
                        output_type="FloatMediaArtifact",
                        tooltip="Unmodified floating-point RGB decoder output for downstream processing.",
                        allowed_modes={ParameterMode.OUTPUT},
                        user_defined=True,
                        serializable=False,
                        hide_property=True,
                    )
                )
        elif raw_output_changed:
            self.remove_parameter_element_by_name("raw_media")

        self._current_raw_output = raw_output

    def _reorder_parameters(self) -> None:
        parameter_names = [element.name for element in self.root_ui_element.children]
        if "latent_tensor" in parameter_names:
            parameter_names.remove("latent_tensor")
            parameter_names.insert(0, "latent_tensor")
        tail = [name for name in ("output_image", "output_video", "raw_media", "Status") if name in parameter_names]
        for name in tail:
            parameter_names.remove(name)
        if "fps" in parameter_names and ("output_video" in tail or "raw_media" in tail):
            parameter_names.remove("fps")
            parameter_names.append("fps")
        parameter_names.extend(tail)
        self.reorder_elements(parameter_names)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        latent = self.get_parameter_value("latent_tensor")
        if latent is None:
            errors.append(ValueError("Missing required 'latent_tensor' input."))
        elif not isinstance(latent, LatentArtifact):
            errors.append(ValueError("'latent_tensor' must be a LatentArtifact."))
        elif self.decoder_params is None:
            errors.append(RuntimeError("Diffusion decoder parameters are not initialized."))
        else:
            expected_rank = 5 if self.decoder_params.spec.media_type == "video" else 4
            if len(latent.shape) != expected_rank:
                media_label = self.decoder_params.spec.media_type
                errors.append(
                    ValueError(
                        f"'latent_tensor' must be a {expected_rank}-D {media_label} latent, got shape {latent.shape}."
                    )
                )

        if self.decoder_params is None:
            errors.append(RuntimeError("Diffusion decoder parameters are not initialized."))
        else:
            parameter_errors = self.decoder_params.validate_before_node_run()
            if parameter_errors:
                errors.extend(parameter_errors)

        if self.decoder_params is not None and self.decoder_params.spec.media_type == "video":
            fps = self.get_parameter_value("fps")
            try:
                fps_int = int(fps)
                if fps_int <= 0:
                    errors.append(ValueError(f"FPS must be positive, got {fps_int}."))
            except (TypeError, ValueError):
                errors.append(ValueError(f"FPS must be an integer, got {fps!r}."))

        return errors or None

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        yield lambda: self._run_with_status(
            self._decode,
            success_msg="Decoded successfully.",
            failure_log="Diffusion decoder failed",
            logger=logger,
        )

    def _decode(self) -> None:
        if self.decoder_params is None:
            raise RuntimeError(f"{self.name}: diffusion decoder parameters are not initialized.")

        latent: LatentArtifact = self.get_parameter_value("latent_tensor")
        if not isinstance(latent, LatentArtifact):
            raise ValueError(f"{self.name}: required 'latent_tensor' is not a LatentArtifact.")
        raw_output = bool(self.get_parameter_value("enable_raw_output"))
        output_type = "np" if raw_output else "pil"
        decoded = self.decoder_params.decode(
            latent,
            seed=int(self.get_parameter_value("seed") or 0),
            output_type=output_type,
        )
        if raw_output:
            if not isinstance(decoded, np.ndarray):
                raise TypeError(f"Decoder returned {type(decoded).__name__}; expected a NumPy array in raw mode.")
            self._publish_raw(decoded)
            display_output = np.clip(decoded, 0.0, 1.0)
        else:
            if not isinstance(decoded, list) or not decoded:
                raise TypeError(f"Decoder returned {type(decoded).__name__}; expected PIL images in normal mode.")
            if self.decoder_params.spec.media_type == "video":
                video_frames = decoded[0] if isinstance(decoded[0], list) else decoded
                if not video_frames or not isinstance(video_frames[0], Image.Image):
                    raise TypeError("Decoder returned invalid PIL video frames in normal mode.")
                display_output = video_frames
            else:
                if not isinstance(decoded[0], Image.Image):
                    raise TypeError("Decoder returned invalid PIL image output in normal mode.")
                display_output = decoded

        if self.decoder_params.spec.media_type == "video":
            self._publish_video(display_output)
        else:
            self._publish_image(display_output)

    def _publish_raw(self, decoded: np.ndarray) -> None:
        if decoded.shape[-1] != 3:
            raise ValueError(f"Decoder returned invalid raw RGB media with shape {decoded.shape}.")
        meta = {}
        if self.decoder_params.spec.media_type == "video":
            if decoded.ndim != 5 or decoded.shape[0] != 1:
                raise ValueError(f"Decoder returned invalid raw video frames with shape {decoded.shape}.")
            meta["fps"] = int(self.get_parameter_value("fps") or self.decoder_params.spec.fps)
            media = decoded
        else:
            if decoded.ndim != 4 or decoded.shape[0] != 1:
                raise ValueError(f"Decoder returned invalid raw image data with shape {decoded.shape}.")
            media = decoded[0]
        raw_media = FloatMediaArtifact(media, meta=meta)
        self.set_parameter_value("raw_media", raw_media)
        self.parameter_output_values["raw_media"] = raw_media

    def _publish_video(self, decoded: list[Image.Image] | np.ndarray) -> None:
        if isinstance(decoded, list):
            frames = np.stack([np.asarray(frame.convert("RGB")) for frame in decoded])
        else:
            frames = decoded[0] if decoded.ndim == 5 else decoded
            if frames.dtype != np.uint8:
                frames = (frames * 255.0 + 0.5).astype(np.uint8)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"Decoder returned invalid video frames with shape {frames.shape}.")
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file_obj:
            temp_path = Path(temp_file_obj.name)
        try:
            fps = int(self.get_parameter_value("fps") or self.decoder_params.spec.fps)
            encode_video(frames, fps, str(temp_path))
            dest = ProjectFileDestination.from_situation(filename="video.mp4", situation="save_node_output")
            saved = dest.write_bytes(temp_path.read_bytes())
            output_video = VideoUrlArtifact(saved.location)
            self.set_parameter_value("output_video", output_video)
            self.parameter_output_values["output_video"] = output_video
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _publish_image(self, decoded: list[Image.Image] | np.ndarray) -> None:
        frame = decoded[0] if isinstance(decoded, list) else (decoded[0] if decoded.ndim == 4 else decoded)
        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"Decoder returned invalid image data with shape {frame.shape}.")
            frame = (frame * 255.0 + 0.5).astype(np.uint8)
            image = Image.fromarray(frame, mode="RGB")
        else:
            image = frame.convert("RGB")
        image_artifact = pil_to_image_artifact(image)
        self.set_parameter_value("output_image", image_artifact)
        self.parameter_output_values["output_image"] = image_artifact
