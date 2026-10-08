import logging
from pathlib import Path
from typing import Any

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.utils.hdr_video_utils import (
    ExrFrameWriteEvent,
    encode_linear_hdr_exr_sequence,
    inverse_log_gamma,
    inverse_logc3,
)
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros, resolve_path_to_macro

logger = logging.getLogger("modular_diffusers_nodes_library")

TRANSFER_FUNCTION_CHOICES = ["None", "Log-Gamma", "ARRI LogC3"]
DEFAULT_TRANSFER_FUNCTION = "Log-Gamma"


def prepare_exr_frames(media: FloatMediaArtifact, transfer_function: str) -> np.ndarray:
    """Convert FloatMediaArtifact frames to float32 linear FHWC values for EXR writing."""
    frames = media.to_frames().astype(np.float32, copy=False)
    if transfer_function == "Log-Gamma":
        return inverse_log_gamma(frames)
    if transfer_function == "ARRI LogC3":
        return inverse_logc3(frames)
    return frames


class SaveExrNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Write decoded float RGB media as an OpenEXR image sequence."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media",
                type="FloatMediaArtifact",
                input_types=["FloatMediaArtifact"],
                tooltip="Float media from Decode HDR Latents.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="transfer_function",
                default_value=DEFAULT_TRANSFER_FUNCTION,
                type="str",
                traits={Options(choices=TRANSFER_FUNCTION_CHOICES)},
                tooltip="Select the encoded transfer function to convert to linear values before writing EXR files.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="output_folder",
                type="str",
                default_value=str(GriptapeNodes.ConfigManager().workspace_path),
                tooltip="Folder where numbered OpenEXR frames are written.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
                traits={
                    FileSystemPicker(
                        allow_files=False,
                        allow_directories=True,
                        multiple=False,
                        initial_path=str(GriptapeNodes.ConfigManager().workspace_path),
                    )
                },
            )
        )
        self.add_parameter(
            Parameter(
                name="file_stem",
                default_value="frame",
                type="str",
                tooltip="Filename prefix for the numbered EXR sequence; do not include a path or extension.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="save_as_half_float",
                default_value=True,
                type="bool",
                tooltip="Write float16 channels when enabled, or float32 channels when disabled.",
                allowed_modes={ParameterMode.PROPERTY},
                user_defined=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="saved_paths",
                type="list[str]",
                output_type="list[str]",
                tooltip="Filesystem paths of the OpenEXR frames written by this node.",
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )
        self.progress_bar_component = ProgressBarComponent(self)
        self.progress_bar_component.add_property_parameters()
        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != "output_folder" or not isinstance(value, str) or not value:
            return

        macro_path = resolve_path_to_macro(value)
        if macro_path != value:
            self.set_parameter_value("output_folder", macro_path, emit_change=False)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        media = self.get_parameter_value("media")
        if not isinstance(media, FloatMediaArtifact):
            errors.append(ValueError(f"{self.name}: connect a FloatMediaArtifact to 'media'."))

        output_folder = self.get_parameter_value("output_folder")
        if not isinstance(output_folder, str) or not output_folder.strip():
            errors.append(ValueError(f"{self.name}: output_folder must be a non-empty directory path."))

        file_stem = self.get_parameter_value("file_stem")
        if not isinstance(file_stem, str) or not file_stem.strip() or Path(file_stem).name != file_stem:
            errors.append(ValueError(f"{self.name}: file_stem must be a non-empty filename prefix without a path."))
        elif Path(file_stem).suffix:
            errors.append(ValueError(f"{self.name}: file_stem must not include a file extension."))
        return errors or None

    def process(self) -> None:
        self._clear_execution_status()
        self.progress_bar_component.reset()
        self.log_params.clear_logs()

        def save() -> None:
            media: FloatMediaArtifact = self.get_parameter_value("media")
            frames = prepare_exr_frames(media, str(self.get_parameter_value("transfer_function")))

            output_folder = Path(expand_path_macros(self.get_parameter_value("output_folder"))).expanduser()
            file_stem = self.get_parameter_value("file_stem")
            save_as_half_float = bool(self.get_parameter_value("save_as_half_float"))
            self.progress_bar_component.initialize(len(frames))
            self.log_params.append_to_logs(
                f"Saving OpenEXR sequence ({len(frames)} frame(s)) to '{output_folder}' with stem '{file_stem}'...\n"
            )

            def on_frame_saved(event: ExrFrameWriteEvent) -> None:
                self.progress_bar_component.increment()
                frame_mb = event.frame_bytes / (1024.0 * 1024.0)
                total_mb = event.total_bytes_written / (1024.0 * 1024.0)
                self.log_params.append_to_logs(
                    f"[{event.current_frame}/{event.total_frames}] Saved {event.output_path} "
                    f"({frame_mb:.2f} MB, cumulative {total_mb:.2f} MB)\n"
                )

            saved_paths = encode_linear_hdr_exr_sequence(
                frames,
                str(output_folder),
                stem=file_stem,
                save_as_half_float=save_as_half_float,
                progress_callback=on_frame_saved,
            )
            self.set_parameter_value("saved_paths", saved_paths)
            self.parameter_output_values["saved_paths"] = saved_paths
            self.log_params.append_to_logs(f"Saved {len(saved_paths)} OpenEXR frame(s).\n")

        self._run_with_status(
            save,
            success_msg="OpenEXR sequence saved successfully.",
            failure_log="OpenEXR sequence save failed",
            logger=logger,
        )
