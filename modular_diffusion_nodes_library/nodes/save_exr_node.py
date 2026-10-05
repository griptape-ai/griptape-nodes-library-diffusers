from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from griptape_nodes.common.macro_parser import ParsedMacro
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file_sequence import build_versioned_sequence_destination
from griptape_nodes.retained_mode.events.project_events import MacroPath

from modular_diffusion_nodes_library.artifact_utils.float_media_artifact import FloatMediaArtifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.utils.hdr_video_utils import encode_float_rgb_exr_frame
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros, resolve_path_to_macro

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger("modular_diffusers_nodes_library")

DEFAULT_OUTPUT_FILE = "raw/frame.exr"


def prepare_exr_frames(media: FloatMediaArtifact) -> np.ndarray:
    """Return float media frames unchanged for EXR writing."""
    return media.to_frames()


def exr_output_folder(output_file: str) -> str:
    """Return the sequence folder as a macro path, under `{outputs}` unless `output_file` is absolute."""
    path = Path(output_file)
    if path.is_absolute():
        return path.parent.as_posix()
    if path.parent == Path("."):
        return "{outputs}"
    return f"{{outputs}}/{path.parent.as_posix()}"


def exr_sequence_macro_path(output_file: str, folder: str) -> MacroPath:
    """Return the sequence macro with the version in the folder and `####` frame numbers."""
    stem = Path(output_file).stem
    template = f"{{folder}}/{stem}_v{{_index:03}}/{stem}.####.exr"
    return MacroPath(ParsedMacro(template), {"folder": folder})


class SaveExrNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Write decoded float RGB media as an OpenEXR image sequence."""

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="media",
                type="FloatMediaArtifact",
                input_types=["FloatMediaArtifact"],
                tooltip="Float media to write unchanged to OpenEXR frames.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename=DEFAULT_OUTPUT_FILE,
            allowed_modes={ParameterMode.PROPERTY},
        )
        self._output_file.add_parameter()
        output_file_param = self.get_parameter_by_name("output_file")
        if output_file_param is not None:
            output_file_param.set_badge(
                variant="info",
                title="Output file",
                message=(
                    "Each run writes a new version folder: `<stem>_v001/<stem>.0001.exr`, `<stem>_v001/<stem>.0002.exr`, and so on.\n\n"
                    "A folder part, such as `renders/shot.exr`, writes frames into a subfolder of the project outputs folder.\n\n"
                    "The extension is ignored."
                ),
            )
        self.add_parameter(
            Parameter(
                name="save_as_half_float",
                default_value=True,
                type="bool",
                tooltip="Write float16 channels when enabled, or float32 channels when disabled.",
                allowed_modes={ParameterMode.PROPERTY},
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
        self._initializing = False

    def add_parameter(self, param: Parameter) -> None:
        # Saved workflows re-add these parameters on load; ignore them after init to avoid duplicates.
        if self._initializing:
            super().add_parameter(param)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != "output_file" or not isinstance(value, str) or not value:
            return

        macro_path = resolve_path_to_macro(value)
        if macro_path != value:
            self.set_parameter_value("output_file", macro_path, emit_change=False)

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        output_file = self._resolve_output_file()
        if "{" in output_file:
            errors.append(
                ValueError(
                    f"{self.name}: Attempted to expand output_file '{output_file}'. Failed because a macro in it could not be resolved."
                )
            )
        elif not Path(output_file).stem:
            errors.append(
                ValueError(f"{self.name}: output_file must name a file, such as 'frame.exr' or 'renders/frame.exr'.")
            )
        return errors or None

    def validate_in_execution_environment(self) -> list[Exception] | None:
        """Check the incoming media; it is held by the process that produced it, so only read it here."""
        media = self.get_parameter_value("media")
        if not isinstance(media, FloatMediaArtifact):
            return [ValueError(f"{self.name}: connect a FloatMediaArtifact to 'media'.")]
        return None

    def _resolve_output_file(self) -> str:
        value = self.get_parameter_value("output_file")
        if not isinstance(value, str) or not value.strip():
            value = DEFAULT_OUTPUT_FILE
        return expand_path_macros(value)

    def process(self) -> AsyncResult:
        self._clear_execution_status()
        self.progress_bar_component.reset()
        self.log_params.clear_logs()
        self.set_parameter_value("saved_paths", [])
        self.parameter_output_values["saved_paths"] = []
        output_file = self._resolve_output_file()

        def save() -> None:
            media: FloatMediaArtifact = self.get_parameter_value("media")
            frames = prepare_exr_frames(media)
            save_as_half_float = bool(self.get_parameter_value("save_as_half_float"))
            total_frames = len(frames)
            sequence_folder = expand_path_macros(exr_output_folder(output_file))
            sequence = build_versioned_sequence_destination(exr_sequence_macro_path(output_file, sequence_folder))
            self.progress_bar_component.initialize(total_frames)
            self.log_params.append_to_logs(
                f"Saving OpenEXR sequence ({total_frames} frame(s)) under '{sequence_folder}'...\n"
            )

            saved_paths: list[str] = []
            for index, frame in enumerate(frames, start=1):
                frame_bytes = encode_float_rgb_exr_frame(frame, save_as_half_float=save_as_half_float)
                saved_path = sequence.entry(index).write_bytes(frame_bytes).resolve()
                saved_paths.append(saved_path)
                self.progress_bar_component.increment()
                frame_mb = len(frame_bytes) / (1024.0 * 1024.0)
                self.log_params.append_to_logs(f"[{index}/{total_frames}] Saved {saved_path} ({frame_mb:.2f} MB)\n")

            self.set_parameter_value("saved_paths", saved_paths)
            self.parameter_output_values["saved_paths"] = saved_paths
            self.log_params.append_to_logs(f"Saved {len(saved_paths)} OpenEXR frame(s).\n")

        def work() -> None:
            self._run_with_status(
                save,
                success_msg="OpenEXR sequence saved successfully.",
                failure_log="OpenEXR sequence save failed",
                logger=logger,
            )

        yield work
