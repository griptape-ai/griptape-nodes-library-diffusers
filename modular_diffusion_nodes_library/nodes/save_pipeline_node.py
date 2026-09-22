from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.utils import resolve_workspace_path

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import save_baked_pipeline
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import serialize_pipeline_artifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")

_SAVE_CONFIG_ONLY = "Config Only"
_SAVE_FULL_PIPELINE = "Full Pipeline"
_SAVE_MODE_CHOICES = [_SAVE_CONFIG_ONLY, _SAVE_FULL_PIPELINE]

_RECIPE_DEFAULT_PATH = "{project_dir}/pipeline-config.json"
_BAKED_DEFAULT_PATH = "{project_dir}/baked-pipeline"


class SavePipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Save a pipeline configuration recipe (JSON) or a baked Diffusers repo."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                input_types=["Pipeline Config"],
                tooltip=(
                    "Pipeline to save. Config Only stores the config; "
                    "Full Pipeline stores fused/quantised weights via save_pretrained."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )
        mode_param = Parameter(
            name="save_mode",
            type="str",
            default_value=_SAVE_CONFIG_ONLY,
            traits={Options(choices=_SAVE_MODE_CHOICES)},
            tooltip="Config Only = lightweight config JSON. Full Pipeline = Diffusers repo with weights.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Save Mode"},
        )
        mode_param.set_badge(
            variant="help",
            title="Config Only vs Full Pipeline",
            message=(
                "**Config Only** — Saves a small JSON that records how to rebuild the pipeline "
                "(repo ids, LoRA paths, quantisation flags). Re-fuses/re-quantises on load.\n\n"
                "**Full Pipeline** — Calls `save_pretrained` into a folder, persisting fused LoRAs "
                "and quantised weights as a valid Diffusers repo. No re-fusing/re-quantising on load."
            ),
        )
        self.add_parameter(mode_param)

        self._file_path_param = FilePathParameter(
            self,
            file_types=[".json"],
            tooltip="Destination for the recipe JSON. Relative paths are resolved against the project directory.",
            display_name="File Path",
            allow_create=True,
            allowed_modes={ParameterMode.PROPERTY},
            default_value=_RECIPE_DEFAULT_PATH,
        )
        self._file_path_param.add_input_parameters()
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._file_path_param.on_after_value_set(parameter, value)

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        if initial_setup:
            return
        if param_name == "save_mode" and isinstance(value, str):
            self._apply_save_mode(value)

    def _apply_save_mode(self, mode: str) -> None:
        """Reshape the destination picker to match the selected save mode."""
        param = self.get_parameter_by_name("file_path")
        if mode == _SAVE_FULL_PIPELINE:
            self._file_path_param.set_picker_mode(allow_files=False, allow_directories=True, file_types=None)
            if param is not None:
                param.ui_options = {**param.ui_options, "display_name": "Output Folder"}
            self.set_parameter_value("file_path", _BAKED_DEFAULT_PATH, emit_change=False)
            return
        self._file_path_param.set_picker_mode(allow_files=True, allow_directories=False, file_types=[".json"])
        if param is not None:
            param.ui_options = {**param.ui_options, "display_name": "File Path"}
        self.set_parameter_value("file_path", _RECIPE_DEFAULT_PATH, emit_change=False)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        pipeline = normalize_diffusion_pipeline_value(self.get_parameter_value("pipeline"))
        if pipeline is None:
            exceptions.append(ValueError(f"Parameter 'pipeline' on node '{self.name}' must be a Pipeline Config."))
        elif self.get_parameter_value("save_mode") == _SAVE_FULL_PIPELINE:
            try:
                pipeline.check_bakeable()
            except ValueError as e:
                exceptions.append(e)

        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            exceptions.append(ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string."))

        if exceptions:
            return exceptions
        return None

    def process(self) -> None:
        self._clear_execution_status()

        def save() -> Path:
            pipeline = normalize_diffusion_pipeline_value(
                self.get_parameter_value("pipeline"), node_name=self.name, raise_on_invalid=True
            )
            if pipeline is None:
                raise ValueError(f"Parameter 'pipeline' on node '{self.name}' must be a Pipeline Config.")

            file_path = self.get_parameter_value("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                raise ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string.")

            expanded_path = expand_path_macros(file_path)
            workspace_path = GriptapeNodes.ConfigManager().workspace_path
            resolved_path = resolve_workspace_path(Path(expanded_path), workspace_path)

            if self.get_parameter_value("save_mode") == _SAVE_FULL_PIPELINE:
                pipeline.check_bakeable()
                pipe = pipeline.get_or_build_pipeline()
                save_baked_pipeline(pipe, pipeline, resolved_path)
                return resolved_path

            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
            temp_path.write_text(serialize_pipeline_artifact(pipeline), encoding="utf-8")
            temp_path.replace(resolved_path)
            return resolved_path

        if self.get_parameter_value("save_mode") == _SAVE_FULL_PIPELINE:
            success_msg = "Full pipeline saved successfully."
            failure_log = "Full pipeline save failed"
        else:
            success_msg = "Pipeline configuration saved successfully."
            failure_log = "Pipeline configuration save failed"

        resolved_path = self._run_with_status(save, success_msg=success_msg, failure_log=failure_log, logger=logger)
        if resolved_path is not None:
            self._set_status_results(was_successful=True, result_details=f"{success_msg}\n\nSaved to: {resolved_path}")
