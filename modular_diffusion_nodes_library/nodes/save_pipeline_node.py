from __future__ import annotations

import logging
from enum import StrEnum
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
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import (
    plan_runtime_lora_copies,
    save_baked_pipeline,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import serialize_pipeline_artifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")


class SaveMode(StrEnum):
    CONFIG_ONLY = "Config Only"
    FULL_PIPELINE = "Full Pipeline"


_RECIPE_DEFAULT_PATH = "{project_dir}/config.yaml"
_BAKED_DEFAULT_PATH = "{project_dir}/baked-pipeline"


class SavePipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Save a pipeline configuration recipe or a baked Diffusers repo."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                input_types=["Pipeline Config"],
                tooltip=(
                    "Pipeline to save. Config Only stores how to rebuild it; Full Pipeline stores the prepared weights."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )
        mode_param = Parameter(
            name="save_mode",
            type="str",
            default_value=SaveMode.CONFIG_ONLY,
            traits={Options(choices=list(SaveMode))},
            tooltip="Config Only = lightweight rebuild recipe. Full Pipeline = folder with weights.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Save Mode"},
        )
        mode_param.set_badge(
            variant="help",
            title="Config Only vs Full Pipeline",
            message=(
                "**Config Only** — Saves a small YAML recipe that records how to rebuild the pipeline "
                "(model, fused and runtime LoRA paths, quantisation). Rebuilds, re-fuses, "
                "re-quantises, and re-attaches runtime LoRAs on load.\n\n"
                "**Full Pipeline** — Saves the weights into a folder, including fused LoRAs "
                "and quantised weights. Runtime LoRAs are copied into the folder and re-attached on load."
            ),
        )
        self.add_parameter(mode_param)

        self._file_path_param = FilePathParameter(
            self,
            file_types=[".yaml"],
            tooltip="Destination for the recipe YAML. Relative paths are resolved against the project directory.",
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
        if mode == SaveMode.FULL_PIPELINE:
            self._file_path_param.set_picker_mode(allow_files=False, allow_directories=True, file_types=None)
            if param is not None:
                param.ui_options = {**param.ui_options, "display_name": "Output Folder"}
            self.set_parameter_value("file_path", _BAKED_DEFAULT_PATH, emit_change=False)
            return
        self._file_path_param.set_picker_mode(allow_files=True, allow_directories=False, file_types=[".yaml"])
        if param is not None:
            param.ui_options = {**param.ui_options, "display_name": "File Path"}
        self.set_parameter_value("file_path", _RECIPE_DEFAULT_PATH, emit_change=False)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        pipeline = normalize_diffusion_pipeline_value(self.get_parameter_value("pipeline"))
        if pipeline is None:
            exceptions.append(ValueError(f"Parameter 'pipeline' on node '{self.name}' must be a Pipeline Config."))
        elif self.get_parameter_value("save_mode") == SaveMode.FULL_PIPELINE:
            try:
                pipeline.check_full_pipeline_bake_policy()
                plan_runtime_lora_copies(pipeline)
            except (FileNotFoundError, ValueError) as e:
                exceptions.append(e)

        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            exceptions.append(ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string."))
        elif (
            self.get_parameter_value("save_mode") == SaveMode.CONFIG_ONLY and Path(file_path).suffix.lower() != ".yaml"
        ):
            exceptions.append(ValueError(f"Parameter 'file_path' on node '{self.name}' must point to a .yaml file."))

        if exceptions:
            self._set_status_results(
                was_successful=False,
                result_details="\n".join(str(exception) for exception in exceptions),
            )
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

            if self.get_parameter_value("save_mode") == SaveMode.FULL_PIPELINE:
                pipeline.check_full_pipeline_bake_policy()
                runtime_lora_copies = plan_runtime_lora_copies(pipeline)
                pipe = pipeline.get_or_build_pipeline()
                save_baked_pipeline(
                    pipe,
                    pipeline,
                    resolved_path,
                    runtime_lora_copies=runtime_lora_copies,
                )
                return resolved_path

            if resolved_path.suffix.lower() != ".yaml":
                raise ValueError(f"Config Only destination must point to a .yaml file: {resolved_path}")
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
            temp_path.write_text(serialize_pipeline_artifact(pipeline), encoding="utf-8")
            temp_path.replace(resolved_path)
            return resolved_path

        if self.get_parameter_value("save_mode") == SaveMode.FULL_PIPELINE:
            success_msg = "Full pipeline saved successfully."
            failure_log = "Full pipeline save failed"
        else:
            success_msg = "Pipeline configuration saved successfully."
            failure_log = "Pipeline configuration save failed"

        resolved_path = self._run_with_status(save, success_msg=success_msg, failure_log=failure_log, logger=logger)
        if resolved_path is not None:
            self._set_status_results(was_successful=True, result_details=f"{success_msg}\n\nSaved to: {resolved_path}")
