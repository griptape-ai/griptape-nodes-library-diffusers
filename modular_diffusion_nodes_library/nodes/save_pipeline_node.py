from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils import resolve_workspace_path

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import serialize_pipeline_artifact
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")


class SavePipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Save a pipeline configuration recipe to a JSON file."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                input_types=["Pipeline Config"],
                tooltip="Pipeline configuration to serialize without model weights.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self._file_path_param = FilePathParameter(
            self,
            file_types=[".json"],
            tooltip="Destination JSON path. Relative paths are resolved against the project directory.",
            display_name="File Path",
            allow_create=True,
            allowed_modes={ParameterMode.PROPERTY},
            default_value="{project_dir}/pipeline-config.json",
        )
        self._file_path_param.add_input_parameters()
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._file_path_param.on_after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        pipeline = self.get_parameter_value("pipeline")
        if normalize_diffusion_pipeline_value(pipeline) is None:
            exceptions.append(ValueError(f"Parameter 'pipeline' on node '{self.name}' must be a Pipeline Config."))

        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            exceptions.append(ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string."))

        if exceptions:
            return exceptions
        return None

    def process(self) -> None:
        self._clear_execution_status()

        def save() -> None:
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
            resolved_path.parent.mkdir(parents=True, exist_ok=True)

            temp_path = resolved_path.with_suffix(f"{resolved_path.suffix}.tmp")
            temp_path.write_text(serialize_pipeline_artifact(pipeline), encoding="utf-8")
            temp_path.replace(resolved_path)

        self._run_with_status(
            save,
            success_msg="Pipeline configuration saved successfully.",
            failure_log="Pipeline configuration save failed",
            logger=logger,
        )