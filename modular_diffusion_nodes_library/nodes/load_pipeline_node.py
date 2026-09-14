from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import NodeResolutionState, SuccessFailureNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils import resolve_workspace_path

from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    deserialize_pipeline_artifact,
    format_pipeline_artifact_summary,
    validate_pipeline_recipe_dependencies,
)
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")


class LoadPipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Load a pipeline configuration recipe from a JSON file."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._file_path_param = FilePathParameter(
            self,
            file_types=[".json"],
            tooltip="Pipeline configuration JSON path. Relative paths are resolved against the project directory.",
            display_name="File Path",
            allowed_modes={ParameterMode.PROPERTY},
            default_value="{project_dir}/pipeline-config.json",
        )
        self._file_path_param.add_input_parameters()
        self.add_parameter(
            Parameter(
                name="pipeline",
                output_type="Pipeline Config",
                default_value=None,
                tooltip="Pipeline configuration reconstructed from the saved recipe.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self._create_status_parameters()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._file_path_param.on_after_value_set(parameter, value)

    @property
    def state(self) -> NodeResolutionState:
        """Always reload the recipe instead of trusting a serialized output artifact."""
        if self.get_parameter_value("pipeline") is not None:
            return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        self._state = new_state

    def validate_before_node_run(self) -> list[Exception] | None:
        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            return [ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string.")]
        return None

    def process(self) -> None:
        self._clear_execution_status()

        def load() -> None:
            file_path = self.get_parameter_value("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                raise ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string.")

            expanded_path = expand_path_macros(file_path)
            workspace_path = GriptapeNodes.ConfigManager().workspace_path
            resolved_path = resolve_workspace_path(Path(expanded_path), workspace_path)
            if not resolved_path.is_file():
                raise FileNotFoundError(f"Pipeline configuration file does not exist: {resolved_path}")

            pipeline = deserialize_pipeline_artifact(resolved_path.read_text(encoding="utf-8"))
            dependency_issues = validate_pipeline_recipe_dependencies(pipeline)
            if dependency_issues:
                details = "\n".join(f"- {issue}" for issue in dependency_issues)
                raise RuntimeError(f"Pipeline recipe dependencies are unavailable:\n{details}")
            self.set_parameter_value("pipeline", pipeline)
            self.parameter_output_values["pipeline"] = pipeline
            return pipeline

        pipeline = self._run_with_status(
            load,
            success_msg="Pipeline configuration loaded successfully.",
            failure_log="Pipeline configuration load failed",
            logger=logger,
        )
        if pipeline is not None:
            summary = format_pipeline_artifact_summary(pipeline)
            self._set_status_results(was_successful=True, result_details=f"Loaded successfully\n\n{summary}")
