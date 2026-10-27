from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import NodeResolutionState, SuccessFailureNode
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils import resolve_workspace_path

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import (
    BAKED_CONFIG_FILENAME,
    detect_baked_layout,
    make_baked_artifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    deserialize_pipeline_artifact,
    format_baked_pipeline_summary,
    format_pipeline_artifact_summary,
    validate_pipeline_recipe_dependencies,
)
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.utils.config_utils import get_workspace_path
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")


class LoadPipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Load a pipeline from a YAML manifest."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loaded_from_baked_dir = False
        self._source_pipeline: DiffusionPipelineArtifact | None = None
        self._setting_optimization_parameters = False
        self._file_path_param = FilePathParameter(
            self,
            file_types=[".yaml"],
            tooltip=(
                "A Config Only recipe YAML or baked-config.yaml manifest. "
                "Relative paths are resolved against the project directory."
            ),
            display_name="File Path",
            allowed_modes={ParameterMode.PROPERTY},
            default_value="{project_dir}/config.yaml",
        )
        self._file_path_param.add_input_parameters()
        self._file_path_param.set_picker_mode(allow_files=True, allow_directories=False, file_types=[".yaml"])
        self.add_parameter(
            Parameter(
                name="pipeline",
                output_type="Pipeline Config",
                default_value=None,
                tooltip="Pipeline configuration reconstructed from the selected recipe or baked directory.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.huggingface_pipeline_params = HuggingFacePipelineParameter(self, include_quantization=False)
        self.huggingface_pipeline_params.add_input_parameters()
        self._create_status_parameters()

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
        if (
            initial_setup
            and param_name == "memory_optimization_strategy"
            and hasattr(self, "huggingface_pipeline_params")
        ):
            parameter = self.get_parameter_by_name(param_name)
            if parameter is not None:
                self.huggingface_pipeline_params.after_value_set(parameter, self.get_parameter_value(param_name))

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._file_path_param.on_after_value_set(parameter, value)
        self.huggingface_pipeline_params.after_value_set(parameter, value)
        if parameter.name in self.huggingface_pipeline_params.parameter_names:
            if not self._setting_optimization_parameters and self._source_pipeline is not None:
                self._publish_pipeline(self._effective_pipeline())
            return
        if parameter.name != "file_path":
            return
        # Macro rewrite may replace the path via a nested set; that call publishes.
        current = self.get_parameter_value("file_path")
        if current != value:
            return
        self._publish_from_committed_path()

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
        if Path(file_path).suffix.lower() != ".yaml":
            return [ValueError(f"Parameter 'file_path' on node '{self.name}' must point to a .yaml file.")]
        return None

    def _publish_pipeline(self, pipeline: DiffusionPipelineArtifact | None) -> None:
        # The request unresolves connected nodes and pushes the artifact into
        # them so parameters derived from it are rebuilt.
        result = GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="pipeline",
                node_name=self.name,
                value=pipeline,  # pyright: ignore[reportArgumentType]
            )
        )
        if result.failed():
            raise RuntimeError(
                f"Attempted to publish pipeline on node '{self.name}'. "
                f"Failed because the engine rejected the value: {result.result_details}"
            )
        self.parameter_output_values["pipeline"] = pipeline

    def _load_pipeline_from_path(self) -> DiffusionPipelineArtifact:
        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError(f"Parameter 'file_path' on node '{self.name}' must be a non-empty string.")

        expanded_path = expand_path_macros(file_path)
        workspace_path = get_workspace_path()
        resolved_path = resolve_workspace_path(Path(expanded_path), workspace_path)

        if resolved_path.suffix.lower() != ".yaml":
            raise ValueError(f"Pipeline source must point to a .yaml file: {resolved_path}")

        if resolved_path.is_dir():
            raise ValueError(f"Pipeline source must point to a .yaml file, not a directory: {resolved_path}")

        if not resolved_path.is_file():
            raise FileNotFoundError(f"Pipeline manifest does not exist: {resolved_path}")

        if resolved_path.name == BAKED_CONFIG_FILENAME and detect_baked_layout(resolved_path.parent) is not None:
            self._loaded_from_baked_dir = True
            return make_baked_artifact(resolved_path.parent)

        self._loaded_from_baked_dir = False
        pipeline = deserialize_pipeline_artifact(resolved_path.read_text(encoding="utf-8"))
        dependency_issues = validate_pipeline_recipe_dependencies(pipeline)
        if dependency_issues:
            details = "\n".join(f"- {issue}" for issue in dependency_issues)
            raise RuntimeError(f"Pipeline recipe dependencies are unavailable:\n{details}")
        return pipeline

    def _set_optimization_parameters_from_pipeline(self, pipeline: DiffusionPipelineArtifact) -> None:
        saved_parameters = pipeline.optimization_kwargs
        self._setting_optimization_parameters = True
        try:
            for parameter_name in self.huggingface_pipeline_params.parameter_names:
                parameter = self.get_parameter_by_name(parameter_name)
                if parameter is None:
                    raise RuntimeError(f"Load Pipeline optimization parameter is missing: {parameter_name}")
                value = saved_parameters.get(parameter_name, parameter.default_value)
                self.set_parameter_value(parameter_name, value, emit_change=False)
        finally:
            self._setting_optimization_parameters = False

    def _effective_pipeline(self) -> DiffusionPipelineArtifact:
        if self._source_pipeline is None:
            raise RuntimeError("No pipeline source has been loaded.")
        return self._source_pipeline.with_optimization_overrides(
            self.huggingface_pipeline_params.get_hf_pipeline_parameters()
        )

    def _clear_pipeline_output(self) -> None:
        self._source_pipeline = None
        self._publish_pipeline(None)

    def _clear_pipeline_output_safely(self) -> None:
        try:
            self._clear_pipeline_output()
        except Exception:
            logger.exception("%s: Failed to clear pipeline output", self.name)

    def _success_status_details(self, pipeline: DiffusionPipelineArtifact) -> str:
        if self._loaded_from_baked_dir:
            lead_in = "Loaded baked pipeline"
            body = format_baked_pipeline_summary(pipeline)
        else:
            lead_in = "Loaded successfully"
            body = format_pipeline_artifact_summary(pipeline)
        return f"{lead_in}\n\n{body}"

    def _publish_from_committed_path(self) -> None:
        """Load and publish when the editor commits a path. Never reject the path change."""
        self._loaded_from_baked_dir = False
        self._source_pipeline = None
        try:
            pipeline = self._load_pipeline_from_path()
        except Exception as error:
            logger.exception("%s: Pipeline load failed after path change", self.name)
            self._clear_pipeline_output_safely()
            self._set_status_results(was_successful=False, result_details=str(error))
            return

        try:
            self._set_optimization_parameters_from_pipeline(pipeline)
            self._source_pipeline = pipeline
            pipeline = self._effective_pipeline()
            self._publish_pipeline(pipeline)
        except Exception as error:
            logger.exception("%s: Failed to publish pipeline after path change", self.name)
            self._clear_pipeline_output_safely()
            self._set_status_results(was_successful=False, result_details=str(error))
            return

        self._set_status_results(was_successful=True, result_details=self._success_status_details(pipeline))

    def process(self) -> None:
        self._clear_execution_status()
        self._loaded_from_baked_dir = False

        def load() -> DiffusionPipelineArtifact:
            self._source_pipeline = None
            self._source_pipeline = self._load_pipeline_from_path()
            pipeline = self._effective_pipeline()
            self._publish_pipeline(pipeline)
            return pipeline

        pipeline = self._run_with_status(
            load,
            success_msg="Pipeline loaded successfully.",
            failure_log="Pipeline load failed",
            logger=logger,
            on_error=self._clear_pipeline_output_safely,
        )
        if pipeline is None:
            return

        self._set_status_results(was_successful=True, result_details=self._success_status_details(pipeline))
