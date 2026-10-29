from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import NodeResolutionState, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
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
from modular_diffusion_nodes_library.nodes.load_pipeline_optimization import (
    ReconciledOptimizationSettings,
    reconcile_optimization_settings,
    validated_optimization_overrides,
)
from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.utils.config_utils import get_workspace_path
from modular_diffusion_nodes_library.utils.path_macros import expand_path_macros

logger = logging.getLogger("modular_diffusers_nodes_library")

_OPTIMIZATION_OVERRIDES_PARAMETER = "_optimization_overrides"


class LoadPipelineNode(SuccessFailureExecutionMixin, SuccessFailureNode):
    """Load a pipeline from a YAML manifest."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._loaded_from_baked_dir = False
        self._source_pipeline: DiffusionPipelineArtifact | None = None
        self._is_synchronizing_optimization_parameters = False
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
        self.add_parameter(
            ParameterButton(
                name=_OPTIMIZATION_OVERRIDES_PARAMETER,
                default_value={},
                label="Use Recipe Optimization Settings",
                icon="list-restart",
                variant="secondary",
                full_width=True,
                on_click=self._on_use_recipe_optimization_settings,
                tooltip="Clear Load Pipeline overrides and use the optimization settings from the loaded recipe.",
                hide=True,
            )
        )
        self._create_status_parameters()

    def _optimization_parameters(self) -> dict[str, Parameter]:
        parameters: dict[str, Parameter] = {}
        for name in self.huggingface_pipeline_params.parameter_names:
            parameter = self.get_parameter_by_name(name)
            if parameter is None:
                raise RuntimeError(f"Load Pipeline optimization parameter is missing: {name}")
            parameters[name] = parameter
        return parameters

    def _validated_optimization_overrides(self, value: Any) -> dict[str, Any]:
        return validated_optimization_overrides(value, self._optimization_parameters())

    def _current_optimization_overrides(self) -> dict[str, Any]:
        return self._validated_optimization_overrides(self.get_parameter_value(_OPTIMIZATION_OVERRIDES_PARAMETER))

    def _store_optimization_overrides(self, overrides: dict[str, Any]) -> None:
        super().set_parameter_value(
            _OPTIMIZATION_OVERRIDES_PARAMETER,
            dict(overrides),
            emit_change=False,
        )
        self._update_optimization_reset_button_visibility()

    def _update_optimization_reset_button_visibility(self) -> None:
        if self._source_pipeline is not None and self._current_optimization_overrides():
            self.show_parameter_by_name(_OPTIMIZATION_OVERRIDES_PARAMETER)
            return
        self.hide_parameter_by_name(_OPTIMIZATION_OVERRIDES_PARAMETER)

    def _set_source_pipeline(self, pipeline: DiffusionPipelineArtifact | None) -> None:
        self._source_pipeline = pipeline
        self._update_optimization_reset_button_visibility()

    @contextmanager
    def _synchronizing_optimization_parameters(self) -> Iterator[None]:
        was_synchronizing_optimization_parameters = self._is_synchronizing_optimization_parameters
        self._is_synchronizing_optimization_parameters = True
        try:
            yield
        finally:
            self._is_synchronizing_optimization_parameters = was_synchronizing_optimization_parameters

    def _apply_reconciled_optimization_settings(self, settings: ReconciledOptimizationSettings) -> None:
        self._store_optimization_overrides(settings.overrides)
        with self._synchronizing_optimization_parameters():
            for parameter_name, value in settings.displayed_values.items():
                self.set_parameter_value(parameter_name, value)

    def _report_optimization_override_error(self, error: Exception) -> None:
        details = f"Could not restore Load Pipeline optimization overrides: {error}"
        logger.error("%s: %s", self.name, details)
        self._set_status_results(was_successful=False, result_details=details)

    def _on_use_recipe_optimization_settings(
        self,
        _button: Button,
        _details: ButtonDetailsMessagePayload,
    ) -> NodeMessageResult:
        if self._source_pipeline is None:
            return NodeMessageResult(
                success=False,
                details="Cannot use recipe optimization settings because no pipeline recipe is loaded.",
                altered_workflow_state=False,
            )

        try:
            settings = reconcile_optimization_settings(
                parameters=self._optimization_parameters(),
                recipe_values=self._source_pipeline.optimization_kwargs,
                overrides={},
            )
        except Exception as error:
            logger.exception("%s: Failed to prepare recipe optimization settings", self.name)
            details = f"Could not use recipe optimization settings: {error}"
            self._set_status_results(was_successful=False, result_details=details)
            return NodeMessageResult(success=False, details=details, altered_workflow_state=False)

        try:
            self._apply_reconciled_optimization_settings(settings)
            self._publish_pipeline(self._effective_pipeline())
        except Exception as error:
            logger.exception("%s: Failed to apply recipe optimization settings", self.name)
            details = f"Could not use recipe optimization settings: {error}"
            self._set_status_results(was_successful=False, result_details=details)
            return NodeMessageResult(success=False, details=details, altered_workflow_state=True)

        return NodeMessageResult(
            success=True,
            details="",
            altered_workflow_state=True,
        )

    def _record_user_optimization_override(self, param_name: str, value: Any) -> None:
        overrides = self._current_optimization_overrides()
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            raise RuntimeError(f"Load Pipeline optimization parameter is missing: {param_name}")
        if self._source_pipeline is None:
            overrides[param_name] = value
        elif value == self._source_pipeline.optimization_kwargs.get(param_name, parameter.default_value):
            overrides.pop(param_name, None)
        else:
            overrides[param_name] = value
        self._store_optimization_overrides(overrides)

    def _prepare_parameter_assignment(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool,
    ) -> tuple[Any, bool]:
        if param_name == _OPTIMIZATION_OVERRIDES_PARAMETER:
            try:
                return self._validated_optimization_overrides(value), True
            except (TypeError, ValueError) as error:
                if not initial_setup:
                    raise
                self._report_optimization_override_error(error)
                return value, False

        track_user_override = (
            param_name in self.huggingface_pipeline_params.parameter_names
            and not initial_setup
            and not self._is_synchronizing_optimization_parameters
        )
        if track_user_override:
            self._record_user_optimization_override(param_name, value)
        return value, True

    def _handle_parameter_assignment(self, param_name: str, *, initial_setup: bool) -> None:
        if initial_setup and param_name in {"file_path", "pipeline"}:
            self.state = NodeResolutionState.UNRESOLVED

        if param_name == _OPTIMIZATION_OVERRIDES_PARAMETER:
            self._update_optimization_reset_button_visibility()
            if self._source_pipeline is not None:
                try:
                    self._reconcile_optimization_parameters(self._source_pipeline)
                    self._publish_pipeline(self._effective_pipeline())
                except Exception as error:
                    self._report_optimization_override_error(error)

        if (
            initial_setup
            and param_name == "memory_optimization_strategy"
            and hasattr(self, "huggingface_pipeline_params")
        ):
            parameter = self.get_parameter_by_name(param_name)
            if parameter is not None:
                self.huggingface_pipeline_params.after_value_set(parameter, self.get_parameter_value(param_name))

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        # Restored output-only values bypass this setter, so file_path is the reliable signal to reload the manifest.
        discard_restored_pipeline = initial_setup and param_name == "pipeline"
        value, should_assign = self._prepare_parameter_assignment(
            param_name,
            value,
            initial_setup=initial_setup,
        )
        if not should_assign:
            return

        super().set_parameter_value(
            param_name,
            None if discard_restored_pipeline else value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        self._handle_parameter_assignment(param_name, initial_setup=initial_setup)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == _OPTIMIZATION_OVERRIDES_PARAMETER:
            return
        self._file_path_param.on_after_value_set(parameter, value)
        self.huggingface_pipeline_params.after_value_set(parameter, value)
        if parameter.name in self.huggingface_pipeline_params.parameter_names:
            if not self._is_synchronizing_optimization_parameters and self._source_pipeline is not None:
                self._publish_pipeline(self._effective_pipeline())
            return
        if parameter.name != "file_path":
            return
        # Macro rewrite may replace the path via a nested set; that call publishes.
        current = self.get_parameter_value("file_path")
        if current != value:
            return
        self._publish_from_committed_path()

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

    def _reconcile_optimization_parameters(self, pipeline: DiffusionPipelineArtifact) -> None:
        settings = reconcile_optimization_settings(
            parameters=self._optimization_parameters(),
            recipe_values=pipeline.optimization_kwargs,
            overrides=self._current_optimization_overrides(),
        )
        self._apply_reconciled_optimization_settings(settings)

    def _effective_pipeline(self) -> DiffusionPipelineArtifact:
        if self._source_pipeline is None:
            raise RuntimeError("No pipeline source has been loaded.")
        return self._source_pipeline.with_optimization_overrides(self._current_optimization_overrides())

    def _clear_pipeline_output(self) -> None:
        self._set_source_pipeline(None)
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
        self._set_source_pipeline(None)
        try:
            pipeline = self._load_pipeline_from_path()
        except Exception as error:
            logger.exception("%s: Pipeline load failed after path change", self.name)
            self._clear_pipeline_output_safely()
            self._set_status_results(was_successful=False, result_details=str(error))
            return

        try:
            self._reconcile_optimization_parameters(pipeline)
            self._set_source_pipeline(pipeline)
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
            self._set_source_pipeline(None)
            source_pipeline = self._load_pipeline_from_path()
            self._reconcile_optimization_parameters(source_pipeline)
            self._set_source_pipeline(source_pipeline)
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
