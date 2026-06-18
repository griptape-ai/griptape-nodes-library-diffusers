import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterList
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from modular_diffusion_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.generate_latent_parameters import (
    DiffusionPipelineGenerateLatentParameters,
)
from modular_diffusion_nodes_library.parameters.pipeline_parameters import (
    ModularDiffusionPipelineParameters,
)
from modular_diffusion_nodes_library.utils.connection_utils import delete_parameter_list_child_connections
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


class DiffusionPipelineGenerateLatentNode(
    ParameterConnectionPreservationMixin, SuccessFailureExecutionMixin, SuccessFailureNode
):
    STATIC_PARAMS: ClassVar = ["pipeline", "input_latent"]
    START_PARAMS: ClassVar = ["pipeline", "input_latent"]
    END_PARAMS: ClassVar = ["output_latent", "preview_image", "progress", "Status", "logs"]

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)

        self.pipe_params = ModularDiffusionPipelineParameters(self)
        self.pipe_params.add_input_parameters()

        self.latent_parameter = DiffusionPipelineGenerateLatentParameters(self)  # type: ignore[reportOptionalMemberAccess]
        self.latent_parameter.add_input_parameters()

        self.progress_bar_component = ProgressBarComponent(self)
        self.progress_bar_component.add_property_parameters()

        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()

        self.latent_parameter.add_output_parameters()
        self.latent_parameter.add_property_parameters()
        self.latent_parameter.add_additional_parameters()
        self._initializing = False
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
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return

        current_pipeline = self.get_parameter_value("pipeline")
        did_pipeline_change = False
        # Handle pipeline change detection before setting the value
        if parameter.name == "pipeline":
            value = normalize_diffusion_pipeline_value(
                value,
                node_name=self.name,
                raise_on_invalid=True,
            )
            did_pipeline_change = current_pipeline != value

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        saved_incoming = []
        saved_outgoing = []
        if did_pipeline_change:
            saved_incoming, saved_outgoing = self._save_connections()
            self.pipe_params.tear_down_runtime_parameters()
            # Strip user_defined params left from workflow load so the new runtime helper installs correctly.
            self._remove_user_defined_parameters()
            self.latent_parameter.add_or_remove_control_net_parameter(current_pipeline, value)

        self.pipe_params.after_value_set(parameter, value)

        if did_pipeline_change:
            start_params = DiffusionPipelineGenerateLatentNode.START_PARAMS
            end_params = DiffusionPipelineGenerateLatentNode.END_PARAMS
            is_control_net_pipeline = self.latent_parameter._is_control_net_pipeline(value)
            end_params = DiffusionPipelineGenerateLatentParameters.update_end_parameters(
                end_params, is_control_net_pipeline
            )
            excluded_params = {*start_params, *end_params}

            middle_elements = [
                element.name for element in self.root_ui_element._children if element.name not in excluded_params
            ]
            sorted_parameters = [*start_params, *middle_elements, *end_params]

            self.reorder_elements(sorted_parameters)

        if did_pipeline_change:
            self._restore_connections(saved_incoming, saved_outgoing)
            self._validate_pipeline(value)

    def _validate_pipeline(self, pipeline_value: Any) -> None:
        normalized_pipeline = normalize_diffusion_pipeline_value(
            pipeline_value,
            node_name=self.name,
            raise_on_invalid=True,
        )
        if normalized_pipeline is None:
            return None
        pipeline_class = normalized_pipeline.pipeline_name
        if not self.latent_parameter.validate_pipeline_class(pipeline_class):
            logger.warning(
                f"Pipeline '{pipeline_class}' is not compatible with this node. Consider using a compatible pipeline."
            )
        return None

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        self.pipe_params.runtime_parameters.after_value_set(parameter, value)
        self.latent_parameter.after_value_set(parameter, value)

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        During initialization, parameters are added normally.
        After initialization (dynamic mode), parameters are marked as user-defined
        for serialization and duplicates are prevented.
        """
        if self._initializing:
            super().add_parameter(param)
            return

        # Dynamic mode: prevent duplicates and mark as user-defined
        if not self.does_name_exist(param.name):
            param.user_defined = True
            super().add_parameter(param)

    def _remove_user_defined_parameters(self) -> None:
        """Remove user_defined parameters before the runtime helper reinstalls them. Does not clear `parameter_values`."""

        excluded_user_params = {
            DiffusionPipelineGenerateLatentParameters.CONTROL_NET_PARAM_NAME,
        }

        # Snapshot first; remove_parameter_element_by_name mutates the children list.
        targets = [
            child
            for child in list(self.root_ui_element._children)
            if isinstance(child, Parameter)
            and getattr(child, "user_defined", False)
            and child.name not in excluded_user_params
        ]
        # Parent removal does not cascade to ParameterList child connections; clean them up first.
        delete_parameter_list_child_connections(
            self,
            [target for target in targets if isinstance(target, ParameterList)],
        )
        for target in targets:
            self.remove_parameter_element_by_name(target.name)

    def preprocess(self) -> None:
        self.pipe_params.runtime_parameters.preprocess()
        self.progress_bar_component.reset()
        self.log_params.clear_logs()

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,  # noqa: ARG002
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "pipeline":
            self.pipe_params.runtime_parameters.remove_input_parameters()
            self.pipe_params.runtime_parameters.remove_output_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        result = self.pipe_params.validate_before_node_run()
        if result is not None:
            return result

        result = self.latent_parameter.validate_before_node_run()
        if result is not None:
            return result

        input_pipeline = self.get_parameter_value("pipeline")
        control_net_parameters = self.latent_parameter.get_control_net_parameters()
        if not control_net_parameters and self.latent_parameter._is_control_net_pipeline(input_pipeline):
            return [
                ValueError(
                    "Control net pipeline selected but no control net parameters provided. Please provide control net parameters for the pipeline."
                )
            ]
        elif not isinstance(control_net_parameters, dict) and self.latent_parameter._is_control_net_pipeline(
            input_pipeline
        ):
            return [
                ValueError(
                    f"Control net parameters should be provided as a dict input. Got: {control_net_parameters!r}"
                )
            ]

        return self.pipe_params.runtime_parameters.validate_before_node_run()

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        # HACK: `node.remove_parameter_element_by_name` does not remove connections so we need to use the retained mode request which does.  # noqa: FIX004
        # To avoid updating a ton of callers, we just override this method here.
        # TODO: Remove after https://github.com/griptape-ai/griptape-nodes/issues/2511
        element = self.get_element_by_name_and_type(element_name)
        if element:
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(parameter_name=element_name, node_name=self.name)
            )

    def process(self) -> AsyncResult:
        self.preprocess()
        self._clear_execution_status()
        pipeline_artifact = self.pipe_params.get_pipeline_artifact()
        pipe = self.pipe_params.get_pipeline()

        def generate() -> Any:
            pipeline_class = self.pipe_params.get_pipeline_class()
            pipe_kwargs = self.pipe_params.runtime_parameters._get_pipe_kwargs()
            with pipeline_artifact.activate(pipe, node_name=self.name) as active_pipe:
                return self.latent_parameter.process_pipeline(active_pipe, pipeline_class, pipe_kwargs)

        def work() -> Any:
            return self._run_with_status(
                generate,
                success_msg="Generation completed successfully.",
                failure_log="Diffusion Pipeline execution failed",
                logger=logger,
                on_error=cleanup_memory_caches,
            )

        yield work
