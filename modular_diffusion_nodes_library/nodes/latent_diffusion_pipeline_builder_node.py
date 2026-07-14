import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult, NodeResolutionState, SuccessFailureNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    BasePipelineIdentity,
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from modular_diffusion_nodes_library.mixins.success_failure_execution_mixin import SuccessFailureExecutionMixin
from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModelParamsError,
)
from modular_diffusion_nodes_library.parameters.pipeline_builder_parameters import (
    LatentDiffusionPipelineBuilderParameters,
)
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.lora_utils import LorasParameter
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was duplicated/copied from diffusers_nodes_library/common/nodes/diffusion_pipeline_builder_node.py.


class LatentDiffusionPipelineBuilderNode(
    ParameterConnectionPreservationMixin, SuccessFailureExecutionMixin, SuccessFailureNode
):
    STATIC_PARAMS: ClassVar = ["provider", "pipeline"]
    START_PARAMS: ClassVar = ["pipeline", "provider"]
    END_PARAMS: ClassVar = ["component_overrides", "loras", "Status", "logs"]

    def __init__(self, **kwargs) -> None:
        self._initializing = True
        super().__init__(**kwargs)
        self.params = LatentDiffusionPipelineBuilderParameters(self)
        self.huggingface_pipeline_params = HuggingFacePipelineParameter(self)
        self.log_params = LogParameter(self)

        self.params.add_output_parameters()
        self.params.add_input_parameters()
        self.huggingface_pipeline_params.add_input_parameters()

        self.loras_params = LorasParameter(self)
        self.loras_params.add_input_parameters()

        self._create_status_parameters()
        self.log_params.add_output_parameters()

        self._initializing = False
        self.params.refresh_component_override_ports(initial_setup=True)
        self.set_pipeline_artifact()

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to compute state based on pipeline's existence in model_cache, ensuring pipeline rebuild if missing."""
        pipeline_artifact = self.get_pipeline_artifact()
        if pipeline_artifact is None or pipeline_artifact.config_hash is None:
            return super().state
        if self._state == NodeResolutionState.RESOLVED and not model_cache.has_pipeline(pipeline_artifact.config_hash):
            logger.debug("Pipeline not found in cache, marking node as UNRESOLVED")
            return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        self._state = new_state

    def set_pipeline_artifact(self) -> None:
        pipeline_artifact = self.build_pipeline_artifact()
        if pipeline_artifact is None:
            self.set_parameter_value("pipeline", None)
            self.parameter_output_values["pipeline"] = None
        else:
            self.log_params.append_to_logs(f"Pipeline configuration hash: {pipeline_artifact.config_hash}\n")
            self.set_parameter_value("pipeline", pipeline_artifact)
            self.parameter_output_values["pipeline"] = pipeline_artifact

    @property
    def optimization_kwargs(self) -> dict[str, Any]:
        """Get optimization settings for the pipeline."""
        return self.huggingface_pipeline_params.get_hf_pipeline_parameters()

    @property
    def _config_hash(self) -> str:
        """Generate a hash for the current configuration to use as cache key."""
        identity = BasePipelineIdentity(
            pipeline_name=self.params.pipeline_type_parameters.pipeline_type_pipeline_params.pipeline_name,
            config_kwargs=self.params.get_config_kwargs(),
            loras=self.loras_params.get_loras(),
            optimization_kwargs=self.huggingface_pipeline_params.get_hf_pipeline_parameters(),
            torch_dtype="bfloat16",  # Currently hardcoded
        )
        return identity.cache_key()

    def _build_pipeline_artifact_strict(self) -> DiffusionPipelineArtifact:
        pipeline_params = self.params.pipeline_type_parameters.pipeline_type_pipeline_params
        build_data_error: str | None = None
        try:
            build_data = pipeline_params.get_build_data()
        except ModelParamsError as e:
            build_data = {}
            build_data_error = (
                f"{self.name}: Failed to collect pipeline build data for "
                f"pipeline '{pipeline_params.pipeline_name}': {e}"
            )

        component_overrides = self.params.component_override_params.get_component_overrides()
        override_is_quantized = self.params.component_override_params.has_quantized_overrides
        if component_overrides:
            build_data["_component_overrides"] = component_overrides

        return DiffusionPipelineArtifact(
            pipeline_name=pipeline_params.pipeline_name,
            config_hash=self._config_hash,
            builder_module=pipeline_params.__class__.__module__,
            builder_class_name=pipeline_params.__class__.__name__,
            build_data=build_data,
            build_data_error=build_data_error,
            loras=self.loras_params.get_loras(),
            optimization_kwargs=self.optimization_kwargs,
            is_prequantized=pipeline_params.is_prequantized() or override_is_quantized,
            supports_layerwise_casting=pipeline_params.supports_layerwise_casting() and not override_is_quantized,
            requires_device_map=pipeline_params.requires_device_map(),
        )

    def build_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        try:
            return self._build_pipeline_artifact_strict()
        except Exception as e:  # noqa: BLE001
            logger.warning("%s: Failed to build pipeline artifact: %s: %s", self.name, type(e).__name__, e)
            return None

    def get_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        pipeline_value = self.get_parameter_value("pipeline")
        if isinstance(pipeline_value, DiffusionPipelineArtifact):
            return pipeline_value

        return self.build_pipeline_artifact()

    def get_pipeline_artifact_or_raise(self) -> DiffusionPipelineArtifact:
        pipeline_value = self.get_parameter_value("pipeline")
        if isinstance(pipeline_value, DiffusionPipelineArtifact):
            return pipeline_value

        return self._build_pipeline_artifact_strict()

    def add_parameter(self, param: Parameter) -> None:
        """Add a parameter to the node.

        During initialization, parameters are added normally.
        After initialization (dynamic mode), parameters are marked as user-defined
        for serialization and duplicates are prevented — unless the parent group
        manages its own serialization (see ``_parent_manages_own_serialization``).
        """
        if self._initializing:
            super().add_parameter(param)
            return

        # Dynamic mode: prevent duplicates and mark as user-defined
        if not self.does_name_exist(param.name):
            if not self._parent_manages_own_serialization(param):
                param.user_defined = True

            # Restore cached parameter properties using mixin method
            self.restore_cached_parameter_properties(param)

            super().add_parameter(param)

    def _parent_manages_own_serialization(self, param: Parameter) -> bool:
        """Return True if param's parent group has user_defined=False.

        Such groups reconstruct their children on load, so serializing the children
        would create duplicates (e.g., component_transformer_1).
        """
        if param.parent_element_name is None:
            return False
        parent_group = self.get_group_by_name_or_element_id(param.parent_element_name)
        return parent_group is not None and not parent_group.user_defined

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

        if parameter.name == "pipeline":
            value = normalize_diffusion_pipeline_value(value, node_name=self.name)

        self.params.before_value_set(parameter, value)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        self.params.after_value_set(parameter, value)
        self.huggingface_pipeline_params.after_value_set(parameter, value)
        if parameter.name != "pipeline":
            self.set_pipeline_artifact()

    def validate_before_node_run(self) -> list[Exception] | None:
        result = self.params.pipeline_type_parameters.pipeline_type_pipeline_params.validate_before_node_run()
        if result is not None:
            return result

        try:
            self.get_pipeline_artifact_or_raise()
        except Exception as e:
            return [ValueError(f"Failed to build pipeline artifact for node_name='{self.name}': {e}")]

        return None

    def preprocess(self) -> None:
        self.log_params.clear_logs()

    def process(self) -> AsyncResult:
        self.preprocess()
        self._clear_execution_status()
        self.log_params.append_to_logs("Building pipeline...\n")

        yield self._build_pipeline

        if self._execution_succeeded:
            self.log_params.append_to_logs("Pipeline building complete.\n")

    def _build_pipeline(self) -> Any:
        self.set_pipeline_artifact()
        pipeline_artifact = self.get_parameter_value("pipeline")

        def build() -> Any:
            with self.log_params.append_profile_to_logs("Pipeline building/caching"):
                return pipeline_artifact.get_or_build_pipeline(log_params=self.log_params)

        def cleanup() -> None:
            self.log_params.append_to_logs("Pipeline building failed.\n")
            if pipeline_artifact.config_hash is not None:
                model_cache.remove_pipeline(pipeline_artifact.config_hash)
            cleanup_memory_caches()

        return self._run_with_status(
            build,
            success_msg="Pipeline built successfully.",
            failure_log="Diffusion Pipeline build failed",
            logger=logger,
            on_error=cleanup,
        )
