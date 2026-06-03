import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode, NodeResolutionState
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    BasePipelineIdentity,
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.mixins.parameter_connection_preservation_mixin import (
    ParameterConnectionPreservationMixin,
)
from modular_diffusion_nodes_library.parameters.huggingface_pipeline_parameter import HuggingFacePipelineParameter
from modular_diffusion_nodes_library.parameters.pipeline_builder_parameters import (
    LatentDiffusionPipelineBuilderParameters,
)
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.lora_utils import LorasParameter
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was duplicated/copied from diffusers_nodes_library/common/nodes/diffusion_pipeline_builder_node.py.

# Additional postfix bits must be powers of two (1, 2, 4, 8, etc.) to ensure unique combinations
UNION_PRO_2_CONFIG_HASH_POSTFIX = 1  # 0001


class LatentDiffusionPipelineBuilderNode(ParameterConnectionPreservationMixin, ControlNode):
    STATIC_PARAMS: ClassVar = ["provider", "pipeline"]
    START_PARAMS: ClassVar = ["pipeline", "provider"]
    END_PARAMS: ClassVar = ["logs"]

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

        self.log_params.add_output_parameters()

        self._initializing = False
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

    def _get_config_hash_postfix(self) -> int:
        config_bits = 0
        controlnet_model = self.get_parameter_value("controlnet_model")
        if controlnet_model and controlnet_model.startswith("Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"):
            # Set the UNION_PRO_2_CONFIG_HASH_POSTFIX bit
            config_bits |= UNION_PRO_2_CONFIG_HASH_POSTFIX
        return config_bits

    @property
    def _config_hash(self) -> str:
        """Generate a hash for the current configuration to use as cache key."""
        identity = BasePipelineIdentity(
            pipeline_name=self.params.pipeline_type_parameters.pipeline_type_pipeline_params.pipeline_name,
            config_kwargs=self.params.get_config_kwargs(),
            loras=self.loras_params.get_loras(),
            optimization_kwargs=self.huggingface_pipeline_params.get_hf_pipeline_parameters(),
            torch_dtype="bfloat16",  # Currently hardcoded
            postfix_bits=self._get_config_hash_postfix(),
        )
        return identity.cache_key()

    def _build_pipeline_artifact_strict(self) -> DiffusionPipelineArtifact:
        pipeline_params = self.params.pipeline_type_parameters.pipeline_type_pipeline_params
        build_data_error: str | None = None
        try:
            build_data = pipeline_params.get_build_data()
        except Exception as e:
            build_data = {}
            build_data_error = (
                f"{self.name}: Failed to collect pipeline build data for "
                f"pipeline '{pipeline_params.pipeline_name}': {e}"
            )

        return DiffusionPipelineArtifact(
            pipeline_name=pipeline_params.pipeline_name,
            config_hash=self._config_hash,
            builder_module=pipeline_params.__class__.__module__,
            builder_class_name=pipeline_params.__class__.__name__,
            build_data=build_data,
            build_data_error=build_data_error,
            loras=self.loras_params.get_loras(),
            optimization_kwargs=self.optimization_kwargs,
            is_prequantized=pipeline_params.is_prequantized(),
            supports_layerwise_casting=pipeline_params.supports_layerwise_casting(),
            requires_device_map=pipeline_params.requires_device_map(),
        )

    def build_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        try:
            return self._build_pipeline_artifact_strict()
        except Exception as e:
            logger.warning("%s: Failed to build pipeline artifact due to error: %s", self.name, str(e))
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
        for serialization and duplicates are prevented.
        """
        if self._initializing:
            super().add_parameter(param)
            return

        # Dynamic mode: prevent duplicates and mark as user-defined
        if not self.does_name_exist(param.name):
            param.user_defined = True

            # Restore cached parameter properties using mixin method
            self.restore_cached_parameter_properties(param)

            super().add_parameter(param)

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
            value = normalize_diffusion_pipeline_value(value, node_name=self.name) or self.build_pipeline_artifact()

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
        self.log_params.append_to_logs("Building pipeline...\n")

        self.set_pipeline_artifact()
        pipeline_artifact = self.get_pipeline_artifact_or_raise()

        def work() -> Any:
            try:
                with self.log_params.append_profile_to_logs("Pipeline building/caching"):
                    return pipeline_artifact.get_or_build_pipeline(log_params=self.log_params)
            except Exception:
                logger.exception("%s: Diffusion Pipeline build failed", self.name)
                # Remove partial/corrupted pipeline from cache
                if pipeline_artifact.config_hash is not None:
                    model_cache.remove_pipeline(pipeline_artifact.config_hash)
                # Aggressive cleanup on failure
                cleanup_memory_caches()
                raise

        yield work

        self.log_params.append_to_logs("Pipeline building complete.\n")
