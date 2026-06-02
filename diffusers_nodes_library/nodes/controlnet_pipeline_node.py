import hashlib
import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

from diffusers_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from diffusers_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from diffusers_nodes_library.parameters.controlnet_pipeline_builder_parameters import (
    LatentDiffusionPipelineBuilderControlNetParameter,
)
from diffusers_nodes_library.parameters.pipelinetype_parameters import find_provider_for_pipeline_type
from diffusers_nodes_library.utils.huggingface_utils import model_cache
from diffusers_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


class ControlNetDiffusionPipelineBuilderNode(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip="Base 🤗 Diffusion pipeline to wrap with ControlNet conditioning.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="controlnet_pipeline",
                output_type="Pipeline Config",
                default_value=None,
                tooltip="🤗 Diffusion pipeline with ControlNet integrated. Connect to a Generate Latents node.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "ControlNet Pipeline"},
            )
        )

        self.controlnet_params = LatentDiffusionPipelineBuilderControlNetParameter(self)  # type: ignore[reportOptionalMemberAccess]
        self.controlnet_params.add_output_parameters()
        self.controlnet_params.add_input_parameters()

        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()
        self.set_controlnet_pipeline_artifact()

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

        if parameter.name in ("pipeline", "controlnet_pipeline"):
            value = normalize_diffusion_pipeline_value(value, node_name=self.name)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if parameter.name != "controlnet_pipeline":
            self.set_controlnet_pipeline_artifact()

        if parameter.name == "control_nets":
            self.controlnet_params.sync_output_parameters()

    def set_controlnet_pipeline_artifact(self) -> None:
        controlnet_artifact = self.build_pipeline_artifact()
        if controlnet_artifact is None:
            self.set_parameter_value("controlnet_pipeline", None)
            self.parameter_output_values["controlnet_pipeline"] = None
            return

        self.log_params.append_to_logs(f"Pipeline configuration hash: {controlnet_artifact.config_hash}\n")
        self.set_parameter_value("controlnet_pipeline", controlnet_artifact)
        self.parameter_output_values["controlnet_pipeline"] = controlnet_artifact

    def validate_before_node_run(self) -> list[Exception] | None:
        raw_input_pipeline = normalize_diffusion_pipeline_value(
            self.get_parameter_value("pipeline"), node_name=self.name
        )
        if isinstance(raw_input_pipeline, ControlNetDiffusionPipelineArtifact):
            return [
                ValueError(
                    f"{self.name}: Input pipeline appears to already have control net configuration. "
                    "Please provide a base pipeline without control net parameters."
                )
            ]

        input_artifact = self._get_input_pipeline_artifact()
        if input_artifact is None:
            return [ValueError(f"{self.name}: Missing required 'pipeline' input.")]

        pipeline_class = self._get_pipeline_class()
        if not pipeline_class:
            return [ValueError(f"{self.name}: Unable to determine pipeline class from input pipeline.")]

        provider = self._get_pipeline_provider(pipeline_class)
        if not provider:
            return [
                ValueError(f"{self.name}: Unable to determine pipeline provider for pipeline class: {pipeline_class}")
            ]

        result = self.controlnet_params.validate_before_node_run(provider)
        if result is not None:
            return result

        driver_class = get_driver_class(pipeline_class)
        if not driver_class:
            return [ValueError(f"{self.name}: No driver found for pipeline class: {pipeline_class}")]

        control_nets = self.controlnet_params.get_control_nets()
        if not control_nets:
            return [
                ValueError(
                    f"{self.name}: Control net configuration is required. Please provide at least one control net."
                )
            ]

        missing_model = [index for index, control_net in enumerate(control_nets) if "model" not in control_net]
        if missing_model:
            return [
                ValueError(
                    f"{self.name}: Control net(s) at index {missing_model} are missing a required 'model' field."
                )
            ]

        control_net_models: list[str] = [control_net["model"] for control_net in control_nets]
        if not driver_class.can_make_control_pipe_from_standard(control_net_models):
            return [
                ValueError(
                    f"{self.name}: Pipeline class {pipeline_class} does not support the provided control net configuration."
                )
            ]

        return None

    def _get_pipeline_class(self) -> str | None:
        input_pipeline = self._get_input_pipeline_artifact()
        if input_pipeline is None:
            return None
        return input_pipeline.pipeline_name

    def _get_pipeline_provider(self, pipeline_class: str) -> str | None:
        provider = find_provider_for_pipeline_type(pipeline_class)
        return provider

    def preprocess(self) -> None:
        self.log_params.clear_logs()

    def _get_input_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        input_pipeline_artefact = normalize_diffusion_pipeline_value(
            self.get_parameter_value("pipeline"), node_name=self.name
        )
        if isinstance(input_pipeline_artefact, ControlNetDiffusionPipelineArtifact):
            return None
        return input_pipeline_artefact

    def build_pipeline_artifact(self) -> ControlNetDiffusionPipelineArtifact | None:
        base_artifact = self._get_input_pipeline_artifact()
        if base_artifact is None:
            return None

        config_hash = self._config_hash()
        if config_hash is None:
            return None

        control_nets = self.controlnet_params.get_control_nets()
        control_net_models = [control_net["model"] for control_net in control_nets if "model" in control_net]
        return ControlNetDiffusionPipelineArtifact(
            base_artifact=base_artifact,
            controlnet_models=control_net_models,
            config_hash=config_hash,
        )

    def _config_hash(self) -> str | None:
        """Extend the input pipeline hash with controlnet configuration."""
        base_artifact = self._get_input_pipeline_artifact()
        if base_artifact is None:
            return None

        input_pipeline_hash = base_artifact.config_hash
        if not input_pipeline_hash:
            return None

        control_nets = self.controlnet_params.get_control_nets()
        control_net_models: list[str] = [control_net["model"] for control_net in control_nets if "model" in control_net]
        controlnet_data = {
            "control_nets": control_net_models,
        }

        controlnet_hash = hashlib.sha256(json.dumps(controlnet_data, sort_keys=True).encode()).hexdigest()

        return f"{input_pipeline_hash}--ControlNet--{controlnet_hash}"

    def process(self) -> AsyncResult:
        self.preprocess()
        self.log_params.append_to_logs("Building pipeline...\n")
        self.controlnet_params.sync_output_parameters()

        def work() -> Any:
            return self._build_pipeline()

        yield work
        self.log_params.append_to_logs("Pipeline building complete.\n")

    def _build_pipeline(self) -> Any:
        self.set_controlnet_pipeline_artifact()
        pipeline_artifact = self.get_parameter_value("controlnet_pipeline")
        if pipeline_artifact is None:
            raise ValueError(f"{self.name}: Failed to build controlnet pipeline artifact.")

        try:
            with self.log_params.append_profile_to_logs("Pipeline building/caching"):
                return pipeline_artifact.get_or_build_pipeline(log_params=self.log_params)
        except Exception:
            logger.exception("%s: Diffusion Pipeline build failed", self.name)
            # Remove partial/corrupted pipeline from cache
            model_cache.remove_pipeline(pipeline_artifact.config_hash)
            # Aggressive cleanup on failure
            cleanup_memory_caches()
            raise
