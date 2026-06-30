import logging
from typing import Any, cast

from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from modular_diffusion_nodes_library.runtime_parameters.ltx2_runtime_parameters import (
    LTX2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.null_runtime_parameters import NullRuntimeParameters
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_params_registry import get_runtime_params_class

logger = logging.getLogger("modular_diffusers_nodes_library")


class ModularDiffusionPipelineParameters:
    def __init__(self, node: BaseNode):
        self._node: BaseNode = node
        self._runtime_parameters: DiffusionPipelineRuntimeParameters = NullRuntimeParameters(node)

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip="🤗 Diffusion pipeline produced by a Pipeline Builder.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

    def set_runtime_parameters(self, pipeline_artifact: DiffusionPipelineArtifact) -> None:
        pipeline_class = pipeline_artifact.pipeline_name

        runtime_cls = get_runtime_params_class(pipeline_class)
        if runtime_cls is None:
            msg = f"Unsupported pipeline class: {pipeline_class}"
            logger.error(msg)
            raise ValueError(msg)
        # Special case: LTX2 runtime-param class takes more than (node,) — it needs the artifact's metadata.
        if runtime_cls is LTX2PipelineRuntimeParameters:
            self._runtime_parameters = LTX2PipelineRuntimeParameters(self._node, pipeline_artifact.metadata)
            return
        self._runtime_parameters = runtime_cls(self._node)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != "pipeline":
            return

        if value is None:
            logger.warning("Value was None, ignoring...")
            return

        pipeline_class = self._get_pipeline_class_from_value(value)
        if pipeline_class is not None:
            self.set_runtime_parameters(value)

            self.runtime_parameters.add_input_parameters()
            self.runtime_parameters.add_output_parameters()

    def tear_down_runtime_parameters(self) -> None:
        """Remove current runtime parameters and rebind to NullRuntimeParameters."""
        self._runtime_parameters.remove_input_parameters()
        self._runtime_parameters.remove_output_parameters()
        self._runtime_parameters = NullRuntimeParameters(self._node)

    @property
    def runtime_parameters(self) -> DiffusionPipelineRuntimeParameters:
        return self._runtime_parameters

    def get_pipeline_class(self) -> str | None:
        pipeline_value = self._node.get_parameter_value("pipeline")
        if pipeline_value is None:
            return None
        return self._get_pipeline_class_from_value(pipeline_value)

    def get_pipeline(self) -> DiffusionPipeline:
        node_name = self._node.name
        pipeline_value = self._node.get_parameter_value("pipeline")
        if pipeline_value is None:
            raise ValueError(f"{node_name}: Pipeline value is None. Ensure pipeline parameter is set.")
        if not isinstance(pipeline_value, DiffusionPipelineArtifact):
            raise ValueError(
                f"{node_name}: Pipeline value must be DiffusionPipelineArtifact. "
                f"Got type '{type(pipeline_value).__name__}'."
            )
        pipeline = pipeline_value.get_or_build_pipeline()
        if pipeline is None:
            raise RuntimeError(f"{node_name}: Pipeline build returned None.")
        return cast(DiffusionPipeline, pipeline)

    def get_pipeline_artifact(self) -> DiffusionPipelineArtifact:
        node_name = self._node.name
        pipeline_value = self._node.get_parameter_value("pipeline")
        if not isinstance(pipeline_value, DiffusionPipelineArtifact):
            raise ValueError(
                f"{node_name}: Pipeline value must be DiffusionPipelineArtifact. "
                f"Got type '{type(pipeline_value).__name__}'."
            )
        return pipeline_value

    def validate_before_node_run(self) -> list[Exception] | None:
        node_name = self._node.name
        pipeline = self._node.get_parameter_value("pipeline")
        if pipeline is None:
            return [ValueError(f"{node_name}: Pipeline is required but not connected.")]

        if not isinstance(pipeline, DiffusionPipelineArtifact):
            return [
                ValueError(
                    f"{node_name}: Pipeline must be a DiffusionPipelineArtifact. Got type '{type(pipeline).__name__}'."
                )
            ]

        pipeline_class = pipeline.pipeline_name
        if get_driver_class(pipeline_class) is None:
            return [
                ValueError(f"{node_name}: Pipeline class '{pipeline_class}' is not supported for latent generation.")
            ]

        return None

    def _get_pipeline_class_from_value(self, pipeline_value: Any) -> str | None:
        if isinstance(pipeline_value, DiffusionPipelineArtifact):
            return pipeline_value.pipeline_name
        return None
