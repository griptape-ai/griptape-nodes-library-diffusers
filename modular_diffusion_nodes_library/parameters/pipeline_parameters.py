import logging
from typing import Any, cast

from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from modular_diffusion_nodes_library.runtime_parameters.flux2_runtime_parameters import (
    Flux2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux_fill_runtime_parameters import (
    FluxFillPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux_runtime_parameters import (
    FluxPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.ltx2_runtime_parameters import (
    LTX2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.ltx_runtime_parameters import (
    LTXPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.qwen_edit_runtime_parameters import (
    QwenEditPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.qwen_runtime_parameters import (
    QwenPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.stable_diffusion_3_runtime_parameters import (
    StableDiffusion3PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.stable_diffusion_xl_runtime_parameters import (
    StableDiffusionXLPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_i2v_runtime_parameters import (
    WanImageToVideoPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_runtime_parameters import (
    WanPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.z_image_runtime_parameters import (
    ZImagePipelineRuntimeParameters,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class ModularDiffusionPipelineParameters:
    def __init__(self, node: BaseNode):
        self._node: BaseNode = node
        self._runtime_parameters: DiffusionPipelineRuntimeParameters
        self.set_runtime_parameters(DiffusionPipelineArtifact(pipeline_name="FluxPipeline"))

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip="🤗 Diffusion pipeline produced by a Pipeline Builder.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

    def set_runtime_parameters(self, pipeline_artifact: DiffusionPipelineArtifact) -> None:  # noqa: C901 PLR0912 PLR0915
        pipeline_class = pipeline_artifact.pipeline_name
        match pipeline_class:
            case "FluxPipeline":
                self._runtime_parameters = FluxPipelineRuntimeParameters(self._node)
            case "FluxFillPipeline":
                self._runtime_parameters = FluxFillPipelineRuntimeParameters(self._node)
            case "Flux2Pipeline":
                self._runtime_parameters = Flux2PipelineRuntimeParameters(self._node)
            case "Flux2KleinPipeline":
                self._runtime_parameters = Flux2PipelineRuntimeParameters(self._node)
            case "LTXPipeline":
                self._runtime_parameters = LTXPipelineRuntimeParameters(self._node)
            case "LTX2Pipeline":
                self._runtime_parameters = LTX2PipelineRuntimeParameters(self._node, pipeline_artifact.metadata)
            case "QwenImagePipeline":
                self._runtime_parameters = QwenPipelineRuntimeParameters(self._node)
            case "QwenImageEditPipeline":
                self._runtime_parameters = QwenEditPipelineRuntimeParameters(self._node)
            case "StableDiffusion3Pipeline":
                self._runtime_parameters = StableDiffusion3PipelineRuntimeParameters(self._node)
            case "StableDiffusionXLPipeline":
                self._runtime_parameters = StableDiffusionXLPipelineRuntimeParameters(self._node)
            case "WanPipeline":
                self._runtime_parameters = WanPipelineRuntimeParameters(self._node)
            case "WanImageToVideoPipeline":
                self._runtime_parameters = WanImageToVideoPipelineRuntimeParameters(self._node)
            case "ZImagePipeline":
                self._runtime_parameters = ZImagePipelineRuntimeParameters(self._node)

            case _:
                msg = f"Unsupported pipeline class: {pipeline_class}"
                logger.error(msg)
                raise ValueError(msg)

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

    @property
    def runtime_parameters(self) -> DiffusionPipelineRuntimeParameters:
        if self._runtime_parameters is None:
            msg = "Runtime parameters not initialized. Ensure pipeline parameter is set."
            logger.error(msg)
            raise ValueError(msg)
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
