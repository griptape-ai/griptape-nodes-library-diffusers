from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode


class PropagationLoadPipelineNode(LoadPipelineNode):
    pass


class PipelineSinkNode(BaseNode):
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name=name, metadata=metadata)
        self.received: list[DiffusionPipelineArtifact] = []
        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                input_types=["Pipeline Config"],
                allowed_modes={ParameterMode.INPUT},
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "pipeline" and isinstance(value, DiffusionPipelineArtifact):
            self.received.append(value)

    def process(self) -> None:
        return None
