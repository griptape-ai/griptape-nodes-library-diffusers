"""No-op runtime parameters used before any pipeline is bound."""

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)


class NullRuntimeParameters(DiffusionPipelineRuntimeParameters):
    """No-op runtime parameters used before any pipeline is bound."""

    def __init__(self, node: BaseNode):
        # Skip the base `__init__` so we don't construct a SeedParameter.
        self._node = node

    def add_input_parameters(self) -> None:
        pass

    def add_output_parameters(self) -> None:
        pass

    def remove_input_parameters(self) -> None:
        pass

    def remove_output_parameters(self) -> None:
        pass

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        pass

    def preprocess(self) -> None:
        pass

    def validate_before_node_run(self) -> list[Exception] | None:
        return None

    def _add_input_parameters(self) -> None:
        pass

    def _remove_input_parameters(self) -> None:
        pass

    def _get_pipe_kwargs(self) -> dict:
        return {}
