# Copied from diffusers_nodes_library.common.parameters.diffusion.runtime_parameters
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter

logger = logging.getLogger("diffusers_nodes_library")


DEFAULT_NUM_INFERENCE_STEPS = 20


class DiffusionPipelineRuntimeParameters(ABC):
    def __init__(self, node: BaseNode):
        self._node = node
        self._seed_parameter = SeedParameter(node)

    @abstractmethod
    def _add_input_parameters(self) -> None:
        raise NotImplementedError

    def add_input_parameters(self) -> None:
        self._add_input_parameters()
        self._node.add_parameter(
            Parameter(
                name="num_inference_steps",
                default_value=DEFAULT_NUM_INFERENCE_STEPS,
                type="int",
                tooltip="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
            )
        )
        self._seed_parameter.add_input_parameters()

    def add_output_parameters(self) -> None:
        pass

    @abstractmethod
    def _remove_input_parameters(self) -> None:
        raise NotImplementedError

    def remove_input_parameters(self) -> None:
        self._node.remove_parameter_element_by_name("num_inference_steps")
        self._seed_parameter.remove_input_parameters()
        self._remove_input_parameters()

    def remove_output_parameters(self) -> None:
        pass

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._seed_parameter.after_value_set(parameter, value)

    def preprocess(self) -> None:
        self._seed_parameter.preprocess()

    def get_num_inference_steps(self) -> int:
        return int(self._node.get_parameter_value("num_inference_steps"))

    @abstractmethod
    def _get_pipe_kwargs(self) -> dict:
        raise NotImplementedError

    def get_pipe_kwargs(self) -> dict:
        return {
            **self._get_pipe_kwargs(),
            "num_inference_steps": self.get_num_inference_steps(),
            "generator": torch.Generator().manual_seed(self._seed_parameter.get_seed()),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        return None
