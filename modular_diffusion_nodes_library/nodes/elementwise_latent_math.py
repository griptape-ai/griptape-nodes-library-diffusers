from __future__ import annotations

import logging
from collections.abc import Callable

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class ElementwiseBinaryLatentNode(DataNode):
    """Base class for elementwise latent artifact math nodes."""

    output_tooltip: str
    operation_name: str

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="left_latent",
                type="LatentArtifact",
                input_types=["LatentArtifact"],
                tooltip="Left latent input.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="right_latent",
                type="LatentArtifact",
                input_types=["LatentArtifact"],
                tooltip="Right latent input.",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_latent",
                output_type="LatentArtifact",
                tooltip=self.output_tooltip,
                allowed_modes={ParameterMode.OUTPUT},
                serializable=False,
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        left_latent = self.get_parameter_value("left_latent")
        if left_latent is None:
            exceptions.append(ValueError(f"Parameter \"left_latent\" was left blank for node '{self.name}'."))  # noqa: PERF401
        elif not isinstance(left_latent, LatentArtifact):
            exceptions.append(
                TypeError(
                    f"Parameter \"left_latent\" on node '{self.name}' must be a LatentArtifact, got '{type(left_latent).__name__}'."
                )
            )

        right_latent = self.get_parameter_value("right_latent")
        if right_latent is None:
            exceptions.append(
                ValueError(f"Parameter \"right_latent\" was left blank for node '{self.name}'.")  # noqa: PERF401
            )
        elif not isinstance(right_latent, LatentArtifact):
            exceptions.append(
                TypeError(
                    f"Parameter \"right_latent\" on node '{self.name}' must be a LatentArtifact, got '{type(right_latent).__name__}'."
                )
            )

        if exceptions:
            return exceptions

        try:
            self._apply_operation(left_latent, right_latent)
        except (TypeError, ValueError, RuntimeError) as error:
            return [error]

        return None

    def process(self) -> None:
        left_latent = self.get_parameter_value("left_latent")
        right_latent = self.get_parameter_value("right_latent")
        result = self._apply_operation(left_latent, right_latent)

        self.set_parameter_value("output_latent", result)
        self.parameter_output_values["output_latent"] = result
        logger.debug("[%s] %s result: %s", self.name, self.operation_name, result.to_dict())

    def _apply_operation(self, left_latent: LatentArtifact, right_latent: LatentArtifact) -> LatentArtifact:
        operation = self.get_operation()
        return operation(left_latent, right_latent)

    def get_operation(self) -> Callable[[LatentArtifact, LatentArtifact], LatentArtifact]:
        raise NotImplementedError
