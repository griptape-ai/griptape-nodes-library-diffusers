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

    def validate_in_execution_environment(self) -> list[Exception] | None:
        """Both inputs are held latents, so only the process running the node can read them.

        The operation is its own check: the shape and dtype compatibility this node can fail on lives in
        the tensors, and `_apply_operation` is where they meet. That needs torch, which the orchestrator
        does not have.
        """
        exceptions: list[Exception] = []
        latents: dict[str, LatentArtifact] = {}
        for name in ("left_latent", "right_latent"):
            value = self.get_parameter_value(name)
            if value is None:
                exceptions.append(ValueError(f"Parameter \"{name}\" was left blank for node '{self.name}'."))
            elif not isinstance(value, LatentArtifact):
                exceptions.append(
                    TypeError(
                        f"Parameter \"{name}\" on node '{self.name}' must be a LatentArtifact, "
                        f"got '{type(value).__name__}'."
                    )
                )
            else:
                latents[name] = value

        if exceptions:
            return exceptions

        try:
            self._apply_operation(latents["left_latent"], latents["right_latent"])
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
