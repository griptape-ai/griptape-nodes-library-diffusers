from __future__ import annotations

import logging
from pathlib import Path

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.utils import resolve_workspace_path

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.utils.config_utils import get_workspace_path

logger = logging.getLogger("modular_diffusers_nodes_library")


class SaveLatentTensorNode(DataNode):
    """Save a latent artifact to a torch .pt file."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="latent_tensor",
                type="LatentArtifact",
                input_types=["LatentArtifact"],
                tooltip="Latent artifact to serialise to disk as a torch .pt file.",
                allowed_modes={ParameterMode.INPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="file_path",
                type="str",
                default_value="debug/latent.pt",
                tooltip="Destination .pt file path. Relative paths are resolved against the workspace directory.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="saved_path",
                type="str",
                output_type="str",
                tooltip="Absolute filesystem path the tensor was written to.",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        file_path = self.get_parameter_value("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            exceptions.append(ValueError(f"Parameter \"file_path\" on node '{self.name}' must be a non-empty string."))

        return exceptions if exceptions else None

    def validate_in_execution_environment(self) -> list[Exception] | None:
        """Check the latent about to be saved.

        The latent is held by the process that produced it, so even asking whether it is connected has
        to happen where it lives.
        """
        latent_tensor = self.get_parameter_value("latent_tensor")
        if latent_tensor is None:
            return [ValueError(f"Parameter \"latent_tensor\" was left blank for node '{self.name}'.")]
        if not isinstance(latent_tensor, LatentArtifact):
            return [
                TypeError(
                    f"Parameter \"latent_tensor\" on node '{self.name}' must be a LatentArtifact, "
                    f"got '{type(latent_tensor).__name__}'."
                )
            ]
        return None

    def process(self) -> None:
        import torch  # type: ignore[reportMissingImports]

        latent_artifact = self.get_parameter_value("latent_tensor")
        file_path = self.get_parameter_value("file_path")

        workspace_path = get_workspace_path()
        resolved_path = resolve_workspace_path(Path(file_path), workspace_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)

        tensor = latent_artifact.to_torch().detach().cpu()
        torch.save(tensor, resolved_path)

        self.set_parameter_value("saved_path", str(resolved_path))
        self.parameter_output_values["saved_path"] = str(resolved_path)
        logger.info("[%s] saved latent tensor to %s", self.name, resolved_path)
