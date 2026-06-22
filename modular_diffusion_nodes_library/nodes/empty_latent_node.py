from typing import ClassVar, override

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import (
    LatentArtifact,  # type: ignore[reportMissingImports]
)
from modular_diffusion_nodes_library.nodes.noise_latent_node import NoiseLatentNode


class EmptyLatentNode(NoiseLatentNode):
    _DOC_URL: ClassVar[str] = (
        "https://github.com/griptape-ai/griptape-nodes-library-diffusers/blob/main/docs/nodes/empty_latents.md"
    )

    @override
    def add_seed_parameter(self) -> None:
        pass

    @override
    def _process(self) -> LatentArtifact:
        noise_artifact = super()._process()
        zero_tensor = torch.zeros_like(noise_artifact.to_torch())
        return LatentArtifact.from_torch(
            zero_tensor,
            source_shape=noise_artifact.source_shape,
            meta={},
        )
