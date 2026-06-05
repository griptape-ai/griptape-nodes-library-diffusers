from typing import Any, override

import torch  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.nodes.noise_latent_node import NoiseLatentNode


class EmptyLatentNode(NoiseLatentNode):
    @override
    def add_seed_parameter(self) -> None:
        pass

    @override
    def _process(self) -> Any:
        latents, latents_source_shape = super()._process()
        return torch.zeros_like(latents), latents_source_shape
