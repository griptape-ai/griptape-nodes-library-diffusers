from __future__ import annotations

from modular_diffusion_nodes_library.nodes.elementwise_latent_math import ElementwiseBinaryLatentNode


class SubtractLatentsNode(ElementwiseBinaryLatentNode):
    """Subtract the right latent from the left latent elementwise."""

    output_tooltip = "Elementwise difference of the two latent inputs."
    operation_name = "subtract"

    def get_operation(self):
        return lambda left_latent, right_latent: left_latent - right_latent
