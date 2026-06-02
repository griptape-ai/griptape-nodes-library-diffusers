from __future__ import annotations

from modular_diffusion_nodes_library.nodes.elementwise_latent_math import ElementwiseBinaryLatentNode


class MultiplyLatentsNode(ElementwiseBinaryLatentNode):
    """Multiply two latent artifacts elementwise."""

    output_tooltip = "Elementwise product of the two latent inputs."
    operation_name = "multiply"

    def get_operation(self):
        return lambda left_latent, right_latent: left_latent * right_latent
