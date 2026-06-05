from __future__ import annotations

from diffusers_nodes_library.nodes.elementwise_latent_math import ElementwiseBinaryLatentNode


class AddLatentsNode(ElementwiseBinaryLatentNode):
    """Add two latent artifacts elementwise."""

    output_tooltip = "Elementwise sum of the two latent inputs."
    operation_name = "add"

    def get_operation(self):
        return lambda left_latent, right_latent: left_latent + right_latent
