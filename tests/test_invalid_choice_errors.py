"""An unsupported provider or pipeline type must name the valid choices.

Clients that set values through `SetParameterValueRequest` never see the dropdown, so the
error is the only place they can learn what the node accepts.
"""

from __future__ import annotations

import pytest

from modular_diffusion_nodes_library.nodes.controlnet_node import ControlNetNode
from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
    LatentDiffusionPipelineBuilderNode,
)


def test_builder_unsupported_provider_lists_valid_providers() -> None:
    node = LatentDiffusionPipelineBuilderNode(name="builder")

    with pytest.raises(ValueError, match="provider='Wan'") as exc_info:
        node.set_parameter_value("provider", "Wan")

    assert "WAN" in str(exc_info.value)
    assert "Flux2" in str(exc_info.value)


def test_builder_unsupported_provider_leaves_node_unchanged() -> None:
    node = LatentDiffusionPipelineBuilderNode(name="builder")
    node.set_parameter_value("provider", "WAN")
    parameter_names = [parameter.name for parameter in node.parameters]

    with pytest.raises(ValueError, match="Valid providers"):
        node.set_parameter_value("provider", "Wan")

    assert node.get_parameter_value("provider") == "WAN"
    assert [parameter.name for parameter in node.parameters] == parameter_names


def test_builder_unsupported_pipeline_type_lists_valid_pipeline_types() -> None:
    node = LatentDiffusionPipelineBuilderNode(name="builder")
    node.set_parameter_value("provider", "WAN")
    node.set_parameter_value("pipeline_type", "WanImageToVideoPipeline")
    parameter_names = [parameter.name for parameter in node.parameters]

    with pytest.raises(ValueError, match="pipeline_type='WanI2V'") as exc_info:
        node.set_parameter_value("pipeline_type", "WanI2V")

    assert "WanVACEPipeline" in str(exc_info.value)
    assert node.get_parameter_value("pipeline_type") == "WanImageToVideoPipeline"
    assert [parameter.name for parameter in node.parameters] == parameter_names


def test_controlnet_unsupported_provider_lists_valid_providers() -> None:
    node = ControlNetNode(name="controlnet")

    # The Options trait snaps an unknown value to the first choice before the node sees it,
    # so this error is reached only on workflow load, which hands the saved value straight in.
    with pytest.raises(ValueError, match="provider='flux'") as exc_info:
        node.set_controlnet_parameter_type("flux")

    assert "Flux" in str(exc_info.value)
    assert "Z-Image" in str(exc_info.value)
