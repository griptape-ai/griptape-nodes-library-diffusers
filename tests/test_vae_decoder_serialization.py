"""Decode Media's outputs must be saved with the workflow.

``output_image`` and ``output_video`` hold URL artifacts pointing at files already written
to disk, so the engine can save them. A ``serializable=False`` flag on either makes the
engine drop the value on save, and the node comes back empty when the workflow reopens.
"""

from __future__ import annotations

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, SerializedParameterValueTracker
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.node_manager import NodeManager

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import DriverSpec
from modular_diffusion_nodes_library.nodes import vae_decoder
from modular_diffusion_nodes_library.nodes.vae_decoder import VaeDecodeNode


def _switch_output_type(node: VaeDecodeNode, monkeypatch: pytest.MonkeyPatch, *, produces_video: bool) -> None:
    driver_spec = DriverSpec(
        "stable_diffusion_xl:StableDiffusionXLLatentPipelineDriver",
        produces_video=produces_video,
        video_fps=24,
        supports_inpainting=True,
    )
    monkeypatch.setattr(vae_decoder, "get_driver_spec", lambda _pipeline_class: driver_spec)
    # Outside a running engine there are no connections to clean up, so remove directly.
    monkeypatch.setattr(
        node, "remove_parameter_element_by_name", super(VaeDecodeNode, node).remove_parameter_element_by_name
    )
    node._update_output_parameter()


def _saves_output_value(node: VaeDecodeNode, parameter_name: str) -> bool:
    parameter = node.get_parameter_by_name(parameter_name)
    assert parameter is not None
    commands = NodeManager.handle_parameter_value_saving(
        parameter=parameter,
        node=node,
        unique_parameter_uuid_to_values={},
        serialized_parameter_value_tracker=SerializedParameterValueTracker(),
        create_node_request=CreateNodeRequest(node_type="VaeDecodeNode"),
        workflow_manager=GriptapeNodes.WorkflowManager(),
    )
    return any(command.set_parameter_value_command.is_output for command in commands or [])


def test_output_image_is_saved() -> None:
    node = VaeDecodeNode(name="decode")
    node.parameter_output_values["output_image"] = ImageUrlArtifact("https://example.com/decoded.png")

    assert _saves_output_value(node, "output_image")


def test_output_video_is_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    node = VaeDecodeNode(name="decode")
    _switch_output_type(node, monkeypatch, produces_video=True)
    node.parameter_output_values["output_video"] = VideoUrlArtifact("https://example.com/decoded.mp4")

    assert node.get_parameter_by_name("output_image") is None
    assert _saves_output_value(node, "output_video")


def test_output_image_is_saved_after_switching_back_from_video(monkeypatch: pytest.MonkeyPatch) -> None:
    node = VaeDecodeNode(name="decode")
    _switch_output_type(node, monkeypatch, produces_video=True)
    _switch_output_type(node, monkeypatch, produces_video=False)
    node.parameter_output_values["output_image"] = ImageUrlArtifact("https://example.com/decoded.png")

    assert node.get_parameter_by_name("output_video") is None
    assert _saves_output_value(node, "output_image")
