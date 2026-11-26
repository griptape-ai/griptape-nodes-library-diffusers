"""Encode Media reopened from a saved workflow must keep a single input type.

A saved video workflow re-adds `input_video` but, when `pipeline` is fed by a connection, never
sets `pipeline`, so the swap that would normally drop `image` does not run.
"""

from __future__ import annotations

from griptape_nodes.exe_types.core_types import Parameter

from modular_diffusion_nodes_library.nodes.vae_encoder import VaeEncodeNode


def test_loading_saved_video_input_without_pipeline_keeps_one_ordered_input() -> None:
    node = VaeEncodeNode(name="encode")
    node.add_parameter(
        Parameter(name="input_video", type="VideoUrlArtifact", input_types=["VideoUrlArtifact"], user_defined=True)
    )

    names = [element.name for element in node.root_ui_element.children]
    assert "image" not in names
    assert names.index("input_video") == names.index("latent_tensor") - 1
    assert names[-1] == "Status"
    assert node._current_input_type == "video"
