"""The placement editor and the x_offset / y_offset / resize_source params must stay in step.

Either side can set the placement, so a change on one has to reach the other, and a param fed by a
connection belongs to that connection rather than to the widget.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.nodes.latent_composite_mask_node import LatentCompositeMaskNode

SIZES = {"dest_width": 1024, "dest_height": 768, "source_width": 512, "source_height": 512}


def _widget(node: LatentCompositeMaskNode) -> dict:
    return node.get_parameter_value("placement_editor")


def test_widget_drag_sets_offset_params() -> None:
    node = LatentCompositeMaskNode(name="composite")

    node.set_parameter_value(
        "placement_editor", {**_widget(node), **SIZES, "x_offset": 256, "y_offset": 128, "resize_source": True}
    )

    assert node.get_parameter_value("x_offset") == 256
    assert node.get_parameter_value("y_offset") == 128
    assert node.get_parameter_value("resize_source") is True


def test_offset_params_update_widget() -> None:
    node = LatentCompositeMaskNode(name="composite")

    node.set_parameter_value("x_offset", 64)
    node.set_parameter_value("resize_source", True)

    assert _widget(node)["x_offset"] == 64
    assert _widget(node)["resize_source"] is True


def test_widget_leaves_connected_param_alone() -> None:
    node = LatentCompositeMaskNode(name="composite")
    node.set_parameter_value("x_offset", 32)

    node.set_parameter_value(
        "placement_editor", {**_widget(node), "x_offset": 512, "y_offset": 64, "locked": ["x_offset"]}
    )

    assert node.get_parameter_value("x_offset") == 32
    assert node.get_parameter_value("y_offset") == 64


def test_param_change_keeps_sizes_from_last_run() -> None:
    node = LatentCompositeMaskNode(name="composite")
    # What a run leaves behind: the sizes reach parameter_output_values, not the stored value.
    node.parameter_output_values["placement_editor"] = {**_widget(node), **SIZES}

    node.set_parameter_value("y_offset", 200)

    assert _widget(node)["y_offset"] == 200
    assert {key: _widget(node)[key] for key in SIZES} == SIZES


def test_widget_can_place_past_slider_range() -> None:
    node = LatentCompositeMaskNode(name="composite")

    node.set_parameter_value("placement_editor", {**_widget(node), "x_offset": 2048, "y_offset": 64})

    assert node.get_parameter_value("x_offset") == 2048
    assert node.get_parameter_value("y_offset") == 64


def test_run_reports_pixel_sizes_of_both_latents() -> None:
    node = LatentCompositeMaskNode(name="composite")

    # A 5D video latent: (batch, channels, frames, height, width).
    node._publish_latent_sizes((1, 16, 5, 96, 128), (1, 16, 5, 64, 64))

    assert {key: node.parameter_output_values["placement_editor"][key] for key in SIZES} == SIZES
