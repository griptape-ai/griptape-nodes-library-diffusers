"""Unit tests for the small helpers behind VAE meta tracing."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from modular_diffusion_nodes_library.memory_estimation.vae_trace import _extrapolate, _reduced_extent
from modular_diffusion_nodes_library.memory_estimation.vae_trace_common import LivenessTracker, patched_attribute


def test_patched_attribute_restores_previous_value() -> None:
    target = SimpleNamespace(forward="original")

    with patched_attribute(target, "forward", "patched"):
        assert target.forward == "patched"

    assert target.forward == "original"


def test_patched_attribute_removes_attribute_it_added() -> None:
    target = SimpleNamespace()

    with patched_attribute(target, "forward", "patched"):
        assert target.forward == "patched"

    assert not hasattr(target, "forward")


def test_patched_attribute_restores_after_error() -> None:
    target = SimpleNamespace(forward="original")

    with pytest.raises(RuntimeError, match="boom"), patched_attribute(target, "forward", "patched"):
        raise RuntimeError("boom")

    assert target.forward == "original"


def test_paused_tracker_ignores_allocations() -> None:
    tracker = LivenessTracker(torch.nn.Identity())

    with tracker:
        kept = torch.empty(4, device="meta")
        with tracker.paused():
            ignored = torch.empty(8, device="meta")

    assert tracker.peak_bytes == kept.nbytes
    assert ignored.nbytes > kept.nbytes


def test_extrapolate_one_axis_is_linear_in_tiles() -> None:
    # 10 bytes at 4 tiles and 13 at 5 tiles: 3 more bytes per tile, 3 tiles beyond the samples.
    assert _extrapolate({(4,): 10, (5,): 13}, steps=[3]) == 19


def test_extrapolate_two_axes_is_bilinear_in_tiles() -> None:
    samples = {(4, 4): 100, (5, 4): 110, (4, 5): 120, (5, 5): 135}

    # f(h, w) = 100 + 10*h + 20*w + 5*h*w with h=2, w=3.
    assert _extrapolate(samples, steps=[2, 3]) == 210


def test_extrapolate_never_returns_less_than_a_sample() -> None:
    assert _extrapolate({(4,): 10, (5,): 7}, steps=[2]) == 10


@pytest.mark.parametrize("tiles", [4, 5])
def test_reduced_extent_keeps_tile_count_and_edge_remainder(tiles: int) -> None:
    extent, stride = 176, 24

    reduced = _reduced_extent(extent, stride, tiles)

    assert math.ceil(reduced / stride) == tiles
    assert reduced % stride == extent % stride
