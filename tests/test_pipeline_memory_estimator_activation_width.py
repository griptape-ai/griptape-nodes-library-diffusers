"""Regression tests for pipeline_memory_estimator.py's activation-width and
component-role helpers.

`_activation_element_size` guards a specific historical bug: reading the activation
byte width directly off a quantized parameter (int8/uint8, or int4's 0.5-byte stored
width) silently zeroed out denoiser/VAE activation estimates entirely, since
activations are always computed at >=16-bit regardless of how the weights are stored.
`_classify_role` guards a second one: WAN 2.2's second denoiser slot (`transformer_2`)
previously fell into the "other" bucket, reporting zero activation memory and emitting
a spurious "unrecognized component" warning.
"""

from __future__ import annotations

import torch

from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    _MIN_ACTIVATION_BYTES_PER_ELEMENT,
    _activation_element_size,
    _classify_role,
)


class _QuantizedModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False)


class _MixedWidthModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, dtype=torch.int8), requires_grad=False)
        self.norm = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16), requires_grad=False)


def test_activation_element_size_clamps_quantized_weights_to_the_compute_floor() -> None:
    """A fully int8-quantized module must not report activations narrower than the
    2-byte floor -- reading the stored width directly previously zeroed int4 (0.5 bytes)
    activation estimates via `int(0.5) == 0`."""
    module = _QuantizedModule()

    assert _activation_element_size(module) == _MIN_ACTIVATION_BYTES_PER_ELEMENT


def test_activation_element_size_widens_to_the_widest_resident_parameter_dtype() -> None:
    """A quantized module keeps norms/biases at compute dtype even though its linear
    weights are quantized; taking the max recovers the compute width instead of landing
    on whichever parameter `next(parameters())` happens to return first."""
    module = _MixedWidthModule()

    assert _activation_element_size(module) == 2


def test_activation_element_size_falls_back_to_the_floor_with_no_parameters() -> None:
    module = torch.nn.Module()

    assert _activation_element_size(module) == _MIN_ACTIVATION_BYTES_PER_ELEMENT


def test_classify_role_recognizes_wans_second_denoiser_slot() -> None:
    assert _classify_role("transformer_2") == "denoiser"


def test_classify_role_recognizes_the_standard_slots() -> None:
    assert _classify_role("text_encoder") == "text_encoder"
    assert _classify_role("text_encoder_2") == "text_encoder"
    assert _classify_role("transformer") == "denoiser"
    assert _classify_role("unet") == "denoiser"
    assert _classify_role("vae") == "vae"
    assert _classify_role("scheduler") == "other"
