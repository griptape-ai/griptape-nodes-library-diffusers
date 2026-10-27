"""Pure-function tests for quantization_bytes.py's bytes-per-element resolution.

These bytes-per-element values are fixed by which named dtype
`_quantize_diffusion_pipeline` / `_manual_optimize_diffusion_pipeline` casts a
component to (see the module docstring) -- not calibrated -- so a change in the
result for a given input is a real behavior regression, not measurement noise.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("safetensors", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("accelerate", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

import torch  # noqa: E402

from modular_diffusion_nodes_library.memory_estimation.quantization_bytes import (  # noqa: E402
    LAYERWISE_CASTING_BYTES_PER_ELEMENT,
    estimate_component_weight_bytes,
    resolve_effective_bytes_per_element,
)


def test_prequantized_component_ignores_requested_optimization() -> None:
    """A pre-quantized checkpoint is already resident at its final width; re-applying the
    requested quantization_mode/layerwise-casting on top would double-count the shrink."""
    effective = resolve_effective_bytes_per_element(
        1.0,
        slot="transformer",
        quantization_mode="int4",
        transformer_layerwise_casting=True,
        is_prequantized=True,
        supports_layerwise_casting=True,
    )

    assert effective == 1.0


class _ProjectionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 2, bias=True, dtype=torch.bfloat16, device="meta")
        self.proj_out = torch.nn.Linear(4, 2, bias=True, dtype=torch.bfloat16, device="meta")


def test_quantization_estimate_keeps_excluded_proj_out_at_stored_width() -> None:
    module = _ProjectionModule()

    total_weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())
    estimated = estimate_component_weight_bytes(module, quantization_mode="int8", total_weight_bytes=total_weight_bytes)
    expected = module.linear.weight.numel() + module.linear.bias.numel() * 2
    expected += module.proj_out.weight.numel() * 2 + module.proj_out.bias.numel() * 2

    assert estimated == expected


def test_int4_quantization_resolves_to_half_a_byte() -> None:
    effective = resolve_effective_bytes_per_element(
        2.0,
        slot="transformer",
        quantization_mode="int4",
        transformer_layerwise_casting=False,
        is_prequantized=False,
        supports_layerwise_casting=False,
    )

    assert effective == 0.5


def test_unrecognized_quantization_mode_leaves_the_stored_width_unchanged() -> None:
    effective = resolve_effective_bytes_per_element(
        2.0,
        slot="transformer",
        quantization_mode="None",
        transformer_layerwise_casting=False,
        is_prequantized=False,
        supports_layerwise_casting=False,
    )

    assert effective == 2.0


def test_layerwise_casting_only_applies_to_the_transformer_slot() -> None:
    non_transformer = resolve_effective_bytes_per_element(
        2.0,
        slot="vae",
        quantization_mode="None",
        transformer_layerwise_casting=True,
        is_prequantized=False,
        supports_layerwise_casting=True,
    )
    transformer = resolve_effective_bytes_per_element(
        2.0,
        slot="transformer",
        quantization_mode="None",
        transformer_layerwise_casting=True,
        is_prequantized=False,
        supports_layerwise_casting=True,
    )

    assert non_transformer == 2.0
    assert transformer == LAYERWISE_CASTING_BYTES_PER_ELEMENT


def test_layerwise_casting_requires_the_component_to_support_it() -> None:
    """A transformer that doesn't support layerwise casting keeps its quantized width."""
    effective = resolve_effective_bytes_per_element(
        2.0,
        slot="transformer",
        quantization_mode="None",
        transformer_layerwise_casting=True,
        is_prequantized=False,
        supports_layerwise_casting=False,
    )

    assert effective == 2.0


def test_layerwise_casting_applies_on_top_of_quantization_mode() -> None:
    """quantization_mode is checked first (every component), then layerwise casting on
    top (transformer slot only) -- same order as _manual_optimize_diffusion_pipeline."""
    effective = resolve_effective_bytes_per_element(
        2.0,
        slot="transformer",
        quantization_mode="int8",
        transformer_layerwise_casting=True,
        is_prequantized=False,
        supports_layerwise_casting=True,
    )

    assert effective == LAYERWISE_CASTING_BYTES_PER_ELEMENT
