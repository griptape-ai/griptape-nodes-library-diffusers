"""Field-mapping regression tests for family_registry.py's per-pipeline denoiser adapters.

Diffusers config field names for layers/heads/hidden-dim/patch-size are NOT uniform
across pipeline classes (see the family_registry.py module docstring), so a
copy-pasted adapter can silently read the wrong attribute name. That failure mode is
not loud: pipeline_memory_estimator.py's `_estimate_denoiser_activation_bytes` catches
AttributeError and downgrades to a "showing weights only" warning rather than raising.
Each test below gives an adapter a SimpleNamespace stub carrying exactly the field
names it claims to read (per the adapter's own source), and asserts the right values
land in the right DenoiserConfigFields/UnetLevelConfig slots -- so a wrong field name
fails the test instead of failing silently at runtime.
"""

from __future__ import annotations

from types import SimpleNamespace

from modular_diffusion_nodes_library.memory_estimation.family_registry import (
    _adapt_flux_family,
    _adapt_hunyuan_video15,
    _adapt_ltx_family,
    _adapt_minimax_h3,
    _adapt_qwen_family,
    _adapt_sd3,
    _adapt_wan_family,
    _adapt_zimage,
    get_sdxl_unet_levels,
)


def _pipe_with_transformer_config(**fields: object) -> SimpleNamespace:
    return SimpleNamespace(transformer=SimpleNamespace(config=SimpleNamespace(**fields)))


def test_adapt_flux_family_reads_flux_field_names() -> None:
    pipe = _pipe_with_transformer_config(
        num_attention_heads=24, attention_head_dim=128, num_layers=19, num_single_layers=38, patch_size=1
    )

    fields = _adapt_flux_family(pipe)

    assert fields.hidden_dim == 24 * 128
    assert fields.num_layers == 19 + 38
    assert fields.patch_size_spatial == 1
    assert fields.patch_size_temporal == 1


def test_adapt_qwen_family_reads_qwen_field_names() -> None:
    pipe = _pipe_with_transformer_config(num_attention_heads=16, attention_head_dim=128, num_layers=60, patch_size=2)

    fields = _adapt_qwen_family(pipe)

    assert fields.hidden_dim == 16 * 128
    assert fields.num_layers == 60
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_adapt_sd3_reads_sd3_field_names() -> None:
    pipe = _pipe_with_transformer_config(num_attention_heads=38, attention_head_dim=64, num_layers=24, patch_size=2)

    fields = _adapt_sd3(pipe)

    assert fields.hidden_dim == 38 * 64
    assert fields.num_layers == 24
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_adapt_zimage_reads_zimage_field_names() -> None:
    pipe = _pipe_with_transformer_config(dim=3072, n_layers=30, n_refiner_layers=2, all_patch_size=[2, 2])

    fields = _adapt_zimage(pipe)

    assert fields.hidden_dim == 3072
    assert fields.num_layers == 30 + 2
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_adapt_ltx_family_reads_ltx_field_names() -> None:
    pipe = _pipe_with_transformer_config(
        num_attention_heads=32, attention_head_dim=64, num_layers=28, patch_size=1, patch_size_t=1
    )

    fields = _adapt_ltx_family(pipe)

    assert fields.hidden_dim == 32 * 64
    assert fields.num_layers == 28
    assert fields.patch_size_spatial == 1
    assert fields.patch_size_temporal == 1


def test_adapt_wan_family_reads_wan_field_names_and_unpacks_the_patch_size_tuple() -> None:
    pipe = _pipe_with_transformer_config(
        num_attention_heads=40, attention_head_dim=128, num_layers=40, patch_size=(1, 2, 2)
    )

    fields = _adapt_wan_family(pipe)

    assert fields.hidden_dim == 40 * 128
    assert fields.num_layers == 40
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_adapt_hunyuan_video15_reads_hunyuan_field_names() -> None:
    pipe = _pipe_with_transformer_config(
        num_attention_heads=24,
        attention_head_dim=128,
        num_layers=20,
        num_refiner_layers=2,
        patch_size=2,
        patch_size_t=1,
    )

    fields = _adapt_hunyuan_video15(pipe)

    assert fields.hidden_dim == 24 * 128
    assert fields.num_layers == 20 + 2
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_adapt_minimax_h3_reads_minimax_field_names_and_unpacks_the_patch_size_tuple() -> None:
    pipe = _pipe_with_transformer_config(hidden_size=3072, num_layers=48, num_refiner_layers=4, patch_size=(1, 2, 2))

    fields = _adapt_minimax_h3(pipe)

    assert fields.hidden_dim == 3072
    assert fields.num_layers == 48 + 4
    assert fields.patch_size_spatial == 2
    assert fields.patch_size_temporal == 1


def test_get_sdxl_unet_levels_reads_per_level_shapes_with_a_flat_int_transformer_layer_count() -> None:
    pipe = SimpleNamespace(
        unet=SimpleNamespace(
            config=SimpleNamespace(block_out_channels=[320, 640, 1280], transformer_layers_per_block=2)
        )
    )

    levels = get_sdxl_unet_levels(pipe)

    assert [level.hidden_dim for level in levels] == [320, 640, 1280]
    assert [level.num_transformer_layers for level in levels] == [2, 2, 2]
    assert [level.downsample_factor for level in levels] == [1, 2, 4]


def test_get_sdxl_unet_levels_reads_a_per_level_transformer_layer_list() -> None:
    pipe = SimpleNamespace(
        unet=SimpleNamespace(
            config=SimpleNamespace(block_out_channels=[320, 640, 1280], transformer_layers_per_block=[1, 2, 10])
        )
    )

    levels = get_sdxl_unet_levels(pipe)

    assert [level.num_transformer_layers for level in levels] == [1, 2, 10]
