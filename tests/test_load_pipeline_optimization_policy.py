from copy import deepcopy

import pytest

from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode
from modular_diffusion_nodes_library.nodes.load_pipeline_optimization import (
    reconcile_optimization_settings,
    validated_optimization_overrides,
)


def _optimization_parameters():
    return LoadPipelineNode(name="Load Pipeline")._optimization_parameters()


def test_reconciliation_preserves_only_overrides_that_differ_from_recipe() -> None:
    parameters = _optimization_parameters()

    result = reconcile_optimization_settings(
        parameters=parameters,
        recipe_values={
            "cpu_offload_strategy": "Sequential",
            "vae_slicing": True,
        },
        overrides={
            "cpu_offload_strategy": "Sequential",
            "vae_slicing": False,
        },
    )

    assert result.overrides == {"vae_slicing": False}
    assert result.displayed_values["cpu_offload_strategy"] == "Sequential"
    assert result.displayed_values["vae_slicing"] is False


def test_reconciliation_uses_parameter_defaults_for_recipe_omissions() -> None:
    result = reconcile_optimization_settings(
        parameters=_optimization_parameters(),
        recipe_values={"cpu_offload_strategy": "Sequential"},
        overrides={},
    )

    assert result.displayed_values == {
        "memory_optimization_strategy": "Manual",
        "attention_slicing": False,
        "vae_slicing": False,
        "vae_tiling": False,
        "transformer_layerwise_casting": False,
        "cpu_offload_strategy": "Sequential",
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown_setting": True},
        {"cpu_offload_strategy": "invalid"},
        {"vae_slicing": "true"},
    ],
)
def test_override_validation_rejects_invalid_state(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        validated_optimization_overrides(overrides, _optimization_parameters())


def test_reconciliation_does_not_mutate_inputs() -> None:
    recipe_values = {
        "cpu_offload_strategy": "Sequential",
        "vae_slicing": True,
    }
    overrides = {
        "cpu_offload_strategy": "Sequential",
        "vae_slicing": False,
    }
    expected_recipe_values = deepcopy(recipe_values)
    expected_overrides = deepcopy(overrides)

    reconcile_optimization_settings(
        parameters=_optimization_parameters(),
        recipe_values=recipe_values,
        overrides=overrides,
    )

    assert recipe_values == expected_recipe_values
    assert overrides == expected_overrides
