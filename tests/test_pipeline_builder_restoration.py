"""The pipeline builder while a saved workflow restores its parameters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("initial_setup", [False, True], ids=["normal-edit", "restoration"])
@pytest.mark.parametrize("nested", [False, True], ids=["direct", "nested"])
def test_flux3_restoration_does_not_query_connections(
    monkeypatch: pytest.MonkeyPatch, initial_setup: bool, nested: bool
) -> None:
    from griptape_nodes.exe_types.node_types import BaseNode

    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import (
        Flux3ModelRepoParameter,
        Flux3PipelineParameters,
    )

    node: Any = LatentDiffusionPipelineBuilderNode.__new__(LatentDiffusionPipelineBuilderNode)
    node._restoring_parameters = False
    values = {"model": "org/model/fp8r", "provider": "Flux3", "component_transformer": None}
    queries: list[str] = []
    setup_flags: list[bool] = []
    node.get_parameter_value = values.get
    node.get_parameter_by_name = lambda name: SimpleNamespace(name=name)
    node._param_has_incoming_connection = lambda name: bool(queries.append(name))

    def store(self: Any, name: str, value: Any, **kwargs: Any) -> None:
        values[name] = value
        setup_flags.append(kwargs["initial_setup"])

    def after_value_set(parameter: Any, value: Any, **kwargs: Any) -> None:
        if parameter.name == "provider":
            node.set_parameter_value("model", values["model"])

    monkeypatch.setattr(BaseNode, "set_parameter_value", store)
    node.params = SimpleNamespace(before_value_set=lambda *args, **kwargs: None, after_value_set=after_value_set)
    node.huggingface_pipeline_params = SimpleNamespace(after_value_set=lambda *args: None)
    repo_parameter = Flux3ModelRepoParameter.__new__(Flux3ModelRepoParameter)
    repo_parameter._node = node
    repo_parameter._parameter_name = "model"
    params = Flux3PipelineParameters.__new__(Flux3PipelineParameters)
    params._node = node
    params._model_repo_parameter = repo_parameter
    node.set_pipeline_artifact = params.is_prequantized
    parameter_name = "model"
    expected_setters = 1
    if nested:
        parameter_name = "provider"
        expected_setters = 2

    node.set_parameter_value(parameter_name, values[parameter_name], initial_setup=initial_setup)

    assert setup_flags == [initial_setup] * expected_setters
    assert node._restoring_parameters is False
    if initial_setup:
        assert queries == []
    else:
        assert queries == ["component_transformer"] * expected_setters


def test_parameter_restoration_scope_resets_after_nested_failure() -> None:
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )

    node = LatentDiffusionPipelineBuilderNode.__new__(LatentDiffusionPipelineBuilderNode)
    node._restoring_parameters = False

    with pytest.raises(RuntimeError, match="test restoration failure"):
        with node._parameter_restoration_scope(True) as restoring:
            assert restoring is True
            with node._parameter_restoration_scope(False) as nested_restoring:
                assert nested_restoring is True
                raise RuntimeError("test restoration failure")

    assert node._restoring_parameters is False
