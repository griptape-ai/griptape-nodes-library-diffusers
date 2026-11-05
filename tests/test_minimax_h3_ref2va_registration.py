from __future__ import annotations

import pytest

pytest.importorskip("diffusers", reason="Loads real MiniMax-H3 driver classes; run `make test/exec`.")

from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import MODULAR_PIPELINE_TYPE_PROVIDER_MAP
from modular_diffusion_nodes_library.parameters.providers import Provider
from modular_diffusion_nodes_library.runtime_parameters.minimax_h3_ref2va_runtime_parameters import (
    MiniMaxH3Ref2VAPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_params_registry import get_runtime_params_class
from modular_diffusion_nodes_library.standard_parameters.minimax_h3_ref2va_parameters import (
    MiniMaxH3Ref2VAPipelineParameters,
)


def test_ref2va_pipeline_name_is_distinct_from_the_upstream_class() -> None:
    assert MiniMaxH3Ref2VAPipelineParameters.pipeline_name() == "MiniMaxH3Ref2VAModularPipeline"
    assert MiniMaxH3Ref2VAPipelineParameters.pipeline_cls() is MiniMaxH3ModularPipeline


def test_ref2va_pipeline_type_is_listed_under_minimax_h3_provider() -> None:
    pipeline_types = MODULAR_PIPELINE_TYPE_PROVIDER_MAP[Provider.MINIMAX_H3].get_pipeline_type_dict()

    assert pipeline_types["MiniMaxH3Ref2VAModularPipeline"] is MiniMaxH3Ref2VAPipelineParameters


def test_ref2va_driver_and_runtime_parameters_are_registered() -> None:
    pipeline_name = MiniMaxH3Ref2VAPipelineParameters.pipeline_name()

    driver_class = get_driver_class(pipeline_name)
    assert driver_class is not None
    assert driver_class.__name__ == "MiniMaxH3Ref2VALatentPipelineDriver"
    assert get_runtime_params_class(pipeline_name) is MiniMaxH3Ref2VAPipelineRuntimeParameters


def test_ref2va_driver_shares_minimax_h3_latent_namespace() -> None:
    driver_class = get_driver_class(MiniMaxH3Ref2VAPipelineParameters.pipeline_name())

    assert driver_class is not None
    assert driver_class.__new__(driver_class).driver_namespace == "MiniMaxH3LatentPipelineDriver"
