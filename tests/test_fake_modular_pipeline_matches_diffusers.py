"""Pin `tests/fake_modular_pipeline.py` to the Diffusers behaviour it stands in for.

The loader tests run against the fake, so they are only as good as its imitation. Each case here runs
the same probe on a real Flux3 modular pipeline with no weights and on the fake built with the same
components, and fails when the two disagree. That happens after an edit to the fake, or after a
Diffusers bump that changes behaviour the loader relies on.

If Flux3's blocks gain or lose a component, the component probes fail: update the layout below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from tests.fake_modular_pipeline import FakeModularPipeline, component_spec

if TYPE_CHECKING:
    from collections.abc import Callable

diffusers = pytest.importorskip("diffusers", reason="Compares against real Diffusers; run `make test/exec`.")

UPSAMPLER_COMPONENTS = ["prompt_upsampler", "prompt_upsampler_processor"]
PRETRAINED_COMPONENTS = ["text_encoder", "tokenizer", "vae", "transformer"]
CONFIG_COMPONENTS = ["image_processor", "scheduler", "guider"]


@pytest.fixture
def real() -> Any:
    """A Flux3 modular pipeline built from its blocks alone: every spec declared, no repo, no weights."""
    return diffusers.Flux3AutoBlocks().init_pipeline()


@pytest.fixture
def fake() -> FakeModularPipeline:
    specs = [
        component_spec(name, pretrained_model_name_or_path=None)
        for name in [*UPSAMPLER_COMPONENTS, *PRETRAINED_COMPONENTS]
    ]
    for name in CONFIG_COMPONENTS:
        specs.append(component_spec(name, pretrained_model_name_or_path=None, default_creation_method="from_config"))
    return FakeModularPipeline(
        specs,
        sub_blocks={"prompt_upsampler": UPSAMPLER_COMPONENTS, "core": [*PRETRAINED_COMPONENTS, *CONFIG_COMPONENTS]},
        workflows={"text2image": ["prompt_upsampler", "core"], "image_conditioned": ["prompt_upsampler", "core"]},
    )


def spec_mutation_persists(pipeline: Any) -> bool:
    spec = pipeline.get_component_spec("vae")
    spec.pretrained_model_name_or_path = "org/elsewhere"
    return pipeline.get_component_spec("vae").pretrained_model_name_or_path == "org/elsewhere"


def blocks_is_a_copy(pipeline: Any) -> bool:
    return pipeline.blocks is not pipeline._blocks


def loads_a_component_whose_stored_spec_has_no_path(pipeline: Any) -> bool:
    pipeline.load_components(names=["vae"], pretrained_model_name_or_path={"vae": "/not/a/repo"}, local_files_only=True)
    return pipeline.vae is not None


def components_created_at_init(pipeline: Any) -> dict[str, bool]:
    return {name: pipeline.components[name] is not None for name in sorted(pipeline.components)}


def workflow_components(pipeline: Any) -> list[str]:
    return sorted(spec.name for spec in pipeline.blocks.get_workflow("text2image").expected_components)


def workflow_components_after_popping_a_sub_block(pipeline: Any) -> list[str]:
    pipeline._blocks.sub_blocks.pop("prompt_upsampler")
    return workflow_components(pipeline)


def update_components_sets_an_undeclared_name(pipeline: Any) -> bool:
    pipeline.update_components(not_a_component=object())
    return getattr(pipeline, "not_a_component", None) is not None


PROBES: list[Callable[[Any], Any]] = [
    spec_mutation_persists,
    blocks_is_a_copy,
    loads_a_component_whose_stored_spec_has_no_path,
    components_created_at_init,
    workflow_components,
    workflow_components_after_popping_a_sub_block,
    update_components_sets_an_undeclared_name,
]


@pytest.mark.parametrize("probe", PROBES, ids=[probe.__name__ for probe in PROBES])
def test_fake_answers_like_diffusers(probe: Callable[[Any], Any], real: Any, fake: FakeModularPipeline) -> None:
    assert probe(fake) == probe(real)


def test_unknown_workflow_is_a_value_error_in_both(real: Any, fake: FakeModularPipeline) -> None:
    for pipeline in (real, fake):
        with pytest.raises(ValueError, match="not found"):
            pipeline.blocks.get_workflow("no-such-workflow")


def test_names_and_workflow_together_is_a_value_error_in_both(real: Any, fake: FakeModularPipeline) -> None:
    for pipeline in (real, fake):
        with pytest.raises(ValueError, match="not both"):
            pipeline.load_components(names=["vae"], workflow="text2image")
