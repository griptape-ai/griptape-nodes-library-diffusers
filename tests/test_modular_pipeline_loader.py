"""The shared modular loader's lifecycle: overrides, custom loaders, preflight and sources."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


def test_overrides_precede_defaults_and_scope_workflows(loader_environment: SimpleNamespace) -> None:
    transformer = object()
    pipe = loader_environment.make_loader(workflows=("active",), device="cpu").from_pretrained(
        "org/base", revision="selected", overrides={"transformer": transformer}
    )

    assert pipe.transformer is transformer
    assert pipe.unused is None
    assert pipe.events == ["register:transformer", "register:vae", "offload"]
    assert pipe.vae.source == "org/base"
    assert pipe.vae.load_kwargs["revision"] == "selected"
    assert pipe._manages_runtime_optimisation is True


def test_custom_loader_runs_before_defaults(loader_environment: SimpleNamespace) -> None:
    pipeline = loader_environment.pipeline
    transformer = object()

    def load_transformer(context: Any) -> object:
        assert context.source == "org/base"
        assert context.revision == "selected"
        pipeline.events.append("custom")
        return transformer

    loader_environment.make_loader(
        workflows=("active",), component_loaders={"transformer": load_transformer}, device="cpu"
    ).from_pretrained("org/base", revision="selected")

    assert pipeline.transformer is transformer
    assert pipeline.events == ["custom", "register:transformer", "register:vae", "offload"]


def test_preflight_precedes_override_materialisation(
    loader_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
        ComponentSourceType,
        ModelComponentArtifact,
    )

    def reject(context: Any) -> None:
        raise RuntimeError("unsupported backend")

    monkeypatch.setattr(ModelComponentArtifact, "materialize", lambda *args, **kwargs: pytest.fail("weights loaded"))
    loader = loader_environment.make_loader(workflows=("active",), component_validators={"transformer": reject})
    override = ModelComponentArtifact(
        load_id="test-vae", source_type=ComponentSourceType.LOCAL_DIR, component="vae", file_path="/unused"
    )

    with pytest.raises(RuntimeError, match="unsupported backend"):
        loader.from_pretrained("org/base", overrides={"vae": override})

    assert loader_environment.pipeline.events == []


@pytest.mark.parametrize("explicit_revision", [None, "replacement"])
def test_source_replacement_does_not_inherit_the_base_revision(
    loader_environment: SimpleNamespace, explicit_revision: str | None
) -> None:
    from tests.fake_modular_pipeline import component_spec

    pipeline = loader_environment.make_pipeline(vae=component_spec("vae", revision="original"))
    loader_environment.pipeline = pipeline
    loader_environment.make_loader(
        workflows=("active",),
        device="cpu",
        component_options={"vae": {"pretrained_model_name_or_path": "org/other", "revision": explicit_revision}},
    ).from_pretrained("org/base", revision="selected")

    assert pipeline.vae.source == "org/other"
    assert pipeline.vae.load_kwargs["revision"] == explicit_revision


def test_component_without_a_repo_path_loads_from_the_base_repo(loader_environment: SimpleNamespace) -> None:
    """A component the pipeline's index names no repo for comes from the base repo at the selected revision."""
    from tests.fake_modular_pipeline import BASE_REPO, FakeComponent, component_spec

    pipeline = loader_environment.make_pipeline(vae=component_spec("vae", pretrained_model_name_or_path=None))
    loader_environment.pipeline = pipeline
    loader_environment.make_loader(workflows=("active",), device="cpu").from_pretrained(
        BASE_REPO, revision="selected", overrides={"transformer": object()}
    )

    assert isinstance(pipeline.vae, FakeComponent)
    assert pipeline.vae.source == BASE_REPO
    assert pipeline.vae.load_kwargs["subfolder"] == "vae"
    assert pipeline.vae.load_kwargs["revision"] == "selected"


@pytest.mark.parametrize(
    ("vae_options", "expected_variant"),
    [
        ({}, "fp16"),
        ({"pretrained_model_name_or_path": "org/other"}, None),
        ({"pretrained_model_name_or_path": "org/other", "variant": "bf16"}, "bf16"),
    ],
    ids=["unchanged-source", "replaced-source", "replaced-source-explicit-variant"],
)
def test_spec_variant_is_kept_only_while_the_source_is_unchanged(
    loader_environment: SimpleNamespace, vae_options: dict[str, Any], expected_variant: str | None
) -> None:
    from tests.fake_modular_pipeline import BASE_REPO, component_spec

    pipeline = loader_environment.make_pipeline(vae=component_spec("vae", variant="fp16"))
    loader_environment.pipeline = pipeline
    loader_environment.make_loader(
        workflows=("active",), device="cpu", component_options={"vae": vae_options}
    ).from_pretrained(BASE_REPO, overrides={"transformer": object()})

    assert pipeline.vae.load_kwargs["variant"] == expected_variant


def test_failed_load_reports_its_cause(loader_environment: SimpleNamespace) -> None:
    from tests.fake_modular_pipeline import BASE_REPO, component_spec

    class Unloadable:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> Any:
            raise OSError("no such snapshot")

    loader_environment.pipeline = loader_environment.make_pipeline(vae=component_spec("vae", type_hint=Unloadable))
    loader = loader_environment.make_loader(workflows=("active",), device="cpu")

    with pytest.raises(ValueError, match="no such snapshot"):
        loader.from_pretrained(BASE_REPO, overrides={"transformer": object()})


@pytest.mark.parametrize(
    "override_slot",
    [None, "transformer", "vae", "text_encoder", "tokenizer", "scheduler", "audio_vae", "audio_scheduler"],
)
def test_h3_repository_build_loads_only_missing_workflow_components(
    loader_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, override_slot: str | None
) -> None:
    from modular_diffusion_nodes_library.standard_parameters import minimax_h3_parameters as module
    from tests.fake_modular_pipeline import FakeModularPipeline, component_spec

    params_cls = module.MiniMaxH3PipelineParameters
    required = [*params_cls.get_component_slots(), "processor"]
    pipeline = FakeModularPipeline(
        [component_spec(name) for name in [*required, "transformer_ref"]],
        sub_blocks={"generate": required, "reference": ["transformer_ref"]},
        workflows={"t2va": ["generate"], "fl2va": ["generate"]},
    )
    loader_environment.pipeline = pipeline
    processor = object()
    monkeypatch.setattr(params_cls, "pipeline_cls", lambda: loader_environment.pipeline_cls)
    monkeypatch.setattr(params_cls, "_load_processor", lambda source, revision: processor)
    monkeypatch.setattr(module, "get_best_device", lambda: "cpu")
    overrides = {}
    if override_slot is not None:
        overrides[override_slot] = object()

    result = params_cls.build_pipeline_from_build_data(
        {**params_cls.build_data("org/base", "selected"), "_component_overrides": overrides}
    )

    assert result is pipeline
    assert pipeline.processor is processor
    assert pipeline.transformer_ref is None
    assert pipeline.events[-1] == "offload"
    for name in params_cls.get_component_slots():
        component = getattr(pipeline, name)
        if name in overrides:
            assert component is overrides[name]
        else:
            assert component.source == "org/base"
            assert component.load_kwargs["revision"] == "selected"
