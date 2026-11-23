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
    assert pipe.events == ["register:transformer", "load:vae", "offload"]
    assert pipe.load_kwargs["revision"] == {"vae": "selected"}
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
    assert pipeline.events == ["custom", "register:transformer", "load:vae", "offload"]


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

    assert pipeline.load_kwargs["pretrained_model_name_or_path"]["vae"] == "org/other"
    assert pipeline.load_kwargs["revision"]["vae"] == explicit_revision


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
    assert "transformer_ref" not in pipeline.load_kwargs["revision"]
    assert all(value == "selected" for value in pipeline.load_kwargs["revision"].values())
    assert pipeline.events[-1] == "offload"
    for name, component in overrides.items():
        assert getattr(pipeline, name) is component
        assert f"load:{name}" not in pipeline.events
