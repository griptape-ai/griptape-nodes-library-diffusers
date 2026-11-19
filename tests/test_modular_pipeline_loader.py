from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest


class PipelineStub:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.transformer = None
        self.vae = None
        self.scheduler = object()
        self.unused = None
        self.specs = {
            name: SimpleNamespace(
                name=name,
                default_creation_method="from_pretrained",
                pretrained_model_name_or_path="org/base",
                subfolder=name,
                revision=None,
            )
            for name in ("transformer", "vae", "scheduler", "unused")
        }
        self.specs["scheduler"].default_creation_method = "from_config"
        self.blocks = SimpleNamespace(
            expected_components=list(self.specs.values()),
            get_workflow=lambda workflow: SimpleNamespace(
                expected_components=[self.specs[name] for name in ("transformer", "vae", "scheduler")]
            ),
        )
        self.load_kwargs: dict[str, Any] = {}

    def get_component_spec(self, name: str) -> Any:
        return self.specs[name]

    def update_components(self, **components: Any) -> None:
        for name, component in components.items():
            self.events.append(f"register:{name}")
            setattr(self, name, component)

    def load_components(self, *, names: list[str], **kwargs: Any) -> None:
        self.load_kwargs = kwargs
        for name in names:
            self.events.append(f"load:{name}")
            setattr(self, name, object())


@pytest.fixture
def loader_environment(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.component_loading.modular_pipeline_loader import (
        ModularPipelineLoader,
        ModularPipelineLoadOptions,
    )

    pipeline = PipelineStub()

    class Manager:
        def enable_auto_cpu_offload(self, **kwargs: Any) -> None:
            pipeline.events.append("offload")

    class PipelineClass:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> PipelineStub:
            assert source == "org/base"
            assert kwargs["local_files_only"] is True
            return pipeline

    monkeypatch.setattr("diffusers.modular_pipelines.components_manager.ComponentsManager", Manager)

    def make_loader(**options: Any) -> ModularPipelineLoader:
        return ModularPipelineLoader(pipeline_cls=PipelineClass, options=ModularPipelineLoadOptions(**options))

    return SimpleNamespace(pipeline=pipeline, make_loader=make_loader, pipeline_cls=PipelineClass)


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
    pipeline = loader_environment.pipeline
    pipeline.specs["vae"].revision = "original"
    loader_environment.make_loader(
        workflows=("active",),
        device="cpu",
        component_options={"vae": {"pretrained_model_name_or_path": "org/other", "revision": explicit_revision}},
    ).from_pretrained("org/base", revision="selected")

    assert pipeline.load_kwargs["pretrained_model_name_or_path"]["vae"] == "org/other"
    assert pipeline.load_kwargs["revision"]["vae"] == explicit_revision


@pytest.mark.parametrize("strategy", ["Manual", "Automatic"])
@pytest.mark.parametrize("offload", ["None", "Model", "Sequential"])
def test_loader_owned_placement_preserves_memory_options(
    monkeypatch: pytest.MonkeyPatch, strategy: str, offload: str
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils import pipeline_utils

    events: list[str] = []

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Loader-owned placement or weights were modified")

    pipe: Any = SimpleNamespace(
        _manages_runtime_optimisation=True,
        vae=SimpleNamespace(
            use_slicing=False,
            use_tiling=False,
            enable_slicing=lambda: events.append("slicing"),
            enable_tiling=lambda: events.append("tiling"),
        ),
        transformer=SimpleNamespace(enable_layerwise_casting=forbidden),
        enable_model_cpu_offload=forbidden,
        enable_sequential_cpu_offload=forbidden,
        to=forbidden,
    )
    monkeypatch.setattr(pipeline_utils, "get_best_device", lambda: torch.device("cuda"))
    monkeypatch.setattr(pipeline_utils, "_log_memory_info", lambda *args: None)
    monkeypatch.setattr(pipeline_utils, "_quantize_diffusion_pipeline", forbidden)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", lambda **kwargs: events.append("backend"))

    pipeline_utils.optimize_diffusion_pipeline(
        pipe,
        memory_optimization_strategy=strategy,
        vae_slicing=True,
        vae_tiling=True,
        quantization_mode="int4",
        transformer_layerwise_casting=True,
        cpu_offload_strategy=offload,
    )

    assert events == ["slicing", "tiling", "backend"]


@pytest.mark.parametrize("pipeline_name", ["Flux3ModularPipeline", "MiniMaxH3ModularPipeline"])
def test_modular_component_lookup_uses_real_block_metadata(pipeline_name: str) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.component_loading.pipeline_components import pipeline_component_types
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    pipeline_cls = getattr(diffusers, pipeline_name)
    components = pipeline_component_types(pipeline_cls)

    assert "transformer" in components
    assert "vae" in components
    assert get_component_class(pipeline_cls, "transformer") is components["transformer"]


@pytest.mark.parametrize(
    "override_slot",
    [None, "transformer", "vae", "text_encoder", "tokenizer", "scheduler", "audio_vae", "audio_scheduler"],
)
def test_h3_repository_build_loads_only_missing_workflow_components(
    loader_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, override_slot: str | None
) -> None:
    from modular_diffusion_nodes_library.standard_parameters import minimax_h3_parameters as module

    params_cls = module.MiniMaxH3PipelineParameters
    pipeline = loader_environment.pipeline
    required = [*params_cls.get_component_slots(), "processor"]
    pipeline.specs = {
        name: SimpleNamespace(
            name=name,
            default_creation_method="from_pretrained",
            pretrained_model_name_or_path="org/base",
            subfolder=name,
            revision=None,
        )
        for name in [*required, "transformer_ref"]
    }
    for name in pipeline.specs:
        setattr(pipeline, name, None)
    pipeline.blocks = SimpleNamespace(
        expected_components=list(pipeline.specs.values()),
        get_workflow=lambda workflow: SimpleNamespace(expected_components=[pipeline.specs[name] for name in required]),
    )
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


def test_audio_component_aliases_keep_model_and_scheduler_types() -> None:
    from modular_diffusion_nodes_library.component_loading.component_slots import (
        component_config_filename,
        slot_artifact_type_name,
    )
    from modular_diffusion_nodes_library.nodes.load_component_node import _LOADABLE_COMPONENTS

    assert _LOADABLE_COMPONENTS["Audio VAE"] == "audio_vae"
    assert slot_artifact_type_name("audio_vae") == "VaeComponentArtifact"
    assert slot_artifact_type_name("audio_scheduler") == "SchedulerComponentArtifact"
    assert component_config_filename("audio_scheduler") == "scheduler_config.json"


@pytest.mark.parametrize("family", ["flux3", "h3"])
@pytest.mark.parametrize("compatible", [False, True])
def test_modular_override_compatibility_uses_architecture_metadata(family: str, compatible: bool) -> None:
    from modular_diffusion_nodes_library.component_loading import component_compatibility as module
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import Flux3PipelineParameters
    from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters

    if family == "flux3":
        params_cls = Flux3PipelineParameters
        layers = 31
        if compatible:
            layers = 32
        configs = {
            "transformer": {"in_channels": 128, "joint_attention_dim": 20480},
            "vae": {"latent_channels": 32},
            "text_encoder": {"hidden_size": 2560, "num_hidden_layers": layers},
        }
    else:
        params_cls = MiniMaxH3PipelineParameters
        audio_channels = 8
        if compatible:
            audio_channels = 16
        configs = {
            "transformer": {"in_channels": 64, "audio_in_channels": 16, "text_dim": 224},
            "vae": {"latent_channels": 64},
            "audio_vae": {"latent_channels": audio_channels},
            "text_encoder": {"hidden_size": 224},
        }

    resolver: Any = SimpleNamespace(resolved_config=configs.get)
    findings = module._run_dimensional_checks(resolver, params_cls)

    if compatible:
        assert findings == []
    else:
        assert len(findings) == 1
