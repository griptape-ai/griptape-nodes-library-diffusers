from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
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


@pytest.fixture
def flux3_vae_override(loader_environment: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> Any:
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.utils import pipeline_utils

    vae = diffusers.AutoencoderKLFlux2(
        block_out_channels=(8, 8, 8, 8),
        norm_num_groups=4,
        layers_per_block=1,
        latent_channels=32,
        sample_size=64,
        mid_block_add_attention=False,
    )
    pipe = loader_environment.make_loader(workflows=("active",), device="cpu").from_pretrained(
        "org/base", overrides={"transformer": object(), "vae": vae}
    )
    assert pipe.vae is vae
    monkeypatch.setattr(pipeline_utils, "get_best_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cuda, "sdp_kernel", lambda **kwargs: None)
    return pipe


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_vae_override_slicing_splits_actual_decoder_batches(
    flux3_vae_override: Any, monkeypatch: pytest.MonkeyPatch, enabled: bool, batch_size: int
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

    pipe = flux3_vae_override
    vae = pipe.vae
    batches: list[int] = []
    scale = 2 ** (len(vae.config.block_out_channels) - 1)

    def decode(latents: Any) -> Any:
        batches.append(latents.shape[0])
        return latents.new_zeros((latents.shape[0], 3, latents.shape[-2] * scale, latents.shape[-1] * scale))

    monkeypatch.setattr(vae.decoder, "forward", decode)
    optimize_diffusion_pipeline(pipe, vae_slicing=enabled)

    output = vae.decode(torch.zeros((batch_size, 32, 8, 8)), return_dict=False)[0]

    expected_batches = [batch_size]
    if enabled and batch_size > 1:
        expected_batches = [1] * batch_size
    assert vae.use_slicing is enabled
    assert batches == expected_batches
    assert output.shape == (batch_size, 3, 8 * scale, 8 * scale)


@pytest.mark.parametrize("enabled", [False, True], ids=["disabled", "enabled"])
@pytest.mark.parametrize("offset", [-1, 0, 1], ids=["below", "at", "above"])
@pytest.mark.parametrize("axis", [-2, -1], ids=["height", "width"])
def test_vae_override_tiling_splits_actual_decoder_inputs(
    flux3_vae_override: Any, monkeypatch: pytest.MonkeyPatch, enabled: bool, offset: int, axis: int
) -> None:
    torch = pytest.importorskip("torch")
    from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

    pipe = flux3_vae_override
    vae = pipe.vae
    decoded_shapes: list[tuple[int, ...]] = []
    scale = 2 ** (len(vae.config.block_out_channels) - 1)

    def decode(latents: Any) -> Any:
        decoded_shapes.append(tuple(latents.shape))
        return latents.new_zeros((latents.shape[0], 3, latents.shape[-2] * scale, latents.shape[-1] * scale))

    monkeypatch.setattr(vae.decoder, "forward", decode)
    optimize_diffusion_pipeline(pipe, vae_tiling=enabled)
    threshold = vae.tile_latent_min_size
    shape = [1, 32, threshold, threshold]
    shape[axis] += offset

    output = vae.decode(torch.zeros(shape), return_dict=False)[0]

    assert vae.use_tiling is enabled
    assert output.shape == (1, 3, shape[-2] * scale, shape[-1] * scale)
    if enabled and offset > 0:
        assert len(decoded_shapes) > 1
        assert all(height <= threshold and width <= threshold for _, _, height, width in decoded_shapes)
    else:
        assert decoded_shapes == [tuple(shape)]


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


@pytest.mark.parametrize("subfolder", [None, "vae"], ids=["target-slot", "explicit-subfolder"])
def test_audio_override_validation_reads_the_same_config_as_materialisation(
    monkeypatch: pytest.MonkeyPatch, subfolder: str | None
) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    configs = {"vae": {"latent_channels": 64}, "audio_vae": {"latent_channels": 32}}
    loaded: dict[str, Any] = {}

    def cached(repo_id: str, filename: str, revision: str | None = None) -> str:
        return str(Path("/virtual") / filename)

    def load(**kwargs: Any) -> object:
        loaded.update(kwargs)
        return object()

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cached)
    monkeypatch.setattr(artifacts, "try_load_json_dict", lambda path: configs[Path(path).parent.name])
    pipeline_cls = diffusers.MiniMaxH3ModularPipeline
    monkeypatch.setattr(get_component_class(pipeline_cls, "audio_vae"), "from_pretrained", load)
    artifact = artifacts.ModelComponentArtifact(
        load_id="test-audio-slot",
        source_type=artifacts.ComponentSourceType.HF_REPO,
        component="vae",
        repo_ref=artifacts.HFRepoRef("org/h3", subfolder=subfolder),
    )
    resolver = compatibility._ConfigResolver({"audio_vae": artifact}, "org/base")

    config = resolver.resolved_config("audio_vae")
    artifact.materialize(pipeline_cls=pipeline_cls, slot="audio_vae")

    expected_folder = "audio_vae"
    if subfolder is not None:
        expected_folder = subfolder
    assert loaded["subfolder"] == expected_folder
    assert config == configs[expected_folder]


def test_h3_single_file_override_resolves_its_canonical_config(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    pipeline_cls = diffusers.MiniMaxH3ModularPipeline
    component_cls = get_component_class(pipeline_cls, "transformer")
    checkpoint = {"test-weight": object()}
    component = object()

    def resolve(model_type: str, model_cls: type, config_source: str | None, **kwargs: Any) -> Path:
        assert model_type == "minimax-h3"
        assert model_cls is component_cls
        assert config_source is None
        assert kwargs["pipeline_slot"] == "transformer"
        return Path("/virtual/transformer")

    def load(**kwargs: Any) -> object:
        assert kwargs["config"] == "/virtual/transformer"
        assert kwargs["pretrained_model_link_or_path_or_dict"] is checkpoint
        assert kwargs["local_files_only"] is True
        return component

    monkeypatch.setattr("diffusers.loaders.single_file_utils.load_single_file_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr("diffusers.loaders.single_file_utils.infer_diffusers_model_type", lambda weights: "minimax-h3")
    monkeypatch.setattr(artifacts, "resolve_config_path", resolve)
    monkeypatch.setattr(component_cls, "from_single_file", load)
    artifact = artifacts.ModelComponentArtifact(
        load_id="test-h3-single-file",
        source_type=artifacts.ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path="/virtual/h3.safetensors",
    )

    assert artifact.materialize(pipeline_cls=pipeline_cls, slot="transformer") is component


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


@pytest.fixture
def h3_implicit_gguf_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters

    configs = {
        "transformer": {"in_channels": 24, "audio_in_channels": 32, "text_dim": 5120},
        "vae": {"latent_channels": 24},
        "audio_vae": {"latent_channels": 32},
        "text_encoder": {"text_config": {"hidden_size": 5120}},
    }
    overrides: dict[str, Any] = {
        "transformer": artifacts.ModelComponentArtifact(
            load_id="test-h3-config-only",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="transformer",
            file_path="/unused/h3.gguf",
        )
    }
    monkeypatch.setattr(compatibility, "_read_base_repo_config", lambda repo, slot, revision: configs.get(slot))
    params = MiniMaxH3PipelineParameters.__new__(MiniMaxH3PipelineParameters)
    params.get_build_data = lambda: params.build_data("MiniMaxAI/MiniMax-H3", "selected")
    params.validate_before_node_run = lambda: None
    return SimpleNamespace(configs=configs, overrides=overrides, params=params)


@pytest.mark.parametrize(
    ("slot", "config", "message"),
    [
        ("vae", {"latent_channels": 16}, "VAE latent_channels"),
        ("audio_vae", {"latent_channels": 16}, "Audio VAE latent_channels"),
        ("text_encoder", {"text_config": {"hidden_size": 4096}}, "Text encoder output dimension"),
    ],
)
def test_implicit_gguf_config_still_blocks_incompatible_overrides_before_loading(
    h3_implicit_gguf_config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slot: str,
    config: dict[str, Any],
    message: str,
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )

    (tmp_path / "config.json").write_text(json.dumps(config))
    state = h3_implicit_gguf_config
    state.overrides[slot] = artifacts.ModelComponentArtifact(
        load_id=f"test-incompatible-{slot}",
        source_type=artifacts.ComponentSourceType.LOCAL_DIR,
        component=slot,
        file_path=str(tmp_path),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Compatibility validation must block before artifact construction or weight loading")

    monkeypatch.setattr(artifacts.ModelComponentArtifact, "materialize", forbidden)
    node: Any = LatentDiffusionPipelineBuilderNode.__new__(LatentDiffusionPipelineBuilderNode)
    node.params = SimpleNamespace(
        pipeline_type_parameters=SimpleNamespace(pipeline_type_pipeline_params=state.params),
        component_override_params=SimpleNamespace(get_component_overrides=lambda: state.overrides),
    )
    node.get_pipeline_artifact_or_raise = forbidden

    errors = node.validate_before_node_run()

    assert errors is not None
    assert len(errors) == 1
    assert message in str(errors[0])
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


def test_implicit_gguf_config_keeps_explicit_config_authoritative(
    h3_implicit_gguf_config: SimpleNamespace, tmp_path: Path
) -> None:
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    config = {"in_channels": 16, "audio_in_channels": 16, "text_dim": 4096}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    state = h3_implicit_gguf_config
    state.overrides["transformer"] = replace(state.overrides["transformer"], config_source=str(config_file))
    resolver = _ConfigResolver(state.overrides, "MiniMaxAI/MiniMax-H3", "selected")

    assert resolver.resolved_config("transformer") == config


@pytest.mark.parametrize("source", ["single-file", "directory"])
def test_missing_explicit_or_directory_config_is_not_replaced_by_base_config(
    h3_implicit_gguf_config: SimpleNamespace, source: str
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    source_type = ComponentSourceType.SINGLE_FILE
    config_source = "/missing/explicit/config.json"
    if source == "directory":
        source_type = ComponentSourceType.LOCAL_DIR
        config_source = None
    state = h3_implicit_gguf_config
    state.overrides["transformer"] = replace(
        state.overrides["transformer"], source_type=source_type, config_source=config_source
    )
    resolver = _ConfigResolver(state.overrides, "MiniMaxAI/MiniMax-H3", "selected")

    assert resolver.resolved_config("transformer") is None


@pytest.mark.parametrize("config_source", [None, "local", "hf"], ids=["implicit", "explicit-local", "explicit-repo"])
def test_text_encoder_gguf_validation_uses_the_adjacent_config_used_by_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_source: str | None
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.artifact_utils import text_encoder_component_artifact as text_artifacts
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    adjacent = {"text_config": {"hidden_size": 4096, "num_hidden_layers": 24}}
    other = {"text_config": {"hidden_size": 5120, "num_hidden_layers": 36}}
    (tmp_path / "config.json").write_text(json.dumps(adjacent))
    checkpoint = tmp_path / "encoder.gguf"
    checkpoint.touch()
    if config_source == "local":
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps(other))
        config_source = str(explicit)
    elif config_source == "hf":
        config_source = "org/other-encoder"

    class Encoder:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["gguf_file"] == checkpoint.name
            assert kwargs["local_files_only"] is True
            return json.loads((Path(source) / "config.json").read_text())

    monkeypatch.setattr(artifacts, "get_loadable_component_class", lambda *args: Encoder)
    monkeypatch.setattr(text_artifacts, "get_loadable_component_class", lambda *args: Encoder)
    artifact = text_artifacts.TextEncoderComponentArtifact(
        load_id="test-adjacent-encoder",
        source_type=artifacts.ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        file_path=str(checkpoint),
        config_source=config_source,
    )
    resolver = _ConfigResolver({"text_encoder": artifact}, "org/base")

    assert resolver.resolved_config("text_encoder") == adjacent
    assert artifact.materialize(pipeline_cls=Encoder, slot="text_encoder") == adjacent


@pytest.mark.parametrize(
    ("hidden_size", "num_hidden_layers", "target_dim", "expected_types"),
    [
        (1280, 36, 20480, ["TextEncoderDenoiserDimMismatch"]),
        (2560, 24, 20480, ["TextEncoderHiddenLayerMismatch"]),
        (1280, 24, 20480, ["TextEncoderDenoiserDimMismatch", "TextEncoderHiddenLayerMismatch"]),
        (1280, 36, 10240, []),
    ],
    ids=["width-mismatch", "insufficient-layers", "both-mismatches", "compatible-override-pair"],
)
def test_text_encoder_gguf_compatibility_checks_adjacent_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hidden_size: int,
    num_hidden_layers: int,
    target_dim: int,
    expected_types: list[str],
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (
        TextEncoderComponentArtifact,
    )
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import Flux3PipelineParameters

    adjacent = {"text_config": {"hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers}}
    (tmp_path / "config.json").write_text(json.dumps(adjacent))
    base_configs = {
        "transformer": {"in_channels": 128, "joint_attention_dim": 20480},
        "vae": {"latent_channels": 32},
        "text_encoder": {"text_config": {"hidden_size": 2560, "num_hidden_layers": 36}},
    }
    monkeypatch.setattr(compatibility, "_read_base_repo_config", lambda repo, slot, revision: base_configs.get(slot))
    overrides: dict[str, artifacts.ComponentArtifact] = {
        "text_encoder": TextEncoderComponentArtifact(
            load_id="test-adjacent-dimensions",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="text_encoder",
            file_path=str(tmp_path / "encoder.gguf"),
        )
    }
    if target_dim != base_configs["transformer"]["joint_attention_dim"]:
        transformer_config = tmp_path / "transformer.json"
        transformer_config.write_text(json.dumps({"in_channels": 128, "joint_attention_dim": target_dim}))
        overrides["transformer"] = artifacts.ModelComponentArtifact(
            load_id="test-compatible-transformer",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="transformer",
            file_path="/unused/transformer.gguf",
            config_source=str(transformer_config),
        )

    findings = compatibility.evaluate_component_compatibility(
        overrides, "org/base", pipeline_params_cls=Flux3PipelineParameters
    )

    assert sorted(type(finding).__name__ for finding in findings) == sorted(expected_types)


@pytest.mark.parametrize("adjacent_config", ["missing", "invalid"])
def test_text_encoder_gguf_does_not_borrow_base_config_when_adjacent_config_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adjacent_config: str
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType
    from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (
        TextEncoderComponentArtifact,
    )
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility

    if adjacent_config == "invalid":
        (tmp_path / "config.json").write_text("{invalid json")
    base_reads: list[str] = []

    def base_config(repo: str, slot: str, revision: str | None) -> dict[str, Any]:
        base_reads.append(slot)
        return {"text_config": {"hidden_size": 5120, "num_hidden_layers": 36}}

    monkeypatch.setattr(compatibility, "_read_base_repo_config", base_config)
    artifact = TextEncoderComponentArtifact(
        load_id="test-unavailable-adjacent",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        file_path=str(tmp_path / "encoder.gguf"),
    )
    resolver = compatibility._ConfigResolver({"text_encoder": artifact}, "org/base")

    assert resolver.resolved_config("text_encoder") is None
    assert base_reads == []
