import pytest
import yaml

from modular_diffusion_nodes_library.artifact_utils import pipeline_recipe
from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ComponentSourceType,
    HFRepoRef,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import _write_baked_config
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    deserialize_pipeline_artifact,
    format_pipeline_artifact_summary,
    serialize_pipeline_artifact,
    validate_pipeline_recipe_dependencies,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import TextEncoderComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.tokenizer_component_artifact import TokenizerComponentArtifact
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import (
    LoraPipelineRuntimeAdapterStep,
    lora_adapter_name,
)
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec
from modular_diffusion_nodes_library.utils.pipeline_runtime_adapter_step import PipelineRuntimeAdapterStep


def test_serialize_base_pipeline_artifact() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"repo_id": "org/model", "revision": "commit-hash"},
        loras={"/tmp/style.safetensors": 0.8},
        optimization_kwargs={"vae_slicing": True},
    )

    serialized = serialize_pipeline_artifact(artifact)
    document = yaml.safe_load(serialized)

    assert serialized.startswith("schema_version: 1\npipeline_name: TestPipeline\n")
    assert document == {
        "schema_version": 1,
        "pipeline_name": "TestPipeline",
        "config_hash": "pipeline-hash",
        "builder_module": "test.module",
        "builder_class_name": "TestParameters",
        "build_data": {"repo_id": "org/model", "revision": "commit-hash"},
        "loras": {"/tmp/style.safetensors": 0.8},
        "runtime_loras": [],
        "optimization": {"vae_slicing": True},
        "is_prequantized": False,
        "supports_layerwise_casting": True,
        "requires_device_map": False,
    }


def test_serialize_pipeline_with_build_data_error_is_refused() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        build_data_error="Model parameters are invalid.",
    )

    with pytest.raises(
        ValueError,
        match=r"Pipeline cannot be saved because its build data is unavailable: Model parameters are invalid\.",
    ):
        serialize_pipeline_artifact(artifact)


def test_config_only_and_baked_manifests_share_artifact_shape(tmp_path) -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"repo_id": "org/model", "revision": "commit-hash"},
        optimization_kwargs={"vae_slicing": True},
    )
    config_only = yaml.safe_load(serialize_pipeline_artifact(artifact))
    _write_baked_config(tmp_path, artifact, "float16", packed_quant_slots=["unet"])
    baked = yaml.safe_load((tmp_path / "baked-config.yaml").read_text(encoding="utf-8"))

    common_keys = (
        "pipeline_name",
        "builder_module",
        "builder_class_name",
        "optimization",
        "supports_layerwise_casting",
        "requires_device_map",
    )
    assert {key: baked[key] for key in common_keys} == {key: config_only[key] for key in common_keys}
    assert baked["torch_dtype"] == "float16"
    assert baked["baked_quantization_mode"] is None
    assert baked["packed_quant_slots"] == ["unet"]


def test_serialize_controlnet_pipeline_artifact_is_refused() -> None:
    base_artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="base-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
    )
    artifact = ControlNetDiffusionPipelineArtifact(
        base_artifact=base_artifact,
        controlnet_models=["org/controlnet"],
        config_hash="controlnet-hash",
    )

    with pytest.raises(ValueError, match="ControlNet pipelines cannot be saved"):
        serialize_pipeline_artifact(artifact)


def _with_runtime_lora(
    artifact: DiffusionPipelineArtifact, path: str, weight: float = 1.0
) -> DiffusionPipelineArtifact:
    step = LoraPipelineRuntimeAdapterStep({path: LoraSpec(path=path, weight=weight)})
    return artifact.with_additional_runtime_adapter_steps([step])


def test_runtime_loras_round_trip_through_recipe() -> None:
    artifact = _with_runtime_lora(
        DiffusionPipelineArtifact(
            pipeline_name="TestPipeline", config_hash="base-hash", loras={"/tmp/style.safetensors": 0.8}
        ),
        "/tmp/hdr-ic-lora.safetensors",
        weight=0.9,
    )

    document = yaml.safe_load(serialize_pipeline_artifact(artifact))
    loaded = deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact))

    assert document["runtime_loras"] == [{"path": "/tmp/hdr-ic-lora.safetensors", "weight": 0.9}]
    assert loaded.config_hash == "base-hash"
    assert loaded.loras == {"/tmp/style.safetensors": 0.8}
    assert loaded.runtime_loras() == {
        "/tmp/hdr-ic-lora.safetensors": LoraSpec(path="/tmp/hdr-ic-lora.safetensors", weight=0.9)
    }
    (step,) = loaded.runtime_adapter_steps()
    assert isinstance(step, LoraPipelineRuntimeAdapterStep)


def test_runtime_lora_recipe_preserves_macro_path() -> None:
    artifact = _with_runtime_lora(
        DiffusionPipelineArtifact(pipeline_name="TestPipeline", config_hash="base-hash"),
        "{project_dir}/loras/style.safetensors",
        weight=0.75,
    )

    document = yaml.safe_load(serialize_pipeline_artifact(artifact))

    assert document["runtime_loras"] == [
        {
            "path": "{project_dir}/loras/style.safetensors",
            "weight": 0.75,
        }
    ]


def test_recipe_without_runtime_loras_loads_with_none() -> None:
    document = yaml.safe_load(
        serialize_pipeline_artifact(DiffusionPipelineArtifact(pipeline_name="TestPipeline", config_hash="h"))
    )
    del document["runtime_loras"]

    assert deserialize_pipeline_artifact(yaml.safe_dump(document)).runtime_adapter_steps() == []


def test_serialize_chained_lora_pipeline_nodes_is_refused() -> None:
    artifact = _with_runtime_lora(
        _with_runtime_lora(DiffusionPipelineArtifact(pipeline_name="TestPipeline"), "/tmp/a.safetensors"),
        "/tmp/b.safetensors",
    )

    with pytest.raises(ValueError, match="more than one LoRA Pipeline node"):
        serialize_pipeline_artifact(artifact)


def test_serialize_non_lora_runtime_adapter_is_refused() -> None:
    class _OtherStep(PipelineRuntimeAdapterStep):
        KIND = "other_adapter"

        def _metadata(self) -> dict:
            return {}

        @classmethod
        def _from_metadata(cls, data: dict) -> "_OtherStep":
            return cls()

        def activate(self, pipe, *, node_name=None): ...

    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline").with_additional_runtime_adapter_steps(
        [_OtherStep()]
    )

    with pytest.raises(ValueError, match="'other_adapter' runtime adapter cannot be saved"):
        serialize_pipeline_artifact(artifact)


def test_deserialize_pipeline_artifact_round_trip() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"repo_id": "org/model", "revision": "commit-hash"},
        loras={"/tmp/style.safetensors": 0.8},
        optimization_kwargs={"vae_slicing": True},
    )

    loaded_artifact = deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact))

    assert loaded_artifact == artifact


def test_pipeline_artifact_recipe_dict_round_trips_directly() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"repo_id": "org/model", "revision": "commit-hash"},
        loras={"/tmp/style.safetensors": 0.8},
        optimization_kwargs={"vae_slicing": True},
    )

    assert DiffusionPipelineArtifact.from_recipe_dict(artifact.to_recipe_dict()) == artifact


def test_pipeline_artifact_with_optimization_overrides_updates_cache_identity() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        optimization_kwargs={"vae_slicing": False, "quantization_mode": "int8"},
    )

    updated = artifact.with_optimization_overrides({"vae_slicing": True})

    assert artifact.optimization_kwargs == {"vae_slicing": False, "quantization_mode": "int8"}
    assert updated.optimization_kwargs == {"vae_slicing": True, "quantization_mode": "int8"}
    assert updated.config_hash != artifact.config_hash
    assert artifact.with_optimization_overrides({}) is artifact


def test_component_artifact_recipe_dict_round_trips_directly() -> None:
    override = ModelComponentArtifact(
        load_id="override-id",
        source_type=ComponentSourceType.HF_REPO,
        component="unet",
        repo_ref=HFRepoRef(repo_id="org/model", revision="commit-hash", subfolder="unet"),
    )

    assert ComponentArtifact.from_recipe_dict(override.to_recipe_dict()) == override


def test_component_artifact_recipe_normalizes_local_config_directory(tmp_path) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text("{}\n", encoding="utf-8")
    override = ModelComponentArtifact(
        load_id="override-id",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="unet",
        file_path="/models/unet.gguf",
        config_source=str(tmp_path),
    )

    recipe = override.to_recipe_dict()

    assert recipe["config_source"] == str(config_file)


def test_component_artifact_recipe_preserves_config_repo_id() -> None:
    override = ModelComponentArtifact(
        load_id="override-id",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="unet",
        file_path="/models/unet.gguf",
        config_source="org/model",
    )

    assert override.to_recipe_dict()["config_source"] == "org/model"


def test_deserialize_pipeline_artifact_round_trips_component_overrides() -> None:
    override = ModelComponentArtifact(
        load_id="override-id",
        source_type=ComponentSourceType.HF_REPO,
        component="unet",
        repo_ref=HFRepoRef(repo_id="org/model", revision="commit-hash", subfolder="unet"),
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"_component_overrides": {"unet": override}},
    )

    loaded_artifact = deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact))

    assert loaded_artifact == artifact


@pytest.mark.parametrize(
    ("slot", "override"),
    [
        pytest.param(
            "transformer",
            ModelComponentArtifact(
                load_id="model-override",
                source_type=ComponentSourceType.HF_REPO,
                component="transformer",
                repo_ref=HFRepoRef(repo_id="org/model", revision="model-revision", subfolder="transformer"),
            ),
            id="model",
        ),
        pytest.param(
            "text_encoder",
            TextEncoderComponentArtifact(
                load_id="text-encoder-override",
                source_type=ComponentSourceType.SINGLE_FILE,
                component="text_encoder",
                file_path="/models/text-encoder.gguf",
                config_source="/models/config.json",
            ),
            id="text-encoder",
        ),
        pytest.param(
            "tokenizer",
            TokenizerComponentArtifact(
                load_id="tokenizer-override",
                source_type=ComponentSourceType.LOCAL_DIR,
                component="tokenizer",
                file_path="/models/tokenizer",
            ),
            id="tokenizer",
        ),
        pytest.param(
            "scheduler",
            SchedulerComponentArtifact(
                load_id="scheduler-override",
                source_type=ComponentSourceType.RAW_CONFIG,
                component="scheduler",
                scheduler_class="FlowMatchEulerDiscreteScheduler",
                text_config={"num_train_timesteps": 1000, "shift": 5.0},
            ),
            id="scheduler",
        ),
    ],
)
def test_component_override_subtypes_round_trip_through_yaml(slot: str, override: ComponentArtifact) -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="pipeline-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"_component_overrides": {slot: override}},
    )

    serialized = serialize_pipeline_artifact(artifact)
    document = yaml.safe_load(serialized)
    loaded = deserialize_pipeline_artifact(serialized)
    loaded_override = loaded.build_data["_component_overrides"][slot]

    assert document["build_data"]["_component_overrides"][slot]["artifact_type"] == type(override).__name__
    assert type(loaded_override) is type(override)
    assert loaded_override == override


def test_validate_pipeline_recipe_dependencies_reports_unavailable_items(monkeypatch) -> None:
    def cached_revisions(repo_id: str) -> list[tuple[str, str]]:
        if repo_id == "org/base":
            return [(repo_id, "cached-revision")]
        return []

    monkeypatch.setattr(pipeline_recipe, "_list_cached_repo_revisions", cached_revisions)
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data={"base_repo_id": "org/base", "base_revision": "missing-revision"},
        loras={"/missing/style.safetensors": 0.8},
    )
    artifact = _with_runtime_lora(artifact, "/missing/ic-lora.safetensors")

    issues = validate_pipeline_recipe_dependencies(artifact)

    assert issues == [
        "LoRA file is unavailable: /missing/style.safetensors",
        (
            f"Runtime LoRA adapter '{lora_adapter_name('/missing/ic-lora.safetensors')}' "
            "file is unavailable: /missing/ic-lora.safetensors"
        ),
        "Repository revision is unavailable locally: org/base (missing-revision)",
    ]


def test_validate_pipeline_recipe_dependencies_accepts_cached_items(monkeypatch, tmp_path) -> None:
    lora_path = tmp_path / "style.safetensors"
    lora_path.touch()
    monkeypatch.setattr(
        pipeline_recipe,
        "_list_cached_repo_revisions",
        lambda repo_id: [(repo_id, "cached-revision")],
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data={"base_repo_id": "org/base", "base_revision": "cached-revision"},
        loras={str(lora_path): 0.8},
    )

    assert validate_pipeline_recipe_dependencies(artifact) == []


def test_validate_pipeline_recipe_dependencies_checks_base_repo_revision(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_recipe,
        "_list_cached_repo_revisions",
        lambda repo_id: [(repo_id, "cached-revision")],
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        build_data={"base_repo_id": "org/base", "revision": "missing-revision"},
    )

    assert validate_pipeline_recipe_dependencies(artifact) == [
        "Repository revision is unavailable locally: org/base (missing-revision)"
    ]


@pytest.mark.parametrize(
    ("scheduler_kwargs", "expected_lines"),
    [
        (
            {"source_type": ComponentSourceType.RAW_CONFIG, "text_config": {"shift": 5.0}},
            ["    Config: Inline"],
        ),
        (
            {"source_type": ComponentSourceType.LOCAL_DIR, "config_source": "/models/scheduler_config.json"},
            ["    Path: /models/scheduler_config.json"],
        ),
        (
            {
                "source_type": ComponentSourceType.HF_REPO,
                "repo_ref": HFRepoRef(repo_id="org/model", revision="commit-hash", subfolder="scheduler"),
            },
            ["    Model: org/model", "    Revision: commit-hash", "    Subfolder: scheduler"],
        ),
    ],
)
def test_loaded_summary_describes_scheduler_override(scheduler_kwargs, expected_lines) -> None:
    scheduler = SchedulerComponentArtifact(
        load_id="scheduler-load-id",
        component="scheduler",
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        **scheduler_kwargs,
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="scheduler-hash",
        build_data={"_component_overrides": {"scheduler": scheduler}, "_all_overrides": True},
    )

    summary = format_pipeline_artifact_summary(deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact)))

    assert "    Class: FlowMatchEulerDiscreteScheduler" in summary.splitlines()
    for line in expected_lines:
        assert line in summary.splitlines()


class _OverridesOnlyPipeline:
    def __init__(self, **components) -> None:
        self.components = components


class _OverridesOnlyBuilder(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls_path = f"{__name__}:_OverridesOnlyPipeline"

    def add_input_parameters(self) -> None: ...

    def remove_input_parameters(self) -> None: ...

    def get_config_kwargs(self) -> dict: ...

    def validate_before_node_run(self) -> None: ...

    def get_build_data(self) -> dict: ...

    @classmethod
    def _build_pipeline_from_repo(cls, build_data, overrides): ...


def test_overrides_only_pipeline_saves_and_rebuilds_with_builder_class() -> None:
    scheduler = SchedulerComponentArtifact(
        load_id="scheduler-load-id",
        source_type=ComponentSourceType.RAW_CONFIG,
        component="scheduler",
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        text_config={"num_train_timesteps": 1000, "shift": 5.0},
    )
    artifact = DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="overrides-only-hash",
        builder_module=__name__,
        builder_class_name=_OverridesOnlyBuilder.__name__,
        build_data={"_component_overrides": {"scheduler": scheduler}, "_all_overrides": True},
    )

    loaded = deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact))
    pipe = _OverridesOnlyBuilder.build_pipeline_from_build_data(loaded.build_data)

    assert isinstance(pipe, _OverridesOnlyPipeline)
    assert type(pipe.components["scheduler"]).__name__ == "FlowMatchEulerDiscreteScheduler"
    assert pipe.components["scheduler"].config.shift == 5.0
