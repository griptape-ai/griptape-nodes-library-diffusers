import json

import pytest

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
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    deserialize_pipeline_artifact,
    format_pipeline_artifact_summary,
    serialize_pipeline_artifact,
    validate_pipeline_recipe_dependencies,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
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

    document = json.loads(serialize_pipeline_artifact(artifact))

    assert document == {
        "schema_version": 1,
        "artifact": {
            "type": "base",
            "pipeline_name": "TestPipeline",
            "config_hash": "pipeline-hash",
            "builder": {"module": "test.module", "class_name": "TestParameters"},
            "build_data": {"repo_id": "org/model", "revision": "commit-hash"},
            "build_data_error": None,
            "loras": {"/tmp/style.safetensors": 0.8},
            "runtime_loras": [],
            "optimization": {"vae_slicing": True},
            "flags": {
                "is_prequantized": False,
                "supports_layerwise_casting": True,
                "requires_device_map": False,
            },
        },
    }


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


def _with_runtime_lora(artifact: DiffusionPipelineArtifact, path: str, weight: float = 1.0) -> DiffusionPipelineArtifact:
    step = LoraPipelineRuntimeAdapterStep({path: LoraSpec(path=path, weight=weight)})
    return artifact.with_additional_runtime_adapter_steps([step])


def test_runtime_loras_round_trip_through_recipe() -> None:
    artifact = _with_runtime_lora(
        DiffusionPipelineArtifact(pipeline_name="TestPipeline", config_hash="base-hash", loras={"/tmp/style.safetensors": 0.8}),
        "/tmp/hdr-ic-lora.safetensors",
        weight=0.9,
    )

    document = json.loads(serialize_pipeline_artifact(artifact))
    loaded = deserialize_pipeline_artifact(serialize_pipeline_artifact(artifact))

    assert document["artifact"]["runtime_loras"] == [{"path": "/tmp/hdr-ic-lora.safetensors", "weight": 0.9}]
    assert loaded.config_hash == "base-hash"
    assert loaded.loras == {"/tmp/style.safetensors": 0.8}
    assert loaded.runtime_loras() == {
        "/tmp/hdr-ic-lora.safetensors": LoraSpec(path="/tmp/hdr-ic-lora.safetensors", weight=0.9)
    }
    (step,) = loaded.runtime_adapter_steps()
    assert isinstance(step, LoraPipelineRuntimeAdapterStep)


def test_recipe_without_runtime_loras_loads_with_none() -> None:
    document = json.loads(serialize_pipeline_artifact(DiffusionPipelineArtifact(pipeline_name="TestPipeline", config_hash="h")))
    del document["artifact"]["runtime_loras"]

    assert deserialize_pipeline_artifact(json.dumps(document)).runtime_adapter_steps() == []


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

        def activate(self, pipe, *, node_name=None): ...

    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline").with_additional_runtime_adapter_steps([_OtherStep()])

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


def test_component_artifact_recipe_dict_round_trips_directly() -> None:
    override = ModelComponentArtifact(
        load_id="override-id",
        source_type=ComponentSourceType.HF_REPO,
        component="unet",
        repo_ref=HFRepoRef(repo_id="org/model", revision="commit-hash", subfolder="unet"),
    )

    assert ComponentArtifact.from_recipe_dict(override.to_recipe_dict()) == override


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
        "Runtime LoRA file is unavailable: /missing/ic-lora.safetensors",
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
    _pipeline_cls = _OverridesOnlyPipeline

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
