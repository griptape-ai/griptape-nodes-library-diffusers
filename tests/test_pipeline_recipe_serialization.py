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
    serialize_pipeline_artifact,
    validate_pipeline_recipe_dependencies,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)


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


def test_serialize_pipeline_with_runtime_adapters_fails() -> None:
    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline")
    artifact._extra_runtime_adapter_steps = [object()]

    with pytest.raises(ValueError, match="runtime adapter steps"):
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

    issues = validate_pipeline_recipe_dependencies(artifact)

    assert issues == [
        "LoRA file is unavailable: /missing/style.safetensors",
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
