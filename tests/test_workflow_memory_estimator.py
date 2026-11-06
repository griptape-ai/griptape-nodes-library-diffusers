from __future__ import annotations

import pickle
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("safetensors", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("accelerate", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact  # noqa: E402
from modular_diffusion_nodes_library.memory_estimation import pipeline_memory_estimator  # noqa: E402
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import (  # noqa: E402
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
)
from modular_diffusion_nodes_library.tools import estimate_workflow_memory as workflow_memory  # noqa: E402
from modular_diffusion_nodes_library.tools import workflow_extraction  # noqa: E402

LIBRARY_ROOT = Path(__file__).parents[1]

TEST_PIPELINE_NAME = "TestPipeline"
SYNTHETIC_WIDTH = 640
SYNTHETIC_HEIGHT = 480
SYNTHETIC_FRAME_COUNT = 1
TEMPLATE_SIZE = 1024
NON_ALIGNED_WIDTH = 641
NON_ALIGNED_HEIGHT = 479
VIDEO_SOURCE_FRAME_COUNT = 17
IMAGE_LATENT_SHAPE = (1, 16, 60, 80)
VIDEO_LATENT_SHAPE = (1, 16, 5, 60, 80)


def _pickle_expression(value: object) -> str:
    return f"pickle.loads({pickle.dumps(value)!r})"


def _synthetic_workflow(*, pipeline_name: str = TEST_PIPELINE_NAME, value: object | None = None) -> str:
    pipeline = value or DiffusionPipelineArtifact(
        pipeline_name=pipeline_name,
        config_hash="test-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"base_repo_id": "org/model"},
    )
    return f"""# schema_version = "0.20.0"
import pickle

async def build_workflow():
    top_level_unique_values_dict = {{
        "pipeline": {_pickle_expression(pipeline)},
        "width": {_pickle_expression(SYNTHETIC_WIDTH)},
        "height": {_pickle_expression(SYNTHETIC_HEIGHT)},
        "frames": {_pickle_expression(SYNTHETIC_FRAME_COUNT)},
    }}
    node_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type="NoiseLatentNode"))).node_name
    await GriptapeNodes.ahandle_request(SetParameterValueRequest(
        node_name=node_name,
        parameter_name="pipeline",
        value=top_level_unique_values_dict["pipeline"],
    ))
    await GriptapeNodes.ahandle_request(SetParameterValueRequest(
        node_name=node_name,
        parameter_name="width",
        value=top_level_unique_values_dict["width"],
    ))
    await GriptapeNodes.ahandle_request(SetParameterValueRequest(
        node_name=node_name,
        parameter_name="height",
        value=top_level_unique_values_dict["height"],
    ))
    await GriptapeNodes.ahandle_request(SetParameterValueRequest(
        node_name=node_name,
        parameter_name="num_frames",
        value=top_level_unique_values_dict["frames"],
    ))
"""


def test_extract_workflow_does_not_execute_workflow(tmp_path: Path) -> None:
    path = tmp_path / "workflow.py"
    path.write_text(_synthetic_workflow(), encoding="utf-8")

    extracted = workflow_memory.extract_workflow(path)

    assert len(extracted.builders) == 1
    builder = extracted.builders[0]
    assert builder.artifact.pipeline_name == TEST_PIPELINE_NAME
    assert builder.dimensions.width == SYNTHETIC_WIDTH
    assert builder.dimensions.height == SYNTHETIC_HEIGHT
    assert builder.dimensions.num_frames == SYNTHETIC_FRAME_COUNT


def test_extract_repository_text_to_image_template() -> None:
    path = LIBRARY_ROOT / "workflows" / "templates" / "Text2Image.py"

    extracted = workflow_memory.extract_workflow(path)

    assert extracted.builders
    assert extracted.builders[0].artifact.pipeline_name == "ZImagePipeline"
    assert extracted.builders[0].dimensions.width == TEMPLATE_SIZE
    assert extracted.builders[0].dimensions.height == TEMPLATE_SIZE
    assert extracted.builders[0].dimensions.num_frames == 41


def _builder_class_for(canonical_name: str) -> type:
    return next(
        parameters_class
        for provider_parameters_class in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.values()
        for parameters_class in provider_parameters_class.get_pipeline_type_dict().values()
        if parameters_class.pipeline_name() == canonical_name
    )


@pytest.mark.parametrize(
    ("canonical_name", "loaded_class_name"),
    [
        ("FluxPipeline", "FluxPipeline"),
        ("StableDiffusionXLPipeline", "StableDiffusionXLImg2ImgPipeline"),
        ("StableDiffusion3Pipeline", "StableDiffusion3Img2ImgPipeline"),
    ],
)
def test_resolve_pipeline_cls_uses_registered_builder_identity(canonical_name: str, loaded_class_name: str) -> None:
    builder_class = _builder_class_for(canonical_name)
    artifact = DiffusionPipelineArtifact(
        pipeline_name=canonical_name,
        builder_module=builder_class.__module__,
        builder_class_name=builder_class.__name__,
    )

    resolved = pipeline_memory_estimator.resolve_pipeline_cls(artifact)

    assert resolved is builder_class.pipeline_cls()
    assert resolved.__name__ == loaded_class_name


def test_resolve_pipeline_cls_rejects_mismatched_pipeline_name() -> None:
    builder_class = _builder_class_for("FluxPipeline")
    artifact = DiffusionPipelineArtifact(
        pipeline_name="ZImagePipeline",
        builder_module=builder_class.__module__,
        builder_class_name=builder_class.__name__,
    )

    with pytest.raises(ValueError, match="registered for 'FluxPipeline'"):
        pipeline_memory_estimator.resolve_pipeline_cls(artifact)


def test_resolve_pipeline_cls_rejects_loaded_class_name_as_identity() -> None:
    builder_class = _builder_class_for("StableDiffusionXLPipeline")
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionXLImg2ImgPipeline",
        builder_module=builder_class.__module__,
        builder_class_name=builder_class.__name__,
    )

    with pytest.raises(ValueError, match="registered for 'StableDiffusionXLPipeline'"):
        pipeline_memory_estimator.resolve_pipeline_cls(artifact)


class _SlotPipeline:
    _optional_components: list[str] = []


def test_weight_bearing_slots_drops_unloaded_component_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_memory_estimator,
        "_module_component_slots",
        lambda _pipeline_cls: ["text_encoder", "transformer", "transformer_ref", "vae"],
    )

    slots = pipeline_memory_estimator._weight_bearing_slots(  # noqa: SLF001
        _SlotPipeline,
        {},
        all_overrides=False,
        model_index=None,
        unloaded_slots=frozenset({"transformer"}),
    )

    assert slots == ["text_encoder", "transformer_ref", "vae"]


def test_weight_bearing_slots_drops_unloaded_slots_from_all_override_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline_memory_estimator,
        "_module_component_slots",
        lambda _pipeline_cls: ["transformer", "transformer_ref", "vae"],
    )
    overrides = {"transformer": object(), "transformer_ref": object(), "vae": object()}

    slots = pipeline_memory_estimator._weight_bearing_slots(  # noqa: SLF001
        _SlotPipeline,
        overrides,  # type: ignore[arg-type]
        all_overrides=True,
        model_index=None,
        unloaded_slots=frozenset({"transformer_ref"}),
    )

    assert slots == ["transformer", "vae"]


@pytest.mark.parametrize(
    ("canonical_name", "unloaded_slots"),
    [
        ("MiniMaxH3ModularPipeline", frozenset({"transformer_ref"})),
        ("MiniMaxH3Ref2VAModularPipeline", frozenset({"transformer"})),
        ("FluxPipeline", frozenset()),
    ],
)
def test_builder_declares_the_component_slots_its_workflow_never_loads(
    canonical_name: str, unloaded_slots: frozenset[str]
) -> None:
    builder_class = _builder_class_for(canonical_name)

    assert builder_class.get_unloaded_component_slots() == unloaded_slots


def test_minimax_unloaded_slots_match_the_upstream_workflow_components() -> None:
    """fl2va and ref2va share one upstream class whose specs list both transformers; each
    workflow loads only one, so the other must be declared unloaded."""
    from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks

    blocks = MiniMaxH3Blocks()
    all_components = {spec.name for spec in blocks.expected_components}
    for canonical_name in ("MiniMaxH3ModularPipeline", "MiniMaxH3Ref2VAModularPipeline"):
        builder_class = _builder_class_for(canonical_name)
        workflow = builder_class._workflow  # noqa: SLF001
        loaded = {spec.name for spec in blocks.get_workflow(workflow).expected_components}
        unloaded_denoisers = {slot for slot in all_components - loaded if slot.startswith("transformer")}

        assert builder_class.get_unloaded_component_slots() == unloaded_denoisers


def test_serialized_untrusted_builder_identity_is_rejected_before_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_name = f"untrusted_builder_{tmp_path.name.replace('-', '_')}"
    import_marker = tmp_path / "module_imported.txt"
    malicious_module = tmp_path / f"{module_name}.py"
    malicious_module.write_text(
        "from pathlib import Path\n"
        f"Path({str(import_marker)!r}).write_text('imported', encoding='utf-8')\n"
        "class UntrustedBuilder:\n"
        "    @classmethod\n"
        "    def pipeline_cls(cls):\n"
        "        return object\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        builder_module=module_name,
        builder_class_name="UntrustedBuilder",
        build_data={"base_repo_id": "org/model"},
    )
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(_synthetic_workflow(pipeline_name="FluxPipeline", value=artifact), encoding="utf-8")
    extracted = workflow_memory.extract_workflow(workflow_path)

    with pytest.raises(ValueError, match="is not registered"):
        workflow_memory.estimate_workflow(extracted)

    assert not import_marker.exists()


def test_new_workflow_schema_requires_supported_value_expressions(tmp_path: Path) -> None:
    path = tmp_path / "workflow.py"
    path.write_text(
        '# schema_version = "0.21.0"\n'
        "top_level_unique_values_dict = {'k': 123}\n"
        "value = decode_value(top_level_unique_values_dict['k'])\n",
        encoding="utf-8",
    )
    tree = __import__("ast").parse(path.read_text(encoding="utf-8"))
    table = workflow_extraction._literal_value_table(tree)
    expression = tree.body[1].value

    assert workflow_extraction._resolve_literal_expression(expression, table) == 123


def test_new_workflow_rejects_missing_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "workflow.py"
    path.write_text("async def build_workflow():\n    pass\n", encoding="utf-8")

    with pytest.raises(workflow_extraction.WorkflowExtractionError, match="schema_version"):
        workflow_extraction.extract_workflow(path)


def test_resolve_noise_latent_shape_uses_driver_without_snapping(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = DiffusionPipelineArtifact(pipeline_name=TEST_PIPELINE_NAME)
    sentinel_pipe = object()

    class FakeDriver:
        def create_noise_latent(self, source_shape, generator_state):  # noqa: ARG002
            return workflow_memory.LatentArtifact(
                shape=IMAGE_LATENT_SHAPE,
                dtype="torch.bfloat16",
                source_shape=source_shape,
            )

    monkeypatch.setattr(workflow_memory, "_build_shape_only_pipe", lambda value: sentinel_pipe)
    monkeypatch.setattr(workflow_memory, "create_driver", lambda pipe, pipeline_name: FakeDriver())

    assert (
        workflow_memory.resolve_noise_latent_shape(
            artifact,
            width=NON_ALIGNED_WIDTH,
            height=NON_ALIGNED_HEIGHT,
        )
        == IMAGE_LATENT_SHAPE
    )


def test_resolved_image_latent_shape_is_separate_from_source_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = DiffusionPipelineArtifact(pipeline_name=TEST_PIPELINE_NAME)
    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: IMAGE_LATENT_SHAPE)

    latent = workflow_memory._make_latent(artifact, NON_ALIGNED_WIDTH, NON_ALIGNED_HEIGHT, None)

    assert latent.shape == IMAGE_LATENT_SHAPE
    assert latent.source_shape == (1, 3, NON_ALIGNED_HEIGHT, NON_ALIGNED_WIDTH)


def test_video_latent_requires_and_preserves_raw_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoDriver:
        produces_video = True

    monkeypatch.setattr(workflow_memory, "get_driver_class", lambda _: VideoDriver)
    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: VIDEO_LATENT_SHAPE)
    artifact = DiffusionPipelineArtifact(pipeline_name="VideoPipeline")

    latent = workflow_memory._make_latent(
        artifact,
        NON_ALIGNED_WIDTH,
        NON_ALIGNED_HEIGHT,
        VIDEO_SOURCE_FRAME_COUNT,
    )

    assert latent.shape == VIDEO_LATENT_SHAPE
    assert latent.source_shape == (1, 3, VIDEO_SOURCE_FRAME_COUNT, NON_ALIGNED_HEIGHT, NON_ALIGNED_WIDTH)


def test_restricted_unpickler_rejects_unsupported_global() -> None:
    payload = b"cnot_allowed\nThing\n."

    with pytest.raises(workflow_extraction.WorkflowExtractionError, match="Unsupported pickle global"):
        workflow_extraction._restricted_loads(payload)


def test_estimate_workflow_json_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "workflow.py"
    path.write_text(_synthetic_workflow(), encoding="utf-8")

    class Estimate:
        def to_dict(self) -> dict[str, object]:
            return {
                "pipeline_name": TEST_PIPELINE_NAME,
                "estimated_peak_gb": 1.0,
                "components": [],
                "warnings": [],
            }

    workflow = workflow_memory.extract_workflow(path)
    received_latents: list[workflow_memory.LatentArtifact] = []

    def fake_estimator(artifact, latent):
        assert artifact is workflow.builders[0].artifact
        received_latents.append(latent)
        return Estimate()

    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: IMAGE_LATENT_SHAPE)
    monkeypatch.setattr(workflow_memory, "estimate_pipeline_memory_from_artifact", fake_estimator)
    monkeypatch.setattr(workflow_memory, "_is_video_artifact", lambda artifact: False)
    result = workflow_memory.estimate_workflow(workflow)

    assert len(received_latents) == 1
    assert received_latents[0].shape == IMAGE_LATENT_SHAPE
    assert received_latents[0].source_shape == (1, 3, SYNTHETIC_HEIGHT, SYNTHETIC_WIDTH)
    assert result["builders"][0]["dimensions"] == {
        "width": SYNTHETIC_WIDTH,
        "height": SYNTHETIC_HEIGHT,
    }
    assert result["builders"][0]["estimate"] == {
        "estimated_peak_gb": 1.0,
        "components": [],
        "warnings": [],
    }
