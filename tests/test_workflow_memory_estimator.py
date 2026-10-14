from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.tools import estimate_workflow_memory as workflow_memory
from modular_diffusion_nodes_library.tools import workflow_extraction

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
    return f"""import pickle

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
