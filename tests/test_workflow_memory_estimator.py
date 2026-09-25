from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.tools import estimate_workflow_memory as workflow_memory

LIBRARY_ROOT = Path(__file__).parents[1]


def _pickle_expression(value: object) -> str:
    return f"pickle.loads({pickle.dumps(value)!r})"


def _synthetic_workflow(*, pipeline_name: str = "TestPipeline", value: object | None = None) -> str:
    pipeline = value or DiffusionPipelineArtifact(
        pipeline_name=pipeline_name,
        config_hash="test-hash",
        builder_module="test.module",
        builder_class_name="TestParameters",
        build_data={"base_repo_id": "org/model"},
    )
    values = {
        "pipeline": _pickle_expression(pipeline),
        "width": _pickle_expression(640),
        "height": _pickle_expression(480),
        "frames": _pickle_expression(1),
    }
    return f"""import pickle

async def build_workflow():
    top_level_unique_values_dict = {{
        "pipeline": {values["pipeline"]},
        "width": {values["width"]},
        "height": {values["height"]},
        "frames": {values["frames"]},
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
    assert builder.artifact.pipeline_name == "TestPipeline"
    assert builder.dimensions.width == 640
    assert builder.dimensions.height == 480
    assert builder.dimensions.num_frames == 1


def test_extract_repository_text_to_image_template() -> None:
    path = LIBRARY_ROOT / "workflows" / "templates" / "Text2Image.py"

    extracted = workflow_memory.extract_workflow(path)

    assert extracted.builders
    assert extracted.builders[0].artifact.pipeline_name == "ZImagePipeline"
    assert extracted.builders[0].dimensions.width == 1024
    assert extracted.builders[0].dimensions.height == 1024
    assert extracted.builders[0].dimensions.num_frames == 41


def test_resolve_noise_latent_shape_uses_driver_without_snapping(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline")
    sentinel_pipe = object()

    class FakeDriver:
        def create_noise_latent(self, source_shape, generator_state):  # noqa: ARG002
            return workflow_memory.LatentArtifact(
                shape=(1, 16, 60, 80),
                dtype="torch.bfloat16",
                source_shape=source_shape,
            )

    monkeypatch.setattr(workflow_memory, "_build_shape_only_pipe", lambda value: sentinel_pipe)
    monkeypatch.setattr(workflow_memory, "create_driver", lambda pipe, pipeline_name: FakeDriver())

    assert workflow_memory.resolve_noise_latent_shape(artifact, width=641, height=479) == (1, 16, 60, 80)


def test_resolved_image_latent_shape_is_separate_from_source_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline")
    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: (1, 16, 60, 80))

    latent = workflow_memory._make_latent(artifact, 641, 479, None)

    assert latent.shape == (1, 16, 60, 80)
    assert latent.source_shape == (1, 3, 479, 641)


def test_video_latent_requires_and_preserves_raw_frame_count(monkeypatch: pytest.MonkeyPatch) -> None:
    class VideoDriver:
        produces_video = True

    monkeypatch.setattr(workflow_memory, "get_driver_class", lambda _: VideoDriver)
    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: (1, 16, 5, 60, 80))
    artifact = DiffusionPipelineArtifact(pipeline_name="VideoPipeline")

    latent = workflow_memory._make_latent(artifact, 641, 479, 17)

    assert latent.shape == (1, 16, 5, 60, 80)
    assert latent.source_shape == (1, 3, 17, 479, 641)


def test_restricted_unpickler_rejects_unsupported_global() -> None:
    payload = b"cnot_allowed\nThing\n."

    with pytest.raises(workflow_memory.WorkflowExtractionError, match="Unsupported pickle global"):
        workflow_memory._restricted_loads(payload)


def test_estimate_workflow_json_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "workflow.py"
    path.write_text(_synthetic_workflow(), encoding="utf-8")

    class Estimate:
        def to_dict(self) -> dict[str, object]:
            return {
                "pipeline_name": "TestPipeline",
                "estimated_peak_gb": 1.0,
                "components": [],
                "warnings": [],
            }

    monkeypatch.setattr(workflow_memory, "resolve_noise_latent_shape", lambda *args, **kwargs: (1, 16, 60, 80))
    monkeypatch.setattr(workflow_memory, "estimate_pipeline_memory_from_artifact", lambda artifact, latent: Estimate())
    monkeypatch.setattr(workflow_memory, "_is_video_artifact", lambda artifact: False)
    result = workflow_memory.estimate_workflow(workflow_memory.extract_workflow(path))

    assert result["builders"][0]["dimensions"] == {"width": 640, "height": 480}
    assert result["builders"][0]["estimate"] == {
        "estimated_peak_gb": 1.0,
        "components": [],
        "warnings": [],
    }
