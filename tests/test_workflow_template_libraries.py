"""Keep templates usable with only this library and the optional standard library."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from griptape_nodes.node_library.workflow_registry import read_workflow_metadata
from griptape_nodes.retained_mode.events.base_events import RequestPayload
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.utils import path_macros
from tests.workflows.dependencies import extract_workflow_dependencies
from tests.workflows.workflow_configs import get_workflow_paths

DIFFUSION_LIBRARY = "Griptape Modular Diffusion Nodes Library"
STANDARD_LIBRARY = "Griptape Nodes Library"
ALLOWED_LIBRARIES = {DIFFUSION_LIBRARY, STANDARD_LIBRARY}


@pytest.mark.parametrize(
    "template_path", sorted(Path(path) for path in get_workflow_paths()), ids=lambda path: path.name
)
def test_workflow_templates_only_use_diffusion_and_standard_libraries(template_path: Path) -> None:
    # Deliberately metadata-only since the engine uses the metadata alone to decide which libraries a workflow needs.
    metadata = read_workflow_metadata(template_path)
    declared_libraries = {library.library_name for library in metadata.node_libraries_referenced}
    assert DIFFUSION_LIBRARY in declared_libraries, (
        f"{template_path.name}: node_libraries_referenced must include {DIFFUSION_LIBRARY!r}"
    )
    unexpected = sorted(declared_libraries - ALLOWED_LIBRARIES)
    assert not unexpected, f"{template_path.name}: references disallowed libraries: {unexpected}"


def test_minimax_h3_template_is_listed_in_catalog() -> None:
    root = Path(__file__).parents[1]
    manifest = json.loads((root / "griptape-nodes-library.json").read_text())
    template = root / "workflows/templates/MiniMaxH3Text2Video.py"
    assert str(template.relative_to(root)) in manifest["workflows"]
    metadata = read_workflow_metadata(template)
    assert metadata.is_template and metadata.is_griptape_provided and not metadata.is_internal
    assert metadata.description
    assert metadata.image is not None
    assert metadata.image.endswith("/MiniMaxH3Text2Video.png")
    assert template.with_suffix(".png").is_file()


def test_minimax_h3_template_has_explicit_dependencies() -> None:
    template = Path(__file__).parents[1] / "workflows/templates/MiniMaxH3Text2Video.py"
    dependencies = extract_workflow_dependencies(template)
    assert dependencies.conclusive, dependencies.blockers
    assert {dependency.repo_id for dependency in dependencies.model_dependencies} == {"MiniMaxAI/MiniMax-H3"}
    assert {requirement.path for requirement in dependencies.path_requirements} == {
        "{workflow_dir}/assets/models/minimax-h3/minimax_h3_fl2va-Q4_K_M.gguf",
    }


@pytest.mark.asyncio
async def test_minimax_h3_restores_component_overrides_before_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    requests: list[SetParameterValueRequest] = []
    created_nodes: set[str] = set()

    async def record_request(request: RequestPayload) -> SimpleNamespace:
        if isinstance(request, CreateFlowRequest):
            return SimpleNamespace(flow_name=request.flow_name)
        if isinstance(request, CreateNodeRequest):
            if request.node_name is not None:
                created_nodes.add(request.node_name)
            return SimpleNamespace(node_name=request.node_name)
        if isinstance(request, SetParameterValueRequest):
            requests.append(request)
        return SimpleNamespace()

    context = MagicMock()
    monkeypatch.setattr(GriptapeNodes, "ContextManager", lambda: context)
    monkeypatch.setattr(GriptapeNodes, "ahandle_request", record_request)
    monkeypatch.setattr(
        path_macros, "expand_path_macros", lambda value: str(tmp_path / value.removeprefix("{workflow_dir}/"))
    )
    template = Path(__file__).parents[1] / "workflows/templates/MiniMaxH3Text2Video.py"
    await runpy.run_path(str(template))["build_workflow"]()

    assert "Load Text Encoder (4-bit)" not in created_nodes
    assert all(request.parameter_name != "component_text_encoder" for request in requests)
    for request in requests:
        if request.is_output:
            assert isinstance(request.value, (ComponentArtifact, DiffusionPipelineArtifact))
    values = {
        (request.node_name, request.parameter_name): request.value for request in requests if not request.is_output
    }
    pipelines = [request for request in requests if request.parameter_name == "pipeline"]
    assert {request.node_name for request in pipelines} == {
        "Modular Diffusion Pipeline Builder",
        "Create Noise Latents",
        "Generate Media Latents",
        "Decode Media",
    }
    for request in pipelines:
        pipeline = request.value
        assert isinstance(pipeline, DiffusionPipelineArtifact)
        assert pipeline.config_hash
        overrides = pipeline.build_data.get("_component_overrides", {})
        assert set(overrides) == {"transformer"}
        assert overrides["transformer"].file_path == str(
            tmp_path / "assets/models/minimax-h3/minimax_h3_fl2va-Q4_K_M.gguf"
        )
        assert pipeline.is_prequantized
        assert not pipeline.supports_layerwise_casting
        assert overrides["transformer"] == values[("Modular Diffusion Pipeline Builder", "component_transformer")]
        assert overrides["transformer"] == values[("Load Transformer (GGUF)", "component_output")]
