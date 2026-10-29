import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import griptape_nodes.retained_mode.managers.config_manager as config_manager_module
import griptape_nodes.retained_mode.managers.secrets_manager as secrets_manager_module
import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.engine import Engine, current_engine, reset_root_engine
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.library_events import (
    RegisterLibraryFromFileRequest,
    RegisterLibraryFromFileResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.utils.version_utils import engine_version

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.load_pipeline_node import LoadPipelineNode

FIXTURE_LIBRARY_DIR = Path(__file__).parent / "fixtures" / "load_pipeline_propagation"
FIXTURE_LIBRARY_JSON = FIXTURE_LIBRARY_DIR / "griptape_nodes_library.json"
FIXTURE_NODE_FILE = FIXTURE_LIBRARY_DIR / "propagation_nodes.py"
LIBRARY_NAME = "Load Pipeline Propagation Test Library"


@pytest.fixture(autouse=True)
def isolated_engine_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    reset_root_engine()
    LibraryRegistry._clear()
    WorkflowRegistry.clear_user_workflows()

    for key in list(os.environ):
        if key.startswith(("GT_CLOUD_", "GTN_CONFIG_")):
            monkeypatch.delenv(key, raising=False)

    workspace_path = tmp_path / "workspace"
    config_path = tmp_path / "griptape_nodes_config.json"
    config_path.write_text(json.dumps({"workspace_directory": str(workspace_path)}, indent=2), encoding="utf-8")
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_manager_module, "USER_CONFIG_PATH", config_path)
    monkeypatch.setattr(secrets_manager_module, "ENV_VAR_PATH", env_path)

    try:
        yield
    finally:
        reset_root_engine()
        LibraryRegistry._clear()
        WorkflowRegistry.clear_user_workflows()


@pytest.fixture
def engine() -> Engine:
    return current_engine()


@pytest.fixture
def fixture_library(tmp_path: Path) -> Path:
    fixture_dir = tmp_path / "library"
    fixture_dir.mkdir()
    schema = json.loads(FIXTURE_LIBRARY_JSON.read_text(encoding="utf-8"))
    schema["metadata"]["engine_version"] = engine_version
    library_path = fixture_dir / "griptape_nodes_library.json"
    library_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (fixture_dir / FIXTURE_NODE_FILE.name).write_text(FIXTURE_NODE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    return library_path


def _pipeline_with_sequential_offload() -> DiffusionPipelineArtifact:
    return DiffusionPipelineArtifact(
        pipeline_name="TestPipeline",
        config_hash="source-hash",
        optimization_kwargs={
            "memory_optimization_strategy": "Manual",
            "attention_slicing": False,
            "vae_slicing": False,
            "vae_tiling": False,
            "transformer_layerwise_casting": False,
            "cpu_offload_strategy": "Sequential",
        },
    )


def test_optimization_edit_and_reset_each_propagate_once_downstream(
    engine: Engine,
    fixture_library: Path,
) -> None:
    register_result = engine.handle_request(RegisterLibraryFromFileRequest(file_path=str(fixture_library)))
    assert isinstance(register_result, RegisterLibraryFromFileResultSuccess), register_result

    engine.context_manager.push_workflow(workflow_name="load_pipeline_propagation")
    flow_result = engine.handle_request(
        CreateFlowRequest(
            parent_flow_name=None,
            flow_name="Load Pipeline Propagation",
            set_as_new_context=False,
        )
    )
    assert isinstance(flow_result, CreateFlowResultSuccess), flow_result

    load_result = engine.handle_request(
        CreateNodeRequest(
            node_type="PropagationLoadPipelineNode",
            specific_library_name=LIBRARY_NAME,
            node_name="Load Pipeline",
            override_parent_flow_name=flow_result.flow_name,
        )
    )
    assert isinstance(load_result, CreateNodeResultSuccess), load_result
    sink_result = engine.handle_request(
        CreateNodeRequest(
            node_type="PipelineSinkNode",
            specific_library_name=LIBRARY_NAME,
            node_name="Pipeline Sink",
            override_parent_flow_name=flow_result.flow_name,
        )
    )
    assert isinstance(sink_result, CreateNodeResultSuccess), sink_result

    connection_result = engine.handle_request(
        CreateConnectionRequest(
            source_node_name=load_result.node_name,
            source_parameter_name="pipeline",
            target_node_name=sink_result.node_name,
            target_parameter_name="pipeline",
        )
    )
    assert isinstance(connection_result, CreateConnectionResultSuccess), connection_result

    load_node = engine.object_manager.attempt_get_object_by_name(load_result.node_name)
    sink_node: Any = engine.object_manager.attempt_get_object_by_name(sink_result.node_name)
    assert isinstance(load_node, LoadPipelineNode)
    assert sink_node is not None

    source_pipeline = _pipeline_with_sequential_offload()
    load_node._source_pipeline = source_pipeline
    load_node._publish_pipeline(source_pipeline)
    sink_node.received.clear()

    load_node.set_parameter_value("cpu_offload_strategy", "None")

    assert len(sink_node.received) == 1
    assert sink_node.received[0].optimization_kwargs["cpu_offload_strategy"] == "None"

    sink_node.received.clear()
    result = load_node._on_use_recipe_optimization_settings(None, None)  # type: ignore[arg-type]

    assert result.success
    assert len(sink_node.received) == 1
    assert sink_node.received[0].optimization_kwargs["cpu_offload_strategy"] == "Sequential"
