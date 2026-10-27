"""Loading a node the way a saved workflow does must not touch the graph.

A worker builds the node it runs by replaying a saved workflow's commands inside a graph where the copy has no
connections and no other nodes. While a node is created, or a command is applied with ``initial_setup=True``,
node code must not read or change connections, must not address other nodes, and must not send
``RemoveParameterFromNodeRequest`` (which deletes connections). The loaded node must expose the same parameters
and the same internal state ``process()`` reads as a node built by normal edits.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Any, NamedTuple

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.base_events import RequestPayload
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    DeleteConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    DeleteNodeRequest,
    SerializeNodeToCommandsRequest,
    SerializeNodeToCommandsResultSuccess,
)
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    GetConnectionsForParameterRequest,
    RemoveParameterFromNodeRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.events.resource_events import CreateResourceInstanceRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.event_manager import EventManager
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import LatentFlux2PipelineTypeParameters
from modular_diffusion_nodes_library.standard_parameters.flux2_parameters import Flux2PipelineParameters

LIBRARY_NAME = "Griptape Modular Diffusion Nodes Library"
LIBRARY_MANIFEST = Path(__file__).parents[1] / "griptape-nodes-library.json"

GRAPH_REQUEST_TYPES = (
    ListConnectionsForNodeRequest,
    GetConnectionsForParameterRequest,
    CreateConnectionRequest,
    DeleteConnectionRequest,
    RemoveParameterFromNodeRequest,
)
NODE_NAME_FIELDS = ("node_name", "source_node_name", "target_node_name")
ENGINE_PACKAGE = "griptape_nodes."
DISPATCH_MODULES = {"griptape_nodes.retained_mode.griptape_nodes", "griptape_nodes.retained_mode.engine"}


def _sender_module(frame: FrameType | None) -> str:
    """The module that sent the request being dispatched from `frame`, past the facade and engine dispatch."""
    while frame is not None and frame.f_globals.get("__name__") in DISPATCH_MODULES:
        frame = frame.f_back
    if frame is None:
        return ""
    return frame.f_globals.get("__name__", "")


class SerializedNode(NamedTuple):
    create_command: CreateNodeRequest
    element_commands: list[RequestPayload]
    value_commands: list[SetParameterValueRequest]


class RequestRecorder:
    """Records every request this library's code sends while another request is being handled.

    Requests the engine sends itself are left out: the rule binds node code, and which requests the engine
    makes while loading depends on the engine version.
    """

    def __init__(self) -> None:
        self.depth = 0
        self.nested: list[RequestPayload] = []

    def enter(self, request: RequestPayload, sender_module: str) -> None:
        if self.depth > 0 and not sender_module.startswith(ENGINE_PACKAGE):
            self.nested.append(request)
        self.depth += 1

    def exit(self) -> None:
        self.depth -= 1

    def violations(self, node_name: str) -> list[RequestPayload]:
        found = []
        for request in self.nested:
            names = {getattr(request, field, None) for field in NODE_NAME_FIELDS} - {None}
            if isinstance(request, GRAPH_REQUEST_TYPES) or names - {node_name}:
                found.append(request)
        return found


@pytest.fixture(scope="module")
def engine() -> None:
    GriptapeNodes()
    # The library requires a GPU backend, and these tests only load nodes, so a compute instance that
    # meets the requirement stands in for a GPU machine.
    compute_result = GriptapeNodes.handle_request(
        CreateResourceInstanceRequest(
            resource_type_name="ComputeResourceType", capabilities={"compute": ["cpu", "cuda"]}
        )
    )
    assert compute_result.succeeded(), compute_result
    result = GriptapeNodes.handle_request(RegisterLibraryFromFileRequest(file_path=str(LIBRARY_MANIFEST)))
    assert result.succeeded(), result


@pytest.fixture
def flow(engine: None, tmp_path: Path) -> Generator[None, None, None]:  # noqa: ARG001
    GriptapeNodes.handle_request(ClearAllObjectStateRequest(i_know_what_im_doing=True))
    GriptapeNodes.ContextManager().push_workflow(file_path=str(tmp_path / "workflow.py"))
    result = GriptapeNodes.handle_request(
        CreateFlowRequest(parent_flow_name=None, flow_name="flow", set_as_new_context=True)
    )
    assert result.succeeded(), result
    yield
    GriptapeNodes.handle_request(ClearAllObjectStateRequest(i_know_what_im_doing=True))


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> RequestRecorder:
    recorder = RequestRecorder()
    original_handle = EventManager.handle_request
    original_ahandle = EventManager.ahandle_request

    def handle_request(self: EventManager, request: RequestPayload, **kwargs: Any) -> Any:
        recorder.enter(request, _sender_module(sys._getframe(1)))
        try:
            return original_handle(self, request, **kwargs)
        finally:
            recorder.exit()

    async def ahandle_request(self: EventManager, request: RequestPayload, **kwargs: Any) -> Any:
        recorder.enter(request, _sender_module(sys._getframe(1)))
        try:
            return await original_ahandle(self, request, **kwargs)
        finally:
            recorder.exit()

    monkeypatch.setattr(EventManager, "handle_request", handle_request)
    monkeypatch.setattr(EventManager, "ahandle_request", ahandle_request)
    return recorder


def _request(request: RequestPayload) -> Any:
    result = GriptapeNodes.handle_request(request)
    assert result.succeeded(), result
    return result


def _create(node_type: str, name: str) -> BaseNode:
    _request(CreateNodeRequest(node_type=node_type, specific_library_name=LIBRARY_NAME, node_name=name))
    return GriptapeNodes.NodeManager().get_node_by_name(name)


def _edit(node: BaseNode, values: dict[str, Any]) -> None:
    for name, value in values.items():
        _request(SetParameterValueRequest(parameter_name=name, node_name=node.name, value=value))


def _connect(source: str, source_parameter: str, target: str, target_parameter: str) -> None:
    _request(
        CreateConnectionRequest(
            source_node_name=source,
            source_parameter_name=source_parameter,
            target_node_name=target,
            target_parameter_name=target_parameter,
            initial_setup=True,
        )
    )


def _serialize_and_delete(node: BaseNode) -> SerializedNode:
    request = SerializeNodeToCommandsRequest(node_name=node.name)
    result = _request(request)
    assert isinstance(result, SerializeNodeToCommandsResultSuccess)
    values = request.unique_parameter_uuid_to_values
    value_commands = [
        dataclasses.replace(
            command.set_parameter_value_command,
            value=values[command.unique_value_uuid],
            initial_setup=True,
        )
        for command in result.set_parameter_value_commands
    ]
    _request(DeleteNodeRequest(node_name=node.name))
    commands = result.serialized_node_commands
    create_command = dataclasses.replace(commands.create_node_command, initial_setup=True)
    return SerializedNode(create_command, commands.element_modification_commands, value_commands)


def _create_and_structure(serialized: SerializedNode) -> str:
    name = _request(serialized.create_command).node_name
    with GriptapeNodes.ContextManager().node(name):
        for command in serialized.element_commands:
            _request(command)
    return name


def _set_values(serialized: SerializedNode, name: str) -> BaseNode:
    for command in serialized.value_commands:
        _request(dataclasses.replace(command, node_name=name))
    return GriptapeNodes.NodeManager().get_node_by_name(name)


def _load(serialized: SerializedNode) -> BaseNode:
    """Replay a node's saved commands into a graph with no connections and no other nodes."""
    return _set_values(serialized, _create_and_structure(serialized))


def _parameter_types(node: BaseNode) -> dict[str, str | None]:
    return {parameter.name: parameter.type for parameter in node.parameters}


@contextmanager
def _assert_no_graph_requests(recorder: RequestRecorder, node_name: str) -> Iterator[None]:
    recorder.nested.clear()
    yield
    assert recorder.violations(node_name) == []


def _incoming_connections(node_name: str) -> dict[str, str]:
    result = _request(ListConnectionsForNodeRequest(node_name=node_name))
    assert isinstance(result, ListConnectionsForNodeResultSuccess)
    return {conn.target_parameter_name: conn.source_node_name for conn in result.incoming_connections}


def _builder_state(node: Any) -> dict[str, Any]:
    pipeline_type_parameters = node.params.pipeline_type_parameters
    return {
        "provider_helper": type(pipeline_type_parameters),
        "pipeline_type_helper": type(pipeline_type_parameters.pipeline_type_pipeline_params),
        "pipeline_name": pipeline_type_parameters.pipeline_type_pipeline_params.pipeline_name,
        "slots": node.params.component_override_params.current_slots,
    }


def test_generate_latent_loads_runtime_parameters_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node: Any = _create("DiffusionPipelineGenerateLatentNode", "generate")
    _edit(node, {"pipeline": DiffusionPipelineArtifact(pipeline_name="Flux2KleinPipeline", config_hash="hash")})
    edited_types = _parameter_types(node)
    edited_order = [element.name for element in node.root_ui_element.children]
    edited_runtime = type(node.pipe_params.runtime_parameters)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "generate"):
        loaded: Any = _load(serialized)

    assert _parameter_types(loaded) == edited_types
    assert [element.name for element in loaded.root_ui_element.children] == edited_order
    assert type(loaded.pipe_params.runtime_parameters) is edited_runtime
    assert loaded.pipe_params.runtime_parameters._get_pipe_kwargs()["prompt"] == ""


def test_generate_latent_keeps_saved_connection_to_runtime_parameter(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    conditioning = _create("MediaGenConditioningNode", "conditioning")
    node = _create("DiffusionPipelineGenerateLatentNode", "generate")
    _edit(node, {"pipeline": DiffusionPipelineArtifact(pipeline_name="Flux2KleinPipeline", config_hash="hash")})
    serialized = _serialize_and_delete(node)

    # A saved workflow creates connections after structure and before values.
    name = _create_and_structure(serialized)
    _connect(conditioning.name, "conditioning", name, "reference_images")
    with _assert_no_graph_requests(recorder, name):
        loaded = _set_values(serialized, name)

    assert _incoming_connections(name) == {"reference_images": "conditioning"}
    connections = GriptapeNodes.FlowManager().get_connections()
    (connection_id,) = connections.incoming_index[name]["reference_images"]
    assert connections.connections[connection_id].target_parameter is loaded.get_parameter_by_name("reference_images")


def test_pipeline_builder_rebinds_helpers_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node = _create("LatentDiffusionPipelineBuilderNode", "builder")
    _edit(node, {"provider": "Flux2", "pipeline_type": "Flux2KleinPipeline"})
    edited_types = _parameter_types(node)
    edited_state = _builder_state(node)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "builder"):
        loaded = _load(serialized)

    assert _builder_state(loaded) == edited_state
    loaded_types = _parameter_types(loaded)
    assert {name: loaded_types.get(name) for name in edited_types} == edited_types


def _pipeline_type_parameters(node: BaseNode) -> list[str]:
    return sorted(parameter.name for parameter in node.parameters if parameter.name.startswith("pipeline_type"))


def _pipeline_type_choices(node: BaseNode) -> list[str]:
    parameter = node.get_parameter_by_name("pipeline_type")
    assert parameter is not None
    (options,) = parameter.find_elements_by_type(Options)
    return list(options.choices)


def test_pipeline_builder_reopens_with_one_pipeline_type_of_its_provider(
    flow: None,  # noqa: ARG001
) -> None:
    node = _create("LatentDiffusionPipelineBuilderNode", "builder")
    _edit(node, {"provider": "Flux2", "pipeline_type": "Flux2KleinPipeline"})

    loaded = _load(_serialize_and_delete(node))

    assert _pipeline_type_parameters(loaded) == ["pipeline_type"]
    assert loaded.get_parameter_value("pipeline_type") == "Flux2KleinPipeline"
    assert _pipeline_type_choices(loaded) == list(LatentFlux2PipelineTypeParameters.get_pipeline_type_dict())


def _pipeline_type_as_earlier_releases_saved_it(name: str) -> AddParameterToNodeRequest:
    """The add the shipped templates carry: choices only in ui_options, no Options trait, no alter."""
    return AddParameterToNodeRequest(
        parameter_name=name,
        tooltip="Specific pipeline variant within the selected provider.",
        type="str",
        input_types=["str"],
        output_type="str",
        ui_options={"simple_dropdown": ["Flux2Pipeline", "Flux2KleinPipeline"]},
        mode_allowed_input=False,
        mode_allowed_output=False,
        initial_setup=True,
    )


def test_pipeline_builder_reopens_a_workflow_an_earlier_release_saved(
    flow: None,  # noqa: ARG001
) -> None:
    """Earlier saves describe pipeline_type with no choices, and carry the copies each reopen and save added."""
    node = _create("LatentDiffusionPipelineBuilderNode", "builder")
    _edit(node, {"provider": "Flux2", "pipeline_type": "Flux2KleinPipeline"})
    serialized = _serialize_and_delete(node)
    commands = [
        command
        for command in serialized.element_commands
        if getattr(command, "parameter_name", None) != "pipeline_type"
    ]
    earlier_adds = [
        _pipeline_type_as_earlier_releases_saved_it(name)
        for name in ("pipeline_type", "pipeline_type_1", "pipeline_type_1_1")
    ]
    serialized = serialized._replace(element_commands=[*earlier_adds, *commands])

    loaded = _load(serialized)

    assert _pipeline_type_parameters(loaded) == ["pipeline_type"]
    assert loaded.get_parameter_value("pipeline_type") == "Flux2KleinPipeline"
    assert _pipeline_type_choices(loaded) == list(LatentFlux2PipelineTypeParameters.get_pipeline_type_dict())
    assert _builder_state(loaded)["pipeline_name"] == "Flux2KleinPipeline"

    resaved = _load(_serialize_and_delete(loaded))

    assert resaved.get_parameter_value("pipeline_type") == "Flux2KleinPipeline"
    assert _pipeline_type_choices(resaved) == list(LatentFlux2PipelineTypeParameters.get_pipeline_type_dict())


def _assert_builder_keeps_saved_transformer_override_connection(recorder: RequestRecorder) -> None:
    component = _create("LoadComponent", "component")
    _edit(component, {"component": "Transformer"})
    node = _create("LatentDiffusionPipelineBuilderNode", "builder")
    _edit(node, {"provider": "Flux2", "pipeline_type": "Flux2KleinPipeline"})
    edited_state = _builder_state(node)
    serialized = _serialize_and_delete(node)

    name = _create_and_structure(serialized)
    _connect(component.name, "component_output", name, "component_transformer")
    with _assert_no_graph_requests(recorder, name):
        loaded: Any = _set_values(serialized, name)

    assert _builder_state(loaded) == edited_state
    assert _incoming_connections(name) == {"component_transformer": "component"}
    connections = GriptapeNodes.FlowManager().get_connections()
    (connection_id,) = connections.incoming_index[name]["component_transformer"]
    target = connections.connections[connection_id].target_parameter
    assert target is loaded.get_parameter_by_name("component_transformer")
    group = loaded.get_group_by_name_or_element_id("component_overrides")
    assert target in group.find_elements_by_type(type(target))


def test_pipeline_builder_keeps_saved_component_override_connection(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    _assert_builder_keeps_saved_transformer_override_connection(recorder)


def test_pipeline_builder_keeps_override_connection_across_slot_sets_lacking_it(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Setting the saved provider passes through its first pipeline type, Flux2Pipeline, before the saved
    # Flux2KleinPipeline. Without a transformer slot there, the load drops the port and later re-adds it.
    original_slots = Flux2PipelineParameters.get_component_slots

    def slots_without_transformer(self: Flux2PipelineParameters) -> list[str]:
        return [slot for slot in original_slots(self) if slot != "transformer"]

    monkeypatch.setattr(Flux2PipelineParameters, "get_component_slots", slots_without_transformer)
    _assert_builder_keeps_saved_transformer_override_connection(recorder)


@pytest.mark.parametrize("provider", ["Qwen", "Stable Diffusion 3", "Z-Image"])
def test_controlnet_loads_provider_parameters_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
    provider: str,
) -> None:
    node: Any = _create("ControlNetNode", "controlnet")
    _edit(node, {"provider": provider})
    edited_types = _parameter_types(node)
    edited_helper = type(node.controlnet_parameter_type)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "controlnet"):
        loaded: Any = _load(serialized)

    assert _parameter_types(loaded) == edited_types
    assert type(loaded.controlnet_parameter_type) is edited_helper


def test_vae_decoder_loads_video_output_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node: Any = _create("VaeDecodeNode", "decode")
    _edit(node, {"pipeline": DiffusionPipelineArtifact(pipeline_name="WanPipeline", config_hash="hash")})
    assert node._current_output_type == "video"
    edited_types = _parameter_types(node)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "decode"):
        loaded: Any = _load(serialized)

    assert _parameter_types(loaded) == edited_types
    assert loaded._current_output_type == "video"


def test_vae_encoder_loads_video_input_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node: Any = _create("VaeEncodeNode", "encode")
    _edit(node, {"pipeline": DiffusionPipelineArtifact(pipeline_name="WanPipeline", config_hash="hash")})
    assert node._current_input_type == "video"
    edited_types = _parameter_types(node)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "encode"):
        loaded: Any = _load(serialized)

    assert _parameter_types(loaded) == edited_types
    assert loaded._current_input_type == "video"


def test_latent_upsampler_loads_provider_without_graph_requests(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node: Any = _create("LatentUpsamplerNode", "upsampler")
    _edit(node, {"provider": "LTX2"})
    edited_types = _parameter_types(node)
    edited_helper = type(node.upsampler_params)

    serialized = _serialize_and_delete(node)
    with _assert_no_graph_requests(recorder, "upsampler"):
        loaded: Any = _load(serialized)

    assert _parameter_types(loaded) == edited_types
    assert type(loaded.upsampler_params) is edited_helper


def test_latent_upsampler_edit_after_load_removes_by_request(
    flow: None,  # noqa: ARG001
    recorder: RequestRecorder,
) -> None:
    node: Any = _create("LatentUpsamplerNode", "upsampler")
    _edit(node, {"provider": "LTX2"})
    loaded = _load(_serialize_and_delete(node))

    recorder.nested.clear()
    _edit(loaded, {"provider": "LTX"})

    removed = [
        request.parameter_name for request in recorder.nested if isinstance(request, RemoveParameterFromNodeRequest)
    ]
    assert "upsampler_model" in removed
