"""Create Noise Latents honours `randomize_seed` the way the engine's `SeedParameter` defines it."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import NodeResolutionState
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.object_events import ClearAllObjectStateRequest
from griptape_nodes.retained_mode.events.resource_events import CreateResourceInstanceRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from modular_diffusion_nodes_library.nodes.empty_latent_node import EmptyLatentNode
from modular_diffusion_nodes_library.nodes.noise_latent_node import NoiseLatentNode

if TYPE_CHECKING:
    from collections.abc import Generator

    from griptape_nodes.exe_types.node_types import BaseNode
    from griptape_nodes.retained_mode.events.base_events import RequestPayload

LIBRARY_NAME = "Griptape Modular Diffusion Nodes Library"
LIBRARY_MANIFEST = Path(__file__).parents[1] / "griptape-nodes-library.json"


@pytest.fixture(scope="module")
def engine() -> None:
    GriptapeNodes()
    # The library requires a GPU backend; a compute instance that meets the requirement stands in for one.
    _request(
        CreateResourceInstanceRequest(
            resource_type_name="ComputeResourceType", capabilities={"compute": ["cpu", "cuda"]}
        )
    )
    _request(RegisterLibraryFromFileRequest(file_path=str(LIBRARY_MANIFEST)))


@pytest.fixture
def flow(engine: None, tmp_path: Path) -> Generator[None, None, None]:  # noqa: ARG001
    GriptapeNodes.handle_request(ClearAllObjectStateRequest(i_know_what_im_doing=True))
    GriptapeNodes.ContextManager().push_workflow(file_path=str(tmp_path / "workflow.py"))
    _request(CreateFlowRequest(parent_flow_name=None, flow_name="flow", set_as_new_context=True))
    yield
    GriptapeNodes.handle_request(ClearAllObjectStateRequest(i_know_what_im_doing=True))


def _request(request: RequestPayload) -> None:
    result = GriptapeNodes.handle_request(request)
    assert result.succeeded(), result


def _create(node_type: str, name: str) -> BaseNode:
    _request(CreateNodeRequest(node_type=node_type, specific_library_name=LIBRARY_NAME, node_name=name))
    return GriptapeNodes.NodeManager().get_node_by_name(name)


def _noise_feeding_generate() -> tuple[BaseNode, BaseNode]:
    """A Create Noise Latents node wired only by data into Generate Media Latents, both left RESOLVED by a run."""
    noise = _create("NoiseLatentNode", "noise")
    generate = _create("DiffusionPipelineGenerateLatentNode", "generate")
    _request(
        CreateConnectionRequest(
            source_node_name="noise",
            source_parameter_name="output_latent",
            target_node_name="generate",
            target_parameter_name="input_latent",
        )
    )
    noise.state = NodeResolutionState.RESOLVED
    generate.state = NodeResolutionState.RESOLVED
    return noise, generate


def test_preprocess_picks_a_new_seed_when_randomize_seed_is_on() -> None:
    node = NoiseLatentNode(name="noise")
    node.set_parameter_value("randomize_seed", True)

    with patch("griptape_nodes.exe_types.param_components.seed_parameter.random.randint", return_value=1234):
        node.preprocess()

    assert node.get_parameter_value("seed") == 1234


def test_preprocess_keeps_the_seed_when_randomize_seed_is_off() -> None:
    node = NoiseLatentNode(name="noise")
    node.set_parameter_value("seed", 7)

    node.preprocess()

    assert node.get_parameter_value("seed") == 7


def test_enabling_randomize_seed_locks_the_seed_field() -> None:
    node = NoiseLatentNode(name="noise")
    seed_parameter = node.get_parameter_by_name("seed")
    assert seed_parameter is not None

    node.set_parameter_value("randomize_seed", True)

    assert seed_parameter.allowed_modes == {ParameterMode.OUTPUT}


def test_workflow_run_reruns_the_node_when_randomize_seed_is_on(flow: None) -> None:  # noqa: ARG001
    noise, generate = _noise_feeding_generate()
    noise.set_parameter_value("randomize_seed", True)

    noise.validate_before_workflow_run()

    assert noise.state == NodeResolutionState.UNRESOLVED
    assert generate.state == NodeResolutionState.RESOLVED


def test_workflow_run_keeps_the_node_resolved_when_randomize_seed_is_off(flow: None) -> None:  # noqa: ARG001
    noise, _ = _noise_feeding_generate()

    noise.validate_before_workflow_run()

    assert noise.state == NodeResolutionState.RESOLVED


def test_workflow_run_keeps_a_locked_node_resolved(flow: None) -> None:  # noqa: ARG001
    noise, _ = _noise_feeding_generate()
    noise.set_parameter_value("randomize_seed", True)
    noise.lock = True

    noise.validate_before_workflow_run()

    assert noise.state == NodeResolutionState.RESOLVED


def test_empty_latent_preprocess_has_no_seed_to_randomize() -> None:
    node = EmptyLatentNode(name="empty")

    node.preprocess()

    assert node.get_parameter_by_name("seed") is None
