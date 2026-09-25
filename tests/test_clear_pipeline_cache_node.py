"""Clearing the pipeline cache releases this library's pipelines and nothing else.

The object store cannot list what a library holds, and dropping the library's whole group is too broad:
the engine parks every `serializable=False` output under that same group, so the group drop took the
latents a downstream node was about to read. `Generate latents -> Clear Pipeline Cache -> VAE Decode` is
the natural place to free VRAM before decoding, and it failed at the decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from griptape_nodes.exe_types.local_objects import KeyVerdict, cache_outputs_for_egress

from modular_diffusion_nodes_library.artifact_utils import pipeline_artifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    DiffusionPipelineArtifact,
    release_held_pipelines,
)
from modular_diffusion_nodes_library.nodes.clear_pipeline_cache_node import ClearPipelineCacheNode
from modular_diffusion_nodes_library.nodes.vae_encoder import VaeEncodeNode

if TYPE_CHECKING:
    from collections.abc import Generator

LIBRARY = {"library": "Modular Diffusion"}


class _Pipe:
    """Stands in for a loaded diffusion pipeline, which cannot be built in a unit test."""

    def __init__(self, label: str) -> None:
        self.label = label


class _Latent:
    """Stands in for a latent tensor: not plain data, so the engine parks it on egress."""


class _StubArtifact(DiffusionPipelineArtifact):
    """Caches through the real `get_or_build_pipeline` without diffusers to build with."""

    def _build_pipeline(self, log_params: Any | None = None) -> _Pipe:
        return _Pipe(str(self.config_hash))


@pytest.fixture(autouse=True)
def _forget_held_pipelines() -> Generator[None, None, None]:
    """The record of held keys is module state, so what one test leaves must not reach the next."""
    yield
    release_held_pipelines(_clear_node())


def _clear_node() -> ClearPipelineCacheNode:
    return ClearPipelineCacheNode(name="Clear Pipeline Cache", metadata=LIBRARY)


def _run(node: ClearPipelineCacheNode) -> None:
    """Drive `process()`, which yields the work rather than doing it inline."""
    for step in node.process() or []:
        step()


def _build(node: ClearPipelineCacheNode, config_hash: str) -> None:
    _StubArtifact(pipeline_name="FluxPipeline", config_hash=config_hash).get_or_build_pipeline(node)


def test_clearing_releases_every_pipeline_this_library_built(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[str] = []
    monkeypatch.setattr(pipeline_artifact, "clear_diffusion_pipeline", lambda pipe: released.append(pipe.label))
    node = _clear_node()
    _build(node, "flux-cfg")
    _build(node, "sdxl-cfg")

    _run(node)

    assert node.parameter_output_values["was_successful"] is True
    assert "Cleared 2 pipeline(s)" in node.parameter_output_values["result_details"]
    assert sorted(released) == ["flux-cfg", "sdxl-cfg"]
    assert node.local_objects.get(node.local_objects.key_for("flux-cfg")) is None
    assert node.local_objects.get(node.local_objects.key_for("sdxl-cfg")) is None


def test_clearing_leaves_a_parked_latent_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: a latent held for a downstream node shares the pipelines' group."""
    monkeypatch.setattr(pipeline_artifact, "clear_diffusion_pipeline", lambda pipe: None)
    encoder = VaeEncodeNode(name="VAE Encode", metadata=LIBRARY)
    encoder.parameter_output_values["latent_tensor"] = _Latent()
    reference = cache_outputs_for_egress(encoder.parameter_output_values, node=encoder)["latent_tensor"]
    node = _clear_node()
    _build(node, "flux-cfg")

    _run(node)

    assert encoder.local_objects.look_up(reference).verdict is KeyVerdict.HELD, (
        "The decoder's latent went with the pipelines, so the decode fails with the released-object error."
    )
    assert "Cleared 1 pipeline(s)" in node.parameter_output_values["result_details"]


def test_a_pipeline_released_elsewhere_is_not_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Release Pipeline leaves its key in the record, and a count of what went has to ignore it."""
    monkeypatch.setattr(pipeline_artifact, "clear_diffusion_pipeline", lambda pipe: None)
    node = _clear_node()
    _build(node, "flux-cfg")
    node.local_objects.drop(node.local_objects.key_for("flux-cfg"))

    _run(node)

    assert "Cleared 0 pipeline(s)" in node.parameter_output_values["result_details"]
