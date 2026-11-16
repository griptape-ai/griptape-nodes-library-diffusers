"""Clearing the pipeline cache releases this library's pipelines and nothing else.

The object store cannot list what a library holds, and dropping the library's whole group is too broad:
the engine parks every `serializable=False` output under that same group, so the group drop took the
latents a downstream node was about to read. `Generate latents -> Clear Pipeline Cache -> VAE Decode` is
the natural place to free VRAM before decoding, and it failed at the decode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

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

LIBRARY = {"library": "Diffusion"}


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
    monkeypatch.setattr(pipeline_artifact, "clear_pipeline", lambda pipe: released.append(pipe.label))
    node = _clear_node()
    _build(node, "flux-cfg")

    _run(node)

    assert node.parameter_output_values["was_successful"] is True
    assert "Cleared 1 pipeline(s)" in node.parameter_output_values["result_details"]
    assert released == ["flux-cfg"]
    assert node.local_objects.get(node.local_objects.key_for("flux-cfg")) is None


def test_building_a_pipeline_releases_the_previous_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """One pipeline resident at a time, so switching models does not need two models' worth of VRAM."""
    released: list[str] = []
    monkeypatch.setattr(pipeline_artifact, "clear_pipeline", lambda pipe: released.append(pipe.label))
    node = _clear_node()
    _build(node, "flux-cfg")
    _build(node, "sdxl-cfg")

    assert released == ["flux-cfg"]
    assert node.local_objects.get(node.local_objects.key_for("flux-cfg")) is None
    assert node.local_objects.get(node.local_objects.key_for("sdxl-cfg")) is not None


def test_clearing_leaves_a_parked_latent_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression: a latent held for a downstream node shares the pipelines' group."""
    monkeypatch.setattr(pipeline_artifact, "clear_pipeline", lambda pipe: None)
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
    """An engine-side release leaves its key in the record, and a count of what went has to ignore it."""
    monkeypatch.setattr(pipeline_artifact, "clear_pipeline", lambda pipe: None)
    node = _clear_node()
    _build(node, "flux-cfg")
    node.local_objects.drop(node.local_objects.key_for("flux-cfg"))

    _run(node)

    assert "Cleared 0 pipeline(s)" in node.parameter_output_values["result_details"]


def test_modular_pipeline_cleanup_disables_offload_before_unloading(monkeypatch: pytest.MonkeyPatch) -> None:
    modular_pipeline = pytest.importorskip("diffusers.modular_pipelines.modular_pipeline")

    from modular_diffusion_nodes_library.utils import pipeline_utils

    class Manager:
        def __init__(self, components: dict[str, object]) -> None:
            self.components = dict(components)
            self.offload_enabled = True

        def disable_auto_cpu_offload(self) -> None:
            self.offload_enabled = False

    class ModularPipe(modular_pipeline.ModularPipeline):
        def __init__(self) -> None:
            self.transformer = object()
            self._components_manager = Manager({"transformer": self.transformer})

        @property
        def components(self) -> dict[str, object]:
            return {"transformer": self.transformer}

        def unload_components(self, names: list[str]) -> None:
            assert self._components_manager.offload_enabled is False
            for name in names:
                setattr(self, name, None)
                self._components_manager.components.pop(name)

    pipe = ModularPipe()
    monkeypatch.setattr(pipeline_utils, "cleanup_memory_caches", lambda: None)

    pipeline_utils.clear_pipeline(cast(Any, pipe))

    assert pipe.transformer is None
    assert pipe._components_manager.components == {}


def test_modular_pipeline_cache_uses_dispatcher(monkeypatch: pytest.MonkeyPatch) -> None:
    modular_pipeline = pytest.importorskip("diffusers.modular_pipelines.modular_pipeline")
    released: list[object] = []

    class ModularPipe(_Pipe, modular_pipeline.ModularPipeline):
        def unload_components(self, names: list[str]) -> None:
            del names

    class ModularArtifact(_StubArtifact):
        def _build_pipeline(self, log_params: Any | None = None) -> ModularPipe:
            return ModularPipe("h3")

    monkeypatch.setattr(pipeline_artifact, "clear_pipeline", released.append)
    node = _clear_node()
    pipe = ModularArtifact(pipeline_name="MiniMaxH3ModularPipeline", config_hash="h3-cfg").get_or_build_pipeline(node)

    _run(node)

    assert released == [pipe]


def test_pipeline_cleanup_rejects_unknown_types() -> None:
    pytest.importorskip("diffusers.modular_pipelines.modular_pipeline")
    from modular_diffusion_nodes_library.utils.pipeline_utils import clear_pipeline

    with pytest.raises(TypeError, match="only DiffusionPipeline and ModularPipeline instances are supported"):
        clear_pipeline(cast(Any, object()))
