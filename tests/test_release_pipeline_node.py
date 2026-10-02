"""Releasing one pipeline leaves every other cached pipeline loaded.

These are the first tests that exercise the worker's local object store through this library, so they
also cover the store's side of the pipeline cache: a key derived from the config hash, and a release
hook that runs when the entry goes.
"""

from __future__ import annotations

from typing import Any

import pytest

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.nodes.release_pipeline_node import ReleasePipelineNode


class _Pipe:
    """Stands in for a loaded diffusion pipeline, which cannot be built in a unit test."""

    def __init__(self, label: str) -> None:
        self.label = label


def _artifact(config_hash: str, pipeline_name: str = "FluxPipeline") -> DiffusionPipelineArtifact:
    return DiffusionPipelineArtifact(pipeline_name=pipeline_name, config_hash=config_hash)


def _node(name: str = "Release Pipeline 1") -> ReleasePipelineNode:
    return ReleasePipelineNode(name=name, metadata={"library": "Modular Diffusion"})


def _run(node: ReleasePipelineNode) -> None:
    """Drive `process()`, which yields the work rather than doing it inline."""
    for step in node.process() or []:
        step()


def test_releases_the_pipeline_passed_in_and_runs_its_release_hook() -> None:
    node = _node()
    released: list[str] = []
    artifact = _artifact("flux-cfg")
    node.local_objects.put(_Pipe("flux"), key="flux-cfg", on_drop=lambda pipe: released.append(pipe.label))

    node.set_parameter_value("pipeline", artifact)
    _run(node)

    assert node.parameter_output_values["was_successful"] is True
    assert "Released 'FluxPipeline'" in node.parameter_output_values["result_details"]
    assert released == ["flux"]
    assert node.local_objects.get(node.local_objects.key_for("flux-cfg")) is None


def test_leaves_other_cached_pipelines_alone() -> None:
    node = _node()
    released: list[str] = []
    node.local_objects.put(_Pipe("flux"), key="flux-cfg", on_drop=lambda pipe: released.append(pipe.label))
    node.local_objects.put(_Pipe("sdxl"), key="sdxl-cfg", on_drop=lambda pipe: released.append(pipe.label))

    node.set_parameter_value("pipeline", _artifact("flux-cfg"))
    _run(node)

    assert released == ["flux"]
    survivor = node.local_objects.get(node.local_objects.key_for("sdxl-cfg"))
    assert survivor is not None
    assert survivor.label == "sdxl"


def test_a_pipeline_that_is_not_held_is_a_success() -> None:
    """The state the node was asked for, so re-running a graph that already released must not fail."""
    node = _node()

    node.set_parameter_value("pipeline", _artifact("never-built"))
    _run(node)

    assert node.parameter_output_values["was_successful"] is True
    assert "Nothing to release" in node.parameter_output_values["result_details"]


def test_releasing_twice_reports_the_second_as_nothing_to_release() -> None:
    node = _node()
    node.local_objects.put(_Pipe("flux"), key="flux-cfg")
    artifact = _artifact("flux-cfg")

    node.set_parameter_value("pipeline", artifact)
    _run(node)
    assert "Released 'FluxPipeline'" in node.parameter_output_values["result_details"]

    _run(node)
    assert node.parameter_output_values["was_successful"] is True
    assert "Nothing to release" in node.parameter_output_values["result_details"]


@pytest.mark.parametrize(
    ("pipeline_value", "expected"),
    [
        (None, "No pipeline connected"),
        (_artifact(""), "never built"),
    ],
)
def test_no_built_pipeline_to_name_is_a_failure(pipeline_value: Any, expected: str) -> None:
    """Distinct from a cache miss: there is no pipeline to name, which is a wiring mistake.

    `_handle_failure_exception` re-raises while nothing is wired to the `failure` output, so the status
    results are recorded and the exception still reaches the flow.
    """
    node = _node()
    node.set_parameter_value("pipeline", pipeline_value)

    with pytest.raises(ValueError, match=expected):
        _run(node)

    assert node.parameter_output_values["was_successful"] is False
    assert expected in node.parameter_output_values["result_details"]


def test_the_cache_key_is_namespaced_so_a_bare_config_hash_finds_nothing() -> None:
    """`put` returns a namespaced key, so the raw hash is not a lookup handle."""
    node = _node()
    node.local_objects.put(_Pipe("flux"), key="flux-cfg")

    assert node.local_objects.get("flux-cfg") is None
    assert node.local_objects.get(node.local_objects.key_for("flux-cfg")) is not None


def test_two_nodes_in_this_library_share_one_cached_pipeline() -> None:
    """The store is keyed per library, not per node, so a second builder reuses the first's pipeline.

    This is what keeps two builder nodes on the same config from loading the model twice.
    """
    first = _node("Release Pipeline 1")
    built = _Pipe("flux")
    first.local_objects.put(built, key="flux-cfg")

    second = _node("Release Pipeline 2")
    assert second.local_objects.get(second.local_objects.key_for("flux-cfg")) is built
