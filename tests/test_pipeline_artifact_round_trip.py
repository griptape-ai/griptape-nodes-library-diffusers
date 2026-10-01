"""A pipeline artifact must survive the trip between the orchestrator and a worker.

The artifact is the *description* of a pipeline, not the pipeline: the built pipeline lives in the
worker's object store, but both processes need the description -- the worker to build from it, the
orchestrator to validate and to decide which parameters to show. So it travels as data.

The engine encodes event payloads with `json.dumps(default=str)`, which replaces a type it cannot
unstructure with that type's `str()`. This artifact used to define `__str__` as its config hash, so an
encoding failure produced a plausible-looking string and surfaced two nodes later as "Invalid
'pipeline' value type 'str'". These tests pin the encoding instead of trusting it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.event_converter import safe_unstructure

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec


def _over_the_wire(value: Any) -> Any:
    """`value` as a consumer in another process receives it, encoded exactly as the engine does."""
    payload = {"parameter_output_values": {"pipeline": value}}
    encoded = json.dumps(safe_unstructure(payload), default=str)
    return json.loads(encoded)["parameter_output_values"]["pipeline"]


def _round_trip(value: Any) -> DiffusionPipelineArtifact | None:
    return normalize_diffusion_pipeline_value(_over_the_wire(value))


def _base_artifact(**overrides: Any) -> DiffusionPipelineArtifact:
    fields: dict[str, Any] = {
        "pipeline_name": "ZImagePipeline",
        "config_hash": "z-config-hash",
        "build_data": {"base_repo_id": "org/z-image", "base_revision": "abc123"},
        "loras": {"sharpen": 0.5},
        "optimization_kwargs": {"enable_cpu_offload": True},
        "is_prequantized": True,
        "requires_device_map": True,
    }
    fields.update(overrides)
    return DiffusionPipelineArtifact(**fields)


def test_the_artifact_does_not_arrive_as_a_string() -> None:
    """The regression this file exists for: a str reaching the consumer is the original bug."""
    assert not isinstance(_over_the_wire(_base_artifact()), str)


def test_every_field_survives() -> None:
    original = _base_artifact()
    restored = _round_trip(original)

    assert isinstance(restored, DiffusionPipelineArtifact)
    assert restored.pipeline_name == original.pipeline_name
    assert restored.config_hash == original.config_hash
    assert restored.build_data == original.build_data
    assert restored.loras == original.loras
    assert restored.optimization_kwargs == original.optimization_kwargs
    assert restored.is_prequantized == original.is_prequantized
    assert restored.requires_device_map == original.requires_device_map
    assert restored.supports_layerwise_casting == original.supports_layerwise_casting


def test_a_live_pipeline_class_is_not_sent() -> None:
    """`_pipeline_cls` is a class object, so it cannot be encoded -- and it is derivable anyway."""
    artifact = _base_artifact(build_data={"base_repo_id": "org/z", "_pipeline_cls": DiffusionPipelineArtifact})

    restored = _round_trip(artifact)

    assert restored is not None
    assert "_pipeline_cls" not in restored.build_data
    assert restored.build_data["base_repo_id"] == "org/z"


def test_runtime_adapter_steps_are_rebuilt_as_their_own_class() -> None:
    """A LoRA node hands downstream an artifact carrying a step; losing it silently drops the LoRA."""
    step = LoraPipelineRuntimeAdapterStep({"sharpen": LoraSpec(path="/loras/sharpen.safetensors", weight=0.8)})
    artifact = _base_artifact().with_additional_runtime_adapter_steps([step])

    restored = _round_trip(artifact)

    assert restored is not None
    rebuilt = restored.runtime_adapter_steps()
    assert [type(s) for s in rebuilt] == [LoraPipelineRuntimeAdapterStep]
    assert rebuilt[0].metadata == step.metadata


def test_a_derived_artifact_keeps_its_own_class_and_its_base() -> None:
    base = _base_artifact()
    derived = ControlNetDiffusionPipelineArtifact(
        base_artifact=base, controlnet_models=["diffusers/controlnet-canny"], config_hash="cn-hash"
    )

    restored = _round_trip(derived)

    assert isinstance(restored, ControlNetDiffusionPipelineArtifact)
    assert restored.config_hash == "cn-hash"
    # The base travels whole rather than by hash: rebuilding the derived pipeline needs its description,
    # and the receiving process may not hold the base's cache entry.
    assert restored._base_artifact.config_hash == base.config_hash  # noqa: SLF001
    assert restored._base_artifact.build_data == base.build_data  # noqa: SLF001


def test_an_unknown_tag_is_refused_rather_than_guessed() -> None:
    from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import structure_pipeline_artifact

    with pytest.raises(ValueError, match="no artifact class declares it"):
        structure_pipeline_artifact({"__pipeline_artifact__": "from-a-newer-library"})


def test_a_genuinely_wrong_type_is_still_rejected() -> None:
    """Accepting the wire form must not turn the validator into something that accepts anything."""
    with pytest.raises(ValueError, match="Invalid 'pipeline' value type 'int'"):
        normalize_diffusion_pipeline_value(7, node_name="Some Node", raise_on_invalid=True)
