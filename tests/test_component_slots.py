"""Every pipeline type's declared override slots must match the pinned diffusers.

`get_component_slots()` returns a declared list rather than reading the pipeline's `__init__`
signature, because reading it imports the pipeline class and the orchestrator builds these parameters
for every Pipeline Builder it shows without the execution environment installed. That trade only holds
while the declarations are true, so this recomputes them the way the library used to and fails on
drift -- which is what a diffusers bump that renames or adds a component looks like.
"""

from __future__ import annotations

import pytest

from modular_diffusion_nodes_library.component_loading.component_slots import ALLOWED_COMPONENT_SLOTS
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import (
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
)

pytest.importorskip(
    "diffusers",
    reason="Reads real diffusers classes; run `make test/exec` for the execution environment.",
)


def _pipeline_type_classes() -> list[tuple[str, type]]:
    seen: dict[type, str] = {}
    for provider, params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.items():
        for pipeline_type, pipeline_type_cls in params_cls.get_pipeline_type_dict().items():
            seen.setdefault(pipeline_type_cls, f"{provider}-{pipeline_type}")
    return [(case_id, cls) for cls, case_id in seen.items()]


def _slots_from_diffusers(pipeline_type_cls: type) -> list[str]:
    """What `get_component_slots()` computed before the slots were declared."""
    pipeline_cls = pipeline_type_cls.pipeline_cls()
    modular_cls = pipeline_type_cls._modular_pipeline_cls()  # noqa: SLF001
    if issubclass(pipeline_cls, modular_cls):
        return []
    all_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)  # noqa: SLF001
    return [slot for slot in ALLOWED_COMPONENT_SLOTS if slot in set(all_slots)]


CASES = _pipeline_type_classes()


@pytest.mark.parametrize(("case_id", "pipeline_type_cls"), CASES, ids=[case_id for case_id, _ in CASES])
def test_declared_slots_match_the_pipeline_signature(case_id: str, pipeline_type_cls: type) -> None:
    declared = list(pipeline_type_cls._component_slots)  # noqa: SLF001
    assert declared == _slots_from_diffusers(pipeline_type_cls), (
        f"{case_id}: declared `_component_slots` on {pipeline_type_cls.__name__} no longer matches "
        f"the pipeline's __init__ signature in the pinned diffusers. Update the declaration."
    )


def test_every_pipeline_type_declares_its_slots() -> None:
    """A missing declaration must not read as 'this pipeline has no overridable components'."""
    undeclared = [
        pipeline_type_cls.__name__
        for _, pipeline_type_cls in CASES
        if not any("_component_slots" in klass.__dict__ for klass in pipeline_type_cls.__mro__)
    ]
    assert not undeclared, f"These pipeline types never declare `_component_slots`: {undeclared}."


def test_declared_slots_are_all_allowed_and_ordered() -> None:
    """`ALLOWED_COMPONENT_SLOTS` fixes both the permitted set and the builder's display order."""
    order = {slot: i for i, slot in enumerate(ALLOWED_COMPONENT_SLOTS)}
    for case_id, pipeline_type_cls in CASES:
        declared = list(pipeline_type_cls._component_slots)  # noqa: SLF001
        unknown = [slot for slot in declared if slot not in order]
        assert not unknown, f"{case_id}: slots absent from ALLOWED_COMPONENT_SLOTS: {unknown}."
        positions = [order[slot] for slot in declared]
        assert positions == sorted(positions), (
            f"{case_id}: declared slots are out of ALLOWED_COMPONENT_SLOTS order, which is the order "
            f"the builder displays them in: {declared}."
        )
