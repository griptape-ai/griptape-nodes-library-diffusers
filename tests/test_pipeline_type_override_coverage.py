"""Every registered pipeline type must be buildable from component overrides alone.

A pipeline whose required `__init__` components are neither exposed as override ports
(`ALLOWED_COMPONENT_SLOTS`) nor auto-supplied by its params class
(`get_auto_supplied_components()`) cannot be built without touching a repo, which surfaces as a
confusing builder failure rather than an error at registration.

`verify_overridable_covers_required` reads the real pipeline class, so checking it imports
diffusers. That is why this lives in a test rather than as an import-time loop in
`pipelinetype_parameters`: the orchestrator imports that module for every node it builds, and
diffusers belongs in the execution environment only.
"""

from __future__ import annotations

import pytest

from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import (
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
)


def _pipeline_type_classes() -> list[tuple[str, type]]:
    cases = []
    for provider, params_cls in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.items():
        for pipeline_type, pipeline_type_cls in params_cls.get_pipeline_type_dict().items():
            cases.append((f"{provider}-{pipeline_type}", pipeline_type_cls))
    return cases


@pytest.mark.parametrize(
    ("case_id", "pipeline_type_cls"), _pipeline_type_classes(), ids=lambda v: v if isinstance(v, str) else ""
)
def test_overridable_covers_required(case_id: str, pipeline_type_cls: type) -> None:
    if not pipeline_type_cls.supports_build_from_overrides_only():
        pytest.skip(f"{case_id} opts out of building from overrides alone")
    pipeline_type_cls.verify_overridable_covers_required()
