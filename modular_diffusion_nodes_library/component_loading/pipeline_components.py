"""Discover component classes from classic signatures or modular block metadata."""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from functools import cache


def pipeline_component_types(pipeline_cls: type, workflows: Sequence[str] = ()) -> dict[str, type]:
    from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

    if issubclass(pipeline_cls, ModularPipeline):
        return dict(_modular_component_types(pipeline_cls, tuple(workflows)))
    component_types: dict[str, type] = {}
    for name, candidates in pipeline_cls._get_signature_types().items():
        component_cls = next(
            (
                candidate
                for candidate in candidates
                if candidate is not type(None) and candidate is not inspect.Parameter.empty
            ),
            None,
        )
        if isinstance(component_cls, type):
            component_types[name] = component_cls
    return component_types


@cache
def _modular_component_types(pipeline_cls: type, workflows: tuple[str, ...]) -> dict[str, type]:
    import diffusers

    blocks_name = getattr(pipeline_cls, "default_blocks_name", None)
    blocks_cls = None
    if isinstance(blocks_name, str):
        blocks_cls = getattr(diffusers, blocks_name, None)
    if blocks_cls is None:
        raise ValueError(
            f"Attempted to discover components of '{pipeline_cls.__name__}'. Failed because "
            f"default_blocks_name ({blocks_name!r}) does not name a blocks class in diffusers."
        )
    blocks = blocks_cls()
    if workflows:
        specs = [spec for workflow in workflows for spec in blocks.get_workflow(workflow).expected_components]
    else:
        specs = list(blocks.expected_components)
    component_types: dict[str, type] = {}
    for spec in specs:
        if spec.default_creation_method != "from_pretrained" or not isinstance(spec.type_hint, type):
            continue
        component_types.setdefault(spec.name, spec.type_hint)
    return component_types
