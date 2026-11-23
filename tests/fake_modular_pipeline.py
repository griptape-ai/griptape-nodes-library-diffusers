"""A stand-in for Diffusers' `ModularPipeline`, shared by the modular loader tests.

Each loader test used to carry its own stub, and the stubs drifted from Diffusers in ways that hid
bugs: a spec mutated through `get_component_spec` stayed mutated. Where this fake does more than
record calls, it copies Diffusers 0.42:

- `get_component_spec` and `blocks` return copies, so mutating what they return changes nothing.
- `blocks.get_workflow` keeps only the sub-blocks still present, so popping a sub-block from
  `_blocks.sub_blocks` drops its components from every workflow.
- `from_config` components exist as soon as the pipeline does, as `ModularPipeline.from_pretrained`
  creates them.
- `update_components` ignores a name the pipeline has no spec for.

Specs are real `ComponentSpec` objects for a `FakeComponent`, so loading through a spec runs Diffusers'
own `ComponentSpec.load` and ends in `FakeComponent.from_pretrained`. Callers need the execution
environment. `tests/test_fake_modular_pipeline_matches_diffusers.py` holds the fake to all of this.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import pytest
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

BASE_REPO = "org/base"


class FakeComponent:
    """What a fake spec loads: it records the repo and the kwargs its `from_pretrained` was given."""

    def __init__(self, source: Any, load_kwargs: dict[str, Any]) -> None:
        self.source = source
        self.load_kwargs = load_kwargs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Any, **kwargs: Any) -> FakeComponent:
        return cls(pretrained_model_name_or_path, kwargs)


def component_spec(name: str, **fields: Any) -> ComponentSpec:
    """Build a spec for a `FakeComponent` in `BASE_REPO`, in a subfolder named after it, unless overridden."""
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec

    fields.setdefault("type_hint", FakeComponent)
    fields.setdefault("pretrained_model_name_or_path", BASE_REPO)
    fields.setdefault("subfolder", name)
    return ComponentSpec(name=name, **fields)


class FakeBlocks:
    """Sub-blocks that each declare components, plus workflows naming the sub-blocks they run."""

    def __init__(self, sub_blocks: dict[str, list[ComponentSpec]], workflows: dict[str, list[str]]) -> None:
        self.sub_blocks = sub_blocks
        self._workflows = workflows

    @property
    def expected_components(self) -> list[ComponentSpec]:
        expected: list[ComponentSpec] = []
        for specs in self.sub_blocks.values():
            for spec in specs:
                if spec not in expected:
                    expected.append(spec)
        return expected

    def get_workflow(self, workflow: str) -> FakeBlocks:
        if workflow not in self._workflows:
            msg = f"Workflow {workflow!r} not found. Available workflows: {list(self._workflows)}"
            raise ValueError(msg)
        selected = {name: specs for name, specs in self.sub_blocks.items() if name in self._workflows[workflow]}
        return FakeBlocks(selected, {})


class FakeModularPipeline:
    """Records what the loader registers, in order, in `events`."""

    def __init__(
        self,
        specs: list[ComponentSpec],
        *,
        sub_blocks: dict[str, list[str]],
        workflows: dict[str, list[str]] | None = None,
    ) -> None:
        self._component_specs: dict[str, ComponentSpec] = {}
        for spec in specs:
            if spec.name is None:
                msg = f"Attempted to build a fake pipeline. Failed because spec {spec!r} has no name."
                raise ValueError(msg)
            self._component_specs[spec.name] = spec
        block_specs = {block: [self._component_specs[name] for name in names] for block, names in sub_blocks.items()}
        self._blocks = FakeBlocks(block_specs, workflows or {})
        for name, spec in self._component_specs.items():
            component = None
            if spec.default_creation_method == "from_config":
                component = object()
            setattr(self, name, component)
        self.events: list[str] = []

    def __getattr__(self, name: str) -> Any:
        # Only reached for a name never set, so lookups stay as strict as Diffusers'. Declaring it lets
        # tests read components such as `pipeline.vae`, which are set per spec in `__init__`.
        raise AttributeError(name)

    @property
    def blocks(self) -> FakeBlocks:
        return deepcopy(self._blocks)

    @property
    def components(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._component_specs}

    def get_component_spec(self, name: str) -> ComponentSpec:
        return deepcopy(self._component_specs[name])

    def update_components(self, **components: Any) -> None:
        for name, component in components.items():
            if name not in self._component_specs:
                continue
            self.events.append(f"register:{name}")
            setattr(self, name, component)


def loader_test_pipeline(**specs: ComponentSpec) -> FakeModularPipeline:
    """The default pipeline for loader tests; pass specs by name to replace or add components.

    Workflow "active" runs the "denoise" and "decode" sub-blocks, so "unused" is never required.
    """
    defaults = {
        "transformer": component_spec("transformer"),
        "vae": component_spec("vae"),
        "scheduler": component_spec(
            "scheduler", pretrained_model_name_or_path=None, default_creation_method="from_config"
        ),
        "unused": component_spec("unused"),
    }
    defaults.update(specs)
    return FakeModularPipeline(
        list(defaults.values()),
        sub_blocks={"denoise": ["transformer", "scheduler"], "decode": ["vae"], "extra": ["unused"]},
        workflows={"active": ["denoise", "decode"]},
    )


class FakeComponentsManager:
    """Appends "offload" to the pipeline's events when the loader enables CPU offload."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.offload_kwargs: dict[str, Any] | None = None

    def enable_auto_cpu_offload(self, **kwargs: Any) -> None:
        self.offload_kwargs = kwargs
        self._events.append("offload")


def patch_components_manager(
    monkeypatch: pytest.MonkeyPatch, make_manager: Callable[[], FakeComponentsManager]
) -> None:
    """Make the loader's `ComponentsManager()` call `make_manager` instead."""
    # Diffusers evaluates its `ComponentsManager | None` annotations when this module is first imported,
    # and a function cannot stand in for the class there, so import it before patching.
    import diffusers.modular_pipelines.modular_pipeline  # noqa: F401

    monkeypatch.setattr("diffusers.modular_pipelines.components_manager.ComponentsManager", make_manager)


def install_components_manager(monkeypatch: pytest.MonkeyPatch, pipeline: FakeModularPipeline) -> FakeComponentsManager:
    """Make the loader's `ComponentsManager()` return a fake that reports into `pipeline.events`."""
    manager = FakeComponentsManager(pipeline.events)
    patch_components_manager(monkeypatch, lambda: manager)
    return manager
