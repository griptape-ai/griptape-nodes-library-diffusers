from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

if TYPE_CHECKING:
    from diffusers.modular_pipelines.components_manager import ComponentsManager
    from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
    from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec


@dataclass(frozen=True)
class ComponentLoadContext:
    """Resolved source and loader-specific kwargs shared by preflight checks and loaders."""

    name: str
    source: str | list[str]
    subfolder: str | None
    revision: str | None
    options: Mapping[str, Any]


@dataclass(frozen=True)
class ModularPipelineLoadOptions:
    """Pipeline-specific policies without duplicating the common loading lifecycle.

    Validators run before any weights load. Supplied components skip validators and loaders.
    Keep loader configuration unchanged for the duration of a load.
    """

    workflows: tuple[str, ...] = ()
    dtype: Any = None
    component_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    component_validators: Mapping[str, Callable[[ComponentLoadContext], None]] = field(default_factory=dict)
    component_loaders: Mapping[str, Callable[[ComponentLoadContext], Any]] = field(default_factory=dict)
    prepare_pipeline: Callable[[Any], None] | None = None
    placement: Literal["cpu", "auto_cpu_offload"] = "auto_cpu_offload"
    device: Any = None
    memory_reserve_margin: str = "12GB"


class ModularPipelineLoader:
    def __init__(self, *, pipeline_cls: type, options: ModularPipelineLoadOptions) -> None:
        self.pipeline_cls = pipeline_cls
        self.options = options

    def from_pretrained(
        self,
        source: str,
        *,
        revision: str | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> ModularPipeline:
        import torch
        from diffusers.modular_pipelines.components_manager import ComponentsManager

        self._validate_options()
        overrides = overrides or {}
        manager = ComponentsManager()
        pipe = self.pipeline_cls.from_pretrained(
            source, revision=revision, components_manager=manager, local_files_only=True
        )
        if self.options.prepare_pipeline is not None:
            self.options.prepare_pipeline(pipe)
        required = self._required_component_names(pipe)
        self._validate_component_names(required, overrides)
        dtype = self.options.dtype
        if dtype is None:
            dtype = torch.bfloat16
        pending_components = self._prepare_pending_components(
            pipe, required, overrides, source=source, revision=revision, dtype=dtype
        )
        supplied = self._materialize_overrides(overrides)
        if supplied:
            pipe.update_components(**supplied)
        self._load_custom_components(pipe, pending_components)
        self._load_defaults(pipe, pending_components)
        missing = sorted(name for name in required if getattr(pipe, name, None) is None)
        if missing:
            raise ValueError(
                f"Attempted to build {self.pipeline_cls.__name__} from '{source}'. Failed because components "
                f"{missing} did not load."
            )
        self._apply_placement(pipe, manager)
        return pipe

    def _apply_placement(self, pipe: ModularPipeline, manager: ComponentsManager) -> None:
        if self.options.placement == "auto_cpu_offload":
            device = self.options.device
            if device is None:
                device = get_best_device()
            manager.enable_auto_cpu_offload(device=device, memory_reserve_margin=self.options.memory_reserve_margin)
        else:
            pipe.to("cpu")
        cast(Any, pipe)._manages_runtime_optimisation = True

    def _materialize_overrides(self, overrides: Mapping[str, Any]) -> dict[str, Any]:
        supplied = {}
        for name, component in overrides.items():
            if isinstance(component, ComponentArtifact):
                component = component.materialize(pipeline_cls=self.pipeline_cls, slot=name)
            supplied[name] = component
        return supplied

    def _prepare_pending_components(
        self,
        pipe: ModularPipeline,
        required: Iterable[str],
        overrides: Mapping[str, Any],
        *,
        source: str,
        revision: str | None,
        dtype: Any,
    ) -> list[ComponentLoadContext]:
        pending_components = []
        for name in required:
            if name in overrides or getattr(pipe, name, None) is not None:
                continue
            spec = pipe.get_component_spec(name)
            if name not in self.options.component_loaders and spec.default_creation_method != "from_pretrained":
                continue
            options = dict(self.options.component_options.get(name, {}))
            options.setdefault("dtype", dtype)
            pending_components.append(self._repo_component_context(name, spec, source, revision, options))
        for context in pending_components:
            validator = self.options.component_validators.get(context.name)
            if validator is not None:
                validator(context)
        return pending_components

    @classmethod
    def _repo_component_context(
        cls, name: str, spec: ComponentSpec, base_source: str, base_revision: str | None, options: dict[str, Any]
    ) -> ComponentLoadContext:
        """Assemble the context after resolving its source and revision policies."""
        component_source = cls._resolve_component_source(spec, base_source, options)
        component_revision = cls._resolve_component_revision(
            spec, component_source, base_source, base_revision, options
        )
        return ComponentLoadContext(
            name=name,
            source=component_source,
            subfolder=options.pop("subfolder", spec.subfolder),
            revision=component_revision,
            options=options,
        )

    @staticmethod
    def _resolve_component_source(spec: ComponentSpec, base_source: str, options: dict[str, Any]) -> str | list[str]:
        """Use the spec unless overridden; an explicit None selects the base source."""
        component_source = options.pop("pretrained_model_name_or_path", spec.pretrained_model_name_or_path)
        if component_source is None:
            component_source = base_source
        return component_source

    @staticmethod
    def _resolve_component_revision(
        spec: ComponentSpec,
        component_source: str | list[str],
        base_source: str,
        base_revision: str | None,
        options: dict[str, Any],
    ) -> str | None:
        """Prefer explicit revisions, including None, then the base or unchanged spec revision."""
        spec_source = spec.pretrained_model_name_or_path
        if spec_source is None:
            spec_source = base_source
        inherited_revision = None
        if component_source == spec_source:
            inherited_revision = spec.revision
        if component_source == base_source and base_revision is not None:
            inherited_revision = base_revision
        return options.pop("revision", inherited_revision)

    def _load_custom_components(self, pipe: ModularPipeline, pending_components: list[ComponentLoadContext]) -> None:
        for context in pending_components:
            loader = self.options.component_loaders.get(context.name)
            if loader is None:
                continue
            component = loader(context)
            if component is None:
                raise ValueError(f"Specialised loader for component '{context.name}' returned no component.")
            pipe.update_components(**{context.name: component})

    def _load_defaults(self, pipe: ModularPipeline, pending_components: list[ComponentLoadContext]) -> None:
        names = []
        load_kwargs: dict[str, Any] = {"local_files_only": True}
        for context in pending_components:
            if self.options.component_loaders.get(context.name) is not None:
                continue
            names.append(context.name)
            pipe.get_component_spec(context.name).pretrained_model_name_or_path = context.source
            options = {
                "pretrained_model_name_or_path": context.source,
                "subfolder": context.subfolder,
                "revision": context.revision,
                **context.options,
            }
            for key, value in options.items():
                load_kwargs.setdefault(key, {})[context.name] = value
        if names:
            pipe.load_components(names=names, **load_kwargs)

    def _validate_component_names(self, required: Iterable[str], overrides: Mapping[str, Any]) -> None:
        required_names = set(required)
        unknown = sorted(set(overrides) - required_names)
        if unknown:
            raise ValueError(
                f"Attempted to build {self.pipeline_cls.__name__} with component overrides. Failed with override "
                f"slots {unknown} because the loaded pipeline does not declare them in the selected workflows."
            )
        configured = (
            set(self.options.component_options)
            | set(self.options.component_validators)
            | set(self.options.component_loaders)
        )
        unknown_options = sorted(configured - required_names)
        if unknown_options:
            raise ValueError(
                f"Attempted to configure {self.pipeline_cls.__name__}. Failed because components "
                f"{unknown_options} are not declared by the selected workflows."
            )

    def _validate_options(self) -> None:
        if self.options.placement not in {"cpu", "auto_cpu_offload"}:
            raise ValueError(f"Unsupported modular pipeline placement policy: {self.options.placement!r}.")
        for name, options in self.options.component_options.items():
            forbidden = set(options) & {"names", "workflow", "components_manager", "local_files_only"}
            if forbidden:
                raise ValueError(
                    f"Attempted to configure component '{name}'. Failed because options {sorted(forbidden)} "
                    "are owned by the shared loader."
                )

    def _required_component_names(self, pipe: ModularPipeline) -> list[str]:
        blocks = pipe.blocks
        if self.options.workflows:
            specs = [
                spec
                for workflow in self.options.workflows
                for spec in blocks.get_workflow(workflow).expected_components
            ]
        else:
            specs = list(blocks.expected_components)
        names = []
        for spec in specs:
            name = spec.name
            if name is None:
                raise ValueError(
                    f"Attempted to load {self.pipeline_cls.__name__}. Failed because a component has no name."
                )
            names.append(name)
        return list(dict.fromkeys(names))
