"""Component offload that discards weights on eviction and reloads them from disk, instead of parking them in RAM.

Listed components get a ``DiskBackedOffloadHook``; every other module keeps diffusers' stock ``CustomOffloadHook``.
Import this module inside functions only: it imports torch, diffusers and accelerate.

Scope: only ``torch.nn.Module`` components of a pipeline repo's standard ``from_pretrained`` layout, a folder with
``config.json`` and ``*.safetensors`` (a local folder, or a repo already in the HF cache). Tokenizers, processors and
schedulers hold no weights and are never disk-backed. Components from the Load Component node are not supported
either: a single-file checkpoint has no component folder, and its config (``config_source``) can be any JSON file,
so the ``config.json`` lookups below don't apply to it.

TODO(disk-backed-optimisations): a reload starts from the checkpoint, so post-load changes are lost: quantization,
fused LoRA, layerwise casting, ``enable_*`` hooks. Wrapped (quantized) parameters are rejected; the rest are dropped
silently. Fix by passing load-time options (``quantization_config``) through ``DiskBackedComponent.load``, or by
replaying the change after each reload.

TODO(disk-backed-load-args) **MUST STAY IN SYNC**: the reload arguments (``dtype``, ``quantization_config``, ...)
must match the original load's, or later runs silently use a different configuration than the first.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch  # type: ignore[reportMissingImports]
from accelerate.utils import send_to_device  # type: ignore[reportMissingImports]
from accelerate.utils.memory import clear_device_cache  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.components_manager import (  # type: ignore[reportMissingImports]
    AutoOffloadStrategy,
    ComponentsManager,
    CustomOffloadHook,
    UserCustomOffloadHook,
)
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec  # type: ignore[reportMissingImports]
from huggingface_hub import try_to_load_from_cache

# strategy(hooks=..., model_id=..., model=..., execution_device=...) -> the resident hooks to evict.
type OffloadStrategy = Callable[..., list[UserCustomOffloadHook]]


@dataclass(frozen=True)
class DiskBackedComponent:
    """How to rebuild one component from disk, and which files that depends on."""

    load: Callable[[torch.device], torch.nn.Module]
    weight_files: tuple[Path, ...]


def disk_backed_component_from_spec(spec: ComponentSpec, *, dtype: torch.dtype) -> DiskBackedComponent:
    """Describe a from_pretrained component whose checkpoint is already in the local cache."""
    model_cls = _model_class(spec)
    component_dir = _resolve_component_dir(spec)
    weight_files = tuple(sorted(component_dir.glob("*.safetensors")))
    if not weight_files:
        msg = (
            f"Attempted to describe component '{spec.name}' for disk-backed offload. "
            f"Failed because '{component_dir}' contains no .safetensors files to reload from."
        )
        raise FileNotFoundError(msg)

    def load(device: torch.device) -> torch.nn.Module:
        # Load on the CPU, then move. The files are memory-mapped, so weights pass through RAM.
        # Don't pass `device_map` (this reserves all GPU memory up front and Windows can then run out of commit).
        # Don't pass `disable_mmap=True` (this reads the whole checkpoint into RAM alongside other components,
        # which can cause OOM issues).
        return model_cls.from_pretrained(component_dir, dtype=dtype).to(device)

    return DiskBackedComponent(load=load, weight_files=weight_files)


def empty_component_from_spec(spec: ComponentSpec, *, dtype: torch.dtype) -> torch.nn.Module:
    """Build the component's parameters on the meta device from its config alone, without reading any weights."""
    from modular_diffusion_nodes_library.component_loading.config_resolver import try_load_json_dict
    from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import build_component_on_meta_device

    model_cls = _model_class(spec)
    # Fixed config name is only valid for a from_pretrained folder, not a Load Component single-file config.
    config_path = _resolve_component_dir(spec) / "config.json"
    config = try_load_json_dict(config_path)
    if config is None:
        msg = f"Attempted to build an empty '{spec.name}'. Failed because '{config_path}' is missing or not valid JSON."
        raise FileNotFoundError(msg)
    return build_component_on_meta_device(model_cls, config, dtype)


def enable_disk_backed_offload(
    manager: ComponentsManager,
    device: str | torch.device,
    components: dict[str, DiskBackedComponent],
    *,
    memory_reserve_margin: str = "3GB",
    offload_strategy: OffloadStrategy | None = None,
) -> None:
    """Drop-in for ``manager.enable_auto_cpu_offload`` that disk-backs the modules named in ``components``.

    The manager's auto-offload flag stays off, otherwise ``ComponentsManager.add``/``remove`` would replace these hooks.
    Modules not named in ``components`` get diffusers' stock ``CustomOffloadHook``.
    """
    modules = {
        component_id: component
        for component_id, component in manager.components.items()
        if isinstance(component, torch.nn.Module)
    }
    disk_backed: dict[str, DiskBackedComponent] = {}
    unknown = []
    for name, component in components.items():
        component_ids = [component_id for component_id in manager.get_ids(name) if component_id in modules]
        if not component_ids:
            unknown.append(name)
        disk_backed.update(dict.fromkeys(component_ids, component))
    if unknown:
        msg = (
            f"Attempted to enable disk-backed offload for {unknown}. "
            f"Failed because the manager holds no torch module with that name."
        )
        raise ValueError(msg)

    # Validate everything before attaching any hook, so a rejection leaves the manager untouched.
    for component_id in disk_backed:
        _reject_wrapped_parameters(modules[component_id])

    execution_device = torch.device(device)
    if execution_device.type != "cpu" and execution_device.index is None:
        # `module.device` always carries an index, and `cuda != cuda:0`.
        execution_device = torch.device(f"{execution_device.type}:0")
    if offload_strategy is None:
        offload_strategy = AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)

    user_hooks = []
    for component_id, module in modules.items():
        component = disk_backed.get(component_id)
        if component is None:
            hook = CustomOffloadHook(
                execution_device=execution_device,
                offload_strategy=offload_strategy,  # type: ignore[reportArgumentType]  # upstream annotates the class, documents any callable
            )
        else:
            hook = DiskBackedOffloadHook(
                component,
                execution_device=execution_device,
                offload_strategy=offload_strategy,
            )
        user_hook = UserCustomOffloadHook(model_id=component_id, model=module, hook=hook)
        user_hook.attach()
        user_hooks.append(user_hook)

    for user_hook in user_hooks:
        for other_hook in user_hooks:
            if other_hook is not user_hook:
                user_hook.add_other_hook(other_hook)
    manager.model_hooks = user_hooks


class DiskBackedOffloadHook(CustomOffloadHook):
    """``init_hook`` (attach and eviction) releases the weights; ``pre_forward`` reloads them from disk."""

    def __init__(
        self,
        component: DiskBackedComponent,
        *,
        execution_device: torch.device,
        offload_strategy: OffloadStrategy,
    ):
        super().__init__(
            execution_device=execution_device,
            offload_strategy=offload_strategy,  # type: ignore[reportArgumentType]  # upstream annotates the class, documents any callable
        )
        self._component = component
        self._device = execution_device

    def init_hook(self, module):
        _release_weights(module)
        return module

    def pre_forward(self, module, *args, **kwargs):
        if module.device.type == "meta":
            self._reload_weights(module)
        return send_to_device(args, self._device), send_to_device(kwargs, self._device)

    def _reload_weights(self, module: torch.nn.Module) -> None:
        missing = [str(path) for path in self._component.weight_files if not path.exists()]
        if missing:
            msg = (
                f"Attempted to reload component '{self.model_id}' from disk onto {self._device}. "
                f"Failed because its weight files no longer exist: {missing}. "
                f"Rebuild the pipeline so the model is downloaded again."
            )
            raise FileNotFoundError(msg)

        resident = [hook for hook in self.other_hooks or [] if hook.model.device == self._device]
        if self.offload_strategy is None:
            victims = resident
        else:
            victims = self.offload_strategy(
                hooks=resident,
                model_id=self.model_id,
                model=module,
                execution_device=self._device,
            )
        for victim in victims:
            victim.offload()
        if victims:
            clear_device_cache()

        fresh = self._component.load(self._device)
        _adopt_weights(module, fresh)


def _reject_wrapped_parameters(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        for name, param in submodule._parameters.items():
            if param is not None and type(param) is not torch.nn.Parameter:
                msg = (
                    f"Attempted to release parameter '{name}' of {type(submodule).__name__}. "
                    f"Failed because it is a {type(param).__name__}, and disk-backed offload only supports plain "
                    f"torch.nn.Parameter weights (not quantized or otherwise wrapped ones)."
                )
                raise TypeError(msg)


def _release_weights(module: torch.nn.Module) -> None:
    _reject_wrapped_parameters(module)
    module.to_empty(device="meta")
    clear_device_cache(garbage_collection=True)


def _adopt_weights(module: torch.nn.Module, fresh: torch.nn.Module) -> None:
    """Move ``fresh``'s tensors, non-persistent buffers included, into ``module`` so its hook and references survive."""
    pairs = list(zip(module.modules(), fresh.modules(), strict=True))
    for shell, source in pairs:
        if shell._parameters.keys() != source._parameters.keys() or shell._buffers.keys() != source._buffers.keys():
            msg = (
                f"Attempted to adopt reloaded weights into {type(module).__name__}. "
                f"Failed because submodule {type(shell).__name__} and its reloaded copy declare different "
                f"parameters or buffers."
            )
            raise ValueError(msg)

    for shell, source in pairs:
        for name, param in source._parameters.items():
            shell._parameters[name] = param
        for name, buffer in source._buffers.items():
            shell._buffers[name] = buffer


def _model_class(spec: ComponentSpec) -> type:
    if spec.type_hint is None:
        msg = f"Attempted to locate component '{spec.name}' on disk. Failed because its spec has no type_hint."
        raise ValueError(msg)
    return spec.type_hint


def _resolve_component_dir(spec: ComponentSpec) -> Path:
    repo = spec.pretrained_model_name_or_path
    if not isinstance(repo, str):
        msg = (
            f"Attempted to locate component '{spec.name}' on disk. "
            f"Failed because its spec names {repo!r} rather than a single repo or directory."
        )
        raise ValueError(msg)

    subfolder = spec.subfolder or ""
    local_root = Path(repo)
    if local_root.is_dir():
        return local_root / subfolder

    # `config.json` is the marker for the component folder in the HF cache; a single-file config can be named anything.
    config_name = f"{subfolder}/config.json" if subfolder else "config.json"
    cached = try_to_load_from_cache(repo, filename=config_name, revision=spec.revision)
    if not isinstance(cached, str):
        msg = (
            f"Attempted to locate component '{spec.name}' on disk. "
            f"Failed because '{config_name}' of repo '{repo}' (revision={spec.revision}) is not in the local cache."
        )
        raise FileNotFoundError(msg)
    return Path(cached).parent
