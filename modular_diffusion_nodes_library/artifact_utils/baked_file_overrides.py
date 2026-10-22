"""Copy GGUF (and similar file) overrides into a baked folder, and restore them on load."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.recipe_codec import resolve_recipe_value
from modular_diffusion_nodes_library.component_loading.component_slots import component_config_filename
from modular_diffusion_nodes_library.component_loading.config_resolver import HF_REPO_ID_PATTERN

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ModelComponentArtifact
    from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact

BuiltPipeline: TypeAlias = ModularPipeline | DiffusionPipeline | Any
PipeComponent: TypeAlias = Any


def gguf_overrides_from_artifact(artifact: DiffusionPipelineArtifact) -> dict[str, ModelComponentArtifact]:
    """Slot → component artifact for overrides whose weights are a .gguf file."""
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ModelComponentArtifact

    overrides = artifact.build_data.get("_component_overrides") or {}
    if not isinstance(overrides, dict):
        return {}
    return {
        str(slot): override
        for slot, override in overrides.items()
        if isinstance(override, ModelComponentArtifact) and override.is_quantized
    }


def component_refuses_official_save(component: PipeComponent) -> bool:
    """True when hf_quantizer says the weights cannot use official save_pretrained.

    Diffusers exposes ``is_serializable`` as a property (GGUFQuantizer).
    Transformers exposes it as a method (QuantoHfQuantizer).
    """
    quantizer = getattr(component, "hf_quantizer", None)
    if quantizer is None:
        return False
    value = getattr(quantizer, "is_serializable", None)
    if callable(value):
        value = value()
    return value is False


def _quantizer_type_name(component: PipeComponent) -> str:
    quantizer = getattr(component, "hf_quantizer", None)
    if quantizer is None:
        return ""
    return type(quantizer).__name__.lower()


def is_gguf_quantizer_component(component: PipeComponent) -> bool:
    return "gguf" in _quantizer_type_name(component)


def is_torchao_quantizer_component(component: PipeComponent) -> bool:
    return "torchao" in _quantizer_type_name(component)


def iter_pipe_components(pipe: BuiltPipeline) -> list[tuple[str, PipeComponent]]:
    components = getattr(pipe, "components", None)
    if isinstance(components, dict) and components:
        return [(name, component) for name, component in components.items() if component is not None]
    return []


def skip_official_save_slots(pipe: BuiltPipeline, artifact: DiffusionPipelineArtifact) -> set[str]:
    names = set(gguf_overrides_from_artifact(artifact))
    for name, component in iter_pipe_components(pipe):
        if is_gguf_quantizer_component(component):
            names.add(name)
    return names


def assert_baked_save_feasible(pipe: BuiltPipeline, artifact: DiffusionPipelineArtifact, skip_slots: set[str]) -> None:
    """Raise if a built slot cannot use official save_pretrained or a GGUF file copy."""
    gguf = gguf_overrides_from_artifact(artifact)
    for name, component in iter_pipe_components(pipe):
        if name in gguf or is_gguf_quantizer_component(component):
            if name not in skip_slots:
                msg = f"Attempted to bake slot '{name}'. Failed because it is GGUF and official save was not skipped."
                raise ValueError(msg)
            if name not in gguf:
                msg = (
                    f"Attempted to bake slot '{name}'. Failed because it is GGUF and no file "
                    "override was recorded to copy."
                )
                raise ValueError(msg)
            continue
        if is_torchao_quantizer_component(component):
            msg = (
                f"Attempted to bake slot '{name}'. Failed because it is torchao "
                "and cannot use official save_pretrained."
            )
            raise ValueError(msg)
        if component_refuses_official_save(component):
            msg = f"Attempted to bake slot '{name}'. Failed because it is not serializable and is not GGUF."
            raise ValueError(msg)


def _noop_save_pretrained(*args: Any, **kwargs: Any) -> None:
    return None


@contextmanager
def noop_official_saves(pipe: BuiltPipeline, slot_names: set[str]) -> Iterator[None]:
    """Replace each skipped slot's save_pretrained with a no-op, then restore."""
    restored: list[tuple[PipeComponent, Any]] = []
    for name, component in iter_pipe_components(pipe):
        if name not in slot_names or not hasattr(component, "save_pretrained"):
            continue
        original = component.save_pretrained
        component.save_pretrained = _noop_save_pretrained
        restored.append((component, original))
    try:
        yield
    finally:
        for component, original in restored:
            component.save_pretrained = original


def copy_gguf_override(override: ModelComponentArtifact, slot_dir: Path, baked_root: Path) -> dict[str, Any]:
    """Copy the .gguf (and its config) into `slot_dir`. Return a recipe dict."""
    raw_path = override.file_path
    if not raw_path:
        msg = (
            f"Attempted to copy a GGUF override for '{override.component}'. "
            "Failed because file_path is missing."
        )
        raise ValueError(msg)
    src = Path(raw_path)
    if not src.is_file():
        msg = (
            f"Attempted to copy a GGUF override for '{override.component}'. "
            f"Failed because '{src}' is not a file."
        )
        raise ValueError(msg)

    slot_dir = Path(slot_dir)
    baked_root = Path(baked_root)
    slot_dir.mkdir(parents=True, exist_ok=True)
    dest = slot_dir / src.name
    shutil.copy2(src, dest)

    copied_config = _copy_override_config(override, src, slot_dir)
    recipe = override.to_recipe_dict()
    recipe["file_path"] = dest.relative_to(baked_root).as_posix()
    if copied_config is not None:
        recipe["config_source"] = copied_config.relative_to(baked_root).as_posix()
    return recipe


def _copy_override_config(override: ModelComponentArtifact, src: Path, slot_dir: Path) -> Path | None:
    config_name = component_config_filename(override.component)
    config_source = override.config_source
    # An explicit config source is authoritative; only use a sibling config when none was supplied.
    if config_source:
        cfg = Path(config_source)
        if cfg.is_file():
            return _copy_config_file(cfg, slot_dir)
        if cfg.is_dir():
            config_file = cfg / config_name
            if not config_file.is_file():
                msg = (
                    f"Attempted to copy config for '{override.component}'. "
                    f"Failed because explicit config directory '{cfg}' does not contain '{config_name}'."
                )
                raise FileNotFoundError(msg)
            return _copy_config_file(config_file, slot_dir)
        if HF_REPO_ID_PATTERN.fullmatch(config_source):
            # Preserve an explicit HF repo ID; do not replace it with a sibling config file.
            return None
        msg = (
            f"Attempted to copy config for '{override.component}'. "
            f"Failed because explicit config source '{config_source}' does not exist."
        )
        raise FileNotFoundError(msg)

    sibling = src.parent / config_name
    if sibling.is_file():
        return _copy_config_file(sibling, slot_dir)
    return None


def _copy_config_file(config_file: Path, slot_dir: Path) -> Path:
    dest = slot_dir / config_file.name
    shutil.copy2(config_file, dest)
    return dest


def resolve_file_overrides(file_overrides: dict[str, Any], baked_path: Path) -> dict[str, ModelComponentArtifact]:
    """Rebuild component artifacts and resolve relative paths against the bake folder.

    ``file_overrides`` is the ``file_overrides`` section from ``baked-config.yaml``
    """
    from modular_diffusion_nodes_library.artifact_utils import (  # noqa: F401
        text_encoder_component_artifact as _text_encoder_component_artifact,
    )
    from modular_diffusion_nodes_library.artifact_utils import (  # noqa: F401
        tokenizer_component_artifact as _tokenizer_component_artifact,
    )

    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ModelComponentArtifact

    baked_path = Path(baked_path)
    resolved: dict[str, ModelComponentArtifact] = {}
    for slot, value in file_overrides.items():
        artifact = resolve_recipe_value(value)
        if not isinstance(artifact, ModelComponentArtifact):
            msg = (
                f"Attempted to resolve file override for '{slot}'. "
                f"Failed because it is {type(artifact).__name__}, not a model component."
            )
            raise TypeError(msg)
        resolved[str(slot)] = _resolve_override_paths(artifact, baked_path)
    return resolved


def _resolve_override_paths(artifact: ModelComponentArtifact, baked_path: Path) -> ModelComponentArtifact:
    updates: dict[str, str] = {}
    file_path = artifact.file_path
    if isinstance(file_path, str) and file_path and not Path(file_path).is_absolute():
        updates["file_path"] = str((baked_path / file_path).resolve())
    config_source = artifact.config_source
    if isinstance(config_source, str) and config_source and not Path(config_source).is_absolute():
        candidate = baked_path / config_source
        if candidate.exists():
            updates["config_source"] = str(candidate.resolve())
    if not updates:
        return artifact
    return replace(artifact, **updates)
