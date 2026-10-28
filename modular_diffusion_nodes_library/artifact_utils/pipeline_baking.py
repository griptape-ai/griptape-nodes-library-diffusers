"""Save/load a built pipeline as a local Diffusers/modular repo ("baking").

Baking persists fused LoRAs and (eventually) quantised weights via diffusers'
own `save_pretrained`/`from_pretrained` APIs, so a session's model prep does
not need to be repeated. See `make_baked_artifact()` for how a baked folder is
wrapped back into the normal `DiffusionPipelineArtifact` build/cache path.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from modular_diffusion_nodes_library.artifact_utils.baked_file_overrides import (
    assert_baked_save_feasible,
    copy_gguf_override,
    gguf_overrides_from_artifact,
    is_gguf_quantizer_component,
    iter_pipe_components,
    noop_official_saves,
    skip_official_save_slots,
)
from modular_diffusion_nodes_library.artifact_utils.packed_quant_io import (
    QUANT_MAP_FILENAME,
    write_packed_quant_maps,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    DEFAULT_BAKED_DTYPE,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.recipe_codec import (
    PIPELINE_RECIPE_SCHEMA_VERSION,
)
from modular_diffusion_nodes_library.utils.lora_apply_utils import lora_adapter_name
from modular_diffusion_nodes_library.utils.pipeline_utils import prepare_pipeline_for_baked_save

logger = logging.getLogger("modular_diffusers_nodes_library")

_MODEL_INDEX = "model_index.json"
_MODULAR_MODEL_INDEX = "modular_model_index.json"
BAKED_CONFIG_FILENAME = "baked-config.yaml"
_RUNTIME_LORA_FOLDER = Path("loras")
_TEMP_BAKE_SUFFIX = ".tmp-bake"
_PREVIOUS_BAKE_SUFFIX = ".previous-bake"


class BakedLayout(Enum):
    """Which diffusers repo layout a baked folder uses."""

    STANDARD = "standard"
    MODULAR = "modular"

    def index_filename(self) -> str:
        if self is BakedLayout.MODULAR:
            return _MODULAR_MODEL_INDEX
        return _MODEL_INDEX


def detect_baked_layout(folder: Path) -> BakedLayout | None:
    """Return the repo layout of `folder`, or None if it isn't a recognizable Diffusers/modular repo."""
    for layout in (BakedLayout.MODULAR, BakedLayout.STANDARD):
        if (folder / layout.index_filename()).is_file():
            return layout
    return None


def _validate_replaceable_bake_directory(folder: Path, *, role: str) -> None:
    if not folder.exists() and not folder.is_symlink():
        return
    if not folder.is_dir() or folder.is_symlink():
        msg = f"Cannot use baked pipeline {role} '{folder}' because it is not a regular directory."
        raise NotADirectoryError(msg)
    if next(folder.iterdir(), None) is None:
        return
    if detect_baked_layout(folder) is not None and (folder / BAKED_CONFIG_FILENAME).is_file():
        return
    msg = (
        f"Cannot use baked pipeline {role} '{folder}' because it is nonempty and is not a generated "
        "Full Pipeline bake. No files were changed. Choose an empty directory or an existing Full Pipeline bake."
    )
    raise FileExistsError(msg)


@dataclass(frozen=True)
class RuntimeLoraCopy:
    source_path: Path
    relative_path: Path
    weight: float


def plan_runtime_lora_copies(artifact: DiffusionPipelineArtifact) -> list[RuntimeLoraCopy]:
    """Validate runtime LoRA sources and map them into the baked folder."""
    copies: list[RuntimeLoraCopy] = []
    destinations: dict[str, Path] = {}
    for spec in artifact.runtime_loras().values():
        requested_path = Path(spec.path).expanduser()
        source_path = requested_path.resolve()
        adapter_name = lora_adapter_name(str(source_path))
        if not source_path.is_file():
            msg = f"Runtime LoRA adapter '{adapter_name}' cannot be saved because its file is missing: {source_path}"
            raise FileNotFoundError(msg)

        relative_path = _RUNTIME_LORA_FOLDER / requested_path.name
        destination_key = relative_path.as_posix().casefold()
        previous_source = destinations.get(destination_key)
        if previous_source is not None:
            msg = (
                "Runtime LoRA filename collision\n\n"
                "Full Pipeline saves runtime LoRAs in one 'loras/' folder. Rename or copy one source file to a "
                "unique filename, update its Load LoRA node, and retry.\n\n"
                f"Destination:\n  {relative_path.as_posix()}\n\n"
                f"Source files:\n  1. {previous_source}\n  2. {source_path}"
            )
            raise ValueError(msg)
        destinations[destination_key] = source_path
        copies.append(
            RuntimeLoraCopy(
                source_path=source_path,
                relative_path=relative_path,
                weight=float(spec.weight),
            )
        )
    return copies


def save_baked_pipeline(
    pipe: Any,
    artifact: DiffusionPipelineArtifact,
    folder: Path,
    *,
    runtime_lora_copies: list[RuntimeLoraCopy] | None = None,
) -> None:
    """Atomically save the built pipe (weights + baked config) into `folder`.

    save_pretrained is not atomic, so a failure would otherwise leave a corrupt
    partial repo. Save into a unique sibling temp dir, retain any existing bake
    under a recovery name, then install the completed replacement.
    """
    folder = folder.expanduser().absolute()
    backup_dir = folder.with_name(folder.name + _PREVIOUS_BAKE_SUFFIX)
    _recover_previous_bake(folder, backup_dir)
    _validate_replaceable_bake_directory(folder, role="destination")

    artifact.check_full_pipeline_bake_policy()
    if runtime_lora_copies is None:
        runtime_lora_copies = plan_runtime_lora_copies(artifact)
    folder.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"{folder.name}{_TEMP_BAKE_SUFFIX}-",
        dir=folder.parent,
        ignore_cleanup_errors=True,
    ) as temporary_path:
        tmp_dir = Path(temporary_path) / "bake"
        tmp_dir.mkdir()
        _write_baked_pipeline_contents(pipe, artifact, tmp_dir, runtime_lora_copies)
        _install_completed_bake(tmp_dir, folder, backup_dir)


def _write_baked_pipeline_contents(
    pipe: Any,
    artifact: DiffusionPipelineArtifact,
    folder: Path,
    runtime_lora_copies: list[RuntimeLoraCopy],
) -> None:
    skip_slots = skip_official_save_slots(pipe, artifact)
    assert_baked_save_feasible(pipe, artifact, skip_slots)
    with prepare_pipeline_for_baked_save(pipe, artifact.config_hash):
        if skip_slots:
            logger.warning(
                "Skipping official save_pretrained for %s (GGUF); copying .gguf files.",
                sorted(skip_slots),
            )
        with noop_official_saves(pipe, skip_slots):
            pipe.save_pretrained(str(folder))
        file_overrides = _write_skipped_slots(pipe, artifact, folder, skip_slots)
        packed_slots = write_packed_quant_maps(pipe, folder)

    runtime_loras = _copy_runtime_loras(runtime_lora_copies, folder)
    if packed_slots:
        logger.info("Wrote packed quanto maps for slots: %s", packed_slots)
    # baked-config top-level "torch_dtype": pipe.dtype at bake → _baked_dtype → from_pretrained / quanto reload.
    dtype_name = str(getattr(pipe, "dtype", None) or f"torch.{DEFAULT_BAKED_DTYPE}").replace("torch.", "")
    extra: dict[str, Any] = {
        "packed_quant_slots": packed_slots,
        "runtime_loras": runtime_loras,
    }
    if file_overrides:
        extra["file_overrides"] = file_overrides
    _write_baked_config(folder, artifact, dtype_name, **extra)


def _recover_previous_bake(folder: Path, backup_dir: Path) -> None:
    backup_exists = backup_dir.exists() or backup_dir.is_symlink()
    if not backup_exists:
        return
    if folder.exists() or folder.is_symlink():
        msg = (
            f"Cannot save baked pipeline to '{folder}' because recovery path '{backup_dir}' already exists. "
            "Preserving both directories; remove or relocate the recovery path before retrying."
        )
        raise FileExistsError(msg)
    _validate_replaceable_bake_directory(backup_dir, role="recovery path")
    logger.warning("Restoring interrupted baked pipeline installation from %s", backup_dir)
    os.replace(backup_dir, folder)


def _restore_previous_destination(
    backup_dir: Path,
    folder: Path,
    *,
    operation: str,
    error: BaseException,
) -> None:
    try:
        os.replace(backup_dir, folder)
    except BaseException as restore_error:
        msg = (
            f"{operation} failed, and restoring the previous destination also failed. "
            f"The previous bake remains at '{backup_dir}'. Original error: {error}"
        )
        raise RuntimeError(msg) from restore_error


def _install_completed_bake(tmp_dir: Path, folder: Path, backup_dir: Path) -> None:
    had_previous_bake = folder.exists() or folder.is_symlink()
    if had_previous_bake:
        _validate_replaceable_bake_directory(folder, role="destination")
        if backup_dir.exists() or backup_dir.is_symlink():
            msg = (
                f"Cannot save baked pipeline to '{folder}' because recovery path '{backup_dir}' already exists. "
                "Preserving both directories; remove or relocate the recovery path before retrying."
            )
            raise FileExistsError(msg)
        os.replace(folder, backup_dir)
        try:
            _validate_replaceable_bake_directory(backup_dir, role="previous destination")
        except BaseException as validation_error:
            _restore_previous_destination(
                backup_dir,
                folder,
                operation=f"Validating the previous destination at '{backup_dir}'",
                error=validation_error,
            )
            raise

    try:
        os.replace(tmp_dir, folder)
    except BaseException as install_error:
        if had_previous_bake:
            _restore_previous_destination(
                backup_dir,
                folder,
                operation=f"Installing baked pipeline to '{folder}'",
                error=install_error,
            )
        raise

    if had_previous_bake:
        try:
            shutil.rmtree(backup_dir)
        except OSError:
            logger.warning(
                "Installed baked pipeline at %s but could not remove previous bake at %s",
                folder,
                backup_dir,
                exc_info=True,
            )


def _copy_runtime_loras(copies: list[RuntimeLoraCopy], folder: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in copies:
        adapter_name = lora_adapter_name(str(item.source_path))
        if not item.source_path.is_file():
            msg = (
                f"Runtime LoRA adapter '{adapter_name}' cannot be saved because its file is missing: {item.source_path}"
            )
            raise FileNotFoundError(msg)
        destination = folder / item.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item.source_path, destination)
        entries.append({"path": item.relative_path.as_posix(), "weight": item.weight})
    return entries


def _write_skipped_slots(
    pipe: Any,
    artifact: DiffusionPipelineArtifact,
    folder: Path,
    skip_slots: set[str],
) -> dict[str, Any]:
    """Fill skipped slots: copy GGUF, or raise if there is no file to copy."""
    gguf = gguf_overrides_from_artifact(artifact)
    components = dict(iter_pipe_components(pipe))
    file_overrides: dict[str, Any] = {}
    for name in sorted(set(skip_slots) | set(gguf)):
        if name in gguf:
            file_overrides[name] = copy_gguf_override(gguf[name], folder / name, folder)
            continue
        component = components.get(name)
        if component is None:
            continue
        if is_gguf_quantizer_component(component):
            msg = (
                f"Attempted to bake slot '{name}'. Failed because it is GGUF and no file override was recorded to copy."
            )
            raise ValueError(msg)
        msg = f"Attempted to bake slot '{name}'. Failed because official save was skipped and the slot is not GGUF."
        raise ValueError(msg)
    return file_overrides


def _write_baked_config(
    folder: Path,
    artifact: DiffusionPipelineArtifact,
    dtype_name: str,
    **extra: Any,
) -> None:
    """Write a flat pipeline manifest with baked-only metadata appended."""
    artifact_config = artifact.to_baked_config_dict()
    baked_quantization_mode = artifact_config.pop("baked_quantization_mode")
    config = {
        "schema_version": PIPELINE_RECIPE_SCHEMA_VERSION,
        **artifact_config,
        "torch_dtype": dtype_name,
        "baked_quantization_mode": baked_quantization_mode,
        **extra,
    }
    (folder / BAKED_CONFIG_FILENAME).write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def read_baked_config(folder: Path) -> dict[str, Any]:
    """Read the sidecar baked config, or {} if a folder was baked without one."""
    config_path = folder / BAKED_CONFIG_FILENAME
    if not config_path.is_file():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        msg = f"Baked pipeline manifest must contain a mapping: {config_path}"
        raise ValueError(msg)
    return config


def make_baked_artifact(folder: Path) -> DiffusionPipelineArtifact:
    """Wrap a baked folder as a `DiffusionPipelineArtifact` pointed at the local folder."""
    layout = detect_baked_layout(folder)
    if layout is None:
        msg = f"Folder is not a baked Diffusers/modular repo: {folder}"
        raise ValueError(msg)

    manifest = read_baked_config(folder)
    schema_version = manifest.get("schema_version")
    if schema_version != PIPELINE_RECIPE_SCHEMA_VERSION:
        msg = f"Unsupported baked pipeline manifest schema version: {schema_version!r}"
        raise ValueError(msg)

    if not manifest.get("builder_module") or not manifest.get("builder_class_name"):
        msg = (
            f"Baked folder is missing '{BAKED_CONFIG_FILENAME}' (or builder info within it): {folder}. "
            "It cannot be reloaded without knowing which pipeline builder to use."
        )
        raise ValueError(msg)
    config = dict(manifest)
    config.pop("schema_version")

    # Fold a content signature into the cache key so re-baking the same folder invalidates
    # the in-session pipeline cache instead of returning a stale build.
    folder = folder.resolve()
    signature = _baked_content_signature(folder, layout)
    config_hash = "baked-" + hashlib.sha256(f"{folder}::{signature}".encode()).hexdigest()
    return DiffusionPipelineArtifact.from_baked_config_dict(config, baked_path=folder, config_hash=config_hash)


def _baked_content_signature(folder: Path, layout: BakedLayout) -> str:
    """Fingerprint index + sidecar files so map-only edits invalidate the cache."""
    paths = [folder / layout.index_filename()]
    paths.append(folder / BAKED_CONFIG_FILENAME)
    paths.extend(sorted(folder.rglob(QUANT_MAP_FILENAME)))
    paths.extend(sorted(folder.rglob("*.gguf")))
    parts: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        parts.append(f"{path.relative_to(folder).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
    return ";".join(parts)
