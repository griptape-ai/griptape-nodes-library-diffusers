"""Save/load a built pipeline as a local Diffusers/modular repo ("baking").

Baking persists fused LoRAs and (eventually) quantised weights via diffusers'
own `save_pretrained`/`from_pretrained` APIs, so a session's model prep does
not need to be repeated. See `make_baked_artifact()` for how a baked folder is
wrapped back into the normal `DiffusionPipelineArtifact` build/cache path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

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

logger = logging.getLogger("modular_diffusers_nodes_library")

_MODEL_INDEX = "model_index.json"
_MODULAR_MODEL_INDEX = "modular_model_index.json"
_BAKED_CONFIG_FILE = "baked-config.json"
_BAKED_CONFIG_SCHEMA_VERSION = 1


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


def save_baked_pipeline(pipe: Any, artifact: DiffusionPipelineArtifact, folder: Path) -> None:
    """Atomically save the built pipe (weights + baked config) into `folder`.

    save_pretrained is not atomic, so a failure would otherwise leave a corrupt
    partial repo. Save into a sibling temp dir, then swap it into place in one
    rename so `folder` is only ever a complete bake or untouched.
    """
    folder = folder.resolve()
    tmp_dir = folder.with_name(folder.name + ".tmp-bake")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        skip_slots = skip_official_save_slots(pipe, artifact)
        assert_baked_save_feasible(pipe, artifact, skip_slots)
        if skip_slots:
            logger.warning(
                "Skipping official save_pretrained for %s (GGUF); copying .gguf files.",
                sorted(skip_slots),
            )
        with noop_official_saves(pipe, skip_slots):
            pipe.save_pretrained(str(tmp_dir))
        file_overrides = _write_skipped_slots(pipe, artifact, tmp_dir, skip_slots)
        packed_slots = write_packed_quant_maps(pipe, tmp_dir)
        if packed_slots:
            logger.info("Wrote packed quanto maps for slots: %s", packed_slots)
        # baked-config top-level "torch_dtype": pipe.dtype at bake → _baked_dtype → from_pretrained / quanto reload.
        dtype_name = str(getattr(pipe, "dtype", None) or f"torch.{DEFAULT_BAKED_DTYPE}").replace("torch.", "")
        extra: dict[str, Any] = {"packed_quant_slots": packed_slots}
        if file_overrides:
            extra["file_overrides"] = file_overrides
        _write_baked_config(tmp_dir, artifact, dtype_name, **extra)

        if folder.exists():
            shutil.rmtree(folder)
        os.replace(tmp_dir, folder)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


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
    """Always write a config, wrapping the artifact's own fields in folder-level metadata."""
    config = {
        "schema_version": _BAKED_CONFIG_SCHEMA_VERSION,
        "torch_dtype": dtype_name,  # pipeline-wide
        **artifact.to_baked_config_dict(),
        **extra,
    }
    (folder / _BAKED_CONFIG_FILE).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_baked_config(folder: Path) -> dict[str, Any]:
    """Read the sidecar baked config, or {} if a folder was baked without one."""
    config_path = folder / _BAKED_CONFIG_FILE
    if not config_path.is_file():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def make_baked_artifact(folder: Path) -> DiffusionPipelineArtifact:
    """Wrap a baked folder as a `DiffusionPipelineArtifact` pointed at the local folder."""
    layout = detect_baked_layout(folder)
    if layout is None:
        msg = f"Folder is not a baked Diffusers/modular repo: {folder}"
        raise ValueError(msg)

    config = read_baked_config(folder)
    if not config.get("builder_module") or not config.get("builder_class_name"):
        msg = (
            f"Baked folder is missing '{_BAKED_CONFIG_FILE}' (or builder info within it): {folder}. "
            "It cannot be reloaded without knowing which pipeline builder to use."
        )
        raise ValueError(msg)

    # Fold a content signature into the cache key so re-baking the same folder invalidates
    # the in-session pipeline cache instead of returning a stale build.
    folder = folder.resolve()
    signature = _baked_content_signature(folder, layout)
    config_hash = "baked-" + hashlib.sha256(f"{folder}::{signature}".encode()).hexdigest()
    return DiffusionPipelineArtifact.from_baked_config_dict(config, baked_path=folder, config_hash=config_hash)


def _baked_content_signature(folder: Path, layout: BakedLayout) -> str:
    """Fingerprint index + sidecar files so map-only edits invalidate the cache."""
    paths = [folder / layout.index_filename()]
    paths.append(folder / _BAKED_CONFIG_FILE)
    paths.extend(sorted(folder.rglob(QUANT_MAP_FILENAME)))
    paths.extend(sorted(folder.rglob("*.gguf")))
    parts: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        stat = path.stat()
        parts.append(f"{path.relative_to(folder).as_posix()}:{stat.st_size}:{stat.st_mtime_ns}")
    return ";".join(parts)
