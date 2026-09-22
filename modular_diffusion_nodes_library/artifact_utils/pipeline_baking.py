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

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]

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


def detect_baked_layout(folder: Path) -> BakedLayout | None:
    """Return the repo layout of `folder`, or None if it isn't a recognizable Diffusers/modular repo."""
    if (folder / _MODULAR_MODEL_INDEX).is_file():
        return BakedLayout.MODULAR
    if (folder / _MODEL_INDEX).is_file():
        return BakedLayout.STANDARD
    return None


def save_baked_pipeline(pipe: Any, artifact: DiffusionPipelineArtifact, folder: Path) -> BakedLayout:
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
        pipe.save_pretrained(str(tmp_dir))
        if isinstance(pipe, ModularPipeline):
            layout = BakedLayout.MODULAR
        else:
            layout = BakedLayout.STANDARD
        dtype_name = str(getattr(pipe, "dtype", None) or f"torch.{DEFAULT_BAKED_DTYPE}").replace("torch.", "")
        _write_baked_config(tmp_dir, artifact, dtype_name)

        if folder.exists():
            shutil.rmtree(folder)
        os.replace(tmp_dir, folder)
        return layout
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def _write_baked_config(folder: Path, artifact: DiffusionPipelineArtifact, dtype_name: str) -> None:
    """Always write a config, wrapping the artifact's own fields in folder-level metadata."""
    config = {
        "schema_version": _BAKED_CONFIG_SCHEMA_VERSION,
        "torch_dtype": dtype_name,
        **artifact.to_baked_config_dict(),
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
    """Cheap content fingerprint from the repo index file's size + mtime."""
    if layout is BakedLayout.MODULAR:
        index_path = folder / _MODULAR_MODEL_INDEX
    else:
        index_path = folder / _MODEL_INDEX
    stat = index_path.stat()
    return f"{stat.st_size}-{stat.st_mtime_ns}"
