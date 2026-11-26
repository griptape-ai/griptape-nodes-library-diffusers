"""Locate a config file for a diffusers component.

Resolution order:

1. **config_source as local path** - if ``config_source`` points to an existing
   config JSON file (any filename) or directory containing a ``config.json`` on
   disk, use it directly.
2. **config_source as HF repo** - if ``config_source`` looks like an HF repo_id
   (not a path on disk), try the warm HF cache for that repo.
3. **Base repo** - the repo the pipeline is built from, if cached.
4. **Canonical repo** - the default repo for ``model_type`` (``DIFFUSERS_DEFAULT_PIPELINE_PATHS``), if cached.

The subfolder is derived from ``pipeline_slot`` / ``artifact_component`` candidates,
so this function works for any component regardless of ``SINGLE_FILE_LOADABLE_CLASSES`` registration.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modular_diffusion_nodes_library.component_loading.component_slots import component_config_filename
from modular_diffusion_nodes_library.component_loading.config_guidance import missing_config_message

logger = logging.getLogger("modular_diffusers_nodes_library")

HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


def _subfolder_candidates(pipeline_slot: str, artifact_component: str) -> list[str]:
    """Return subfolder candidates to probe, in preference order.

    1. ``pipeline_slot`` (e.g. ``"vae"``, ``"text_encoder_2"``)
    2. ``artifact_component`` when it differs (e.g. ``"text_encoder"``)
    3. ``""`` — config.json at the repo root.
    """
    candidates = [pipeline_slot]
    if artifact_component != pipeline_slot:
        candidates.append(artifact_component)
    candidates.append("")
    return candidates


def resolve_hf_repo_config_subfolder(
    repo_id: str,
    pipeline_slot: str,
    artifact_component: str,
    revision: str | None = None,
    *,
    config_filename: str = "config.json",
) -> str | None:
    """Return the subfolder within ``repo_id`` whose ``config_filename`` is in the
    warm HuggingFace cache, or ``None`` if no candidate is cached.

    Returns ``""`` when the file is at the repo root.
    Candidates are tried in the order produced by ``_subfolder_candidates``.
    """
    from huggingface_hub import try_to_load_from_cache

    for subfolder in _subfolder_candidates(pipeline_slot, artifact_component):
        filename = config_filename if not subfolder else f"{subfolder}/{config_filename}"
        if isinstance(try_to_load_from_cache(repo_id, filename=filename, revision=revision), str):
            return subfolder
    return None


def is_hf_config_cached(
    repo_id: str,
    subfolder: str,
    config_filename: str,
    revision: str | None = None,
) -> bool:
    """Return True when ``config_filename`` is present in the warm HF cache at ``subfolder``."""
    from huggingface_hub import try_to_load_from_cache

    filename = f"{subfolder}/{config_filename}" if subfolder else config_filename
    return isinstance(try_to_load_from_cache(repo_id, filename=filename, revision=revision), str)


def try_load_json_dict(path: Path) -> dict[str, Any] | None:
    """Load and return a JSON file as a dict, or ``None`` on any failure."""
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(data, dict):
        return data
    return None


def loadable_class_name(component_cls: type) -> str | None:
    """Return the ``SINGLE_FILE_LOADABLE_CLASSES`` key that ``component_cls``
    is a subclass of, or ``None`` if none matches. A subclass of a registered
    loadable (e.g. a project-local subclass of ``FluxTransformer2DModel``)
    resolves to its registered ancestor.
    """
    import diffusers  # type: ignore[reportMissingImports]
    from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES  # type: ignore[reportMissingImports]

    for name in SINGLE_FILE_LOADABLE_CLASSES:
        loadable_cls = getattr(diffusers, name, None)
        if loadable_cls is not None and issubclass(component_cls, loadable_cls):
            return name
    return None


class ConfigLoadError(ValueError):
    """A single file failed to load for a reason the user can fix. The message already says why and how."""


@dataclass(frozen=True)
class ResolvedConfig:
    """A config found for a single file, and where it came from."""

    path: Path
    source: str
    repo_id: str | None = None


def _cached_config_dir(repo_id: str, pipeline_slot: str, artifact_component: str, revision: str | None) -> Path | None:
    """Return the folder holding this component's downloaded config in ``repo_id``, if any."""
    from huggingface_hub import try_to_load_from_cache

    config_name = component_config_filename(pipeline_slot)
    subfolder = resolve_hf_repo_config_subfolder(
        repo_id, pipeline_slot, artifact_component, revision=revision, config_filename=config_name
    )
    if subfolder is None:
        return None

    filename = config_name
    if subfolder:
        filename = f"{subfolder}/{config_name}"
    cached = try_to_load_from_cache(repo_id, filename=filename, revision=revision)
    if not isinstance(cached, str):
        return None
    return Path(cached).parent


def read_cached_config(
    repo_id: str, pipeline_slot: str, artifact_component: str, revision: str | None
) -> dict[str, Any] | None:
    """Return the config dict a repo's warm cache holds for this component, or ``None``. Never downloads.

    The one reader of a repo's cached config. Every caller that needs a single file's base config goes through it,
    so they all look in the same places in the same order.
    """
    cached_dir = _cached_config_dir(repo_id, pipeline_slot, artifact_component, revision)
    if cached_dir is None:
        return None
    return try_load_json_dict(cached_dir / component_config_filename(pipeline_slot))


def resolve_config_path(
    model_type: str,
    config_source: str | None,
    *,
    pipeline_slot: str = "",
    artifact_component: str = "",
    base_repo_id: str | None = None,
    base_revision: str | None = None,
) -> ResolvedConfig:
    """Return a config suitable for ``from_single_file(config=...)``, with where it came from.

    Parameters
    ----------
    model_type:
        The diffusers ``model_type`` string from ``infer_diffusers_model_type``.
    config_source:
        A local path to a JSON file or directory, an HF repo_id, or ``None``
        to use the base repo, then the canonical repo.
    pipeline_slot:
        The slot name in the target pipeline (e.g. ``"vae"``, ``"text_encoder_2"``).
        Used as the first subfolder candidate when probing HF repos.
    artifact_component:
        The component name the artifact declared itself as (e.g. ``"text_encoder"``
        when ``pipeline_slot`` is ``"text_encoder_2"``). Used as a secondary
        subfolder candidate.
    base_repo_id, base_revision:
        The repo the pipeline is built from. Tried before the canonical repo, because it is the model the
        user picked and the canonical repo may be a different release of the family.
    """
    from diffusers.loaders.single_file_utils import (
        DIFFUSERS_DEFAULT_PIPELINE_PATHS,  # type: ignore[reportMissingImports]
    )

    # Config_source supplied by caller. An explicit value must resolve.
    if config_source is not None:
        config_path = Path(config_source)
        if config_path.exists():
            if config_path.is_file() or (config_path / "config.json").is_file():
                logger.info("Resolved config: source=USER_PATH path=%s", config_path)
                return ResolvedConfig(path=config_path, source="the 'Config' you set")
            msg = (
                f"Attempted to resolve config path. "
                f"Failed with config_source='{config_source}' because it is a directory "
                f"that does not contain a 'config.json'."
            )
            raise ConfigLoadError(msg)

        if HF_REPO_ID_PATTERN.match(config_source):
            cached_dir = _cached_config_dir(config_source, pipeline_slot, artifact_component, revision=None)
            if cached_dir is not None:
                logger.info("Resolved config: source=USER_REPO repo=%s path=%s", config_source, cached_dir)
                return ResolvedConfig(
                    path=cached_dir, source=f"the 'Config' you set ('{config_source}')", repo_id=config_source
                )
            msg = (
                f"Attempted to resolve config path. "
                f"Failed with config_source='{config_source}' because no config for '{pipeline_slot}' "
                f"is in the local HuggingFace cache for that repo."
            )
            raise ConfigLoadError(msg)

        msg = (
            f"Attempted to resolve config path. "
            f"Failed with config_source='{config_source}' because it is neither an existing "
            f"local path nor a valid HuggingFace repo id."
        )
        raise ConfigLoadError(msg)

    candidate_repos: list[tuple[str, str | None, str]] = []
    if base_repo_id is not None and HF_REPO_ID_PATTERN.match(base_repo_id):
        candidate_repos.append((base_repo_id, base_revision, f"the Build Pipeline's model '{base_repo_id}'"))
    paths_entry = DIFFUSERS_DEFAULT_PIPELINE_PATHS.get(model_type)
    if paths_entry is not None and paths_entry["pretrained_model_name_or_path"] != base_repo_id:
        default_repo_id = paths_entry["pretrained_model_name_or_path"]
        candidate_repos.append((default_repo_id, None, f"the default repo '{default_repo_id}' for '{model_type}'"))

    for repo_id, revision, source in candidate_repos:
        cached_dir = _cached_config_dir(repo_id, pipeline_slot, artifact_component, revision)
        if cached_dir is not None:
            logger.info("Resolved config: repo=%s path=%s", repo_id, cached_dir)
            return ResolvedConfig(path=cached_dir, source=source, repo_id=repo_id)

    tried_repos = [repo for repo, _, _ in candidate_repos]
    msg = missing_config_message(pipeline_slot=pipeline_slot, tried_repos=tried_repos, example_repos=tried_repos)
    raise ConfigLoadError(msg)
