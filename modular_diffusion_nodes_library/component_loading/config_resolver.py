"""Locate a config file for a diffusers component.

Resolution order:

1. **config_source as local path** - if ``config_source`` points to an existing
   config JSON file (any filename) or directory containing a ``config.json`` on
   disk, use it directly.
2. **config_source as HF repo** - if ``config_source`` looks like an HF repo_id
   (not a path on disk), try the warm HF cache for that repo.
3. **Canonical HF cache** - look up the canonical repo for this ``model_type``
   via ``DIFFUSERS_DEFAULT_PIPELINE_PATHS`` and try the warm HF cache.

The subfolder is derived from ``pipeline_slot`` / ``artifact_component`` candidates,
so this function works for any component regardless of ``SINGLE_FILE_LOADABLE_CLASSES`` registration.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import diffusers  # type: ignore[reportMissingImports]
from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES  # type: ignore[reportMissingImports]
from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS  # type: ignore[reportMissingImports]
from huggingface_hub import try_to_load_from_cache

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
    filename = f"{subfolder}/{config_filename}" if subfolder else config_filename
    return isinstance(try_to_load_from_cache(repo_id, filename=filename, revision=revision), str)


def loadable_class_name(component_cls: type) -> str | None:
    """Return the ``SINGLE_FILE_LOADABLE_CLASSES`` key that ``component_cls``
    is a subclass of, or ``None`` if none matches. A subclass of a registered
    loadable (e.g. a project-local subclass of ``FluxTransformer2DModel``)
    resolves to its registered ancestor.
    """
    for name in SINGLE_FILE_LOADABLE_CLASSES:
        loadable_cls = getattr(diffusers, name, None)
        if loadable_cls is not None and issubclass(component_cls, loadable_cls):
            return name
    return None


def resolve_config_path(
    model_type: str,
    component_cls: type,
    config_source: str | None,
    *,
    pipeline_slot: str = "",
    artifact_component: str = "",
) -> Path:
    """Return a config path suitable for ``from_single_file(config=...)``.

    Parameters
    ----------
    model_type:
        The diffusers ``model_type`` string from ``infer_diffusers_model_type``.
    component_cls:
        The concrete component class (e.g. ``FluxTransformer2DModel``).
    config_source:
        A local path to a JSON file or directory, an HF repo_id, or ``None``
        to go straight to the canonical HF cache.
    pipeline_slot:
        The slot name in the target pipeline (e.g. ``"vae"``, ``"text_encoder_2"``).
        Used as the first subfolder candidate when probing HF repos.
    artifact_component:
        The component name the artifact declared itself as (e.g. ``"text_encoder"``
        when ``pipeline_slot`` is ``"text_encoder_2"``). Used as a secondary
        subfolder candidate.
    """

    config_filename = ""

    # Config_source supplied by caller. An explicit value must resolve.
    if config_source is not None:
        config_path = Path(config_source)
        if config_path.exists():
            if config_path.is_file():
                logger.info("Resolved config: source=USER_PATH path=%s", config_path)
                return config_path
            candidate = config_path / "config.json"
            if candidate.is_file():
                logger.info("Resolved config: source=USER_PATH path=%s", candidate)
                return config_path
            msg = (
                f"Attempted to resolve config path. "
                f"Failed with config_source='{config_source}' because it is a directory "
                f"that does not contain a 'config.json'."
            )
            raise FileNotFoundError(msg)

        if HF_REPO_ID_PATTERN.match(config_source):
            subfolder = resolve_hf_repo_config_subfolder(config_source, pipeline_slot, artifact_component)
            config_filename = f"{subfolder}/config.json" if subfolder else "config.json"
            cached = try_to_load_from_cache(config_source, filename=config_filename)
            if isinstance(cached, str):
                cached_dir = Path(cached).parent
                logger.info("Resolved config: source=USER_REPO repo=%s path=%s", config_source, cached_dir)
                return cached_dir
            msg = (
                f"Attempted to resolve config path. "
                f"Failed with config_source='{config_source}' because '{config_filename}' "
                f"is not in the local HuggingFace cache for that repo."
            )
            raise FileNotFoundError(msg)

        msg = (
            f"Attempted to resolve config path. "
            f"Failed with config_source='{config_source}' because it is neither an existing "
            f"local path nor a valid HuggingFace repo id."
        )
        raise ValueError(msg)

    # Canonical HF cache for this model_type.
    paths_entry = DIFFUSERS_DEFAULT_PIPELINE_PATHS.get(model_type)
    if paths_entry is not None:
        repo_id = paths_entry["pretrained_model_name_or_path"]
        subfolder = resolve_hf_repo_config_subfolder(repo_id, pipeline_slot, artifact_component)
        config_filename = f"{subfolder}/config.json" if subfolder else "config.json"
        cached = try_to_load_from_cache(repo_id, filename=config_filename)
        if isinstance(cached, str):
            cached_dir = Path(cached).parent
            logger.info("Resolved config: source=HF_CACHE repo=%s path=%s", repo_id, cached_dir)
            return cached_dir

    canonical_repo = paths_entry["pretrained_model_name_or_path"] if paths_entry else "<no canonical repo>"
    msg = (
        f"Attempted to resolve config path. "
        f"Failed with model_type='{model_type}' component='{component_cls.__name__}' "
        f"because no config was found: "
        f"HF_CACHE (repo='{canonical_repo}' file='{config_filename}' not cached). "
        f"Set 'Config Source' to a local config directory or HuggingFace repo_id for this component."
    )
    raise FileNotFoundError(msg)
