"""Locate a ``config.json`` for a diffusers component.

Resolution order:

1. **config_source as local path** - if ``config_source`` points to an existing
   ``config.json`` file or directory on disk, use it directly.
2. **config_source as HF repo** - if ``config_source`` looks like an HF repo_id
   (not a path on disk), try the warm HF cache for that repo.
3. **Canonical HF cache** - look up the canonical repo for this ``model_type``
   via ``DIFFUSERS_DEFAULT_PIPELINE_PATHS`` and try the warm HF cache.
4. **Bundled fallback** - shipped copy under
   ``bundled_configs/<model_type>/<subfolder>/config.json``.

The ``subfolder`` is derived from ``SINGLE_FILE_LOADABLE_CLASSES`` so this
function works for any loadable component (transformer, vae, unet, …).
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

_BUNDLED_ROOT = Path(__file__).parent / "bundled_configs"

HF_REPO_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$")


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


def resolve_config_dir(model_type: str, component_cls: type, config_source: str | None) -> Path:
    """Return the directory containing a ``config.json`` for ``component_cls``.

    Parameters
    ----------
    model_type:
        The diffusers ``model_type`` string from ``infer_diffusers_model_type``.
    component_cls:
        The concrete component class (e.g. ``FluxTransformer2DModel``). Used
        to derive the subfolder name from ``SINGLE_FILE_LOADABLE_CLASSES``.
    config_source:
        Either a local path to a ``config.json`` file or its parent directory,
        an HF repo_id whose warm cache to consult, or ``None`` to skip tiers
        1 and 2 and go straight to the canonical HF cache and bundled fallback.

    The returned path is passed to ``from_single_file(config=...)``.
    Logs which tier (USER_PATH / USER_REPO / HF_CACHE / BUNDLED) was used.
    """
    loadable_name = loadable_class_name(component_cls)
    if loadable_name is None:
        msg = (
            f"Attempted to resolve config dir. "
            f"Failed with component_cls='{component_cls.__name__}' because none of its "
            f"base classes are registered in SINGLE_FILE_LOADABLE_CLASSES."
        )
        raise ValueError(msg)

    subfolder = SINGLE_FILE_LOADABLE_CLASSES[loadable_name].get("default_subfolder")
    if subfolder is None:
        msg = (
            f"Attempted to resolve config dir. "
            f"Failed with component_cls='{component_cls.__name__}' "
            f"(loadable_class='{loadable_name}') because it has no 'default_subfolder' "
            f"in SINGLE_FILE_LOADABLE_CLASSES."
        )
        raise ValueError(msg)

    config_filename = f"{subfolder}/config.json"

    # Config_source supplied by caller. An explicit value must resolve.
    if config_source is not None:
        config_path = Path(config_source)
        if config_path.exists():
            if config_path.is_file():
                logger.info("Resolved config: source=USER_PATH path=%s", config_path)
                return config_path.parent
            candidate = config_path / "config.json"
            if candidate.is_file():
                logger.info("Resolved config: source=USER_PATH path=%s", candidate)
                return config_path
            msg = (
                f"Attempted to resolve config dir. "
                f"Failed with config_source='{config_source}' because it is a directory "
                f"that does not contain a 'config.json'."
            )
            raise FileNotFoundError(msg)

        if HF_REPO_ID_PATTERN.match(config_source):
            cached = try_to_load_from_cache(config_source, filename=config_filename)
            if isinstance(cached, str):
                cached_dir = Path(cached).parent
                logger.info("Resolved config: source=USER_REPO repo=%s path=%s", config_source, cached_dir)
                return cached_dir
            msg = (
                f"Attempted to resolve config dir. "
                f"Failed with config_source='{config_source}' because '{config_filename}' "
                f"is not in the local HuggingFace cache for that repo."
            )
            raise FileNotFoundError(msg)

        msg = (
            f"Attempted to resolve config dir. "
            f"Failed with config_source='{config_source}' because it is neither an existing "
            f"local path nor a valid HuggingFace repo id."
        )
        raise ValueError(msg)

    # Canonical HF cache for this model_type.
    paths_entry = DIFFUSERS_DEFAULT_PIPELINE_PATHS.get(model_type)
    if paths_entry is not None:
        repo_id = paths_entry["pretrained_model_name_or_path"]
        cached = try_to_load_from_cache(repo_id, filename=config_filename)
        if isinstance(cached, str):
            cached_dir = Path(cached).parent
            logger.info("Resolved config: source=HF_CACHE repo=%s path=%s", repo_id, cached_dir)
            return cached_dir

    # Bundled config fallback.
    bundled = _BUNDLED_ROOT / model_type / subfolder / "config.json"
    if bundled.is_file():
        logger.info("Resolved config: source=BUNDLED path=%s", bundled)
        return bundled.parent

    canonical_repo = paths_entry["pretrained_model_name_or_path"] if paths_entry else "<no canonical repo>"
    msg = (
        f"Attempted to resolve config dir. "
        f"Failed with model_type='{model_type}' component='{component_cls.__name__}' "
        f"subfolder='{subfolder}' because no config was found in any tier: "
        f"HF_CACHE (repo='{canonical_repo}' file='{config_filename}' not cached), "
        f"BUNDLED (expected at '{bundled}'). "
        f"Set 'Config Source' to a local config directory or HuggingFace repo_id for this component."
    )
    raise FileNotFoundError(msg)
