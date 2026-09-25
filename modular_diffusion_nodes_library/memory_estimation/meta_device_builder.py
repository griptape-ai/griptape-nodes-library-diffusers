"""Build a diffusers/transformers component on the meta device from its resolved
config, touching no weight data at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch  # type: ignore[reportMissingImports]
from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
from huggingface_hub import try_to_load_from_cache
from transformers import PreTrainedModel  # type: ignore[reportMissingImports]
from transformers.models.auto.configuration_auto import AutoConfig  # type: ignore[reportMissingImports]
from transformers.models.auto.modeling_auto import AutoModel  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.component_loading.component_slots import component_config_filename
from modular_diffusion_nodes_library.component_loading.config_resolver import (
    resolve_hf_repo_config_subfolder,
    try_load_json_dict,
)


class ComponentConfigNotCachedError(RuntimeError):
    """Raised when a component's config file is not in the warm HuggingFace cache."""


def resolve_base_component_config(
    repo_id: str,
    component: str,
    revision: str | None = None,
) -> dict[str, Any]:
    """Read a base-pipeline component's config.json from the warm HF cache.

    Never triggers a download -- raises ComponentConfigNotCachedError if the config
    isn't already cached locally.
    """
    config_filename = component_config_filename(component)
    subfolder = resolve_hf_repo_config_subfolder(
        repo_id,
        component,
        component,
        revision=revision,
        config_filename=config_filename,
    )
    if subfolder is None:
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed because no cached '{config_filename}' was found under repo '{repo_id}' (revision='{revision}')."
        )
        raise ComponentConfigNotCachedError(msg)

    filename = f"{subfolder}/{config_filename}" if subfolder else config_filename
    cached = try_to_load_from_cache(repo_id, filename=filename, revision=revision)
    if not isinstance(cached, str):
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed because '{filename}' is not in the local HuggingFace cache for repo '{repo_id}'."
        )
        raise ComponentConfigNotCachedError(msg)

    config_dict = try_load_json_dict(Path(cached))
    if config_dict is None:
        msg = (
            f"Attempted to resolve config for component '{component}'. "
            f"Failed to parse cached config file '{cached}' as JSON."
        )
        raise ComponentConfigNotCachedError(msg)
    return config_dict


def build_component_on_meta_device(
    component_cls: type,
    config_dict: dict[str, Any],
    torch_dtype: torch.dtype,
    *,
    cast_dtype: bool = True,
) -> Any:
    """Construct component_cls on the meta device from config_dict.

    No weight data is ever read or allocated -- shapes/dtype only, via
    accelerate.init_empty_weights(). Dispatches on whether component_cls is a
    transformers PreTrainedModel (constructed from a `config_class` instance), a
    generic PreTrainedModel annotation (resolved through AutoModel), or a diffusers
    ConfigMixin (whose `from_config` accepts a raw dict directly).
    """
    config_class = getattr(component_cls, "config_class", None)
    with init_empty_weights():
        if config_class is not None and component_cls is not PreTrainedModel:
            config_obj = config_class.from_dict(config_dict)
            component = component_cls(config_obj)
        elif issubclass(component_cls, PreTrainedModel):
            model_type = config_dict.get("model_type")
            if not isinstance(model_type, str):
                msg = (
                    "Attempted to resolve a generic transformer component. Failed because its config has no model_type."
                )
                raise ValueError(msg)
            model_kwargs = {key: value for key, value in config_dict.items() if key != "model_type"}
            model_config = AutoConfig.for_model(model_type, **model_kwargs)
            component = AutoModel.from_config(model_config)
        else:
            component = component_cls.from_config(config_dict)
    if cast_dtype:
        # This is a metadata-only component on the meta device. Call the base
        # implementation so Diffusers' runtime `.to(dtype=...)` warning is not
        # emitted for a cast that never touches model weights.
        return torch.nn.Module.to(component, dtype=torch_dtype)
    return component
