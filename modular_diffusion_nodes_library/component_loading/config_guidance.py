"""Explains the config file a single-file component needs, in plain language."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from modular_diffusion_nodes_library.component_loading.component_slots import (
    component_config_filename,
    slot_component_kind,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.component_loading.config_resolver import ResolvedConfig

#: Kinds with no config file for the user to choose: a tokenizer cannot load from a single file, a scheduler is only a
#: config, and a GGUF text encoder carries its own (transformers builds the config from the file).
_KINDS_WITHOUT_BADGE = frozenset({"tokenizer", "scheduler", "text_encoder"})


def component_config_badge_message(slot: str, base_repo_id: str | None, base_revision: str | None = None) -> str | None:
    """Explain which config file a single-file component needs to load.

    Returns None for a slot that gets no badge.
    """
    kind = slot_component_kind(slot)
    if kind in _KINDS_WITHOUT_BADGE:
        return None

    config_name = component_config_filename(slot)
    what_it_is = (
        "**What is a config file?**\n"
        f"A small file, `{config_name}`, that tells the node how to read your single file. "
        "It must come from the same model as your file. It is not the model itself.\n\n"
    )
    gated_note = "If the model is gated, log in to Hugging Face and accept its license on the model page first.\n\n"
    leave_empty = "Leave **Config (Optional)** empty only if you have downloaded the full model before."

    if base_repo_id is None:
        return (
            f"{what_it_is}"
            "**How to get one?**\n"
            f"1. Open the model page you got your single file from, click **Files and versions**, and open "
            f"`{config_name}` in the `{slot}` folder.\n"
            "2. Click **Download**, then in **Load Component** choose it under **Config (Optional)**.\n\n"
            f"{gated_note}"
            f"{leave_empty}"
        )

    revision = quote(base_revision or "main", safe="")
    config_path = quote(f"{slot}/{config_name}", safe="/")
    config_url = f"https://huggingface.co/{quote(base_repo_id, safe='/')}/blob/{revision}/{config_path}"
    return (
        f"{what_it_is}"
        "**How to get it?**\n"
        f"Open [`{config_name}`]({config_url}) in **{base_repo_id}**. This link opens "
        "the file page; it does not download the model.\n"
        "On that page, click **Download** to save only the config file. Then, in **Load Component**, choose the "
        "downloaded file under **Config**.\n\n"
        f"{gated_note}"
        "If your file came from another model, open that model's matching config page instead.\n\n"
        f"{leave_empty}"
    )


def short_path(path: str | Path) -> str:
    """Label a path by its folder and name (``ltx23/model.gguf``). The full path goes in the log."""
    parts = Path(path).parts
    return "/".join(parts[-2:])


def config_setting_hint(*, slot: str, example_repos: list[str]) -> str:
    """How to point a single file at its model's config, as a clause starting "set"."""
    repos = ", ".join(f"'{repo}'" for repo in example_repos) or "'owner/model'"
    config_name = component_config_filename(slot)
    return (
        f"set 'Config' on the Load Component node to the repo id of the model the file was made from, "
        f"using the same version as the file (e.g. {repos}; its '{slot}/{config_name}' is then used). "
        f"Or download just that {config_name} from the repo's Files tab on huggingface.co and set 'Config' "
        f"to the path you saved it at."
    )


def missing_config_message(*, pipeline_slot: str, tried_repos: list[str], example_repos: list[str]) -> str:
    """Tell the user that a single file has no config, and how to supply one."""
    if tried_repos:
        searched = (
            f"Checked the downloaded models for: {', '.join(tried_repos)}. "
            f"Download '{tried_repos[0]}' in Model Management."
        )
    else:
        searched = "No matching model is known."
    hint = config_setting_hint(slot=pipeline_slot, example_repos=example_repos)
    return (
        f"No config file was found for the {pipeline_slot}. A single file like a .gguf only holds the model's "
        f"weights, so it needs the config of the model it was made from. {searched} Or {hint}"
    )


def config_mismatch_message(
    *,
    component: str,
    file_path: str,
    slot: str,
    config: ResolvedConfig,
    suggested_repos: list[str],
    details: str,
    likely_cause_only: bool,
) -> str:
    """Tell the user that a single file did not load with the config it was given, and what to do.

    ``likely_cause_only`` says the config is a guess at the cause, not a certainty, so the message says so.
    """
    hint = config_setting_hint(slot=slot, example_repos=suggested_repos)
    config_label = config.source
    if config.repo_id is None:
        config_label = f"{config.source} ({short_path(config.path)})"
    caveat = ""
    if likely_cause_only:
        caveat = "The config is the most likely cause; the log has the full traceback. "
    return (
        f"Could not load the {component} from '{short_path(file_path)}'.\n"
        f"**Why:** diffusers could not build it from the file with the config used, which came from {config_label}. "
        f"The file and the config are probably from different versions of the model. "
        f"{caveat}"
        f"To fix it, {hint} \n**Additional details:** {details}"
    )
