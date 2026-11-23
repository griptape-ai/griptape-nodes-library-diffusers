"""Explains the config file a single-file component needs, in plain language."""

from __future__ import annotations

from urllib.parse import quote

from modular_diffusion_nodes_library.component_loading.component_slots import (
    component_config_filename,
    slot_component_kind,
)

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
