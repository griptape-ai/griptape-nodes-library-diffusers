from __future__ import annotations

import pytest

from modular_diffusion_nodes_library.component_loading.component_config_badge import component_config_badge_message


def test_transformer_port_links_the_selected_model_config_page() -> None:
    message = component_config_badge_message("transformer", "Qwen/Qwen-Image")

    assert message is not None
    assert "**What is a config file?**" in message
    assert "[`config.json`](https://huggingface.co/Qwen/Qwen-Image/blob/main/transformer/config.json)" in message
    assert "in **Qwen/Qwen-Image**" in message
    assert "click **Download**" in message
    assert "the file page; it does not download the model" in message
    assert "gated" in message
    assert "log in to Hugging Face" in message
    assert "resolve/" not in message
    assert "settings" not in message.lower()


def test_custom_base_revision_is_used_in_config_page_link() -> None:
    message = component_config_badge_message("transformer", "Qwen/Qwen-Image", "refs/pr/17")

    assert message is not None
    assert "https://huggingface.co/Qwen/Qwen-Image/blob/refs%2Fpr%2F17/transformer/config.json" in message


def test_port_without_a_base_repo_explains_how_to_find_the_config_file() -> None:
    message = component_config_badge_message("vae", None)

    assert message is not None
    assert "**How to get one?**" in message
    assert "click **Files and versions**" in message
    assert "`config.json` in the `vae` folder" in message
    assert "gated" in message
    assert "[Open" not in message


@pytest.mark.parametrize("slot", ["text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"])
def test_slots_with_no_config_file_to_choose_get_no_badge(slot: str) -> None:
    assert component_config_badge_message(slot, "org/base") is None
