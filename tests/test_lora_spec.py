from __future__ import annotations

from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec, normalize_loras


def test_normalize_loras_sums_duplicate_path_weights() -> None:
    normalized = normalize_loras(
        [
            {"path": "/models/style.safetensors", "weight": 0.25},
            {"path": "/models/detail.safetensors", "weight": 0.75},
            {"path": "/models/style.safetensors", "weight": 1.0},
        ]
    )

    assert normalized == {
        "/models/style.safetensors": LoraSpec(path="/models/style.safetensors", weight=1.25),
        "/models/detail.safetensors": LoraSpec(path="/models/detail.safetensors", weight=0.75),
    }


def test_normalize_loras_preserves_duplicate_trigger_phrase() -> None:
    normalized = normalize_loras(
        [
            {"path": "/models/style.safetensors", "weight": 0.25, "trigger_phrase": "style"},
            {"path": "/models/style.safetensors", "weight": 0.75},
        ]
    )

    assert normalized["/models/style.safetensors"] == LoraSpec(
        path="/models/style.safetensors",
        weight=1.0,
        trigger_phrase="style",
    )
