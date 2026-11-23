from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("accelerate", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

import torch  # noqa: E402
from diffusers.modular_pipelines.components_manager import ComponentsManager  # noqa: E402
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec  # noqa: E402
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration  # noqa: E402

from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import (  # noqa: E402
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
)
from modular_diffusion_nodes_library.parameters.providers import Provider  # noqa: E402
from modular_diffusion_nodes_library.standard_parameters import minimax_h3_parameters  # noqa: E402
from modular_diffusion_nodes_library.standard_parameters.minimax_h3_ref2va_parameters import (  # noqa: E402
    MiniMaxH3Ref2VAPipelineParameters,
)
from modular_diffusion_nodes_library.utils.disk_backed_offload import (  # noqa: E402
    DiskBackedComponent,
    DiskBackedOffloadHook,
    enable_disk_backed_offload,
)


@pytest.fixture(scope="module")
def checkpoint_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A tiny Qwen3-VL saved under `<root>/text_encoder`."""
    root = tmp_path_factory.mktemp("checkpoint")
    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "in_channels": 3,
            "patch_size": 4,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 16,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
        tie_word_embeddings=False,
    )
    Qwen3VLForConditionalGeneration(config).save_pretrained(root / "text_encoder")
    return root


def test_ref2va_pipeline_name_is_distinct_from_the_upstream_class() -> None:
    assert MiniMaxH3Ref2VAPipelineParameters.pipeline_name() == "MiniMaxH3Ref2VAModularPipeline"
    assert MiniMaxH3Ref2VAPipelineParameters.pipeline_cls() is MiniMaxH3ModularPipeline


def test_ref2va_pipeline_type_is_listed_under_minimax_h3_provider() -> None:
    pipeline_types = MODULAR_PIPELINE_TYPE_PROVIDER_MAP[Provider.MINIMAX_H3].get_pipeline_type_dict()

    assert pipeline_types["MiniMaxH3Ref2VAModularPipeline"] is MiniMaxH3Ref2VAPipelineParameters


def test_text_encoder_reloads_after_the_hand_fired_hook_and_submodule_call(checkpoint_root: Path) -> None:
    # MiniMax-H3 fires the hook by hand, then calls `text_encoder.model` rather than the top-level forward.
    directory = checkpoint_root / "text_encoder"

    def load(device: torch.device) -> Qwen3VLForConditionalGeneration:
        return Qwen3VLForConditionalGeneration.from_pretrained(directory, dtype=torch.float32, device_map=str(device))

    text_encoder = load(torch.device("cpu"))
    ids = torch.tensor([[1, 2, 3, 4]])

    def hidden_state() -> torch.Tensor:
        outputs = text_encoder.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=torch.zeros_like(ids),
            use_cache=False,
            output_hidden_states=True,
        )
        return outputs.hidden_states[2]

    with torch.no_grad():
        reference = hidden_state()
    manager = ComponentsManager()
    manager.add("text_encoder", text_encoder)
    component = DiskBackedComponent(load=load, weight_files=tuple(directory.glob("*.safetensors")))
    enable_disk_backed_offload(manager, "cpu", {"text_encoder": component}, offload_strategy=lambda **_: [])

    cast(Any, text_encoder)._hf_hook.pre_forward(text_encoder)
    with torch.no_grad():
        torch.testing.assert_close(hidden_state(), reference)


def test_builder_registers_an_empty_text_encoder_before_loading_the_rest(
    checkpoint_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(minimax_h3_parameters, "get_best_device", lambda: "cpu")
    snapshot_download_calls: list[tuple[str, dict[str, Any]]] = []

    def fake_snapshot_download(repo_id: str, **kwargs: Any) -> str:
        snapshot_download_calls.append((repo_id, kwargs))
        return "/cache/snapshot"

    monkeypatch.setattr(minimax_h3_parameters, "snapshot_download", fake_snapshot_download)
    manager = ComponentsManager()
    spec = ComponentSpec(
        name="text_encoder",
        type_hint=Qwen3VLForConditionalGeneration,
        pretrained_model_name_or_path=str(checkpoint_root),
        subfolder="text_encoder",
        default_creation_method="from_pretrained",
    )

    class _FakePipe:
        text_encoder: Qwen3VLForConditionalGeneration
        meta_when_rest_loaded: bool

        def get_component_spec(self, name: str) -> ComponentSpec:
            return spec

        def register_components(self, text_encoder: Qwen3VLForConditionalGeneration) -> None:
            self.text_encoder = text_encoder
            manager.add("text_encoder", text_encoder)

        def load_components(self, workflow: str, **kwargs: Any) -> None:
            assert kwargs["local_files_only"] is True
            assert kwargs["pretrained_model_name_or_path"] == {"processor": "/cache/snapshot"}
            self.meta_when_rest_loaded = all(p.device.type == "meta" for p in self.text_encoder.parameters())

    pipe = _FakePipe()
    minimax_h3_parameters.MiniMaxH3PipelineParameters._load_components_with_disk_backed_offload(
        cast(Any, pipe), manager, "MiniMaxAI/MiniMax-H3", "main"
    )

    assert snapshot_download_calls == [
        ("MiniMaxAI/MiniMax-H3", {"revision": "main", "local_files_only": True, "allow_patterns": "processor/*"})
    ]
    assert pipe.meta_when_rest_loaded
    assert isinstance(cast(Any, pipe.text_encoder)._hf_hook, DiskBackedOffloadHook)
    assert pipe.text_encoder.lm_head.weight.dtype == torch.bfloat16
