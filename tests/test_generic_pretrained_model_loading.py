from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline  # noqa: E402
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline  # noqa: E402
from transformers import AutoModel, CLIPTextModel, Qwen3Config, Qwen3Model  # noqa: E402

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType  # noqa: E402
from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (  # noqa: E402
    TextEncoderComponentArtifact,
)
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import (  # noqa: E402
    get_loadable_component_class,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_generic_pretrained_model_slot_resolves_to_auto_model() -> None:
    assert get_loadable_component_class(ZImagePipeline, "text_encoder") is AutoModel


def test_concrete_slot_keeps_its_annotated_class() -> None:
    assert get_loadable_component_class(FluxPipeline, "text_encoder") is CLIPTextModel


def test_z_image_text_encoder_loads_from_local_dir(tmp_path: Path) -> None:
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
    )
    Qwen3Model(config).save_pretrained(tmp_path)
    artifact = TextEncoderComponentArtifact(
        load_id="z-image-text-encoder",
        source_type=ComponentSourceType.LOCAL_DIR,
        component="text_encoder",
        file_path=str(tmp_path),
        torch_dtype="float32",
    )

    text_encoder = artifact.materialize(pipeline_cls=ZImagePipeline)

    assert isinstance(text_encoder, Qwen3Model)
