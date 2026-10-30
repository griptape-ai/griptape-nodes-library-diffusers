"""Load Pipeline quantization status: one line when types match, per slot when they differ."""

from __future__ import annotations

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_baking import _write_baked_config, make_baked_artifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_recipe import (
    QuantizationStatus,
    collect_quantization_status,
    format_baked_pipeline_summary,
    format_pipeline_artifact_summary,
    format_quantization_status_lines,
)


def _gguf_override(slot: str = "transformer") -> ModelComponentArtifact:
    return ModelComponentArtifact(
        source_type=ComponentSourceType.SINGLE_FILE,
        component=slot,
        file_path=f"/tmp/{slot}.gguf",
    )


def test_format_quantization_one_line_when_types_match() -> None:
    status = QuantizationStatus(slots={"unet": "int8", "vae": "int8", "text_encoder": "int8"})
    assert format_quantization_status_lines(status) == ["Quantization: int8 (text_encoder, unet, vae)"]


def test_format_quantization_per_slot_when_types_differ() -> None:
    status = QuantizationStatus(slots={"transformer": "GGUF", "unet": "int8"})
    assert format_quantization_status_lines(status) == [
        "Quantization",
        "  transformer: GGUF",
        "  unet: int8",
    ]


def test_format_quantization_pipeline_only() -> None:
    status = QuantizationStatus(slots={}, pipeline="int8")
    assert format_quantization_status_lines(status) == ["Quantization: int8"]


def test_format_quantization_empty() -> None:
    assert format_quantization_status_lines(QuantizationStatus()) == []


def test_collect_baked_quanto_slots() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionXLPipeline",
        build_data={
            "_baked_path": "/tmp/baked",
            "_baked_dtype": "bfloat16",
            "_baked_quantization_mode": "int8",
            "_packed_quant_slots": ["unet", "vae"],
        },
    )
    assert collect_quantization_status(artifact) == QuantizationStatus(slots={"unet": "int8", "vae": "int8"})


def test_collect_gguf_override() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        is_prequantized=True,
        build_data={"_component_overrides": {"transformer": _gguf_override()}},
        optimization_kwargs={"quantization_mode": "int8"},
    )
    assert collect_quantization_status(artifact) == QuantizationStatus(slots={"transformer": "GGUF"})


def test_collect_bnb_hub_repo_has_no_quant_label() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="Flux2Pipeline",
        is_prequantized=True,
        build_data={"base_repo_id": "diffusers/FLUX.2-dev-bnb-4bit"},
    )
    assert collect_quantization_status(artifact) == QuantizationStatus()


def test_collect_minimax_prequantized_flag_has_no_hub_label() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="MiniMaxH3ModularPipeline",
        is_prequantized=True,
        build_data={"base_repo_id": "MiniMaxAI/MiniMax-H3"},
    )
    assert collect_quantization_status(artifact) == QuantizationStatus()


def test_collect_recipe_quanto_mode() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionXLPipeline",
        optimization_kwargs={"quantization_mode": "int8"},
    )
    assert collect_quantization_status(artifact) == QuantizationStatus(pipeline="int8")


def test_collect_mixed_gguf_and_packed_slots() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        is_prequantized=True,
        build_data={
            "_component_overrides": {"transformer": _gguf_override()},
            "_baked_quantization_mode": "int8",
            "_packed_quant_slots": ["vae"],
        },
    )
    assert collect_quantization_status(artifact) == QuantizationStatus(slots={"transformer": "GGUF", "vae": "int8"})


def test_baked_artifact_exposes_quant_fields_for_status(tmp_path) -> None:
    (tmp_path / "model_index.json").write_text('{"_class_name": "StableDiffusionXLPipeline"}\n', encoding="utf-8")
    artifact = DiffusionPipelineArtifact(
        pipeline_name="StableDiffusionXLPipeline",
        builder_module="mod",
        builder_class_name="Cls",
        optimization_kwargs={"quantization_mode": "int8"},
    )
    _write_baked_config(tmp_path, artifact, "bfloat16", packed_quant_slots=["unet", "vae"])
    loaded = make_baked_artifact(tmp_path)
    assert loaded.build_data["_baked_quantization_mode"] == "int8"
    assert loaded.build_data["_packed_quant_slots"] == ["unet", "vae"]
    summary = format_baked_pipeline_summary(loaded)
    assert "Dtype: bfloat16" in summary
    assert "Quantization: int8 (unet, vae)" in summary


def test_recipe_summary_includes_quantization() -> None:
    artifact = DiffusionPipelineArtifact(
        pipeline_name="FluxPipeline",
        is_prequantized=True,
        build_data={
            "repo_id": "black-forest-labs/FLUX.1-dev",
            "_component_overrides": {"transformer": _gguf_override()},
        },
    )
    summary = format_pipeline_artifact_summary(artifact)
    assert "Quantization: GGUF (transformer)" in summary
