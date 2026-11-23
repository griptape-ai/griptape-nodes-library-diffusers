"""Which config a GGUF text encoder override is validated and loaded with."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


@pytest.mark.parametrize("config_source", [None, "local", "hf"], ids=["implicit", "explicit-local", "explicit-repo"])
def test_text_encoder_gguf_validation_uses_the_adjacent_config_used_by_loading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, config_source: str | None
) -> None:
    pytest.importorskip("torch", reason="Loads through the real text-encoder path; run `make test/exec`.")
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.artifact_utils import text_encoder_component_artifact as text_artifacts
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    adjacent = {"text_config": {"hidden_size": 4096, "num_hidden_layers": 24}}
    other = {"text_config": {"hidden_size": 5120, "num_hidden_layers": 36}}
    (tmp_path / "config.json").write_text(json.dumps(adjacent))
    checkpoint = tmp_path / "encoder.gguf"
    checkpoint.touch()
    if config_source == "local":
        explicit = tmp_path / "explicit.json"
        explicit.write_text(json.dumps(other))
        config_source = str(explicit)
    elif config_source == "hf":
        config_source = "org/other-encoder"

    class Encoder:
        @classmethod
        def from_pretrained(cls, source: str, **kwargs: Any) -> dict[str, Any]:
            assert kwargs["gguf_file"] == checkpoint.name
            assert kwargs["local_files_only"] is True
            return json.loads((Path(source) / "config.json").read_text())

    monkeypatch.setattr(artifacts, "get_loadable_component_class", lambda *args: Encoder)
    monkeypatch.setattr(text_artifacts, "get_loadable_component_class", lambda *args: Encoder)
    artifact = text_artifacts.TextEncoderComponentArtifact(
        load_id="test-adjacent-encoder",
        source_type=artifacts.ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        file_path=str(checkpoint),
        config_source=config_source,
    )
    resolver = _ConfigResolver({"text_encoder": artifact}, "org/base")

    assert resolver.resolved_config("text_encoder") == adjacent
    assert artifact.materialize(pipeline_cls=Encoder, slot="text_encoder") == adjacent


@pytest.mark.parametrize(
    ("hidden_size", "num_hidden_layers", "target_dim", "expected_types"),
    [
        (1280, 36, 20480, ["TextEncoderDenoiserDimMismatch"]),
        (2560, 24, 20480, ["TextEncoderHiddenLayerMismatch"]),
        (1280, 24, 20480, ["TextEncoderDenoiserDimMismatch", "TextEncoderHiddenLayerMismatch"]),
        (1280, 36, 10240, []),
    ],
    ids=["width-mismatch", "insufficient-layers", "both-mismatches", "compatible-override-pair"],
)
def test_text_encoder_gguf_compatibility_checks_adjacent_dimensions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    hidden_size: int,
    num_hidden_layers: int,
    target_dim: int,
    expected_types: list[str],
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (
        TextEncoderComponentArtifact,
    )
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import Flux3PipelineParameters

    adjacent = {"text_config": {"hidden_size": hidden_size, "num_hidden_layers": num_hidden_layers}}
    (tmp_path / "config.json").write_text(json.dumps(adjacent))
    base_configs = {
        "transformer": {"in_channels": 128, "joint_attention_dim": 20480},
        "vae": {"latent_channels": 32},
        "text_encoder": {"text_config": {"hidden_size": 2560, "num_hidden_layers": 36}},
    }
    monkeypatch.setattr(compatibility, "_read_base_repo_config", lambda repo, slot, revision: base_configs.get(slot))
    overrides: dict[str, artifacts.ComponentArtifact] = {
        "text_encoder": TextEncoderComponentArtifact(
            load_id="test-adjacent-dimensions",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="text_encoder",
            file_path=str(tmp_path / "encoder.gguf"),
        )
    }
    if target_dim != base_configs["transformer"]["joint_attention_dim"]:
        transformer_config = tmp_path / "transformer.json"
        transformer_config.write_text(json.dumps({"in_channels": 128, "joint_attention_dim": target_dim}))
        overrides["transformer"] = artifacts.ModelComponentArtifact(
            load_id="test-compatible-transformer",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="transformer",
            file_path="/unused/transformer.gguf",
            config_source=str(transformer_config),
        )

    findings = compatibility.evaluate_component_compatibility(
        overrides, "org/base", pipeline_params_cls=Flux3PipelineParameters
    )

    assert sorted(type(finding).__name__ for finding in findings) == sorted(expected_types)


@pytest.mark.parametrize("adjacent_config", ["missing", "invalid"])
def test_text_encoder_gguf_does_not_borrow_base_config_when_adjacent_config_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, adjacent_config: str
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType
    from modular_diffusion_nodes_library.artifact_utils.text_encoder_component_artifact import (
        TextEncoderComponentArtifact,
    )
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility

    if adjacent_config == "invalid":
        (tmp_path / "config.json").write_text("{invalid json")
    base_reads: list[str] = []

    def base_config(repo: str, slot: str, revision: str | None) -> dict[str, Any]:
        base_reads.append(slot)
        return {"text_config": {"hidden_size": 5120, "num_hidden_layers": 36}}

    monkeypatch.setattr(compatibility, "_read_base_repo_config", base_config)
    artifact = TextEncoderComponentArtifact(
        load_id="test-unavailable-adjacent",
        source_type=ComponentSourceType.SINGLE_FILE,
        component="text_encoder",
        file_path=str(tmp_path / "encoder.gguf"),
    )
    resolver = compatibility._ConfigResolver({"text_encoder": artifact}, "org/base")

    assert resolver.resolved_config("text_encoder") is None
    assert base_reads == []
