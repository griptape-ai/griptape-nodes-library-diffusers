"""Config-only compatibility checks for overrides on modular pipelines."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("family", ["flux3", "h3"])
@pytest.mark.parametrize("compatible", [False, True])
def test_modular_override_compatibility_uses_architecture_metadata(family: str, compatible: bool) -> None:
    from modular_diffusion_nodes_library.component_loading import component_compatibility as module
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import Flux3PipelineParameters
    from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters

    if family == "flux3":
        params_cls = Flux3PipelineParameters
        layers = 31
        if compatible:
            layers = 32
        configs = {
            "transformer": {"in_channels": 128, "joint_attention_dim": 20480},
            "vae": {"latent_channels": 32},
            "text_encoder": {"hidden_size": 2560, "num_hidden_layers": layers},
        }
    else:
        params_cls = MiniMaxH3PipelineParameters
        audio_channels = 8
        if compatible:
            audio_channels = 16
        configs = {
            "transformer": {"in_channels": 64, "audio_in_channels": 16, "text_dim": 224},
            "vae": {"latent_channels": 64},
            "audio_vae": {"latent_channels": audio_channels},
            "text_encoder": {"hidden_size": 224},
        }

    resolver: Any = SimpleNamespace(resolved_config=configs.get)
    findings = module._run_dimensional_checks(resolver, params_cls)

    if compatible:
        assert findings == []
    else:
        assert len(findings) == 1


@pytest.mark.parametrize("subfolder", [None, "vae"], ids=["target-slot", "explicit-subfolder"])
def test_audio_override_validation_reads_the_same_config_as_materialisation(
    monkeypatch: pytest.MonkeyPatch, subfolder: str | None
) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    configs = {"vae": {"latent_channels": 64}, "audio_vae": {"latent_channels": 32}}
    loaded: dict[str, Any] = {}

    def cached(repo_id: str, filename: str, revision: str | None = None) -> str:
        return str(Path("/virtual") / filename)

    def load(**kwargs: Any) -> object:
        loaded.update(kwargs)
        return object()

    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", cached)
    monkeypatch.setattr(artifacts, "try_load_json_dict", lambda path: configs[Path(path).parent.name])
    pipeline_cls = diffusers.MiniMaxH3ModularPipeline
    monkeypatch.setattr(get_component_class(pipeline_cls, "audio_vae"), "from_pretrained", load)
    artifact = artifacts.ModelComponentArtifact(
        load_id="test-audio-slot",
        source_type=artifacts.ComponentSourceType.HF_REPO,
        component="vae",
        repo_ref=artifacts.HFRepoRef("org/h3", subfolder=subfolder),
    )
    resolver = compatibility._ConfigResolver({"audio_vae": artifact}, "org/base")

    config = resolver.resolved_config("audio_vae")
    artifact.materialize(pipeline_cls=pipeline_cls, slot="audio_vae")

    expected_folder = "audio_vae"
    if subfolder is not None:
        expected_folder = subfolder
    assert loaded["subfolder"] == expected_folder
    assert config == configs[expected_folder]


def test_h3_single_file_override_resolves_its_canonical_config(monkeypatch: pytest.MonkeyPatch) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    pipeline_cls = diffusers.MiniMaxH3ModularPipeline
    component_cls = get_component_class(pipeline_cls, "transformer")
    checkpoint = {"test-weight": object()}
    component = object()

    def resolve(model_type: str, model_cls: type, config_source: str | None, **kwargs: Any) -> Path:
        assert model_type == "minimax-h3"
        assert model_cls is component_cls
        assert config_source is None
        assert kwargs["pipeline_slot"] == "transformer"
        return Path("/virtual/transformer")

    def load(**kwargs: Any) -> object:
        assert kwargs["config"] == "/virtual/transformer"
        assert kwargs["pretrained_model_link_or_path_or_dict"] is checkpoint
        assert kwargs["local_files_only"] is True
        return component

    monkeypatch.setattr("diffusers.loaders.single_file_utils.load_single_file_checkpoint", lambda path: checkpoint)
    monkeypatch.setattr("diffusers.loaders.single_file_utils.infer_diffusers_model_type", lambda weights: "minimax-h3")
    monkeypatch.setattr(artifacts, "resolve_config_path", resolve)
    monkeypatch.setattr(component_cls, "from_single_file", load)
    artifact = artifacts.ModelComponentArtifact(
        load_id="test-h3-single-file",
        source_type=artifacts.ComponentSourceType.SINGLE_FILE,
        component="transformer",
        file_path="/virtual/h3.safetensors",
    )

    assert artifact.materialize(pipeline_cls=pipeline_cls, slot="transformer") is component


@pytest.fixture
def h3_implicit_gguf_config(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.component_loading import component_compatibility as compatibility
    from modular_diffusion_nodes_library.standard_parameters.minimax_h3_parameters import MiniMaxH3PipelineParameters

    configs = {
        "transformer": {"in_channels": 24, "audio_in_channels": 32, "text_dim": 5120},
        "vae": {"latent_channels": 24},
        "audio_vae": {"latent_channels": 32},
        "text_encoder": {"text_config": {"hidden_size": 5120}},
    }
    overrides: dict[str, Any] = {
        "transformer": artifacts.ModelComponentArtifact(
            load_id="test-h3-config-only",
            source_type=artifacts.ComponentSourceType.SINGLE_FILE,
            component="transformer",
            file_path="/unused/h3.gguf",
        )
    }
    monkeypatch.setattr(compatibility, "_read_base_repo_config", lambda repo, slot, revision: configs.get(slot))
    params = MiniMaxH3PipelineParameters.__new__(MiniMaxH3PipelineParameters)
    params.get_build_data = lambda: params.build_data("MiniMaxAI/MiniMax-H3", "selected")
    params.validate_before_node_run = lambda: None
    return SimpleNamespace(configs=configs, overrides=overrides, params=params)


@pytest.mark.parametrize(
    ("slot", "config", "message"),
    [
        ("vae", {"latent_channels": 16}, "VAE latent_channels"),
        ("audio_vae", {"latent_channels": 16}, "Audio VAE latent_channels"),
        ("text_encoder", {"text_config": {"hidden_size": 4096}}, "Text encoder output dimension"),
    ],
)
def test_implicit_gguf_config_still_blocks_incompatible_overrides_before_loading(
    h3_implicit_gguf_config: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    slot: str,
    config: dict[str, Any],
    message: str,
) -> None:
    from modular_diffusion_nodes_library.artifact_utils import component_artifact as artifacts
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )

    (tmp_path / "config.json").write_text(json.dumps(config))
    state = h3_implicit_gguf_config
    state.overrides[slot] = artifacts.ModelComponentArtifact(
        load_id=f"test-incompatible-{slot}",
        source_type=artifacts.ComponentSourceType.LOCAL_DIR,
        component=slot,
        file_path=str(tmp_path),
    )

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Compatibility validation must block before artifact construction or weight loading")

    monkeypatch.setattr(artifacts.ModelComponentArtifact, "materialize", forbidden)
    node: Any = LatentDiffusionPipelineBuilderNode.__new__(LatentDiffusionPipelineBuilderNode)
    node.params = SimpleNamespace(
        pipeline_type_parameters=SimpleNamespace(pipeline_type_pipeline_params=state.params),
        component_override_params=SimpleNamespace(get_component_overrides=lambda: state.overrides),
    )
    node.get_pipeline_artifact_or_raise = forbidden

    errors = node.validate_before_node_run()

    assert errors is not None
    assert len(errors) == 1
    assert message in str(errors[0])
    assert list(tmp_path.iterdir()) == [tmp_path / "config.json"]


def test_implicit_gguf_config_keeps_explicit_config_authoritative(
    h3_implicit_gguf_config: SimpleNamespace, tmp_path: Path
) -> None:
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    config = {"in_channels": 16, "audio_in_channels": 16, "text_dim": 4096}
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config))
    state = h3_implicit_gguf_config
    state.overrides["transformer"] = replace(state.overrides["transformer"], config_source=str(config_file))
    resolver = _ConfigResolver(state.overrides, "MiniMaxAI/MiniMax-H3", "selected")

    assert resolver.resolved_config("transformer") == config


@pytest.mark.parametrize("source", ["single-file", "directory"])
def test_missing_explicit_or_directory_config_is_not_replaced_by_base_config(
    h3_implicit_gguf_config: SimpleNamespace, source: str
) -> None:
    from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentSourceType
    from modular_diffusion_nodes_library.component_loading.component_compatibility import _ConfigResolver

    source_type = ComponentSourceType.SINGLE_FILE
    config_source = "/missing/explicit/config.json"
    if source == "directory":
        source_type = ComponentSourceType.LOCAL_DIR
        config_source = None
    state = h3_implicit_gguf_config
    state.overrides["transformer"] = replace(
        state.overrides["transformer"], source_type=source_type, config_source=config_source
    )
    resolver = _ConfigResolver(state.overrides, "MiniMaxAI/MiniMax-H3", "selected")

    assert resolver.resolved_config("transformer") is None
