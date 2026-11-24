from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from tests.fake_modular_pipeline import (
    FakeComponent,
    FakeModularPipeline,
    component_spec,
    install_components_manager,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FLUX3_BASE_REPO = "black-forest-labs/flux-3-image-klein-base"


@pytest.fixture
def diffusers_module() -> Any:
    return pytest.importorskip("diffusers", reason="Reads real Diffusers APIs; run `make test/exec`.")


def flux3_pipeline(prompt_upsampler_components: Sequence[str]) -> FakeModularPipeline:
    """Flux3's components, with the prompt upsampler's in their own sub-block as in Diffusers."""
    core = ["transformer", "vae", "text_encoder", "tokenizer", "image_processor", "scheduler", "guider"]
    upsampler = list(prompt_upsampler_components)
    return FakeModularPipeline(
        [component_spec(name, pretrained_model_name_or_path=FLUX3_BASE_REPO) for name in [*core, *upsampler]],
        sub_blocks={"core": core, "prompt_upsampler": upsampler},
        workflows={
            "text2image": ["core", "prompt_upsampler"],
            "image_conditioned": ["core", "prompt_upsampler"],
        },
    )


@pytest.mark.parametrize("upsampler_selected", [True, False], ids=["upsampler", "none"])
@pytest.mark.parametrize("variant", ["", "fp8r", "nvfp4"], ids=["default", "fp8r", "nvfp4"])
@pytest.mark.parametrize("override_slot", [None, "transformer", "vae", "text_encoder", "tokenizer"])
def test_loader_loads_or_omits_prompt_upsampler_and_quantized_transformer(
    monkeypatch: pytest.MonkeyPatch,
    diffusers_module: Any,
    upsampler_selected: bool,
    variant: str,
    override_slot: str | None,
) -> None:
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import (
        PROMPT_UPSAMPLER_COMPONENTS,
        PROMPT_UPSAMPLER_REPO_ID,
        Flux3PipelineParameters,
    )

    pipeline = flux3_pipeline(PROMPT_UPSAMPLER_COMPONENTS)
    install_components_manager(monkeypatch, pipeline)
    transformer = object()
    single_file_calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_from_single_file(cls: Any, path: Any, **kwargs: Any) -> object:
        single_file_calls.append((path, kwargs))
        return transformer

    monkeypatch.setattr(
        diffusers_module.ModularPipeline, "from_pretrained", classmethod(lambda cls, *args, **kwargs: pipeline)
    )
    monkeypatch.setattr(
        diffusers_module.Flux3Transformer2DModel, "from_single_file", classmethod(fake_from_single_file)
    )
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.standard_parameters.flux3_parameters.get_best_device", lambda: "cpu"
    )
    fp8r_checks: list[Any] = []
    monkeypatch.setattr(
        "modular_diffusion_nodes_library.standard_parameters.flux3_parameters.require_fp8r_support",
        fp8r_checks.append,
    )
    weights_path = f"/cache/snapshots/base-revision/variants/{variant}/model.safetensors"
    build_data = {
        "base_repo_id": FLUX3_BASE_REPO,
        "variant": variant,
        "variant_weights_path": weights_path if variant else None,
        "revision": "base-revision",
        "prompt_upsampler_repo_id": PROMPT_UPSAMPLER_REPO_ID if upsampler_selected else None,
        "prompt_upsampler_revision": "pup-revision" if upsampler_selected else None,
    }

    overrides: dict[str, Any] = {}
    if override_slot is not None:
        overrides[override_slot] = object()
    result = Flux3PipelineParameters._build_pipeline_from_repo(build_data, overrides)

    assert result is pipeline
    # Components the loader fetched from a repo, as opposed to overrides and the single-file transformer.
    loaded = {name: c for name, c in pipeline.components.items() if isinstance(c, FakeComponent)}
    assert all(component.load_kwargs["local_files_only"] is True for component in loaded.values())
    with_mistral_fix = [name for name, component in loaded.items() if "fix_mistral_regex" in component.load_kwargs]
    if override_slot == "tokenizer":
        assert with_mistral_fix == []
    else:
        assert with_mistral_fix == ["tokenizer"]
        assert loaded["tokenizer"].load_kwargs["fix_mistral_regex"] is True
    if variant and override_slot != "transformer":
        assert pipeline.transformer is transformer
        assert len(single_file_calls) == 1
        path, kwargs = single_file_calls[0]
        assert path == weights_path
        assert kwargs["config"] == FLUX3_BASE_REPO
        assert kwargs["config_revision"] == "base-revision"
        assert kwargs["subfolder"] == "transformer"
        assert kwargs["local_files_only"] is True
        assert isinstance(kwargs["quantization_config"], diffusers_module.BFLQuantizationConfig)
        assert fp8r_checks == (["cpu"] if variant == "fp8r" else [])
    else:
        if override_slot == "transformer":
            assert "transformer" not in loaded
        else:
            assert "transformer" in loaded
        assert single_file_calls == []
        assert fp8r_checks == []
    if override_slot is not None:
        assert override_slot not in loaded
        assert getattr(pipeline, override_slot) is overrides[override_slot]
    if upsampler_selected:
        for name in PROMPT_UPSAMPLER_COMPONENTS:
            assert loaded[name].source == PROMPT_UPSAMPLER_REPO_ID
            assert loaded[name].load_kwargs["subfolder"] == ""
            assert loaded[name].load_kwargs["revision"] == "pup-revision"
        base_components = [c for name, c in loaded.items() if name not in PROMPT_UPSAMPLER_COMPONENTS]
    else:
        assert all(name not in loaded for name in PROMPT_UPSAMPLER_COMPONENTS)
        base_components = list(loaded.values())
    assert all(component.source == FLUX3_BASE_REPO for component in base_components)
    assert all(component.load_kwargs["revision"] == "base-revision" for component in base_components)
    assert ("prompt_upsampler" in pipeline.blocks.sub_blocks) is upsampler_selected


def test_prompt_upsampler_picker_offers_none_and_downloadable_repo(
    monkeypatch: pytest.MonkeyPatch, diffusers_module: Any
) -> None:
    from griptape_nodes.exe_types.node_types import BaseNode

    from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
        ConnectableHuggingFaceRepoParameter,
    )
    from modular_diffusion_nodes_library.standard_parameters.flux3_parameters import (
        PROMPT_UPSAMPLER_NONE,
        PROMPT_UPSAMPLER_REPO_ID,
        OptionalPromptUpsamplerRepoParameter,
    )

    monkeypatch.setattr(ConnectableHuggingFaceRepoParameter, "get_choices", lambda self: [PROMPT_UPSAMPLER_REPO_ID])
    parameter = OptionalPromptUpsamplerRepoParameter.__new__(OptionalPromptUpsamplerRepoParameter)
    parameter._repo_ids = [PROMPT_UPSAMPLER_REPO_ID]
    parameter._deprecated_repos = []
    parameter._parameter_name = "prompt_upsampler_model"
    parameter._node = cast(BaseNode, SimpleNamespace(get_parameter_value=lambda name: PROMPT_UPSAMPLER_NONE))

    assert parameter.get_choices() == [PROMPT_UPSAMPLER_REPO_ID, PROMPT_UPSAMPLER_NONE]
    assert parameter.repo_id_for_choice(PROMPT_UPSAMPLER_NONE) is None
    assert parameter.repo_id_for_choice(PROMPT_UPSAMPLER_REPO_ID) == PROMPT_UPSAMPLER_REPO_ID
    assert parameter.get_download_models() == [PROMPT_UPSAMPLER_REPO_ID]
    assert parameter.validate_before_node_run() is None
