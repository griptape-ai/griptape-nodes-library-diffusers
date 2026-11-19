from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast

import pytest


@pytest.fixture
def diffusers_module() -> Any:
    return pytest.importorskip("diffusers", reason="Reads real Diffusers APIs; run `make test/exec`.")


class FakePipeline:
    def __init__(self, prompt_upsampler_components: tuple[str, ...]) -> None:
        self._blocks = SimpleNamespace(sub_blocks={"prompt_upsampler": object()})
        self._prompt_upsampler_components = prompt_upsampler_components
        self.components = dict.fromkeys(
            (
                "transformer",
                "vae",
                "text_encoder",
                "tokenizer",
                "image_processor",
                "scheduler",
                "guider",
                *prompt_upsampler_components,
            )
        )
        for name in self.components:
            setattr(self, name, None)
        self._specs = {
            name: SimpleNamespace(
                name=name,
                default_creation_method="from_pretrained",
                pretrained_model_name_or_path="black-forest-labs/flux-3-image-klein-base",
                subfolder=name,
                revision=None,
            )
            for name in self.components
        }
        self.loaded_names: list[str] | None = None
        self.load_kwargs: dict[str, Any] | None = None
        self.updated_components: dict[str, Any] = {}

    @property
    def blocks(self) -> Any:
        blocks = deepcopy(self._blocks)
        blocks.expected_components = [
            spec
            for name, spec in self._specs.items()
            if name not in self._prompt_upsampler_components or "prompt_upsampler" in blocks.sub_blocks
        ]
        blocks.get_workflow = lambda workflow: blocks
        return blocks

    def get_component_spec(self, name: str) -> SimpleNamespace:
        return self._specs[name]

    def load_components(self, *, names: list[str], **kwargs: Any) -> None:
        self.loaded_names = names
        self.load_kwargs = kwargs
        for name in names:
            setattr(self, name, object())

    def update_components(self, **components: Any) -> None:
        self.updated_components.update(components)
        for name, component in components.items():
            setattr(self, name, component)


class FakeComponentsManager:
    def enable_auto_cpu_offload(self, **kwargs: Any) -> None:
        self.offload_kwargs = kwargs


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

    pipeline = FakePipeline(PROMPT_UPSAMPLER_COMPONENTS)
    manager = FakeComponentsManager()
    transformer = object()
    single_file_calls: list[tuple[Any, dict[str, Any]]] = []

    def fake_from_single_file(cls: Any, path: Any, **kwargs: Any) -> object:
        single_file_calls.append((path, kwargs))
        return transformer

    monkeypatch.setattr(
        diffusers_module.ModularPipeline, "from_pretrained", classmethod(lambda cls, *args, **kwargs: pipeline)
    )
    monkeypatch.setattr("diffusers.modular_pipelines.components_manager.ComponentsManager", lambda: manager)
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
        "base_repo_id": "black-forest-labs/flux-3-image-klein-base",
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
    assert pipeline.loaded_names is not None
    assert pipeline.load_kwargs is not None
    assert pipeline.load_kwargs["local_files_only"] is True
    if override_slot == "tokenizer":
        assert "fix_mistral_regex" not in pipeline.load_kwargs
    else:
        assert pipeline.load_kwargs["fix_mistral_regex"] == {"tokenizer": True}
    expected_updates = dict(overrides)
    if variant and override_slot != "transformer":
        assert "transformer" not in pipeline.loaded_names
        assert len(single_file_calls) == 1
        path, kwargs = single_file_calls[0]
        assert path == weights_path
        assert kwargs["config"] == "black-forest-labs/flux-3-image-klein-base"
        assert kwargs["config_revision"] == "base-revision"
        assert kwargs["subfolder"] == "transformer"
        assert kwargs["local_files_only"] is True
        assert isinstance(kwargs["quantization_config"], diffusers_module.BFLQuantizationConfig)
        expected_updates["transformer"] = transformer
        assert fp8r_checks == (["cpu"] if variant == "fp8r" else [])
    else:
        if override_slot == "transformer":
            assert "transformer" not in pipeline.loaded_names
        else:
            assert "transformer" in pipeline.loaded_names
        assert single_file_calls == []
        assert fp8r_checks == []
    assert pipeline.updated_components == expected_updates
    if override_slot is not None:
        assert override_slot not in pipeline.loaded_names
        assert getattr(pipeline, override_slot) is overrides[override_slot]
    if upsampler_selected:
        assert all(name in pipeline.loaded_names for name in PROMPT_UPSAMPLER_COMPONENTS)
        for name in PROMPT_UPSAMPLER_COMPONENTS:
            assert pipeline.load_kwargs["pretrained_model_name_or_path"][name] == PROMPT_UPSAMPLER_REPO_ID
            assert pipeline.load_kwargs["subfolder"][name] == ""
            assert pipeline.load_kwargs["revision"][name] == "pup-revision"
    else:
        assert all(name not in pipeline.loaded_names for name in PROMPT_UPSAMPLER_COMPONENTS)
        assert all(
            source == "black-forest-labs/flux-3-image-klein-base"
            for source in pipeline.load_kwargs["pretrained_model_name_or_path"].values()
        )
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
