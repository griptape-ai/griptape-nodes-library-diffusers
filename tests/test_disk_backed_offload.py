from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

pytest.importorskip("torch", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("diffusers", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("accelerate", reason="Uses execution dependencies; run in the execution environment.")
pytest.importorskip("transformers", reason="Uses execution dependencies; run in the execution environment.")

import torch  # noqa: E402
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL  # noqa: E402
from diffusers.modular_pipelines.components_manager import ComponentsManager, CustomOffloadHook  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks  # noqa: E402
from diffusers.modular_pipelines.modular_pipeline_utils import ComponentSpec  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from modular_diffusion_nodes_library.utils.disk_backed_offload import (  # noqa: E402
    DiskBackedComponent,
    DiskBackedOffloadHook,
    disk_backed_component_from_spec,
    empty_component_from_spec,
    enable_disk_backed_offload,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import clear_pipeline  # noqa: E402


def _evict_all(hooks: list[Any], model_id: str, model: torch.nn.Module, execution_device: torch.device) -> list[Any]:
    return list(hooks)


def _hook(module: torch.nn.Module) -> Any:
    return cast(Any, module)._hf_hook


def _decode(vae: AutoencoderKL, latents: torch.Tensor) -> torch.Tensor:
    return cast(Any, vae).decode(latents).sample


def _is_meta(module: torch.nn.Module) -> bool:
    return all(tensor.device.type == "meta" for tensor in (*module.parameters(), *module.buffers()))


def _is_resident(module: torch.nn.Module) -> bool:
    return all(tensor.device.type != "meta" for tensor in (*module.parameters(), *module.buffers()))


@dataclass
class _Checkpoints:
    lm_dir: Path
    vae_dir: Path


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> _Checkpoints:
    root = tmp_path_factory.mktemp("checkpoints")
    torch.manual_seed(0)

    config = Qwen3Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(root / "lm")

    AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        latent_channels=4,
        layers_per_block=1,
        norm_num_groups=4,
        sample_size=16,
    ).save_pretrained(root / "vae")
    return _Checkpoints(lm_dir=root / "lm", vae_dir=root / "vae")


def _lm_component(directory: Path) -> DiskBackedComponent:
    return DiskBackedComponent(
        load=lambda device: Qwen3ForCausalLM.from_pretrained(directory, dtype=torch.float32, device_map=str(device)),
        weight_files=tuple(directory.glob("*.safetensors")),
    )


def _vae_component(directory: Path) -> DiskBackedComponent:
    return DiskBackedComponent(
        load=lambda device: AutoencoderKL.from_pretrained(directory, dtype=torch.float32, device_map=str(device)),
        weight_files=tuple(directory.glob("*.safetensors")),
    )


@dataclass
class _Rig:
    lm: Qwen3ForCausalLM
    vae: AutoencoderKL
    manager: ComponentsManager
    ids: torch.Tensor
    latents: torch.Tensor
    ref_logits: torch.Tensor
    ref_hidden: torch.Tensor
    ref_decoded: torch.Tensor
    ref_inv_freq: torch.Tensor


def _make_rig(checkpoints: _Checkpoints, device: str, *, disk_backed_vae: bool = True, **enable_kwargs: Any) -> _Rig:
    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map=device)
    vae = AutoencoderKL.from_pretrained(checkpoints.vae_dir, dtype=torch.float32, device_map=device)
    ids = torch.tensor([[1, 2, 3, 4]], device=device)
    latents = torch.randn(1, 4, 4, 4, device=device, generator=torch.Generator(device).manual_seed(0))
    with torch.no_grad():
        ref_logits = lm(ids).logits
        ref_hidden = lm.model(input_ids=ids).last_hidden_state
        ref_decoded = _decode(vae, latents)
    ref_inv_freq = lm.model.rotary_emb.inv_freq.clone()

    manager = ComponentsManager()
    manager.add("text_encoder", lm)
    manager.add("vae", vae)
    components = {"text_encoder": _lm_component(checkpoints.lm_dir)}
    if disk_backed_vae:
        components["vae"] = _vae_component(checkpoints.vae_dir)
    enable_disk_backed_offload(manager, device, components, **enable_kwargs)
    return _Rig(lm, vae, manager, ids, latents, ref_logits, ref_hidden, ref_decoded, ref_inv_freq)


def _make_mixed_rig(checkpoints: _Checkpoints, device: str, **enable_kwargs: Any) -> _Rig:
    """The text encoder is disk-backed; the VAE keeps diffusers' stock CPU offload."""
    return _make_rig(checkpoints, device, disk_backed_vae=False, **enable_kwargs)


@pytest.fixture
def rig(checkpoints: _Checkpoints) -> _Rig:
    return _make_rig(checkpoints, "cpu", offload_strategy=_evict_all)


def test_enabling_releases_every_component(rig: _Rig) -> None:
    assert _is_meta(rig.lm)
    assert _is_meta(rig.vae)
    assert rig.lm.device.type == "meta"
    assert isinstance(_hook(rig.lm), DiskBackedOffloadHook)
    assert isinstance(_hook(rig.vae), DiskBackedOffloadHook)


def test_forward_rebuilds_the_same_module_in_place(rig: _Rig) -> None:
    lm_before = rig.lm
    forward_before = rig.lm.forward

    with torch.no_grad():
        logits = rig.lm(rig.ids).logits

    torch.testing.assert_close(logits, rig.ref_logits)
    assert rig.lm is lm_before
    assert rig.lm.forward is forward_before
    assert _is_resident(rig.lm)
    # The checkpoint stores one tensor for both, so reloading has to keep them tied.
    assert rig.lm.lm_head.weight is rig.lm.model.embed_tokens.weight
    # Non-persistent: not in the checkpoint, so it only survives if the reload supplies it.
    torch.testing.assert_close(rig.lm.model.rotary_emb.inv_freq, rig.ref_inv_freq)


def test_hand_fired_hook_then_submodule_call_rebuilds_the_weights(rig: _Rig) -> None:
    # A block may fire the hook by hand and then call a submodule, bypassing the top-level `forward`.
    _hook(rig.lm).pre_forward(rig.lm)

    with torch.no_grad():
        hidden = rig.lm.model(input_ids=rig.ids).last_hidden_state

    torch.testing.assert_close(hidden, rig.ref_hidden)


def test_vae_decode_triggers_a_reload(rig: _Rig) -> None:
    with torch.no_grad():
        decoded = _decode(rig.vae, rig.latents)

    torch.testing.assert_close(decoded, rig.ref_decoded)


def test_using_another_component_releases_the_resident_one(rig: _Rig) -> None:
    with torch.no_grad():
        rig.lm(rig.ids)
        assert _is_resident(rig.lm)
        assert _is_meta(rig.vae)

        _decode(rig.vae, rig.latents)
        assert _is_resident(rig.vae)
        assert _is_meta(rig.lm)

        logits = rig.lm(rig.ids).logits
        assert _is_meta(rig.vae)

    torch.testing.assert_close(logits, rig.ref_logits)


def test_a_resident_component_is_not_reloaded_on_later_calls(rig: _Rig) -> None:
    with torch.no_grad():
        rig.lm(rig.ids)
        weight = rig.lm.lm_head.weight
        rig.lm(rig.ids)

    assert rig.lm.lm_head.weight is weight


def test_missing_weight_files_fail_before_anything_is_evicted(checkpoints: _Checkpoints, tmp_path: Path) -> None:
    lm_dir = tmp_path / "lm"
    shutil.copytree(checkpoints.lm_dir, lm_dir)
    local = _Checkpoints(lm_dir=lm_dir, vae_dir=checkpoints.vae_dir)
    rig = _make_rig(local, "cpu", offload_strategy=_evict_all)

    with torch.no_grad():
        _decode(rig.vae, rig.latents)
        for weight_file in lm_dir.glob("*.safetensors"):
            weight_file.unlink()

        with pytest.raises(FileNotFoundError, match="no longer exist"):
            rig.lm(rig.ids)

    assert _is_resident(rig.vae)
    assert _is_meta(rig.lm)


def test_a_reload_with_a_different_structure_is_rejected_without_partial_adoption(
    checkpoints: _Checkpoints,
) -> None:
    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    manager = ComponentsManager()
    manager.add("text_encoder", lm)
    wrong = DiskBackedComponent(load=lambda device: torch.nn.Linear(2, 2), weight_files=())
    enable_disk_backed_offload(manager, "cpu", {"text_encoder": wrong}, offload_strategy=_evict_all)

    with pytest.raises(ValueError, match="zip"):
        _hook(lm).pre_forward(lm)

    assert _is_meta(lm)


def test_wrapped_parameters_are_rejected(checkpoints: _Checkpoints) -> None:
    class Wrapped(torch.nn.Parameter):
        pass

    module = torch.nn.Linear(2, 2)
    module.weight = Wrapped(module.weight.data)
    manager = ComponentsManager()
    manager.add("quantized", module)
    component = DiskBackedComponent(load=lambda device: torch.nn.Linear(2, 2), weight_files=())

    with pytest.raises(TypeError, match="plain torch.nn.Parameter"):
        enable_disk_backed_offload(manager, "cpu", {"quantized": component}, offload_strategy=_evict_all)


def test_a_rejected_component_leaves_the_manager_and_other_components_untouched(checkpoints: _Checkpoints) -> None:
    class Wrapped(torch.nn.Parameter):
        pass

    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    module = torch.nn.Linear(2, 2)
    module.weight = Wrapped(module.weight.data)
    manager = ComponentsManager()
    # `text_encoder` is valid; the `quantized` one must still stop everything before any hook is attached.
    manager.add("text_encoder", lm)
    manager.add("quantized", module)
    unused = DiskBackedComponent(load=lambda device: torch.nn.Linear(2, 2), weight_files=())
    components = {"text_encoder": _lm_component(checkpoints.lm_dir), "quantized": unused}

    with pytest.raises(TypeError, match="plain torch.nn.Parameter"):
        enable_disk_backed_offload(manager, "cpu", components, offload_strategy=_evict_all)

    assert not hasattr(lm, "_hf_hook")
    assert not hasattr(module, "_hf_hook")
    assert _is_resident(lm)
    assert manager.model_hooks is None


def test_a_module_without_a_reload_description_falls_back_to_the_stock_cpu_hook(checkpoints: _Checkpoints) -> None:
    rig = _make_mixed_rig(checkpoints, "cpu", offload_strategy=_evict_all)

    assert isinstance(_hook(rig.lm), DiskBackedOffloadHook)
    assert type(_hook(rig.vae)) is CustomOffloadHook
    assert _is_meta(rig.lm)
    assert _is_resident(rig.vae)
    assert rig.vae.device == torch.device("cpu")


def test_disk_backed_and_stock_components_both_still_produce_reference_outputs(checkpoints: _Checkpoints) -> None:
    # On a CPU execution device the stock hook sees its module as already in place and never evicts, so this pins
    # correctness only; the CUDA test below covers the eviction in each direction.
    rig = _make_mixed_rig(checkpoints, "cpu", offload_strategy=_evict_all)

    with torch.no_grad():
        logits = rig.lm(rig.ids).logits
        decoded = _decode(rig.vae, rig.latents)
        logits_again = rig.lm(rig.ids).logits

    # The disk-backed hook evicted the VAE through the stock hook, which parks it on CPU, never on meta.
    assert _is_resident(rig.vae)
    torch.testing.assert_close(logits, rig.ref_logits)
    torch.testing.assert_close(logits_again, rig.ref_logits)
    torch.testing.assert_close(decoded, rig.ref_decoded)


def test_a_reload_description_for_a_component_the_manager_lacks_is_rejected(checkpoints: _Checkpoints) -> None:
    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    manager = ComponentsManager()
    manager.add("text_encoder", lm)

    with pytest.raises(ValueError, match=r"\['txt_encoder'\]"):
        enable_disk_backed_offload(
            manager, "cpu", {"txt_encoder": _lm_component(checkpoints.lm_dir)}, offload_strategy=_evict_all
        )

    assert not hasattr(lm, "_hf_hook")
    assert _is_resident(lm)


def _lm_spec(checkpoints: _Checkpoints) -> ComponentSpec:
    return ComponentSpec(
        name="text_encoder",
        type_hint=Qwen3ForCausalLM,
        pretrained_model_name_or_path=str(checkpoints.lm_dir.parent),
        subfolder="lm",
        default_creation_method="from_pretrained",
    )


def test_an_empty_component_matches_the_checkpoint_and_reloads_on_first_use(checkpoints: _Checkpoints) -> None:
    loaded = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        ref_logits = loaded(ids).logits
    spec = _lm_spec(checkpoints)

    lm = cast(Qwen3ForCausalLM, empty_component_from_spec(spec, dtype=torch.float32))

    assert all(p.device.type == "meta" for p in lm.parameters())
    assert {name: (t.shape, t.dtype) for name, t in lm.state_dict().items()} == {
        name: (t.shape, t.dtype) for name, t in loaded.state_dict().items()
    }

    manager = ComponentsManager()
    manager.add("text_encoder", lm)
    enable_disk_backed_offload(
        manager, "cpu", {"text_encoder": _lm_component(checkpoints.lm_dir)}, offload_strategy=_evict_all
    )
    with torch.no_grad():
        logits = lm(ids).logits

    torch.testing.assert_close(logits, ref_logits)


def test_an_empty_diffusers_component_matches_the_checkpoint(checkpoints: _Checkpoints) -> None:
    loaded = AutoencoderKL.from_pretrained(checkpoints.vae_dir, dtype=torch.float32)
    spec = ComponentSpec(
        name="vae",
        type_hint=AutoencoderKL,
        pretrained_model_name_or_path=str(checkpoints.vae_dir.parent),
        subfolder="vae",
        default_creation_method="from_pretrained",
    )

    vae = empty_component_from_spec(spec, dtype=torch.bfloat16)

    assert all(p.device.type == "meta" for p in vae.parameters())
    assert [(name, p.shape) for name, p in vae.named_parameters()] == [
        (name, p.shape) for name, p in loaded.named_parameters()
    ]
    assert {p.dtype for p in vae.parameters()} == {torch.bfloat16}


def test_clearing_a_modular_pipeline_drops_references_without_moving_modules(
    checkpoints: _Checkpoints, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Blocks(ModularPipelineBlocks):
        @property
        def expected_components(self) -> list[ComponentSpec]:
            return [ComponentSpec("text_encoder", Qwen3ForCausalLM), ComponentSpec("vae", AutoencoderKL)]

        def __call__(self, components: Any, state: Any) -> Any:
            return components, state

    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    vae = AutoencoderKL.from_pretrained(checkpoints.vae_dir, dtype=torch.float32, device_map="cpu")
    latents = torch.randn(1, 4, 4, 4, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref_decoded = _decode(vae, latents)
    manager = ComponentsManager()
    pipe = _Blocks().init_pipeline(components_manager=manager)
    pipe.update_components(text_encoder=lm, vae=vae)
    enable_disk_backed_offload(
        manager, "cpu", {"text_encoder": _lm_component(checkpoints.lm_dir)}, offload_strategy=_evict_all
    )
    assert _is_meta(lm)

    def _no_move(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("clearing moved a module")

    with monkeypatch.context() as patch:
        patch.setattr(torch.nn.Module, "to", _no_move)
        clear_pipeline(pipe)

    assert manager.components == {}
    assert manager.added_time == {}
    assert manager.model_hooks is None
    assert pipe.components.get("text_encoder") is None
    assert pipe.components.get("vae") is None
    assert not hasattr(vae, "_hf_hook")
    # Another cached pipeline sharing the module can still use it.
    with torch.no_grad():
        torch.testing.assert_close(_decode(vae, latents), ref_decoded)


def test_clearing_a_pipeline_leaves_a_disk_backed_module_released_and_unhooked(checkpoints: _Checkpoints) -> None:
    # Pins the limit noted on `_disk_backed_component_names`: removing the hook does not reload the weights, so a
    # disk-backed module must not be shared with another cached pipeline.
    class _Blocks(ModularPipelineBlocks):
        @property
        def expected_components(self) -> list[ComponentSpec]:
            return [ComponentSpec("text_encoder", Qwen3ForCausalLM)]

        def __call__(self, components: Any, state: Any) -> Any:
            return components, state

    lm = Qwen3ForCausalLM.from_pretrained(checkpoints.lm_dir, dtype=torch.float32, device_map="cpu")
    manager = ComponentsManager()
    pipe = _Blocks().init_pipeline(components_manager=manager)
    pipe.update_components(text_encoder=lm)
    enable_disk_backed_offload(
        manager, "cpu", {"text_encoder": _lm_component(checkpoints.lm_dir)}, offload_strategy=_evict_all
    )

    clear_pipeline(pipe)

    assert not hasattr(lm, "_hf_hook")
    assert _is_meta(lm)


def test_component_description_reads_a_local_checkpoint(checkpoints: _Checkpoints) -> None:
    spec = ComponentSpec(
        name="text_encoder",
        type_hint=Qwen3ForCausalLM,
        pretrained_model_name_or_path=str(checkpoints.lm_dir.parent),
        subfolder="lm",
        default_creation_method="from_pretrained",
    )

    component = disk_backed_component_from_spec(spec, dtype=torch.float32)
    loaded = component.load(torch.device("cpu"))

    assert [path.name for path in component.weight_files] == ["model.safetensors"]
    assert isinstance(loaded, Qwen3ForCausalLM)
    assert _is_resident(loaded)


def test_a_reload_uses_the_default_memory_mapped_cpu_load(
    checkpoints: _Checkpoints, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `device_map` or `disable_mmap=True` would break the disk-backed offload; see `load`.
    load_kwargs: list[dict[str, Any]] = []
    original = Qwen3ForCausalLM.from_pretrained

    def spy(*args: Any, **kwargs: Any) -> Any:
        load_kwargs.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(Qwen3ForCausalLM, "from_pretrained", spy)
    component = disk_backed_component_from_spec(_lm_spec(checkpoints), dtype=torch.float32)

    component.load(torch.device("cpu"))

    assert load_kwargs == [{"dtype": torch.float32}]


def test_component_description_rejects_a_checkpoint_directory_without_weights(tmp_path: Path) -> None:
    (tmp_path / "lm").mkdir()
    spec = ComponentSpec(
        name="text_encoder",
        type_hint=Qwen3ForCausalLM,
        pretrained_model_name_or_path=str(tmp_path),
        subfolder="lm",
        default_creation_method="from_pretrained",
    )

    with pytest.raises(FileNotFoundError, match="no .safetensors files"):
        disk_backed_component_from_spec(spec, dtype=torch.float32)


def test_component_description_rejects_a_spec_without_a_type_hint(tmp_path: Path) -> None:
    spec = ComponentSpec(name="text_encoder", pretrained_model_name_or_path=str(tmp_path), subfolder="lm")

    with pytest.raises(ValueError, match="no type_hint"):
        disk_backed_component_from_spec(spec, dtype=torch.float32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a CUDA device.")
def test_swap_on_cuda_with_the_real_strategy_frees_device_memory(checkpoints: _Checkpoints) -> None:
    # A margin larger than any card forces the stock strategy to evict whatever is resident.
    rig = _make_rig(checkpoints, "cuda", memory_reserve_margin="100000GB")
    baseline = torch.cuda.memory_allocated()

    with torch.no_grad():
        rig.lm(rig.ids)
        assert torch.cuda.memory_allocated() > baseline
        assert rig.lm.device == torch.device("cuda:0")

        _decode(rig.vae, rig.latents)
        assert _is_meta(rig.lm)
        assert rig.vae.device == torch.device("cuda:0")

        logits = rig.lm(rig.ids).logits
        assert _is_meta(rig.vae)

    torch.testing.assert_close(logits, rig.ref_logits)
    del logits

    for user_hook in rig.manager.model_hooks or []:
        user_hook.offload()
    assert torch.cuda.memory_allocated() == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs a CUDA device.")
def test_mixed_swap_on_cuda_releases_the_text_encoder_and_parks_the_vae_on_cpu(checkpoints: _Checkpoints) -> None:
    rig = _make_mixed_rig(checkpoints, "cuda", memory_reserve_margin="100000GB")

    with torch.no_grad():
        rig.lm(rig.ids)
        assert rig.lm.device == torch.device("cuda:0")

        _decode(rig.vae, rig.latents)
        assert _is_meta(rig.lm)
        assert rig.vae.device == torch.device("cuda:0")

        logits = rig.lm(rig.ids).logits
        assert rig.vae.device == torch.device("cpu")

    torch.testing.assert_close(logits, rig.ref_logits)
