from __future__ import annotations

from pathlib import Path

from diffusers import UNet2DConditionModel  # type: ignore[reportMissingImports]
from peft import LoraConfig  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.utils.lora_apply_utils import release_lora_adapters


class _FakeLoraPipeline:
    def __init__(self, adapter_names: list[str]) -> None:
        self.adapter_names = set(adapter_names)
        self.deleted_adapter_names: list[str] = []
        self.unload_calls = 0

    def delete_adapters(self, adapter_names: list[str]) -> None:
        self.deleted_adapter_names.extend(adapter_names)
        self.adapter_names.difference_update(adapter_names)

    def get_list_adapters(self) -> dict[str, list[str]]:
        return {"unet": sorted(self.adapter_names)}

    def unload_lora_weights(self) -> None:
        self.unload_calls += 1


def test_release_lora_adapters_unloads_peft_layers_after_last_adapter() -> None:
    pipe = _FakeLoraPipeline(["first", "second"])

    release_lora_adapters(pipe, ["first", "second"])

    assert pipe.deleted_adapter_names == ["first", "second"]
    assert pipe.unload_calls == 1


def test_release_lora_adapters_keeps_peft_layers_while_other_adapters_remain() -> None:
    pipe = _FakeLoraPipeline(["outer", "inner"])

    release_lora_adapters(pipe, ["inner"])

    assert pipe.deleted_adapter_names == ["inner"]
    assert pipe.unload_calls == 0


def test_release_lora_adapters_restores_original_peft_state_dict_keys(tmp_path: Path) -> None:
    unet = UNet2DConditionModel(
        block_out_channels=(4, 8),
        layers_per_block=1,
        sample_size=8,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=8,
        norm_num_groups=2,
    )
    unet.add_adapter(
        LoraConfig(r=2, lora_alpha=2, target_modules=["to_q", "to_k", "to_v", "to_out.0"]),
        adapter_name="runtime",
    )

    class _TinyPipeline:
        def delete_adapters(self, adapter_names: list[str]) -> None:
            unet.delete_adapters(adapter_names)

        def get_list_adapters(self) -> dict[str, list[str]]:
            return {"unet": list(getattr(unet, "peft_config", {}))}

        def unload_lora_weights(self) -> None:
            unet.unload_lora()

    assert any(".base_layer." in key for key in unet.state_dict())

    release_lora_adapters(_TinyPipeline(), ["runtime"])

    assert not any(".base_layer." in key for key in unet.state_dict())

    unet.save_pretrained(tmp_path)
    reloaded = UNet2DConditionModel.from_pretrained(tmp_path)

    assert not any(parameter.device.type == "meta" for parameter in reloaded.parameters())
