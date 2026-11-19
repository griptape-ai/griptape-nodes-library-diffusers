"""Restore BF16-stored GGUF tensors that Diffusers loads as raw uint8 parameters.

H3 aligns inputs with projection parameter dtypes. A byte-backed BF16 projection therefore
casts text embeddings to uint8 before a BF16 matmul. Norm weights and biases also need decoding.
Only tensors stored as BF16 are restored; genuinely quantised weights remain quantised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def restore_bf16_gguf_tensors(model: torch.nn.Module) -> int:
    """Restore BF16 parameters and replace fully BF16 GGUF linear layers before offloading."""
    restored = _restore_bf16_linear_layers(model)
    restored += _restore_bf16_parameters(model)
    return restored


def _restore_bf16_linear_layers(model: torch.nn.Module) -> int:
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear

    restored = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if not isinstance(child, GGUFLinear) or not _is_bf16(child.weight):
                continue
            linear = torch.nn.Linear(child.in_features, child.out_features, bias=child.bias is not None, device="meta")
            linear.weight = _to_plain(child.weight)
            restored += 1
            if child.bias is not None:
                linear.bias = child.bias
            parent.add_module(name, linear)
    return restored


def _restore_bf16_parameters(model: torch.nn.Module) -> int:
    from diffusers.quantizers.gguf.utils import GGUFLinear

    restored = 0
    for module in model.modules():
        for name, parameter in list(module.named_parameters(recurse=False, remove_duplicate=False)):
            if isinstance(module, GGUFLinear) and name == "weight":
                continue
            if _is_bf16(parameter):
                module.register_parameter(name, _to_plain(parameter))
                restored += 1
    return restored


def _is_bf16(parameter: object) -> bool:
    import gguf
    from diffusers.quantizers.gguf.utils import GGUFParameter

    return (
        isinstance(parameter, GGUFParameter)
        and getattr(parameter, "quant_type", None) == gguf.GGMLQuantizationType.BF16
    )


def _to_plain(parameter: torch.Tensor) -> torch.nn.Parameter:
    import torch
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor

    return torch.nn.Parameter(dequantize_gguf_tensor(parameter).to(torch.bfloat16), requires_grad=False)
