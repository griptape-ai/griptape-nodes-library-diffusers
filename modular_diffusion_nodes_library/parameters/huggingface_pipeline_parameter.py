import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.traits.options import Options

logger = logging.getLogger("modular_diffusers_nodes_library")


class HuggingFacePipelineParameter:
    def __init__(self, node: BaseNode, *, include_quantization: bool = True):
        self._node = node
        self._include_quantization = include_quantization

    @classmethod
    def get_hf_pipeline_parameter_names(cls, *, include_quantization: bool = True) -> list[str]:
        names = [
            "memory_optimization_strategy",
            "attention_slicing",
            "vae_slicing",
            "vae_tiling",
            "transformer_layerwise_casting",
            "cpu_offload_strategy",
        ]
        if include_quantization:
            names.append("quantization_mode")
        return names

    @property
    def parameter_names(self) -> list[str]:
        return self.get_hf_pipeline_parameter_names(include_quantization=self._include_quantization)

    def get_hf_pipeline_parameters(self) -> dict[str, Any]:
        parameters = {
            "memory_optimization_strategy": self._node.get_parameter_value("memory_optimization_strategy"),
            "attention_slicing": self._node.get_parameter_value("attention_slicing"),
            "vae_slicing": self._node.get_parameter_value("vae_slicing"),
            "vae_tiling": self._node.get_parameter_value("vae_tiling"),
            "transformer_layerwise_casting": self._node.get_parameter_value("transformer_layerwise_casting"),
            "cpu_offload_strategy": self._node.get_parameter_value("cpu_offload_strategy"),
        }
        if self._include_quantization:
            parameters["quantization_mode"] = self._node.get_parameter_value("quantization_mode")
        return parameters

    def add_input_parameters(self) -> None:
        memory_optimization_strategy_choices = ["Manual", "Automatic"]
        memory_optimization_strategy_param = Parameter(
            name="memory_optimization_strategy",
            default_value=memory_optimization_strategy_choices[0],
            type="str",
            traits={
                Options(
                    choices=memory_optimization_strategy_choices,
                )
            },
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Choose Manual to expose individual memory knobs, or Automatic to let Griptape pick reasonable defaults.",
        )
        memory_optimization_strategy_param.set_badge(
            variant="help",
            title="Memory optimization help",
            message=(
                "***Manual*** exposes per-knob toggles (attention slicing, VAE slicing, layerwise casting, "
                f"CPU offload{', quantization' if self._include_quantization else ''}). "
                "***Automatic*** lets Griptape pick reasonable defaults for you.\n\n"
                "Not sure which knob does what? See "
                "[Manual Memory Settings](https://docs.griptapenodes.com/en/latest/nodes/advanced_media_library/diffusion_pipelines/#manual-memory-settings) "
                "for guidance on when to enable each option."
            ),
        )
        self._node.add_parameter(memory_optimization_strategy_param)
        self._node.add_parameter(
            Parameter(
                name="attention_slicing",
                type="bool",
                output_type="bool",
                tooltip="Process attention in slices to reduce peak VRAM at a small speed cost. Enable on memory-constrained GPUs.",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=False,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="vae_slicing",
                type="bool",
                output_type="bool",
                tooltip="Decode the VAE one slice at a time to reduce peak VRAM. Useful for large images or batches.",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=False,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="vae_tiling",
                type="bool",
                output_type="bool",
                tooltip="Enable VAE tiling to reduce memory usage during decode.",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=False,
            )
        )
        self._node.add_parameter(
            Parameter(
                name="transformer_layerwise_casting",
                type="bool",
                output_type="bool",
                tooltip="Cast transformer layers to a lower-precision dtype on the fly to reduce VRAM. May slow inference slightly.",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=False,
            )
        )
        cpu_offload_strategy_choices = ["None", "Model", "Sequential"]
        self._node.add_parameter(
            Parameter(
                name="cpu_offload_strategy",
                default_value=cpu_offload_strategy_choices[0],
                type="str",
                traits={
                    Options(
                        choices=cpu_offload_strategy_choices,
                    )
                },
                tooltip="Offload unused modules to CPU/disk during inference. 'Model' offloads whole submodels; 'Sequential' offloads finer-grained at a larger speed cost.",
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        if self._include_quantization:
            quantization_mode_choices = ["None", "fp8", "int8", "int4"]
            self._node.add_parameter(
                Parameter(
                    name="quantization_mode",
                    type="str",
                    default_value=quantization_mode_choices[0],
                    allowed_modes={ParameterMode.PROPERTY},
                    tooltip="Quantize model weights to reduce VRAM. fp8/int8/int4 trade increasing memory savings for some quality/speed loss.",
                    traits={Options(choices=quantization_mode_choices)},
                )
            )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != "memory_optimization_strategy":
            return

        parameter_names = self.parameter_names[1:]
        update_visibility = (
            self._node.hide_parameter_by_name if value == "Automatic" else self._node.show_parameter_by_name
        )
        for parameter_name in parameter_names:
            update_visibility(parameter_name)
