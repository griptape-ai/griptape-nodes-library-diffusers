import logging

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode

from modular_diffusion_nodes_library.parameters.file_path_parameter import FilePathParameter
from modular_diffusion_nodes_library.parameters.lora_parameters import LoraParameters

logger = logging.getLogger("modular_diffusers_nodes_library")


class LoadLora(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lora_file_path_params = FilePathParameter(
            self,
            file_types=[".safetensors", ".sft", ".pt", ".bin", ".json", ".lora"],
            tooltip="Absolute path to a local LoRA file (.safetensors, .sft, .pt, .bin, .json, .lora).",
        )
        self.lora_weight_and_output_params = LoraParameters(self)
        self.lora_file_path_params.add_input_parameters()
        self.lora_weight_and_output_params.add_input_parameters()
        self.lora_weight_and_output_params.add_output_parameters()
        self.add_parameter(
            Parameter(
                name="trigger_phrase",
                default_value="",
                type="str",
                output_type="str",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                tooltip="Optional phrase the LoRA was trained to respond to. Included in the prompt to activate the LoRA's style.",
                hide=True,
            )
        )

    def process(self) -> None:
        self.lora_file_path_params.validate_parameter_values()
        lora_path = str(self.lora_file_path_params.get_file_path())
        lora_weight = self.lora_weight_and_output_params.get_weight()
        trigger_phrase = self.get_parameter_value("trigger_phrase") or None
        self.lora_weight_and_output_params.set_output_lora(
            {"path": lora_path, "weight": lora_weight, "trigger_phrase": trigger_phrase}
        )
