from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.traits.options import Options

from diffusers_nodes_library.parameters.pipelinetype_parameters import (
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
    LatentPipelineTypeParameters,
)

if TYPE_CHECKING:
    from diffusers_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was duplicated/copied from diffusers_nodes_library/common/parameters/diffusion/builder_parameters.py.


class LatentDiffusionPipelineBuilderParameters:
    def __init__(self, node: LatentDiffusionPipelineBuilderNode):
        self.provider_choices = list(MODULAR_PIPELINE_TYPE_PROVIDER_MAP.keys())
        self._node = node
        self._pipeline_type_parameters: LatentPipelineTypeParameters
        self.did_provider_change = False
        self.set_pipeline_type_parameters(self.provider_choices[0])

    def add_input_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="provider",
                type="str",
                traits={Options(choices=self.provider_choices)},
                tooltip="Select the model family (Flux, Qwen, SDXL, WAN, etc.) to build a pipeline for.",
                allowed_modes={ParameterMode.PROPERTY},
                ui_options={"placeholder_text": "Select Provider"},
            )
        )

    def add_output_parameters(self) -> None:
        self._node.add_parameter(
            Parameter(
                name="pipeline",
                output_type="Pipeline Config",
                default_value=None,
                tooltip="Built and cached 🤗 Diffusion pipeline. Connect to a Generate Latents, Encode Media, or Decode Latents node etc.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "pipeline"},
            )
        )

    def set_pipeline_type_parameters(self, provider: str) -> None:
        if provider not in MODULAR_PIPELINE_TYPE_PROVIDER_MAP:
            msg = f"Unsupported pipeline provider: {provider}"
            logger.error(msg)
            raise ValueError(msg)

        provider_class = MODULAR_PIPELINE_TYPE_PROVIDER_MAP[provider]
        self._pipeline_type_parameters = provider_class(self._node)

    def before_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "provider":
            current_provider = self._node.get_parameter_value("provider")
            self.did_provider_change = current_provider != value
        self.pipeline_type_parameters.before_value_set(parameter, value)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "provider" and self.did_provider_change:
            self.regenerate_pipeline_type_parameters_for_provider(value)
        self.pipeline_type_parameters.after_value_set(parameter, value)

    def regenerate_pipeline_type_parameters_for_provider(self, provider: str) -> None:
        # Save parameter properties and connections before removing parameters
        self._node.save_parameter_properties()
        saved_incoming, saved_outgoing = self._node._save_connections()

        self.pipeline_type_parameters.remove_input_parameters()
        self.set_pipeline_type_parameters(provider)
        self.pipeline_type_parameters.add_input_parameters()

        first_pipeline_type = self.pipeline_type_parameters.pipeline_types[0]
        self._node.set_parameter_value("pipeline_type", first_pipeline_type)

        # Restore connections after adding parameters
        self._node._restore_connections(saved_incoming, saved_outgoing)

        # Reorder parameters to maintain consistent layout
        self._node.reorder_parameters_by_groups()

        self._node.clear_parameter_cache()

    @property
    def pipeline_type_parameters(self) -> LatentPipelineTypeParameters:
        if self._pipeline_type_parameters is None:
            msg = "Pipeline type parameters not initialized. Ensure provider parameter is set."
            logger.error(msg)
            raise ValueError(msg)
        return self._pipeline_type_parameters

    def get_provider(self) -> str:
        return self._node.get_parameter_value("provider")

    def get_config_kwargs(self) -> dict:
        return {
            **self.pipeline_type_parameters.get_config_kwargs(),
            "provider": self.get_provider(),
        }
