from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.traits.options import Options

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import (
    MODULAR_PIPELINE_TYPE_PROVIDER_MAP,
    LatentPipelineTypeParameters,
)

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.nodes.latent_diffusion_pipeline_builder_node import (
        LatentDiffusionPipelineBuilderNode,
    )

logger = logging.getLogger("modular_diffusers_nodes_library")

# This code was duplicated/copied from diffusers_nodes_library/common/parameters/diffusion/builder_parameters.py.

_ACRONYMS = {"vae", "unet"}


def _slot_display_name(slot: str) -> str:
    """Convert a component slot name to a human-readable display name.

    Examples: "vae" -> "VAE", "text_encoder_2" -> "Text Encoder 2"
    """
    parts = slot.split("_")
    result: list[str] = []
    for part in parts:
        if part in _ACRONYMS:
            result.append(part.upper())
        else:
            result.append(part.capitalize())
    return " ".join(result)


class ComponentOverrideParameters:
    """Delegate that manages per-component override input ports on the builder node."""

    GROUP_NAME = "component_overrides"

    def __init__(self, node: LatentDiffusionPipelineBuilderNode) -> None:
        self._node = node
        self._current_slots: list[str] = []

    def update_slots(self, slots: list[str]) -> None:
        """Replace the current override ports with ports for the given slot names."""
        if slots == self._current_slots:
            return

        self._node.save_parameter_properties()
        saved_incoming, saved_outgoing = self._node._save_connections()

        self._remove_override_group()
        self._current_slots = list(slots)
        self._add_override_group()

        self._node._restore_connections(saved_incoming, saved_outgoing)
        self._node.reorder_parameters_by_groups()
        self._node.clear_parameter_cache()

    def _add_override_group(self) -> None:
        if not self._current_slots:
            return

        self._node.add_node_element(
            ParameterGroup(
                name=self.GROUP_NAME,
                ui_options={"display_name": "Component Overrides", "collapsed": True},
                user_defined=True,
            )
        )
        for slot in self._current_slots:
            param = Parameter(
                name=f"component_{slot}",
                type="ComponentArtifact",
                input_types=["ComponentArtifact"],
                default_value=None,
                tooltip=f"Optional override for the {_slot_display_name(slot)} component. Connect a Load Component node to replace the default.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": _slot_display_name(slot)},
                parent_element_name=self.GROUP_NAME,
            )
            self._node.add_parameter(param)

    def _remove_override_group(self) -> None:
        self._node.remove_parameter_element_by_name(self.GROUP_NAME)

    @property
    def current_slots(self) -> list[str]:
        return list(self._current_slots)

    def get_component_overrides(self) -> dict[str, ComponentArtifact]:
        """Return connected ComponentArtifact instances keyed by slot name."""
        overrides: dict[str, ComponentArtifact] = {}
        for slot in self._current_slots:
            value = self._node.get_parameter_value(f"component_{slot}")
            if isinstance(value, ComponentArtifact):
                overrides[slot] = value
        return overrides

    def get_override_config_kwargs(self) -> dict[str, str]:
        """Return cache-key entries for connected component overrides."""
        return {
            f"component_override_{slot}": artifact.load_id for slot, artifact in self.get_component_overrides().items()
        }


class LatentDiffusionPipelineBuilderParameters:
    def __init__(self, node: LatentDiffusionPipelineBuilderNode):
        self.provider_choices = list(MODULAR_PIPELINE_TYPE_PROVIDER_MAP.keys())
        self._node = node
        self._pipeline_type_parameters: LatentPipelineTypeParameters
        self._component_override_params = ComponentOverrideParameters(node)
        self.did_provider_change = False
        self.set_pipeline_type_parameters(self.provider_choices[0])

    def add_input_parameters(self) -> None:
        provider_param = Parameter(
            name="provider",
            type="str",
            traits={Options(choices=self.provider_choices)},
            tooltip="Select the model family (Flux, Qwen, SDXL, WAN, etc.) to build a pipeline for.",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"placeholder_text": "Select Provider"},
        )
        provider_param.set_badge(
            variant="help",
            title="Choosing a provider",
            message=(
                "Each provider maps to a model family from a specific organisation:\n\n"
                "- ***Flux / Flux2 / Flux Fill*** — Black Forest Labs. Image generation and inpainting.\n"
                "- ***Qwen / Qwen Edit*** — Alibaba. Image generation and image editing.\n"
                "- ***Stable Diffusion XL / SD3*** — Stability AI. Image generation.\n"
                "- ***WAN / WAN I2V*** — Alibaba. Text-to-video and image-to-video.\n"
                "- ***LTX / LTX2*** — Lightricks. Text-to-video and image-to-video.\n"
                "- ***ZImage*** — ZImage. Image generation and inpainting.\n\n"
                "The provider you select determines which pipeline types and model checkpoints "
                "are available in the fields below."
            ),
        )
        self._node.add_parameter(provider_param)

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

        provider_class = MODULAR_PIPELINE_TYPE_PROVIDER_MAP[provider]  # type: ignore[reportArgumentType]
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

        if parameter.name in {"provider", "pipeline_type"}:
            self._refresh_component_override_ports()

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

    def _refresh_component_override_ports(self) -> None:
        """Update component override ports to match the current pipeline type's slots."""
        slots = self.pipeline_type_parameters.pipeline_type_pipeline_params.get_component_slots()
        self._component_override_params.update_slots(slots)

    @property
    def component_override_params(self) -> ComponentOverrideParameters:
        return self._component_override_params

    def get_config_kwargs(self) -> dict:
        config = {
            **self.pipeline_type_parameters.get_config_kwargs(),
            "provider": self.get_provider(),
        }
        config.update(self._component_override_params.get_override_config_kwargs())
        return config
