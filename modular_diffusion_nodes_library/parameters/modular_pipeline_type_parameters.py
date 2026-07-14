import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact

logger = logging.getLogger("modular_diffusers_nodes_library")

# Copied from diffusers_nodes_library/common/parameters/diffusion/pipeline_type_parameters


class ModularDiffusionPipelineTypePipelineParameters(ABC):
    EXCLUDED_COMPONENT_SLOTS: ClassVar[set[str]] = {
        "image_encoder",
        "feature_extractor",
        "image_processor",
        "audio_vae",
        "connectors",
        "vocoder",
        "processor",
    }

    COMPONENT_SLOT_PRIORITY: ClassVar[list[str]] = [
        "transformer",
        "unet",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "tokenizer",
        "tokenizer_2",
        "tokenizer_3",
        "transformer_2",
        "scheduler",
    ]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        self._node = node
        self._list_all_models = list_all_models

    @abstractmethod
    def add_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_config_kwargs(self) -> dict:
        raise NotImplementedError

    @property
    @abstractmethod
    def pipeline_class(self) -> type[DiffusionPipeline] | type[ModularPipeline]:
        raise NotImplementedError

    @property
    def pipeline_name(self) -> str:
        return self.pipeline_class.__name__

    def get_component_slots(self) -> list[str]:
        """Return component slot names available for override on this pipeline type.

        Derives slots from the pipeline class's __init__ signature via
        DiffusionPipeline._get_signature_keys, filtering out slots listed
        in EXCLUDED_COMPONENT_SLOTS.
        """
        pipeline_cls = self.pipeline_class
        all_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)
        overridable_slots = set(all_slots) - self.EXCLUDED_COMPONENT_SLOTS
        return sorted(overridable_slots, key=self._slot_sort_key)

    @classmethod
    def _slot_sort_key(cls, name: str) -> tuple[int, str]:
        """Return (priority_index, name) for sorting. Lower index = higher priority."""
        try:
            priority = cls.COMPONENT_SLOT_PRIORITY.index(name)
        except ValueError:
            priority = len(cls.COMPONENT_SLOT_PRIORITY)
        return (priority, name)

    @classmethod
    def _materialize_overrides(cls, build_data: dict[str, Any]) -> dict[str, Any]:
        """Extract and materialize component overrides from build_data."""
        raw: dict[str, ComponentArtifact] = build_data.get("_component_overrides", {})
        return {slot: artifact.materialize() for slot, artifact in raw.items()}

    @abstractmethod
    def validate_before_node_run(self) -> list[Exception] | None:
        raise NotImplementedError

    @abstractmethod
    def get_build_data(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def build_pipeline_from_build_data(
        cls, build_data: dict[str, Any]
    ) -> ModularPipeline | DiffusionPipeline | Any | None:
        raise NotImplementedError

    def is_prequantized(self) -> bool:
        """Return True if the model is already quantized (e.g., bnb-4bit).

        Pre-quantized models should not have layerwise casting or additional
        quantization applied.
        """
        return False

    def supports_layerwise_casting(self) -> bool:
        """Return True if the pipeline's transformer supports layerwise casting.

        Some transformers (e.g., ZImage) check weight dtype before calling modules,
        which is incompatible with layerwise casting hooks that cast weights during
        the forward pass.
        """
        return True

    def requires_device_map(self) -> bool:
        """Return True if the pipeline requires device_map during loading.

        Some pipelines (e.g., GLM-Image) have components that must be loaded with
        accelerate's device_map to properly materialize weights. When True:
        - build_pipeline() should use device_map parameter
        - optimize_diffusion_pipeline() should skip .to(device) and CPU offload calls
        """
        return False
