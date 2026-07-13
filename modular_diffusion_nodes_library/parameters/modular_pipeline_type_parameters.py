import logging
from abc import ABC, abstractmethod
from typing import Any

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import ALLOWED_COMPONENT_SLOTS

logger = logging.getLogger("modular_diffusers_nodes_library")

# Copied from diffusers_nodes_library/common/parameters/diffusion/pipeline_type_parameters


class ModularDiffusionPipelineTypePipelineParameters(ABC):
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

        Intersects the pipeline's __init__ signature (via
        DiffusionPipeline._get_signature_keys) with ALLOWED_COMPONENT_SLOTS,
        preserving the priority order defined there.
        """
        pipeline_cls = self.pipeline_class
        all_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)
        all_slots_set = set(all_slots)
        return [slot for slot in ALLOWED_COMPONENT_SLOTS if slot in all_slots_set]

    @classmethod
    def _materialize_overrides(cls, build_data: dict[str, Any], *, pipeline_cls: type) -> dict[str, Any]:
        """Extract and materialize component overrides from build_data."""
        raw: dict[str, ComponentArtifact] = build_data.get("_component_overrides", {})
        return {slot: artifact.materialize(pipeline_cls=pipeline_cls) for slot, artifact in raw.items()}

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
