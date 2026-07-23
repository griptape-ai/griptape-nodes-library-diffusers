import inspect
import logging
from abc import ABC, abstractmethod
from typing import Any

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_model_parameter import HuggingFaceModelParameter

from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import ALLOWED_COMPONENT_SLOTS

logger = logging.getLogger("modular_diffusers_nodes_library")

# Copied from diffusers_nodes_library/common/parameters/diffusion/pipeline_type_parameters


def _required_init_components(pipeline_cls: type) -> set[str]:
    """Init params with no default (excluding self). These MUST be supplied
    to build the pipeline, whether via from_pretrained or direct construction.
    """
    parameters = inspect.signature(pipeline_cls.__init__).parameters
    return {name for name, param in parameters.items() if name != "self" and param.default is inspect.Parameter.empty}


class ModelParamsError(RuntimeError):
    """Raised when there is an issue resolving model parameters (e.g. no HF
    model selected).
    """


class ModularDiffusionPipelineTypePipelineParameters(ABC):
    _pipeline_cls: type[DiffusionPipeline] | type[ModularPipeline]

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        self._node = node
        self._list_all_models = list_all_models

    def _resolve_repo(self, hf_param: HuggingFaceModelParameter) -> tuple[str, str]:
        """Return ``(repo_id, revision)`` for ``hf_param``, or raise
        ``ModelParamsError`` if the user hasn't picked a value yet.
        """
        param_name = hf_param._parameter_name  # noqa: SLF001
        value = self._node.get_parameter_value(param_name)
        if value is None:
            msg = f"Required input '{param_name}' on node '{self._node.name}' has no value selected."
            raise ModelParamsError(msg)
        if not hf_param.list_repo_revisions():
            msg = (
                f"Required input '{param_name}' on node '{self._node.name}' has no models available "
                f"(nothing cached locally that matches this parameter's filter)."
            )
            raise ModelParamsError(msg)
        return hf_param.get_repo_revision()

    def _resolve_fixed_repo(self, hf_param: HuggingFaceModelParameter) -> tuple[str, str]:
        """Return ``(repo_id, revision)`` for a parameter not shown in the UI.
        Unlike ``_resolve_repo``, this does not require the parameter to be
        registered on the node.  It scans the HF cache directly.
        """
        revisions = hf_param.fetch_repo_revisions()
        if not revisions:
            param_name = hf_param._parameter_name  # noqa: SLF001
            msg = (
                f"Required model '{param_name}' on node '{self._node.name}' "
                f"has no models available (nothing cached locally)."
            )
            raise ModelParamsError(msg)
        return revisions[0]

    @abstractmethod
    def add_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def remove_input_parameters(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_config_kwargs(self) -> dict:
        raise NotImplementedError

    @classmethod
    def pipeline_cls(cls) -> type[DiffusionPipeline] | type[ModularPipeline]:
        return cls._pipeline_cls

    @property
    def pipeline_name(self) -> str:
        return self._pipeline_cls.__name__

    def get_component_slots(self) -> list[str]:
        """Return component slot names available for override on this pipeline type.

        Intersects the pipeline's __init__ signature (via
        DiffusionPipeline._get_signature_keys) with ALLOWED_COMPONENT_SLOTS,
        preserving the priority order defined there.
        """
        all_slots, _ = self._pipeline_cls._get_signature_keys(self._pipeline_cls)
        all_slots_set = set(all_slots)
        return [slot for slot in ALLOWED_COMPONENT_SLOTS if slot in all_slots_set]

    @classmethod
    def _materialize_overrides(cls, build_data: dict[str, Any], *, pipeline_cls: type) -> dict[str, Any]:
        """Extract and materialize component overrides from build_data."""
        raw: dict[str, ComponentArtifact] = build_data.get("_component_overrides", {})
        return {slot: artifact.materialize(pipeline_cls=pipeline_cls, slot=slot) for slot, artifact in raw.items()}

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """Return False if this pipeline deliberately opts out of being built from component overrides alone."""
        return True

    @classmethod
    def get_auto_supplied_components(cls) -> set[str]:
        """Return the set of required __init__ components this class supplies itself.

        Subclasses that override ``_build_pipeline_from_overrides_only`` to inject
        components not in ``ALLOWED_COMPONENT_SLOTS`` (e.g. a default-constructed
        guider) must list those component names here so
        ``verify_overridable_covers_required`` accepts the class.
        """
        return set()

    @classmethod
    def verify_overridable_covers_required(cls) -> None:
        """Raise if any required init arg of ``cls._pipeline_cls`` is not covered."""
        pipeline_cls = cls._pipeline_cls
        required = _required_init_components(pipeline_cls)
        missing = required - set(ALLOWED_COMPONENT_SLOTS) - cls.get_auto_supplied_components()
        if missing:
            msg = (
                f"Attempted to register pipeline class '{pipeline_cls.__name__}'. "
                f"Failed because these required __init__ components are not in "
                f"ALLOWED_COMPONENT_SLOTS and not declared in "
                f"{cls.__name__}.get_auto_supplied_components(): {sorted(missing)}. "
                f"Add them to modular_diffusion_nodes_library/component_loading/component_slots.py "
                f"(both ALLOWED_COMPONENT_SLOTS and SLOT_DISPLAY_NAMES) so a user can supply "
                f"them via an override port, list them in get_auto_supplied_components() if "
                f"{cls.__name__} injects them itself, or override "
                f"supports_build_from_overrides_only() to return False."
            )
            raise RuntimeError(msg)

    @abstractmethod
    def validate_before_node_run(self) -> list[Exception] | None:
        raise NotImplementedError

    @abstractmethod
    def get_build_data(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def _build_pipeline_from_overrides_only(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> ModularPipeline | DiffusionPipeline | Any | None:
        """Build pipeline directly from materialized component overrides."""
        pipeline_cls = build_data["_pipeline_cls"]
        return pipeline_cls(**overrides)

    @classmethod
    def build_pipeline_from_build_data(
        cls, build_data: dict[str, Any]
    ) -> ModularPipeline | DiffusionPipeline | Any | None:
        """Build pipeline from build_data. Routes to overrides-only or repo path."""
        pipeline_cls = build_data.get("_pipeline_cls") or cls._pipeline_cls
        overrides = cls._materialize_overrides(build_data, pipeline_cls=pipeline_cls)

        if build_data.get("_all_overrides"):
            return cls._build_pipeline_from_overrides_only(build_data, overrides)
        return cls._build_pipeline_from_repo(build_data, overrides)

    @classmethod
    @abstractmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
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
