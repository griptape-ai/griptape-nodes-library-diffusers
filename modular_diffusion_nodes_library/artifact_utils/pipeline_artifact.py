from __future__ import annotations

import copy
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.pipeline_build_steps import (
    ApplyOptimizationStep,
    AttachControlNetStep,
    BuildStep,
    FuseLorasStep,
    LoadPipelineStep,
    run_build_steps,
)
from modular_diffusion_nodes_library.artifact_utils.recipe_codec import (
    json_safe,
    optional_string,
    require_bool,
    require_dict,
    require_loras,
    require_string,
    require_string_list,
    resolve_recipe_value,
)
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.pipeline_runtime_adapter_step import PipelineRuntimeAdapterStep

logger = logging.getLogger("modular_diffusers_nodes_library")

DEFAULT_BAKED_DTYPE = "bfloat16"


def _digest(parts: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


@dataclass(frozen=True)
class BasePipelineIdentity:
    """Cache-key identity for a pipeline build.

    Owns both the digest input (the typed fields below) and the public
    cache-key layout (``{pipeline_name}-{HASH}-{postfix:x}``). Not
    runtime-hashable (contains dicts); use `cache_key()`.
    """

    pipeline_name: str
    config_kwargs: dict[str, Any]
    loras: dict[str, float]
    optimization_kwargs: dict[str, Any]
    torch_dtype: str
    postfix_bits: int = 0

    def cache_key(self) -> str:
        parts: dict[str, Any] = {
            **self.config_kwargs,
            **self.loras,
            "torch_dtype": self.torch_dtype,
            **{f"opt_{k}": v for k, v in self.optimization_kwargs.items()},
        }
        return f"{self.pipeline_name}-{_digest(parts)}-{self.postfix_bits:x}"


class DiffusionPipelineArtifact:
    """Serializable description of a diffusion pipeline.

    The artifact itself is not a built pipeline; it captures only the
    information required to reproduce one (`build_data`, LoRAs, optimization
    flags, etc.) plus a config hash used as the model cache key. The actual
    pipeline instance is built lazily by `get_or_build_pipeline()`, which
    delegates to the global `model_cache`.

    Subclasses (e.g. `ControlNetDiffusionPipelineArtifact`) layer additional
    configuration on top of a base artifact.

    Field mutability
    -------------------------
    `with_additional_runtime_adapter_steps()` uses `copy.copy()` (shallow copy) to clone this
    object. This is safe only because every field is either immutable (str,
    bool) or guarded by a copy-returning property (`build_data`, `loras`,
    `optimization_kwargs`). If a new mutable field is added that is directly
    settable from outside the class — i.e. not protected by such a property —
    `with_additional_runtime_adapter_steps()` must be updated to use `copy.deepcopy()` instead.
    """

    def __init__(
        self,
        *,
        pipeline_name: str,
        config_hash: str | None = None,
        builder_module: str | None = None,
        builder_class_name: str | None = None,
        build_data: dict[str, Any] | None = None,
        build_data_error: str | None = None,
        loras: dict[str, float] | None = None,
        optimization_kwargs: dict[str, Any] | None = None,
        is_prequantized: bool = False,
        supports_layerwise_casting: bool = True,
        requires_device_map: bool = False,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.config_hash = config_hash
        self._builder_module = builder_module
        self._builder_class_name = builder_class_name
        if build_data is not None:
            self._build_data = copy.deepcopy(build_data)
        else:
            self._build_data = {}
        self._build_data_error = build_data_error
        self._loras = dict(loras) if loras else {}
        self._optimization_kwargs = copy.deepcopy(optimization_kwargs) if optimization_kwargs else {}
        self._is_prequantized = is_prequantized
        self._supports_layerwise_casting = supports_layerwise_casting
        self._requires_device_map = requires_device_map
        self._extra_runtime_adapter_steps: list[PipelineRuntimeAdapterStep] = []

    @property
    def builder_module(self) -> str | None:
        return self._builder_module

    @property
    def builder_class_name(self) -> str | None:
        return self._builder_class_name

    @property
    def build_data(self) -> dict[str, Any]:
        return copy.deepcopy(self._build_data)

    @property
    def build_data_error(self) -> str | None:
        return self._build_data_error

    @property
    def loras(self) -> dict[str, float]:
        return dict(self._loras)

    @property
    def optimization_kwargs(self) -> dict[str, Any]:
        return copy.deepcopy(self._optimization_kwargs)

    @property
    def is_prequantized(self) -> bool:
        return self._is_prequantized

    @property
    def supports_layerwise_casting(self) -> bool:
        return self._supports_layerwise_casting

    @property
    def requires_device_map(self) -> bool:
        return self._requires_device_map

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "pipeline_name": self.pipeline_name,
            "builder_module": self._builder_module,
            "builder_class_name": self._builder_class_name,
            "build_data": self.build_data,
            "build_data_error": self._build_data_error,
            "loras": self.loras,
            "optimization_kwargs": self.optimization_kwargs,
            "is_prequantized": self._is_prequantized,
            "supports_layerwise_casting": self._supports_layerwise_casting,
            "requires_device_map": self._requires_device_map,
            "runtime_adapter_steps": [step.metadata for step in self.runtime_adapter_steps()],
        }

    def _ensure_recipe_serializable(self) -> None:
        if self.runtime_adapter_steps():
            msg = "Pipelines with runtime adapter steps cannot be saved. Save the base pipeline and reconnect the adapter."
            raise ValueError(msg)

    def to_recipe_dict(self) -> dict[str, Any]:
        """Render this pipeline configuration as a JSON-safe recipe dict."""
        self._ensure_recipe_serializable()
        if type(self) is not DiffusionPipelineArtifact:
            msg = f"Unsupported pipeline artifact type: {type(self).__name__}"
            raise ValueError(msg)
        return {
            "type": "base",
            "pipeline_name": self.pipeline_name,
            "config_hash": self.config_hash,
            "builder": {"module": self.builder_module, "class_name": self.builder_class_name},
            "build_data": json_safe(self.build_data),
            "build_data_error": self.build_data_error,
            "loras": json_safe(self.loras),
            "optimization": json_safe(self.optimization_kwargs),
            "flags": {
                "is_prequantized": self.is_prequantized,
                "supports_layerwise_casting": self.supports_layerwise_casting,
                "requires_device_map": self.requires_device_map,
            },
        }

    def check_bakeable(self) -> None:
        """Raise if this pipeline uses a feature Full Pipeline save doesn't support yet."""
        if self.is_prequantized:
            msg = (
                "Full Pipeline save does not yet support pre-quantised models "
                "(e.g. bnb-4bit repos); their quantised weights are not verified to round-trip "
                "through save_pretrained. Use Config Only instead."
            )
            raise ValueError(msg)

        quantization_mode = self.optimization_kwargs.get("quantization_mode", "None")
        if quantization_mode != "None":
            msg = (
                f"Full Pipeline save does not yet support quantization_mode={quantization_mode!r} "
                "(quantised weights are not guaranteed to round-trip through save_pretrained). "
                "Use Config Only, or set Quantization Mode to 'None' before saving."
            )
            raise ValueError(msg)

        overrides = self.build_data.get("_component_overrides", {})
        for slot, override in overrides.items():
            if getattr(override, "is_quantized", False):
                msg = (
                    f"Full Pipeline save does not yet support quantised component overrides "
                    f"(slot '{slot}' is quantised, e.g. GGUF). Use Config Only instead."
                )
                raise ValueError(msg)

    def to_baked_config_dict(self) -> dict[str, Any]:
        """Render the artifact-owned half of a baked folder's sidecar config.

        Fields describing the folder rather than the artifact (schema version,
        layout, dtype of the built pipe) are added by the caller, mirroring how
        `serialize_pipeline_artifact` wraps `to_recipe_dict`. LoRAs and
        quantization settings are dropped: baking fuses them into the weights.
        """
        self.check_bakeable()
        return {
            "pipeline_name": self.pipeline_name,
            "builder_module": self.builder_module,
            "builder_class_name": self.builder_class_name,
            "optimization_kwargs": self.optimization_kwargs,
            "is_prequantized": self.is_prequantized,
            "supports_layerwise_casting": self.supports_layerwise_casting,
            "requires_device_map": self.requires_device_map,
        }

    @classmethod
    def from_baked_config_dict(
        cls, data: dict[str, Any], *, baked_path: Path, config_hash: str
    ) -> DiffusionPipelineArtifact:
        """Reconstruct an artifact that rebuilds from `baked_path` instead of a hub repo.

        Rebuilding still goes through the pipeline type's own builder, so any
        pipeline-specific load behavior (e.g. MiniMaxH3's ComponentsManager
        offload) is preserved.
        """
        return cls(
            pipeline_name=data.get("pipeline_name", baked_path.name),
            config_hash=config_hash,
            builder_module=optional_string(data.get("builder_module"), "Baked pipeline builder module"),
            builder_class_name=optional_string(data.get("builder_class_name"), "Baked pipeline builder class name"),
            build_data={
                "_baked_path": str(baked_path),
                "_baked_dtype": data.get("torch_dtype", DEFAULT_BAKED_DTYPE),
            },
            loras={},  # already fused into the saved weights
            optimization_kwargs=data.get("optimization_kwargs"),
            is_prequantized=data.get("is_prequantized", False),
            supports_layerwise_casting=data.get("supports_layerwise_casting", True),
            requires_device_map=data.get("requires_device_map", False),
        )

    @classmethod
    def from_recipe_dict(cls, data: dict[str, Any]) -> DiffusionPipelineArtifact:
        """Reconstruct a pipeline artifact, dispatching on its recorded type."""
        artifact_type = data.get("type")
        if artifact_type == "controlnet":
            return ControlNetDiffusionPipelineArtifact.from_recipe_dict(data)
        if artifact_type != "base":
            msg = f"Unsupported pipeline recipe artifact type: {artifact_type!r}"
            raise ValueError(msg)
        builder = require_dict(data.get("builder"), "Pipeline builder")
        flags = require_dict(data.get("flags"), "Pipeline flags")
        return cls(
            pipeline_name=require_string(data.get("pipeline_name"), "Pipeline name"),
            config_hash=require_string(data.get("config_hash"), "Pipeline config hash"),
            builder_module=optional_string(builder.get("module"), "Pipeline builder module"),
            builder_class_name=optional_string(builder.get("class_name"), "Pipeline builder class name"),
            build_data=require_dict(resolve_recipe_value(data.get("build_data")), "Pipeline build data"),
            build_data_error=optional_string(data.get("build_data_error"), "Pipeline build data error"),
            loras=require_loras(resolve_recipe_value(data.get("loras"))),
            optimization_kwargs=require_dict(resolve_recipe_value(data.get("optimization")), "Pipeline optimization"),
            is_prequantized=require_bool(flags.get("is_prequantized"), "Pipeline is_prequantized flag"),
            supports_layerwise_casting=require_bool(
                flags.get("supports_layerwise_casting"), "Pipeline supports_layerwise_casting flag"
            ),
            requires_device_map=require_bool(flags.get("requires_device_map"), "Pipeline requires_device_map flag"),
        )

    def get_or_build_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        if not self.config_hash:
            raise ValueError("Config hash is required to get or build pipeline from artifact.")

        if model_cache.has_pipeline(self.config_hash):
            self._append_log(log_params, "Using cached pipeline.\n")
            return model_cache.get_pipeline(self.config_hash)

        self._append_log(log_params, "No cached pipeline found. Building new pipeline.\n")
        return model_cache.get_or_build_pipeline(self.config_hash, lambda: self._build_pipeline(log_params=log_params))

    def runtime_adapter_steps(self) -> list[PipelineRuntimeAdapterStep]:
        """Per-generation context-managed transformations applied around each generation call."""
        return list(getattr(self, "_extra_runtime_adapter_steps", []))

    def with_additional_runtime_adapter_steps(
        self, steps: list[PipelineRuntimeAdapterStep]
    ) -> DiffusionPipelineArtifact:
        """Return a shallow copy of this artifact with `steps` appended to its activation chain.

        The copy shares the same `config_hash` — no pipeline rebuild is triggered.
        Calling this on an artifact that already has runtime adapter steps chains them:
        existing steps run first, then `steps`.
        """
        clone = copy.copy(self)
        clone._extra_runtime_adapter_steps = [*self.runtime_adapter_steps(), *steps]
        return clone

    @contextmanager
    def activate(self, pipe: Any, *, node_name: str | None = None):
        """Fold `runtime_adapter_steps()` around `pipe` for one generation."""
        steps = self.runtime_adapter_steps()
        if not steps:
            yield pipe
            return
        with ExitStack() as stack:
            for step in steps:
                pipe = stack.enter_context(step.activate(pipe, node_name=node_name))
            yield pipe

    def _build_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        return run_build_steps(self._compose_build_steps(), log_params=log_params)

    def _compose_build_steps(self) -> list[BuildStep]:
        """Step list for a from-scratch build: load → fuse LoRAs → optimize."""
        return [*self._load_and_fuse_steps(), self._optimize_step(is_reuse=False)]

    def _load_and_fuse_steps(self) -> list[BuildStep]:
        """Common prefix used by both base and subclass step compositions."""
        return [
            LoadPipelineStep(
                builder_module=self._builder_module,
                builder_class_name=self._builder_class_name,
                build_data=self._build_data,
                build_data_error=self._build_data_error,
            ),
            FuseLorasStep(self._loras),
        ]

    def _optimize_step(self, *, is_reuse: bool) -> ApplyOptimizationStep:
        return ApplyOptimizationStep(
            optimization_kwargs=self._optimization_kwargs,
            is_prequantized=self._is_prequantized,
            supports_layerwise_casting=self._supports_layerwise_casting,
            requires_device_map=self._requires_device_map,
            is_reuse=is_reuse,
        )

    def __call__(self) -> ModularPipeline | DiffusionPipeline | Any:
        return self.get_or_build_pipeline()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiffusionPipelineArtifact):
            return False
        if type(self) is not type(other):
            return False

        return self.metadata == other.metadata

    def __repr__(self) -> str:
        return f"DiffusionPipelineArtifact(config_hash={self.config_hash!r}, pipeline_name={self.pipeline_name!r})"

    def __str__(self) -> str:
        return self.config_hash or f"DiffusionPipelineArtifact({self.pipeline_name})"

    @contextmanager
    def _profile(self, log_params: Any | None, label: str):
        if log_params is None:
            yield
            return

        with log_params.append_profile_to_logs(label):
            yield

    def _append_log(self, log_params: Any | None, message: str) -> None:
        if log_params is None:
            return
        log_params.append_to_logs(message)


def normalize_diffusion_pipeline_value(
    value: Any,
    *,
    node_name: str | None = None,
    raise_on_invalid: bool = False,
) -> DiffusionPipelineArtifact | None:
    if value is None:
        return None

    if isinstance(value, DiffusionPipelineArtifact):
        return value

    message = f"Invalid 'pipeline' value type '{type(value).__name__}'. Expected DiffusionPipelineArtifact."
    if node_name is not None:
        message = f"{node_name}: {message}"

    if raise_on_invalid:
        raise ValueError(message)

    if node_name is not None:
        logger.warning("%s: ignoring 'pipeline' value of type %s.", node_name, type(value).__name__)

    return None


class BaseDiffusionPipelineArtifact(DiffusionPipelineArtifact, ABC):
    """Base class for artifacts that compose/derive from another artifact.

    Handles common cache reuse pattern for ControlNet and LoRA pipelines.

    Subclasses must implement:
    - `_get_reuse_log_context()`: Return a context string for log messages.
    - `_build_pipeline_with_base()`: Build the derived pipeline from a base pipeline.
    """

    def __init__(
        self,
        *,
        base_artifact: DiffusionPipelineArtifact,
        config_hash: str,
        loras: dict[str, float] | None = None,
    ) -> None:
        super().__init__(
            config_hash=config_hash,
            pipeline_name=base_artifact.pipeline_name,
            builder_module=base_artifact.builder_module,
            builder_class_name=base_artifact.builder_class_name,
            build_data=base_artifact.build_data,
            build_data_error=base_artifact.build_data_error,
            loras=loras if loras is not None else base_artifact.loras,
            optimization_kwargs=base_artifact.optimization_kwargs,
            is_prequantized=base_artifact.is_prequantized,
            supports_layerwise_casting=base_artifact.supports_layerwise_casting,
            requires_device_map=base_artifact.requires_device_map,
        )
        self._base_artifact = base_artifact

    @property
    def base_artifact(self) -> DiffusionPipelineArtifact:
        return self._base_artifact

    @abstractmethod
    def _get_reuse_log_context(self) -> str:
        """Return context string for reuse log messages (e.g., 'control net', 'LoRA pipeline')."""
        ...

    def get_or_build_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        if not self.config_hash:
            raise ValueError("Config hash is required to get or build pipeline from artifact.")

        if model_cache.has_pipeline(self.config_hash):
            self._append_log(log_params, "Using cached pipeline.\n")
            return model_cache.get_pipeline(self.config_hash)

        self._append_log(log_params, "No cached pipeline found. Building new pipeline.\n")
        base_pipe_ref = None
        base_config_hash = self._base_artifact.config_hash
        if not base_config_hash:
            raise ValueError("Base artifact config hash is required to get or build pipeline from artifact.")

        if model_cache.has_pipeline(base_config_hash):
            self._append_log(log_params, "Base pipeline found in cache. Capturing reference for reuse.\n")
            base_pipe_ref = model_cache.get_pipeline(base_config_hash)
            model_cache.take_pipeline(base_config_hash)

        try:
            derived_pipeline = model_cache.get_or_build_pipeline(
                self.config_hash,
                lambda: self._build_pipeline_with_base(base_pipe_ref, log_params=log_params),
            )
        finally:
            if base_pipe_ref is not None and not model_cache.has_pipeline(base_config_hash):
                if self._should_return_base_to_cache():
                    context = self._get_reuse_log_context()
                    self._append_log(log_params, f"Re-adding base pipeline to cache after {context} build.\n")
                    model_cache.add_pipeline(base_config_hash, base_pipe_ref)

        return derived_pipeline

    def _should_return_base_to_cache(self) -> bool:
        """Whether the base pipeline should be re-added to cache after the derived pipeline is built.

        Subclasses that mutate shared components should return False to prevent a contaminated base from re-entering the cache.
        """
        return True

    @abstractmethod
    def _build_pipeline_with_base(
        self,
        base_pipe: Any | None,
        *,
        log_params: Any | None = None,
    ) -> ModularPipeline | DiffusionPipeline | Any: ...


class ControlNetDiffusionPipelineArtifact(BaseDiffusionPipelineArtifact):
    def __init__(
        self,
        *,
        base_artifact: DiffusionPipelineArtifact,
        controlnet_models: list[str],
        config_hash: str,
    ) -> None:
        super().__init__(
            base_artifact=base_artifact,
            config_hash=config_hash,
        )
        self._controlnet_models = list(controlnet_models)

    @property
    def controlnet_models(self) -> list[str]:
        return list(self._controlnet_models)

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata
        metadata.update(
            {
                "base_config_hash": self._base_artifact.config_hash,
                "controlnet_models": list(self._controlnet_models),
            }
        )
        return metadata

    def check_bakeable(self) -> None:
        msg = "Full Pipeline save does not yet support ControlNet pipelines. Use Config Only instead."
        raise ValueError(msg)

    def to_recipe_dict(self) -> dict[str, Any]:
        self._ensure_recipe_serializable()
        return {
            "type": "controlnet",
            "base": self.base_artifact.to_recipe_dict(),
            "config_hash": self.config_hash,
            "controlnet_models": json_safe(self.controlnet_models),
        }

    @classmethod
    def from_recipe_dict(cls, data: dict[str, Any]) -> ControlNetDiffusionPipelineArtifact:
        return cls(
            base_artifact=DiffusionPipelineArtifact.from_recipe_dict(
                require_dict(data.get("base"), "ControlNet base artifact")
            ),
            controlnet_models=require_string_list(data.get("controlnet_models"), "ControlNet models"),
            config_hash=require_string(data.get("config_hash"), "ControlNet config hash"),
        )

    def _get_reuse_log_context(self) -> str:
        return "control net"

    def _build_pipeline_with_base(
        self,
        base_pipe: Any | None,
        *,
        log_params: Any | None = None,
    ) -> ModularPipeline | DiffusionPipeline | Any:
        is_reuse = base_pipe is not None
        if is_reuse:
            self._append_log(log_params, "Reusing cached base pipeline — skipping base model load.\n")
        else:
            self._append_log(log_params, "Base pipeline not available — performing full build.\n")

        steps = self._compose_build_steps_with_base(is_reuse=is_reuse)
        return run_build_steps(steps, log_params=log_params, initial_pipe=base_pipe)

    def _compose_build_steps_with_base(self, *, is_reuse: bool) -> list[BuildStep]:
        """On reuse, skip load/fuse and only attach ControlNet + re-optimize."""
        steps: list[BuildStep] = [] if is_reuse else list(self._load_and_fuse_steps())
        steps.append(
            AttachControlNetStep(
                pipeline_name=self.pipeline_name,
                controlnet_models=list(self._controlnet_models),
            )
        )
        steps.append(self._optimize_step(is_reuse=is_reuse))
        return steps
