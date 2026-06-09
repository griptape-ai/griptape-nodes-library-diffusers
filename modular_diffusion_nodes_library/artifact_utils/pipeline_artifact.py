from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from importlib import import_module
from typing import Any

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.lora_apply_utils import configure_loras_on_pipeline
from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

logger = logging.getLogger("modular_diffusers_nodes_library")


class DiffusionPipelineArtifact:
    """Serializable description of a diffusion pipeline.

    The artifact itself is not a built pipeline; it captures only the
    information required to reproduce one (`build_data`, LoRAs, optimization
    flags, etc.) plus a config hash used as the model cache key. The actual
    pipeline instance is built lazily by `get_or_build_pipeline()`, which
    delegates to the global `model_cache`.

    Subclasses (e.g. `ControlNetDiffusionPipelineArtifact`) layer additional
    configuration on top of a base artifact.
    """

    def __init__(
        self,
        *,
        pipeline_name: str,
        config_hash: str | None = None,
        builder_module: str | None = None,
        builder_class_name: str | None = None,
        build_data: dict[str, Any] = None,
        build_data_error: str | None = None,
        loras: dict[str, float] | None = None,
        optimization_kwargs: dict[str, Any] | None = None,
        is_prequantized: bool = False,
        supports_layerwise_casting: bool = True,
        requires_device_map: bool = False,
    ) -> None:
        if build_data is None:
            build_data = {}
        self.pipeline_name = pipeline_name
        self.config_hash = config_hash
        self._builder_module = builder_module
        self._builder_class_name = builder_class_name
        self._build_data = copy.deepcopy(build_data)
        self._build_data_error = build_data_error
        self._loras = dict(loras) if loras else {}
        self._optimization_kwargs = copy.deepcopy(optimization_kwargs) if optimization_kwargs else {}
        self._is_prequantized = is_prequantized
        self._supports_layerwise_casting = supports_layerwise_casting
        self._requires_device_map = requires_device_map

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
        }

    def get_or_build_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        if not self.config_hash:
            raise ValueError("Config hash is required to get or build pipeline from artifact.")

        if model_cache.has_pipeline(self.config_hash):
            self._append_log(log_params, "Using cached pipeline.\n")
            return model_cache.get_pipeline(self.config_hash)

        self._append_log(log_params, "No cached pipeline found. Building new pipeline.\n")
        return model_cache.get_or_build_pipeline(self.config_hash, lambda: self._build_pipeline(log_params=log_params))

    def _build_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        base_pipeline = self._build_base_pipeline(log_params=log_params)
        return self._optimize_pipeline(base_pipeline, log_params=log_params, is_reuse=False)

    def _build_base_pipeline(self, log_params: Any | None = None) -> ModularPipeline | DiffusionPipeline | Any:
        self._append_log(log_params, "Creating new pipeline instance...\n")

        if self._build_data_error is not None:
            raise RuntimeError(self._build_data_error)

        if not self._builder_module or not self._builder_class_name:
            raise RuntimeError("Builder module and class name must be specified to build pipeline.")

        with self._profile(log_params, "Loading pipeline"):
            builder_class = self._resolve_builder_class()
            pipe = builder_class.build_pipeline_from_build_data(self._build_data)

        with self._profile(log_params, "Configuring LoRAs"):
            configure_loras_on_pipeline(pipe, self._loras)

        return pipe

    def _optimize_pipeline(
        self,
        base_pipe: Any,
        log_params: Any | None = None,
        *,
        is_reuse: bool = False,
    ) -> ModularPipeline | DiffusionPipeline | Any:
        pipe = base_pipe
        is_prequantized = self._is_prequantized
        if is_reuse:
            is_prequantized = True

        with self._profile(log_params, "Applying optimizations"):
            optimize_diffusion_pipeline(
                pipe=pipe,
                is_prequantized=is_prequantized,
                supports_layerwise_casting=self._supports_layerwise_casting,
                requires_device_map=self._requires_device_map,
                **self._optimization_kwargs,
            )

        if is_reuse:
            self._append_log(log_params, "Pipeline reuse complete.\n")
        else:
            self._append_log(log_params, "Pipeline creation complete.\n")
        return pipe

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

    def _resolve_builder_class(self) -> Any:
        if self._builder_module is None or self._builder_class_name is None:
            raise ValueError("Builder module and class name must be specified to resolve builder class.")
        module = import_module(self._builder_module)
        return getattr(module, self._builder_class_name)


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


class ControlNetDiffusionPipelineArtifact(DiffusionPipelineArtifact):
    def __init__(
        self,
        *,
        base_artifact: DiffusionPipelineArtifact,
        controlnet_models: list[str],
        config_hash: str,
    ) -> None:
        super().__init__(
            config_hash=config_hash,
            pipeline_name=base_artifact.pipeline_name,
            builder_module=base_artifact.builder_module,
            builder_class_name=base_artifact.builder_class_name,
            build_data=base_artifact.build_data,
            build_data_error=base_artifact.build_data_error,
            loras=base_artifact.loras,
            optimization_kwargs=base_artifact.optimization_kwargs,
            is_prequantized=base_artifact.is_prequantized,
            supports_layerwise_casting=base_artifact.supports_layerwise_casting,
            requires_device_map=base_artifact.requires_device_map,
        )
        self._base_artifact = base_artifact
        self._controlnet_models = list(controlnet_models)

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
            control_pipeline = model_cache.get_or_build_pipeline(
                self.config_hash,
                lambda: self._build_pipeline_with_base(base_pipe_ref, log_params=log_params),
            )
        finally:
            if base_pipe_ref is not None and not model_cache.has_pipeline(base_config_hash):
                self._append_log(log_params, "Re-adding base pipeline to cache after control net build.\n")
                model_cache.add_pipeline(base_config_hash, base_pipe_ref)

        return control_pipeline

    def _build_pipeline_with_base(
        self,
        base_pipe: Any | None,
        *,
        log_params: Any | None = None,
    ) -> ModularPipeline | DiffusionPipeline | Any:
        is_reuse = base_pipe is not None
        if base_pipe is not None:
            self._append_log(log_params, "Reusing cached base pipeline — skipping base model load.\n")
        else:
            self._append_log(log_params, "Base pipeline not available — performing full build.\n")
            base_pipe = self._build_base_pipeline(log_params=log_params)

        controlnet_pipe = self._configure_controlnet(base_pipe)
        return self._optimize_pipeline(controlnet_pipe, log_params=log_params, is_reuse=is_reuse)

    def _configure_controlnet(self, pipe: Any) -> Any:
        if not self._controlnet_models:
            return pipe
        driver_class = get_driver_class(self.pipeline_name)
        if driver_class is None:
            return pipe
        return driver_class.control_pipe_from_standard(pipe, self._controlnet_models)
