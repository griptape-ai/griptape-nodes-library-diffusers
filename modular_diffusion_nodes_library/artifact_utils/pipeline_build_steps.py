"""Typed build steps for pipeline artifacts.

Each step takes the in-flight pipe (or `None` for the initial load) and
returns the next pipe. `run_build_steps()` folds a list of steps.
"""

from __future__ import annotations

from contextlib import contextmanager
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.utils.lora_apply_utils import configure_loras_on_pipeline
from modular_diffusion_nodes_library.utils.pipeline_utils import optimize_diffusion_pipeline

Pipe = ModularPipeline | DiffusionPipeline | Any


def _append_log(log_params: Any | None, message: str) -> None:
    if log_params is None:
        return
    log_params.append_to_logs(message)


@contextmanager
def _profile(log_params: Any | None, label: str):
    if log_params is None:
        yield
        return
    with log_params.append_profile_to_logs(label):
        yield


@runtime_checkable
class BuildStep(Protocol):
    """One step in a pipeline-build sequence."""

    kind: str

    def apply(self, pipe: Pipe | None, *, log_params: Any | None = None) -> Pipe: ...


class LoadPipelineStep:
    """Instantiate a fresh pipeline from build_data."""

    kind = "load"

    def __init__(
        self,
        *,
        builder_module: str | None,
        builder_class_name: str | None,
        build_data: dict[str, Any],
        build_data_error: str | None = None,
    ) -> None:
        self._builder_module = builder_module
        self._builder_class_name = builder_class_name
        self._build_data = build_data
        self._build_data_error = build_data_error

    def apply(self, pipe: Pipe | None, *, log_params: Any | None = None) -> Pipe:
        if pipe is not None:
            raise RuntimeError(
                f"LoadPipelineStep must run first. Failed with pipe of type "
                f"'{type(pipe).__name__}' because a pipe is already in flight."
            )
        if self._build_data_error is not None:
            raise RuntimeError(self._build_data_error)
        if not self._builder_module or not self._builder_class_name:
            raise RuntimeError("Builder module and class name must be specified to build pipeline.")

        _append_log(log_params, "Creating new pipeline instance...\n")
        with _profile(log_params, "Loading pipeline"):
            module = import_module(self._builder_module)
            builder_class = getattr(module, self._builder_class_name)
            return builder_class.build_pipeline_from_build_data(self._build_data)


class FuseLorasStep:
    """Permanently fuse LoRA adapters into the pipeline's weights."""

    kind = "fuse_loras"

    def __init__(self, loras: dict[str, float]) -> None:
        self._loras = dict(loras)

    def apply(self, pipe: Pipe | None, *, log_params: Any | None = None) -> Pipe:
        if pipe is None:
            raise RuntimeError("FuseLorasStep requires a pipe; run LoadPipelineStep first.")
        with _profile(log_params, "Configuring LoRAs"):
            configure_loras_on_pipeline(pipe, self._loras)
        return pipe


class AttachControlNetStep:
    """Wrap the pipeline with the driver's ControlNet variant."""

    kind = "attach_controlnet"

    def __init__(self, *, pipeline_name: str, controlnet_models: list[str]) -> None:
        self._pipeline_name = pipeline_name
        self._controlnet_models = list(controlnet_models)

    def apply(self, pipe: Pipe | None, *, log_params: Any | None = None) -> Pipe:
        # Imported lazily to avoid a import cycle
        from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import (
            get_driver_class,
        )

        if pipe is None:
            raise RuntimeError("AttachControlNetStep requires a pipe.")
        if not self._controlnet_models:
            return pipe
        driver_class = get_driver_class(self._pipeline_name)
        if driver_class is None:
            return pipe
        with _profile(log_params, "Attaching ControlNet"):
            return driver_class.control_pipe_from_standard(pipe, self._controlnet_models)


class ApplyOptimizationStep:
    """Apply offload / casting / device-map optimizations to the pipeline."""

    kind = "optimize"

    def __init__(
        self,
        *,
        optimization_kwargs: dict[str, Any],
        is_prequantized: bool,
        supports_layerwise_casting: bool,
        requires_device_map: bool,
        is_reuse: bool = False,
    ) -> None:
        self._optimization_kwargs = dict(optimization_kwargs)
        # Reused pipelines are treated as prequantized to skip re-quantization.
        self._is_prequantized = is_prequantized or is_reuse
        self._supports_layerwise_casting = supports_layerwise_casting
        self._requires_device_map = requires_device_map
        self._is_reuse = is_reuse

    def apply(self, pipe: Pipe | None, *, log_params: Any | None = None) -> Pipe:
        if pipe is None:
            raise RuntimeError("ApplyOptimizationStep requires a pipe.")
        with _profile(log_params, "Applying optimizations"):
            optimize_diffusion_pipeline(
                pipe=pipe,
                is_prequantized=self._is_prequantized,
                supports_layerwise_casting=self._supports_layerwise_casting,
                requires_device_map=self._requires_device_map,
                **self._optimization_kwargs,
            )
        msg = "Pipeline reuse complete.\n" if self._is_reuse else "Pipeline creation complete.\n"
        _append_log(log_params, msg)
        return pipe


def run_build_steps(
    steps: list[BuildStep],
    *,
    log_params: Any | None = None,
    initial_pipe: Pipe | None = None,
) -> Pipe:
    """Fold `steps` over `initial_pipe` (defaults to None for fresh builds)."""
    if not steps:
        raise RuntimeError("run_build_steps: step list is empty.")

    pipe: Pipe | None = initial_pipe
    for step in steps:
        pipe = step.apply(pipe, log_params=log_params)

    if pipe is None:
        raise RuntimeError("run_build_steps: no step produced a pipeline.")
    return pipe
