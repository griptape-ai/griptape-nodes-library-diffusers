import logging
from typing import Any

import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.traits.widget import Widget

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import normalize_diffusion_pipeline_value
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import (
    estimate_pipeline_memory_from_artifact,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import MEMORY_HEADROOM_FACTOR
from modular_diffusion_nodes_library.utils.torch_utils import to_human_readable_size

logger = logging.getLogger("modular_diffusers_nodes_library")

_MEMORY_WIDGET_LIBRARY = "Griptape Modular Diffusion Nodes Library"
_MEMORY_WIDGET_NAME = "MemoryBreakdown"


def _empty_memory_report(status: str, *, message: str | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "status": status,
        "unit": "bytes",
        "measurement": "estimated_component_memory",
        "pipeline_name": None,
        "offload_mode": None,
        "device": "unknown",
        "total_device_memory_bytes": 0,
        "free_device_memory_bytes": 0,
        "headroom_factor": MEMORY_HEADROOM_FACTOR,
        "headroom_percent": round((MEMORY_HEADROOM_FACTOR - 1) * 100),
        "components": [],
        "total_bytes": 0,
        "estimated_peak_bytes": 0,
        "warnings": [],
    }
    if message is not None:
        report["message"] = message
    return report


def _memory_report_from_estimate(estimate: Any, optimization_summary: str) -> dict[str, Any]:
    components = []
    total_bytes = 0
    for component in estimate.components:
        component_total = max(0, int(component.total_bytes))
        total_bytes += component_total
        components.append(
            {
                "id": component.component_name,
                "label": component.component_name.replace("_", " ").title(),
                "role": component.role,
                "weight_bytes": max(0, int(component.weight_bytes)),
                "activation_bytes": max(0, int(component.activation_bytes)),
                "total_bytes": component_total,
                "is_estimated": component.is_estimated,
                "warning": component.warning,
            }
        )

    device = "unknown"
    total_device_memory_bytes = 0
    free_device_memory_bytes = 0
    if torch.cuda.is_available():
        device = "cuda"
        total_device_memory_bytes = int(torch.cuda.get_device_properties(torch.device("cuda")).total_memory)
        free_device_memory_bytes = max(
            0, total_device_memory_bytes - int(torch.cuda.memory_allocated(torch.device("cuda")))
        )
    elif torch.backends.mps.is_available():
        device = "mps"
        total_device_memory_bytes = int(torch.mps.recommended_max_memory())
        free_device_memory_bytes = max(0, total_device_memory_bytes - int(torch.mps.current_allocated_memory()))

    return {
        "status": "ready",
        "unit": "bytes",
        "measurement": "estimated_component_memory",
        "pipeline_name": estimate.pipeline_name,
        "offload_mode": estimate.offload_mode,
        "device": device,
        "total_device_memory_bytes": total_device_memory_bytes,
        "free_device_memory_bytes": free_device_memory_bytes,
        "optimization_summary": optimization_summary,
        "headroom_factor": MEMORY_HEADROOM_FACTOR,
        "headroom_percent": round((MEMORY_HEADROOM_FACTOR - 1) * 100),
        "components": components,
        "total_bytes": total_bytes,
        "estimated_peak_bytes": max(0, int(estimate.estimated_peak_bytes)),
        "warnings": list(estimate.warnings),
    }


def _format_optimization_summary(optimization_kwargs: dict[str, Any], offload_mode: str | None) -> str:
    optimization_details: list[str] = []
    offload_labels = {
        "model": "model CPU offload",
        "sequential": "sequential CPU offload",
    }
    if offload_mode in offload_labels:
        optimization_details.append(offload_labels[offload_mode])

    quantization_mode = optimization_kwargs.get("quantization_mode")
    if quantization_mode and quantization_mode != "None":
        optimization_details.append(f"{str(quantization_mode).upper()} quantization")
    if optimization_kwargs.get("transformer_layerwise_casting"):
        optimization_details.append("transformer layerwise casting")
    if optimization_kwargs.get("attention_slicing"):
        optimization_details.append("attention slicing")
    if optimization_kwargs.get("vae_slicing"):
        optimization_details.append("VAE slicing")
    if optimization_kwargs.get("vae_tiling"):
        optimization_details.append("VAE tiling")

    if not optimization_details:
        return "No additional memory optimizations"
    return "\n".join(optimization_details)


class PipelineMemoryEstimateNode(SuccessFailureNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip=(
                    "Diffusion pipeline to estimate. Connect from Pipeline Builder. "
                    "This node estimates memory usage from the connected pipeline and latent shape."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="latent",
                input_types=["LatentArtifact"],
                tooltip="Latent whose shape determines the token count / resolution used in the estimate.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="memory_breakdown",
                type="dict",
                output_type="dict",
                default_value=_empty_memory_report("waiting"),
                tooltip="Visual breakdown of estimated component memory and analytical peak memory.",
                allowed_modes={ParameterMode.PROPERTY},
                settable=False,
                traits={Widget(name=_MEMORY_WIDGET_NAME, library=_MEMORY_WIDGET_LIBRARY)},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the memory estimate.",
            result_details_placeholder="Memory estimate details appear here after execution.",
        )

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        if param_name == "pipeline":
            value = normalize_diffusion_pipeline_value(value, node_name=self.name)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

    def process(self) -> AsyncResult | None:
        yield lambda: self._process()

    def _process(self) -> None:
        self._clear_execution_status()
        self.publish_update_to_parameter("memory_breakdown", _empty_memory_report("loading"))

        pipeline_artifact = normalize_diffusion_pipeline_value(
            self.get_parameter_value("pipeline"), node_name=self.name
        )
        if pipeline_artifact is None:
            self.publish_update_to_parameter(
                "memory_breakdown", _empty_memory_report("error", message="Missing required 'pipeline' input.")
            )
            self._set_status_results(was_successful=False, result_details="Missing required 'pipeline' input.")
            return

        latent_artifact = self.get_parameter_value("latent")
        if not isinstance(latent_artifact, LatentArtifact):
            self.publish_update_to_parameter(
                "memory_breakdown", _empty_memory_report("error", message="Missing required 'latent' input.")
            )
            self._set_status_results(was_successful=False, result_details="Missing required 'latent' input.")
            return

        try:
            estimate = estimate_pipeline_memory_from_artifact(pipeline_artifact, latent_artifact)
        except Exception as e:
            logger.exception("%s: Pipeline memory estimation failed", self.name)
            self.publish_update_to_parameter("memory_breakdown", _empty_memory_report("error", message=str(e)))
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        optimization_summary = _format_optimization_summary(
            pipeline_artifact.optimization_kwargs, estimate.offload_mode
        )
        self.publish_update_to_parameter(
            "memory_breakdown", _memory_report_from_estimate(estimate, optimization_summary)
        )
        headroom_percent = round((MEMORY_HEADROOM_FACTOR - 1) * 100)
        peak_memory = to_human_readable_size(estimate.estimated_peak_bytes)

        self._set_status_results(
            was_successful=True,
            result_details=(
                f"Estimated peak: {peak_memory} (includes {headroom_percent}% safety headroom) "
                f"across {len(estimate.components)} components."
            ),
        )
