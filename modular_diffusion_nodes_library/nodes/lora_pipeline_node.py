import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.log_parameter import LogParameter

from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    DiffusionPipelineArtifact,
    normalize_diffusion_pipeline_value,
)
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec, normalize_loras
from modular_diffusion_nodes_library.utils.pipeline_utils import cleanup_memory_caches

logger = logging.getLogger("modular_diffusers_nodes_library")


class LoraActivationPipelineNode(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="pipeline",
                type="Pipeline Config",
                tooltip=(
                    "Base diffusion pipeline. Connect from Pipeline Builder. The base pipeline is reused, "
                    "not modified — wire it to other nodes simultaneously."
                ),
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="lora_pipeline",
                output_type="Pipeline Config",
                default_value=None,
                tooltip=(
                    "Pipeline reference that activates the listed LoRAs around each generation call. "
                    "Shares the cache entry of the input pipeline — wiring this to a Generate Latent "
                    "node does not trigger a rebuild."
                ),
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "LoRA Pipeline"},
            )
        )

        loras_param = ParameterList(
            name="loras",
            input_types=["loras"],
            default_value=[],
            type="loras",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            tooltip=(
                "One or more LoRA payloads from Load LoRA nodes. Adapters are activated dynamically "
                "at generation time and deactivated afterward; the base pipeline weights are never "
                "permanently modified."
            ),
        )
        loras_param.set_badge(
            variant="help",
            title="Activation vs. fused LoRAs",
            message=(
                "**Activation (this node)** \u2014 adapters are loaded and switched on per generation, "
                "then released. The base pipeline is **not** mutated, so the same cached pipeline can "
                "power multiple branches (e.g. one branch with LoRAs, one without) without rebuilding.\n\n"
                "**Fused** (the `loras` input on the Modular Diffusion Pipeline Builder) \u2014 adapters "
                "are baked permanently into the cached weights. Changing the LoRA set evicts and "
                "rebuilds the entire pipeline.\n\n"
                "Prefer activation for in-context (IC) LoRAs, distillation/acceleration LoRAs, slider "
                "LoRAs, or any workflow that swaps adapters between runs. Prefer fused only when the "
                "same LoRA stack is used for every generation and the per-run activation cost matters."
            ),
        )
        self.add_parameter(loras_param)

        self.log_params = LogParameter(self)
        self.log_params.add_output_parameters()
        self.set_lora_pipeline_artifact()

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        parameter = self.get_parameter_by_name(param_name)
        if parameter is None:
            return

        if parameter.name in ("pipeline", "lora_pipeline"):
            value = normalize_diffusion_pipeline_value(value, node_name=self.name)

        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if parameter.name != "lora_pipeline":
            self.set_lora_pipeline_artifact()

    def set_lora_pipeline_artifact(self) -> None:
        lora_artifact = self.build_pipeline_artifact()
        if lora_artifact is None:
            self.set_parameter_value("lora_pipeline", None)
            self.parameter_output_values["lora_pipeline"] = None
            return

        self.log_params.append_to_logs(f"Pipeline configuration hash: {lora_artifact.config_hash}\n")
        self.set_parameter_value("lora_pipeline", lora_artifact)
        self.parameter_output_values["lora_pipeline"] = lora_artifact

    def validate_before_node_run(self) -> list[Exception] | None:
        input_artifact = self._get_input_pipeline_artifact()
        if input_artifact is None:
            return [ValueError(f"{self.name}: Missing required 'pipeline' input.")]

        try:
            loras = self._get_loras()
        except (TypeError, ValueError) as err:
            return [ValueError(f"{self.name}: Invalid 'loras' input. {err}")]

        if not loras:
            return [ValueError(f"{self.name}: Missing required 'loras' input. Connect one or more Load LoRA nodes.")]

        return None

    def preprocess(self) -> None:
        self.log_params.clear_logs()

    def _get_input_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        input_pipeline_artefact = normalize_diffusion_pipeline_value(
            self.get_parameter_value("pipeline"), node_name=self.name
        )
        if input_pipeline_artefact is None:
            return None
        return input_pipeline_artefact

    def build_pipeline_artifact(self) -> DiffusionPipelineArtifact | None:
        base_artifact = self._get_input_pipeline_artifact()
        if base_artifact is None:
            return None

        try:
            loras = self._get_loras()
        except (TypeError, ValueError):
            return None
        if not loras:
            return None

        return base_artifact.with_additional_runtime_adapter_steps([LoraPipelineRuntimeAdapterStep(loras)])

    def _get_loras(self) -> dict[str, LoraSpec]:
        loras_list = self.get_parameter_value("loras") or []
        return normalize_loras(loras_list)

    def process(self) -> AsyncResult:
        self.preprocess()
        self.log_params.append_to_logs("Building pipeline...\n")

        def work() -> Any:
            return self._build_pipeline()

        yield work
        self.log_params.append_to_logs("Pipeline building complete.\n")

    def _build_pipeline(self) -> Any:
        self.set_lora_pipeline_artifact()
        pipeline_artifact = self.get_parameter_value("lora_pipeline")
        if pipeline_artifact is None:
            raise ValueError(f"{self.name}: Failed to build lora pipeline artifact.")

        try:
            with self.log_params.append_profile_to_logs("Pipeline building/caching"):
                return pipeline_artifact.get_or_build_pipeline(log_params=self.log_params)
        except Exception:
            logger.exception("%s: Diffusion Pipeline build failed", self.name)
            model_cache.remove_pipeline(pipeline_artifact.config_hash)
            cleanup_memory_caches()
            raise
