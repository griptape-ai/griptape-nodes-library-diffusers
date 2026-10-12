from __future__ import annotations

from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact
from modular_diffusion_nodes_library.memory_estimation.lora_memory import collect_runtime_lora_memory
from modular_diffusion_nodes_library.memory_estimation.pipeline_memory_estimator import estimate_pipeline_memory
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec


def _artifact_with_loras(paths: list[str]) -> DiffusionPipelineArtifact:
    loras = {path: LoraSpec(path=path, weight=0.5) for path in paths}
    artifact = DiffusionPipelineArtifact(pipeline_name="TestPipeline")
    return artifact.with_additional_runtime_adapter_steps([LoraPipelineRuntimeAdapterStep(loras)])


def _latent() -> LatentArtifact:
    return LatentArtifact.from_torch(torch.zeros(1, 4, 8, 8), source_shape=(1, 3, 64, 64))


def test_collect_runtime_lora_memory_reads_header_without_loading_tensor_data(tmp_path) -> None:
    path = tmp_path / "style.safetensors"
    save_file(
        {
            "lora_A.weight": torch.zeros((2, 3), dtype=torch.float16),
            "lora_B.weight": torch.zeros((4, 2), dtype=torch.float32),
        },
        str(path),
    )

    estimates = collect_runtime_lora_memory(_artifact_with_loras([str(path)]))

    assert len(estimates) == 1
    assert estimates[0].weight_bytes == (2 * 3 * 2) + (4 * 2 * 4)
    assert estimates[0].is_estimated is False
    assert estimates[0].warning is None


def test_collect_runtime_lora_memory_deduplicates_chained_steps(tmp_path) -> None:
    path = tmp_path / "style.safetensors"
    save_file({"weight": torch.zeros((4,), dtype=torch.float16)}, str(path))
    artifact = _artifact_with_loras([str(path)])
    artifact = artifact.with_additional_runtime_adapter_steps(
        [LoraPipelineRuntimeAdapterStep({str(path): LoraSpec(path=str(path), weight=1.0)})]
    )

    estimates = collect_runtime_lora_memory(artifact)

    assert len(estimates) == 1
    assert estimates[0].weight_bytes == 8


def test_estimate_merges_multiple_runtime_loras_into_transformer(tmp_path) -> None:
    path_a = tmp_path / "style.safetensors"
    path_b = tmp_path / "character.safetensors"
    save_file({"weight": torch.zeros((4,), dtype=torch.float16)}, str(path_a))
    save_file({"weight": torch.zeros((8,), dtype=torch.float16)}, str(path_b))
    artifact = _artifact_with_loras([str(path_a), str(path_b)])
    lora_adapters = collect_runtime_lora_memory(artifact)
    pipe = SimpleNamespace(components={"transformer": torch.nn.Linear(4, 4, bias=False)})

    estimate = estimate_pipeline_memory(pipe, _latent(), {}, "SomeUnregisteredPipeline", lora_adapters=lora_adapters)

    transformer = estimate.components[0]
    base_bytes = 4 * 4 * 4
    assert transformer.weight_bytes == base_bytes + 8 + 16
    assert transformer.tooltip is not None
    assert "of base transformer weights" in transformer.tooltip
    assert "of combined transformer weights" in transformer.tooltip
    assert "bytes" not in transformer.tooltip
    assert len(estimate.lora_adapters) == 2
    assert "lora_adapter" not in {component.role for component in estimate.components}


def test_missing_lora_file_is_reported_without_failing_estimate(tmp_path) -> None:
    path = tmp_path / "missing.safetensors"
    estimates = collect_runtime_lora_memory(_artifact_with_loras([str(path)]))

    assert len(estimates) == 1
    assert estimates[0].weight_bytes == 0
    assert estimates[0].is_estimated
    assert "Attempted to read LoRA weights" in estimates[0].warning
