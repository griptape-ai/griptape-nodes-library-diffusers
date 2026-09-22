from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huggingface_hub import scan_cache_dir  # pyright: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils import (
    scheduler_component_artifact as _scheduler_component_artifact,  # noqa: F401  (registers deserializer)
)
from modular_diffusion_nodes_library.artifact_utils import (
    text_encoder_component_artifact as _text_encoder_component_artifact,  # noqa: F401
)
from modular_diffusion_nodes_library.artifact_utils import (
    tokenizer_component_artifact as _tokenizer_component_artifact,  # noqa: F401
)
from modular_diffusion_nodes_library.artifact_utils.component_artifact import ComponentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    DEFAULT_BAKED_DTYPE,
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.recipe_codec import (
    PIPELINE_RECIPE_SCHEMA_VERSION,
    require_dict,
)


def serialize_pipeline_artifact(artifact: DiffusionPipelineArtifact) -> str:
    """Serialize a supported pipeline artifact to a versioned JSON recipe."""
    recipe = {
        "schema_version": PIPELINE_RECIPE_SCHEMA_VERSION,
        "artifact": artifact.to_recipe_dict(),
    }
    return json.dumps(recipe, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def deserialize_pipeline_artifact(recipe_json: str) -> DiffusionPipelineArtifact:
    """Deserialize a supported versioned pipeline recipe."""
    try:
        document = json.loads(recipe_json)
    except json.JSONDecodeError as error:
        msg = f"Invalid pipeline recipe JSON: {error.msg}"
        raise ValueError(msg) from error

    document_data = require_dict(document, "Pipeline recipe")
    schema_version = document_data.get("schema_version")
    if schema_version != PIPELINE_RECIPE_SCHEMA_VERSION:
        msg = f"Unsupported pipeline recipe schema version: {schema_version!r}"
        raise ValueError(msg)

    return DiffusionPipelineArtifact.from_recipe_dict(
        require_dict(document_data.get("artifact"), "Pipeline recipe artifact")
    )


def validate_pipeline_recipe_dependencies(artifact: DiffusionPipelineArtifact) -> list[str]:
    """Return local files and pinned Hugging Face revisions unavailable for an artifact."""
    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        issues = []
        for repo_id in artifact.controlnet_models:
            if not _list_cached_repo_revisions(repo_id):
                issues.append(f"ControlNet repository is unavailable locally: {repo_id}")
        return [*issues, *validate_pipeline_recipe_dependencies(artifact.base_artifact)]

    issues = []
    for lora_path in artifact.loras:
        if not Path(lora_path).is_file():
            issues.append(f"LoRA file is unavailable: {lora_path}")

    return [*issues, *_validate_build_data_repositories(artifact.build_data)]


@dataclass(frozen=True)
class QuantizationStatus:
    """Recorded quant types for Load Pipeline status.

    `slots` is slot name → type when we know which components are quantized.
    `pipeline` is a single type when we only know it at pipeline level.
    """

    slots: dict[str, str] = field(default_factory=dict)
    pipeline: str | None = None


def collect_quantization_status(artifact: DiffusionPipelineArtifact) -> QuantizationStatus:
    """Read GGUF overrides, baked packed slots, pre-quantized Hub repos, or recipe quanto mode."""
    base = artifact.base_artifact if isinstance(artifact, ControlNetDiffusionPipelineArtifact) else artifact
    build_data = base.build_data
    slots: dict[str, str] = {}

    overrides = build_data.get("_component_overrides")
    if isinstance(overrides, dict):
        for slot, override in overrides.items():
            label = _override_quant_label(override)
            if label is not None:
                slots[str(slot)] = label

    packed = build_data.get("_packed_quant_slots") or []
    baked_mode = build_data.get("_baked_quantization_mode")
    if isinstance(packed, list) and packed:
        mode = baked_mode if isinstance(baked_mode, str) and baked_mode else "quanto"
        for slot in packed:
            if isinstance(slot, str) and slot not in slots:
                slots[slot] = mode

    if slots:
        return QuantizationStatus(slots=slots)

    if base.is_prequantized:
        label = _hub_prequant_label(build_data)
        if label is not None:
            return QuantizationStatus(pipeline=label)
        return QuantizationStatus()

    mode = base.optimization_kwargs.get("quantization_mode")
    if isinstance(mode, str) and mode not in {"", "None"}:
        return QuantizationStatus(pipeline=mode)
    return QuantizationStatus()


def format_quantization_status_lines(status: QuantizationStatus) -> list[str]:
    """One line when every slot shares a type; per slot when they differ."""
    if status.slots:
        types = set(status.slots.values())
        if len(types) == 1:
            qtype = next(iter(types))
            names = ", ".join(sorted(status.slots))
            return [f"Quantization: {qtype} ({names})"]
        return ["Quantization", *[f"  {slot}: {status.slots[slot]}" for slot in sorted(status.slots)]]
    if status.pipeline:
        return [f"Quantization: {status.pipeline}"]
    return []


def format_baked_pipeline_summary(artifact: DiffusionPipelineArtifact) -> str:
    """Load Pipeline status for a baked folder."""
    build_data = artifact.build_data
    lines = [
        "Loaded baked pipeline",
        "",
        f"Type: {artifact.pipeline_name}",
        f"Path: {build_data.get('_baked_path', '')}",
        f"Dtype: {build_data.get('_baked_dtype', DEFAULT_BAKED_DTYPE)}",
    ]
    quant_lines = format_quantization_status_lines(collect_quantization_status(artifact))
    if quant_lines:
        lines.extend(quant_lines)
    return "\n".join(lines)


def format_pipeline_artifact_summary(artifact: DiffusionPipelineArtifact) -> str:
    """Format the persisted pipeline configuration for a Load Pipeline status message."""
    base_artifact = artifact.base_artifact if isinstance(artifact, ControlNetDiffusionPipelineArtifact) else artifact
    build_data = base_artifact.build_data
    repo_id = build_data.get("base_repo_id") or build_data.get("repo_id") or "Unknown"
    revision = build_data.get("base_revision") or build_data.get("revision") or "Unpinned"
    lines = ["Pipeline", f"  Type: {base_artifact.pipeline_name}", f"  Model: {repo_id}", f"  Revision: {revision}"]

    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        lines.extend(["", "ControlNets"])
        lines.extend(f"  {index}. {model}" for index, model in enumerate(artifact.controlnet_models, start=1))

    if base_artifact.loras:
        lines.extend(["", "LoRAs"])
        for index, (path, weight) in enumerate(base_artifact.loras.items(), start=1):
            lines.extend([f"  {index}. {Path(path).name}", f"     Path: {path}", f"     Weight: {weight:g}"])

    component_overrides = build_data.get("_component_overrides", {})
    if isinstance(component_overrides, dict) and component_overrides:
        lines.extend(["", "Component Overrides"])
        for slot, override in component_overrides.items():
            if not isinstance(override, ComponentArtifact):
                lines.append(f"  {slot}: Unsupported override configuration")
                continue
            lines.extend(
                [
                    f"  {slot.title()}",
                    f"    Type: {type(override).__name__}",
                    f"    Source: {override.source_type.value}",
                ]
            )
            if override.repo_ref is not None:
                lines.append(f"    Model: {override.repo_ref.repo_id}")
                if override.repo_ref.revision:
                    lines.append(f"    Revision: {override.repo_ref.revision}")
                if override.repo_ref.subfolder:
                    lines.append(f"    Subfolder: {override.repo_ref.subfolder}")
            else:
                lines.append(f"    Path: {override.file_path or override.config_source or 'Unknown'}")
            if override.torch_dtype:
                lines.append(f"    Dtype: {override.torch_dtype}")

    quant_lines = format_quantization_status_lines(collect_quantization_status(artifact))
    if quant_lines:
        lines.extend(["", *quant_lines])

    return "\n".join(lines)


def _override_quant_label(override: Any) -> str | None:
    if not getattr(override, "is_quantized", False):
        return None
    path = getattr(override, "file_path", None) or ""
    if str(path).lower().endswith(".gguf"):
        return "GGUF"
    return None


def _hub_prequant_label(build_data: dict[str, Any]) -> str | None:
    repo = str(build_data.get("base_repo_id") or build_data.get("repo_id") or "")
    lower = repo.lower()
    if "nvfp4" in lower:
        return "fp4"
    return None


def _validate_build_data_repositories(build_data: dict[str, Any]) -> list[str]:
    issues = []
    for key, value in build_data.items():
        if not key.endswith("repo_id") or not isinstance(value, str):
            continue

        revisions = _list_cached_repo_revisions(value)
        if not revisions:
            issues.append(f"Repository is unavailable locally: {value}")
            continue

        revision_key = key.removesuffix("repo_id") + "revision"
        revision = build_data.get(revision_key)
        if revision is None and key in {"repo_id", "base_repo_id"}:
            revision = build_data.get("revision")
        if isinstance(revision, str) and revision:
            cached_revisions = {cached_revision for _, cached_revision in revisions}
            if revision not in cached_revisions:
                issues.append(f"Repository revision is unavailable locally: {value} ({revision})")
    return issues


def _list_cached_repo_revisions(repo_id: str) -> list[tuple[str, str]]:
    # Use scan_cache_dir() so we get every cached revision for repo_id.
    # list_repo_revisions_in_cache() / list_all_repo_revisions_in_cache() in
    # griptape_nodes...huggingface_utils use quick_scan_diffuser_repos(), which
    # returns one snapshot hash per repo — enough for presence checks, not for
    # validating a specific pinned revision in _validate_build_data_repositories().
    cache_info = scan_cache_dir()
    for repo in cache_info.repos:
        if repo.repo_id == repo_id:
            return [(repo_id, revision.commit_hash) for revision in repo.revisions]
    return []
