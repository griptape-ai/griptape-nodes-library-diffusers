from __future__ import annotations

import json
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
            lines.extend([f"  {slot.title()}", f"    Type: {type(override).__name__}", f"    Source: {override.source_type.value}"])
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

    return "\n".join(lines)


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
