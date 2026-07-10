from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from typing import Any

import torch  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class HFRepoRef:
    repo_id: str
    revision: str | None = None
    subfolder: str | None = None


class ComponentSourceType(StrEnum):
    HF_REPO = "hf_repo"
    SINGLE_FILE = "single_file"
    LOCAL_DIR = "local_dir"


@dataclass(frozen=True)
class ComponentArtifact:
    """Lazy descriptor for a single pipeline component.

    Produced by component loader nodes, consumed by the pipeline builder
    at build time. Does not hold the loaded component itself.
    """

    load_id: str

    source_type: ComponentSourceType

    repo_ref: HFRepoRef | None = None
    file_path: str | None = None

    config_ref: HFRepoRef | None = None
    config_local_path: str | None = None

    component_class: str | None = None
    torch_dtype: str | None = None

    def materialize(self) -> Any:
        """Load this component from its descriptor.

        Currently supports HF_REPO source type only.
        """
        if self.source_type != ComponentSourceType.HF_REPO:
            msg = (
                f"Attempted to materialize component '{self.load_id}'. "
                f"Failed because source type '{self.source_type}' is not yet supported."
            )
            raise NotImplementedError(msg)

        if not self.component_class:
            msg = (
                f"Attempted to materialize component '{self.load_id}'. Failed because component_class is not specified."
            )
            raise ValueError(msg)

        if not self.repo_ref:
            msg = (
                f"Attempted to materialize component '{self.load_id}'. "
                f"Failed because repo_ref is required for HF_REPO source type."
            )
            raise ValueError(msg)

        module_name, class_name = self.component_class.rsplit(".", 1)
        module = import_module(module_name)
        component_cls = getattr(module, class_name)

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.repo_ref.repo_id,
            "local_files_only": True,
        }
        if self.repo_ref.revision is not None:
            kwargs["revision"] = self.repo_ref.revision
        if self.repo_ref.subfolder is not None:
            kwargs["subfolder"] = self.repo_ref.subfolder
        if self.torch_dtype is not None:
            kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        return component_cls.from_pretrained(**kwargs)
