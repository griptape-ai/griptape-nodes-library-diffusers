from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import diffusers  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ComponentSourceType,
    HFRepoRef,
)


@dataclass(frozen=True)
class SchedulerComponentArtifact(ComponentArtifact):
    """Descriptor for a scheduler component (config-only, no weights).

    The scheduler class is chosen explicitly; its configuration is loaded from a
    local ``scheduler_config.json`` (a file or its containing folder) or from a
    HuggingFace repo's ``scheduler`` subfolder, then applied via
    ``ChosenClass.from_config(...)``. The config's recorded ``_class_name`` does
    not dictate the instantiated class — the chosen class always wins.
    """

    scheduler_class: str = ""
    config_source: str | None = None  # LOCAL_DIR: path to scheduler_config.json or its folder
    repo_ref: HFRepoRef | None = None  # HF_REPO

    @override
    def materialize(self, *, pipeline_cls: type, slot: str | None = None) -> Any:  # noqa: ARG002
        scheduler_cls = self._resolve_scheduler_class()
        config = self._read_config_for_scheduler_class(scheduler_cls)
        return scheduler_cls.from_config(config)

    def read_config(self) -> dict[str, Any]:
        """Load the scheduler config dict from the source. Raises if it cannot be read.

        Raises ``FileNotFoundError`` / ``OSError`` when the source has no
        ``scheduler_config.json``, ``json.JSONDecodeError`` if it is malformed,
        and ``ValueError`` for an unknown scheduler class.
        """
        scheduler_cls = self._resolve_scheduler_class()
        return self._read_config_for_scheduler_class(scheduler_cls)

    def _read_config_for_scheduler_class(self, scheduler_cls: type) -> dict[str, Any]:
        return self._load_config_dict(scheduler_cls)

    def read_config_class_name(self) -> str | None:
        """Return the ``_class_name`` recorded in the source config, or ``None``.

        Best-effort and side-effect-free — used by the loader node to compare the
        config's family against the chosen scheduler class. Never raises.
        """
        try:
            config = self.read_config()
        except (ValueError, FileNotFoundError, OSError, json.JSONDecodeError):
            return None
        name = config.get("_class_name")
        return name if isinstance(name, str) else None

    def _resolve_scheduler_class(self) -> type:
        scheduler_cls = getattr(diffusers, self.scheduler_class, None)
        if not (isinstance(scheduler_cls, type) and issubclass(scheduler_cls, diffusers.SchedulerMixin)):
            msg = f"Failed to resolve scheduler class '{self.scheduler_class}' as a diffusers scheduler class."
            raise ValueError(msg)
        return scheduler_cls

    def _load_config_dict(self, scheduler_cls: type) -> dict[str, Any]:
        if self.source_type == ComponentSourceType.LOCAL_DIR:
            return self._load_local_config()
        if self.source_type == ComponentSourceType.HF_REPO:
            return self._load_hf_config(scheduler_cls)
        msg = f"Unsupported scheduler source type '{self.source_type}' for config loading."
        raise NotImplementedError(msg)

    def _load_local_config(self) -> dict[str, Any]:
        if not self.config_source:
            msg = "Local scheduler config loading requires config_source."
            raise ValueError(msg)

        path = Path(self.config_source)
        if path.is_file():
            config_file = path
        elif path.is_dir():
            config_file = path / "scheduler_config.json"
        else:
            msg = f"Scheduler config source '{self.config_source}' is not an existing file or directory."
            raise FileNotFoundError(msg)

        if not config_file.is_file():
            msg = f"No 'scheduler_config.json' found at scheduler config source '{self.config_source}'."
            raise FileNotFoundError(msg)

        with config_file.open(encoding="utf-8") as f:
            return json.load(f)

    def _load_hf_config(self, scheduler_cls: type) -> dict[str, Any]:
        if self.repo_ref is None:
            msg = (
                "HuggingFace scheduler config loading requires repo_ref, but none was provided "
                f"for scheduler_class='{self.scheduler_class}'."
            )
            raise ValueError(msg)

        config = scheduler_cls.load_config(
            self.repo_ref.repo_id,
            subfolder=self.repo_ref.subfolder or "scheduler",
            revision=self.repo_ref.revision,
            local_files_only=True,
        )
        # load_config returns the config dict by default; unwrap if a tuple slips through.
        if isinstance(config, tuple):
            config = config[0]
        return config
