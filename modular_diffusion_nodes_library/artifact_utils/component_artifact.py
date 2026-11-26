from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path
from typing import Any, override

from griptape_nodes.retained_mode.events.event_converter import converter

from modular_diffusion_nodes_library.component_loading.component_slots import component_config_filename
from modular_diffusion_nodes_library.component_loading.config_guidance import (
    config_mismatch_message,
    missing_config_message,
)
from modular_diffusion_nodes_library.component_loading.config_resolver import (
    ConfigLoadError,
    loadable_class_name,
    read_cached_config,
    resolve_config_path,
    resolve_hf_repo_config_subfolder,
    try_load_json_dict,
)
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import (
    MODEL_TYPE_TO_PIPELINE_TYPE,
    default_repo_for_model_type,
    get_component_class,
    get_loadable_component_class,
)

logger = logging.getLogger("modular_diffusers_nodes_library")

#: Names the concrete subclass in the wire form. Present in the wire form only.
_COMPONENT_TAG = "__component_artifact__"

#: Every subclass by name, so `structure_component_artifact` can rebuild the one a wire form names.
#: Populated by `__init_subclass__`, so a subclass in a sibling module registers itself on import.
_COMPONENT_CLASSES: dict[str, type[ComponentArtifact]] = {}


def is_encoded_component_artifact(value: Any) -> bool:
    """Whether `value` is a component artifact that crossed a process boundary and needs rebuilding."""
    return isinstance(value, dict) and _COMPONENT_TAG in value


def structure_component_artifact(data: dict[str, Any]) -> ComponentArtifact:
    """Rebuild the concrete `ComponentArtifact` subclass named by `data`'s tag.

    Naming the class rather than letting the converter choose between the subclasses is what makes
    this work: they are indistinguishable by shape, since every field of every one of them has a
    default and two of them share `config_source` and `repo_ref`.

    Raises:
        ValueError: if the tag names a class absent from the registry. Registration is an import side
            effect, so the likely cause is that this process never imported the module declaring that
            subclass, rather than a version skew between the two ends.
    """
    tag = data.get(_COMPONENT_TAG)
    artifact_cls = _COMPONENT_CLASSES.get(tag) if isinstance(tag, str) else None
    if artifact_cls is None:
        msg = (
            f"Attempted to rebuild a component artifact. Failed with tag='{tag}' because no component "
            f"artifact class declares it. Known tags: {sorted(_COMPONENT_CLASSES)}."
        )
        raise ValueError(msg)
    return converter.structure({k: v for k, v in data.items() if k != _COMPONENT_TAG}, artifact_cls)


def _pipeline_model_types(pipeline_cls: type) -> list[str]:
    """Return the ``model_type`` strings registered for ``pipeline_cls``."""
    return [
        model_type
        for model_type, pipeline_type in MODEL_TYPE_TO_PIPELINE_TYPE.items()
        if pipeline_type == pipeline_cls.__name__
    ]


@dataclass(frozen=True)
class HFRepoRef:
    repo_id: str
    revision: str | None = None
    subfolder: str | None = None


def base_repo_ref(build_data: dict[str, Any]) -> HFRepoRef | None:
    """The repo a pipeline is built from, or ``None``. The one reader of it in ``build_data``."""
    base_repo_id = build_data.get("base_repo_id") or build_data.get("repo_id")
    if not base_repo_id:
        return None
    return HFRepoRef(repo_id=base_repo_id, revision=build_data.get("base_revision") or build_data.get("revision"))


@dataclass(frozen=True)
class MaterializeContext:
    """Where a component's config can come from, when it is loaded from a single file."""

    #: The repo the pipeline is built from. A single file without a ``config_source`` takes its config from it.
    base_repo: HFRepoRef | None = None
    #: The repos Build Pipeline offers for the pipeline class. One of them holds a config for a single file.
    config_repos: list[str] = field(default_factory=list)


class ComponentSourceType(StrEnum):
    HF_REPO = "hf_repo"
    SINGLE_FILE = "single_file"
    LOCAL_DIR = "local_dir"
    RAW_CONFIG = "raw_config"


@dataclass(frozen=True)
class ComponentArtifact(ABC):
    """Descriptor for a single pipeline component.

    Produced by component loader nodes, consumed by the Build Pipeline node
    at build time. Does not hold the loaded component itself.

    Subclasses specialize by component kind (model weights, tokenizer,
    text encoder, ...).
    """

    load_id: str
    source_type: ComponentSourceType
    component: str  # slot name, e.g. "transformer", "vae", "tokenizer"
    torch_dtype: str = "bfloat16"

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _COMPONENT_CLASSES[cls.__name__] = cls

    def _cattrs_unstructure(self) -> dict[str, Any]:
        """The wire form: the concrete subclass's name plus every field it declares.

        The engine's documented seam for a class that knows its own wire form. Tagging here rather
        than registering a converter hook means the tag is present however the value gets encoded,
        including when the engine serializes this as a bare parameter value having called nothing in
        this library first.
        """
        return {
            _COMPONENT_TAG: type(self).__name__,
            **{field.name: converter.unstructure(getattr(self, field.name)) for field in fields(self)},
        }

    @property
    def is_quantized(self) -> bool:
        """True if the underlying weights use an embedded quantization format."""
        return False

    @abstractmethod
    def materialize(
        self, *, pipeline_cls: type, slot: str | None = None, context: MaterializeContext = MaterializeContext()
    ) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class ModelComponentArtifact(ComponentArtifact):
    """Descriptor for a Transformer, UNet, or VAE component loaded via diffusers."""

    # HF_REPO
    repo_ref: HFRepoRef | None = None

    # SINGLE_FILE / LOCAL_DIR
    file_path: str | None = None
    config_source: str | None = None  # local path OR HF repo_id

    @property
    @override
    def is_quantized(self) -> bool:
        """True if the weights use an embedded quantization format (e.g. GGUF)."""
        return self.file_path is not None and self.file_path.lower().endswith(".gguf")

    @override
    def materialize(
        self, *, pipeline_cls: type, slot: str | None = None, context: MaterializeContext = MaterializeContext()
    ) -> Any:
        """Load this component from its descriptor.

        ``pipeline_cls`` is the diffusers pipeline class (e.g. ``FluxPipeline``)
        used to derive the concrete component class and to validate that the weights
        are compatible with the target pipeline.

        ``slot`` is the actual pipeline slot being filled (e.g. ``"text_encoder_2"``).
        When provided it takes precedence over ``self.component`` for class lookup,
        so an artifact created for one slot can be correctly loaded into another
        (e.g. a generic text-encoder artifact wired to the ``text_encoder_2`` port).

        ``context`` carries the repos a single file's config can come from.
        """
        effective_slot = slot if slot is not None else self.component
        component_cls = get_loadable_component_class(pipeline_cls, effective_slot)
        try:
            if self.source_type == ComponentSourceType.HF_REPO:
                return self._materialize_hf_repo(pipeline_cls=pipeline_cls, effective_slot=effective_slot)
            if self.source_type == ComponentSourceType.SINGLE_FILE:
                return self._materialize_single_file(
                    pipeline_cls=pipeline_cls, effective_slot=effective_slot, context=context
                )
            if self.source_type == ComponentSourceType.LOCAL_DIR:
                return self._materialize_local_dir(pipeline_cls=pipeline_cls, effective_slot=effective_slot)

            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because source type '{self.source_type}' is not supported."
            )
            raise NotImplementedError(msg)
        except ConfigLoadError:
            # Already says why and how to fix it; wrapping it again only buries that.
            raise
        except Exception as e:
            source_info = self._describe_source()
            msg = (
                f"Failed to load {self.component} as {component_cls.__name__} from {source_info}. Original error: {e!s}"
            )
            # The traceback is dropped from the raised error to keep the node's message short, so keep it in the log.
            logger.exception("Failed to load %s as %s from %s.", self.component, component_cls.__name__, source_info)
            raise RuntimeError(msg) from None

    def _describe_source(self) -> str:
        """Return a human-readable description of this component's source."""
        if self.source_type == ComponentSourceType.HF_REPO and self.repo_ref:
            parts = [f"HF repo '{self.repo_ref.repo_id}'"]
            if self.repo_ref.subfolder:
                parts.append(f"subfolder '{self.repo_ref.subfolder}'")
            if self.repo_ref.revision:
                parts.append(f"revision '{self.repo_ref.revision}'")
            return ", ".join(parts)
        if self.source_type == ComponentSourceType.SINGLE_FILE and self.file_path:
            return f"single file '{self.file_path}'"
        if self.source_type == ComponentSourceType.LOCAL_DIR and self.file_path:
            return f"local directory '{self.file_path}'"
        return f"{self.source_type} (details unavailable)"

    def try_read_config(self, base_repo: HFRepoRef | None = None) -> dict[str, Any] | None:
        """Read this artifact's config file without materializing weights.

        Reads only what is already on disk. Deriving a config path from a component class, which is what
        a single-file checkpoint's HF-repo `config_source` would need, requires diffusers -- and the only
        caller is the builder's compatibility check, which runs on the orchestrator.

        ``base_repo`` is the repo the pipeline is built from. A single file with no ``config_source`` reads its
        config from it, as ``materialize`` does. The canonical repo for the checkpoint's model type is not tried
        here: naming it needs the checkpoint read, which this method avoids.
        """
        from huggingface_hub import try_to_load_from_cache

        config_name = component_config_filename(self.component)
        if self.source_type == ComponentSourceType.HF_REPO:
            if self.repo_ref is None:
                return None
            subfolder = self.repo_ref.subfolder or ""
            if not subfolder:
                subfolder = (
                    resolve_hf_repo_config_subfolder(
                        self.repo_ref.repo_id,
                        self.component,
                        self.component,
                        revision=self.repo_ref.revision,
                        config_filename=config_name,
                    )
                    or ""
                )
            filename = f"{subfolder}/{config_name}" if subfolder else config_name
            cached = try_to_load_from_cache(self.repo_ref.repo_id, filename, revision=self.repo_ref.revision)
            if not isinstance(cached, str):
                return None
            return try_load_json_dict(Path(cached))

        if self.source_type == ComponentSourceType.LOCAL_DIR:
            if self.file_path is None:
                return None
            return try_load_json_dict(Path(self.file_path) / config_name)

        if self.source_type == ComponentSourceType.SINGLE_FILE:
            return self._try_read_single_file_config(base_repo)

        return None

    def _infer_single_file_model_type(self, *, pipeline_cls: type, checkpoint: Any) -> tuple[str, str]:
        """Infer the checkpoint model type and the effective config lookup model type.

        The guess from the key names picks the variant. When it names a type this pipeline does not register:

        - A pipeline with one type takes that type.
        - A pipeline with several types keeps the guess if the library knows it, otherwise takes the pipeline's first type.
        """
        from diffusers.loaders.single_file_utils import (  # type: ignore[reportMissingImports]
            infer_diffusers_model_type,
        )

        inferred_model_type = infer_diffusers_model_type(checkpoint)
        pipeline_model_types = _pipeline_model_types(pipeline_cls)
        if inferred_model_type in pipeline_model_types or not pipeline_model_types:
            model_type = inferred_model_type
        elif len(pipeline_model_types) == 1 or inferred_model_type not in MODEL_TYPE_TO_PIPELINE_TYPE:
            model_type = pipeline_model_types[0]
        else:
            model_type = inferred_model_type
        return inferred_model_type, model_type

    def _try_read_single_file_config(self, base_repo: HFRepoRef | None) -> dict[str, Any] | None:
        """Read config for a single-file artifact from an explicit config source, or else the base repo's cache.

        A `config_source` naming an HF repo rather than a local path reads as no config: resolving it
        needs the component class, and `materialize` resolves it for real at build time anyway.
        """
        if self.config_source is None:
            if base_repo is None:
                return None
            return read_cached_config(base_repo.repo_id, self.component, self.component, base_repo.revision)

        config_name = component_config_filename(self.component)
        config_path = Path(self.config_source)
        if config_path.is_file():
            return try_load_json_dict(config_path)
        if config_path.is_dir():
            return try_load_json_dict(config_path / config_name)
        return None

    def _materialize_hf_repo(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        import torch  # type: ignore[reportMissingImports]

        if not self.repo_ref:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because repo_ref is required for HF_REPO source type."
            )
            raise ValueError(msg)

        component_cls = get_loadable_component_class(pipeline_cls, effective_slot)

        subfolder = self.repo_ref.subfolder or ""
        if not subfolder:
            subfolder = (
                resolve_hf_repo_config_subfolder(
                    self.repo_ref.repo_id,
                    effective_slot,
                    self.component,
                    revision=self.repo_ref.revision,
                )
                or ""
            )

        kwargs: dict[str, Any] = {
            "pretrained_model_name_or_path": self.repo_ref.repo_id,
            "local_files_only": True,
        }
        if self.repo_ref.revision:
            kwargs["revision"] = self.repo_ref.revision
        if subfolder:
            kwargs["subfolder"] = subfolder
        kwargs["torch_dtype"] = getattr(torch, self.torch_dtype)

        logger.info(
            "Materializing %s (%s) from HF_REPO repo='%s' subfolder='%s' revision='%s'.",
            self.component,
            component_cls.__name__,
            self.repo_ref.repo_id,
            subfolder,
            self.repo_ref.revision,
        )
        return component_cls.from_pretrained(**kwargs)

    def _materialize_single_file(self, *, pipeline_cls: type, effective_slot: str, context: MaterializeContext) -> Any:
        import torch  # type: ignore[reportMissingImports]
        from diffusers import GGUFQuantizationConfig  # type: ignore[reportMissingImports]
        from diffusers.loaders.single_file_utils import (  # type: ignore[reportMissingImports]
            load_single_file_checkpoint,
        )

        config_repo = context.base_repo
        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because file_path is required for SINGLE_FILE source type."
            )
            raise ValueError(msg)

        component_cls = get_component_class(pipeline_cls, effective_slot)

        # Cannot use diffusers' from_single_file path.
        loadable_name = loadable_class_name(component_cls)
        if loadable_name is None:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with component_cls='{component_cls.__name__}' pipeline_cls='{pipeline_cls.__name__}' because "
                f"none of its base classes are registered in SINGLE_FILE_LOADABLE_CLASSES."
            )
            raise ValueError(msg)

        checkpoint = load_single_file_checkpoint(self.file_path)
        inferred_model_type, model_type = self._infer_single_file_model_type(
            pipeline_cls=pipeline_cls,
            checkpoint=checkpoint,
        )

        logger.info(
            "Inferred model_type='%s' (effective='%s') for %s (%s) from single file '%s'.",
            inferred_model_type,
            model_type,
            self.component,
            component_cls.__name__,
            self.file_path,
        )

        kwargs: dict[str, Any] = {
            "pretrained_model_link_or_path_or_dict": checkpoint,
            "torch_dtype": getattr(torch, self.torch_dtype),
            "local_files_only": True,
        }
        if self.is_quantized:
            kwargs["quantization_config"] = GGUFQuantizationConfig(compute_dtype=getattr(torch, self.torch_dtype))

        # Load with a config if the model type is known, or the user or the builder supplies one.
        if model_type in MODEL_TYPE_TO_PIPELINE_TYPE or self.config_source or config_repo is not None:
            expected_pipeline_type = MODEL_TYPE_TO_PIPELINE_TYPE.get(inferred_model_type)
            if expected_pipeline_type is not None and expected_pipeline_type != pipeline_cls.__name__:
                logger.warning(
                    "%s: checkpoint model_type='%s' maps to pipeline '%s' "
                    "but builder requested pipeline_type='%s'. Proceeding with builder's "
                    "pipeline_type — ensure weights are compatible.",
                    self.component,
                    inferred_model_type,
                    expected_pipeline_type,
                    pipeline_cls.__name__,
                )

            base_repo_id = None
            base_revision = None
            if config_repo is not None:
                base_repo_id = config_repo.repo_id
                base_revision = config_repo.revision

            config = resolve_config_path(
                model_type,
                self.config_source,
                pipeline_slot=effective_slot,
                artifact_component=self.component,
                base_repo_id=base_repo_id,
                base_revision=base_revision,
            )
            checkpoint_keys = list(checkpoint)  # diffusers' converters empty `checkpoint`, so read them first
            try:
                component = component_cls.from_single_file(config=str(config.path), **kwargs)
            except (NotImplementedError, ValueError) as e:
                # Every failure here is about the file or the config: the arguments diffusers validates at this
                # stage are the ones this code always passes. A meta tensor means part of the model was never filled.
                part_unfilled = isinstance(e, NotImplementedError) and "meta tensor" in str(e)
                # exc_info keeps the traceback in the log: the message the user sees does not carry it.
                logger.warning(
                    "%s: diffusers reported: %s (pipeline_cls=%s, model_type='%s', config looked up as '%s')",
                    component_cls.__name__,
                    e,
                    pipeline_cls.__name__,
                    inferred_model_type,
                    model_type,
                    exc_info=True,
                )
                # The user sees the first line of diffusers' message. The full text is in the log above.
                details = str(e).split("\n", 1)[0]
                if part_unfilled:
                    key_comparison = self._compare_keys_with_model(
                        component_cls=component_cls,
                        loadable_name=loadable_name,
                        config_path=config.path,
                        checkpoint_keys=checkpoint_keys,
                    )
                    if key_comparison is not None:
                        details = f"{details} {key_comparison}"
                logger.warning("%s: weights '%s' with config '%s'", self.component, self.file_path, config.path)
                # The config was looked up under `model_type`, so the repo to suggest is the one for that type.
                candidate_repos = [default_repo_for_model_type(model_type)]
                candidate_repos += context.config_repos
                suggested_repos = list(dict.fromkeys(r for r in candidate_repos if r and r != config.repo_id))
                msg = config_mismatch_message(
                    component=self.component,
                    file_path=self.file_path,
                    slot=effective_slot,
                    config=config,
                    suggested_repos=suggested_repos,
                    details=details,
                    likely_cause_only=not part_unfilled,
                )
                raise ConfigLoadError(msg) from None
            kwargs.pop("pretrained_model_link_or_path_or_dict", None)
            del checkpoint
            return component

        msg = missing_config_message(
            pipeline_slot=effective_slot,
            tried_repos=[],
            example_repos=context.config_repos,
        )
        raise ConfigLoadError(msg)

    @staticmethod
    def _compare_keys_with_model(
        *, component_cls: type, loadable_name: str, config_path: Path, checkpoint_keys: list[str]
    ) -> str | None:
        """Say which model parameters the file's keys did not fill, or None if that cannot be told.

        Only runs after a failed load, from inside its error handler, so it must not raise.
        """
        try:
            from accelerate import init_empty_weights  # type: ignore[reportMissingImports]
            from diffusers.loaders.single_file_model import (  # type: ignore[reportMissingImports]
                SINGLE_FILE_LOADABLE_CLASSES,
            )

            config = component_cls.load_config(str(config_path))  # type: ignore[attr-defined]
            with init_empty_weights():
                model = component_cls.from_config(config)  # type: ignore[attr-defined]
            mapping_fn = SINGLE_FILE_LOADABLE_CLASSES[loadable_name]["checkpoint_mapping_fn"]
            try:
                converted = set(mapping_fn(checkpoint=dict.fromkeys(checkpoint_keys), config=config))
            except (AttributeError, TypeError, KeyError, ValueError, IndexError, RuntimeError) as e:
                # Converters that read tensor values cannot run on placeholders.
                logger.warning(
                    "Could not compare the key names of '%s' with the model. Failed with %s: %s",
                    component_cls.__name__,
                    type(e).__name__,
                    e,
                )
                return None

            expected = set(model.state_dict())
            missing = sorted(expected - converted)
            unmatched = sorted(converted - expected)
            logger.warning(
                "%s with config '%s': %d model parameters not filled, e.g. %s; %d file keys matched nothing, e.g. %s",
                component_cls.__name__,
                config_path,
                len(missing),
                missing[:5],
                len(unmatched),
                unmatched[:5],
            )
            return f"{len(missing)} model parameters were not filled and {len(unmatched)} file keys matched nothing."
        except Exception as e:
            # This is best-effort context for an existing load error; never replace that actionable error.
            logger.warning(
                "Could not compare the key names of '%s' with the model. Failed with %s: %s",
                component_cls.__name__,
                type(e).__name__,
                e,
                exc_info=True,
            )
            return None

    def _materialize_local_dir(self, *, pipeline_cls: type, effective_slot: str) -> Any:
        import torch  # type: ignore[reportMissingImports]

        if not self.file_path:
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed because file_path is required for LOCAL_DIR source type."
            )
            raise ValueError(msg)

        folder = Path(self.file_path)
        if not folder.is_dir():
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with file_path='{self.file_path}' because it is not an existing directory."
            )
            raise FileNotFoundError(msg)

        if not (folder / "config.json").is_file():
            msg = (
                f"Attempted to materialize {self.component}. "
                f"Failed with file_path='{self.file_path}' because it does not contain a 'config.json'. "
                f"Pick a diffusers-format component folder (e.g. '.../FLUX.1-dev/transformer/')."
            )
            raise FileNotFoundError(msg)

        component_cls = get_loadable_component_class(pipeline_cls, effective_slot)

        logger.info(
            "Materializing %s (%s) from LOCAL_DIR path='%s'.",
            self.component,
            component_cls.__name__,
            self.file_path,
        )
        return component_cls.from_pretrained(
            self.file_path,
            torch_dtype=getattr(torch, self.torch_dtype),
            local_files_only=True,
        )
