"""Config-only compatibility checks for pipeline component overrides.

:func:`evaluate_component_compatibility` runs every check over the components and returns a
list of findings. When a config cannot be resolved quickly the relevant check is
silently skipped.

Validation checks overview
-----------------
Checks are grouped into two passes: dimensional checks (:func:`_run_dimensional_checks`) and
config-consistency checks (:func:`_run_config_consistency_checks`).

Dimensional checks — compare the config of a component against the denoiser:

- :class:`VaeDenoiserChannelMismatch` (:func:`_check_vae_denoiser`): flagged when the VAE's
  ``latent_channels`` (or ``z_dim``) multiplied by the pipeline's latent packing ratio
  (``PipelineParameters.latent_packing_ratio``) does not equal the denoiser's ``in_channels``.
- :class:`TextEncoderDenoiserDimMismatch` (:func:`_check_text_conditioning_denoiser`): flagged
  when the text-conditioning width (resolved by the pipeline-params class) does not equal the
  denoiser dimension named by ``text_conditioning_target_dim_key``.

Config-consistency checks — compare an *overridden* component against the *base repo's*
component:

- :class:`VaeScalingMismatch` (:func:`_check_vae_scaling`): one finding per key when an
  override VAE's ``scaling_factor`` or ``shift_factor`` differs from the base
  VAE's value.
- :class:`SchedulerObjectiveMismatch` (:func:`_check_scheduler_prediction_type`): flagged when
  an override scheduler's ``prediction_type`` differs from the base's, or when only one side
  has a ``prediction_type`` at all.
- :class:`TokenizerTextEncoderMismatch` (:func:`_check_tokenizer_text_encoder_pairing`): flagged
  when an override tokenizer's coarse family (CLIP vs T5) differs from its paired text
  encoder's, or when their ``vocab_size`` values disagree.
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from huggingface_hub import try_to_load_from_cache

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.scheduler_component_artifact import SchedulerComponentArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import component_config_filename
from modular_diffusion_nodes_library.component_loading.config_resolver import try_load_json_dict

logger = logging.getLogger("modular_diffusers_nodes_library")

# ---------------------------------------------------------------------------
# Config readers (pure helpers, no side effects)
# ---------------------------------------------------------------------------


def _get_vae_latent_channels(config: dict[str, Any]) -> int | None:
    """Extract latent channel count from a VAE config.

    ``AutoencoderKL`` uses ``latent_channels``; WAN / Qwen VAEs use ``z_dim``.
    """
    value = config.get("latent_channels") or config.get("z_dim")
    if isinstance(value, int):
        return value
    return None


def _get_denoiser_in_channels(config: dict[str, Any]) -> int | None:
    """Extract ``in_channels`` from a transformer or UNet config."""
    value = config.get("in_channels")
    if isinstance(value, int):
        return value
    return None


def _check_text_conditioning_denoiser(
    text_conditioning_dim: int | None,
    denoiser_config: dict[str, Any],
    denoiser_key: str | None,
) -> TextEncoderDenoiserDimMismatch | None:
    """Check resolved text-conditioning width against a denoiser dimension key."""
    if text_conditioning_dim is None or denoiser_key is None:
        return None

    denoiser_dim = denoiser_config.get(denoiser_key)
    if not isinstance(denoiser_dim, int):
        return None

    if text_conditioning_dim != denoiser_dim:
        return TextEncoderDenoiserDimMismatch(
            slot="text_conditioning",
            text_encoder_dim=text_conditioning_dim,
            denoiser_key=denoiser_key,
            denoiser_dim=denoiser_dim,
        )

    return None


def _read_base_repo_config(base_repo_id: str, slot: str, revision: str | None) -> dict[str, Any] | None:
    """Read a non-overridden component's config from the base repo cache."""
    filename = f"{slot}/{component_config_filename(slot)}"
    cached = try_to_load_from_cache(base_repo_id, filename, revision=revision)
    if not isinstance(cached, str):
        return None
    return try_load_json_dict(Path(cached))


class _ConfigResolver:
    """Memoized reader for component configs.

    Distinguishes the two views the checks need — the *overridden* component's
    config and the *base repo's* config for a slot.
    """

    def __init__(
        self,
        overrides: dict[str, ComponentArtifact],
        pipeline_cls: type,
        base_repo_id: str | None,
        base_revision: str | None = None,
    ) -> None:
        self._overrides = overrides
        self._pipeline_cls = pipeline_cls
        self._base_repo_id = base_repo_id
        self._base_revision = base_revision
        self._override_cache: dict[str, dict[str, Any] | None] = {}
        self._base_cache: dict[str, dict[str, Any] | None] = {}

    def override_config(self, slot: str) -> dict[str, Any] | None:
        """Config of the component overriding ``slot``, or ``None`` if not overridden."""
        if slot not in self._override_cache:
            artifact = self._overrides.get(slot)
            if isinstance(artifact, ModelComponentArtifact):
                self._override_cache[slot] = artifact.try_read_config(
                    pipeline_cls=self._pipeline_cls,
                )
            elif isinstance(artifact, SchedulerComponentArtifact):
                self._override_cache[slot] = self._try_read_scheduler_config(artifact)
            else:
                self._override_cache[slot] = None
        return self._override_cache[slot]

    @staticmethod
    def _try_read_scheduler_config(artifact: SchedulerComponentArtifact) -> dict[str, Any] | None:
        try:
            return artifact.read_config()
        except (ValueError, FileNotFoundError, OSError):
            return None

    def base_config(self, slot: str) -> dict[str, Any] | None:
        """Config of ``slot`` from the base repo, or ``None`` if unavailable."""
        if slot not in self._base_cache:
            if self._base_repo_id is not None:
                self._base_cache[slot] = _read_base_repo_config(self._base_repo_id, slot, self._base_revision)
            else:
                self._base_cache[slot] = None
        return self._base_cache[slot]

    def resolved_config(self, slot: str) -> dict[str, Any] | None:
        """Effective config for ``slot`` — the override if present, else the base repo's."""
        if slot in self._overrides:
            return self.override_config(slot)
        return self.base_config(slot)


class FindingCategory(StrEnum):
    """Machine-readable type tag for a compatibility finding."""

    VAE_DENOISER_CHANNELS = "vae_denoiser_channels"
    TEXT_ENCODER_DENOISER_DIM = "text_encoder_denoiser_dim"
    VAE_SCALING = "vae_scaling"
    SCHEDULER_OBJECTIVE = "scheduler_objective"
    TOKENIZER_TEXT_ENCODER = "tokenizer_text_encoder"


class ComponentFamily(StrEnum):
    """Coarse model family used to pair tokenizers with text encoders."""

    CLIP = "clip"
    T5 = "t5"

    @classmethod
    def from_config_value(cls, value: Any) -> ComponentFamily | None:
        """Map a tokenizer/text-encoder identifier string to a coarse family."""
        if not isinstance(value, str):
            return None
        lowered = value.lower()
        if lowered.startswith("clip"):
            return cls.CLIP
        if lowered.startswith("t5"):
            return cls.T5
        return None


class TokenizerMismatchReason(StrEnum):
    """Why a tokenizer/text-encoder pairing was flagged."""

    FAMILY = "family"
    VOCAB = "vocab"


@dataclass(frozen=True)
class ComponentCompatibilityFinding(ABC):
    """A structured, self-describing compatibility observation.

    Subclasses carry the data behind the finding and implement ``__str__`` (the
    user-facing message). ``category`` is a machine-readable type tag the caller
    uses to decide severity; the dataclass-generated ``__repr__`` exposes the
    raw fields.
    """

    category: ClassVar[FindingCategory]

    slot: str

    @abstractmethod
    def __str__(self) -> str:
        """Return the user-facing description of this finding."""
        raise NotImplementedError


@dataclass(frozen=True)
class VaeDenoiserChannelMismatch(ComponentCompatibilityFinding):
    """VAE latent channels are incompatible with the denoiser's in_channels."""

    category: ClassVar[FindingCategory] = FindingCategory.VAE_DENOISER_CHANNELS

    vae_latent_channels: int = 0
    packing_ratio: int = 1
    denoiser_in_channels: int = 0

    def __str__(self) -> str:
        expected = self.vae_latent_channels * self.packing_ratio
        if self.packing_ratio == 1:
            return (
                f"VAE latent_channels ({self.vae_latent_channels}) "
                f"does not match denoiser in_channels ({self.denoiser_in_channels})."
            )
        return (
            f"VAE latent_channels ({self.vae_latent_channels}) * packing_ratio ({self.packing_ratio}) "
            f"= {expected}, but denoiser in_channels = {self.denoiser_in_channels}."
        )


@dataclass(frozen=True)
class TextEncoderDenoiserDimMismatch(ComponentCompatibilityFinding):
    """Text-encoder output dim is incompatible with the denoiser's cross-attention dim."""

    category: ClassVar[FindingCategory] = FindingCategory.TEXT_ENCODER_DENOISER_DIM

    text_encoder_dim: int = 0
    denoiser_key: str = ""
    denoiser_dim: int = 0

    def __str__(self) -> str:
        return (
            f"Text encoder output dimension ({self.text_encoder_dim}) "
            f"does not match denoiser {self.denoiser_key} ({self.denoiser_dim})."
        )


@dataclass(frozen=True)
class VaeScalingMismatch(ComponentCompatibilityFinding):
    """One VAE normalization key differs from the base VAE."""

    category: ClassVar[FindingCategory] = FindingCategory.VAE_SCALING

    key: str = ""
    override_value: float | None = None
    base_value: float | None = None

    def __str__(self) -> str:
        return (
            f"Override VAE {self.key} ({self.override_value}) "
            f"differs from base VAE {self.key} ({self.base_value}). "
            "This may produce washed-out or over-saturated output."
        )


@dataclass(frozen=True)
class SchedulerObjectiveMismatch(ComponentCompatibilityFinding):
    """An overridden scheduler's objective differs from the base scheduler."""

    category: ClassVar[FindingCategory] = FindingCategory.SCHEDULER_OBJECTIVE

    override_prediction_type: str | None = None
    base_prediction_type: str | None = None
    override_class: str | None = None
    base_class: str | None = None
    family_mismatch: bool = False

    def __str__(self) -> str:
        if self.family_mismatch:
            if self.override_prediction_type is None:
                flow, non_flow = self.override_class, self.base_class
            else:
                flow, non_flow = self.base_class, self.override_class
            return (
                f"'{flow}' is a flow-matching scheduler but '{non_flow}' is not. "
                "Mixing sampling families will produce incorrect results."
            )
        return (
            f"Override scheduler uses prediction_type='{self.override_prediction_type}', "
            f"but the base model was trained with prediction_type='{self.base_prediction_type}'."
        )


@dataclass(frozen=True)
class TokenizerTextEncoderMismatch(ComponentCompatibilityFinding):
    """An overridden tokenizer may not match its paired text encoder."""

    category: ClassVar[FindingCategory] = FindingCategory.TOKENIZER_TEXT_ENCODER

    text_encoder_slot: str = ""
    override_tokenizer_class: str | None = None
    text_encoder_model_type: str | None = None
    tokenizer_family: ComponentFamily | None = None
    text_encoder_family: ComponentFamily | None = None
    tokenizer_vocab_size: int | None = None
    text_encoder_vocab_size: int | None = None
    reason: TokenizerMismatchReason = TokenizerMismatchReason.FAMILY

    def __str__(self) -> str:
        if self.reason is TokenizerMismatchReason.VOCAB:
            return (
                f"Tokenizer '{self.slot}' vocab_size ({self.tokenizer_vocab_size}) does not match "
                f"text encoder '{self.text_encoder_slot}' vocab_size ({self.text_encoder_vocab_size})."
            )
        return (
            f"Tokenizer '{self.slot}' is {self.tokenizer_family} "
            f"('{self.override_tokenizer_class}'), but text encoder "
            f"'{self.text_encoder_slot}' is {self.text_encoder_family} "
            f"('{self.text_encoder_model_type}'). These are different model families."
        )


# ---------------------------------------------------------------------------
# Dimensional checks (VAE / text-encoder vs denoiser)
# ---------------------------------------------------------------------------


def _check_vae_denoiser(
    vae_config: dict[str, Any],
    denoiser_config: dict[str, Any],
    packing_ratio: int,
) -> VaeDenoiserChannelMismatch | None:
    """Check VAE latent channels against denoiser in_channels.

    Returns a finding if incompatible, ``None`` if compatible or if the check
    cannot be performed (missing keys).
    """
    vae_channels = _get_vae_latent_channels(vae_config)
    if vae_channels is None:
        return None

    denoiser_channels = _get_denoiser_in_channels(denoiser_config)
    if denoiser_channels is None:
        return None

    if vae_channels * packing_ratio != denoiser_channels:
        return VaeDenoiserChannelMismatch(
            slot="vae",
            vae_latent_channels=vae_channels,
            packing_ratio=packing_ratio,
            denoiser_in_channels=denoiser_channels,
        )

    return None


def _find_denoiser_slot(pipeline_cls: type) -> str | None:
    """Return ``'transformer'`` or ``'unet'`` depending on which the pipeline declares."""
    params = set(inspect.signature(pipeline_cls.__init__).parameters.keys())
    if "transformer" in params:
        return "transformer"
    if "unet" in params:
        return "unet"
    return None


def _run_dimensional_checks(
    resolver: _ConfigResolver,
    pipeline_cls: type,
    pipeline_params_cls: type | None,
) -> list[ComponentCompatibilityFinding]:
    """VAE / text-encoder vs denoiser dimensional checks."""
    findings: list[ComponentCompatibilityFinding] = []

    denoiser_slot = _find_denoiser_slot(pipeline_cls)
    if denoiser_slot is None:
        return findings

    denoiser_config = resolver.resolved_config(denoiser_slot)
    if denoiser_config is None:
        return findings

    packing_ratio = pipeline_params_cls.latent_packing_ratio if pipeline_params_cls is not None else 1

    vae_config = resolver.resolved_config("vae")
    if vae_config is not None:
        finding = _check_vae_denoiser(vae_config, denoiser_config, packing_ratio)
        if finding is not None:
            findings.append(finding)

    if pipeline_params_cls is not None:
        target_dim_key = pipeline_params_cls.text_conditioning_target_dim_key
        text_conditioning_dim = pipeline_params_cls.text_conditioning_width(resolver.resolved_config)
        finding = _check_text_conditioning_denoiser(
            text_conditioning_dim,
            denoiser_config,
            target_dim_key,
        )
        if finding is not None:
            findings.append(finding)

    return findings


# ---------------------------------------------------------------------------
# Config-consistency checks (VAE normalization / scheduler / tokenizer pairing)
# ---------------------------------------------------------------------------

_SCALING_TOLERANCE = 1e-4


def _values_differ(override_val: Any, base_val: Any) -> bool:
    """True if two config values differ (float-tolerant; None-aware)."""
    if override_val is None or base_val is None:
        return override_val is not base_val
    if isinstance(override_val, (int, float)) and isinstance(base_val, (int, float)):
        return abs(float(override_val) - float(base_val)) > _SCALING_TOLERANCE
    return override_val != base_val


def _check_vae_scaling(
    slot: str,
    override_vae_config: dict[str, Any],
    base_vae_config: dict[str, Any],
) -> list[VaeScalingMismatch]:
    """Compare scaling_factor / shift_factor between override and base VAE.

    Emits one finding per mismatched key. Only compares a key when both configs
    expose it; a key present on just one side (e.g. Qwen/Wan VAEs, which
    normalize via latents_mean/std instead) is skipped rather than flagged.
    """
    findings: list[VaeScalingMismatch] = []
    for key in ("scaling_factor", "shift_factor"):
        override_val = override_vae_config.get(key)
        base_val = base_vae_config.get(key)
        if override_val is None or base_val is None:
            continue
        if _values_differ(override_val, base_val):
            findings.append(VaeScalingMismatch(slot=slot, key=key, override_value=override_val, base_value=base_val))
    return findings


def _check_scheduler_prediction_type(
    slot: str,
    override_cfg: dict[str, Any],
    base_cfg: dict[str, Any],
) -> SchedulerObjectiveMismatch | None:
    """Compare scheduler objective between override and base.

    Compares ``prediction_type`` when both expose it; when only one side does (a
    flow-matching scheduler has none) treats it as a sampling-family mismatch.
    Returns ``None`` when neither exposes a comparable field.
    """
    override_pt = override_cfg.get("prediction_type")
    base_pt = base_cfg.get("prediction_type")

    if override_pt is not None and base_pt is not None:
        if override_pt != base_pt:
            return SchedulerObjectiveMismatch(
                slot=slot,
                override_prediction_type=override_pt,
                base_prediction_type=base_pt,
                family_mismatch=False,
            )
        return None

    if (override_pt is None) != (base_pt is None):
        return SchedulerObjectiveMismatch(
            slot=slot,
            override_prediction_type=override_pt,
            base_prediction_type=base_pt,
            override_class=override_cfg.get("_class_name"),
            base_class=base_cfg.get("_class_name"),
            family_mismatch=True,
        )

    return None


def _check_tokenizer_text_encoder_pairing(
    tokenizer_slot: str,
    text_encoder_slot: str,
    tokenizer_cfg: dict[str, Any],
    text_encoder_cfg: dict[str, Any],
) -> TokenizerTextEncoderMismatch | None:
    """Coarse tokenizer/text-encoder family match, with a vocab_size cross-check.

    Returns ``None`` when the family can't be determined and vocab sizes agree
    (or aren't both present).
    """
    tok_family = ComponentFamily.from_config_value(tokenizer_cfg.get("tokenizer_class"))
    te_family = ComponentFamily.from_config_value(text_encoder_cfg.get("model_type"))

    if tok_family is not None and te_family is not None and tok_family != te_family:
        return TokenizerTextEncoderMismatch(
            slot=tokenizer_slot,
            text_encoder_slot=text_encoder_slot,
            override_tokenizer_class=tokenizer_cfg.get("tokenizer_class"),
            text_encoder_model_type=text_encoder_cfg.get("model_type"),
            tokenizer_family=tok_family,
            text_encoder_family=te_family,
            reason=TokenizerMismatchReason.FAMILY,
        )

    tok_vocab = tokenizer_cfg.get("vocab_size")
    te_vocab = text_encoder_cfg.get("vocab_size")
    if isinstance(tok_vocab, int) and isinstance(te_vocab, int) and tok_vocab != te_vocab:
        return TokenizerTextEncoderMismatch(
            slot=tokenizer_slot,
            text_encoder_slot=text_encoder_slot,
            tokenizer_vocab_size=tok_vocab,
            text_encoder_vocab_size=te_vocab,
            reason=TokenizerMismatchReason.VOCAB,
        )

    return None


def _paired_text_encoder_slot(tokenizer_slot: str) -> str:
    """Return the text-encoder slot paired with a tokenizer slot by index."""
    if tokenizer_slot == "tokenizer":
        return "text_encoder"
    if tokenizer_slot.startswith("tokenizer_"):
        return f"text_encoder{tokenizer_slot.removeprefix('tokenizer')}"
    raise ValueError(f"'{tokenizer_slot}' is not a tokenizer slot")


def _run_config_consistency_checks(
    resolver: _ConfigResolver,
    overrides: dict[str, ComponentArtifact],
) -> list[ComponentCompatibilityFinding]:
    """VAE normalization / scheduler objective / tokenizer pairing checks."""
    findings: list[ComponentCompatibilityFinding] = []

    # VAE scaling/shift (needs a base VAE to compare against)
    if "vae" in overrides:
        override_vae = resolver.override_config("vae")
        base_vae = resolver.base_config("vae")
        if override_vae is not None and base_vae is not None:
            findings.extend(_check_vae_scaling("vae", override_vae, base_vae))

    # Scheduler objective (dormant until a scheduler producer exists)
    if "scheduler" in overrides:
        override_sched = resolver.override_config("scheduler")
        base_sched = resolver.base_config("scheduler")
        if override_sched is not None and base_sched is not None:
            scheduler_finding = _check_scheduler_prediction_type("scheduler", override_sched, base_sched)
            if scheduler_finding is not None:
                findings.append(scheduler_finding)

    # Tokenizer / text-encoder pairing
    for tok_slot in overrides:
        if tok_slot != "tokenizer" and not tok_slot.startswith("tokenizer_"):
            continue
        tok_cfg = resolver.override_config(tok_slot)
        if tok_cfg is None:
            continue
        te_slot = _paired_text_encoder_slot(tok_slot)
        te_cfg = resolver.override_config(te_slot)
        if te_cfg is None:
            te_cfg = resolver.base_config(te_slot)
        if te_cfg is None:
            continue
        pairing_finding = _check_tokenizer_text_encoder_pairing(tok_slot, te_slot, tok_cfg, te_cfg)
        if pairing_finding is not None:
            findings.append(pairing_finding)

    return findings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_component_compatibility(
    overrides: dict[str, ComponentArtifact],
    pipeline_cls: type,
    base_repo_id: str | None,
    *,
    base_revision: str | None = None,
    pipeline_params_cls: type | None = None,
) -> list[ComponentCompatibilityFinding]:
    """Run every config-only compatibility check in a single pass.

    Parameters
    ----------
    overrides:
        Component overrides from the builder node (slot name → artifact).
    pipeline_cls:
        The target diffusers pipeline class.
    base_repo_id:
        The base HF repo for non-overridden components, or ``None`` when
        building entirely from overrides.
    base_revision:
        The revision of ``base_repo_id`` to read cached configs from.
    """
    try:
        resolver = _ConfigResolver(overrides, pipeline_cls, base_repo_id, base_revision)
        findings = _run_dimensional_checks(resolver, pipeline_cls, pipeline_params_cls)
        findings.extend(_run_config_consistency_checks(resolver, overrides))
        return findings
    except Exception:
        logger.warning(
            "Component compatibility evaluation raised an unexpected error; skipping.",
            exc_info=True,
        )
        return []
