"""Ordered list of pipeline component slots that may be surfaced as override ports.

Only slots present in ``ALLOWED_COMPONENT_SLOTS`` can appear in a pipeline
builder's Component Overrides group. The list order determines the display
order in the UI — more fundamental components (transformer, VAE) appear first.

When a new overridable component type is added to the library, append it here.
Slots absent from this list are never exposed as override inputs, regardless of
what ``DiffusionPipeline._get_signature_keys`` returns for a given pipeline.
"""

import re

ALLOWED_COMPONENT_SLOTS: list[str] = [
    "transformer",
    "unet",
    "vae",
    "text_encoder",
    "text_encoder_2",
    "text_encoder_3",
    "tokenizer",
    "tokenizer_2",
    "tokenizer_3",
    "transformer_2",
    "scheduler",
]

# Human-readable display name for each slot. Covers all entries in
# ALLOWED_COMPONENT_SLOTS; update both together when adding a new slot.
SLOT_DISPLAY_NAMES: dict[str, str] = {
    "transformer": "Transformer",
    "unet": "UNet",
    "vae": "VAE",
    "text_encoder": "Text Encoder",
    "text_encoder_2": "Text Encoder 2",
    "text_encoder_3": "Text Encoder 3",
    "tokenizer": "Tokenizer",
    "tokenizer_2": "Tokenizer 2",
    "tokenizer_3": "Tokenizer 3",
    "transformer_2": "Transformer 2",
    "scheduler": "Scheduler",
}

# Import-time invariant: every ALLOWED_COMPONENT_SLOTS entry must have a
# display name, and no display name may exist without a corresponding slot.
_missing_display_names = sorted(set(ALLOWED_COMPONENT_SLOTS) - set(SLOT_DISPLAY_NAMES))
_orphan_display_names = sorted(set(SLOT_DISPLAY_NAMES) - set(ALLOWED_COMPONENT_SLOTS))
if _missing_display_names or _orphan_display_names:
    _msg = (
        f"Attempted to load component_slots module. "
        f"Failed because ALLOWED_COMPONENT_SLOTS and SLOT_DISPLAY_NAMES are out of sync: "
        f"missing display names for {_missing_display_names}, "
        f"orphan display names for {_orphan_display_names}."
    )
    raise RuntimeError(_msg)


def slot_component_kind(slot: str) -> str:
    """Strip trailing numeric suffix from slot name to get the component kind.

    Examples:
        "transformer" → "transformer"
        "transformer_2" → "transformer"
        "vae" → "vae"
    """
    return re.sub(r"_\d+$", "", slot)


def slot_artifact_type_name(slot: str) -> str:
    """Derive the artifact type name for a given slot.

    Examples:
        "transformer" → "TransformerComponentArtifact"
        "transformer_2" → "TransformerComponentArtifact"
        "vae" → "VaeComponentArtifact"
    """
    kind = slot_component_kind(slot)
    parts = kind.split("_")
    pascal_name = "".join(part.capitalize() for part in parts)
    return f"{pascal_name}ComponentArtifact"
