"""Ordered list of pipeline component slots that may be surfaced as override ports.

Only slots present in ``ALLOWED_COMPONENT_SLOTS`` can appear in a pipeline
builder's Component Overrides group. The list order determines the display
order in the UI — more fundamental components (transformer, VAE) appear first.

When a new overridable component type is added to the library, append it here.
Slots absent from this list are never exposed as override inputs, regardless of
what ``DiffusionPipeline._get_signature_keys`` returns for a given pipeline.
"""

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
