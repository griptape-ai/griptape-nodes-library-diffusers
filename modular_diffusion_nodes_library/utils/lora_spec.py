"""LoraSpec — a single LoRA adapter's identity + weight + trigger phrase.

Wire format on the node graph stays as `list[dict]`; `LoraSpec.from_raw`
normalises legacy `{path: weight}` payloads and new `{path, weight,
trigger_phrase}` payloads into a typed object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LoraSpec:
    path: str
    weight: float = 1.0
    trigger_phrase: str | None = None

    @classmethod
    def from_raw(cls, raw: Any) -> LoraSpec:
        """Accept a LoraSpec, a new-form dict `{path, weight, trigger_phrase}`,
        or a legacy single-entry dict `{path: weight}`."""
        if isinstance(raw, LoraSpec):
            return raw
        if not isinstance(raw, dict):
            raise TypeError(f"Attempted to build LoraSpec. Failed with raw={raw!r} because expected LoraSpec or dict.")
        if "path" in raw:
            return cls(
                path=str(raw["path"]),
                weight=float(raw.get("weight", 1.0)),
                trigger_phrase=raw.get("trigger_phrase"),
            )
        # Legacy: single-key dict {path: weight}
        if len(raw) == 1:
            ((path, weight),) = raw.items()
            return cls(path=str(path), weight=float(weight))
        raise TypeError(
            f"Attempted to build LoraSpec. Failed with raw={raw!r} because dict has no 'path' "
            "and is not a legacy single-entry mapping."
        )


def normalize_loras(raw_list: list[Any]) -> dict[str, LoraSpec]:
    """Normalise a list of raw LoRA payloads into `{path: LoraSpec}`.

    Duplicate paths are combined by summing their weights.
    """
    out: dict[str, LoraSpec] = {}
    for raw in raw_list:
        # Skip empty placeholders emitted by the UI for unset ParameterList entries.
        if raw == [] or raw == {}:
            continue
        spec = LoraSpec.from_raw(raw)
        if not spec.path:
            raise ValueError(f"Attempted to normalise LoRAs. Failed with raw={raw!r} because path is empty.")
        existing = out.get(spec.path)
        if existing is None:
            out[spec.path] = spec
            continue
        out[spec.path] = LoraSpec(
            path=spec.path,
            weight=existing.weight + spec.weight,
            trigger_phrase=spec.trigger_phrase or existing.trigger_phrase,
        )
    return out
