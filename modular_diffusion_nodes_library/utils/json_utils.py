from __future__ import annotations

import json
from typing import Any


def parse_json_to_dict(value: Any) -> dict[str, Any]:
    """Accepts dict, JSON string, or None/empty string; raises ValueError otherwise."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as e:
            msg = f"Could not parse value as JSON: {e}"
            raise ValueError(msg) from e
        if not isinstance(parsed, dict):
            msg = "JSON value must be an object (dict), not a list or scalar."
            raise ValueError(msg)
        return parsed
    try:
        return dict(value)
    except (TypeError, ValueError) as e:
        msg = f"Could not convert value to dict: {e}"
        raise ValueError(msg) from e
