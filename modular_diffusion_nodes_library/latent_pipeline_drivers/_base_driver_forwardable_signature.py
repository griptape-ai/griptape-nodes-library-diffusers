"""Signature contract enforcement for ``LatentPipelineDriver`` forwardable methods.

Drivers expose a small set of public methods that all callers invoke uniformly.
Those methods MUST match the cross-driver signature listed in
:data:`FORWARDABLE_METHOD_POSITIONAL` *exactly* — no extra positional parameters,
no extra keyword-only parameters, no ``*args`` / ``**kwargs``. Driver-specific
tunables are not permitted on the forwardable surface; route them through other
methods or through artifact metadata.

This module owns:

- :data:`FORWARDABLE_METHODS` — the set of method names that participate
- :data:`FORWARDABLE_METHOD_POSITIONAL` — the positional contract per method
- :func:`validate_forwardable_signature` — the import-time check, invoked from
  ``LatentPipelineDriver.__init_subclass__``
"""

from inspect import Parameter, signature
from typing import Any

FORWARDABLE_METHODS: tuple[str, ...] = (
    "encode_media",
    "decode_latent",
    "create_noise_latent",
    "add_noise_to_latent",
)

FORWARDABLE_METHOD_POSITIONAL: dict[str, tuple[str, ...]] = {
    "encode_media": ("media", "generator_state"),
    "decode_latent": ("latent",),
    "create_noise_latent": ("source_shape", "generator_state"),
    "add_noise_to_latent": ("latent", "generator_state", "num_inference_steps", "strength"),
}


def validate_forwardable_signature(owner_name: str, method_name: str, method: Any) -> None:
    """Enforce the forwardable-method signature contract for a driver class.

    The driver method's parameters (after ``self``) must match
    :data:`FORWARDABLE_METHOD_POSITIONAL` exactly — same names, same order,
    same count. No additional parameters of any kind are permitted.

    ``owner_name`` is used only for error messages (typically ``cls.__name__``).
    """
    contract = FORWARDABLE_METHOD_POSITIONAL.get(method_name, ())
    try:
        sig = signature(method)
    except (TypeError, ValueError):
        return
    params = list(sig.parameters.values())[1:]  # skip ``self``

    if len(params) != len(contract):
        msg = (
            f"{owner_name}.{method_name} declares {len(params)} parameter(s) "
            f"({[p.name for p in params]}) but the base-class contract expects "
            f"exactly {len(contract)} ({list(contract)}). Driver-specific "
            f"tunables are not permitted on forwardable methods."
        )
        raise TypeError(msg)

    for index, (param, expected_name) in enumerate(zip(params, contract, strict=True)):
        if param.kind is Parameter.VAR_POSITIONAL or param.kind is Parameter.VAR_KEYWORD:
            msg = (
                f"{owner_name}.{method_name} declares '{param.name}' as {param.kind.name}.\n"
                f"Forwardable driver methods may not accept *args / **kwargs."
            )
            raise TypeError(msg)
        if param.name != expected_name:
            msg = (
                f"{owner_name}.{method_name} positional parameter at index {index} is named "
                f"'{param.name}' but the base-class contract expects '{expected_name}'."
            )
            raise TypeError(msg)
        if param.kind not in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.POSITIONAL_ONLY):
            msg = (
                f"{owner_name}.{method_name} contract parameter '{param.name}' has kind "
                f"{param.kind.name}; expected POSITIONAL_OR_KEYWORD."
            )
            raise TypeError(msg)
