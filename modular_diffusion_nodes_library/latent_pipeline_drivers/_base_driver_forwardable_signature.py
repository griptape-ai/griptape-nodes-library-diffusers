"""Signature contract enforcement for ``LatentPipelineDriver`` forwardable methods.

Drivers expose a small set of public methods that all callers invoke uniformly.
Those methods MUST keep the cross-driver positional contract listed in
:data:`FORWARDABLE_METHOD_POSITIONAL`; any driver-specific tunable beyond the
contract must be declared keyword-only so it cannot drift into the positional
signature.

This module owns:

- :data:`FORWARDABLE_METHODS` — the set of method names that participate
- :data:`FORWARDABLE_METHOD_POSITIONAL` — the positional contract per method
- :func:`validate_forwardable_signature` — the import-time check, invoked from
  ``LatentPipelineDriver.__init_subclass__``
"""

from inspect import Parameter, signature
from typing import Any

FORWARDABLE_METHODS: tuple[str, ...] = (
    "encode_image",
    "encode_video",
    "decode_latent",
    "create_noise_latent",
    "add_noise_to_latent",
)

FORWARDABLE_METHOD_POSITIONAL: dict[str, tuple[str, ...]] = {
    "encode_image": ("image", "source_shape"),
    "encode_video": ("frames", "source_shape"),
    "decode_latent": ("latent",),
    "create_noise_latent": ("source_shape", "seed"),
    "add_noise_to_latent": ("latent", "seed", "num_inference_steps", "strength"),
}


def validate_forwardable_signature(owner_name: str, method_name: str, method: Any) -> None:
    """Enforce the forwardable-method signature contract for a driver class.

    ``owner_name`` is used only for error messages (typically ``cls.__name__``).
    """
    contract = FORWARDABLE_METHOD_POSITIONAL.get(method_name, ())
    try:
        sig = signature(method)
    except (TypeError, ValueError):
        return
    params = list(sig.parameters.values())[1:]  # skip ``self``
    for index, param in enumerate(params):
        if param.kind is Parameter.VAR_POSITIONAL or param.kind is Parameter.VAR_KEYWORD:
            msg = (
                f"{owner_name}.{method_name} declares '{param.name}' as {param.kind.name}.\n"
                f"Forwardable driver methods may not accept *args / **kwargs — declare each "
                f"driver-specific tunable explicitly as keyword-only."
            )
            raise TypeError(msg)
        if index < len(contract):
            expected_name = contract[index]
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
            continue
        if param.kind is not Parameter.KEYWORD_ONLY:
            msg = (
                f"{owner_name}.{method_name} has parameter '{param.name}' as {param.kind.name}.\n"
                f"Driver-specific tunables on forwardable methods must be keyword-only "
                f"(declare them after `*`)."
            )
            raise TypeError(msg)
