from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter


@dataclass(frozen=True)
class ReconciledOptimizationSettings:
    overrides: dict[str, Any]
    displayed_values: dict[str, Any]


def _validated_optimization_parameter_value(parameter: Parameter, value: Any) -> Any:
    if type(value) is not type(parameter.default_value):
        raise ValueError(f"Load Pipeline optimization setting '{parameter.name}' has an invalid value: {value!r}.")
    candidate = value
    try:
        for converter in parameter.converters:
            candidate = converter(candidate)
        for validator in parameter.validators:
            validator(parameter, candidate)
    except Exception as error:
        raise ValueError(
            f"Load Pipeline optimization setting '{parameter.name}' has an invalid value: {value!r}."
        ) from error
    if type(candidate) is not type(value) or candidate != value:
        raise ValueError(f"Load Pipeline optimization setting '{parameter.name}' has an invalid value: {value!r}.")
    return candidate


def validated_optimization_overrides(
    value: Any,
    parameters: Mapping[str, Parameter],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Load Pipeline optimization override state must be a dictionary.")
    if not all(isinstance(name, str) for name in value):
        raise ValueError("Load Pipeline optimization override names must be strings.")
    unknown_names = set(value) - set(parameters)
    if unknown_names:
        names = ", ".join(sorted(unknown_names))
        raise ValueError(f"Load Pipeline optimization override state contains unknown parameters: {names}.")
    return {
        name: _validated_optimization_parameter_value(parameters[name], override_value)
        for name, override_value in value.items()
    }


def reconcile_optimization_settings(
    *,
    parameters: Mapping[str, Parameter],
    recipe_values: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> ReconciledOptimizationSettings:
    reconciled_overrides = validated_optimization_overrides(dict(overrides), parameters)
    displayed_values: dict[str, Any] = {}
    for name, parameter in parameters.items():
        recipe_value = _validated_optimization_parameter_value(
            parameter,
            recipe_values.get(name, parameter.default_value),
        )
        if name in reconciled_overrides and reconciled_overrides[name] == recipe_value:
            reconciled_overrides.pop(name)
        displayed_values[name] = reconciled_overrides.get(name, recipe_value)
    return ReconciledOptimizationSettings(
        overrides=reconciled_overrides,
        displayed_values=displayed_values,
    )
