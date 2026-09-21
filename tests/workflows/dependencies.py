from __future__ import annotations

import ast
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple


@dataclass(frozen=True, order=True)
class SourceLocation:
    line: int
    column: int


@dataclass(frozen=True, order=True)
class ModelDependency:
    repo_id: str
    roles: tuple[str, ...]
    nodes: tuple[str, ...]
    sources: tuple[SourceLocation, ...]


@dataclass(frozen=True, order=True)
class ExtractionBlocker:
    code: str
    detail: str
    source: SourceLocation | None = None


@dataclass(frozen=True)
class WorkflowDependencies:
    workflow_path: Path
    model_dependencies: tuple[ModelDependency, ...] = ()
    library_names: tuple[str, ...] = ()
    blockers: tuple[ExtractionBlocker, ...] = ()

    @property
    def conclusive(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class _EmbeddedValue:
    value: Any = None
    error: str | None = None


type EmbeddedValues = dict[str, _EmbeddedValue]


def _source_location(node: ast.AST) -> SourceLocation:
    return SourceLocation(line=node.lineno, column=node.col_offset)


def _is_named_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _is_pickle_loads_bytes(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call) or len(node.args) != 1 or node.keywords:
        return False
    if not isinstance(node.func, ast.Attribute) or node.func.attr != "loads":
        return False
    if not isinstance(node.func.value, ast.Name) or node.func.value.id != "pickle":
        return False
    payload = node.args[0]
    return isinstance(payload, ast.Constant) and isinstance(payload.value, bytes)


def _extract_embedded_values(module: ast.Module) -> EmbeddedValues:
    """Resolve trusted repository-controlled pickle.loads(<bytes literal>) table entries."""
    values: EmbeddedValues = {}

    for node in ast.walk(module):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "top_level_unique_values_dict":
            continue
        if not isinstance(node.value, ast.Dict):
            continue

        for key_node, value_node in zip(node.value.keys, node.value.values, strict=False):
            if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
                continue
            if not _is_pickle_loads_bytes(value_node):
                values[key_node.value] = _EmbeddedValue(error="unsupported embedded value expression")
                continue

            payload = value_node.args[0].value
            try:
                value = pickle.loads(payload)  # noqa: S301 - literal bytes from a trusted, version-controlled template
            except (EOFError, pickle.UnpicklingError, ValueError) as error:
                values[key_node.value] = _EmbeddedValue(error=f"pickle deserialization failed: {error}")
                continue

            values[key_node.value] = _EmbeddedValue(value=value)

    return values


def _extract_library_names(module: ast.Module) -> tuple[tuple[str, ...], tuple[ExtractionBlocker, ...]]:
    """Extract every literal specific_library_name from CreateNodeRequest calls."""
    library_names: set[str] = set()
    blockers: list[ExtractionBlocker] = []

    calls = [node for node in ast.walk(module) if _is_named_call(node, "CreateNodeRequest")]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    for call in calls:
        for keyword in call.keywords:
            if keyword.arg != "specific_library_name":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                library_names.add(keyword.value.value)
                continue
            blockers.append(
                ExtractionBlocker(
                    code="unresolved-library-name",
                    detail="CreateNodeRequest specific_library_name is not a string literal",
                    source=_source_location(keyword.value),
                )
            )

    return tuple(sorted(library_names)), tuple(sorted(blockers))


REPO_ID_PATTERN = re.compile(r"\b[\w.-]+/[\w.-]+\b")
# Mirrors HuggingFaceModelParameter._key_to_repo_revision's key format (griptape_nodes engine):
# saved model selections may be pinned to a revision as "{repo_id} ({40-hex commit hash})".
REPO_REVISION_KEY_PATTERN = re.compile(r"^(.+) \(([a-f0-9]{40})\)$")
REPO_PARAMETERS = {"model", "controlnet_model", "upsampler_model"}
ROLE_BY_PARAMETER = {
    "model": "primary",
    "controlnet_model": "controlnet",
    "upsampler_model": "upscaler",
}


@dataclass(frozen=True)
class _ValueSlot:
    """A candidate model-parameter value awaiting classification, tied to its node identity and source."""

    value_node: ast.expr
    source: SourceLocation
    node_key: str | None


class RepoAssignmentKey(NamedTuple):
    """Identifies a repo parameter slot: the node it belongs to (or a synthetic per-call fallback) and its name."""

    node_key: str
    parameter_name: str


type RepoAssignmentSlots = dict[RepoAssignmentKey, _ValueSlot]


@dataclass(frozen=True)
class _ResolvedValue:
    """The outcome of classifying a model-parameter value: kind is 'empty', 'repo', or 'blocked'."""

    kind: str
    repos: tuple[str, ...] = ()
    detail: str | None = None


class ModelDependencyEvidence(NamedTuple):
    """One resolved (repo, role, node) fact, later merged by repo_id into a ModelDependency."""

    repo_id: str
    role: str
    node_key: str | None
    source: SourceLocation


def _get_keyword(call: ast.Call, name: str) -> ast.expr | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _keyword_str(call: ast.Call, name: str) -> str | None:
    value = _get_keyword(call, name)
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _keyword_bool_literal(call: ast.Call, name: str) -> bool | None:
    value = _get_keyword(call, name)
    if isinstance(value, ast.Constant) and isinstance(value.value, bool):
        return value.value
    return None


def _node_context_regions(module: ast.Module) -> list[tuple[int, int, str | None]]:
    """Map each `with ...ContextManager().node(<var>):` block to its statically known node variable."""
    regions: list[tuple[int, int, str | None]] = []

    for node in ast.walk(module):
        if not isinstance(node, ast.With) or len(node.items) != 1:
            continue
        context_expr = node.items[0].context_expr
        if not isinstance(context_expr, ast.Call):
            continue
        if not isinstance(context_expr.func, ast.Attribute) or context_expr.func.attr != "node":
            continue
        if len(context_expr.args) != 1:
            continue

        arg = context_expr.args[0]
        node_var = arg.id if isinstance(arg, ast.Name) else None
        end_line = node.end_lineno if node.end_lineno is not None else node.lineno
        regions.append((node.lineno, end_line, node_var))

    return regions


def _node_identity_for_line(regions: list[tuple[int, int, str | None]], line: int) -> str | None:
    """Return the node variable of the innermost enclosing node(...) region containing line, if any."""
    innermost: tuple[int, int, str | None] | None = None
    for start, end, node_var in regions:
        if start <= line <= end and (innermost is None or start > innermost[0]):
            innermost = (start, end, node_var)
    return innermost[2] if innermost is not None else None


def _classify_repo_value(value: Any) -> _ResolvedValue:
    """Classify an already-resolved Python value from an in-scope repo parameter."""
    if value is None:
        return _ResolvedValue(kind="empty")
    if isinstance(value, str) and value == "":
        return _ResolvedValue(kind="empty")
    if isinstance(value, list) and not value:
        return _ResolvedValue(kind="empty")

    if isinstance(value, str):
        if REPO_ID_PATTERN.fullmatch(value):
            return _ResolvedValue(kind="repo", repos=(value,))
        revision_key_match = REPO_REVISION_KEY_PATTERN.match(value)
        if revision_key_match and REPO_ID_PATTERN.fullmatch(revision_key_match.group(1)):
            return _ResolvedValue(kind="repo", repos=(revision_key_match.group(1),))
        return _ResolvedValue(kind="blocked", detail=f"value {value!r} is not a recognized repository id")

    if isinstance(value, list):
        repos: set[str] = set()
        for element in value:
            element_result = _classify_repo_value(element)
            if element_result.kind == "blocked":
                return element_result
            repos.update(element_result.repos)
        if not repos:
            return _ResolvedValue(kind="empty")
        return _ResolvedValue(kind="repo", repos=tuple(sorted(repos)))

    return _ResolvedValue(kind="blocked", detail=f"unsupported value type {type(value).__name__!r}")


def _resolve_embedded_table_reference(node: ast.Subscript, embedded: EmbeddedValues) -> _ResolvedValue:
    key_node = node.slice
    if not isinstance(key_node, ast.Constant) or not isinstance(key_node.value, str):
        return _ResolvedValue(kind="blocked", detail="top_level_unique_values_dict key is not a string literal")

    embedded_value = embedded.get(key_node.value)
    if embedded_value is None:
        return _ResolvedValue(kind="blocked", detail=f"no embedded value for key {key_node.value!r}")
    if embedded_value.error is not None:
        return _ResolvedValue(kind="blocked", detail=embedded_value.error)

    return _classify_repo_value(embedded_value.value)


def _resolve_repo_expression(node: ast.expr, embedded: EmbeddedValues) -> _ResolvedValue:
    """Resolve a model-parameter value expression: direct literals and embedded table references."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "top_level_unique_values_dict"
    ):
        return _resolve_embedded_table_reference(node, embedded)

    try:
        literal_value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _ResolvedValue(
            kind="blocked", detail="value is not a literal or a top_level_unique_values_dict reference"
        )

    return _classify_repo_value(literal_value)


def _repo_assignment_key(node_var: str | None, call: ast.Call, parameter_name: str) -> RepoAssignmentKey:
    """Build a RepoAssignmentKey, falling back to a per-call synthetic node key when identity is unresolved."""
    node_key = node_var if node_var is not None else f"@{call.lineno}:{call.col_offset}"
    return RepoAssignmentKey(node_key=node_key, parameter_name=parameter_name)


def _collect_repo_assignments(module: ast.Module) -> RepoAssignmentSlots:
    """Replay AddParameterToNodeRequest/SetParameterValueRequest calls in source order."""
    regions = _node_context_regions(module)
    calls = [
        node
        for node in ast.walk(module)
        if _is_named_call(node, "AddParameterToNodeRequest") or _is_named_call(node, "SetParameterValueRequest")
    ]
    calls.sort(key=lambda call: (call.lineno, call.col_offset))

    dynamically_added_keys: set[RepoAssignmentKey] = set()
    for call in calls:
        if not _is_named_call(call, "AddParameterToNodeRequest"):
            continue
        parameter_name = _keyword_str(call, "parameter_name")
        if parameter_name is None or parameter_name not in REPO_PARAMETERS:
            continue
        node_var = _node_identity_for_line(regions, call.lineno)
        dynamically_added_keys.add(_repo_assignment_key(node_var, call, parameter_name))

    current: RepoAssignmentSlots = {}
    established: set[RepoAssignmentKey] = set()

    for call in calls:
        parameter_name = _keyword_str(call, "parameter_name")
        if parameter_name is None or parameter_name not in REPO_PARAMETERS:
            continue

        if _is_named_call(call, "AddParameterToNodeRequest"):
            node_var = _node_identity_for_line(regions, call.lineno)
            storage_key = _repo_assignment_key(node_var, call, parameter_name)
            established.add(storage_key)
            # ui_options.simple_dropdown is never consulted, so dropdown alternatives can never
            # become defaults or dependencies.
            default_value_node = _get_keyword(call, "default_value")
            if default_value_node is None:
                continue
            current[storage_key] = _ValueSlot(
                value_node=default_value_node, source=_source_location(call), node_key=node_var
            )
            continue

        # SetParameterValueRequest
        if _keyword_bool_literal(call, "is_output") is True:
            continue  # literal output assignments never contribute model dependencies
        value_node = _get_keyword(call, "value")
        if value_node is None:
            continue
        node_name_node = _get_keyword(call, "node_name")
        node_var = node_name_node.id if isinstance(node_name_node, ast.Name) else None
        storage_key = _repo_assignment_key(node_var, call, parameter_name)
        if storage_key in dynamically_added_keys and storage_key not in established:
            continue  # this parameter is only ever created by a later Add in this file, so this assignment fails
        # The final assignment for a given node/parameter wins because calls are replayed in
        # lexical order and each overwrites any prior entry for the same key.
        current[storage_key] = _ValueSlot(value_node=value_node, source=_source_location(call), node_key=node_var)

    return current


def _resolve_model_assignments(
    module: ast.Module, embedded: EmbeddedValues
) -> tuple[list[ModelDependencyEvidence], list[ExtractionBlocker]]:
    """Select the authoritative value per node/parameter and classify it."""
    assignments = _collect_repo_assignments(module)

    evidence: list[ModelDependencyEvidence] = []
    blockers: list[ExtractionBlocker] = []

    for key, slot in assignments.items():
        parameter_name = key.parameter_name
        resolved = _resolve_repo_expression(slot.value_node, embedded)

        if resolved.kind == "blocked":
            blockers.append(
                ExtractionBlocker(
                    code="unresolved-model-dependency",
                    detail=f"{parameter_name}: {resolved.detail}",
                    source=slot.source,
                )
            )
            continue
        if resolved.kind == "empty":
            continue

        # A certain repo-shaped value is retained even when its node identity is unresolved;
        # slot.node_key is None in that case and simply contributes no node evidence below.
        role = ROLE_BY_PARAMETER[parameter_name]
        for repo_id in resolved.repos:
            evidence.append(
                ModelDependencyEvidence(repo_id=repo_id, role=role, node_key=slot.node_key, source=slot.source)
            )

    return evidence, blockers


def _merge_model_dependencies(
    evidence: list[ModelDependencyEvidence],
) -> tuple[ModelDependency, ...]:
    """Deduplicate by repo_id while unioning all role, node, and source evidence."""
    roles_by_repo: dict[str, set[str]] = {}
    nodes_by_repo: dict[str, set[str]] = {}
    sources_by_repo: dict[str, set[SourceLocation]] = {}

    for repo_id, role, node_key, source in evidence:
        roles_by_repo.setdefault(repo_id, set()).add(role)
        sources_by_repo.setdefault(repo_id, set()).add(source)
        if node_key is not None:
            nodes_by_repo.setdefault(repo_id, set()).add(node_key)

    dependencies = [
        ModelDependency(
            repo_id=repo_id,
            roles=tuple(sorted(roles)),
            nodes=tuple(sorted(nodes_by_repo.get(repo_id, set()))),
            sources=tuple(sorted(sources_by_repo[repo_id])),
        )
        for repo_id, roles in roles_by_repo.items()
    ]
    return tuple(sorted(dependencies))


def _extract_model_dependencies(
    module: ast.Module, embedded: EmbeddedValues
) -> tuple[tuple[ModelDependency, ...], tuple[ExtractionBlocker, ...]]:
    """Replay and classify primary/ControlNet/upscaler model dependencies."""
    evidence, blockers = _resolve_model_assignments(module, embedded)
    return _merge_model_dependencies(evidence), tuple(sorted(blockers))


def extract_workflow_dependencies(workflow_path: Path) -> WorkflowDependencies:
    try:
        source = workflow_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        blocker = ExtractionBlocker(code="template-read-failed", detail=str(error))
        return WorkflowDependencies(workflow_path=workflow_path, blockers=(blocker,))

    try:
        module = ast.parse(source, filename=str(workflow_path))
    except SyntaxError as error:
        source_location = None
        if error.lineno is not None and error.offset is not None:
            source_location = SourceLocation(line=error.lineno, column=max(error.offset - 1, 0))
        blocker = ExtractionBlocker(code="template-parse-failed", detail=error.msg, source=source_location)
        return WorkflowDependencies(workflow_path=workflow_path, blockers=(blocker,))

    embedded_values = _extract_embedded_values(module)
    library_names, library_blockers = _extract_library_names(module)
    model_dependencies, model_blockers = _extract_model_dependencies(module, embedded_values)
    blockers = tuple(sorted(library_blockers + model_blockers))
    return WorkflowDependencies(
        workflow_path=workflow_path,
        model_dependencies=model_dependencies,
        library_names=library_names,
        blockers=blockers,
    )
