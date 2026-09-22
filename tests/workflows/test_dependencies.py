from __future__ import annotations

import ast
import importlib
import importlib.util
import pickle
import runpy
import sys
from pathlib import Path

import pytest

from tests.workflows.dependencies import (
    ExtractionBlocker,
    WorkflowDependencies,
    _extract_embedded_values,
    extract_workflow_dependencies,
)


class TestTemplateBoundary:
    def test_valid_dependency_free_template_is_conclusive(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text("value = 1\n", encoding="utf-8")

        result = extract_workflow_dependencies(workflow_path)

        assert result.conclusive
        assert result.model_dependencies == ()
        assert result.library_names == ()
        assert result.blockers == ()

    def test_missing_template_has_read_blocker(self, tmp_path: Path) -> None:
        result = extract_workflow_dependencies(tmp_path / "missing.py")

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["template-read-failed"]

    def test_invalid_utf8_has_read_blocker(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_bytes(b"\xff\xfe")

        result = extract_workflow_dependencies(workflow_path)

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["template-read-failed"]

    def test_invalid_syntax_has_parse_blocker(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text("if True print('no')\n", encoding="utf-8")

        result = extract_workflow_dependencies(workflow_path)

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["template-parse-failed"]

    def test_template_body_is_not_executed(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text("raise RuntimeError('template executed')\n", encoding="utf-8")

        result = extract_workflow_dependencies(workflow_path)

        assert result.conclusive


class TestEmbeddedValues:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("owner/model", "owner/model"),
            (None, None),
            (["owner/model"], ["owner/model"]),
        ],
    )
    def test_supported_pickle_values(self, value: object, expected: object) -> None:
        payload = repr(pickle.dumps(value))
        module = ast.parse(f"top_level_unique_values_dict = {{'key': pickle.loads({payload})}}")

        values = _extract_embedded_values(module)

        assert values["key"].value == expected
        assert values["key"].error is None

    def test_failed_pickle_is_retained(self) -> None:
        module = ast.parse("top_level_unique_values_dict = {'key': pickle.loads(b'not-a-pickle')}")

        values = _extract_embedded_values(module)

        assert values["key"].error is not None

    def test_unsupported_expression_is_retained(self) -> None:
        module = ast.parse("top_level_unique_values_dict = {'key': get_value()}")

        values = _extract_embedded_values(module)

        assert values["key"].error == "unsupported embedded value expression"


class TestLibraryExtraction:
    def test_literal_library_names_are_exact_sorted_and_deduplicated(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text(
            "CreateNodeRequest(node_type='A', specific_library_name='Library B')\n"
            "CreateNodeRequest(node_type='B', specific_library_name='Library A')\n"
            "CreateNodeRequest(node_type='C', specific_library_name='Library B')\n",
            encoding="utf-8",
        )

        result = extract_workflow_dependencies(workflow_path)

        assert result.conclusive
        assert result.library_names == ("Library A", "Library B")

    def test_dynamic_library_name_blocks_extraction(self, tmp_path: Path) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text(
            "CreateNodeRequest(node_type='A', specific_library_name=library_name)\n",
            encoding="utf-8",
        )

        result = extract_workflow_dependencies(workflow_path)

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-library-name"]


def _write_workflow(tmp_path: Path, source: str) -> Path:
    workflow_path = tmp_path / "workflow.py"
    workflow_path.write_text(source, encoding="utf-8")
    return workflow_path


class TestLoraExtraction:
    def test_load_lora_node_file_paths_are_extracted(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'load_lora'\n"
            "node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='LoadLora', "
            "specific_library_name='Library A', node_name='Load LoRA')) )\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(parameter_name='file_path', default_value='/tmp/first.safetensors')\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='file_path', node_name=node0_name, value='/tmp/second.safetensors', is_output=False\n"
            "    )\n"
        )

        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert result.lora_file_paths == ('/tmp/second.safetensors',)

    def test_dynamic_lora_path_blocks_extraction(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'load_lora'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='file_path', node_name=node0_name, value=some_dynamic_expression(), is_output=False\n"
            "    )\n"
        )

        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-lora-path"]


class TestValuePrecedence:
    """Default/saved-assignment replay precedence for model parameters."""

    def test_default_used_when_no_saved_assignment(self, tmp_path: Path) -> None:
        """The default applies when no saved assignment exists for that node/parameter."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(parameter_name='model', default_value='owner/default-model')\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert len(result.model_dependencies) == 1
        dependency = result.model_dependencies[0]
        assert dependency.repo_id == "owner/default-model"
        assert dependency.roles == ("primary",)
        assert dependency.nodes == ("node0_name",)

    def test_saved_assignment_overrides_default(self, tmp_path: Path) -> None:
        """The final non-output saved assignment is authoritative over the default."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(parameter_name='model', default_value='owner/default-model')\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/saved-model', is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/saved-model"]

    def test_repeated_saved_assignment_final_wins(self, tmp_path: Path) -> None:
        """When the same node/parameter is saved multiple times, the lexically final one wins."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/first-model', is_output=False\n"
            "    )\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/second-model', is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/second-model"]

    def test_saved_assignment_before_its_add_fails_and_default_applies(self, tmp_path: Path) -> None:
        """A saved assignment lexically before the Add that creates its parameter has no effect."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/decoy-model', is_output=False\n"
            "    )\n"
            "    AddParameterToNodeRequest(parameter_name='model', default_value='owner/default-model')\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/default-model"]

    def test_unresolved_saved_assignment_blocks_without_fallback(self, tmp_path: Path) -> None:
        """An unresolved saved assignment blocks and never falls back to the default."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(parameter_name='model', default_value='owner/default-model')\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value=some_dynamic_expression(), is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert result.model_dependencies == ()
        assert [blocker.code for blocker in result.blockers] == ["unresolved-model-dependency"]

    def test_output_assignment_ignored(self, tmp_path: Path) -> None:
        """A literal is_output=True assignment never contributes and the default still applies."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(parameter_name='model', default_value='owner/default-model')\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/output-model', is_output=True\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/default-model"]

    def test_dropdown_alternatives_ignored(self, tmp_path: Path) -> None:
        """ui_options.simple_dropdown alternatives never become dependencies."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    AddParameterToNodeRequest(\n"
            "        parameter_name='model',\n"
            "        default_value='owner/default-model',\n"
            "        ui_options={'simple_dropdown': ['owner/alt-model-a', 'owner/alt-model-b']},\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/default-model"]


class TestValueResolution:
    """Classifying resolved values for in-scope model parameters."""

    def test_direct_string_literal_resolves_to_repo(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/direct-model', is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/direct-model"]

    def test_pickled_table_value_resolves_to_repo(self, tmp_path: Path) -> None:
        payload = repr(pickle.dumps("owner/pickled-model"))
        source = (
            f"top_level_unique_values_dict = {{'k1': pickle.loads({payload})}}\n"
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model',\n"
            "        node_name=node0_name,\n"
            "        value=top_level_unique_values_dict['k1'],\n"
            "        is_output=False,\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert [dependency.repo_id for dependency in result.model_dependencies] == ["owner/pickled-model"]

    @pytest.mark.parametrize("literal", ["None", "''", "[]"])
    def test_empty_values_are_conclusively_no_selection(self, tmp_path: Path, literal: str) -> None:
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            f"    SetParameterValueRequest(parameter_name='model', node_name=node0_name, value={literal}, is_output=False)\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert result.model_dependencies == ()

    def test_dynamic_expression_blocks(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value=some_call(), is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-model-dependency"]

    def test_missing_table_key_blocks(self, tmp_path: Path) -> None:
        payload = repr(pickle.dumps("owner/irrelevant"))
        source = (
            f"top_level_unique_values_dict = {{'other-key': pickle.loads({payload})}}\n"
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model',\n"
            "        node_name=node0_name,\n"
            "        value=top_level_unique_values_dict['missing-key'],\n"
            "        is_output=False,\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-model-dependency"]

    def test_failed_pickle_entry_blocks(self, tmp_path: Path) -> None:
        source = (
            "top_level_unique_values_dict = {'k1': pickle.loads(b'not-a-pickle')}\n"
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model',\n"
            "        node_name=node0_name,\n"
            "        value=top_level_unique_values_dict['k1'],\n"
            "        is_output=False,\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-model-dependency"]

    def test_unsupported_consumed_form_blocks(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=123, is_output=False)\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert not result.conclusive
        assert [blocker.code for blocker in result.blockers] == ["unresolved-model-dependency"]

    def test_recursive_list_of_repos_resolves_to_multiple_dependencies(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='controlnet_model',\n"
            "        node_name=node0_name,\n"
            "        value=['owner/model-a', ['owner/model-b', []]],\n"
            "        is_output=False,\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert {dependency.repo_id for dependency in result.model_dependencies} == {"owner/model-a", "owner/model-b"}
        for dependency in result.model_dependencies:
            assert dependency.roles == ("controlnet",)


class TestDependencyEvidence:
    """Node-identity uncertainty and cross-source evidence merging for ModelDependency."""

    def test_unresolved_node_identity_does_not_block_certain_repo_value(self, tmp_path: Path) -> None:
        """A certain repo-shaped value is retained with unknown node evidence, not blocked."""
        source = (
            "node0_name = 'n0'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model',\n"
            "        node_name=get_dynamic_node_name(),\n"
            "        value='owner/certain-model',\n"
            "        is_output=False,\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert len(result.model_dependencies) == 1
        dependency = result.model_dependencies[0]
        assert dependency.repo_id == "owner/certain-model"
        assert dependency.nodes == ()

    def test_all_three_roles_are_recognized(self, tmp_path: Path) -> None:
        source = (
            "node0_name = 'n0'\n"
            "node1_name = 'n1'\n"
            "node2_name = 'n2'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='owner/primary-model', is_output=False\n"
            "    )\n"
            "with GriptapeNodes.ContextManager().node(node1_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='controlnet_model', node_name=node1_name, value='owner/controlnet-model', is_output=False\n"
            "    )\n"
            "with GriptapeNodes.ContextManager().node(node2_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='upsampler_model', node_name=node2_name, value='owner/upscaler-model', is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        roles_by_repo = {dependency.repo_id: dependency.roles for dependency in result.model_dependencies}
        assert roles_by_repo == {
            "owner/primary-model": ("primary",),
            "owner/controlnet-model": ("controlnet",),
            "owner/upscaler-model": ("upscaler",),
        }

    def test_duplicate_repo_ids_merge_roles_nodes_and_sources(self, tmp_path: Path) -> None:
        """The same repository id across roles/nodes merges into one ModelDependency."""
        source = (
            "node0_name = 'n0'\n"
            "node1_name = 'n1'\n"
            "with GriptapeNodes.ContextManager().node(node0_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='model', node_name=node0_name, value='shared/repo', is_output=False\n"
            "    )\n"
            "with GriptapeNodes.ContextManager().node(node1_name):\n"
            "    SetParameterValueRequest(\n"
            "        parameter_name='controlnet_model', node_name=node1_name, value='shared/repo', is_output=False\n"
            "    )\n"
        )
        result = extract_workflow_dependencies(_write_workflow(tmp_path, source))

        assert result.conclusive
        assert len(result.model_dependencies) == 1
        dependency = result.model_dependencies[0]
        assert dependency.repo_id == "shared/repo"
        assert dependency.roles == ("controlnet", "primary")
        assert dependency.nodes == ("node0_name", "node1_name")
        assert len(dependency.sources) == 2


class TestNonExecutionSentinel:
    """Static extraction never imports, executes, or compiles a workflow template."""

    def test_extraction_never_imports_or_executes_template(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        workflow_path = tmp_path / "workflow.py"
        workflow_path.write_text(
            "import sys\n"
            "sys.modules['tests_workflows_import_sentinel'] = object()\n"
            "raise RuntimeError('workflow template body was executed')\n",
            encoding="utf-8",
        )

        forbidden_calls: list[str] = []

        def _forbid(name: str) -> object:
            def _fail(*_args: object, **_kwargs: object) -> object:
                forbidden_calls.append(name)
                raise AssertionError(f"{name} must not be called by static extraction")

            return _fail

        monkeypatch.setattr(importlib, "import_module", _forbid("importlib.import_module"))
        monkeypatch.setattr(
            importlib.util, "spec_from_file_location", _forbid("importlib.util.spec_from_file_location")
        )
        monkeypatch.setattr(runpy, "run_path", _forbid("runpy.run_path"))

        result = extract_workflow_dependencies(workflow_path)

        assert result.conclusive
        assert forbidden_calls == []
        assert "tests_workflows_import_sentinel" not in sys.modules


class TestPreflightAdapter:
    """preflight._build_preflight_data must be a thin adapter over extract_workflow_dependencies.

    It must parse each discovered workflow exactly once and must never let a blocked
    workflow's absence of repos count as a conclusive "no repos required" workflow.
    """

    def test_blocked_workflow_is_excluded_from_workflows_with_no_repos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests import preflight

        conclusive_empty_result = WorkflowDependencies(workflow_path=Path("conclusive_empty.py"))
        blocked_result = WorkflowDependencies(
            workflow_path=Path("blocked.py"),
            blockers=(ExtractionBlocker(code="unresolved-model-dependency", detail="canonical result is blocked"),),
        )
        canonical_results = {
            "conclusive_empty.py": conclusive_empty_result,
            "blocked.py": blocked_result,
        }
        extraction_call_count = 0

        def _fake_extract_workflow_dependencies(workflow_path: Path) -> WorkflowDependencies:
            nonlocal extraction_call_count
            extraction_call_count += 1
            return canonical_results[workflow_path.name]

        monkeypatch.setattr(preflight, "_discover_workflow_templates", lambda: sorted(canonical_results))
        monkeypatch.setattr(preflight, "extract_workflow_dependencies", _fake_extract_workflow_dependencies)
        monkeypatch.setattr(preflight, "_discover_installed_library_manifests", lambda: {})
        monkeypatch.setattr(preflight, "list_repo_revisions_in_cache", lambda repo_id: [])  # noqa: ARG005

        preflight_data = preflight._build_preflight_data()

        assert extraction_call_count == len(canonical_results)
        assert preflight_data["workflows_with_no_repos"] == ["conclusive_empty.py"]
        assert preflight_data["workflow_extraction_blockers"] == {"blocked.py": blocked_result.blockers}
        assert "conclusive_empty.py" not in preflight_data["workflow_extraction_blockers"]
        assert "blocked.py" not in preflight_data["workflows_with_no_repos"]

    def test_preflight_tracks_missing_lora_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests import preflight

        workflow_path = Path("workflow.py")
        result = WorkflowDependencies(
            workflow_path=workflow_path,
            lora_file_paths=("/tmp/does-not-exist.safetensors",),
        )
        monkeypatch.setattr(preflight, "_discover_workflow_templates", lambda: [workflow_path.name])
        monkeypatch.setattr(preflight, "extract_workflow_dependencies", lambda _: result)
        monkeypatch.setattr(preflight, "_discover_installed_library_manifests", lambda: {})
        monkeypatch.setattr(preflight, "list_repo_revisions_in_cache", lambda repo_id: [])

        preflight_data = preflight._build_preflight_data()

        assert preflight_data["workflow_required_loras"][workflow_path.name] == ("/tmp/does-not-exist.safetensors",)
        assert preflight_data["missing_loras_by_workflow"][workflow_path.name] == ("/tmp/does-not-exist.safetensors",)

    def test_lora_path_is_resolved_to_assets_lora_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests import preflight

        # The repo-local install convention is: user-created LoRAs live under workflows/assets/lora.
        # We accept repo-relative references to that folder, but reject any file outside it.
        # Point the library root at tmp_path so this test doesn't write into the real repo.
        lora_dir = tmp_path / "workflows" / "assets" / "lora"
        lora_dir.mkdir(parents=True)
        lora_file = lora_dir / "example.safetensors"
        lora_file.write_bytes(b"stub")
        monkeypatch.setattr(preflight, "LIBRARY_ROOT", tmp_path)
        monkeypatch.setattr(preflight, "LORA_ASSETS_DIR", lora_dir)

        monkeypatch.setattr(preflight, "_discover_workflow_templates", lambda: [])
        assert preflight._resolve_lora_path("{project_dir}/../workflows/assets/lora/example.safetensors") == lora_file
        assert preflight._resolve_lora_path("../workflows/assets/lora/example.safetensors") == lora_file

        outside = tmp_path / "outside.safetensors"
        outside.write_bytes(b"stub")
        assert preflight._resolve_lora_path(str(outside)) is None
