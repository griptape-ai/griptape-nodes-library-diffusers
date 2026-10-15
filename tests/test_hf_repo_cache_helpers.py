from typing import Any

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.user_specified_hf_repo_parameter import (
    UserSpecifiedHuggingFaceRepoParameter,
)


def test_user_specified_repo_parameter_fetches_exact_repo_revisions(monkeypatch) -> None:
    class NodeStub(BaseNode):
        def __init__(self) -> None:
            # Skip BaseNode.__init__, so every attribute the engine reads off a node has to be set
            # here. Empty metadata is the documented fallback in the engine's model policy: the node
            # type comes from the class name and the library is treated as unknown.
            self.metadata = {}
            self._engine = None

        def get_parameter_value(self, param_name: str) -> Any:
            return "org/model"

    parameter = UserSpecifiedHuggingFaceRepoParameter(NodeStub(), "repo_id")
    parameter._invalidate_cached_repo_choices()
    captured: dict[str, str] = {}

    def fake_list_all_repo_revisions_in_cache() -> list[tuple[str, str]]:
        captured["called"] = "yes"
        return [("org/model", "abc123"), ("other/repo", "def456")]

    monkeypatch.setattr(
        "modular_diffusion_nodes_library.parameters.user_specified_hf_repo_parameter.list_all_repo_revisions_in_cache",
        fake_list_all_repo_revisions_in_cache,
    )

    assert parameter.fetch_repo_revisions() == [("org/model", "abc123")]
    assert captured["called"] == "yes"
