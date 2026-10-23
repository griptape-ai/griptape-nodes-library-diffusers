"""Mixin that removes parameters by request, or locally while the node loads."""

from collections.abc import Generator
from contextlib import contextmanager

from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class ParameterRemovalMixin:
    """Routes ``remove_parameter_element_by_name`` through ``RemoveParameterFromNodeRequest``.

    Inside ``removing_locally_while_loading(initial_setup=True)`` parameters are removed from the node only,
    because loading must not edit the graph. That is safe only for parameters no loaded connection targets.
    """

    _removes_locally: bool = False

    @contextmanager
    def removing_locally_while_loading(self, *, initial_setup: bool) -> Generator[None, None, None]:
        previous = self._removes_locally
        self._removes_locally = previous or initial_setup
        try:
            yield
        finally:
            self._removes_locally = previous

    def remove_parameter_element_by_name(self, element_name: str) -> None:
        if self._removes_locally:
            super().remove_parameter_element_by_name(element_name)  # type: ignore[misc]
            return
        # HACK: `node.remove_parameter_element_by_name` does not remove connections so we need to use the retained mode request which does.  # noqa: FIX004
        # To avoid updating a ton of callers, we just override this method here.
        # TODO: Remove after https://github.com/griptape-ai/griptape-nodes/issues/2511
        if self.get_element_by_name_and_type(element_name):  # type: ignore[attr-defined]
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(parameter_name=element_name, node_name=self.name)  # type: ignore[attr-defined]
            )
