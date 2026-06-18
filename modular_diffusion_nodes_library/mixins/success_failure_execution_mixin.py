import logging
from collections.abc import Callable
from typing import Any


class SuccessFailureExecutionMixin:
    """Mixin that wraps a callable in the standard status-tracking try/except.

    Requires the host class to also inherit from SuccessFailureNode.
    """

    def _run_with_status(
        self,
        fn: Callable[[], Any],
        *,
        success_msg: str,
        failure_log: str,
        logger: logging.Logger,
        on_error: Callable[[], None] | None = None,
    ) -> Any:
        try:
            result = fn()
            self._set_status_results(was_successful=True, result_details=success_msg)  # type: ignore[attr-defined]
            return result
        except Exception as e:
            logger.exception("%s: %s", self.name, failure_log)  # type: ignore[attr-defined]
            if on_error is not None:
                on_error()
            self._set_status_results(was_successful=False, result_details=str(e))  # type: ignore[attr-defined]
            self._handle_failure_exception(e)  # type: ignore[attr-defined]
