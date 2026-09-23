"""`DriverSpec` restates facts that live on the driver class, and the two must not drift.

The orchestrator decides which parameters a node shows from these: whether a pipeline produces video,
what frame rate to offer, whether a mask input makes sense. Reading them off the driver means importing
it, and every driver subclasses a diffusers pipeline block -- so the orchestrator would need the whole
execution stack to answer whether a checkbox appears. Declaring them here is the trade; this test is
what keeps the declaration honest.
"""

from __future__ import annotations

import pytest

pytest.importorskip("diffusers", reason="Reads real driver classes; run `make test/exec`.")

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import (  # noqa: E402
    _DRIVER_REGISTRY,
    DriverSpec,
    get_driver_class,
    get_driver_spec,
    supported_pipeline_classes,
)

CASES = sorted(_DRIVER_REGISTRY.items())


@pytest.mark.parametrize(("pipeline_class", "spec"), CASES, ids=[name for name, _ in CASES])
def test_the_spec_matches_the_driver_it_describes(pipeline_class: str, spec: DriverSpec) -> None:
    driver_class = get_driver_class(pipeline_class)

    assert driver_class is not None, f"{pipeline_class}: `target` does not resolve to a driver."
    assert spec.produces_video == driver_class.produces_video
    assert spec.video_fps == driver_class.video_fps
    assert spec.supports_inpainting == (driver_class._inpaint_pipeline_class is not None)  # noqa: SLF001


def test_every_declared_target_resolves() -> None:
    """A typo in `target` would read as "unsupported pipeline" rather than as a broken driver."""
    unresolvable = [name for name in supported_pipeline_classes() if get_driver_class(name) is None]

    assert not unresolvable, f"These specs name a driver that does not exist: {unresolvable}."


class TestTheFactsAreReadableWithoutTheDriver:
    """The whole point: these answers must not require importing diffusers."""

    def test_an_unsupported_pipeline_has_no_spec(self) -> None:
        assert get_driver_spec("NotAPipeline") is None

    def test_none_has_no_spec(self) -> None:
        """A node with nothing connected asks this, and must not be a special case at every call site."""
        assert get_driver_spec(None) is None

    def test_a_supported_pipeline_has_one(self) -> None:
        assert get_driver_spec("WanPipeline") is not None
