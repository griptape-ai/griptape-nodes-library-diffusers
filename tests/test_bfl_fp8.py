from __future__ import annotations

from typing import Any

import pytest

from modular_diffusion_nodes_library.misc.bfl_fp8 import require_fp8r_support


@pytest.fixture
def torch_module() -> Any:
    torch_module = pytest.importorskip("torch", reason="Reads real torch; run `make test/exec`.")
    if not hasattr(torch_module, "randn"):
        pytest.skip("Requires an installed PyTorch execution environment; run `make test/exec`.")
    return torch_module


def test_cpu_device_is_supported(torch_module: Any) -> None:
    require_fp8r_support(torch_module.device("cpu"))


def test_cuda_device_either_passes_or_raises_a_clear_error(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        pytest.skip("Needs a CUDA device.")

    try:
        require_fp8r_support(torch_module.device("cuda"))
    except RuntimeError as e:
        assert "fp8r" in str(e)
