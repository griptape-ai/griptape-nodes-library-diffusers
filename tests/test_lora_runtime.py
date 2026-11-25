from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modular_diffusion_nodes_library.utils import lora_apply_utils
from modular_diffusion_nodes_library.utils.lora_apply_utils import LoraPipelineRuntimeAdapterStep
from modular_diffusion_nodes_library.utils.lora_spec import LoraSpec


@pytest.fixture
def step(monkeypatch: pytest.MonkeyPatch) -> LoraPipelineRuntimeAdapterStep:
    monkeypatch.setattr(lora_apply_utils, "_load_missing_lora_adapters", lambda _pipe, _loras: None)
    monkeypatch.setattr(lora_apply_utils, "cleanup_memory_caches", lambda: None)
    return LoraPipelineRuntimeAdapterStep({"a": LoraSpec(path="/loras/a.safetensors")})


def test_error_inside_the_generation_is_not_reported_as_a_lora_failure(
    step: LoraPipelineRuntimeAdapterStep,
) -> None:
    pipe = MagicMock()

    with pytest.raises(ValueError, match="encode failed") as exc_info, step.activate(pipe, node_name="Gen"):
        raise ValueError("encode failed")

    assert "Attempted to activate LoRAs" not in str(exc_info.value)
    pipe.delete_adapters.assert_called_once()


def test_activation_failure_is_reported_as_a_lora_failure(step: LoraPipelineRuntimeAdapterStep) -> None:
    pipe = MagicMock()
    pipe.set_adapters.side_effect = ValueError("bad adapter")

    with pytest.raises(RuntimeError, match="Gen: Attempted to activate LoRAs"), step.activate(pipe, node_name="Gen"):
        pytest.fail("the body should not run when activation fails")

    pipe.delete_adapters.assert_called_once()


def test_adapters_are_released_after_a_successful_generation(step: LoraPipelineRuntimeAdapterStep) -> None:
    pipe = MagicMock()

    with step.activate(pipe) as active_pipe:
        assert active_pipe is pipe

    pipe.delete_adapters.assert_called_once()
