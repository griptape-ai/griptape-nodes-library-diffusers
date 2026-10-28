from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from modular_diffusion_nodes_library.artifact_utils import packed_quant_io


def test_apply_packed_quant_reuses_loaded_slot_state(tmp_path, monkeypatch) -> None:
    component = nn.Linear(2, 2)
    pipe = SimpleNamespace(components={"transformer": component})
    slot_dir = tmp_path / "transformer"
    qmap = {"fc": {"weights": "qint8"}}
    packed_quant_io.write_quantization_map(slot_dir, qmap)
    state = {"fc.weight._data": torch.ones(2, 2)}
    load_state = Mock(return_value=state)
    requantize = Mock()

    monkeypatch.setattr(packed_quant_io, "load_slot_state_dict", load_state)
    monkeypatch.setattr("optimum.quanto.requantize", requantize)
    monkeypatch.setattr(packed_quant_io, "_assert_qmodules_frozen", Mock())

    applied = packed_quant_io.apply_packed_quant_from_folder(pipe, tmp_path)

    assert applied == ["transformer"]
    load_state.assert_called_once_with(slot_dir)
    requantize.assert_called_once_with(component, state, qmap, device=torch.device("cpu"))
