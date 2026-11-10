"""Create Noise Latents honours `randomize_seed` the way the engine's `SeedParameter` defines it."""

from __future__ import annotations

from unittest.mock import patch

from griptape_nodes.exe_types.core_types import ParameterMode

from modular_diffusion_nodes_library.nodes.empty_latent_node import EmptyLatentNode
from modular_diffusion_nodes_library.nodes.noise_latent_node import NoiseLatentNode


def test_preprocess_picks_a_new_seed_when_randomize_seed_is_on() -> None:
    node = NoiseLatentNode(name="noise")
    node.set_parameter_value("randomize_seed", True)

    with patch("griptape_nodes.exe_types.param_components.seed_parameter.random.randint", return_value=1234):
        node.preprocess()

    assert node.get_parameter_value("seed") == 1234


def test_preprocess_keeps_the_seed_when_randomize_seed_is_off() -> None:
    node = NoiseLatentNode(name="noise")
    node.set_parameter_value("seed", 7)

    node.preprocess()

    assert node.get_parameter_value("seed") == 7


def test_enabling_randomize_seed_locks_the_seed_field() -> None:
    node = NoiseLatentNode(name="noise")
    seed_parameter = node.get_parameter_by_name("seed")
    assert seed_parameter is not None

    node.set_parameter_value("randomize_seed", True)

    assert seed_parameter.allowed_modes == {ParameterMode.OUTPUT}


def test_empty_latent_preprocess_has_no_seed_to_randomize() -> None:
    node = EmptyLatentNode(name="empty")

    node.preprocess()

    assert node.get_parameter_by_name("seed") is None
