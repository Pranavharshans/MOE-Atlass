"""Compatibility exports for the package-owned model-free fixtures."""

from moeatlas.fixtures import SyntheticConfig, SyntheticMoE, SyntheticParameter

from .mixtral import FakeParameter, MixtralConfig, MixtralConfigMapping, MixtralForCausalLM

__all__ = [
    "FakeParameter",
    "MixtralConfig",
    "MixtralConfigMapping",
    "MixtralForCausalLM",
    "SyntheticConfig",
    "SyntheticMoE",
    "SyntheticParameter",
]
