"""Compatibility exports for the package-owned model-free fixtures."""

from moeatlas.fixtures import SyntheticConfig, SyntheticMoE, SyntheticParameter

from .mixtral import FakeParameter, MixtralConfig, MixtralConfigMapping, MixtralForCausalLM
from .qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeConfigMapping,
    Qwen3MoeForCausalLM,
)

__all__ = [
    "FakeParameter",
    "MixtralConfig",
    "MixtralConfigMapping",
    "MixtralForCausalLM",
    "Qwen3MoeConfig",
    "Qwen3MoeConfigMapping",
    "Qwen3MoeForCausalLM",
    "SyntheticConfig",
    "SyntheticMoE",
    "SyntheticParameter",
]
