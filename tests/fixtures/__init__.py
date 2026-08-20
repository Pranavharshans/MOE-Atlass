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
from .qwen3_5_moe import (
    FakeParameter as Qwen35FakeParameter,
)
from .qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeModel,
    Qwen3_5MoeTextConfig,
)

__all__ += [
    "Qwen35FakeParameter",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]
