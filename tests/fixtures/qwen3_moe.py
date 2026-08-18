"""Standard-library Qwen3-MoE reference-layout fixtures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class Qwen3MoeConfig:
    model_type: str | None = "qwen3_moe"
    architectures: tuple[str, ...] = ("Qwen3MoeForCausalLM",)
    num_hidden_layers: int = 4
    hidden_size: int = 8
    intermediate_size: int = 16
    moe_intermediate_size: int = 12
    num_experts: int = 4
    num_experts_per_tok: int = 2
    decoder_sparse_step: int = 2
    norm_topk_prob: bool = True
    mlp_only_layers: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class FakeParameter:
    shape: tuple[int, ...]


class _FakeModule:
    def __init__(self, **attributes: object) -> None:
        self.__dict__.update(attributes)


class Qwen3MoeForCausalLM:
    """A deterministic module tree for the official dense/sparse schedules."""

    def __init__(
        self,
        *,
        layout: str = "legacy_indexed",
        prefix: str = "",
        config: object | None = None,
    ) -> None:
        if layout == "legacy":
            layout = "legacy_indexed"
        if layout not in {"legacy_indexed", "packed"}:
            raise ValueError("layout must be legacy_indexed or packed")
        self.config = config if config is not None else Qwen3MoeConfig()
        self.layout = layout
        self.prefix = prefix
        self._modules: list[tuple[str, object]] = []
        self._parameters: list[tuple[str, FakeParameter]] = []
        self._build()

    def _path(self, *parts: object) -> str:
        return ".".join(str(part) for part in parts if str(part))

    def _value(self, config: object, name: str) -> object:
        if isinstance(config, Mapping):
            return config[name]
        return getattr(config, name)

    def _build(self) -> None:
        config = self.config
        layers = self._value(config, "num_hidden_layers")
        experts = self._value(config, "num_experts")
        hidden = self._value(config, "hidden_size")
        dense_intermediate = self._value(config, "intermediate_size")
        sparse_intermediate = self._value(config, "moe_intermediate_size")
        top_k = self._value(config, "num_experts_per_tok")
        step = self._value(config, "decoder_sparse_step")
        only_dense = set(self._value(config, "mlp_only_layers"))
        layer_root = self._path(self.prefix, "layers")
        self._modules.append((layer_root, _FakeModule()))

        for layer_index in range(layers):
            layer_path = self._path(layer_root, layer_index)
            self._modules.append((layer_path, _FakeModule()))
            sparse = layer_index not in only_dense and (layer_index + 1) % step == 0
            mlp_path = self._path(layer_path, "mlp")
            if not sparse:
                self._modules.extend(
                    (
                        (
                            mlp_path,
                            _FakeModule(
                                hidden_size=hidden,
                                intermediate_size=dense_intermediate,
                            ),
                        ),
                        (self._path(mlp_path, "gate_proj"), _FakeModule()),
                        (self._path(mlp_path, "up_proj"), _FakeModule()),
                        (self._path(mlp_path, "down_proj"), _FakeModule()),
                        (self._path(mlp_path, "act_fn"), _FakeModule()),
                    )
                )
                self._parameters.extend(
                    (
                        (
                            self._path(mlp_path, "gate_proj.weight"),
                            FakeParameter((dense_intermediate, hidden)),
                        ),
                        (
                            self._path(mlp_path, "up_proj.weight"),
                            FakeParameter((dense_intermediate, hidden)),
                        ),
                        (
                            self._path(mlp_path, "down_proj.weight"),
                            FakeParameter((hidden, dense_intermediate)),
                        ),
                    )
                )
                continue

            gate_path = self._path(mlp_path, "gate")
            experts_path = self._path(mlp_path, "experts")
            if self.layout == "legacy_indexed":
                self._modules.extend(
                    (
                        (
                            mlp_path,
                            _FakeModule(
                                num_experts=experts,
                                top_k=top_k,
                                norm_topk_prob=self._value(config, "norm_topk_prob"),
                            ),
                        ),
                        (gate_path, _FakeModule()),
                        (experts_path, _FakeModule()),
                    )
                )
                self._parameters.append(
                    (
                        self._path(gate_path, "weight"),
                        FakeParameter((experts, hidden)),
                    )
                )
                for expert_index in range(experts):
                    expert_path = self._path(experts_path, expert_index)
                    self._modules.extend(
                        (
                            (
                                expert_path,
                                _FakeModule(
                                    hidden_size=hidden,
                                    intermediate_size=sparse_intermediate,
                                ),
                            ),
                            (self._path(expert_path, "gate_proj"), _FakeModule()),
                            (self._path(expert_path, "up_proj"), _FakeModule()),
                            (self._path(expert_path, "down_proj"), _FakeModule()),
                            (self._path(expert_path, "act_fn"), _FakeModule()),
                        )
                    )
                    self._parameters.extend(
                        (
                            (
                                self._path(expert_path, "gate_proj.weight"),
                                FakeParameter((sparse_intermediate, hidden)),
                            ),
                            (
                                self._path(expert_path, "up_proj.weight"),
                                FakeParameter((sparse_intermediate, hidden)),
                            ),
                            (
                                self._path(expert_path, "down_proj.weight"),
                                FakeParameter((hidden, sparse_intermediate)),
                            ),
                        )
                    )
            else:
                self._modules.extend(
                    (
                        (mlp_path, _FakeModule()),
                        (
                            gate_path,
                            _FakeModule(
                                top_k=top_k,
                                num_experts=experts,
                                norm_topk_prob=self._value(config, "norm_topk_prob"),
                                hidden_dim=hidden,
                            ),
                        ),
                        (
                            experts_path,
                            _FakeModule(
                                num_experts=experts,
                                hidden_dim=hidden,
                                intermediate_dim=sparse_intermediate,
                            ),
                        ),
                        (self._path(experts_path, "act_fn"), _FakeModule()),
                    )
                )
                self._parameters.extend(
                    (
                        (
                            self._path(gate_path, "weight"),
                            FakeParameter((experts, hidden)),
                        ),
                        (
                            self._path(experts_path, "gate_up_proj"),
                            FakeParameter((experts, 2 * sparse_intermediate, hidden)),
                        ),
                        (
                            self._path(experts_path, "down_proj"),
                            FakeParameter((experts, hidden, sparse_intermediate)),
                        ),
                    )
                )

    def named_modules(self):
        return iter((("", self), *self._modules))

    def named_parameters(self):
        return iter(self._parameters)


class Qwen3MoeConfigMapping(dict[str, object]):
    """Mapping-shaped equivalent of Qwen3MoeConfig."""

    def __init__(self, config: Qwen3MoeConfig | None = None) -> None:
        source = config or Qwen3MoeConfig()
        super().__init__(
            {
                "model_type": source.model_type,
                "architectures": list(source.architectures),
                "num_hidden_layers": source.num_hidden_layers,
                "hidden_size": source.hidden_size,
                "intermediate_size": source.intermediate_size,
                "moe_intermediate_size": source.moe_intermediate_size,
                "num_experts": source.num_experts,
                "num_experts_per_tok": source.num_experts_per_tok,
                "decoder_sparse_step": source.decoder_sparse_step,
                "norm_topk_prob": source.norm_topk_prob,
                "mlp_only_layers": list(source.mlp_only_layers),
            }
        )


__all__ = [
    "FakeParameter",
    "Qwen3MoeConfig",
    "Qwen3MoeConfigMapping",
    "Qwen3MoeForCausalLM",
]
