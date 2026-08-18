"""Standard-library Mixtral layout fixtures for adapter tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class MixtralConfig:
    model_type: str = "mixtral"
    architectures: tuple[str, ...] = ("MixtralForCausalLM",)
    num_hidden_layers: int = 2
    hidden_size: int = 8
    intermediate_size: int = 16
    num_local_experts: int = 4
    num_experts_per_tok: int = 2


@dataclass(frozen=True)
class FakeParameter:
    shape: tuple[int, ...]


class _FakeModule:
    def __init__(self, **attributes: object) -> None:
        self.__dict__.update(attributes)


class MixtralForCausalLM:
    """A named-module tree with either the indexed or packed public layout."""

    def __init__(
        self,
        *,
        layout: str = "legacy",
        prefix: str = "",
        config: object | None = None,
    ) -> None:
        if layout not in {"legacy", "packed"}:
            raise ValueError("layout must be legacy or packed")
        self.config = config if config is not None else MixtralConfig()
        self.layout = layout
        self.prefix = prefix
        self._modules: list[tuple[str, object]] = []
        self._parameters: list[tuple[str, FakeParameter]] = []
        self._build()

    def _path(self, *parts: object) -> str:
        values = [str(part) for part in parts if str(part)]
        return ".".join(values)

    def _build(self) -> None:
        config = self.config

        def value(name: str) -> object:
            if isinstance(config, Mapping):
                return config[name]
            return getattr(config, name)

        layers = int(value("num_hidden_layers"))
        experts = int(value("num_local_experts"))
        hidden = int(value("hidden_size"))
        intermediate = int(value("intermediate_size"))
        layer_root = self._path(self.prefix, "layers")
        self._modules.append((layer_root, _FakeModule()))
        for layer_index in range(layers):
            layer_path = self._path(layer_root, layer_index)
            self._modules.append((layer_path, _FakeModule()))
            if self.layout == "legacy":
                moe_path = self._path(layer_path, "block_sparse_moe")
                gate_path = self._path(moe_path, "gate")
                experts_path = self._path(moe_path, "experts")
                self._modules.extend(
                    (
                        (
                            moe_path,
                            _FakeModule(
                                num_experts=experts,
                                top_k=int(value("num_experts_per_tok")),
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
                            (expert_path, _FakeModule()),
                            (self._path(expert_path, "w1"), _FakeModule()),
                            (self._path(expert_path, "w2"), _FakeModule()),
                            (self._path(expert_path, "w3"), _FakeModule()),
                            (self._path(expert_path, "act_fn"), _FakeModule()),
                        )
                    )
                    self._parameters.extend(
                        (
                            (
                                self._path(expert_path, "w1.weight"),
                                FakeParameter((intermediate, hidden)),
                            ),
                            (
                                self._path(expert_path, "w2.weight"),
                                FakeParameter((hidden, intermediate)),
                            ),
                            (
                                self._path(expert_path, "w3.weight"),
                                FakeParameter((intermediate, hidden)),
                            ),
                        )
                    )
            else:
                mlp_path = self._path(layer_path, "mlp")
                gate_path = self._path(mlp_path, "gate")
                experts_path = self._path(mlp_path, "experts")
                self._modules.extend(
                    (
                        (
                            mlp_path,
                            _FakeModule(top_k=int(value("num_experts_per_tok"))),
                        ),
                        (
                            gate_path,
                            _FakeModule(
                                num_experts=experts,
                                top_k=int(value("num_experts_per_tok")),
                            ),
                        ),
                        (experts_path, _FakeModule(num_experts=experts)),
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
                            FakeParameter((experts, 2 * intermediate, hidden)),
                        ),
                        (
                            self._path(experts_path, "down_proj"),
                            FakeParameter((experts, hidden, intermediate)),
                        ),
                    )
                )

    def named_modules(self):
        return iter((("", self), *self._modules))

    def named_parameters(self):
        return iter(self._parameters)


class MixtralConfigMapping(dict[str, object]):
    """Mapping-shaped config with the same canonical fields as MixtralConfig."""

    def __init__(self, config: MixtralConfig | None = None) -> None:
        source = config or MixtralConfig()
        super().__init__(
            {
                "model_type": source.model_type,
                "architectures": list(source.architectures),
                "num_hidden_layers": source.num_hidden_layers,
                "hidden_size": source.hidden_size,
                "intermediate_size": source.intermediate_size,
                "num_local_experts": source.num_local_experts,
                "num_experts_per_tok": source.num_experts_per_tok,
            }
        )


__all__ = ["FakeParameter", "MixtralConfig", "MixtralConfigMapping", "MixtralForCausalLM"]
