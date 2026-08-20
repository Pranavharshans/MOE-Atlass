"""Dependency-free Qwen3.5-MoE packed structure fixtures.

The fixture mirrors only the names and shapes consumed by the static adapter;
it has no tensor runtime and never performs a forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeParameter:
    shape: tuple[int, ...]


@dataclass
class Qwen3_5MoeTextConfig:
    model_type: str = "qwen3_5_moe_text"
    architectures: tuple[str, ...] = ("Qwen3_5MoeForCausalLM",)
    num_hidden_layers: int = 2
    hidden_size: int = 8
    moe_intermediate_size: int = 12
    shared_expert_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    layer_types: tuple[str, ...] = ("full_attention", "linear_attention")
    mlp_only_layers: tuple[int, ...] = ()


@dataclass
class Qwen3_5MoeConfig:
    model_type: str = "qwen3_5_moe"
    architectures: tuple[str, ...] = ("Qwen3_5MoeForConditionalGeneration",)
    text_config: Qwen3_5MoeTextConfig = field(default_factory=Qwen3_5MoeTextConfig)


class _Module:
    def __init__(self, *, config: object | None = None) -> None:
        if config is not None:
            self.config = config


class Qwen3_5MoeModel:
    """Small conditional or text-only named-surface model."""

    def __init__(
        self,
        *,
        surface: str = "conditional",
        layout: str = "packed",
        config: Qwen3_5MoeConfig | Qwen3_5MoeTextConfig | None = None,
        num_layers: int | None = None,
        num_experts: int | None = None,
        hidden_size: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        top_k: int | None = None,
        extra_modules: tuple[str, ...] = (),
        extra_parameters: tuple[str, ...] = (),
    ) -> None:
        if surface not in {"conditional", "text", "bare"}:
            raise ValueError("surface must be conditional, text, or bare")
        if layout not in {"packed", "legacy_indexed", "mixed"}:
            raise ValueError("layout must be packed, legacy_indexed, or mixed")
        if config is None:
            config = Qwen3_5MoeConfig() if surface == "conditional" else Qwen3_5MoeTextConfig()
        self.config = config
        text_config = config.text_config if surface == "conditional" else config
        layers = num_layers if num_layers is not None else text_config.num_hidden_layers
        experts = num_experts if num_experts is not None else text_config.num_experts
        hidden = hidden_size if hidden_size is not None else text_config.hidden_size
        moe = (
            moe_intermediate_size
            if moe_intermediate_size is not None
            else text_config.moe_intermediate_size
        )
        shared = (
            shared_expert_intermediate_size
            if shared_expert_intermediate_size is not None
            else text_config.shared_expert_intermediate_size
        )
        self._modules: dict[str, object] = {"": _Module()}
        if surface == "conditional":
            prefix = "model.language_model"
            self._modules.update(
                {
                    "model": _Module(),
                    "model.language_model": _Module(config=config.text_config),
                }
            )
        elif surface == "text":
            prefix = "model"
            self._modules["model"] = _Module()
        else:
            prefix = ""
        self._modules[f"{prefix}.layers" if prefix else "layers"] = _Module()
        self._parameters: dict[str, FakeParameter] = {}
        for layer in range(layers):
            layer_path = ".".join(part for part in (prefix, "layers", str(layer)) if part)
            self._modules[layer_path] = _Module()
            suffixes = (
                "mlp",
                "mlp.gate",
                "mlp.experts",
                "mlp.experts.act_fn",
                "mlp.shared_expert",
                "mlp.shared_expert.gate_proj",
                "mlp.shared_expert.up_proj",
                "mlp.shared_expert.down_proj",
                "mlp.shared_expert.act_fn",
                "mlp.shared_expert_gate",
            )
            if layout in {"legacy_indexed", "mixed"}:
                suffixes += tuple(f"mlp.experts.{index}" for index in range(experts))
            for suffix in suffixes:
                self._modules[f"{layer_path}.{suffix}"] = _Module()
            packed_parameters = {
                f"{layer_path}.mlp.gate.weight": FakeParameter((experts, hidden)),
                f"{layer_path}.mlp.experts.gate_up_proj": FakeParameter((experts, 2 * moe, hidden)),
                f"{layer_path}.mlp.experts.down_proj": FakeParameter((experts, hidden, moe)),
            }
            indexed_parameters = {
                f"{layer_path}.mlp.experts.{index}.gate_proj.weight": FakeParameter((moe, hidden))
                for index in range(experts)
            }
            indexed_parameters.update(
                {
                    f"{layer_path}.mlp.experts.{index}.up_proj.weight": FakeParameter((moe, hidden))
                    for index in range(experts)
                }
            )
            indexed_parameters.update(
                {
                    f"{layer_path}.mlp.experts.{index}.down_proj.weight": FakeParameter(
                        (hidden, moe)
                    )
                    for index in range(experts)
                }
            )
            self._parameters.update(
                indexed_parameters if layout == "legacy_indexed" else packed_parameters
            )
            self._parameters.update(
                {
                    f"{layer_path}.mlp.gate.weight": FakeParameter((experts, hidden)),
                    f"{layer_path}.mlp.shared_expert.gate_proj.weight": FakeParameter(
                        (shared, hidden)
                    ),
                    f"{layer_path}.mlp.shared_expert.up_proj.weight": FakeParameter(
                        (shared, hidden)
                    ),
                    f"{layer_path}.mlp.shared_expert.down_proj.weight": FakeParameter(
                        (hidden, shared)
                    ),
                    f"{layer_path}.mlp.shared_expert_gate.weight": FakeParameter((1, hidden)),
                }
            )
        for path in extra_modules:
            self._modules[path] = _Module()
        for name in extra_parameters:
            self._parameters[name] = FakeParameter((1,))

    def named_modules(self):
        return tuple(self._modules.items())

    def named_parameters(self):
        return tuple(self._parameters.items())


class Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeModel):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("surface", "conditional")
        super().__init__(**kwargs)


class Qwen3_5MoeForCausalLM(Qwen3_5MoeModel):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("surface", "text")
        super().__init__(**kwargs)


class Qwen3_5MoeHookHandle:
    """Torch-free removable handle for one Qwen3.5 gate hook."""

    def __init__(self, owner: Qwen3_5MoeHookModule, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner.callbacks:
            self.owner.callbacks.remove(self.callback)


class Qwen3_5MoeHookModule(_Module):
    """A Qwen-owned gate surface with the standard forward-hook API."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = path
        self.callbacks: list[object] = []

    def register_forward_hook(self, callback: object) -> Qwen3_5MoeHookHandle:
        self.callbacks.append(callback)
        return Qwen3_5MoeHookHandle(self, callback)

    def fire(self, output: object, *, inputs: tuple[object, ...] = ()) -> None:
        for callback in tuple(self.callbacks):
            callback(self, inputs, output)


class Qwen3_5MoeHookableModel:
    """Conditional/text Qwen3.5 fixture whose routed gates are hookable."""

    def __init__(self, *, surface: str = "conditional", **kwargs: object) -> None:
        if surface == "conditional":
            source = Qwen3_5MoeForConditionalGeneration(**kwargs)
        elif surface == "text":
            source = Qwen3_5MoeForCausalLM(**kwargs)
        else:
            raise ValueError("surface must be conditional or text")
        self.config = source.config
        self.nodes: dict[str, Qwen3_5MoeHookModule] = {}
        self._entries: list[tuple[str, object]] = []
        for path, module in source.named_modules():
            if path.endswith(".mlp.gate"):
                node = Qwen3_5MoeHookModule(path)
                self.nodes[path] = node
                self._entries.append((path, node))
            else:
                self._entries.append((path, module))
        self._parameters = tuple(source.named_parameters())

    def named_modules(self):
        return iter(self._entries)

    def named_parameters(self):
        return iter(self._parameters)


class Qwen3_5MoeHookableForConditionalGeneration(Qwen3_5MoeHookableModel):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("surface", "conditional")
        super().__init__(**kwargs)


class Qwen3_5MoeHookableForCausalLM(Qwen3_5MoeHookableModel):
    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("surface", "text")
        super().__init__(**kwargs)


__all__ = [
    "FakeParameter",
    "Qwen3_5MoeConfig",
    "Qwen3_5MoeTextConfig",
    "Qwen3_5MoeModel",
    "Qwen3_5MoeForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeHookHandle",
    "Qwen3_5MoeHookModule",
    "Qwen3_5MoeHookableModel",
    "Qwen3_5MoeHookableForConditionalGeneration",
    "Qwen3_5MoeHookableForCausalLM",
]
