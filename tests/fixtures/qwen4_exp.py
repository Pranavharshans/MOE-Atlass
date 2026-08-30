"""Dependency-free Qwen4-Exp packed structure fixtures.

Only the static names and shapes consumed by the adapter are represented.  No
PyTorch or Transformers runtime is imported, and fixture parameters expose
shape metadata only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeParameter:
    shape: tuple[int, ...]


@dataclass
class Qwen4ExpTextConfig:
    model_type: str = "qwen4_exp_text"
    num_hidden_layers: int = 2
    hidden_size: int = 8
    moe_intermediate_size: int = 12
    shared_expert_intermediate_size: int = 16
    num_experts: int = 4
    num_experts_per_tok: int = 2
    layer_types: tuple[str, ...] = ("linear_attention", "full_attention")
    mlp_only_layers: tuple[int, ...] = ()


@dataclass
class Qwen4ExpConfig:
    model_type: str = "qwen4_exp"
    architectures: tuple[str, ...] = ("Qwen4ExpForConditionalGeneration",)
    text_config: Qwen4ExpTextConfig = field(default_factory=Qwen4ExpTextConfig)


class _HookHandle:
    def __init__(self, owner: _Module, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner._hooks:
            self.owner._hooks.remove(self.callback)


class _Module:
    def __init__(self, *, config: object | None = None) -> None:
        self._hooks: list[object] = []
        if config is not None:
            self.config = config

    def register_forward_hook(self, callback: object) -> _HookHandle:
        self._hooks.append(callback)
        return _HookHandle(self, callback)

    def fire(self, payload: object) -> None:
        for callback in tuple(self._hooks):
            callback(self, (), payload)


class Qwen4ExpForConditionalGeneration:
    """Small conditional-generation model exposing official packed names."""

    def __init__(
        self,
        *,
        config: Qwen4ExpConfig | None = None,
        num_layers: int | None = None,
        num_experts: int | None = None,
        hidden_size: int | None = None,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_size: int | None = None,
        extra_modules: tuple[str, ...] = (),
        extra_parameters: tuple[str, ...] = (),
    ) -> None:
        config = Qwen4ExpConfig() if config is None else config
        self.config = config
        text_config = config.text_config
        layers = text_config.num_hidden_layers if num_layers is None else num_layers
        experts = text_config.num_experts if num_experts is None else num_experts
        hidden = text_config.hidden_size if hidden_size is None else hidden_size
        moe = (
            text_config.moe_intermediate_size
            if moe_intermediate_size is None
            else moe_intermediate_size
        )
        shared = (
            text_config.shared_expert_intermediate_size
            if shared_expert_intermediate_size is None
            else shared_expert_intermediate_size
        )

        self._modules: dict[str, object] = {
            "": _Module(),
            "model": _Module(),
            "model.language_model": _Module(config=text_config),
            "model.language_model.layers": _Module(),
        }
        self._parameters: dict[str, FakeParameter] = {}
        for layer in range(layers):
            prefix = f"model.language_model.layers.{layer}"
            for suffix in (
                "",
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
            ):
                self._modules[f"{prefix}.{suffix}" if suffix else prefix] = _Module()
            self._parameters.update(
                {
                    f"{prefix}.mlp.gate.weight": FakeParameter((experts, hidden)),
                    f"{prefix}.mlp.experts.gate_up_proj": FakeParameter((experts, 2 * moe, hidden)),
                    f"{prefix}.mlp.experts.down_proj": FakeParameter((experts, hidden, moe)),
                    f"{prefix}.mlp.shared_expert.gate_proj.weight": FakeParameter((shared, hidden)),
                    f"{prefix}.mlp.shared_expert.up_proj.weight": FakeParameter((shared, hidden)),
                    f"{prefix}.mlp.shared_expert.down_proj.weight": FakeParameter((hidden, shared)),
                    f"{prefix}.mlp.shared_expert_gate.weight": FakeParameter((1, hidden)),
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


class Qwen4ExpHookableForConditionalGeneration(Qwen4ExpForConditionalGeneration):
    """Small official-shaped router double for generic capture tests.

    The packed expert tensors remain logical-only: only the real per-layer
    ``mlp.gate`` modules expose forward hooks, and one opaque native tuple is
    emitted by each gate during the single test forward.
    """

    def __init__(self, payload: object, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.payload = payload
        self.calls = 0
        self.output = object()
        self.router_paths = tuple(
            path for path in self._modules if path.endswith(".mlp.gate")
        )

    def __call__(self, **kwargs: object) -> object:
        del kwargs
        self.calls += 1
        for path in self.router_paths:
            self._modules[path].fire(self.payload)  # type: ignore[attr-defined]
        return self.output


__all__ = [
    "FakeParameter",
    "Qwen4ExpConfig",
    "Qwen4ExpTextConfig",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpHookableForConditionalGeneration",
]
