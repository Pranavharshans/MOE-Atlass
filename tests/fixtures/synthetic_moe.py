"""A deterministic, torch-free MoE-shaped object for discovery tests."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SyntheticParameter:
    """Minimal parameter stand-in exposing only a tuple-like ``shape``."""

    shape: tuple[int, ...]


class _Node:
    """Tiny named-module tree with deterministic traversal and no mutation."""

    def __init__(
        self,
        *,
        children: dict[str, _Node] | None = None,
        parameters: dict[str, SyntheticParameter] | None = None,
    ) -> None:
        self._children = dict(children or {})
        self._parameters = dict(parameters or {})

    def named_modules(self):
        yield "", self
        for child_name, child in self._children.items():
            for nested_name, nested_module in child.named_modules():
                full_name = child_name if not nested_name else f"{child_name}.{nested_name}"
                yield full_name, nested_module

    def named_parameters(self):
        for parameter_name, parameter in self._parameters.items():
            yield parameter_name, parameter
        for child_name, child in self._children.items():
            for nested_name, parameter in child.named_parameters():
                yield f"{child_name}.{nested_name}", parameter


class SyntheticRouter(_Node):
    pass


class SyntheticExpert(_Node):
    pass


class SyntheticExperts(_Node):
    pass


class SyntheticSharedExpert(_Node):
    pass


class SyntheticMoEBlock(_Node):
    pass


class SyntheticLayers(_Node):
    pass


@dataclass(frozen=True)
class SyntheticConfig:
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    num_shared_experts: int = 1


class SyntheticMoE(_Node):
    """Two-layer MoE with four routed and one shared expert per layer."""

    def __init__(self) -> None:
        def make_block() -> SyntheticMoEBlock:
            experts = SyntheticExperts(
                children={
                    str(index): SyntheticExpert(
                        parameters={
                            "w1.weight": SyntheticParameter((16, 8)),
                            "w2.weight": SyntheticParameter((8, 16)),
                        }
                    )
                    for index in range(4)
                }
            )
            return SyntheticMoEBlock(
                children={
                    "router": SyntheticRouter(parameters={"weight": SyntheticParameter((4, 8))}),
                    "experts": experts,
                    "shared_expert": SyntheticSharedExpert(
                        parameters={"weight": SyntheticParameter((8, 8))}
                    ),
                }
            )

        super().__init__(
            children={"layers": SyntheticLayers(children={"0": make_block(), "1": make_block()})}
        )
        self.config = SyntheticConfig()


__all__ = ["SyntheticMoE", "SyntheticParameter"]
