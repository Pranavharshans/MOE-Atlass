"""Evidence-bound resolution of the routed universe from one static report.

Static candidates are scored partly from name tokens, so foreign-family module
trees can publish noisy semantic kinds: every SwiGLU ``gate_proj`` Linear
tokenizes like a router, and class names such as ``BailingMoeV3RMSNorm``
tokenize like MoE layers. Consumers must therefore bind to the structure the
scanner actually proved — expert containers published as ``EXPERT_CONTAINER``
components — instead of trusting every candidate that carries a semantic kind.

Name matching survives only as a strictly guarded fallback for selections that
would otherwise be empty: the final dotted path segment must be exactly
``gate`` and the router's layer must publish expert-container or routed-expert
evidence. MoE-layer identity prefers the component the scanner published at
the router's parent block, then the nearest published ancestor block, and only
then a deterministic synthesized key guarded by whole-word ``moe`` markers on
dotted path segments — never substrings inside CamelCase identifiers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..core import ComponentKind, ComponentManifest, make_component_key

_EXACT_GATE_LEAF = "gate"


def _parent_path(module_path: str) -> str:
    return module_path.rpartition(".")[0]


def _leaf(module_path: str) -> str:
    return module_path.rpartition(".")[2]


def has_whole_word_moe_marker(module_path: str) -> bool:
    """Match ``moe`` as a whole word on dotted segments, never CamelCase parts.

    Splitting stops at non-alphanumeric boundaries only, so ``moe_block`` and
    ``sparse-moe`` match while ``BailingMoeV3RMSNorm`` does not — CamelCase
    substrings inside class-derived identifiers are precisely the noise this
    guard exists to reject.
    """

    for segment in module_path.split("."):
        for word in re.split(r"[^A-Za-z0-9]+", segment):
            if word.lower() == "moe":
                return True
    return False


def trusted_routers(components: Sequence[ComponentManifest]) -> tuple[ComponentManifest, ...]:
    """Return the router components bound to published structure evidence.

    A ROUTER component is trusted when its parent block hosts a published
    expert container as a direct child — the topology the scanner proved for a
    real routing decision point. Reports without such a publication fall back
    to strict name guards: the final path segment must be exactly ``gate`` and
    the router's layer must carry expert-container or routed-expert evidence.
    The result is ordered by module path and may be empty.
    """

    routers = [
        component
        for component in components
        if component.kind is ComponentKind.ROUTER and component.layer_index is not None
    ]
    container_parents = {
        _parent_path(component.module_path)
        for component in components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    }
    structured = [
        router
        for router in routers
        if _parent_path(router.module_path) in container_parents
    ]
    if structured:
        return tuple(sorted(structured, key=lambda item: item.module_path))

    containers = [
        component.module_path
        for component in components
        if component.kind is ComponentKind.EXPERT_CONTAINER
    ]
    expert_layers = {
        component.layer_index
        for component in components
        if component.kind in {ComponentKind.EXPERT, ComponentKind.EXPERT_CONTAINER}
        and component.layer_index is not None
    }
    fallback: list[ComponentManifest] = []
    for router in routers:
        if _leaf(router.module_path) != _EXACT_GATE_LEAF:
            continue
        block = _parent_path(router.module_path)
        hosted = any(
            _parent_path(container) == block or container.startswith(f"{block}.")
            for container in containers
        ) or router.layer_index in expert_layers
        if hosted:
            fallback.append(router)
    return tuple(sorted(fallback, key=lambda item: item.module_path))


def bind_moe_layer_key(
    model_key: str,
    components: Sequence[ComponentManifest],
    router: ComponentManifest,
) -> str:
    """Return the canonical MoE-layer identity binding one trusted router.

    Resolution prefers the MOE_LAYER component published at the router's
    parent block; otherwise the nearest published ancestor block; otherwise a
    deterministic key synthesized from the parent block, allowed only when the
    block path carries a whole-word ``moe`` marker. Anything else is genuinely
    unresolvable and raises ``ValueError``.
    """

    if router.layer_index is None:
        raise ValueError("router layer index is not exact")
    block = _parent_path(router.module_path)
    exact = [
        component
        for component in components
        if component.kind is ComponentKind.MOE_LAYER and component.module_path == block
    ]
    if len(exact) > 1:
        raise ValueError(
            f"router on layer {router.layer_index} must bind exactly one MoE layer"
        )
    if len(exact) == 1:
        return exact[0].component_key
    ancestors = sorted(
        (
            component
            for component in components
            if component.kind is ComponentKind.MOE_LAYER
            and router.module_path.startswith(f"{component.module_path}.")
        ),
        key=lambda item: (-len(item.module_path), item.module_path),
    )
    if ancestors:
        return ancestors[0].component_key
    if has_whole_word_moe_marker(block):
        return make_component_key(
            model_key,
            ComponentKind.MOE_LAYER.value,
            block,
            layer_index=router.layer_index,
        )
    raise ValueError(
        f"router on layer {router.layer_index} must bind exactly one MoE layer"
    )


__all__ = [
    "bind_moe_layer_key",
    "has_whole_word_moe_marker",
    "trusted_routers",
]
