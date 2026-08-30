"""MoE naming, structure, and semantic-shape heuristics.

The functions here consume already-collected structural surfaces. They do not
call model methods, inspect tensor values, or assemble public reports.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..core import ComponentKind
from .models import DiscoveryEvidence, DiscoveryFacts, DiscoverySignal
from .surface import ConfigSnapshot, ModuleEntry, child_path, parent_path, qualified_type_name

_TOKEN_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")
_ROUTER_TOKENS = frozenset({"router", "gate", "gates", "gating", "routing"})
_EXPERT_CONTAINER_TOKENS = frozenset({"experts", "expertcontainer", "expertlist"})
_MOE_TOKENS = frozenset({"moe", "mixture", "switch", "sparsemoe", "moelayer"})
# A shared expert is the module/container that owns the shared FFN.  Its
# descendants may inherit ``SharedExpert`` in a concrete class name, but a
# projection (``gate_proj``, ``up_proj``/``down_proj``) or scalar gate is not
# itself that component.  Keep this guard token based so it applies equally
# to foreign families and to alternate shared-expert spellings.
_SHARED_PROJECTION_TOKENS = frozenset(
    {
        "gate",
        "gates",
        "gating",
        "proj",
        "projection",
        "up",
        "down",
        "weight",
        "bias",
        "act",
        "activation",
    }
)


@dataclass(frozen=True)
class CandidateSpec:
    """Internal candidate data before canonical manifests are assembled."""

    kind: ComponentKind
    module_path: str
    layer_index: int | None
    expert_index: int | None
    confidence: float
    evidence: tuple[DiscoveryEvidence, ...]
    routed: bool | None
    shared: bool | None
    warnings: tuple[str, ...]
    tensor_shapes: dict[str, list[int]]


def name_tokens(value: str) -> tuple[str, ...]:
    """Tokenize snake paths and CamelCase class names into exact words."""

    tokens: list[str] = []
    for segment in re.split(r"[^A-Za-z0-9]+", value):
        if not segment:
            continue
        pieces = _TOKEN_RE.findall(segment)
        index = 0
        while index < len(pieces):
            if (
                index + 1 < len(pieces)
                and pieces[index].lower() == "mo"
                and pieces[index + 1].lower() == "e"
            ):
                tokens.append("moe")
                index += 2
                continue
            tokens.append(pieces[index].lower())
            index += 1
    return tuple(tokens)


def _class_tokens(value: object) -> frozenset[str]:
    """Tokenize only the concrete class name, not enclosing test/local scopes."""

    return frozenset(name_tokens(type(value).__name__))


def _layer_index(path: str) -> int | None:
    parts = path.split(".")
    for index, part in enumerate(parts[:-1]):
        if part.lower() in {"layer", "layers", "block", "blocks"}:
            following = parts[index + 1]
            if following.isdigit():
                return int(following)
    return None


def _direct_children(
    modules: Mapping[str, ModuleEntry],
) -> dict[str, list[tuple[str, ModuleEntry]]]:
    children: dict[str, list[tuple[str, ModuleEntry]]] = defaultdict(list)
    for path, entry in modules.items():
        if not path:
            continue
        children[parent_path(path)].append((path.rpartition(".")[2], entry))
    for values in children.values():
        values.sort(key=lambda pair: (pair[0], qualified_type_name(pair[1].value)))
    return dict(children)


def _is_expert_container(entry: ModuleEntry, numeric_child_count: int) -> bool:
    leaf_tokens = frozenset(name_tokens(entry.path.rpartition(".")[2]))
    tokens = leaf_tokens | _class_tokens(entry.value)
    return bool(
        tokens & _EXPERT_CONTAINER_TOKENS
        or ("expert" in tokens and numeric_child_count >= 2)
        or ("experts" in tokens and numeric_child_count >= 1)
    )


def _is_shared_owner(entry: ModuleEntry) -> bool:
    """Identify the shared-expert owner, excluding projections and gates."""

    leaf_tokens = frozenset(name_tokens(entry.path.rpartition(".")[2]))
    class_tokens = _class_tokens(entry.value)
    shared_markers = frozenset({"shared"}) & (leaf_tokens | class_tokens)
    expert_markers = frozenset({"expert", "experts"}) & (leaf_tokens | class_tokens)
    return bool(
        shared_markers
        and expert_markers
        and not (leaf_tokens & _SHARED_PROJECTION_TOKENS)
    )


def _evidence(signal: DiscoverySignal, detail: str, weight: float) -> DiscoveryEvidence:
    return DiscoveryEvidence(signal=signal, detail=detail, weight=float(weight))


def _shape_with_expert_axis(
    shapes: Mapping[str, list[int]],
    expert_count: int | None,
    *,
    packed: bool,
) -> tuple[str, list[int]] | None:
    if expert_count is None:
        return None
    for name, shape in sorted(shapes.items()):
        if packed and len(shape) < 3:
            continue
        if expert_count in shape:
            return name, list(shape)
    return None


def _has_packed_shape(shapes: Mapping[str, list[int]]) -> bool:
    return any(len(shape) >= 3 for shape in shapes.values())


def _shape_detail(name: str, shape: list[int], expert_count: int, *, packed: bool) -> str:
    axis_kind = "packed expert axis" if packed else "expert dimension"
    return (
        f"parameter {name!r} shape {tuple(shape)!r} contains configured {axis_kind} {expert_count}"
    )


def _build_spec(
    kind: ComponentKind,
    module_path: str,
    evidence: Iterable[DiscoveryEvidence],
    *,
    layer_index: int | None,
    expert_index: int | None,
    routed: bool | None,
    shared: bool | None,
    shapes: Mapping[str, list[int]],
    warnings: Iterable[str],
) -> CandidateSpec:
    ordered_evidence = tuple(
        sorted(evidence, key=lambda item: (item.signal.value, item.detail, item.weight))
    )
    confidence = min(1.0, round(sum(item.weight for item in ordered_evidence), 3))
    final_warnings = set(warnings)
    if confidence < 0.60:
        final_warnings.add(
            f"ambiguous candidate: confidence {confidence:.3f} is below the 0.600 review threshold"
        )
    return CandidateSpec(
        kind=kind,
        module_path=module_path,
        layer_index=layer_index,
        expert_index=expert_index,
        confidence=confidence,
        evidence=ordered_evidence,
        routed=routed,
        shared=shared,
        warnings=tuple(sorted(final_warnings)),
        tensor_shapes={name: list(shape) for name, shape in sorted(shapes.items())},
    )


def candidate_specs(
    modules: Mapping[str, ModuleEntry],
    parameter_shapes: Mapping[str, Mapping[str, list[int]]],
    config: ConfigSnapshot,
) -> list[CandidateSpec]:
    """Score semantic candidates from names, children, config, and shapes."""

    children = _direct_children(modules)
    numeric_child_counts = {
        path: sum(leaf.isdigit() for leaf, _ in values) for path, values in children.items()
    }
    container_paths = {
        path
        for path, entry in modules.items()
        if path and _is_expert_container(entry, numeric_child_counts.get(path, 0))
    }
    shared_owner_paths = {
        path for path, entry in modules.items() if path and _is_shared_owner(entry)
    }

    moe_paths: set[str] = set()
    for path, entry in modules.items():
        if not path:
            continue
        role_tokens = frozenset(name_tokens(path.rpartition(".")[2])) | frozenset(
            _class_tokens(entry.value)
        )
        child_values = children.get(path, [])
        child_tokens = [
            frozenset(name_tokens(child_path_value)) | _class_tokens(child.value)
            for child_path_value, child in child_values
        ]
        has_router_child = any(tokens & _ROUTER_TOKENS for tokens in child_tokens)
        has_container_child = any(
            child_path(path, child_leaf) in container_paths for child_leaf, _ in child_values
        )
        if role_tokens & _MOE_TOKENS or (has_router_child and has_container_child):
            moe_paths.add(path)

    specs: list[CandidateSpec] = []
    for path in sorted(modules):
        if not path:
            continue
        entry = modules[path]
        path_tokens = frozenset(name_tokens(path))
        leaf_tokens = frozenset(name_tokens(path.rpartition(".")[2]))
        class_tokens = _class_tokens(entry.value)
        role_tokens = leaf_tokens | class_tokens
        child_values = children.get(path, [])
        child_tokens = [
            frozenset(name_tokens(child_path_value)) | _class_tokens(child.value)
            for child_path_value, child in child_values
        ]
        numeric_children = numeric_child_counts.get(path, 0)
        parent = parent_path(path)
        parent_entry = modules.get(parent)
        parent_class_tokens = (
            _class_tokens(parent_entry.value) if parent_entry is not None else frozenset()
        )
        parent_is_container = parent in container_paths
        parent_is_moe = parent in moe_paths
        parent_child_paths = {
            child_path(parent, child_leaf) for child_leaf, _ in children.get(parent, [])
        }
        has_moe_context = parent_is_moe or bool(parent_child_paths & container_paths)
        has_router_child = any(tokens & _ROUTER_TOKENS for tokens in child_tokens)
        has_container_child = any(
            child_path(path, child_leaf) in container_paths for child_leaf, _ in child_values
        )
        shapes = dict(parameter_shapes.get(path, {}))
        config_detail = (
            f"normalized expert_count={config.expert_count}, "
            f"routed_top_k={config.routed_top_k}, "
            f"shared_expert_count={config.shared_expert_count}"
        )

        is_router = bool(role_tokens & _ROUTER_TOKENS)
        # Restrict shared-expert identity to the component's own leaf/class
        # marker.  Looking at all path tokens made ``shared_expert_gate`` and
        # nested ``shared_expert.*`` projections look like additional shared
        # experts when real model classes carried shared-expert names.
        is_shared = path in shared_owner_paths
        # A path such as ``shared_experts`` can be a container, but it is the
        # shared component rather than a routed expert container.  Keep the
        # path in ``container_paths`` for surrounding MoE-context evidence;
        # only publish it as EXPERT_CONTAINER when it is not shared itself.
        is_container = path in container_paths and not is_shared
        is_expert = bool(
            path.rpartition(".")[2].isdigit()
            and parent_is_container
            and parent not in shared_owner_paths
        )
        is_moe_layer = bool(
            (role_tokens & _MOE_TOKENS or (has_router_child and has_container_child))
            and not (is_router or is_container or is_expert or is_shared)
        )

        if is_moe_layer:
            evidence: list[DiscoveryEvidence] = []
            if path_tokens & _MOE_TOKENS:
                evidence.append(
                    _evidence(
                        DiscoverySignal.PATH_NAME,
                        "module path contains a MoE/mixture marker",
                        0.34,
                    )
                )
            if class_tokens & _MOE_TOKENS:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CLASS_NAME,
                        "module class contains a MoE/mixture marker",
                        0.24,
                    )
                )
            if has_router_child and has_container_child:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CHILD_STRUCTURE,
                        "module has both router-like and expert-container children",
                        0.32,
                    )
                )
            if config.has_valid_field:
                evidence.append(_evidence(DiscoverySignal.CONFIG_FIELD, config_detail, 0.10))
            warnings_for_candidate: list[str] = []
            if not (has_router_child and has_container_child):
                warnings_for_candidate.append(
                    "ambiguous MoE layer: router and expert-container child signals are incomplete"
                )
            specs.append(
                _build_spec(
                    ComponentKind.MOE_LAYER,
                    path,
                    evidence,
                    layer_index=_layer_index(path),
                    expert_index=None,
                    routed=None,
                    shared=None,
                    shapes=shapes,
                    warnings=warnings_for_candidate,
                )
            )

        if is_router:
            evidence = []
            if path_tokens & _ROUTER_TOKENS:
                evidence.append(
                    _evidence(
                        DiscoverySignal.PATH_NAME,
                        "module path contains a router/gate marker",
                        0.30,
                    )
                )
            if class_tokens & _ROUTER_TOKENS:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CLASS_NAME,
                        "module class contains a router/gate marker",
                        0.20,
                    )
                )
            if has_moe_context:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CHILD_STRUCTURE,
                        "router is nested in or adjacent to a detected MoE structure",
                        0.24,
                    )
                )
            if config.has_valid_field:
                evidence.append(_evidence(DiscoverySignal.CONFIG_FIELD, config_detail, 0.12))
            warnings_for_candidate: list[str] = []
            if config.expert_count is not None and shapes:
                match = _shape_with_expert_axis(shapes, config.expert_count, packed=False)
                if match is not None:
                    evidence.append(
                        _evidence(
                            DiscoverySignal.PARAMETER_SHAPE,
                            _shape_detail(match[0], match[1], config.expert_count, packed=False),
                            0.14,
                        )
                    )
                else:
                    warnings_for_candidate.append(
                        "router parameter shapes do not expose a dimension matching "
                        f"configured expert_count={config.expert_count}"
                    )
            if not has_moe_context:
                warnings_for_candidate.append(
                    "ambiguous router: name signal is not accompanied by MoE structure"
                )
            specs.append(
                _build_spec(
                    ComponentKind.ROUTER,
                    path,
                    evidence,
                    layer_index=_layer_index(path),
                    expert_index=None,
                    routed=None,
                    shared=None,
                    shapes=shapes,
                    warnings=warnings_for_candidate,
                )
            )

        if is_container:
            evidence = []
            if path_tokens & _EXPERT_CONTAINER_TOKENS:
                evidence.append(
                    _evidence(
                        DiscoverySignal.PATH_NAME,
                        "module path contains an expert-container marker",
                        0.30,
                    )
                )
            if class_tokens & _EXPERT_CONTAINER_TOKENS or "expert" in class_tokens:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CLASS_NAME,
                        "module class contains an expert-container marker",
                        0.18,
                    )
                )
            if numeric_children >= 2:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CHILD_STRUCTURE,
                        f"expert container exposes {numeric_children} indexed child modules",
                        0.34,
                    )
                )
            if config.has_valid_field:
                evidence.append(_evidence(DiscoverySignal.CONFIG_FIELD, config_detail, 0.12))
            warnings_for_candidate: list[str] = []
            if numeric_children < 2:
                warnings_for_candidate.append(
                    "ambiguous expert container: fewer than two indexed child modules were observed"
                )
            if config.expert_count is not None and _has_packed_shape(shapes):
                match = _shape_with_expert_axis(shapes, config.expert_count, packed=True)
                if match is not None:
                    evidence.append(
                        _evidence(
                            DiscoverySignal.PARAMETER_SHAPE,
                            _shape_detail(match[0], match[1], config.expert_count, packed=True),
                            0.06,
                        )
                    )
                else:
                    warnings_for_candidate.append(
                        "packed expert-container shapes do not expose an axis matching "
                        f"configured expert_count={config.expert_count}"
                    )
            specs.append(
                _build_spec(
                    ComponentKind.EXPERT_CONTAINER,
                    path,
                    evidence,
                    layer_index=_layer_index(path),
                    expert_index=None,
                    routed=None,
                    shared=None,
                    shapes=shapes,
                    warnings=warnings_for_candidate,
                )
            )

        if is_expert:
            evidence = []
            if "expert" in path_tokens or "experts" in path_tokens:
                evidence.append(
                    _evidence(
                        DiscoverySignal.PATH_NAME,
                        "indexed module is below an expert-container path",
                        0.18,
                    )
                )
            if "expert" in class_tokens:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CLASS_NAME,
                        "module class contains an expert marker",
                        0.12,
                    )
                )
            expert_index = int(path.rpartition(".")[2])
            evidence.append(
                _evidence(
                    DiscoverySignal.INDEXED_EXPERT,
                    f"module is indexed child {expert_index} of an expert container",
                    0.34,
                )
            )
            if parent_is_container and numeric_children >= 2:
                evidence.append(
                    _evidence(
                        DiscoverySignal.CHILD_STRUCTURE,
                        "parent exposes multiple indexed expert children",
                        0.20,
                    )
                )
            if config.has_valid_field:
                evidence.append(_evidence(DiscoverySignal.CONFIG_FIELD, config_detail, 0.04))
            warnings_for_candidate: list[str] = []
            if not shapes:
                warnings_for_candidate.append(
                    "expert candidate has no readable parameter shape evidence"
                )
            specs.append(
                _build_spec(
                    ComponentKind.EXPERT,
                    path,
                    evidence,
                    layer_index=_layer_index(path),
                    expert_index=expert_index,
                    routed=True,
                    shared=False,
                    shapes=shapes,
                    warnings=warnings_for_candidate,
                )
            )

        if is_shared:
            evidence = []
            if "shared" in path_tokens or "shared" in class_tokens:
                evidence.append(
                    _evidence(
                        DiscoverySignal.SHARED_NAME,
                        "module path or class contains a shared-expert marker",
                        0.35,
                    )
                )
            if role_tokens & {"expert", "experts"}:
                evidence.append(
                    _evidence(
                        DiscoverySignal.PATH_NAME,
                        "module path or class contains an expert marker",
                        0.16,
                    )
                )
            if has_moe_context or (
                parent_entry is not None
                and "moe" in (frozenset(name_tokens(parent_entry.path)) | parent_class_tokens)
            ):
                evidence.append(
                    _evidence(
                        DiscoverySignal.CHILD_STRUCTURE,
                        "shared expert is nested in a detected MoE context",
                        0.22,
                    )
                )
            if config.shared_expert_count is not None:
                evidence.append(_evidence(DiscoverySignal.CONFIG_FIELD, config_detail, 0.15))
            warnings_for_candidate: list[str] = []
            if not has_moe_context:
                warnings_for_candidate.append(
                    "ambiguous shared expert: no surrounding MoE structure was observed"
                )
            specs.append(
                _build_spec(
                    ComponentKind.SHARED_EXPERT,
                    path,
                    evidence,
                    layer_index=_layer_index(path),
                    expert_index=None,
                    routed=False,
                    shared=True,
                    shapes=shapes,
                    warnings=warnings_for_candidate,
                )
            )

    unique: dict[tuple[str, ComponentKind, int | None], CandidateSpec] = {}
    for spec in specs:
        identity = (spec.module_path, spec.kind, spec.expert_index)
        previous = unique.get(identity)
        spec_key = tuple((item.signal.value, item.detail, item.weight) for item in spec.evidence)
        previous_key = (
            tuple((item.signal.value, item.detail, item.weight) for item in previous.evidence)
            if previous is not None
            else ()
        )
        if previous is None or (spec.confidence, spec_key) > (
            previous.confidence,
            previous_key,
        ):
            unique[identity] = spec
    return sorted(
        unique.values(),
        key=lambda spec: (
            spec.module_path,
            spec.kind.value,
            spec.expert_index if spec.expert_index is not None else -1,
        ),
    )


def _group_counts(specs: Iterable[CandidateSpec], kind: ComponentKind) -> dict[str, int]:
    groups: dict[str, int] = defaultdict(int)
    for spec in specs:
        if spec.kind is kind:
            groups[parent_path(spec.module_path)] += 1
    return dict(groups)


def _fact_count(
    label: str,
    configured: int | None,
    groups: Mapping[str, int],
    warnings: list[str],
) -> tuple[int | None, str | None]:
    ordered_groups = sorted(groups.items())
    if configured is not None:
        for group, count in ordered_groups:
            if count != configured:
                warnings.append(
                    f"{label} configuration={configured} conflicts with per-layer/container "
                    f"count {group}={count}"
                )
        return configured, None
    if not ordered_groups:
        return None, None
    counts = {count for _, count in ordered_groups}
    if len(counts) > 1:
        summary = ", ".join(f"{group}={count}" for group, count in ordered_groups)
        warnings.append(f"conflicting per-layer/container {label} counts: {summary}")
        return None, None
    return ordered_groups[0][1], "static per-layer/container structure"


def facts_from_specs(
    config: ConfigSnapshot,
    specs: Iterable[CandidateSpec],
) -> tuple[DiscoveryFacts, tuple[str, ...]]:
    """Normalize configured facts and per-container structural fallbacks."""

    specs_list = list(specs)
    warnings: list[str] = []
    expert_groups = _group_counts(specs_list, ComponentKind.EXPERT)
    shared_groups = _group_counts(specs_list, ComponentKind.SHARED_EXPERT)
    expert_count, expert_source = _fact_count(
        "expert_count", config.expert_count, expert_groups, warnings
    )
    shared_count, shared_source = _fact_count(
        "shared_expert_count",
        config.shared_expert_count,
        shared_groups,
        warnings,
    )
    if config.expert_count is not None:
        expert_source = config.expert_count_source
    if config.shared_expert_count is not None:
        shared_source = config.shared_expert_count_source
    if (
        config.routed_top_k is not None
        and expert_count is not None
        and config.routed_top_k > expert_count
    ):
        warnings.append(
            f"routed_top_k configuration={config.routed_top_k} exceeds expert_count={expert_count}"
        )
    return (
        DiscoveryFacts(
            expert_count=expert_count,
            expert_count_source=expert_source,
            routed_top_k=config.routed_top_k,
            routed_top_k_source=config.routed_top_k_source,
            shared_expert_count=shared_count,
            shared_expert_count_source=shared_source,
        ),
        tuple(sorted(set(warnings))),
    )


__all__ = ["CandidateSpec", "candidate_specs", "facts_from_specs", "name_tokens"]
