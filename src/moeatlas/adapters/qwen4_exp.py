"""Static structure adapter for the official Qwen4Exp conditional wrapper.

This module deliberately stays model-runtime independent.  It accepts only the
exact ``qwen4_exp``/``Qwen4ExpForConditionalGeneration`` identity, inspects
named module/parameter paths and shapes, and publishes unverified STRUCTURE
evidence.  It does not import Transformers or read tensor values.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .. import __version__
from ..core import (
    CapabilityLabel,
    CaptureProvenance,
    CaptureSource,
    ComponentKind,
    ComponentManifest,
    ModelManifest,
    Provenance,
    make_component_key,
)
from ..discovery import (
    DiscoveryCandidate,
    DiscoveryEvidence,
    DiscoveryFacts,
    DiscoveryReport,
    DiscoverySignal,
)
from .contracts import AdapterDescriptor, AdapterDetection

_MISSING = object()
_NAME = "huggingface-qwen4-exp-static"
_VERSION = "1.0"
_METHOD = "qwen4-exp-static-structure-v1"
_LAYOUT = "packed"
_PACKED_WARNING = "packed expert slices are logical and are not independently hookable"
_UNSUPPORTED = "Qwen4Exp static structure is unsupported on this object"
_FAMILY = "Qwen4Exp family identity is missing or conflicting"
_CONFIG = "Qwen4Exp configuration fields or layer schedule are invalid"
_TOPOLOGY = "Qwen4Exp module topology is incomplete, conflicting, or mixed"
_SHAPES = "Qwen4Exp parameter shapes do not match the packed layout"
_PARAM_ROOT = "Qwen4Exp parameter roots are conflicting or malformed"
_ROOT = "model.language_model"
_OUTER_ARCH = "Qwen4ExpForConditionalGeneration"
_LAYER_TYPES = frozenset({"full_attention", "linear_attention"})
_INT_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "moe_intermediate_size",
    "shared_expert_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
)

_DESCRIPTOR = AdapterDescriptor(
    name=_NAME,
    version=_VERSION,
    architecture_families=("qwen4_exp",),
    compatibility_notes=(
        "official Qwen4Exp conditional-generation packed text surface is supported",
        "shared experts are structural and not router targets",
        "structure-only; routing and FP8 model certification are not provided",
    ),
)


@dataclass(frozen=True)
class _ConfigFacts:
    layers: int
    hidden: int
    moe_intermediate: int
    shared_intermediate: int
    experts: int
    top_k: int
    family_evidence: tuple[str, ...]


@dataclass(frozen=True)
class _Layer:
    index: int
    mlp: str
    gate: str
    experts: str
    shared_expert: str
    shared_gate: str


@dataclass(frozen=True)
class _Analysis:
    config: _ConfigFacts | None
    layers: tuple[_Layer, ...]
    parameters: dict[str, object]
    warnings: tuple[str, ...]
    family_evidence: tuple[str, ...]
    topology_valid: bool = False
    shapes_valid: bool = False

    @property
    def valid(self) -> bool:
        return (
            self.config is not None
            and len(self.layers) == self.config.layers
            and self.topology_valid
            and self.shapes_valid
        )


class _ConfigReadError(Exception):
    def __init__(self, field: str, cause: Exception) -> None:
        self.field = field
        self.cause = cause
        super().__init__(field)


def _failure(operation: str, exc: Exception) -> str:
    return f"{operation} failed with {type(exc).__name__}"


def _value(config: object, field: str) -> object:
    if isinstance(config, Mapping):
        try:
            return config[field] if field in config else _MISSING
        except Exception as exc:
            raise _ConfigReadError(field, exc) from exc
    try:
        return getattr(config, field)
    except AttributeError:
        return _MISSING
    except Exception as exc:
        raise _ConfigReadError(field, exc) from exc


def _model_config(model: object) -> tuple[object | None, str | None]:
    try:
        config = getattr(model, "config")
    except AttributeError:
        return None, "model.config is unavailable"
    except Exception as exc:
        return None, _failure("model.config access", exc)
    return (config, None) if config is not None else (None, "model.config is unavailable")


def _sequence(value: object) -> tuple[object, ...] | None:
    if not isinstance(value, list | tuple):
        return None
    try:
        return tuple(value)
    except Exception:
        return None


def _identity(config: object) -> tuple[tuple[str, ...], object | None, str | None]:
    try:
        model_type = _value(config, "model_type")
        architectures = _value(config, "architectures")
    except _ConfigReadError as exc:
        return (), None, _failure(f"config.{exc.field} access", exc.cause)
    values = _sequence(architectures)
    if (
        type(model_type) is not str
        or values is None
        or len(values) != 1
        or type(values[0]) is not str
        or model_type != "qwen4_exp"
        or values[0] != _OUTER_ARCH
    ):
        return (), None, _FAMILY
    try:
        nested = _value(config, "text_config")
        nested_type = (
            _value(nested, "model_type")
            if nested is not _MISSING and nested is not None
            else _MISSING
        )
    except _ConfigReadError as exc:
        return (), None, _failure(f"config.text_config.{exc.field} access", exc.cause)
    if (
        nested is _MISSING
        or nested is None
        or type(nested_type) is not str
        or nested_type != "qwen4_exp_text"
    ):
        return (), None, _FAMILY
    return (
        ("architecture:qwen4-exp-conditional-allowlist", "model_type:qwen4_exp"),
        nested,
        None,
    )


def _strict_config(
    config: object, identity: tuple[str, ...], nested: object
) -> tuple[_ConfigFacts | None, str | None]:
    values: dict[str, int] = {}
    for field in _INT_FIELDS:
        try:
            raw = _value(nested, field)
        except _ConfigReadError as exc:
            return None, _failure(f"config.text_config.{exc.field} access", exc.cause)
        if type(raw) is not int or raw <= 0:
            return None, _CONFIG
        values[field] = raw
    if values["num_experts_per_tok"] > values["num_experts"]:
        return None, _CONFIG
    try:
        layer_types = _value(nested, "layer_types")
    except _ConfigReadError as exc:
        return None, _failure(f"config.text_config.{exc.field} access", exc.cause)
    layer_values = _sequence(layer_types)
    if layer_values is None or len(layer_values) != values["num_hidden_layers"]:
        return None, _CONFIG
    if any(type(item) is not str or item not in _LAYER_TYPES for item in layer_values):
        return None, _CONFIG
    # Some Transformers configs carry this inherited field.  All Qwen4Exp
    # layers are MoE, so an omitted or empty value is the only valid form.
    try:
        mlp_only = _value(nested, "mlp_only_layers")
    except _ConfigReadError as exc:
        return None, _failure(f"config.text_config.{exc.field} access", exc.cause)
    if mlp_only is not _MISSING:
        mlp_values = _sequence(mlp_only)
        if mlp_values is None or mlp_values:
            return None, _CONFIG
    return (
        _ConfigFacts(
            layers=values["num_hidden_layers"],
            hidden=values["hidden_size"],
            moe_intermediate=values["moe_intermediate_size"],
            shared_intermediate=values["shared_expert_intermediate_size"],
            experts=values["num_experts"],
            top_k=values["num_experts_per_tok"],
            family_evidence=identity,
        ),
        None,
    )


def _pairs(surface: object) -> object:
    if isinstance(surface, Mapping):
        return iter(surface.items())
    return iter(surface)  # type: ignore[arg-type]


def _collect(model: object, name: str) -> tuple[dict[str, object], tuple[str, ...]]:
    try:
        method = getattr(model, name)
    except AttributeError:
        return {}, (f"{name}() is unavailable",)
    except Exception as exc:
        return {}, (_failure(f"{name}() access", exc),)
    if not callable(method):
        return {}, (f"{name}() is unavailable",)
    try:
        iterator = _pairs(method())
    except Exception as exc:
        return {}, (_failure(f"{name}() traversal", exc),)
    entries: dict[str, object] = {}
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            return {}, (_failure(f"{name}() iteration", exc),)
        if not isinstance(item, tuple | list) or len(item) != 2:
            return {}, (f"{name}() surface is malformed or duplicated",)
        path, value = item
        if type(path) is not str or path in entries:
            return {}, (f"{name}() surface is malformed or duplicated",)
        entries[path] = value
    if not entries:
        return {}, (f"{name}() surface is empty",)
    return dict(sorted(entries.items())), ()


def _index(token: str) -> int | None:
    if not token or not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
        return None
    return int(token)


def _layer_parse(path: str) -> tuple[str, int, tuple[str, ...]] | None:
    if not path:
        return None
    parts = tuple(path.split("."))
    if any(not part for part in parts):
        raise ValueError
    positions = [i for i, part in enumerate(parts) if part == "layers"]
    if not positions:
        return None
    if len(positions) != 1:
        raise ValueError
    pos = positions[0]
    # The collection includes the ``...layers`` container itself; it is not a
    # layer entry until a canonical numeric index follows it.
    if pos + 1 >= len(parts):
        return None
    index = _index(parts[pos + 1])
    if index is None:
        raise ValueError
    return ".".join(parts[:pos]), index, parts[pos + 2 :]


def _paths(index: int, *suffix: str) -> str:
    return ".".join((_ROOT, "layers", str(index), *suffix))


def _topology(
    modules: Mapping[str, object], config: _ConfigFacts
) -> tuple[tuple[_Layer, ...], tuple[str, ...]]:
    # The conditional wrapper has one and only one decoder-layer container.
    # Reject alternate ``*.layers`` roots even when they contain no indexed
    # descendants (which would otherwise be invisible to ``_layer_parse``).
    layer_containers = {
        path[: -len(".layers")] if path.endswith(".layers") else ""
        for path in modules
        if path == "layers" or path.endswith(".layers")
    }
    if layer_containers != {_ROOT}:
        return (), (_TOPOLOGY,)
    roots: dict[str, dict[int, set[tuple[str, ...]]]] = {}
    try:
        for path in modules:
            parsed = _layer_parse(path)
            if parsed is None:
                continue
            prefix, index, suffix = parsed
            roots.setdefault(prefix, {}).setdefault(index, set()).add(suffix)
    except ValueError:
        return (), (_TOPOLOGY,)
    if set(roots) != {_ROOT}:
        return (), (_TOPOLOGY,)
    indexed = roots[_ROOT]
    if set(indexed) != set(range(config.layers)) or f"{_ROOT}.layers" not in modules:
        return (), (_TOPOLOGY,)
    expected = {
        (),
        ("mlp",),
        ("mlp", "gate"),
        ("mlp", "experts"),
        ("mlp", "experts", "act_fn"),
        ("mlp", "shared_expert"),
        ("mlp", "shared_expert", "gate_proj"),
        ("mlp", "shared_expert", "up_proj"),
        ("mlp", "shared_expert", "down_proj"),
        ("mlp", "shared_expert", "act_fn"),
        ("mlp", "shared_expert_gate"),
    }
    layers: list[_Layer] = []
    for index in range(config.layers):
        suffixes = indexed[index]
        if {suffix for suffix in suffixes if suffix[:1] == ("mlp",)} != expected - {()}:
            return (), (_TOPOLOGY,)
        mlp = _paths(index, "mlp")
        layers.append(
            _Layer(
                index,
                mlp,
                f"{mlp}.gate",
                f"{mlp}.experts",
                f"{mlp}.shared_expert",
                f"{mlp}.shared_expert_gate",
            )
        )
    return tuple(layers), ()


def _validate_exposed_config(modules: Mapping[str, object], nested: object) -> tuple[str, ...]:
    language_model = modules.get(_ROOT, _MISSING)
    if language_model is _MISSING:
        return (_TOPOLOGY,)
    try:
        exposed = getattr(language_model, "config")
    except AttributeError:
        return (f"{_ROOT}.config is unavailable",)
    except Exception as exc:
        return (_failure(f"{_ROOT}.config access", exc),)
    if exposed is not nested:
        return (f"{_ROOT}.config identity does not match config.text_config",)
    return ()


def _parameter_roots(parameters: Mapping[str, object], count: int) -> tuple[str, ...]:
    roots: set[str] = set()
    try:
        for name in parameters:
            parsed = _layer_parse(name)
            if parsed is None:
                continue
            prefix, index, suffix = parsed
            if index >= count:
                return (_PARAM_ROOT,)
            if suffix[:1] == ("mlp",):
                roots.add(prefix)
    except ValueError:
        return (_PARAM_ROOT,)
    return () if not roots or roots == {_ROOT} else (_PARAM_ROOT,)


def _shape(parameter: object) -> tuple[int, ...] | None:
    try:
        shape = tuple(getattr(parameter, "shape"))
    except Exception:
        return None
    return shape if all(type(dim) is int and dim >= 0 for dim in shape) else None


def _expected(layer: _Layer, config: _ConfigFacts) -> dict[str, tuple[int, ...]]:
    return {
        f"{layer.gate}.weight": (config.experts, config.hidden),
        f"{layer.experts}.gate_up_proj": (
            config.experts,
            2 * config.moe_intermediate,
            config.hidden,
        ),
        f"{layer.experts}.down_proj": (config.experts, config.hidden, config.moe_intermediate),
        f"{layer.shared_expert}.gate_proj.weight": (config.shared_intermediate, config.hidden),
        f"{layer.shared_expert}.up_proj.weight": (config.shared_intermediate, config.hidden),
        f"{layer.shared_expert}.down_proj.weight": (config.hidden, config.shared_intermediate),
        f"{layer.shared_gate}.weight": (1, config.hidden),
    }


def _validate_shapes(
    layers: tuple[_Layer, ...], parameters: Mapping[str, object], config: _ConfigFacts
) -> bool:
    for layer in layers:
        expected = _expected(layer, config)
        if {name for name in parameters if name.startswith(f"{layer.mlp}.")} != set(expected):
            return False
        if any(_shape(parameters.get(name, _MISSING)) != shape for name, shape in expected.items()):
            return False
    return True


def _analyze(model: object, expected_config: object = _MISSING) -> _Analysis:
    config, warning = _model_config(model)
    if config is None:
        return _Analysis(None, (), {}, (warning or _UNSUPPORTED,), ())
    if expected_config is not _MISSING and config is not expected_config:
        return _Analysis(
            None, (), {}, ("model.config identity does not match supplied config",), ()
        )
    identity, nested, identity_warning = _identity(config)
    if identity_warning or nested is None:
        return _Analysis(None, (), {}, (identity_warning or _FAMILY,), identity)
    facts, config_warning = _strict_config(config, identity, nested)
    if facts is None:
        return _Analysis(None, (), {}, (config_warning or _CONFIG,), identity)
    modules, module_warnings = _collect(model, "named_modules")
    if module_warnings:
        return _Analysis(facts, (), {}, module_warnings, identity)
    parameters, parameter_warnings = _collect(model, "named_parameters")
    if parameter_warnings:
        return _Analysis(facts, (), parameters, parameter_warnings, identity)
    layers, topology_warnings = _topology(modules, facts)
    if topology_warnings:
        return _Analysis(facts, layers, parameters, topology_warnings, identity)
    exposed_warnings = _validate_exposed_config(modules, nested)
    if exposed_warnings:
        return _Analysis(facts, layers, parameters, exposed_warnings, identity)
    roots = _parameter_roots(parameters, facts.layers)
    if roots:
        return _Analysis(facts, layers, parameters, roots, identity, True, False)
    shapes_valid = _validate_shapes(layers, parameters, facts)
    return _Analysis(
        facts, layers, parameters, () if shapes_valid else (_SHAPES,), identity, True, shapes_valid
    )


def _detection(analysis: _Analysis) -> AdapterDetection:
    if not analysis.valid:
        return AdapterDetection(
            score=0.0, warnings=tuple(sorted(set(analysis.warnings or (_UNSUPPORTED,))))
        )
    return AdapterDetection(
        score=1.0,
        evidence=tuple(
            sorted(
                (
                    *analysis.family_evidence,
                    "config:strict-fields-and-schedule",
                    "topology:packed-shared-expert",
                    "shapes:exact",
                )
            )
        ),
    )


def _component(
    manifest: ModelManifest,
    layer: _Layer,
    descriptor: AdapterDescriptor,
    *,
    kind: ComponentKind,
    path: str,
    expert_index: int | None,
    shapes: dict[str, list[int]],
    routed: bool | None,
    shared: bool | None,
    warnings: list[str],
) -> tuple[DiscoveryCandidate, ComponentManifest]:
    key = make_component_key(
        manifest.model_key, kind.value, path, layer_index=layer.index, expert_index=expert_index
    )
    evidence = [
        DiscoveryEvidence(
            signal=DiscoverySignal.CHILD_STRUCTURE, detail="Qwen4Exp packed layout", weight=1.0
        )
    ]
    candidate = DiscoveryCandidate(
        component_key=key,
        model_key=manifest.model_key,
        kind=kind,
        module_path=path,
        layer_index=layer.index,
        expert_index=expert_index,
        confidence=1.0,
        evidence=evidence,
        routed=routed,
        shared=shared,
        warnings=list(warnings),
    )
    capture = CaptureProvenance(
        source=CaptureSource.STATIC_STRUCTURE,
        method=_METHOD,
        adapter=descriptor.name,
        adapter_version=descriptor.version,
        verified=False,
        metadata={"layout": _LAYOUT},
    )
    provenance = Provenance(
        source=_METHOD,
        tool_version=__version__,
        metadata={"layout": _LAYOUT, "evidence": ["config", "topology", "shapes"]},
    )
    component = ComponentManifest(
        component_key=key,
        model_key=manifest.model_key,
        kind=kind,
        module_path=path,
        layer_index=layer.index,
        expert_index=expert_index,
        tensor_shapes=shapes,
        capabilities=[CapabilityLabel.STRUCTURE],
        routed=routed,
        shared=shared,
        capture=capture,
        provenance=provenance,
        warnings=list(warnings),
    )
    return candidate, component


def _report(
    manifest: ModelManifest, analysis: _Analysis, descriptor: AdapterDescriptor
) -> DiscoveryReport:
    if not analysis.valid or analysis.config is None:
        return DiscoveryReport(
            model_key=manifest.model_key,
            model_manifest=manifest,
            warnings=sorted(set((_UNSUPPORTED, *analysis.warnings))),
        )
    config = analysis.config
    candidates: list[DiscoveryCandidate] = []
    components: list[ComponentManifest] = []
    report_warnings: list[str] = []
    for layer in analysis.layers:

        def add(**kwargs: object) -> None:
            candidate, component = _component(manifest, layer, descriptor, **kwargs)  # type: ignore[arg-type]
            candidates.append(candidate)
            components.append(component)

        add(
            kind=ComponentKind.MOE_LAYER,
            path=layer.mlp,
            expert_index=None,
            shapes={},
            routed=None,
            shared=None,
            warnings=[],
        )
        add(
            kind=ComponentKind.ROUTER,
            path=layer.gate,
            expert_index=None,
            shapes={"weight": [config.experts, config.hidden]},
            routed=None,
            shared=None,
            warnings=[],
        )
        add(
            kind=ComponentKind.EXPERT_CONTAINER,
            path=layer.experts,
            expert_index=None,
            shapes={
                "gate_up_proj": [config.experts, 2 * config.moe_intermediate, config.hidden],
                "down_proj": [config.experts, config.hidden, config.moe_intermediate],
            },
            routed=None,
            shared=None,
            warnings=[],
        )
        for expert_index in range(config.experts):
            add(
                kind=ComponentKind.EXPERT,
                path=layer.experts,
                expert_index=expert_index,
                shapes={
                    "gate_up_proj": [2 * config.moe_intermediate, config.hidden],
                    "down_proj": [config.hidden, config.moe_intermediate],
                },
                routed=True,
                shared=False,
                warnings=[_PACKED_WARNING],
            )
            report_warnings.append(_PACKED_WARNING)
        add(
            kind=ComponentKind.SHARED_EXPERT,
            path=layer.shared_expert,
            expert_index=None,
            shapes={
                "gate_proj.weight": [config.shared_intermediate, config.hidden],
                "up_proj.weight": [config.shared_intermediate, config.hidden],
                "down_proj.weight": [config.hidden, config.shared_intermediate],
                "shared_expert_gate.weight": [1, config.hidden],
            },
            routed=False,
            shared=True,
            warnings=[],
        )
    return DiscoveryReport(
        model_key=manifest.model_key,
        model_manifest=manifest,
        scanner_version=__version__,
        facts=DiscoveryFacts(
            expert_count=config.experts,
            expert_count_source="config.text_config.num_experts",
            routed_top_k=config.top_k,
            routed_top_k_source="config.text_config.num_experts_per_tok",
            shared_expert_count=1,
            shared_expert_count_source="topology.shared_expert",
        ),
        candidates=candidates,
        components=components,
        warnings=sorted(set(report_warnings)),
    )


class Qwen4ExpStaticAdapter:
    """Stateless packed-only static adapter for Qwen4Exp."""

    __slots__ = ()

    @property
    def descriptor(self) -> AdapterDescriptor:
        return _DESCRIPTOR

    def detect(self, model: object, config: object) -> AdapterDetection:
        return _detection(_analyze(model, expected_config=config))

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        return _report(model_manifest, _analyze(model), self.descriptor)


__all__ = ["Qwen4ExpStaticAdapter"]
