"""Static, model-runtime-independent Qwen3.5-MoE inspection.

This adapter is intentionally independent from the older Qwen3 adapter.  The
Transformers 5.14 Qwen3.5 surface uses packed expert tensors and an optional
conditional-generation wrapper; neither is safely represented by the older
indexed Qwen3 contract.  The seam reads only names, configuration fields, and
parameter shapes.  It never imports a model runtime, reads tensor values, or
retains model state.
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
_NAME = "huggingface-qwen3.5-moe-static"
_VERSION = "1.0"
_METHOD = "qwen3.5-moe-static-structure-v1"
_LAYOUT = "packed"

_UNSUPPORTED = "Qwen3.5-MoE static structure is unsupported on this object"
_FAMILY = "Qwen3.5-MoE family identity is missing or conflicting"
_CONFIG = "Qwen3.5-MoE configuration fields or sparse schedule are invalid"
_TOPOLOGY = "Qwen3.5-MoE module topology is incomplete, conflicting, or mixed"
_SHAPES = "Qwen3.5-MoE parameter shapes do not match the packed layout"
_PARAM_ROOT = "Qwen3.5-MoE parameter roots are conflicting or malformed"
_PACKED_WARNING = "packed expert slices are logical and are not independently hookable"

_OUTER_ARCHITECTURES = frozenset({"Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeModel"})
_TEXT_ARCHITECTURES = frozenset({"Qwen3_5MoeForCausalLM", "Qwen3_5MoeTextModel"})
_LAYER_TYPES = frozenset({"linear_attention", "full_attention"})
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
    architecture_families=("qwen3_5_moe",),
    compatibility_notes=(
        "official Transformers v5.14 packed conditional and text surfaces are supported",
        "shared experts are structural and not router targets",
        "structure-only; routing and model certification are not provided",
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
    root: str
    family_evidence: tuple[str, ...]


@dataclass(frozen=True)
class _Layer:
    index: int
    prefix: str
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


def _sequence(value: object, field: str) -> tuple[tuple[object, ...] | None, str | None]:
    if not isinstance(value, list | tuple):
        return None, None
    try:
        return tuple(value), None
    except Exception as exc:
        return None, _failure(f"config.{field} iteration", exc)


def _architectures(config: object) -> tuple[tuple[str, ...] | None, str | None, bool]:
    try:
        raw = _value(config, "architectures")
    except _ConfigReadError as exc:
        return None, _failure("config.architectures access", exc.cause), True
    if raw is _MISSING:
        return None, None, False
    values, warning = _sequence(raw, "architectures")
    if warning is not None:
        return None, warning, True
    if values is None or not values:
        return None, None, True
    if any(type(item) is not str for item in values) or len(set(values)) != len(values):
        return None, None, True
    return tuple(values), None, True


def _identity(config: object) -> tuple[tuple[str, ...], str | None, str | None]:
    """Return evidence, surface kind, and a safe warning."""

    try:
        model_type = _value(config, "model_type")
        architectures, architecture_warning, _architecture_present = _architectures(config)
    except _ConfigReadError as exc:
        return (), None, _failure(f"config.{exc.field} access", exc.cause)
    if architecture_warning is not None:
        return (), None, architecture_warning
    if type(model_type) is not str or architectures is None:
        return (), None, _FAMILY
    if model_type == "qwen3_5_moe":
        if any(item not in _OUTER_ARCHITECTURES for item in architectures):
            return (), None, _FAMILY
        if not any(item in _OUTER_ARCHITECTURES for item in architectures):
            return (), None, _FAMILY
        try:
            nested = _value(config, "text_config")
        except _ConfigReadError as exc:
            return (), None, _failure(f"config.{exc.field} access", exc.cause)
        if nested is _MISSING or nested is None:
            return (), None, _FAMILY
        try:
            nested_type = _value(nested, "model_type")
            nested_arch, nested_arch_warning, nested_arch_present = _architectures(nested)
        except _ConfigReadError as exc:
            return (), None, _failure(f"config.text_config.{exc.field} access", exc.cause)
        if nested_arch_warning is not None:
            return (), None, nested_arch_warning
        if nested_type != "qwen3_5_moe_text" or (nested_arch_present and nested_arch is None):
            return (), None, _FAMILY
        if nested_arch is not None and any(item not in _TEXT_ARCHITECTURES for item in nested_arch):
            return (), None, _FAMILY
        return (
            ("architecture:qwen3.5-conditional-allowlist", "model_type:qwen3_5_moe"),
            "conditional",
            None,
        )
    if model_type == "qwen3_5_moe_text":
        if any(item not in _TEXT_ARCHITECTURES for item in architectures):
            return (), None, _FAMILY
        return (
            ("architecture:qwen3.5-text-allowlist", "model_type:qwen3_5_moe_text"),
            "text",
            None,
        )
    return (), None, _FAMILY


def _strict_config(
    config: object, identity: tuple[str, ...], surface: str
) -> tuple[_ConfigFacts | None, str | None]:
    source = config
    if surface == "conditional":
        try:
            source = _value(config, "text_config")
        except _ConfigReadError as exc:
            return None, _failure(f"config.{exc.field} access", exc.cause)
        if source is _MISSING or source is None:
            return None, _CONFIG
    values: dict[str, int] = {}
    for field in _INT_FIELDS:
        try:
            raw = _value(source, field)
        except _ConfigReadError as exc:
            return None, _failure(f"config.{exc.field} access", exc.cause)
        if type(raw) is not int or raw <= 0:
            return None, _CONFIG
        values[field] = raw
    if values["num_experts_per_tok"] > values["num_experts"]:
        return None, _CONFIG
    try:
        layer_types = _value(source, "layer_types")
        mlp_only = _value(source, "mlp_only_layers")
    except _ConfigReadError as exc:
        return None, _failure(f"config.{exc.field} access", exc.cause)
    layer_values, layer_warning = _sequence(layer_types, "layer_types")
    if layer_warning is not None:
        return None, layer_warning
    if layer_values is None or len(layer_values) != values["num_hidden_layers"]:
        return None, _CONFIG
    if any(type(item) is not str or item not in _LAYER_TYPES for item in layer_values):
        return None, _CONFIG
    if mlp_only is not _MISSING:
        mlp_values, mlp_warning = _sequence(mlp_only, "mlp_only_layers")
        if mlp_warning is not None:
            return None, mlp_warning
        if mlp_values is None or len(mlp_values) != 0:
            return None, _CONFIG
    root = "model.language_model" if surface == "conditional" else "model"
    return (
        _ConfigFacts(
            layers=values["num_hidden_layers"],
            hidden=values["hidden_size"],
            moe_intermediate=values["moe_intermediate_size"],
            shared_intermediate=values["shared_expert_intermediate_size"],
            experts=values["num_experts"],
            top_k=values["num_experts_per_tok"],
            root=root,
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


def _canonical_index(token: str) -> int | None:
    if not token or not token.isdecimal() or (len(token) > 1 and token.startswith("0")):
        return None
    return int(token)


def _layer_parse(path: str) -> tuple[str, int, tuple[str, ...]] | None:
    if path == "":
        return None
    parts = tuple(path.split("."))
    if any(not part for part in parts):
        raise ValueError
    positions = [i for i, part in enumerate(parts) if part == "layers"]
    candidates = [i for i in positions if i + 1 < len(parts)]
    if not candidates:
        return None
    if len(candidates) != 1:
        raise ValueError
    pos = candidates[0]
    index = _canonical_index(parts[pos + 1])
    if index is None:
        raise ValueError
    return ".".join(parts[:pos]), index, parts[pos + 2 :]


def _paths(prefix: str, index: int, suffix: tuple[str, ...] = ()) -> str:
    return ".".join(part for part in (prefix, "layers", str(index), *suffix) if part)


def _topology(
    modules: Mapping[str, object], config: _ConfigFacts, surface: str
) -> tuple[tuple[_Layer, ...], tuple[str, ...]]:
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
    if surface == "conditional":
        accepted_prefixes = {config.root}
    elif set(roots) == {""}:
        accepted_prefixes = {""}
    else:
        accepted_prefixes = {config.root}
    if set(roots) != accepted_prefixes:
        return (), (_TOPOLOGY,)
    prefix = next(iter(accepted_prefixes))
    indexed = roots.get(prefix, {})
    if set(indexed) != set(range(config.layers)):
        return (), (_TOPOLOGY,)
    layers_root = ".".join(part for part in (prefix, "layers") if part)
    if layers_root not in modules:
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
        mlp_suffixes = {suffix for suffix in suffixes if suffix[:1] == ("mlp",)}
        if () not in suffixes or mlp_suffixes != expected - {()}:
            return (), (_TOPOLOGY,)
        mlp = _paths(prefix, index, ("mlp",))
        layers.append(
            _Layer(
                index=index,
                prefix=prefix,
                mlp=mlp,
                gate=f"{mlp}.gate",
                experts=f"{mlp}.experts",
                shared_expert=f"{mlp}.shared_expert",
                shared_gate=f"{mlp}.shared_expert_gate",
            )
        )
    return tuple(layers), ()


def _validate_exposed_config(
    modules: Mapping[str, object], config: object, surface: str
) -> tuple[str, ...]:
    """Require the official conditional wrapper to expose its text config."""

    if surface != "conditional":
        return ()
    try:
        nested = _value(config, "text_config")
    except _ConfigReadError as exc:
        return (_failure(f"config.{exc.field} access", exc.cause),)
    language_model = modules.get("model.language_model", _MISSING)
    if language_model is _MISSING:
        return (_TOPOLOGY,)
    try:
        exposed = getattr(language_model, "config")
    except AttributeError:
        return ("model.language_model.config is unavailable",)
    except Exception as exc:
        return (_failure("model.language_model.config access", exc),)
    if nested is _MISSING or exposed is not nested:
        return ("model.language_model.config identity does not match config.text_config",)
    return ()


def _parameter_roots(parameters: Mapping[str, object], prefix: str, count: int) -> tuple[str, ...]:
    roots: set[str] = set()
    for name in parameters:
        try:
            parsed = _layer_parse(name)
        except ValueError:
            return (_PARAM_ROOT,)
        if parsed is None:
            continue
        parsed_prefix, index, suffix = parsed
        if index >= count:
            return (_PARAM_ROOT,)
        if suffix[:1] == ("mlp",):
            roots.add(parsed_prefix)
    if roots and roots != {prefix}:
        return (_PARAM_ROOT,)
    return ()


def _shape(parameter: object) -> tuple[int, ...] | None:
    try:
        result = tuple(getattr(parameter, "shape"))
    except Exception:
        return None
    if any(type(item) is not int or item < 0 for item in result):
        return None
    return result


def _expected(layer: _Layer, config: _ConfigFacts) -> dict[str, tuple[int, ...]]:
    return {
        f"{layer.gate}.weight": (config.experts, config.hidden),
        f"{layer.experts}.gate_up_proj": (
            config.experts,
            2 * config.moe_intermediate,
            config.hidden,
        ),
        f"{layer.experts}.down_proj": (
            config.experts,
            config.hidden,
            config.moe_intermediate,
        ),
        f"{layer.shared_expert}.gate_proj.weight": (
            config.shared_intermediate,
            config.hidden,
        ),
        f"{layer.shared_expert}.up_proj.weight": (
            config.shared_intermediate,
            config.hidden,
        ),
        f"{layer.shared_expert}.down_proj.weight": (
            config.hidden,
            config.shared_intermediate,
        ),
        f"{layer.shared_gate}.weight": (1, config.hidden),
    }


def _validate_shapes(
    layers: tuple[_Layer, ...], parameters: Mapping[str, object], config: _ConfigFacts
) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    for layer in layers:
        expected = _expected(layer, config)
        actual = {name for name in parameters if name.startswith(f"{layer.mlp}.")}
        if actual != set(expected):
            warnings.append(_SHAPES)
        for name, shape in expected.items():
            parameter = parameters.get(name, _MISSING)
            if parameter is _MISSING or _shape(parameter) != shape:
                warnings.append(_SHAPES)
    return not warnings, tuple(sorted(set(warnings)))


def _analyze(model: object, expected_config: object = _MISSING) -> _Analysis:
    config, config_warning = _model_config(model)
    if config is None:
        return _Analysis(None, (), {}, (config_warning or _UNSUPPORTED,), ())
    if expected_config is not _MISSING and config is not expected_config:
        return _Analysis(
            None,
            (),
            {},
            ("model.config identity does not match supplied config",),
            (),
        )
    identity, surface, identity_warning = _identity(config)
    if identity_warning is not None or surface is None:
        return _Analysis(None, (), {}, (identity_warning or _FAMILY,), identity)
    facts, config_warning = _strict_config(config, identity, surface)
    if facts is None:
        return _Analysis(None, (), {}, (config_warning or _CONFIG,), identity)
    modules, module_warnings = _collect(model, "named_modules")
    if module_warnings:
        return _Analysis(facts, (), {}, module_warnings, identity)
    parameters, parameter_warnings = _collect(model, "named_parameters")
    if parameter_warnings:
        return _Analysis(facts, (), parameters, parameter_warnings, identity)
    layers, topology_warnings = _topology(modules, facts, surface)
    if topology_warnings:
        return _Analysis(facts, layers, parameters, topology_warnings, identity)
    exposed_config_warnings = _validate_exposed_config(modules, config, surface)
    if exposed_config_warnings:
        return _Analysis(facts, layers, parameters, exposed_config_warnings, identity)
    roots = _parameter_roots(parameters, layers[0].prefix, facts.layers)
    if roots:
        return _Analysis(facts, layers, parameters, roots, identity, True, False)
    shapes_valid, shape_warnings = _validate_shapes(layers, parameters, facts)
    return _Analysis(facts, layers, parameters, shape_warnings, identity, True, shapes_valid)


def _detection(analysis: _Analysis) -> AdapterDetection:
    if not analysis.valid:
        messages = tuple(sorted(set(analysis.warnings or (_UNSUPPORTED,))))
        return AdapterDetection(score=0.0, warnings=messages)
    evidence = (
        *analysis.family_evidence,
        "config:strict-fields-and-schedule",
        "topology:packed-shared-expert",
        "shapes:exact",
    )
    return AdapterDetection(score=1.0, evidence=evidence)


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
        manifest.model_key,
        kind.value,
        path,
        layer_index=layer.index,
        expert_index=expert_index,
    )
    evidence = [
        DiscoveryEvidence(
            signal=DiscoverySignal.CHILD_STRUCTURE,
            detail="Qwen3.5-MoE packed layout",
            weight=1.0,
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
            candidate, component = _component(  # type: ignore[arg-type]
                manifest, layer, descriptor, **kwargs
            )
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
    facts = DiscoveryFacts(
        expert_count=config.experts,
        expert_count_source="config.num_experts",
        routed_top_k=config.top_k,
        routed_top_k_source="config.num_experts_per_tok",
        shared_expert_count=1,
        shared_expert_count_source="topology.shared_expert",
    )
    return DiscoveryReport(
        model_key=manifest.model_key,
        model_manifest=manifest,
        scanner_version=__version__,
        facts=facts,
        candidates=candidates,
        components=components,
        warnings=sorted(set(report_warnings)),
    )


class Qwen3_5MoeStaticAdapter:
    """Stateless packed-only static adapter for current Qwen3.5-MoE."""

    __slots__ = ()

    @property
    def descriptor(self) -> AdapterDescriptor:
        return _DESCRIPTOR

    def detect(self, model: object, config: object) -> AdapterDetection:
        return _detection(_analyze(model, expected_config=config))

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        return _report(model_manifest, _analyze(model), self.descriptor)


__all__ = ["Qwen3_5MoeStaticAdapter"]
