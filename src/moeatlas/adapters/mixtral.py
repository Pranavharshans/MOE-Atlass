"""Static, model-runtime-independent Mixtral structure adapter.

This module intentionally knows only the two public structural layouts used by
Mixtral-shaped module trees.  It does not import Transformers, inspect tensor
values, or retain observations between ``detect`` and ``discover`` calls.
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
_LAYOUT_LEGACY = "legacy_indexed"
_LAYOUT_PACKED = "packed"
_ADAPTER_NAME = "huggingface-mixtral-static"
_ADAPTER_VERSION = "1.0"
_METHOD = "mixtral-static-structure-v1"
_PACKED_SLICE_WARNING = "packed expert slices are logical and are not independently hookable"
_UNSUPPORTED_WARNING = "Mixtral static structure is unsupported on this object"
_FAMILY_WARNING = "Mixtral family identity is missing or conflicting"
_CONFIG_WARNING = "Mixtral configuration fields are invalid or inconsistent"
_SHAPE_WARNING = "Mixtral parameter shapes do not match the selected layout"

_ARCHITECTURES = frozenset(
    {
        "MixtralModel",
        "MixtralForCausalLM",
        "MixtralForSequenceClassification",
        "MixtralForTokenClassification",
        "MixtralForQuestionAnswering",
    }
)
_CONFIG_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "num_local_experts",
    "num_experts_per_tok",
)


_DESCRIPTOR = AdapterDescriptor(
    name=_ADAPTER_NAME,
    version=_ADAPTER_VERSION,
    architecture_families=("mixtral",),
    compatibility_notes=(
        "current Transformers packed layouts are supported as logical slices",
        "official Transformers 4.50 indexed layouts are supported",
        "structure-only; routing certification is not provided",
    ),
)


@dataclass(frozen=True)
class _ConfigFacts:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    family_evidence: tuple[str, ...]


@dataclass(frozen=True)
class _Layer:
    prefix: str
    index: int
    layout: str
    moe_path: str
    gate_path: str
    experts_path: str
    expert_paths: tuple[str, ...]


@dataclass(frozen=True)
class _Analysis:
    config: _ConfigFacts | None
    layers: tuple[_Layer, ...]
    parameter_shapes: dict[str, tuple[int, ...]]
    layout: str | None
    warnings: tuple[str, ...]
    family_evidence: tuple[str, ...]
    topology_valid: bool
    shapes_valid: bool

    @property
    def valid(self) -> bool:
        return (
            self.config is not None
            and bool(self.layers)
            and self.layout is not None
            and self.topology_valid
            and self.shapes_valid
        )


def _warning_for_exception(operation: str, exc: Exception) -> str:
    return f"{operation} failed with {type(exc).__name__}"


def _zero_detection(*warnings: str) -> AdapterDetection:
    normalized = tuple(sorted(set(warnings))) or (_UNSUPPORTED_WARNING,)
    return AdapterDetection(score=0.0, warnings=normalized)


def _config_value(config: object, field_name: str) -> object:
    if isinstance(config, Mapping):
        try:
            return config[field_name] if field_name in config else _MISSING
        except Exception as exc:
            raise _ConfigReadError(field_name, exc) from exc
    try:
        return getattr(config, field_name)
    except AttributeError:
        return _MISSING
    except Exception as exc:
        raise _ConfigReadError(field_name, exc) from exc


class _ConfigReadError(Exception):
    def __init__(self, field_name: str, cause: Exception) -> None:
        self.field_name = field_name
        self.cause = cause
        super().__init__()


def _model_config(model: object) -> tuple[object | None, str | None]:
    try:
        config = getattr(model, "config")
    except AttributeError:
        return None, "model.config is unavailable"
    except Exception as exc:
        return None, _warning_for_exception("model.config access", exc)
    if config is None:
        return None, "model.config is unavailable"
    return config, None


def _family_facts(config: object) -> tuple[tuple[str, ...], str | None]:
    try:
        model_type = _config_value(config, "model_type")
        architectures = _config_value(config, "architectures")
    except _ConfigReadError as exc:
        return (), _warning_for_exception(f"config.{exc.field_name} access", exc.cause)

    if model_type is not _MISSING and (type(model_type) is not str or model_type != "mixtral"):
        return (), _FAMILY_WARNING

    config_architectures: tuple[str, ...] = ()
    if architectures is not _MISSING:
        if not isinstance(architectures, list | tuple):
            return (), _FAMILY_WARNING
        if any(type(value) is not str for value in architectures):
            return (), _FAMILY_WARNING
        config_architectures = tuple(architectures)
        if len(set(config_architectures)) != len(config_architectures) or any(
            value not in _ARCHITECTURES for value in config_architectures
        ):
            return (), _FAMILY_WARNING

    has_model_type = model_type is not _MISSING
    has_architecture = bool(config_architectures)
    if not has_model_type and not has_architecture:
        return (), _FAMILY_WARNING

    evidence: list[str] = []
    if has_model_type:
        evidence.append("model_type:mixtral")
    if has_architecture:
        evidence.append("architecture:mixtral-allowlist")
    return tuple(sorted(evidence)), None


def _strict_config(config: object) -> tuple[_ConfigFacts | None, str | None]:
    values: dict[str, int] = {}
    for field_name in _CONFIG_FIELDS:
        try:
            value = _config_value(config, field_name)
        except _ConfigReadError as exc:
            return None, _warning_for_exception(f"config.{exc.field_name} access", exc.cause)
        if type(value) is not int or value <= 0:
            return None, f"invalid config field: {field_name}"
        values[field_name] = value
    if values["num_experts_per_tok"] > values["num_local_experts"]:
        return None, "config num_experts_per_tok exceeds num_local_experts"
    return (
        _ConfigFacts(
            num_hidden_layers=values["num_hidden_layers"],
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            num_local_experts=values["num_local_experts"],
            num_experts_per_tok=values["num_experts_per_tok"],
            family_evidence=(),
        ),
        None,
    )


def _pair_iterator(surface: object) -> object:
    if isinstance(surface, Mapping):
        return iter(surface.items())
    return iter(surface)  # type: ignore[arg-type]


def _collect_modules(model: object) -> tuple[dict[str, object], tuple[str, ...]]:
    try:
        method = getattr(model, "named_modules")
    except AttributeError:
        return {}, ("named_modules() is unavailable",)
    except Exception as exc:
        return {}, (_warning_for_exception("named_modules() access", exc),)
    if not callable(method):
        return {}, ("named_modules() is unavailable",)
    try:
        iterator = _pair_iterator(method())
    except Exception as exc:
        return {}, (_warning_for_exception("named_modules() traversal", exc),)
    entries: dict[str, object] = {}
    position = 0
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            return {}, (_warning_for_exception("named_modules() iteration", exc),)
        if not isinstance(item, tuple | list) or len(item) != 2:
            return {}, ("named_modules() surface is malformed",)
        path, value = item
        if type(path) is not str:
            return {}, ("named_modules() surface is malformed",)
        if path in entries:
            return {}, ("named_modules() returned duplicate paths",)
        entries[path] = value
        position += 1
    if position == 0:
        return {}, ("named_modules() surface is empty",)
    return dict(sorted(entries.items())), ()


def _structural_attribute(
    modules: Mapping[str, object], path: str, attribute: str, expected: int, label: str
) -> str | None:
    module = modules.get(path, _MISSING)
    if module is _MISSING:
        return f"Mixtral structural attribute unavailable: {label}"
    try:
        value = getattr(module, attribute)
    except AttributeError:
        return f"Mixtral structural attribute unavailable: {label}"
    except Exception as exc:
        return _warning_for_exception(f"structural attribute {label} access", exc)
    if type(value) is not int or value <= 0:
        return f"Mixtral structural attribute is invalid: {label}"
    if value != expected:
        return f"Mixtral structural attribute disagrees with config: {label}"
    return None


def _validate_structure_attributes(
    layers: tuple[_Layer, ...], modules: Mapping[str, object], config: _ConfigFacts
) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    for layer in layers:
        if layer.layout == _LAYOUT_LEGACY:
            checks = (
                (layer.moe_path, "num_experts", config.num_local_experts, "legacy.num_experts"),
                (layer.moe_path, "top_k", config.num_experts_per_tok, "legacy.top_k"),
            )
        else:
            checks = (
                (layer.moe_path, "top_k", config.num_experts_per_tok, "packed.mlp.top_k"),
                (
                    layer.gate_path,
                    "num_experts",
                    config.num_local_experts,
                    "packed.gate.num_experts",
                ),
                (layer.gate_path, "top_k", config.num_experts_per_tok, "packed.gate.top_k"),
                (
                    layer.experts_path,
                    "num_experts",
                    config.num_local_experts,
                    "packed.experts.num_experts",
                ),
            )
        for path, attribute, expected, label in checks:
            warning = _structural_attribute(modules, path, attribute, expected, label)
            if warning is not None:
                warnings.append(warning)
    return not warnings, tuple(sorted(set(warnings)))


def _shape(parameter: object) -> tuple[tuple[int, ...] | None, str | None]:
    try:
        raw_shape = getattr(parameter, "shape")
        dimensions = tuple(raw_shape)
    except Exception as exc:
        return None, _warning_for_exception("parameter.shape access", exc)
    if any(type(dimension) is not int or dimension < 0 for dimension in dimensions):
        return None, "parameter.shape is invalid"
    return tuple(dimensions), None


def _collect_parameters(model: object) -> tuple[dict[str, tuple[int, ...]], tuple[str, ...]]:
    try:
        method = getattr(model, "named_parameters")
    except AttributeError:
        return {}, ("named_parameters() is unavailable",)
    except Exception as exc:
        return {}, (_warning_for_exception("named_parameters() access", exc),)
    if not callable(method):
        return {}, ("named_parameters() is unavailable",)
    try:
        iterator = _pair_iterator(method())
    except Exception as exc:
        return {}, (_warning_for_exception("named_parameters() traversal", exc),)
    entries: dict[str, tuple[int, ...]] = {}
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            return {}, (_warning_for_exception("named_parameters() iteration", exc),)
        if not isinstance(item, tuple | list) or len(item) != 2:
            return {}, ("named_parameters() surface is malformed",)
        name, parameter = item
        if type(name) is not str or not name or name in entries:
            return {}, ("named_parameters() surface is malformed",)
        parameter_shape, shape_warning = _shape(parameter)
        if shape_warning is not None or parameter_shape is None:
            return {}, (shape_warning or "named_parameters() parameter shape is unavailable",)
        entries[name] = parameter_shape
    if not entries:
        return {}, ("named_parameters() surface is empty",)
    return dict(sorted(entries.items())), ()


def _layer_parts(path: str) -> tuple[str, int, tuple[str, ...]] | None:
    parts = tuple(path.split("."))
    for index, part in enumerate(parts[:-1]):
        if part == "layers":
            layer_index = _canonical_index(parts[index + 1])
            if layer_index is None:
                return None
            suffix = parts[index + 2 :]
            return ".".join(parts[:index]), layer_index, suffix
    return None


def _canonical_index(token: str) -> int | None:
    """Return an ordinary decimal index, rejecting aliases such as ``01``."""

    if not token or any(character not in "0123456789" for character in token):
        return None
    if len(token) > 1 and token.startswith("0"):
        return None
    return int(token)


def _module_path(prefix: str, index: int, suffix: tuple[str, ...]) -> str:
    parts = tuple(part for part in (prefix, "layers", str(index), *suffix) if part)
    return ".".join(parts)


def _topology(
    modules: Mapping[str, object], config: _ConfigFacts
) -> tuple[tuple[_Layer, ...], str | None, tuple[str, ...]]:
    roots: dict[str, dict[int, set[tuple[str, ...]]]] = {}
    layer_tokens: dict[str, dict[int, set[str]]] = {}
    for path in modules:
        parts = tuple(path.split("."))
        layer_positions = tuple(
            position for position, part in enumerate(parts[:-1]) if part == "layers"
        )
        if layer_positions and any(
            _canonical_index(parts[position + 1]) is None for position in layer_positions
        ):
            return (), None, ("Mixtral layer indices are not canonical",)
        parsed = _layer_parts(path)
        if parsed is None:
            continue
        prefix, index, suffix = parsed
        roots.setdefault(prefix, {}).setdefault(index, set()).add(suffix)
        layer_position = next(
            position for position, part in enumerate(parts[:-1]) if part == "layers"
        )
        layer_tokens.setdefault(prefix, {}).setdefault(index, set()).add(parts[layer_position + 1])
    candidate_roots = {
        prefix: indexed
        for prefix, indexed in roots.items()
        if any(
            suffix and suffix[0] in {"block_sparse_moe", "mlp"}
            for layer_suffixes in indexed.values()
            for suffix in layer_suffixes
        )
    }
    if len(candidate_roots) != 1:
        return (), None, ("Mixtral module roots are conflicting",)
    prefix, indexed = next(iter(candidate_roots.items()))
    if any(len(tokens) != 1 for tokens in layer_tokens[prefix].values()):
        return (), None, ("Mixtral layer indices are duplicated",)
    expected_indices = set(range(config.num_hidden_layers))
    if set(indexed) != expected_indices:
        return (), None, ("Mixtral layer indices are not contiguous",)

    layers: list[_Layer] = []
    layouts: set[str] = set()
    for index in sorted(indexed):
        suffixes = {
            suffix
            for suffix in indexed[index]
            if suffix and suffix[0] in {"block_sparse_moe", "mlp"}
        }
        legacy_root = ("block_sparse_moe",)
        packed_root = ("mlp",)
        has_legacy = any(suffix[:1] == legacy_root for suffix in suffixes)
        has_packed = any(suffix[:1] == packed_root for suffix in suffixes)
        if has_legacy and has_packed:
            return (), None, ("Mixtral layouts are mixed",)
        if not has_legacy and not has_packed:
            return (), None, ("Mixtral layer topology is incomplete",)
        if has_legacy:
            required = {
                ("block_sparse_moe",),
                ("block_sparse_moe", "gate"),
                ("block_sparse_moe", "experts"),
            }
            if not required.issubset(suffixes):
                return (), None, ("Mixtral legacy layer triple is incomplete",)
            expert_indices = {
                _canonical_index(suffix[-1])
                for suffix in suffixes
                if len(suffix) == 3
                and suffix[:2] == ("block_sparse_moe", "experts")
                and _canonical_index(suffix[2]) is not None
            }
            expected_expert_indices = set(range(config.num_local_experts))
            expected_suffixes = (
                {()}
                | required
                | {
                    suffix
                    for expert_index in expected_expert_indices
                    for suffix in (
                        ("block_sparse_moe", "experts", str(expert_index)),
                        ("block_sparse_moe", "experts", str(expert_index), "w1"),
                        ("block_sparse_moe", "experts", str(expert_index), "w2"),
                        ("block_sparse_moe", "experts", str(expert_index), "w3"),
                        ("block_sparse_moe", "experts", str(expert_index), "act_fn"),
                    )
                }
            )
            required_suffixes = required | {
                suffix
                for expert_index in expected_expert_indices
                for suffix in (
                    ("block_sparse_moe", "experts", str(expert_index)),
                    ("block_sparse_moe", "experts", str(expert_index), "w1"),
                    ("block_sparse_moe", "experts", str(expert_index), "w2"),
                    ("block_sparse_moe", "experts", str(expert_index), "w3"),
                    ("block_sparse_moe", "experts", str(expert_index), "act_fn"),
                )
            }
            if expert_indices != expected_expert_indices or suffixes not in (
                required_suffixes,
                expected_suffixes,
            ):
                return (), None, ("Mixtral legacy expert indices are incomplete",)
            layers.append(
                _Layer(
                    prefix=prefix,
                    index=index,
                    layout=_LAYOUT_LEGACY,
                    moe_path=_module_path(prefix, index, ("block_sparse_moe",)),
                    gate_path=_module_path(prefix, index, ("block_sparse_moe", "gate")),
                    experts_path=_module_path(prefix, index, ("block_sparse_moe", "experts")),
                    expert_paths=tuple(
                        _module_path(
                            prefix,
                            index,
                            ("block_sparse_moe", "experts", str(expert_index)),
                        )
                        for expert_index in range(config.num_local_experts)
                    ),
                )
            )
            layouts.add(_LAYOUT_LEGACY)
        else:
            required = {
                ("mlp",),
                ("mlp", "gate"),
                ("mlp", "experts"),
                ("mlp", "experts", "act_fn"),
            }
            if suffixes not in (required, {()} | required):
                return (), None, ("Mixtral packed layer triple is incomplete",)
            layers.append(
                _Layer(
                    prefix=prefix,
                    index=index,
                    layout=_LAYOUT_PACKED,
                    moe_path=_module_path(prefix, index, ("mlp",)),
                    gate_path=_module_path(prefix, index, ("mlp", "gate")),
                    experts_path=_module_path(prefix, index, ("mlp", "experts")),
                    expert_paths=(),
                )
            )
            layouts.add(_LAYOUT_PACKED)
    if len(layouts) != 1:
        return (), None, ("Mixtral layouts are mixed",)
    return tuple(layers), next(iter(layouts)), ()


def _parameter_shape(
    parameters: Mapping[str, tuple[int, ...]], module_path: str, relative_name: str
) -> tuple[int, ...] | None:
    return parameters.get(f"{module_path}.{relative_name}")


def _validate_shapes(
    layers: tuple[_Layer, ...], parameters: Mapping[str, tuple[int, ...]], config: _ConfigFacts
) -> bool:
    def relative_names(root: str) -> set[str]:
        prefix = f"{root}."
        return {name[len(prefix) :] for name in parameters if name.startswith(prefix)}

    for layer in layers:
        if layer.layout == _LAYOUT_LEGACY:
            expected_names = {"gate.weight"}
            expected_names.update(
                f"experts.{expert_index}.{weight_name}"
                for expert_index in range(config.num_local_experts)
                for weight_name in ("w1.weight", "w2.weight", "w3.weight")
            )
        else:
            expected_names = {
                "gate.weight",
                "experts.gate_up_proj",
                "experts.down_proj",
            }
        if relative_names(layer.moe_path) != expected_names:
            return False
        router_shape = _parameter_shape(parameters, layer.gate_path, "weight")
        if router_shape != (config.num_local_experts, config.hidden_size):
            return False
        if layer.layout == _LAYOUT_LEGACY:
            for expert_path in layer.expert_paths:
                if _parameter_shape(parameters, expert_path, "w1.weight") != (
                    config.intermediate_size,
                    config.hidden_size,
                ):
                    return False
                if _parameter_shape(parameters, expert_path, "w2.weight") != (
                    config.hidden_size,
                    config.intermediate_size,
                ):
                    return False
                if _parameter_shape(parameters, expert_path, "w3.weight") != (
                    config.intermediate_size,
                    config.hidden_size,
                ):
                    return False
        else:
            if _parameter_shape(parameters, layer.experts_path, "gate_up_proj") != (
                config.num_local_experts,
                2 * config.intermediate_size,
                config.hidden_size,
            ):
                return False
            if _parameter_shape(parameters, layer.experts_path, "down_proj") != (
                config.num_local_experts,
                config.hidden_size,
                config.intermediate_size,
            ):
                return False
    return True


def _analyze(model: object, expected_config: object = _MISSING) -> _Analysis:
    config, config_warning = _model_config(model)
    if config is None:
        warning = config_warning or "model.config is unavailable"
        return _Analysis(None, (), {}, None, (warning,), (), False, False)
    if expected_config is not _MISSING and config is not expected_config:
        return _Analysis(
            None,
            (),
            {},
            None,
            ("model.config identity does not match supplied config",),
            (),
            False,
            False,
        )
    family_evidence, family_warning = _family_facts(config)
    if family_warning is not None:
        return _Analysis(None, (), {}, None, (family_warning,), (), False, False)
    config_facts, config_warning = _strict_config(config)
    if config_warning is not None or config_facts is None:
        return _Analysis(
            None,
            (),
            {},
            None,
            (config_warning or _CONFIG_WARNING,),
            family_evidence,
            False,
            False,
        )
    config_facts = _ConfigFacts(
        num_hidden_layers=config_facts.num_hidden_layers,
        hidden_size=config_facts.hidden_size,
        intermediate_size=config_facts.intermediate_size,
        num_local_experts=config_facts.num_local_experts,
        num_experts_per_tok=config_facts.num_experts_per_tok,
        family_evidence=family_evidence,
    )
    modules, module_warnings = _collect_modules(model)
    if module_warnings:
        return _Analysis(
            config_facts,
            (),
            {},
            None,
            module_warnings,
            family_evidence,
            False,
            False,
        )
    parameters, parameter_warnings = _collect_parameters(model)
    if parameter_warnings:
        return _Analysis(
            config_facts,
            (),
            {},
            None,
            parameter_warnings,
            family_evidence,
            False,
            False,
        )
    layers, layout, topology_warnings = _topology(modules, config_facts)
    if topology_warnings:
        return _Analysis(
            config_facts,
            layers,
            parameters,
            layout,
            topology_warnings,
            family_evidence,
            False,
            False,
        )
    attributes_valid, attribute_warnings = _validate_structure_attributes(
        layers, modules, config_facts
    )
    if not attributes_valid:
        return _Analysis(
            config_facts,
            layers,
            parameters,
            layout,
            attribute_warnings,
            family_evidence,
            False,
            False,
        )
    shapes_valid = _validate_shapes(layers, parameters, config_facts)
    return _Analysis(
        config_facts,
        layers,
        parameters,
        layout,
        () if shapes_valid else (_SHAPE_WARNING,),
        family_evidence,
        True,
        shapes_valid,
    )


def _detection(analysis: _Analysis) -> AdapterDetection:
    if not analysis.valid:
        return _zero_detection(*analysis.warnings)
    evidence = list(analysis.family_evidence)
    evidence.extend(("config:strict-fields", "topology:complete-layout", "shapes:exact"))
    weights = {
        "model_type:mixtral": 0.25,
        "architecture:mixtral-allowlist": 0.15,
        "config:strict-fields": 0.20,
        "topology:complete-layout": 0.20,
        "shapes:exact": 0.20,
    }
    score = round(sum(weights[item] for item in evidence), 3)
    return AdapterDetection(score=float(score), evidence=tuple(evidence))


def _evidence(signal: DiscoverySignal, detail: str) -> list[DiscoveryEvidence]:
    return [DiscoveryEvidence(signal=signal, detail=detail, weight=1.0)]


def _component_data(
    model_manifest: ModelManifest,
    layer: _Layer,
    config: _ConfigFacts,
    descriptor: AdapterDescriptor,
    *,
    kind: ComponentKind,
    module_path: str,
    expert_index: int | None,
    tensor_shapes: dict[str, list[int]],
    routed: bool | None,
    warnings: list[str],
) -> tuple[DiscoveryCandidate, ComponentManifest]:
    component_key = make_component_key(
        model_manifest.model_key,
        kind.value,
        module_path,
        layer_index=layer.index,
        expert_index=expert_index,
    )
    layout_detail = f"Mixtral {layer.layout} layout"
    evidence = _evidence(DiscoverySignal.CHILD_STRUCTURE, layout_detail)
    candidate = DiscoveryCandidate(
        component_key=component_key,
        model_key=model_manifest.model_key,
        kind=kind,
        module_path=module_path,
        layer_index=layer.index,
        expert_index=expert_index,
        confidence=1.0,
        evidence=evidence,
        routed=routed,
        shared=None,
        warnings=list(warnings),
    )
    metadata = {"layout": layer.layout, "evidence": ["config", "topology", "shapes"]}
    provenance = Provenance(source="mixtral-static", tool_version=__version__, metadata=metadata)
    capture = CaptureProvenance(
        source=CaptureSource.STATIC_STRUCTURE,
        method=_METHOD,
        adapter=descriptor.name,
        adapter_version=descriptor.version,
        verified=False,
        metadata={"layout": layer.layout},
    )
    component = ComponentManifest(
        component_key=component_key,
        model_key=model_manifest.model_key,
        kind=kind,
        module_path=module_path,
        layer_index=layer.index,
        expert_index=expert_index,
        tensor_shapes=tensor_shapes,
        capabilities=[CapabilityLabel.STRUCTURE],
        routed=routed,
        shared=None,
        capture=capture,
        provenance=provenance,
        warnings=list(warnings),
    )
    return candidate, component


def _report(
    model_manifest: ModelManifest,
    analysis: _Analysis,
    descriptor: AdapterDescriptor,
) -> DiscoveryReport:
    if not analysis.valid or analysis.config is None or analysis.layout is None:
        return DiscoveryReport(
            model_key=model_manifest.model_key,
            model_manifest=model_manifest,
            warnings=sorted(set((_UNSUPPORTED_WARNING, *analysis.warnings))),
        )
    config = analysis.config
    candidates: list[DiscoveryCandidate] = []
    components: list[ComponentManifest] = []
    report_warnings: list[str] = []
    for layer in analysis.layers:
        layer_candidate, layer_component = _component_data(
            model_manifest,
            layer,
            config,
            descriptor,
            kind=ComponentKind.MOE_LAYER,
            module_path=layer.moe_path,
            expert_index=None,
            tensor_shapes={},
            routed=None,
            warnings=[],
        )
        candidates.append(layer_candidate)
        components.append(layer_component)
        gate_shape = _parameter_shape(analysis.parameter_shapes, layer.gate_path, "weight")
        gate_candidate, gate_component = _component_data(
            model_manifest,
            layer,
            config,
            descriptor,
            kind=ComponentKind.ROUTER,
            module_path=layer.gate_path,
            expert_index=None,
            tensor_shapes={"weight": list(gate_shape or ())},
            routed=None,
            warnings=[],
        )
        candidates.append(gate_candidate)
        components.append(gate_component)
        if layer.layout == _LAYOUT_LEGACY:
            container_shapes: dict[str, list[int]] = {}
        else:
            container_shapes = {
                "gate_up_proj": [
                    config.num_local_experts,
                    2 * config.intermediate_size,
                    config.hidden_size,
                ],
                "down_proj": [
                    config.num_local_experts,
                    config.hidden_size,
                    config.intermediate_size,
                ],
            }
        container_candidate, container_component = _component_data(
            model_manifest,
            layer,
            config,
            descriptor,
            kind=ComponentKind.EXPERT_CONTAINER,
            module_path=layer.experts_path,
            expert_index=None,
            tensor_shapes=container_shapes,
            routed=None,
            warnings=[],
        )
        candidates.append(container_candidate)
        components.append(container_component)
        for expert_index in range(config.num_local_experts):
            expert_path = (
                layer.expert_paths[expert_index]
                if layer.layout == _LAYOUT_LEGACY
                else layer.experts_path
            )
            if layer.layout == _LAYOUT_LEGACY:
                expert_shapes = {
                    "w1.weight": [config.intermediate_size, config.hidden_size],
                    "w2.weight": [config.hidden_size, config.intermediate_size],
                    "w3.weight": [config.intermediate_size, config.hidden_size],
                }
                expert_warning: list[str] = []
            else:
                expert_shapes = {
                    "gate_up_proj": [2 * config.intermediate_size, config.hidden_size],
                    "down_proj": [config.hidden_size, config.intermediate_size],
                }
                expert_warning = [_PACKED_SLICE_WARNING]
                report_warnings.append(_PACKED_SLICE_WARNING)
            expert_candidate, expert_component = _component_data(
                model_manifest,
                layer,
                config,
                descriptor,
                kind=ComponentKind.EXPERT,
                module_path=expert_path,
                expert_index=expert_index,
                tensor_shapes=expert_shapes,
                routed=True,
                warnings=expert_warning,
            )
            candidates.append(expert_candidate)
            components.append(expert_component)
    facts = DiscoveryFacts(
        expert_count=config.num_local_experts,
        expert_count_source="config.num_local_experts",
        routed_top_k=config.num_experts_per_tok,
        routed_top_k_source="config.num_experts_per_tok",
    )
    return DiscoveryReport(
        model_key=model_manifest.model_key,
        model_manifest=model_manifest,
        scanner_version=__version__,
        facts=facts,
        candidates=candidates,
        components=components,
        warnings=sorted(set(report_warnings)),
    )


class MixtralStaticAdapter:
    """Stateless static adapter for exact Mixtral module layouts."""

    __slots__ = ()

    @property
    def descriptor(self) -> AdapterDescriptor:
        return _DESCRIPTOR

    def detect(self, model: object, config: object) -> AdapterDetection:
        analysis = _analyze(model, expected_config=config)
        return _detection(analysis)

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        analysis = _analyze(model)
        return _report(model_manifest, analysis, self.descriptor)


__all__ = ["MixtralStaticAdapter"]
