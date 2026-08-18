"""Static, model-runtime-independent Qwen3-MoE structure adapter.

Only the two explicit Qwen3-MoE reference surfaces described in the adapter
documentation are accepted here. The implementation intentionally uses
duck-typed named_modules(), named_parameters(), and configuration attributes;
it never imports a model runtime, reads parameter values, or retains state
between calls.
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

_LAYOUT_DENSE = "dense"
_LAYOUT_LEGACY = "legacy_indexed"
_LAYOUT_PACKED = "packed"

_ADAPTER_NAME = "huggingface-qwen3-moe-static"
_ADAPTER_VERSION = "1.0"
_METHOD = "qwen3-moe-static-structure-v1"
_PACKED_SLICE_WARNING = "packed expert slices are logical and are not independently hookable"

_UNSUPPORTED_WARNING = "Qwen3-MoE static structure is unsupported on this object"
_FAMILY_WARNING = "Qwen3-MoE family identity is missing or conflicting"
_CONFIG_WARNING = "Qwen3-MoE configuration fields or sparse schedule are invalid"
_TOPOLOGY_WARNING = "Qwen3-MoE module topology is incomplete, conflicting, or mixed"
_ATTRIBUTE_WARNING = "Qwen3-MoE structural attributes are missing or inconsistent"
_SHAPE_WARNING = "Qwen3-MoE parameter shapes do not match the selected layout"
_PARAMETER_ROOT_WARNING = "Qwen3-MoE parameter roots are conflicting or malformed"

_ARCHITECTURES = frozenset(
    {
        "Qwen3MoeModel",
        "Qwen3MoeForCausalLM",
        "Qwen3MoeForSequenceClassification",
        "Qwen3MoeForTokenClassification",
        "Qwen3MoeForQuestionAnswering",
    }
)

_INT_CONFIG_FIELDS = (
    "num_hidden_layers",
    "hidden_size",
    "intermediate_size",
    "moe_intermediate_size",
    "num_experts",
    "num_experts_per_tok",
    "decoder_sparse_step",
)

_DESCRIPTOR = AdapterDescriptor(
    name=_ADAPTER_NAME,
    version=_ADAPTER_VERSION,
    architecture_families=("qwen3_moe",),
    compatibility_notes=(
        "official Transformers 4.51.3 and 4.57.1 indexed reference layouts are supported",
        "official Transformers 5.0.0 packed reference layout is supported as logical slices",
        "structure-only; routing certification is not provided",
    ),
)


@dataclass(frozen=True)
class _ConfigFacts:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    decoder_sparse_step: int
    norm_topk_prob: bool
    mlp_only_layers: tuple[int, ...]
    sparse_indices: tuple[int, ...]
    dense_indices: tuple[int, ...]
    family_evidence: tuple[str, ...]


@dataclass(frozen=True)
class _Layer:
    prefix: str
    index: int
    layout: str
    mlp_path: str
    gate_path: str
    experts_path: str
    expert_paths: tuple[str, ...]


@dataclass(frozen=True)
class _Analysis:
    config: _ConfigFacts | None
    layers: tuple[_Layer, ...]
    parameters: dict[str, object]
    layout: str | None
    warnings: tuple[str, ...]
    family_evidence: tuple[str, ...]
    topology_valid: bool
    attributes_valid: bool
    shapes_valid: bool

    @property
    def valid(self) -> bool:
        return (
            self.config is not None
            and bool(self.layers)
            and self.layout in {_LAYOUT_LEGACY, _LAYOUT_PACKED}
            and self.topology_valid
            and self.attributes_valid
            and self.shapes_valid
        )


class _ConfigReadError(Exception):
    """Private marker retaining only the safe field name and exception type."""

    def __init__(self, field_name: str, cause: Exception) -> None:
        self.field_name = field_name
        self.cause = cause
        super().__init__()


def _warning_for_exception(operation: str, exc: Exception) -> str:
    """Format an ordinary failure without copying its potentially sensitive text."""

    return f"{operation} failed with {type(exc).__name__}"


def _zero_detection(*warnings: str) -> AdapterDetection:
    normalized = tuple(sorted(set(warnings))) or (_UNSUPPORTED_WARNING,)
    return AdapterDetection(score=0.0, warnings=normalized)


def _config_value(config: object, field_name: str) -> object:
    if isinstance(config, Mapping):
        try:
            if field_name not in config:
                return _MISSING
            return config[field_name]
        except Exception as exc:
            raise _ConfigReadError(field_name, exc) from exc
    try:
        return getattr(config, field_name)
    except AttributeError:
        return _MISSING
    except Exception as exc:
        raise _ConfigReadError(field_name, exc) from exc


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

    if model_type is not _MISSING and (type(model_type) is not str or model_type != "qwen3_moe"):
        return (), _FAMILY_WARNING

    architecture_values: tuple[str, ...] = ()
    if architectures is not _MISSING:
        if not isinstance(architectures, list | tuple):
            return (), _FAMILY_WARNING
        try:
            architecture_values = tuple(architectures)
        except Exception as exc:
            return (), _warning_for_exception("config.architectures iteration", exc)
        if any(type(value) is not str for value in architecture_values):
            return (), _FAMILY_WARNING
        if len(set(architecture_values)) != len(architecture_values):
            return (), _FAMILY_WARNING
        if any(value not in _ARCHITECTURES for value in architecture_values):
            return (), _FAMILY_WARNING

    has_model_type = model_type is not _MISSING
    has_architecture = bool(architecture_values)
    if not has_model_type and not has_architecture:
        return (), _FAMILY_WARNING

    evidence: list[str] = []
    if has_model_type:
        evidence.append("model_type:qwen3_moe")
    if has_architecture:
        evidence.append("architecture:qwen3-moe-allowlist")
    return tuple(sorted(evidence)), None


def _strict_config(
    config: object, family_evidence: tuple[str, ...]
) -> tuple[_ConfigFacts | None, str | None]:
    values: dict[str, int] = {}
    for field_name in _INT_CONFIG_FIELDS:
        try:
            value = _config_value(config, field_name)
        except _ConfigReadError as exc:
            return None, _warning_for_exception(f"config.{exc.field_name} access", exc.cause)
        if type(value) is not int or value <= 0:
            return None, _CONFIG_WARNING
        values[field_name] = value

    if values["num_experts_per_tok"] > values["num_experts"]:
        return None, _CONFIG_WARNING

    try:
        norm_topk_prob = _config_value(config, "norm_topk_prob")
        mlp_only_layers = _config_value(config, "mlp_only_layers")
    except _ConfigReadError as exc:
        return None, _warning_for_exception(f"config.{exc.field_name} access", exc.cause)

    if type(norm_topk_prob) is not bool:
        return None, _CONFIG_WARNING
    if type(mlp_only_layers) is not list:
        return None, _CONFIG_WARNING
    if any(type(index) is not int for index in mlp_only_layers):
        return None, _CONFIG_WARNING
    if len(set(mlp_only_layers)) != len(mlp_only_layers):
        return None, _CONFIG_WARNING
    if any(index < 0 or index >= values["num_hidden_layers"] for index in mlp_only_layers):
        return None, _CONFIG_WARNING

    excluded = set(mlp_only_layers)
    sparse_indices = tuple(
        index
        for index in range(values["num_hidden_layers"])
        if index not in excluded and (index + 1) % values["decoder_sparse_step"] == 0
    )
    if not sparse_indices:
        return None, _CONFIG_WARNING
    dense_indices = tuple(
        index for index in range(values["num_hidden_layers"]) if index not in sparse_indices
    )

    return (
        _ConfigFacts(
            num_hidden_layers=values["num_hidden_layers"],
            hidden_size=values["hidden_size"],
            intermediate_size=values["intermediate_size"],
            moe_intermediate_size=values["moe_intermediate_size"],
            num_experts=values["num_experts"],
            num_experts_per_tok=values["num_experts_per_tok"],
            decoder_sparse_step=values["decoder_sparse_step"],
            norm_topk_prob=norm_topk_prob,
            mlp_only_layers=tuple(mlp_only_layers),
            sparse_indices=sparse_indices,
            dense_indices=dense_indices,
            family_evidence=family_evidence,
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
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            return {}, (_warning_for_exception("named_modules() iteration", exc),)
        if not isinstance(item, tuple | list) or len(item) != 2:
            return {}, ("named_modules() surface is malformed or duplicated",)
        path, module = item
        if type(path) is not str or path in entries:
            return {}, ("named_modules() surface is malformed or duplicated",)
        entries[path] = module
    if not entries:
        return {}, ("named_modules() surface is empty",)
    return dict(sorted(entries.items())), ()


def _collect_parameters(model: object) -> tuple[dict[str, object], tuple[str, ...]]:
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

    entries: dict[str, object] = {}
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            return {}, (_warning_for_exception("named_parameters() iteration", exc),)
        if not isinstance(item, tuple | list) or len(item) != 2:
            return {}, ("named_parameters() surface is malformed or duplicated",)
        name, parameter = item
        if type(name) is not str or not name or name in entries:
            return {}, ("named_parameters() surface is malformed or duplicated",)
        entries[name] = parameter
    if not entries:
        return {}, ("named_parameters() surface is empty",)
    return dict(sorted(entries.items())), ()


def _canonical_index(token: str) -> int | None:
    """Return an ordinary decimal index, rejecting aliases such as 01."""

    if not token or any(character not in "0123456789" for character in token):
        return None
    if len(token) > 1 and token.startswith("0"):
        return None
    return int(token)


def _parse_layer_path(path: str) -> tuple[str, int, tuple[str, ...]] | None:
    if path == "":
        return None
    parts = tuple(path.split("."))
    if any(part == "" for part in parts):
        raise ValueError("empty path component")
    positions = [position for position, part in enumerate(parts) if part == "layers"]
    candidate_positions = [position for position in positions if position + 1 < len(parts)]
    if not candidate_positions:
        return None
    if len(candidate_positions) != 1:
        raise ValueError("multiple layers segments")
    position = candidate_positions[0]
    index = _canonical_index(parts[position + 1])
    if index is None:
        raise ValueError("non-canonical layer index")
    return ".".join(parts[:position]), index, parts[position + 2 :]


def _module_path(prefix: str, index: int, suffix: tuple[str, ...]) -> str:
    return ".".join(part for part in (prefix, "layers", str(index), *suffix) if part)


def _expected_dense_suffixes() -> set[tuple[str, ...]]:
    return {
        ("mlp",),
        ("mlp", "gate_proj"),
        ("mlp", "up_proj"),
        ("mlp", "down_proj"),
        ("mlp", "act_fn"),
    }


def _expected_legacy_suffixes(expert_count: int) -> set[tuple[str, ...]]:
    suffixes = {
        ("mlp",),
        ("mlp", "gate"),
        ("mlp", "experts"),
    }
    for expert_index in range(expert_count):
        expert = ("mlp", "experts", str(expert_index))
        suffixes.add(expert)
        suffixes.update(
            expert + (name,) for name in ("gate_proj", "up_proj", "down_proj", "act_fn")
        )
    return suffixes


def _expected_packed_suffixes() -> set[tuple[str, ...]]:
    return {
        ("mlp",),
        ("mlp", "gate"),
        ("mlp", "experts"),
        ("mlp", "experts", "act_fn"),
    }


def _topology(
    modules: Mapping[str, object], config: _ConfigFacts
) -> tuple[tuple[_Layer, ...], str | None, tuple[str, ...]]:
    roots: dict[str, dict[int, set[tuple[str, ...]]]] = {}
    parsed_prefixes: dict[str, set[int]] = {}
    try:
        for path in modules:
            try:
                parsed = _parse_layer_path(path)
            except ValueError:
                return (), None, (_TOPOLOGY_WARNING,)
            if parsed is None:
                continue
            prefix, index, suffix = parsed
            parsed_prefixes.setdefault(prefix, set()).add(index)
            if suffix and suffix[0] != "mlp":
                if index >= config.num_hidden_layers:
                    return (), None, (_TOPOLOGY_WARNING,)
                continue
            roots.setdefault(prefix, {}).setdefault(index, set()).add(tuple(suffix))
    except Exception as exc:
        return (), None, (_warning_for_exception("module topology analysis", exc),)

    if len(parsed_prefixes) != 1 or len(roots) != 1:
        return (), None, (_TOPOLOGY_WARNING,)
    prefix, indexed = next(iter(roots.items()))
    if set(indexed) != set(range(config.num_hidden_layers)):
        return (), None, (_TOPOLOGY_WARNING,)
    layers_path = ".".join(part for part in (prefix, "layers") if part)
    if layers_path not in modules:
        return (), None, (_TOPOLOGY_WARNING,)

    layers: list[_Layer] = []
    sparse_layout: str | None = None
    for index in range(config.num_hidden_layers):
        suffixes = indexed[index]
        if () not in suffixes:
            return (), None, (_TOPOLOGY_WARNING,)
        mlp_suffixes = {suffix for suffix in suffixes if suffix[:1] == ("mlp",)}
        if not mlp_suffixes:
            return (), None, (_TOPOLOGY_WARNING,)

        expected_sparse = index in config.sparse_indices
        if not expected_sparse:
            if mlp_suffixes != _expected_dense_suffixes():
                return (), None, (_TOPOLOGY_WARNING,)
            layers.append(
                _Layer(
                    prefix=prefix,
                    index=index,
                    layout=_LAYOUT_DENSE,
                    mlp_path=_module_path(prefix, index, ("mlp",)),
                    gate_path="",
                    experts_path="",
                    expert_paths=(),
                )
            )
            continue

        has_legacy = any(
            len(suffix) >= 3
            and suffix[:2] == ("mlp", "experts")
            and _canonical_index(suffix[2]) is not None
            for suffix in mlp_suffixes
        )
        has_packed = ("mlp", "experts", "act_fn") in mlp_suffixes
        if has_legacy and has_packed:
            return (), None, (_TOPOLOGY_WARNING,)
        if has_legacy:
            layout = _LAYOUT_LEGACY
            expected = _expected_legacy_suffixes(config.num_experts)
            if mlp_suffixes != expected:
                return (), None, (_TOPOLOGY_WARNING,)
            expert_paths = tuple(
                _module_path(prefix, index, ("mlp", "experts", str(expert_index)))
                for expert_index in range(config.num_experts)
            )
        elif has_packed:
            layout = _LAYOUT_PACKED
            if mlp_suffixes != _expected_packed_suffixes():
                return (), None, (_TOPOLOGY_WARNING,)
            expert_paths = ()
        else:
            return (), None, (_TOPOLOGY_WARNING,)

        if sparse_layout is not None and sparse_layout != layout:
            return (), None, (_TOPOLOGY_WARNING,)
        sparse_layout = layout
        layers.append(
            _Layer(
                prefix=prefix,
                index=index,
                layout=layout,
                mlp_path=_module_path(prefix, index, ("mlp",)),
                gate_path=_module_path(prefix, index, ("mlp", "gate")),
                experts_path=_module_path(prefix, index, ("mlp", "experts")),
                expert_paths=expert_paths,
            )
        )

    if sparse_layout is None:
        return (), None, (_TOPOLOGY_WARNING,)
    return tuple(layers), sparse_layout, ()


def _parameter_roots(
    parameters: Mapping[str, object], expected_prefix: str, layer_count: int
) -> tuple[str, ...]:
    roots: set[str] = set()
    parsed_prefixes: set[str] = set()
    for name in parameters:
        try:
            parsed = _parse_layer_path(name)
        except ValueError:
            return (_PARAMETER_ROOT_WARNING,)
        if parsed is None:
            continue
        prefix, index, suffix = parsed
        if index >= layer_count:
            return (_PARAMETER_ROOT_WARNING,)
        parsed_prefixes.add(prefix)
        if suffix[:1] == ("mlp",):
            roots.add(prefix)
    if parsed_prefixes and parsed_prefixes != {expected_prefix}:
        return (_PARAMETER_ROOT_WARNING,)
    if roots and roots != {expected_prefix}:
        return (_PARAMETER_ROOT_WARNING,)
    return ()


def _attribute_warning(
    modules: Mapping[str, object],
    path: str,
    attribute: str,
    expected: object,
    label: str,
) -> str | None:
    module = modules.get(path, _MISSING)
    if module is _MISSING:
        return _ATTRIBUTE_WARNING
    try:
        value = getattr(module, attribute)
    except AttributeError:
        return _ATTRIBUTE_WARNING
    except Exception as exc:
        return _warning_for_exception(f"Qwen3-MoE {label} access", exc)
    if type(value) is not type(expected) or value != expected:
        return _ATTRIBUTE_WARNING
    return None


def _validate_attributes(
    layers: tuple[_Layer, ...],
    modules: Mapping[str, object],
    config: _ConfigFacts,
) -> tuple[bool, tuple[str, ...]]:
    warnings: list[str] = []
    for layer in layers:
        if layer.layout == _LAYOUT_DENSE:
            checks = (
                (layer.mlp_path, "hidden_size", config.hidden_size, "dense hidden_size"),
                (
                    layer.mlp_path,
                    "intermediate_size",
                    config.intermediate_size,
                    "dense intermediate_size",
                ),
            )
        elif layer.layout == _LAYOUT_LEGACY:
            checks = (
                (layer.mlp_path, "num_experts", config.num_experts, "legacy num_experts"),
                (layer.mlp_path, "top_k", config.num_experts_per_tok, "legacy top_k"),
                (
                    layer.mlp_path,
                    "norm_topk_prob",
                    config.norm_topk_prob,
                    "legacy norm_topk_prob",
                ),
            )
            checks += tuple(
                (expert_path, "hidden_size", config.hidden_size, "legacy expert hidden_size")
                for expert_path in layer.expert_paths
            )
            checks += tuple(
                (
                    expert_path,
                    "intermediate_size",
                    config.moe_intermediate_size,
                    "legacy expert intermediate_size",
                )
                for expert_path in layer.expert_paths
            )
        else:
            checks = (
                (layer.gate_path, "top_k", config.num_experts_per_tok, "packed gate top_k"),
                (layer.gate_path, "num_experts", config.num_experts, "packed gate num_experts"),
                (
                    layer.gate_path,
                    "norm_topk_prob",
                    config.norm_topk_prob,
                    "packed gate norm_topk_prob",
                ),
                (layer.gate_path, "hidden_dim", config.hidden_size, "packed gate hidden_dim"),
                (
                    layer.experts_path,
                    "num_experts",
                    config.num_experts,
                    "packed experts num_experts",
                ),
                (
                    layer.experts_path,
                    "hidden_dim",
                    config.hidden_size,
                    "packed experts hidden_dim",
                ),
                (
                    layer.experts_path,
                    "intermediate_dim",
                    config.moe_intermediate_size,
                    "packed experts intermediate_dim",
                ),
            )
        for path, attribute, expected, label in checks:
            warning = _attribute_warning(modules, path, attribute, expected, label)
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
        return None, _SHAPE_WARNING
    return tuple(dimensions), None


def _expected_parameter_shapes(layer: _Layer, config: _ConfigFacts) -> dict[str, tuple[int, ...]]:
    if layer.layout == _LAYOUT_DENSE:
        return {
            f"{layer.mlp_path}.gate_proj.weight": (
                config.intermediate_size,
                config.hidden_size,
            ),
            f"{layer.mlp_path}.up_proj.weight": (
                config.intermediate_size,
                config.hidden_size,
            ),
            f"{layer.mlp_path}.down_proj.weight": (
                config.hidden_size,
                config.intermediate_size,
            ),
        }
    if layer.layout == _LAYOUT_LEGACY:
        expected = {f"{layer.mlp_path}.gate.weight": (config.num_experts, config.hidden_size)}
        for expert_path in layer.expert_paths:
            expected.update(
                {
                    f"{expert_path}.gate_proj.weight": (
                        config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                    f"{expert_path}.up_proj.weight": (
                        config.moe_intermediate_size,
                        config.hidden_size,
                    ),
                    f"{expert_path}.down_proj.weight": (
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ),
                }
            )
        return expected
    return {
        f"{layer.gate_path}.weight": (config.num_experts, config.hidden_size),
        f"{layer.experts_path}.gate_up_proj": (
            config.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
        ),
        f"{layer.experts_path}.down_proj": (
            config.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
        ),
    }


def _validate_shapes(
    layers: tuple[_Layer, ...],
    parameters: Mapping[str, object],
    config: _ConfigFacts,
) -> tuple[bool, tuple[str, ...]]:
    valid = True
    warnings: list[str] = []
    for layer in layers:
        expected = _expected_parameter_shapes(layer, config)
        actual = {name for name in parameters if name.startswith(f"{layer.mlp_path}.")}
        if actual != set(expected):
            valid = False
            warnings.append(_SHAPE_WARNING)
        for name, expected_shape in expected.items():
            parameter = parameters.get(name, _MISSING)
            if parameter is _MISSING:
                continue
            actual_shape, shape_warning = _shape(parameter)
            if shape_warning is not None or actual_shape != expected_shape:
                valid = False
                warnings.append(shape_warning or _SHAPE_WARNING)
    return valid, tuple(sorted(set(warnings)))


def _analyze(model: object, expected_config: object = _MISSING) -> _Analysis:
    config, config_warning = _model_config(model)
    if config is None:
        return _Analysis(
            None,
            (),
            {},
            None,
            (config_warning or _UNSUPPORTED_WARNING,),
            (),
            False,
            False,
            False,
        )
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
            False,
        )

    family_evidence, family_warning = _family_facts(config)
    if family_warning is not None:
        return _Analysis(None, (), {}, None, (family_warning,), (), False, False, False)
    config_facts, config_warning = _strict_config(config, family_evidence)
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
            False,
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
            False,
        )
    parameters, parameter_warnings = _collect_parameters(model)
    if parameter_warnings:
        return _Analysis(
            config_facts,
            (),
            parameters,
            None,
            parameter_warnings,
            family_evidence,
            False,
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
            False,
        )

    parameter_root_warnings = _parameter_roots(
        parameters, layers[0].prefix, config_facts.num_hidden_layers
    )
    if parameter_root_warnings:
        return _Analysis(
            config_facts,
            layers,
            parameters,
            layout,
            parameter_root_warnings,
            family_evidence,
            False,
            False,
            False,
        )

    attributes_valid, attribute_warnings = _validate_attributes(layers, modules, config_facts)
    shapes_valid, shape_warnings = _validate_shapes(layers, parameters, config_facts)
    warnings = tuple(sorted(set((*attribute_warnings, *shape_warnings))))
    return _Analysis(
        config_facts,
        layers,
        parameters,
        layout,
        warnings,
        family_evidence,
        True,
        attributes_valid,
        shapes_valid,
    )


def _detection(analysis: _Analysis) -> AdapterDetection:
    if not analysis.valid:
        return _zero_detection(*analysis.warnings)
    evidence = list(analysis.family_evidence)
    evidence.extend(
        (
            "config:strict-fields-and-schedule",
            "topology:complete-layout",
            "shapes:exact",
        )
    )
    weights = {
        "model_type:qwen3_moe": 0.25,
        "architecture:qwen3-moe-allowlist": 0.15,
        "config:strict-fields-and-schedule": 0.20,
        "topology:complete-layout": 0.20,
        "shapes:exact": 0.20,
    }
    score = round(sum(weights[item] for item in evidence), 3)
    return AdapterDetection(score=float(score), evidence=tuple(evidence))


def _component_data(
    model_manifest: ModelManifest,
    layer: _Layer,
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
    detail = f"Qwen3-MoE {layer.layout} layout"
    evidence = [
        DiscoveryEvidence(
            signal=DiscoverySignal.CHILD_STRUCTURE,
            detail=detail,
            weight=1.0,
        )
    ]
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
    provenance = Provenance(source="qwen3-moe-static", tool_version=__version__, metadata=metadata)
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
        if layer.layout == _LAYOUT_DENSE:
            continue

        layer_candidate, layer_component = _component_data(
            model_manifest,
            layer,
            descriptor,
            kind=ComponentKind.MOE_LAYER,
            module_path=layer.mlp_path,
            expert_index=None,
            tensor_shapes={},
            routed=None,
            warnings=[],
        )
        candidates.append(layer_candidate)
        components.append(layer_component)

        router_candidate, router_component = _component_data(
            model_manifest,
            layer,
            descriptor,
            kind=ComponentKind.ROUTER,
            module_path=layer.gate_path,
            expert_index=None,
            tensor_shapes={"weight": [config.num_experts, config.hidden_size]},
            routed=None,
            warnings=[],
        )
        candidates.append(router_candidate)
        components.append(router_component)

        if layer.layout == _LAYOUT_LEGACY:
            container_shapes: dict[str, list[int]] = {}
        else:
            container_shapes = {
                "gate_up_proj": [
                    config.num_experts,
                    2 * config.moe_intermediate_size,
                    config.hidden_size,
                ],
                "down_proj": [
                    config.num_experts,
                    config.hidden_size,
                    config.moe_intermediate_size,
                ],
            }
        container_candidate, container_component = _component_data(
            model_manifest,
            layer,
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

        for expert_index in range(config.num_experts):
            if layer.layout == _LAYOUT_LEGACY:
                expert_path = layer.expert_paths[expert_index]
                expert_shapes = {
                    "gate_proj.weight": [
                        config.moe_intermediate_size,
                        config.hidden_size,
                    ],
                    "up_proj.weight": [
                        config.moe_intermediate_size,
                        config.hidden_size,
                    ],
                    "down_proj.weight": [
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ],
                }
                expert_warnings: list[str] = []
            else:
                expert_path = layer.experts_path
                expert_shapes = {
                    "gate_up_proj": [
                        2 * config.moe_intermediate_size,
                        config.hidden_size,
                    ],
                    "down_proj": [
                        config.hidden_size,
                        config.moe_intermediate_size,
                    ],
                }
                expert_warnings = [_PACKED_SLICE_WARNING]
                report_warnings.append(_PACKED_SLICE_WARNING)

            expert_candidate, expert_component = _component_data(
                model_manifest,
                layer,
                descriptor,
                kind=ComponentKind.EXPERT,
                module_path=expert_path,
                expert_index=expert_index,
                tensor_shapes=expert_shapes,
                routed=True,
                warnings=expert_warnings,
            )
            candidates.append(expert_candidate)
            components.append(expert_component)

    facts = DiscoveryFacts(
        expert_count=config.num_experts,
        expert_count_source="config.num_experts",
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


class Qwen3MoeStaticAdapter:
    """Stateless static adapter for exact Qwen3-MoE module layouts."""

    __slots__ = ()

    @property
    def descriptor(self) -> AdapterDescriptor:
        return _DESCRIPTOR

    def detect(self, model: object, config: object) -> AdapterDetection:
        return _detection(_analyze(model, expected_config=config))

    def discover(self, model: object, model_manifest: ModelManifest) -> DiscoveryReport:
        return _report(model_manifest, _analyze(model), self.descriptor)


__all__ = ["Qwen3MoeStaticAdapter"]
