"""Public orchestration for model-runtime-independent static discovery."""

from __future__ import annotations

from .. import __version__
from ..core import (
    CapabilityLabel,
    ComponentManifest,
    ModelManifest,
    Provenance,
    make_component_key,
    parse_model_key,
)
from .heuristics import candidate_specs, facts_from_specs
from .models import DiscoveryCandidate, DiscoveryReport
from .surface import collect_config, collect_named_modules, collect_parameter_shapes


def scan(model: object, model_manifest: ModelManifest) -> DiscoveryReport:
    """Scan a PyTorch-compatible object without importing or mutating it.

    The object must expose a callable ``named_modules()`` method. A callable
    ``named_parameters()`` method and a ``config`` mapping/object are optional.
    All malformed or unavailable optional surfaces become deterministic report
    warnings rather than runtime/model imports.
    """

    if not isinstance(model_manifest, ModelManifest):
        raise TypeError(
            "model_manifest must be a validated ModelManifest; static discovery never builds one"
        )
    parse_model_key(model_manifest.model_key)

    warnings: list[str] = []
    modules = collect_named_modules(model, warnings)
    parameter_shapes = collect_parameter_shapes(model, warnings)
    config = collect_config(model, warnings)
    specs = candidate_specs(modules, parameter_shapes, config)
    facts, fact_warnings = facts_from_specs(config, specs)
    warnings.extend(fact_warnings)

    candidates: list[DiscoveryCandidate] = []
    components: list[ComponentManifest] = []
    for spec in specs:
        component_key = make_component_key(
            model_manifest.model_key,
            spec.kind.value,
            spec.module_path,
            layer_index=spec.layer_index,
            expert_index=spec.expert_index,
        )
        candidate = DiscoveryCandidate(
            component_key=component_key,
            model_key=model_manifest.model_key,
            kind=spec.kind,
            module_path=spec.module_path,
            layer_index=spec.layer_index,
            expert_index=spec.expert_index,
            confidence=spec.confidence,
            evidence=list(spec.evidence),
            routed=spec.routed,
            shared=spec.shared,
            warnings=list(spec.warnings),
        )
        provenance = Provenance(
            source="static-discovery",
            tool_version=__version__,
            metadata={
                "scanner": "duck-typed-static-v1",
                "confidence": spec.confidence,
                "signals": [item.signal.value for item in spec.evidence],
            },
        )
        component = ComponentManifest(
            component_key=component_key,
            model_key=model_manifest.model_key,
            kind=spec.kind,
            module_path=spec.module_path,
            layer_index=spec.layer_index,
            expert_index=spec.expert_index,
            tensor_shapes=spec.tensor_shapes,
            capabilities=[CapabilityLabel.STRUCTURE],
            routed=spec.routed,
            shared=spec.shared,
            provenance=provenance,
            warnings=list(spec.warnings),
        )
        candidates.append(candidate)
        components.append(component)

    warnings.extend(spec_warning for spec in specs for spec_warning in spec.warnings)
    return DiscoveryReport(
        model_key=model_manifest.model_key,
        model_manifest=model_manifest,
        scanner_version=__version__,
        facts=facts,
        candidates=candidates,
        components=components,
        warnings=sorted(set(warnings)),
    )


discover = scan


__all__ = ["discover", "scan"]
