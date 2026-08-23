"""Structure-driven expert interventions for loaded Transformers models.

This module intentionally contains no model-family allowlist.  It resolves a
human layer/expert coordinate against the immutable discovery report and owns
only temporary forward-hook handles.  Models that do not expose each routed
expert as an independently hookable module are rejected explicitly; a fused
or packed implementation must never be reported as successfully ablated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from moeatlas.discovery import DiscoveryReport
from moeatlas.interventions.recipes import InterventionOperation, InterventionRecipe
from moeatlas.runtime.generic_capture import StructuredCaptureError, structured_expert_targets

_TARGET = re.compile(r"^layer:(0|[1-9][0-9]*)/expert:(0|[1-9][0-9]*)$")


class TransformersInterventionError(RuntimeError):
    """Safe failure for unsupported or invalid live expert interventions."""


@dataclass(frozen=True, slots=True)
class ExpertInterventionTarget:
    """One user-facing layer × expert coordinate bound to a real module."""

    label: str
    layer_index: int
    expert_index: int
    layer_key: str
    expert_key: str
    module_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "layer_index": self.layer_index,
            "expert_index": self.expert_index,
            "layer_key": self.layer_key,
            "expert_key": self.expert_key,
            "module_path": self.module_path,
        }


def intervention_targets(report: DiscoveryReport) -> tuple[ExpertInterventionTarget, ...]:
    """Return stable layer × expert coordinates for independently exposed experts."""

    try:
        discovered = structured_expert_targets(report)
    except StructuredCaptureError as exc:
        raise TransformersInterventionError(
            "model does not expose independently hookable routed experts"
        ) from exc
    grouped: dict[int, list[Any]] = {}
    for target in discovered:
        grouped.setdefault(target.layer_index, []).append(target)
    resolved: list[ExpertInterventionTarget] = []
    for layer_index in sorted(grouped):
        row = grouped[layer_index]
        for expert_index, target in enumerate(row):
            resolved.append(
                ExpertInterventionTarget(
                    label=f"layer:{layer_index}/expert:{expert_index}",
                    layer_index=layer_index,
                    expert_index=expert_index,
                    layer_key=target.layer_key,
                    expert_key=target.component_key,
                    module_path=target.module_path,
                )
            )
    if not resolved:
        raise TransformersInterventionError(
            "model does not expose independently hookable routed experts"
        )
    return tuple(resolved)


def _scale_output(output: object, factor: float) -> object:
    """Scale one tensor-like expert output without guessing nested semantics."""

    if output is None or isinstance(output, bool | int | float | str | bytes):
        raise TransformersInterventionError("expert output is not a tensor-like value")
    multiply = getattr(output, "mul", None)
    if callable(multiply):
        return multiply(factor)
    try:
        scaled = output * factor  # type: ignore[operator]
    except Exception as exc:
        raise TransformersInterventionError("expert output cannot be scaled safely") from exc
    if scaled is NotImplemented:
        raise TransformersInterventionError("expert output cannot be scaled safely")
    return scaled


class TransformersExpertInterventionCapability:
    """Temporary ablate/scale hooks resolved only from discovery evidence."""

    def __init__(self, report: DiscoveryReport) -> None:
        if type(report) is not DiscoveryReport:
            raise TypeError("report must be an exact DiscoveryReport")
        self._targets = intervention_targets(report)
        self._handles: list[object] = []
        self._invocations: dict[str, int] = {}

    @property
    def target_inventory(self) -> tuple[ExpertInterventionTarget, ...]:
        return self._targets

    @property
    def invocation_counts(self) -> dict[str, int]:
        return dict(sorted(self._invocations.items()))

    def capture(self, module: object) -> tuple[()]:
        del module
        if self._handles:
            raise TransformersInterventionError("intervention capability is already active")
        self._invocations = {}
        return ()

    def apply(self, module: object, recipe: InterventionRecipe) -> None:
        if recipe.operation not in {InterventionOperation.ABLATE, InterventionOperation.SCALE}:
            raise TransformersInterventionError(
                "live execution currently supports only expert ablation and scaling"
            )
        named_modules = getattr(module, "named_modules", None)
        if not callable(named_modules):
            raise TransformersInterventionError("loaded model does not expose named_modules()")
        try:
            modules = dict(named_modules())
        except Exception as exc:
            raise TransformersInterventionError("model module inventory is unavailable") from exc
        by_label = {target.label: target for target in self._targets}
        unknown = tuple(label for label in recipe.targets if label not in by_label)
        if unknown:
            raise TransformersInterventionError(
                "recipe targets are outside the discovered routed-expert universe"
            )
        factor = 0.0 if recipe.operation is InterventionOperation.ABLATE else recipe.factor
        assert factor is not None
        try:
            for label in recipe.targets:
                target = by_label[label]
                expert = modules.get(target.module_path)
                register = getattr(expert, "register_forward_hook", None)
                if not callable(register):
                    raise TransformersInterventionError(
                        "a selected expert does not support temporary forward hooks"
                    )

                def intervention_hook(
                    _module: object,
                    _inputs: object,
                    output: object,
                    *,
                    target_label: str = label,
                    scale_factor: float = factor,
                ) -> object:
                    self._invocations[target_label] = self._invocations.get(target_label, 0) + 1
                    return _scale_output(output, scale_factor)

                handle = register(intervention_hook)
                if not callable(getattr(handle, "remove", None)):
                    raise TransformersInterventionError(
                        "selected expert returned an unremovable hook handle"
                    )
                self._handles.append(handle)
        except BaseException:
            self._remove_handles()
            raise

    def restore(self, module: object, snapshot: object) -> None:
        del module
        if snapshot != ():
            raise TransformersInterventionError("intervention snapshot is invalid")
        self._remove_handles()

    def _remove_handles(self) -> None:
        failures: list[BaseException] = []
        while self._handles:
            handle = self._handles.pop()
            try:
                handle.remove()
            except BaseException as exc:
                failures.append(exc)
        if failures:
            raise TransformersInterventionError(
                f"failed to remove {len(failures)} intervention hook(s)"
            ) from failures[0]


def parse_intervention_target(label: str) -> tuple[int, int]:
    """Parse the public ``layer:N/expert:M`` coordinate format."""

    if type(label) is not str:
        raise TypeError("intervention target label must be a string")
    match = _TARGET.fullmatch(label)
    if match is None:
        raise ValueError("intervention target must use layer:N/expert:M")
    return int(match.group(1)), int(match.group(2))


__all__ = [
    "ExpertInterventionTarget",
    "TransformersExpertInterventionCapability",
    "TransformersInterventionError",
    "intervention_targets",
    "parse_intervention_target",
]
