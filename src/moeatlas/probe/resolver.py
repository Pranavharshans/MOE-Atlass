"""Deterministic resolution of probe targets against named modules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..core import validate_stable_identifier
from .plan import HookPoint, ProbePlan, ProbeTarget


class ProbeResolutionError(ValueError):
    """Raised when a serializable plan cannot bind to a module surface."""


@dataclass(frozen=True)
class ResolvedTarget:
    """One validated plan target and its duck-typed module object."""

    target: ProbeTarget
    module: object


@dataclass(frozen=True)
class ResolvedProbePlan:
    """Immutable resolution result used by the hook manager."""

    plan: ProbePlan
    targets: tuple[ResolvedTarget, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ProbePlan):
            raise TypeError("resolved plan must contain a validated ProbePlan")
        try:
            targets = tuple(self.targets)
        except TypeError as exc:
            raise TypeError(
                "resolved targets must be an iterable of ResolvedTarget objects"
            ) from exc
        if not targets:
            raise ProbeResolutionError("a resolved probe plan must contain at least one target")
        if any(not isinstance(target, ResolvedTarget) for target in targets):
            raise TypeError("resolved targets must contain only ResolvedTarget objects")
        if any(not isinstance(target.target, ProbeTarget) for target in targets):
            raise TypeError("resolved targets must contain validated ProbeTarget objects")
        paths = [target.target.module_path for target in targets]
        if len(set(paths)) != len(paths):
            raise ProbeResolutionError(
                "a resolved probe plan must not contain duplicate module paths or bindings"
            )
        ordered = tuple(sorted(targets, key=lambda target: target.target.module_path))
        expected = tuple(
            target
            for target in self.plan.targets
            if (not self.plan.include or target.module_path in self.plan.include)
            and target.module_path not in self.plan.exclude
        )
        actual = tuple(target.target for target in ordered)
        if actual != expected:
            raise ProbeResolutionError(
                "resolved targets must exactly match the source-plan selection "
                "after include/exclude filters"
            )
        object.__setattr__(self, "targets", ordered)

    @property
    def bindings(self) -> tuple[tuple[str, HookPoint, object], ...]:
        return tuple(
            (resolved.target.module_path, hook_point, resolved.module)
            for resolved in self.targets
            for hook_point in self.plan.hook_points
        )


_HOOK_METHODS = {
    HookPoint.FORWARD_PRE: "register_forward_pre_hook",
    HookPoint.FORWARD: "register_forward_hook",
    HookPoint.FULL_BACKWARD: "register_full_backward_hook",
}


def _pair_iterator(values: object) -> object:
    if isinstance(values, Mapping):
        return iter(values.items())
    return iter(values)  # type: ignore[arg-type]


def _collect_modules(model: object) -> dict[str, object]:
    method = getattr(model, "named_modules", None)
    if not callable(method):
        raise ProbeResolutionError(
            "probe resolution requires a callable model.named_modules() surface"
        )
    try:
        iterator = _pair_iterator(method())
    except Exception as exc:
        raise ProbeResolutionError(
            f"model.named_modules() could not be started: {type(exc).__name__}"
        ) from exc

    modules: dict[str, object] = {}
    position = 0
    while True:
        try:
            item = next(iterator)  # type: ignore[call-overload]
        except StopIteration:
            break
        except Exception as exc:
            raise ProbeResolutionError(
                f"model.named_modules() failed at item {position}: {type(exc).__name__}"
            ) from exc
        if not isinstance(item, tuple) or len(item) != 2:
            raise ProbeResolutionError(
                f"model.named_modules() item {position} is not a (name, module) pair"
            )
        path, module = item
        if not isinstance(path, str):
            raise ProbeResolutionError(
                f"model.named_modules() item {position} has a non-string path"
            )
        if path:
            try:
                validate_stable_identifier(path, field_name="module path")
            except (TypeError, ValueError) as exc:
                raise ProbeResolutionError(
                    f"model.named_modules() item {position} has an invalid path {path!r}"
                ) from exc
        if path in modules:
            raise ProbeResolutionError(f"model.named_modules() returned duplicate path {path!r}")
        modules[path] = module
        position += 1
    return dict(sorted(modules.items()))


def resolve_probe_plan(plan: ProbePlan, model: object) -> ResolvedProbePlan:
    """Resolve all selected targets and verify their hook surfaces."""

    if not isinstance(plan, ProbePlan):
        raise TypeError("plan must be a validated ProbePlan")
    modules = _collect_modules(model)
    selectors = set(plan.include) | set(plan.exclude)
    missing_selectors = sorted(path for path in selectors if path not in modules)
    if missing_selectors:
        raise ProbeResolutionError(
            f"probe include/exclude paths are missing from named_modules(): {missing_selectors!r}"
        )

    selected: list[ResolvedTarget] = []
    for target in plan.targets:
        if target.module_path not in modules:
            raise ProbeResolutionError(
                f"probe target module_path {target.module_path!r} is missing from named_modules()"
            )
        if plan.include and target.module_path not in plan.include:
            continue
        if target.module_path in plan.exclude:
            continue
        selected.append(ResolvedTarget(target=target, module=modules[target.module_path]))
    if not selected:
        raise ProbeResolutionError("probe include/exclude filters selected no targets")

    for resolved in selected:
        for hook_point in plan.hook_points:
            method_name = _HOOK_METHODS[hook_point]
            method = getattr(resolved.module, method_name, None)
            if not callable(method):
                raise ProbeResolutionError(
                    f"target {resolved.target.module_path!r} does not support "
                    f"{hook_point.value} ({method_name})"
                )
    return ResolvedProbePlan(plan=plan, targets=tuple(selected))


__all__ = [
    "ProbeResolutionError",
    "ResolvedProbePlan",
    "ResolvedTarget",
    "resolve_probe_plan",
]
