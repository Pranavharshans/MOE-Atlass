"""Immutable intervention recipes (PRD §9.3).

A recipe is a bounded, content-addressed description of exactly one
causal manipulation — expert ablation, expert scaling, expert rerouting,
or router-logit alteration — expressed purely in family-neutral labels.
Recipes carry no clocks, randomness, storage, or model knowledge; native
interpretation lives behind adapter capabilities (see
``moeatlas.interventions.engine``).

The fingerprint of a recipe is the ``sha256:<64 hex>`` digest of its
canonical form and is the value recorded in ``InterventionLineage`` on
derived run specifications, so lineage binds to exact recipe content.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from moeatlas.core.identity import stable_digest

INTERVENTION_SCHEMA_VERSION = "1.0"
"""Schema version of the intervention recipe contracts."""

_RECIPE_ARTIFACT_TYPE = "moeatlas.intervention_recipe"

_BUDGET_ARTIFACT_TYPE = "moeatlas.intervention_budget"

_ERROR_STAGES = frozenset({"contract", "serialization"})

_MAX_TARGETS = 1024
_MAX_TARGET_LABEL = 256


class InterventionBudgetError(RuntimeError):
    """Safe fixed-stage failure for intervention budget handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("intervention budget error stage is not supported")
        self.stage = stage
        text = f"intervention budget failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


class InterventionOperation(str, Enum):
    """The fixed vocabulary of bounded causal manipulations."""

    ABLATE = "ablate"
    SCALE = "scale"
    REROUTE = "reroute"
    ALTER_ROUTER = "alter_router"

    def __str__(self) -> str:
        return str(self.value)


def _strict_targets(value: object) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError("targets must be a tuple of strings")
    if not value:
        raise ValueError("targets must not be empty")
    if len(value) > _MAX_TARGETS:
        raise ValueError(f"targets must hold at most {_MAX_TARGETS} entries")
    for entry in value:
        if type(entry) is not str:
            raise TypeError("targets entries must be strings")
        if not entry:
            raise ValueError("targets entries must not be empty")
        if len(entry) > _MAX_TARGET_LABEL:
            raise ValueError(f"targets entries must hold at most {_MAX_TARGET_LABEL} characters")
    if list(value) != sorted(set(value)):
        raise ValueError("targets must be unique and sorted")
    return value  # type: ignore[return-value]


def _strict_optional_number(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if type(value) is not float and type(value) is not int:
        raise TypeError(f"{field_name} must be a number or None")
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{field_name} must be finite")
    return float(value)  # type: ignore[return-value]


def _strict_alternates(value: object) -> tuple[tuple[str, str], ...]:
    if type(value) is not tuple:
        raise TypeError("alternates must be a tuple of (target, alternate) pairs")
    pairs: list[tuple[str, str]] = []
    for entry in value:
        if type(entry) is not tuple or len(entry) != 2:
            raise TypeError("alternates entries must be (target, alternate) tuples")
        target, alternate = entry
        if type(target) is not str or type(alternate) is not str:
            raise TypeError("alternates entries must hold strings")
        if not target or not alternate:
            raise ValueError("alternates entries must not be empty")
        if len(target) > _MAX_TARGET_LABEL or len(alternate) > _MAX_TARGET_LABEL:
            raise ValueError(
                f"alternates entries must hold at most {_MAX_TARGET_LABEL} characters"
            )
        pairs.append((target, alternate))
    keys = [(target, alternate) for target, alternate in pairs]
    if keys != sorted(set(keys)):
        raise ValueError("alternates must be unique and sorted")
    return tuple(keys)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class InterventionRecipe:
    """One bounded causal manipulation over family-neutral target labels.

    Parameter exclusivity is enforced per operation: ``ablate`` takes no
    parameters, ``scale`` requires exactly ``factor``, ``alter_router``
    requires exactly ``bias``, and ``reroute`` requires ``alternates``
    covering exactly the target set with each alternate distinct from its
    target.
    """

    operation: InterventionOperation
    targets: tuple[str, ...]
    factor: float | None = None
    bias: float | None = None
    alternates: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.operation) is not InterventionOperation:
            raise TypeError("operation must be an InterventionOperation")
        _strict_targets(self.targets)
        factor = _strict_optional_number(self.factor, "factor")
        bias = _strict_optional_number(self.bias, "bias")
        alternates = _strict_alternates(self.alternates)
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "bias", bias)
        object.__setattr__(self, "alternates", alternates)
        if self.operation is InterventionOperation.SCALE and self.factor is None:
            raise ValueError("scale recipes require a finite factor")
        if self.operation is not InterventionOperation.SCALE and self.factor is not None:
            raise ValueError("factor is only valid on scale recipes")
        if self.operation is InterventionOperation.ALTER_ROUTER and self.bias is None:
            raise ValueError("alter_router recipes require a finite bias")
        if self.operation is not InterventionOperation.ALTER_ROUTER and self.bias is not None:
            raise ValueError("bias is only valid on alter_router recipes")
        if self.operation is not InterventionOperation.REROUTE and self.alternates:
            raise ValueError("alternates are only valid on reroute recipes")
        if self.operation is InterventionOperation.REROUTE:
            covered = tuple(target for target, _ in self.alternates)
            if covered != self.targets:
                raise ValueError("reroute alternates must cover exactly the target set")
            for target, alternate in self.alternates:
                if alternate == target:
                    raise ValueError("reroute alternates must differ from their targets")

    @property
    def fingerprint(self) -> str:
        """Content address of this exact recipe as ``sha256:<64 hex>``."""

        return f"sha256:{stable_digest(self.to_dict())}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _RECIPE_ARTIFACT_TYPE,
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "operation": str(self.operation.value),
            "targets": list(self.targets),
            "factor": self.factor,
            "bias": self.bias,
            "alternates": [[target, alternate] for target, alternate in self.alternates],
        }

    def to_json(self) -> str:
        """Serialize this recipe with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> InterventionRecipe:
        """Validate one canonical JSON document into an exact recipe value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("intervention recipe document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("intervention recipe document must be a JSON object")
        if (
            document.get("artifact_type") != _RECIPE_ARTIFACT_TYPE
            or document.get("schema_version") != INTERVENTION_SCHEMA_VERSION
        ):
            raise ValueError("document is not an intervention recipe artifact")
        try:
            operation = InterventionOperation(document["operation"])
        except KeyError as exc:
            raise ValueError("intervention recipe document is missing fields") from exc
        except ValueError as exc:
            raise ValueError("intervention recipe operation is not supported") from exc
        try:
            return cls(
                operation=operation,
                targets=tuple(document["targets"]),
                factor=document["factor"],
                bias=document["bias"],
                alternates=tuple(tuple(pair) for pair in document["alternates"]),
            )
        except KeyError as exc:
            raise ValueError("intervention recipe document is missing fields") from exc


@dataclass(frozen=True, slots=True)
class InterventionBudget:
    """Caller-enforced bounds for exactly one intervention execution.

    Budgets are immutable inputs to the engine, never ambient state: the
    same recipe run under a smaller budget fails loudly instead of
    silently narrowing the manipulation.
    """

    max_targets: int = 64

    def __post_init__(self) -> None:
        if type(self.max_targets) is not int or isinstance(self.max_targets, bool):
            raise TypeError("max_targets must be an integer")
        if self.max_targets <= 0:
            raise InterventionBudgetError("contract", "max_targets must be strictly positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _BUDGET_ARTIFACT_TYPE,
            "schema_version": INTERVENTION_SCHEMA_VERSION,
            "max_targets": self.max_targets,
        }

    def to_json(self) -> str:
        """Serialize this budget with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> InterventionBudget:
        """Validate one canonical JSON document into an exact budget value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("intervention budget document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("intervention budget document must be a JSON object")
        if (
            document.get("artifact_type") != _BUDGET_ARTIFACT_TYPE
            or document.get("schema_version") != INTERVENTION_SCHEMA_VERSION
        ):
            raise ValueError("document is not an intervention budget artifact")
        try:
            return cls(max_targets=document["max_targets"])
        except KeyError as exc:
            raise ValueError("intervention budget document is missing fields") from exc


def recipe_budget_from_json(payload: str | bytes | bytearray) -> InterventionBudget:
    """Alias of :meth:`InterventionBudget.from_json` for surface symmetry."""

    return InterventionBudget.from_json(payload)
