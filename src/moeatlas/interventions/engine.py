"""Failure-safe intervention execution over capability adapters (PRD §9.3).

The engine is the only component allowed to mutate a module for causal
observation, and it does so under one hard contract: **no outcome is
produced unless the module was restored to its pre-intervention state.**
Every path — apply failure, observed-execution failure, cancellation, or
restore failure on the success path — restores first and only then
reports. Restoration itself failing is surfaced as a distinct ``restore``
stage so a dirty module is never mistaken for a clean one.

The engine stays family-blind: native snapshot/apply/restore semantics
live behind an ``InterventionCapability`` supplied by an adapter.
Synthetic modules prove the mechanics locally; real-model causal claims
stay deferred to the validation ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from moeatlas.interventions.recipes import (
    InterventionBudget,
    InterventionRecipe,
)

INTERVENTION_ENGINE_SCHEMA_VERSION = "1.0"
"""Schema version of the intervention engine contracts."""

_OUTCOME_ARTIFACT_TYPE = "moeatlas.intervention_outcome"

_ERROR_STAGES = frozenset({"contract", "capture", "apply", "execute", "restore"})


class InterventionEngineError(RuntimeError):
    """Safe fixed-stage failure for intervention execution."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("intervention engine error stage is not supported")
        self.stage = stage
        text = f"intervention failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


class InterventionCapability(Protocol):
    """Adapter-owned primitives the engine sequences; never family-aware."""

    def capture(self, module: object) -> object:
        """Return an opaque snapshot of every state the recipe may touch."""

    def restore(self, module: object, snapshot: object) -> None:
        """Return the module exactly to the captured state."""

    def apply(self, module: object, recipe: InterventionRecipe) -> None:
        """Apply one validated recipe to the module in place."""


@dataclass(frozen=True, slots=True)
class InterventionOutcome:
    """Evidence that one recipe was applied, observed, and fully restored.

    An outcome exists only after successful restoration; it is published
    as a ``moeatlas.intervention_outcome`` artifact and binds to the
    recipe by fingerprint so Evidence Cards can join causal observations
    to exact manipulations.
    """

    schema_version: str
    recipe_fingerprint: str
    operation: str
    targets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _OUTCOME_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "recipe_fingerprint": self.recipe_fingerprint,
            "operation": self.operation,
            "targets": list(self.targets),
        }

    def to_json(self) -> str:
        """Serialize this outcome with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> InterventionOutcome:
        """Validate one canonical JSON document into an exact outcome value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("intervention outcome document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("intervention outcome document must be a JSON object")
        if (
            document.get("artifact_type") != _OUTCOME_ARTIFACT_TYPE
            or document.get("schema_version") != INTERVENTION_ENGINE_SCHEMA_VERSION
        ):
            raise ValueError("document is not an intervention outcome artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                recipe_fingerprint=document["recipe_fingerprint"],
                operation=document["operation"],
                targets=tuple(document["targets"]),
            )
        except KeyError as exc:
            raise ValueError("intervention outcome document is missing fields") from exc


def _require_callable(capability: object, name: str) -> None:
    attribute = getattr(capability, name, None)
    if not callable(attribute):
        raise TypeError(f"capability must provide a callable {name}()")


def _restore_or_raise(
    capability: InterventionCapability,
    module: object,
    snapshot: object,
) -> None:
    """Restore, translating cleanup failure into the distinct restore stage."""

    try:
        capability.restore(module, snapshot)
    except BaseException as cleanup:
        raise InterventionEngineError("restore", "module restoration failed") from cleanup


def run_intervention(
    module: object,
    recipe: InterventionRecipe,
    capability: InterventionCapability,
    execute: Any,
    *,
    budget: InterventionBudget | None = None,
) -> InterventionOutcome:
    """Apply one recipe, observe once, and guarantee restoration.

    ``execute`` receives the mutated module exactly once; its return value
    is intentionally discarded because the outcome records the
    manipulation, not the observation — callers own their evidence. Any
    failure restores the module before reporting, and a failed restoration
    is reported at the ``restore`` stage with no outcome produced.
    """

    if module is None:
        raise TypeError("module must be provided")
    if type(recipe) is not InterventionRecipe:
        raise TypeError("recipe must be an InterventionRecipe")
    _require_callable(capability, "capture")
    _require_callable(capability, "restore")
    _require_callable(capability, "apply")
    if not callable(execute):
        raise TypeError("execute must be callable")
    if budget is None:
        budget = InterventionBudget()
    if type(budget) is not InterventionBudget:
        raise TypeError("budget must be an InterventionBudget or None")
    if len(recipe.targets) > budget.max_targets:
        raise InterventionEngineError(
            "contract",
            f"recipe has {len(recipe.targets)} targets; budget is {budget.max_targets}",
        )
    try:
        snapshot = capability.capture(module)
    except Exception as exc:
        raise InterventionEngineError("capture", "snapshot capture failed", cause=exc) from exc
    try:
        capability.apply(module, recipe)
    except BaseException as exc:
        _restore_or_raise(capability, module, snapshot)
        if not isinstance(exc, Exception):
            raise
        raise InterventionEngineError("apply", "recipe application failed", cause=exc) from exc
    try:
        execute(module)
    except BaseException as exc:
        _restore_or_raise(capability, module, snapshot)
        if not isinstance(exc, Exception):
            raise
        raise InterventionEngineError("execute", "observed execution failed", cause=exc) from exc
    _restore_or_raise(capability, module, snapshot)
    return InterventionOutcome(
        schema_version=INTERVENTION_ENGINE_SCHEMA_VERSION,
        recipe_fingerprint=recipe.fingerprint,
        operation=str(recipe.operation.value),
        targets=recipe.targets,
    )
