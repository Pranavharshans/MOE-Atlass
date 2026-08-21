"""Reproducible benchmark plans and result bundles (PRD §19).

A benchmark **plan** is a reproducible workload definition: named cases,
each carrying a canonical-JSON workload descriptor and an operation count.
A **result bundle** joins measured values back to a plan by digest.
Measurements are always caller-supplied — this module owns no clock, so a
bundle cannot exist without explicit ``environment`` and ``recorded_at``
provenance, and every bundle is stamped ``release_evidence: false``:
developer-machine timing becomes release evidence only through the review
process, never through this API.

The layer is pure and deterministic: no storage reads, randomness, model
knowledge, or wall-clock access.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from moeatlas.core.identity import stable_digest

BENCHMARK_SCHEMA_VERSION = "1.0"
"""Schema version of the benchmark contracts."""

_PLAN_ARTIFACT_TYPE = "moeatlas.benchmark_plan"

_RESULT_ARTIFACT_TYPE = "moeatlas.benchmark_results"

_ERROR_STAGES = frozenset({"contract", "budget"})

_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_MAX_NAME = 200


class BenchmarkError(RuntimeError):
    """Safe fixed-stage failure for benchmark handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("benchmark error stage is not supported")
        self.stage = stage
        text = f"benchmark failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_name(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field_name} must be a string")
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if len(value) > _MAX_NAME:
        raise ValueError(f"{field_name} must hold at most {_MAX_NAME} characters")
    return value


def _canonical_workload(value: object) -> str:
    _strict_name(value, "workload")
    try:
        parsed = json.loads(value)  # type: ignore[arg-type]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("workload must be valid JSON") from exc
    canonical = json.dumps(
        parsed,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if canonical != value:
        raise ValueError("workload must be canonical JSON")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One reproducible workload definition.

    ``workload`` is a canonical-JSON descriptor of the deterministic
    inputs and settings the case measures; ``operations`` counts the work
    units so results can be normalized across machines honestly.
    """

    name: str
    workload: str
    operations: int

    def __post_init__(self) -> None:
        _strict_name(self.name, "name")
        _canonical_workload(self.workload)
        if type(self.operations) is not int or isinstance(self.operations, bool):
            raise TypeError("operations must be an integer")
        if self.operations <= 0:
            raise BenchmarkError("contract", "operations must be strictly positive")


@dataclass(frozen=True, slots=True)
class BenchmarkPlan:
    """Sorted set of benchmark cases forming one reproducible plan."""

    schema_version: str
    cases: tuple[BenchmarkCase, ...]

    def __post_init__(self) -> None:
        if type(self.cases) is not tuple:
            raise TypeError("cases must be a tuple of BenchmarkCase entries")
        if not self.cases:
            raise BenchmarkError("contract", "cases must not be empty")
        names = [case.name for case in self.cases]
        if names != sorted(set(names)):
            raise ValueError("case names must be unique and sorted")
        for case in self.cases:
            if type(case) is not BenchmarkCase:
                raise TypeError("cases entries must be BenchmarkCase values")

    @property
    def fingerprint(self) -> str:
        """Content address of this exact plan as ``sha256:<64 hex>``."""

        return f"sha256:{stable_digest(self.to_dict())}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _PLAN_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "cases": [
                {
                    "name": case.name,
                    "workload": json.loads(case.workload),
                    "operations": case.operations,
                }
                for case in self.cases
            ],
        }

    def to_json(self) -> str:
        """Serialize this plan with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> BenchmarkPlan:
        """Validate one canonical JSON document into an exact plan value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("benchmark plan document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("benchmark plan document must be a JSON object")
        if (
            document.get("artifact_type") != _PLAN_ARTIFACT_TYPE
            or document.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        ):
            raise ValueError("document is not a benchmark plan artifact")
        try:
            cases = []
            for entry in document["cases"]:
                workload = entry["workload"]
                cases.append(
                    BenchmarkCase(
                        name=entry["name"],
                        workload=json.dumps(
                            workload,
                            ensure_ascii=False,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        operations=entry["operations"],
                    )
                )
            return cls(schema_version=document["schema_version"], cases=tuple(cases))
        except KeyError as exc:
            raise ValueError("benchmark plan document is missing fields") from exc


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """One caller-measured value for exactly one plan case.

    ``value``, ``unit``, ``environment``, and ``recorded_at`` are all
    caller-supplied provenance; this module never measures anything.
    """

    case_name: str
    value: float
    unit: str
    environment: str
    recorded_at: str

    def __post_init__(self) -> None:
        _strict_name(self.case_name, "case_name")
        if type(self.value) is not float and type(self.value) is not int:
            raise TypeError("value must be a number")
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise ValueError("value must be finite")
        _strict_name(self.unit, "unit")
        if len(self.unit) > 32:
            raise ValueError("unit must hold at most 32 characters")
        _strict_name(self.environment, "environment")
        if type(self.recorded_at) is not str:
            raise TypeError("recorded_at must be a string")
        if _TIMESTAMP.fullmatch(self.recorded_at) is None:
            raise BenchmarkError(
                "contract", "recorded_at must use canonical YYYY-MM-DDTHH:MM:SSZ form"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkResults:
    """Measured bundle joined to a plan by digest.

    Every plan case must carry exactly one result; extra cases are
    rejected. ``release_evidence`` is pinned to ``False`` — promoting
    measurements to release evidence is a human review decision recorded
    outside this module, never an API flag.
    """

    schema_version: str
    plan_fingerprint: str
    release_evidence: bool
    results: tuple[BenchmarkResult, ...]

    def __post_init__(self) -> None:
        if self.release_evidence is not False:
            raise BenchmarkError(
                "contract",
                "release_evidence is reserved for the review process and stays false here",
            )
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple of BenchmarkResult entries")
        names = [result.case_name for result in self.results]
        if names != sorted(set(names)):
            raise ValueError("result case names must be unique and sorted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _RESULT_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "plan_fingerprint": self.plan_fingerprint,
            "release_evidence": self.release_evidence,
            "results": [
                {
                    "case_name": result.case_name,
                    "value": result.value,
                    "unit": result.unit,
                    "environment": result.environment,
                    "recorded_at": result.recorded_at,
                }
                for result in self.results
            ],
        }

    def to_json(self) -> str:
        """Serialize this bundle with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> BenchmarkResults:
        """Validate one canonical JSON document into an exact bundle value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("benchmark results document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("benchmark results document must be a JSON object")
        if (
            document.get("artifact_type") != _RESULT_ARTIFACT_TYPE
            or document.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        ):
            raise ValueError("document is not a benchmark results artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                plan_fingerprint=document["plan_fingerprint"],
                release_evidence=document["release_evidence"],
                results=tuple(
                    BenchmarkResult(
                        case_name=entry["case_name"],
                        value=entry["value"],
                        unit=entry["unit"],
                        environment=entry["environment"],
                        recorded_at=entry["recorded_at"],
                    )
                    for entry in document["results"]
                ),
            )
        except KeyError as exc:
            raise ValueError("benchmark results document is missing fields") from exc


def collect_benchmark_results(
    plan: BenchmarkPlan, results: tuple[BenchmarkResult, ...]
) -> BenchmarkResults:
    """Join caller measurements to a plan, requiring exactly one per case."""

    if type(plan) is not BenchmarkPlan:
        raise TypeError("plan must be a BenchmarkPlan")
    if type(results) is not tuple:
        raise TypeError("results must be a tuple of BenchmarkResult entries")
    for result in results:
        if type(result) is not BenchmarkResult:
            raise TypeError("results entries must be BenchmarkResult values")
    expected = {case.name for case in plan.cases}
    seen = [result.case_name for result in results]
    duplicates = sorted({name for name in seen if seen.count(name) > 1})
    if duplicates:
        raise BenchmarkError("contract", f"duplicate results for cases: {duplicates}")
    missing = sorted(expected - set(seen))
    if missing:
        raise BenchmarkError("contract", f"missing results for cases: {missing}")
    unknown = sorted(set(seen) - expected)
    if unknown:
        raise BenchmarkError("contract", f"unknown case names: {unknown}")
    ordered = tuple(sorted(results, key=lambda result: result.case_name))
    return BenchmarkResults(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        plan_fingerprint=plan.fingerprint,
        release_evidence=False,
        results=ordered,
    )
