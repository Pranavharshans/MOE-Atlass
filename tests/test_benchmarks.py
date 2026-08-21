"""Contract tests for reproducible benchmark plans and result bundles."""

from __future__ import annotations

import pytest

from moeatlas.benchmarks import (
    BENCHMARK_SCHEMA_VERSION,
    BenchmarkCase,
    BenchmarkError,
    BenchmarkPlan,
    BenchmarkResult,
    BenchmarkResults,
    collect_benchmark_results,
)


def _case(name: str = "a/append") -> BenchmarkCase:
    return BenchmarkCase(
        name=name,
        workload='{"rows":100,"seed":7}',
        operations=100,
    )


def _result(case_name: str = "a/append", value: float = 1.5) -> BenchmarkResult:
    return BenchmarkResult(
        case_name=case_name,
        value=value,
        unit="s",
        environment="ci-runner-ubuntu-24.04",
        recorded_at="2026-08-21T00:00:00Z",
    )


def test_surface_is_pinned() -> None:
    assert BENCHMARK_SCHEMA_VERSION == "1.0"
    assert BenchmarkError("contract").stage == "contract"
    with pytest.raises(ValueError, match="stage is not supported"):
        BenchmarkError("unknown")


def test_case_requires_canonical_json_workload() -> None:
    case = _case()
    assert case.operations == 100
    with pytest.raises(ValueError, match="must be valid JSON"):
        BenchmarkCase(name="a", workload="{not json", operations=1)
    # Valid JSON but not canonical (key order and whitespace differ).
    with pytest.raises(ValueError, match="must be canonical JSON"):
        BenchmarkCase(name="a", workload='{"seed": 7, "rows": 100}', operations=1)
    with pytest.raises(TypeError):
        BenchmarkCase(name="a", workload=b'{"a":1}', operations=1)  # type: ignore[arg-type]
    with pytest.raises(BenchmarkError, match="strictly positive"):
        BenchmarkCase(name="a", workload="{}", operations=0)


def test_plan_orders_cases_and_rejects_duplicates() -> None:
    plan = BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case(),))
    assert plan.fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="unique and sorted"):
        BenchmarkPlan(
            schema_version=BENCHMARK_SCHEMA_VERSION,
            cases=(_case("b"), _case("a")),
        )
    with pytest.raises(BenchmarkError, match="must not be empty"):
        BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=())
    with pytest.raises(TypeError):
        BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=[_case()])  # type: ignore[arg-type]


def test_plan_digest_is_content_addressed() -> None:
    first = BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case(),))
    second = BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case(),))
    changed = BenchmarkPlan(
        schema_version=BENCHMARK_SCHEMA_VERSION,
        cases=(BenchmarkCase(name="a/append", workload='{"rows":101,"seed":7}', operations=100),),
    )
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint


def test_result_provenance_is_caller_supplied_and_validated() -> None:
    result = _result()
    assert result.value == 1.5
    with pytest.raises(ValueError, match="must be finite"):
        BenchmarkResult(
            case_name="a",
            value=float("nan"),
            unit="s",
            environment="ci",
            recorded_at="2026-08-21T00:00:00Z",
        )
    with pytest.raises(BenchmarkError, match="canonical YYYY-MM-DDTHH:MM:SSZ form"):
        BenchmarkResult(
            case_name="a",
            value=1.0,
            unit="s",
            environment="ci",
            recorded_at="yesterday",
        )
    with pytest.raises(ValueError, match="unit must hold at most 32 characters"):
        BenchmarkResult(
            case_name="a",
            value=1.0,
            unit="x" * 33,
            environment="ci",
            recorded_at="2026-08-21T00:00:00Z",
        )


def test_collect_requires_exactly_one_result_per_case() -> None:
    plan = BenchmarkPlan(
        schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case("a/x"), _case("b/y"))
    )
    bundle = collect_benchmark_results(plan, (_result("b/y"), _result("a/x")))
    assert [r.case_name for r in bundle.results] == ["a/x", "b/y"]
    assert bundle.plan_fingerprint == plan.fingerprint
    assert bundle.release_evidence is False

    with pytest.raises(BenchmarkError, match="missing results"):
        collect_benchmark_results(plan, (_result("a/x"),))
    with pytest.raises(BenchmarkError, match="duplicate results"):
        collect_benchmark_results(plan, (_result("a/x"), _result("a/x")))
    extra_plan = BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case("a/x"),))
    with pytest.raises(BenchmarkError, match="unknown case names"):
        collect_benchmark_results(extra_plan, (_result("a/x"), _result("b/y")))
    with pytest.raises(BenchmarkError, match="stays false here"):
        BenchmarkResults(
            schema_version=BENCHMARK_SCHEMA_VERSION,
            plan_fingerprint=plan.fingerprint,
            release_evidence=True,
            results=(_result("a/x"),),
        )


def test_canonical_round_trip_and_rejection() -> None:
    plan = BenchmarkPlan(schema_version=BENCHMARK_SCHEMA_VERSION, cases=(_case(),))
    document = plan.to_json()
    assert '"artifact_type":"moeatlas.benchmark_plan"' in document
    restored = BenchmarkPlan.from_json(document)
    assert restored == plan
    assert restored.fingerprint == plan.fingerprint

    bundle = collect_benchmark_results(plan, (_result(),))
    bundle_document = bundle.to_json()
    assert '"artifact_type":"moeatlas.benchmark_results"' in bundle_document
    assert '"release_evidence":false' in bundle_document
    assert BenchmarkResults.from_json(bundle_document) == bundle

    with pytest.raises(ValueError, match="not a benchmark plan artifact"):
        BenchmarkPlan.from_json('{"schema_version":"1.0"}')
    with pytest.raises(ValueError, match="not a benchmark results artifact"):
        BenchmarkResults.from_json('{"schema_version":"1.0"}')
    with pytest.raises(ValueError, match="not valid JSON"):
        BenchmarkPlan.from_json("{oops")
