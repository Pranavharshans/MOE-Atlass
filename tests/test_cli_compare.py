from __future__ import annotations

import ast
import builtins
import re
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import moeatlas.analysis as analysis_module
import moeatlas.cli as cli_module
from moeatlas.analysis import RoutingLoadError
from moeatlas.cli import build_parser, main
from moeatlas.events import RoutingEvent, TokenEvent
from moeatlas.runtime import MixtralRoutingForwardResult
from moeatlas.scan import ScanOutputError
from moeatlas.store import RoutingShardError, append_mixtral_routing_shard

from .test_runtime_routing_forward import _ForwardModel, _run


def _command(
    workspace: Path,
    inspection: Path,
    output: Path,
    *,
    baseline: str = "run-1",
    comparison: str = "run-2",
    metric: str = "count_deltas",
    inspection_bytes: str = "1000000",
    routing_rows: str = "1000000",
    source_bytes: str = "100000000",
    matrix_cells: str = "100000",
    force: bool = False,
) -> list[str]:
    argv = [
        "compare",
        str(workspace),
        "--inspection",
        str(inspection),
        "--baseline-run-key",
        baseline,
        "--comparison-run-key",
        comparison,
        "--metric",
        metric,
        "--max-inspection-bytes",
        inspection_bytes,
        "--max-routing-rows",
        routing_rows,
        "--max-source-bytes",
        source_bytes,
        "--max-matrix-cells",
        matrix_cells,
        "--output",
        str(output),
    ]
    if force:
        argv.append("--force")
    return argv


def _retagged(result: MixtralRoutingForwardResult, *, run_key: str) -> MixtralRoutingForwardResult:
    tokens: list[TokenEvent] = []
    remapped: dict[str, str] = {}
    for index, event in enumerate(result.token_events):
        payload = event.model_dump(mode="json")
        payload["run_key"] = run_key
        payload["token_id"] = event.token_id + index + 1
        payload.pop("token_key", None)
        retokened = TokenEvent.model_validate(payload)
        remapped[event.token_key] = retokened.token_key
        tokens.append(retokened)
    routes = tuple(
        RoutingEvent.model_validate(
            {**route.model_dump(mode="json"), "token_key": remapped[route.token_key]}
        )
        for route in result.routing_events
    )
    return MixtralRoutingForwardResult(result.output, tuple(tokens), routes)


def _two_run_fixture(tmp_path: Path, layout: str = "legacy") -> tuple[Path, Path, str, str]:
    baseline_result, _model, inspection = _run(layout, token_count=1)
    comparison_source, _, _ = _run(
        layout, token_count=1, model=_ForwardModel(layout, rows=[[3.0, 0.0, 2.0, 1.0]])
    )
    workspace = tmp_path / f"workspace-{layout}"
    workspace.mkdir()
    append_mixtral_routing_shard(workspace, baseline_result)
    append_mixtral_routing_shard(workspace, _retagged(comparison_source, run_key="run-2"))
    inspection_path = tmp_path / f"inspection-{layout}.json"
    inspection_path.write_bytes(inspection.to_json().encode())
    return workspace, inspection_path, "run-1", "run-2"


def test_parser_exposes_only_the_bounded_compare_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        _command(Path("workspace"), Path("inspection.json"), Path("output.html"))
    )
    assert args.command == "compare"
    assert args.workspace == "workspace"
    assert args.inspection == "inspection.json"
    assert args.baseline_run_key == "run-1"
    assert args.comparison_run_key == "run-2"
    assert args.metric == "count_deltas"
    assert args.output == "output.html"
    assert args.force is False


def test_compare_flags_are_required_and_budgets_have_no_defaults() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    compare = subparsers.choices["compare"]
    required = {
        "inspection",
        "baseline_run_key",
        "comparison_run_key",
        "metric",
        "max_inspection_bytes",
        "max_routing_rows",
        "max_source_bytes",
        "max_matrix_cells",
        "output",
    }
    actions = {action.dest: action for action in compare._actions}
    assert required.issubset(actions)
    assert all(actions[name].required for name in required)
    assert all(actions[name].default is None for name in required if name.startswith("max_"))


def test_compare_help_describes_exact_frozen_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["compare", "--help"])
    assert caught.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for term in (
        "--inspection",
        "--baseline-run-key",
        "--comparison-run-key",
        "--metric",
        "--max-inspection-bytes",
        "--max-routing-rows",
        "--max-source-bytes",
        "--max-matrix-cells",
        "--output",
        "--force",
        "AdapterInspection.to_json()",
        "the two run keys must differ",
        "exact lowercase .html",
        "DuckDB store extra",
        "write_report_atomic()",
        "no model, browser, network, cache, or generation path",
    ):
        assert term in help_text


@pytest.mark.parametrize("metric", ["count_deltas", "share_deltas", "ratio_deltas"])
def test_real_two_run_workspace_publishes_one_complete_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], metric: str
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / f"{metric}.html"
    argv = _command(
        workspace, inspection, output, baseline=baseline, comparison=comparison, metric=metric
    )

    assert main(argv) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"saved routing comparison to {output}\n"
    html = output.read_text(encoding="utf-8")
    assert html.endswith("\n")
    assert html.count("<table") == 1
    assert '<meta name="moeatlas-routing-compare-heatmap-schema" content="1.0">' in html
    assert "Layer × Expert routing-load comparison" in html
    assert f"{baseline} vs {comparison}" in html
    assert "Baseline run key" in html
    assert "Comparison run key" in html
    assert "Model key" in html
    assert "Inspection digest" in html
    assert "legacy_indexed" in html
    assert (
        "Routing-load deltas only. Differences in selection frequency are association "
        "evidence, not expert specialization or causal effect."
    ) in html
    assert 'id="delta-formula"' in html
    assert metric in html
    assert re.search(r'<td class="(heat|cold)-\d"', html)
    assert "token_text" not in html


def test_packed_layout_publishes_one_complete_comparison(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path, "packed")
    output = tmp_path / "packed.html"
    argv = _command(workspace, inspection, output, baseline=baseline, comparison=comparison)

    assert main(argv) == 0
    capsys.readouterr()
    html = output.read_text(encoding="utf-8")
    assert "packed" in html
    assert re.search(r'<td class="(heat|cold)-\d"', html)


def test_comparison_is_deterministic_and_workspace_readonly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"

    def snapshot() -> tuple[str, ...]:
        return tuple(sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*")))

    argv = _command(workspace, inspection, output, baseline=baseline, comparison=comparison)
    before = snapshot()
    assert main(argv) == 0
    capsys.readouterr()
    after = snapshot()
    assert set(after) == set(before) | {"output.html"}
    first = output.read_bytes()
    assert (
        main(
            _command(
                workspace, inspection, output, baseline=baseline, comparison=comparison, force=True
            )
        )
        == 0
    )
    capsys.readouterr()
    assert output.read_bytes() == first
    assert snapshot() == after


def test_equal_run_keys_are_rejected_before_budgets_and_output_preflight(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "missing" / "output.html"
    argv = _command(
        workspace, inspection, output, baseline="run-1", comparison="run-1", inspection_bytes="0"
    )

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: baseline and comparison run keys must differ\n"
    assert not output.parent.exists()

    plain_output = tmp_path / "output.html"
    assert main(_command(workspace, inspection, plain_output, baseline="r", comparison="r")) == 2
    captured = capsys.readouterr()
    assert captured.err == "moeatlas compare: baseline and comparison run keys must differ\n"
    assert not plain_output.exists()


@pytest.mark.parametrize(
    "field", ["inspection_bytes", "routing_rows", "source_bytes", "matrix_cells"]
)
@pytest.mark.parametrize("bad", ["0", "-1", "1.5", "01", "+5", ""])
def test_required_budgets_are_canonical_positive_decimals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, bad: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    args = _command(workspace, inspection, output, **{field: bad})

    assert main(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "canonical positive decimal integer" in captured.err
    assert not output.exists()


@pytest.mark.parametrize("suffix", [".htm", ".HTML", ".json", ""])
def test_output_requires_exact_lowercase_html_suffix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], suffix: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / f"output{suffix}"
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "exact .html suffix" in captured.err
    assert not output.exists()


def test_existing_output_requires_force_and_force_publishes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"
    output.write_text("keep", encoding="utf-8")
    argv = _command(workspace, inspection, output, baseline=baseline, comparison=comparison)

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert output.read_text(encoding="utf-8") == "keep"

    forced = _command(
        workspace, inspection, output, baseline=baseline, comparison=comparison, force=True
    )
    assert main(forced) == 0
    captured = capsys.readouterr()
    assert captured.err == f"saved routing comparison to {output}\n"
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_output_directory_and_nondirectory_parent_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    directory = tmp_path / "output.html"
    directory.mkdir()
    assert main(_command(workspace, inspection, directory)) == 2
    assert "directory" in capsys.readouterr().err

    parent = tmp_path / "not-a-directory"
    parent.write_text("file", encoding="utf-8")
    output = parent / "output.html"
    assert main(_command(workspace, inspection, output)) == 2
    assert "parent is not a directory" in capsys.readouterr().err


def test_missing_output_parent_is_rejected_without_creating_anything(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "missing" / "output.html"
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("inspection must not be read")

    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", forbidden)
    assert main(_command(tmp_path / "workspace", inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "parent does not exist" in captured.err
    assert called is False
    assert not output.parent.exists()


def test_output_preflight_happens_before_inspection_workspace_or_optional_dependency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.html"
    output.write_text("keep", encoding="utf-8")
    inspection = tmp_path / "inspection.json"
    inspection.write_text("not read", encoding="utf-8")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("inspection/dependency path was traversed")

    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", forbidden)
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise AssertionError("duckdb must remain lazy after output preflight")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert main(_command(tmp_path / "missing-workspace", inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "already exists" in captured.err
    assert called is False
    assert output.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("case", "fragment"),
    [
        ("missing", "regular non-symlink file"),
        ("symlink", "regular non-symlink file"),
        ("oversized", "exceeds --max-inspection-bytes"),
        ("malformed", "valid AdapterInspection"),
    ],
)
def test_inspection_problems_are_rejected_with_fixed_messages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, fragment: str
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"
    target = tmp_path / "input.json"
    if case == "symlink":
        try:
            target.symlink_to(inspection)
        except OSError as exc:  # pragma: no cover - platform-specific CI fallback
            pytest.skip(f"symlink creation unavailable: {exc}")
    elif case == "oversized":
        target.write_bytes(inspection.read_bytes())
    elif case == "malformed":
        target.write_text("{not-json", encoding="utf-8")
    kwargs = {"inspection_bytes": "10"} if case == "oversized" else {}
    argv = _command(workspace, target, output, baseline=baseline, comparison=comparison, **kwargs)

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert fragment in captured.err
    assert not output.exists()


def test_invalid_inspection_is_rejected_without_analysis_or_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("analysis must not run for invalid inspection")

    monkeypatch.setattr(cli_module, "_run_compare_analysis", forbidden)
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid AdapterInspection" in captured.err
    assert called is False
    assert not output.exists()


@pytest.mark.parametrize("missing", ["baseline", "comparison", "both"])
def test_missing_or_uncommitted_run_keys_fail_at_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"
    requested_baseline = "run-absent-a" if missing in {"baseline", "both"} else baseline
    requested_comparison = "run-absent-b" if missing in {"comparison", "both"} else comparison
    argv = _command(
        workspace, inspection, output, baseline=requested_baseline, comparison=requested_comparison
    )

    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: routing load aggregation failed at source\n"
    assert not output.exists()


def test_empty_workspace_fails_at_source(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "empty-workspace"
    workspace.mkdir()
    _result, _model, inspection_object = _run("legacy", token_count=1)
    inspection = tmp_path / "inspection.json"
    inspection.write_bytes(inspection_object.to_json().encode())
    output = tmp_path / "output.html"

    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: routing load aggregation failed at source\n"
    assert not output.exists()


def test_unknown_metric_is_rejected_by_argparse(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(
            _command(
                Path("workspace"),
                Path("inspection.json"),
                Path("output.html"),
                metric="count_delta",
            )
        )
    assert caught.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_incomparable_universes_fail_with_generic_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline_result, _model, inspection = _run("legacy", token_count=1)
    comparison_source, _, _ = _run("legacy", token_count=2)
    workspace = tmp_path / "workspace-incomparable"
    workspace.mkdir()
    append_mixtral_routing_shard(workspace, baseline_result)
    append_mixtral_routing_shard(workspace, _retagged(comparison_source, run_key="run-2"))
    inspection_path = tmp_path / "inspection.json"
    inspection_path.write_bytes(inspection.to_json().encode())
    output = tmp_path / "output.html"

    assert main(_command(workspace, inspection_path, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: routing comparison failed\n"
    assert not output.exists()


def test_aggregate_compare_render_are_called_once_with_identity_and_exact_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    inspection_object = object()
    matrix = object()
    comparison_value = object()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: inspection_object)

    def aggregate(
        received_workspace: object, received_inspection: object, **kwargs: object
    ) -> object:
        calls.append((received_workspace, received_inspection, kwargs))
        return matrix

    def compare(received_baseline: object, received_comparison: object, **kwargs: object) -> object:
        calls.append((received_baseline, received_comparison, kwargs))
        return comparison_value

    def render(received_comparison: object, **kwargs: object) -> str:
        calls.append((received_comparison, kwargs))
        return "<!doctype html>\n"

    monkeypatch.setattr(analysis_module, "aggregate_routing_load", aggregate)
    monkeypatch.setattr(analysis_module, "compare_routing_load", compare)
    monkeypatch.setattr(analysis_module, "render_routing_load_comparison", render)
    argv = _command(
        workspace,
        inspection,
        output,
        baseline="run-a",
        comparison="run-b",
        metric="share_deltas",
        routing_rows="23",
        source_bytes="29",
        matrix_cells="31",
    )
    workspace_arg = argv[1]
    assert main(argv) == 0
    capsys.readouterr()
    expected_budgets = {
        "max_routing_rows": 23,
        "max_source_bytes": 29,
        "max_matrix_cells": 31,
    }
    assert calls == [
        (str(workspace), inspection_object, {"run_key": "run-a"} | expected_budgets),
        (str(workspace), inspection_object, {"run_key": "run-b"} | expected_budgets),
        (matrix, matrix, {"max_cells": 31}),
        (comparison_value, {"metric": "share_deltas", "max_cells": 31}),
    ]
    assert type(calls[0][0]) is str
    assert calls[0][0] is workspace_arg
    assert output.read_text(encoding="utf-8") == "<!doctype html>\n"


def test_generic_failure_never_echoes_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
    monkeypatch.setattr(analysis_module, "aggregate_routing_load", lambda *_a, **_k: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise ValueError("SECRET_DETAIL")

    monkeypatch.setattr(analysis_module, "compare_routing_load", fail)
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: routing comparison failed\n"
    assert "SECRET_DETAIL" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RoutingLoadError("source"), "routing load aggregation failed at source"),
        (RoutingLoadError("query"), "routing load aggregation failed at query"),
        (RoutingShardError("reopen"), "routing shard failed at reopen"),
        (ValueError("SECRET"), "routing comparison failed"),
    ],
)
def test_known_analysis_store_errors_keep_only_exact_fixed_stage_messages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    message: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
    monkeypatch.setattr(
        cli_module,
        "_run_compare_analysis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.err == f"moeatlas compare: {message}\n"
    assert not output.exists()


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_propagate_without_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error_type: type[BaseException]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise error_type("control-flow")

    monkeypatch.setattr(analysis_module, "aggregate_routing_load", fail)
    with pytest.raises(error_type):
        main(_command(workspace, inspection, output))
    assert not output.exists()


@pytest.mark.parametrize("phase", ["inspection", "aggregate", "compare", "render", "publication"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_exact_at_every_compare_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    error_type: type[BaseException],
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    if phase == "inspection":
        monkeypatch.setattr(
            cli_module,
            "_read_heatmap_inspection",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("inspection")),
        )
    elif phase == "aggregate":
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(
            analysis_module,
            "aggregate_routing_load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("aggregate")),
        )
    elif phase == "compare":
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(analysis_module, "aggregate_routing_load", lambda *_a, **_k: object())
        monkeypatch.setattr(
            analysis_module,
            "compare_routing_load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("compare")),
        )
    elif phase == "render":
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(analysis_module, "aggregate_routing_load", lambda *_a, **_k: object())
        monkeypatch.setattr(analysis_module, "compare_routing_load", lambda *_a, **_k: object())
        monkeypatch.setattr(
            analysis_module,
            "render_routing_load_comparison",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("render")),
        )
    else:
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(cli_module, "_run_compare_analysis", lambda *_a, **_k: "HTML\n")
        monkeypatch.setattr(
            cli_module,
            "write_report_atomic",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("publication")),
        )
    with pytest.raises(error_type) as caught:
        main(_command(workspace, inspection, output))
    assert str(caught.value) == phase
    assert not output.exists()


def test_writer_is_reused_once_and_force_is_forwarded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
    monkeypatch.setattr(cli_module, "_run_compare_analysis", lambda *args, **kwargs: "HTML\n")
    calls: list[tuple[object, ...]] = []

    def writer(payload: str, path: Path, *, force: bool = False) -> Path:
        calls.append((payload, path, force))
        return path

    monkeypatch.setattr(cli_module, "write_report_atomic", writer)
    assert main(_command(workspace, inspection, output, force=True)) == 0
    capsys.readouterr()
    assert calls == [("HTML\n", output, True)]

    monkeypatch.setattr(
        cli_module,
        "write_report_atomic",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ScanOutputError("writer failed")),
    )
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert "writer failed" in captured.err


def test_missing_duckdb_is_a_safe_dependency_stage_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise ModuleNotFoundError("duckdb intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    argv = _command(workspace, inspection, output, baseline=baseline, comparison=comparison)
    assert main(argv) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas compare: routing shard failed at dependency\n"
    assert not output.exists()


def test_cli_actual_path_does_not_touch_network_cache_or_browser(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, inspection, baseline, comparison = _two_run_fixture(tmp_path)
    output = tmp_path / "output.html"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("compare CLI must not use network or browser")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)

    def snapshot() -> tuple[str, ...]:
        return tuple(sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*")))

    before = snapshot()
    argv = _command(workspace, inspection, output, baseline=baseline, comparison=comparison)
    assert main(argv) == 0
    capsys.readouterr()
    after = snapshot()
    assert set(after) == set(before) | {"output.html"}
    assert not list(tmp_path.glob("**/*.tmp"))


@pytest.mark.parametrize("command", [["compare", "--help"], ["doctor"]])
def test_existing_cli_commands_remain_lazy_without_analysis_import(command: list[str]) -> None:
    script = (
        "import sys; import moeatlas.cli as cli; "
        "assert not any(name.startswith('moeatlas.analysis') for name in sys.modules); "
        f"cli.main({command!r})"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0
    assert "Traceback" not in completed.stderr


def test_cli_source_keeps_compare_boundary_model_free_and_non_networked() -> None:
    source = Path("src/moeatlas/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    lowered = source.lower()
    for term in ("webbrowser", "urlopen", "create_connection", "torch", "transformers"):
        assert term not in lowered
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "compare_routing_load"
            for node in calls
        )
        == 1
    )
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "render_routing_load_comparison"
            for node in calls
        )
        == 1
    )
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "write_report_atomic"
            for node in calls
        )
        == 3
    )  # scan plus the heatmap and compare delegations
    assert "baseline and comparison run keys must differ" in source
