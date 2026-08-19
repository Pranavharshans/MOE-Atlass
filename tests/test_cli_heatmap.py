from __future__ import annotations

import ast
import builtins
import json
import os
import socket
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

import moeatlas.analysis as analysis_module
import moeatlas.cli as cli_module
import moeatlas.scan as scan_module
from moeatlas.analysis import RoutingLoadError
from moeatlas.cli import build_parser, main
from moeatlas.scan import ScanOutputError
from moeatlas.store import RoutingShardError, append_mixtral_routing_shard

from .test_mixtral_routing_decoder import _inspection
from .test_runtime_routing_forward import _run


def _command(
    workspace: Path,
    inspection: Path,
    output: Path,
    *,
    run_key: str = "run-1",
    metric: str = "load_ratios",
    inspection_bytes: str = "1000000",
    routing_rows: str = "1000000",
    source_bytes: str = "100000000",
    matrix_cells: str = "100000",
    force: bool = False,
) -> list[str]:
    argv = [
        "heatmap",
        str(workspace),
        "--inspection",
        str(inspection),
        "--run-key",
        run_key,
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


def _real_fixture(tmp_path: Path, layout: str) -> tuple[Path, Path, str]:
    result, _model, inspection = _run(layout, token_count=1)
    workspace = tmp_path / f"workspace-{layout}"
    workspace.mkdir()
    receipt = append_mixtral_routing_shard(workspace, result)
    inspection_path = tmp_path / f"inspection-{layout}.json"
    inspection_path.write_bytes(inspection.to_json().encode())
    return workspace, inspection_path, receipt.run_key


def test_parser_exposes_only_the_bounded_heatmap_contract() -> None:
    parser = build_parser()
    args = parser.parse_args(
        _command(Path("workspace"), Path("inspection.json"), Path("output.html"))
    )
    assert args.command == "heatmap"
    assert args.workspace == "workspace"
    assert args.inspection == "inspection.json"
    assert args.output == "output.html"
    assert args.force is False


def test_heatmap_flags_are_required_and_budgets_have_no_defaults() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    heatmap = subparsers.choices["heatmap"]
    required = {
        "inspection",
        "run_key",
        "metric",
        "max_inspection_bytes",
        "max_routing_rows",
        "max_source_bytes",
        "max_matrix_cells",
        "output",
    }
    actions = {action.dest: action for action in heatmap._actions}
    assert required.issubset(actions)
    assert all(actions[name].required for name in required)
    assert all(actions[name].default is None for name in required if name.startswith("max_"))


def test_heatmap_help_describes_exact_frozen_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["heatmap", "--help"])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    for term in (
        "--inspection",
        "--run-key",
        "--metric",
        "--max-inspection-bytes",
        "--max-routing-rows",
        "--max-source-bytes",
        "--max-matrix-cells",
        "--output",
        "--force",
        "AdapterInspection.to_json()",
        "exact lowercase .html",
        "DuckDB store extra",
        "write_report_atomic()",
    ):
        assert term in help_text


@pytest.mark.parametrize("layout", ["legacy", "packed"])
@pytest.mark.parametrize("metric", ["assignment_counts", "assignment_shares", "load_ratios"])
def test_real_feature19_layouts_and_metrics_publish_one_complete_heatmap(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], layout: str, metric: str
) -> None:
    workspace, inspection, run_key = _real_fixture(tmp_path, layout)
    output = tmp_path / f"{layout}-{metric}.html"

    assert main(_command(workspace, inspection, output, run_key=run_key, metric=metric)) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"saved routing heatmap to {output}\n"
    html = output.read_text(encoding="utf-8")
    assert html.endswith("\n")
    assert html.count("<table") == 1
    assert html.count("<td") == 8
    assert '<meta name="moeatlas-routing-heatmap-schema" content="1.0">' in html
    assert "Layer × Expert routing-load heatmap" in html
    assert run_key in html
    assert "Model key" in html
    assert "Inspection digest" in html
    assert ("legacy_indexed" if layout == "legacy" else "packed") in html
    assert "Routing load only. Selection frequency is association evidence" in html
    assert "token_text" not in html


@pytest.mark.parametrize(
    "field", ["inspection_bytes", "routing_rows", "source_bytes", "matrix_cells"]
)
@pytest.mark.parametrize("bad", ["0", "01", "+1", "-1", "1_0", " 1", "1 ", "1.0", "abc"])
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


@pytest.mark.parametrize("suffix", [".html.bak", ".HTML", ".json", ""])
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


def test_existing_output_requires_force_and_force_publishes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, inspection, run_key = _real_fixture(tmp_path, "legacy")
    output = tmp_path / "output.html"
    output.write_text("keep", encoding="utf-8")
    assert main(_command(workspace, inspection, output, run_key=run_key)) == 2
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert output.read_text(encoding="utf-8") == "keep"
    assert main(_command(workspace, inspection, output, run_key=run_key, force=True)) == 0
    captured = capsys.readouterr()
    assert captured.err == f"saved routing heatmap to {output}\n"
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_determinism_redaction_equivalence_and_workspace_readonly(tmp_path: Path) -> None:
    result, _model, inspection = _run("legacy", token_count=1)
    workspaces: list[Path] = []
    inspection_path = tmp_path / "inspection.json"
    inspection_path.write_text(inspection.to_json(), encoding="utf-8")
    outputs: list[str] = []
    for index, store_token_text in enumerate((False, True)):
        workspace = tmp_path / f"workspace-{index}"
        workspace.mkdir()
        append_mixtral_routing_shard(workspace, result, store_token_text=store_token_text)
        workspaces.append(workspace)
        before = tuple(
            sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*"))
        )
        output = tmp_path / f"output-{index}.html"
        assert main(_command(workspace, inspection_path, output)) == 0
        outputs.append(output.read_text(encoding="utf-8"))
        after = tuple(
            sorted(item.relative_to(workspace).as_posix() for item in workspace.rglob("*"))
        )
        assert after == before
    assert outputs[0].split("<table", 1)[1] == outputs[1].split("<table", 1)[1]

    repeat = tmp_path / "repeat.html"
    assert main(_command(workspaces[0], inspection_path, repeat)) == 0
    first = repeat.read_bytes()
    assert main(_command(workspaces[0], inspection_path, repeat, force=True)) == 0
    assert repeat.read_bytes() == first


def test_cli_publication_races_and_atomic_failures_leave_no_temporary_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, inspection, run_key = _real_fixture(tmp_path, "packed")
    output = tmp_path / "race.html"

    def race(_source: Path, destination: Path) -> None:
        destination.write_text("racer", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr(scan_module.os, "link", race)
    assert main(_command(workspace, inspection, output, run_key=run_key)) == 2
    assert "already exists" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "racer"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))

    output.write_text("original", encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace")

    monkeypatch.setattr(scan_module.os, "replace", fail_replace)
    assert main(_command(workspace, inspection, output, run_key=run_key, force=True)) == 2
    assert "could not write output" in capsys.readouterr().err
    assert output.read_text(encoding="utf-8") == "original"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@pytest.mark.parametrize("field", ["routing_rows", "source_bytes", "matrix_cells"])
def test_inspection_is_bounded_and_non_symlink(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_bytes(b"x" * 100)
    output = tmp_path / "output.html"
    kwargs = {field: "1"}
    assert main(_command(workspace, inspection, output, **kwargs)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert output.exists() is False

    link = tmp_path / "inspection-link.json"
    try:
        link.symlink_to(inspection)
    except OSError as exc:  # pragma: no cover - platform-specific CI fallback
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert main(_command(workspace, link, output, inspection_bytes="1000")) == 2
    captured = capsys.readouterr()
    assert "non-symlink" in captured.err


def test_inspection_stat_is_checked_before_and_after_bounded_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Stat:
        def __init__(self, size: int) -> None:
            self.st_size = size

    class _Stream:
        def __enter__(self) -> _Stream:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _limit: int) -> bytes:
            return b"x"

    class _FakePath:
        def __init__(self, sizes: list[int]) -> None:
            self.sizes = sizes
            self.opens = 0

        def is_symlink(self) -> bool:
            return False

        def is_file(self) -> bool:
            return True

        def stat(self) -> _Stat:
            return _Stat(self.sizes.pop(0))

        def open(self, _mode: str) -> _Stream:
            self.opens += 1
            return _Stream()

    before = _FakePath([2])
    monkeypatch.setattr(cli_module, "Path", lambda _value: before)
    with pytest.raises(cli_module._HeatmapInputError, match="exceeds"):
        cli_module._read_heatmap_inspection("ignored", 1)
    assert before.opens == 0

    after = _FakePath([1, 2])
    monkeypatch.setattr(cli_module, "Path", lambda _value: after)
    with pytest.raises(cli_module._HeatmapInputError, match="exceeds"):
        cli_module._read_heatmap_inspection("ignored", 1)
    assert after.opens == 1


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

    monkeypatch.setattr(cli_module, "_run_heatmap_analysis", forbidden)
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid AdapterInspection" in captured.err
    assert called is False
    assert not output.exists()


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "directory",
        "symlink",
        "nonregular",
        "empty",
        "malformed",
        "schema",
        "extra",
        "manifest",
    ],
)
def test_inspection_file_shape_and_manifest_cases_are_rejected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    valid = _inspection("legacy")
    if case == "missing":
        input_path = inspection
    elif case == "directory":
        inspection.mkdir()
        input_path = inspection
    elif case == "symlink":
        target = tmp_path / "target.json"
        target.write_text(valid.to_json(), encoding="utf-8")
        try:
            inspection.symlink_to(target)
        except OSError as exc:  # pragma: no cover - platform-specific CI fallback
            pytest.skip(f"symlink creation unavailable: {exc}")
        input_path = inspection
    elif case == "nonregular":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO unavailable")
        os.mkfifo(inspection)
        input_path = inspection
    else:
        input_path = inspection
        if case == "empty":
            inspection.write_bytes(b"")
        elif case == "malformed":
            inspection.write_text("{not-json", encoding="utf-8")
        else:
            payload = json.loads(valid.to_json())
            if case == "schema":
                payload["schema_version"] = "9.9"
            elif case == "extra":
                payload["unexpected"] = True
            else:
                payload["manifest_type"] = "not_adapter_inspection"
            inspection.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "output.html"
    assert main(_command(workspace, input_path, output, inspection_bytes="1000000")) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not output.exists()


def test_aggregate_and_render_are_each_called_once_with_identity_and_exact_arguments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    inspection_object = object()
    matrix = object()
    calls: list[tuple[object, ...]] = []

    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: inspection_object)

    def aggregate(
        received_workspace: object, received_inspection: object, **kwargs: object
    ) -> object:
        calls.append((received_workspace, received_inspection, kwargs))
        return matrix

    def render(received_matrix: object, **kwargs: object) -> str:
        calls.append((received_matrix, kwargs))
        return "<!doctype html>\n"

    monkeypatch.setattr(analysis_module, "aggregate_mixtral_routing_load", aggregate)
    monkeypatch.setattr(analysis_module, "render_mixtral_routing_load_heatmap", render)
    argv = _command(
        workspace,
        inspection,
        output,
        metric="assignment_counts",
        routing_rows="23",
        source_bytes="29",
        matrix_cells="31",
    )
    workspace_arg = argv[1]
    assert main(argv) == 0
    capsys.readouterr()
    assert calls == [
        (
            str(workspace),
            inspection_object,
            {
                "run_key": "run-1",
                "max_routing_rows": 23,
                "max_source_bytes": 29,
                "max_matrix_cells": 31,
            },
        ),
        (matrix, {"metric": "assignment_counts", "max_cells": 31}),
    ]
    assert type(calls[0][0]) is str
    assert calls[0][0] is workspace_arg
    assert output.read_text(encoding="utf-8") == "<!doctype html>\n"


def test_analysis_failure_is_generic_and_never_publishes_or_echoes_details(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inspection = tmp_path / "inspection.json"
    inspection.write_text("{}", encoding="utf-8")
    output = tmp_path / "output.html"
    monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())

    def fail(*_args: object, **_kwargs: object) -> object:
        raise ValueError("SECRET_DETAIL")

    monkeypatch.setattr(analysis_module, "aggregate_mixtral_routing_load", fail)
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas heatmap: heatmap generation failed\n"
    assert "SECRET_DETAIL" not in captured.err
    assert not output.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RoutingLoadError("source"), "mixtral routing load aggregation failed at source"),
        (RoutingShardError("reopen"), "routing shard failed at reopen"),
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
        "_run_heatmap_analysis",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    assert main(_command(workspace, inspection, output)) == 2
    captured = capsys.readouterr()
    assert captured.err == f"moeatlas heatmap: {message}\n"
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

    monkeypatch.setattr(analysis_module, "aggregate_mixtral_routing_load", fail)
    with pytest.raises(error_type):
        main(_command(workspace, inspection, output))
    assert not output.exists()


@pytest.mark.parametrize("phase", ["inspection", "aggregate", "render", "publication"])
@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_control_flow_exceptions_are_exact_at_every_heatmap_phase(
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
            "aggregate_mixtral_routing_load",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("aggregate")),
        )
    elif phase == "render":
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(
            analysis_module, "aggregate_mixtral_routing_load", lambda *_a, **_k: object()
        )
        monkeypatch.setattr(
            analysis_module,
            "render_mixtral_routing_load_heatmap",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(error_type("render")),
        )
    else:
        monkeypatch.setattr(cli_module, "_read_heatmap_inspection", lambda *_: object())
        monkeypatch.setattr(cli_module, "_run_heatmap_analysis", lambda *_a, **_k: "HTML\n")
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
    monkeypatch.setattr(cli_module, "_run_heatmap_analysis", lambda *args, **kwargs: "HTML\n")
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


def test_cli_actual_path_does_not_touch_network_cache_or_browser(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, inspection, run_key = _real_fixture(tmp_path, "legacy")
    output = tmp_path / "output.html"
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setenv("HF_HOME", str(cache))
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(cache))

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("heatmap CLI must not use network or browser")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    before = tuple(sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*")))
    assert main(_command(workspace, inspection, output, run_key=run_key)) == 0
    capsys.readouterr()
    after = tuple(sorted(item.relative_to(tmp_path).as_posix() for item in tmp_path.rglob("*")))
    assert set(after) == set(before) | {"output.html"}
    assert not list(tmp_path.glob("**/*.tmp"))


def test_missing_duckdb_is_a_safe_dependency_stage_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, inspection, run_key = _real_fixture(tmp_path, "legacy")
    output = tmp_path / "output.html"
    original_import = builtins.__import__

    def blocked_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise ModuleNotFoundError("duckdb intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    assert main(_command(workspace, inspection, output, run_key=run_key)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas heatmap: routing shard failed at dependency\n"
    assert not output.exists()


@pytest.mark.parametrize("command", [["--version"], ["doctor"], ["scan", "fixture:synthetic"]])
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


def test_cli_source_keeps_heatmap_boundary_model_free_and_non_networked() -> None:
    source = Path("src/moeatlas/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "duckdb",
        "webbrowser",
        "socket",
        "urllib",
        "requests",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".", 1)[0] not in forbidden for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".", 1)[0] not in forbidden
    lowered = source.lower()
    for term in ("webbrowser", "urlopen", "create_connection", "torch", "transformers"):
        assert term not in lowered
    assert "write_report_atomic" in source
    assert "from .analysis import aggregate_mixtral_routing_load" in source
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "aggregate_mixtral_routing_load"
            for node in calls
        )
        == 1
    )
    assert (
        sum(
            isinstance(node.func, ast.Name)
            and node.func.id == "render_mixtral_routing_load_heatmap"
            for node in calls
        )
        == 1
    )
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "write_report_atomic"
            for node in calls
        )
        == 2
    )  # existing scan plus the single heatmap delegation
    forbidden_attrs = {
        "mkdir",
        "glob",
        "rglob",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "touch",
    }
    for node in calls:
        if isinstance(node.func, ast.Name):
            assert node.func.id != "open"
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr == "replace" and isinstance(node.func.value, ast.Name):
                assert node.func.value.id == "field_name"
                continue
            assert node.func.attr not in forbidden_attrs
            if node.func.attr == "open":
                assert node.args and isinstance(node.args[0], ast.Constant)
                assert node.args[0].value == "rb"
    assert "import duckdb" not in lowered
    assert "read_parquet" not in lowered
    assert "select " not in lowered


def test_existing_command_version_remains_available(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
    assert "moeatlas" in capsys.readouterr().out
