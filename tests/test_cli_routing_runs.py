from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import pytest

import moeatlas.cli as cli_module
from moeatlas.cli import build_parser, main
from moeatlas.scan import ScanOutputError
from moeatlas.store import (
    STORE_SCHEMA_VERSION,
    MixtralRoutingRunInventory,
    append_mixtral_routing_shard,
)

from .test_runtime_routing_forward import _run


def _args(
    workspace: Path, *, output: Path | None = None, force: bool = False, **overrides: str
) -> list[str]:
    values = {
        "max_runs": "10",
        "max_shards": "10",
        "max_event_rows": "10000",
        "max_source_bytes": "10000000",
    }
    values.update(overrides)
    argv = [
        "routing-runs",
        str(workspace),
        "--max-runs",
        values["max_runs"],
        "--max-shards",
        values["max_shards"],
        "--max-event-rows",
        values["max_event_rows"],
        "--max-source-bytes",
        values["max_source_bytes"],
    ]
    if output is not None:
        argv.extend(["--output", str(output)])
    if force:
        argv.append("--force")
    return argv


def _fixture(tmp_path: Path, layout: str = "legacy") -> tuple[Path, str]:
    result, _model, _inspection = _run(layout, token_count=1)
    workspace = tmp_path / f"workspace-{layout}"
    workspace.mkdir()
    receipt = append_mixtral_routing_shard(workspace, result)
    return workspace, receipt.run_key


def test_parser_requires_all_budgets_and_has_no_defaults() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    command = subparsers.choices["routing-runs"]
    actions = {action.dest: action for action in command._actions}
    assert {"max_runs", "max_shards", "max_event_rows", "max_source_bytes"}.issubset(actions)
    assert all(
        actions[name].required
        for name in ("max_runs", "max_shards", "max_event_rows", "max_source_bytes")
    )
    assert all(
        actions[name].default is None
        for name in ("max_runs", "max_shards", "max_event_rows", "max_source_bytes")
    )


@pytest.mark.parametrize("field", ["max_runs", "max_shards", "max_event_rows", "max_source_bytes"])
@pytest.mark.parametrize("bad", ["0", "01", "+1", "-1", "1_0", "1.0", " 1", "1 "])
def test_cli_budgets_are_canonical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], field: str, bad: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert main(_args(workspace, **{field: bad})) == 2
    assert "canonical positive decimal integer" in capsys.readouterr().err


def test_empty_inventory_is_stdout_json_plus_one_newline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert main(_args(workspace)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.endswith("\n") and not captured.out.endswith("\n\n")
    assert json.loads(captured.out)["manifest_type"] == "mixtral_routing_run_inventory"


def test_real_legacy_inventory_is_stdout_and_workspace_str_is_unchanged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace_path, run_key = _fixture(tmp_path, "legacy")
    workspace = str(workspace_path)
    calls: list[tuple[object, dict[str, int]]] = []
    original = cli_module._run_routing_run_inventory

    def wrapped(value: str, **kwargs: int):
        calls.append((value, kwargs))
        return original(value, **kwargs)

    monkeypatch.setattr(cli_module, "_run_routing_run_inventory", wrapped)
    assert main(_args(workspace)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["runs"][0]["run_key"] == run_key
    assert len(calls) == 1
    assert calls[0][0] is workspace
    assert type(calls[0][0]) is str
    assert calls[0][1] == {
        "max_runs": 10,
        "max_shards": 10,
        "max_event_rows": 10000,
        "max_source_bytes": 10000000,
    }


def test_empty_inventory_keeps_duckdb_lazy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    original_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise AssertionError("empty inventory must not import duckdb")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert main(_args(workspace)) == 0
    assert capsys.readouterr().err == ""


def test_real_inventory_file_publication_is_atomic_and_deterministic(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace, run_key = _fixture(tmp_path, "packed")
    output = tmp_path / "inventory.json"
    args = _args(workspace, output=output)
    assert main(args) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"saved routing run inventory to {output}\n"
    first = output.read_bytes()
    document = json.loads(first)
    assert document["runs"][0]["run_key"] == run_key
    assert first.endswith(b"\n")
    assert main(_args(workspace, output=output)) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(_args(workspace, output=output, force=True)) == 0
    capsys.readouterr()
    assert output.read_bytes() == first
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_output_symlink_and_writer_once_exact_payload_and_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _run_key = _fixture(tmp_path, "packed")
    target = tmp_path / "target.json"
    target.write_text("old")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    assert main(_args(workspace, output=link)) == 2
    assert "already exists" in capsys.readouterr().err

    calls: list[tuple[str, Path, bool]] = []

    def writer(payload: str, output: Path, *, force: bool) -> Path:
        calls.append((payload, output, force))
        return output

    monkeypatch.setattr(cli_module, "_write_routing_runs_report", writer)
    output = tmp_path / "writer.json"
    assert main(_args(workspace, output=output, force=True)) == 0
    captured = capsys.readouterr()
    assert len(calls) == 1
    assert calls[0][0].endswith("\n")
    assert json.loads(calls[0][0])["manifest_type"] == "mixtral_routing_run_inventory"
    assert calls[0][1] == output
    assert calls[0][2] is True
    assert captured.err == f"saved routing run inventory to {output}\n"


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(9)])
def test_inventory_serialization_and_writer_control_flow_are_exact(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    inventory = MixtralRoutingRunInventory(
        "1.0",
        "mixtral_routing_run_inventory",
        STORE_SCHEMA_VERSION,
        "1.0",
        0,
        0,
        0,
        0,
        0,
        (),
    )

    class SerializationFailure:
        def to_json(self):
            raise failure

    monkeypatch.setattr(
        cli_module, "_run_routing_run_inventory", lambda *_args, **_kwargs: SerializationFailure()
    )
    with pytest.raises(type(failure)) as caught:
        main(_args(workspace))
    assert caught.value is failure
    assert capsys.readouterr().out == ""
    assert inventory.to_json().endswith("}")

    monkeypatch.setattr(
        cli_module, "_run_routing_run_inventory", lambda *_args, **_kwargs: inventory
    )
    monkeypatch.setattr(
        cli_module,
        "_write_routing_runs_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    output = tmp_path / "control.json"
    with pytest.raises(type(failure)) as caught:
        main(_args(workspace, output=output))
    assert caught.value is failure


def test_unexpected_error_is_fixed_and_redacts_path_and_secret(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(
        cli_module,
        "_run_routing_run_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("/secret/token/path")),
    )
    assert main(_args(workspace)) == 2
    captured = capsys.readouterr()
    assert captured.err == "moeatlas routing-runs: routing run inventory failed\n"
    assert "/secret" not in captured.err


def test_writer_failure_uses_existing_safe_output_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _run_key = _fixture(tmp_path, "legacy")
    monkeypatch.setattr(
        cli_module,
        "_write_routing_runs_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ScanOutputError("writer failed")),
    )
    output = tmp_path / "failure.json"
    assert main(_args(workspace, output=output)) == 2
    assert capsys.readouterr().err == "moeatlas routing-runs: writer failed\n"


def test_routing_runs_cli_ast_has_no_model_network_cache_or_alternate_paths() -> None:
    source = Path("src/moeatlas/cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {"_handle_routing_runs", "_run_routing_run_inventory", "_preflight_routing_runs_output"}
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    }
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
        "importlib",
        "glob",
        "mkdir",
        "write_text",
        "write_bytes",
        "unlink",
        "rename",
        "replace",
        "tempfile",
        "catalog",
        "cache",
    }
    for function in functions.values():
        for node in ast.walk(function):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".", 1)[0] not in forbidden for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".", 1)[0] not in forbidden
            if isinstance(node, ast.Name):
                assert node.id not in forbidden
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden
    routing_handler = functions["_handle_routing_runs"]
    calls = [node for node in ast.walk(routing_handler) if isinstance(node, ast.Call)]
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "_run_routing_run_inventory"
            for node in calls
        )
        == 1
    )
    assert (
        sum(isinstance(node.func, ast.Attribute) and node.func.attr == "to_json" for node in calls)
        == 1
    )
    assert (
        sum(
            isinstance(node.func, ast.Name) and node.func.id == "_write_routing_runs_report"
            for node in calls
        )
        == 1
    )


def test_force_without_output_is_rejected_before_traversal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = tmp_path / "missing-workspace"
    assert main(_args(workspace, force=True)) == 2
    assert capsys.readouterr().err == "moeatlas routing-runs: --force requires --output PATH\n"
    assert not workspace.exists()


def test_output_preflight_precedes_inventory_import_and_workspace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("keep")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("inventory must not run after output preflight")

    monkeypatch.setattr(cli_module, "_run_routing_run_inventory", forbidden)
    original_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise AssertionError("duckdb must remain lazy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert main(_args(tmp_path / "missing", output=output)) == 2
    assert "already exists" in capsys.readouterr().err
    assert called is False
    assert output.read_text() == "keep"


def test_missing_duckdb_is_fixed_dependency_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, _run_key = _fixture(tmp_path)
    original = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name == "duckdb":
            raise ModuleNotFoundError("duckdb blocked")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    assert main(_args(workspace)) == 2
    assert capsys.readouterr().err == "moeatlas routing-runs: routing shard failed at dependency\n"
