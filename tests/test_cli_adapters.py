"""Contract tests for the `moeatlas adapters list` command."""

from __future__ import annotations

import json

import pytest

from moeatlas.cli import build_parser, main


def test_parser_exposes_adapters_list_with_policy_flags() -> None:
    parser = build_parser()
    subparsers = next(action for action in parser._actions if action.dest == "command")
    adapters = subparsers.choices["adapters"]
    nested = next(action for action in adapters._actions if action.dest == "adapters_command")
    command = nested.choices["list"]
    dests = {action.dest for action in command._actions}
    assert {"json", "builtin_only", "enable", "disable", "family"}.issubset(dests)


def test_list_prints_sorted_builtin_plugins(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["adapters", "list"]) == 0
    out = capsys.readouterr().out.splitlines()
    names = [line.split(" ")[0] for line in out[1:] if not line.startswith("adapter plugins:")]
    assert names == sorted(names)
    assert {
        "huggingface-mixtral-static",
        "huggingface-qwen3-moe-static",
        "huggingface-qwen3.5-moe-static",
    }.issubset(set(names))
    header = out[0]
    assert header.startswith("adapter plugins: ")
    assert "enabled" in header


def test_json_emits_canonical_registry_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["adapters", "list", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["artifact_type"] == "moeatlas.adapter_registry"
    assert document["schema_version"] == "1.0"
    names = [item["name"] for item in document["entries"]]
    assert names == sorted(names)
    assert all(item["status"] == "enabled" for item in document["entries"])


def test_disable_flag_marks_plugin_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["adapters", "list", "--disable", "huggingface-mixtral-static"]) == 0
    lines = capsys.readouterr().out.splitlines()
    mixtral = next(line for line in lines if line.startswith("huggingface-mixtral-static "))
    assert " disabled " in f" {mixtral} "


def test_enable_allowlist_disables_unlisted_plugins(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["adapters", "list", "--enable", "huggingface-qwen3-moe-static"]) == 0
    lines = capsys.readouterr().out.splitlines()
    mixtral = next(line for line in lines if line.startswith("huggingface-mixtral-static "))
    qwen = next(line for line in lines if line.startswith("huggingface-qwen3-moe-static "))
    assert " disabled " in f" {mixtral} "
    assert " enabled " in f" {qwen} "


def test_conflicting_enable_and_disable_is_rejected(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "adapters",
            "list",
            "--enable",
            "acme",
            "--disable",
            "acme",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "invalid adapter policy" in captured.err


def test_builtin_only_keeps_builtins_enabled(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["adapters", "list", "--builtin-only"]) == 0
    lines = capsys.readouterr().out.splitlines()
    mixtral = next(line for line in lines if line.startswith("huggingface-mixtral-static "))
    assert " enabled " in f" {mixtral} "


def test_family_filter_keeps_only_matching_enabled_records(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["adapters", "list", "--family", "mixtral"]) == 0
    lines = capsys.readouterr().out.splitlines()
    listed = [line for line in lines[1:] if not line.startswith("adapter plugins:")]
    assert [line.split(" ")[0] for line in listed] == ["huggingface-mixtral-static"]

    assert main(["adapters", "list", "--family", "no_such_family"]) == 0
    captured = capsys.readouterr()
    assert "0 listed" in captured.out
