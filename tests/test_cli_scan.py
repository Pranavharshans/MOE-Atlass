from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

import moeatlas.scan as scan_module
from moeatlas.cli import main
from moeatlas.core import make_config_hash
from moeatlas.discovery import DiscoveryReport
from moeatlas.fixtures import SyntheticConfig, SyntheticMoE
from moeatlas.scan import PHASE0_FIXTURE_SOURCE, SYNTHETIC_MODEL_KEY, synthetic_model_manifest


def test_scan_stdout_is_complete_deterministic_json(capsys: pytest.CaptureFixture[str]) -> None:
    first_code = main(["scan", PHASE0_FIXTURE_SOURCE])
    first = capsys.readouterr()
    second_code = main(["scan", PHASE0_FIXTURE_SOURCE])
    second = capsys.readouterr()

    assert first_code == second_code == 0
    assert first.err == second.err == ""
    assert first.out == second.out
    report = DiscoveryReport.model_validate_json(first.out)
    assert report.model_key == SYNTHETIC_MODEL_KEY
    assert report.model_manifest.provenance is not None
    assert report.model_manifest.provenance.source == PHASE0_FIXTURE_SOURCE
    assert report.model_manifest.warnings
    assert report.facts.expert_count == 4
    assert report.facts.routed_top_k == 2
    assert report.facts.shared_expert_count == 1
    assert len(report.candidates) == len(report.components) == 16


def test_scan_output_file_matches_stdout_and_validates(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "nested" / "report.json"
    output_path.parent.mkdir()

    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--output", str(output_path)]) == 0
    saved = capsys.readouterr()
    assert saved.out == ""
    assert f"saved scan report to {output_path}" in saved.err
    file_bytes = output_path.read_bytes()
    DiscoveryReport.model_validate_json(file_bytes)

    assert main(["scan", PHASE0_FIXTURE_SOURCE]) == 0
    stdout = capsys.readouterr()
    assert file_bytes == stdout.out.encode()


def test_scan_refuses_overwrite_without_force_and_force_replaces(tmp_path: Path, capsys) -> None:
    output_path = tmp_path / "report.json"
    original = b"preserve me\n"
    output_path.write_bytes(original)

    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--output", str(output_path)]) == 2
    refused = capsys.readouterr()
    assert refused.out == ""
    assert "already exists" in refused.err
    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []

    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--output", str(output_path), "--force"]) == 0
    replaced = capsys.readouterr()
    assert replaced.out == ""
    assert output_path.read_bytes() != original
    DiscoveryReport.model_validate_json(output_path.read_bytes())


def test_scan_output_failures_are_concise_and_leave_no_partial_file(
    tmp_path: Path, capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_parent_output = tmp_path / "missing" / "report.json"
    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--output", str(missing_parent_output)]) == 2
    missing = capsys.readouterr()
    assert missing.out == ""
    assert "parent does not exist" in missing.err
    assert not missing_parent_output.parent.exists()

    output_path = tmp_path / "injected.json"
    original = b"keep this original\n"
    output_path.write_bytes(original)

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(scan_module.os, "replace", fail_replace)
    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--output", str(output_path), "--force"]) == 2
    failed = capsys.readouterr()
    assert failed.out == ""
    assert "could not write output" in failed.err
    assert "injected replace failure" in failed.err
    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


def test_scan_no_force_publication_is_no_clobber_for_races_and_broken_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "race.json"

    def publish_race(source: Path, destination: Path) -> None:
        raise FileExistsError(destination)

    monkeypatch.setattr(scan_module.os, "link", publish_race)
    with pytest.raises(scan_module.ScanOutputError, match="already exists"):
        scan_module.write_report_atomic("{}\n", output_path)
    assert not output_path.exists()
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []

    broken_target = tmp_path / "missing-target"
    broken_link = tmp_path / "broken.json"
    try:
        broken_link.symlink_to(broken_target)
    except OSError as exc:  # pragma: no cover - platform-specific CI fallback
        pytest.skip(f"symlink creation unavailable: {exc}")
    with pytest.raises(scan_module.ScanOutputError, match="already exists"):
        scan_module.write_report_atomic("{}\n", broken_link)
    assert broken_link.is_symlink()
    assert list(tmp_path.glob(f".{broken_link.name}.*.tmp")) == []


def test_synthetic_manifest_hash_matches_canonical_fixture_config() -> None:
    fixture = SyntheticMoE()
    manifest = synthetic_model_manifest()
    assert fixture.config == SyntheticConfig()
    assert manifest.config_hash == make_config_hash(asdict(fixture.config))


def test_scan_rejects_unsupported_sources_without_loader_or_fixture_fallback(
    tmp_path: Path, capsys
) -> None:
    unsupported = str(tmp_path / "local-model")
    assert main(["scan", unsupported]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not available in Phase 0" in captured.err
    assert "HF/local model loading is deferred" in captured.err
    assert not (tmp_path / "local-model").exists()


def test_scan_help_and_argument_errors_are_normal_cli_failures(capsys) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["scan", "--help"])
    help_output = capsys.readouterr()
    assert help_exit.value.code == 0
    assert "fixture:synthetic" in help_output.out
    assert "deferred" in help_output.out
    assert help_output.err == ""

    with pytest.raises(SystemExit) as parse_exit:
        main(["scan"])
    parse_output = capsys.readouterr()
    assert parse_exit.value.code == 2
    assert "usage:" in parse_output.err
    assert "Traceback" not in parse_output.err

    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--force"]) == 2
    force_output = capsys.readouterr()
    assert force_output.out == ""
    assert "--force requires --output PATH" in force_output.err


def test_scan_source_and_report_do_not_import_model_runtime() -> None:
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))


def test_scan_json_is_parseable_by_stdlib() -> None:
    report = scan_module.scan_source(PHASE0_FIXTURE_SOURCE)
    decoded = json.loads(scan_module.report_payload(report))
    assert decoded["manifest_type"] == "discovery_report"
