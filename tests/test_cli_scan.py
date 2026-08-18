from __future__ import annotations

import io
import json
import socket
import sys
import urllib.request
from dataclasses import asdict
from pathlib import Path

import pytest

import moeatlas.runtime as runtime_module
import moeatlas.scan as scan_module
from moeatlas.cli import main
from moeatlas.core import (
    DType,
    ModelManifest,
    Provenance,
    TokenizerIdentity,
    make_config_hash,
    make_model_key,
)
from moeatlas.discovery import DiscoveryReport
from moeatlas.discovery import scan as discover_scan
from moeatlas.fixtures import SyntheticConfig, SyntheticMoE
from moeatlas.loading import (
    CustomLoaderSource,
    DownloadPolicy,
    HuggingFaceSource,
    ImmutableRevisionEvidence,
    InstanceSource,
    LoadConfig,
    LoadingPlan,
    LocalSource,
    ResolvedSource,
    RevisionEvidenceKind,
    SourceKind,
    TokenizerRequest,
)
from moeatlas.runtime import RuntimeCleanupError
from moeatlas.scan import (
    PHASE0_FIXTURE_SOURCE,
    SYNTHETIC_MODEL_KEY,
    report_payload,
    synthetic_model_manifest,
)

PLAN_MODEL_ID = "fixture/plan-moe"
PLAN_MODEL_REQUEST = "requested-model"
PLAN_MODEL_COMMIT = "a" * 40
PLAN_TOKENIZER_ID = "fixture/plan-tokenizer"
PLAN_TOKENIZER_REQUEST = "requested-tokenizer"
PLAN_TOKENIZER_COMMIT = "b" * 40


def _loading_plan(
    source_kind: SourceKind = SourceKind.HUGGINGFACE,
    *,
    path: str = "/private/tmp/plan-model",
    config: LoadConfig | None = None,
    resolved: bool = True,
) -> LoadingPlan:
    effective_config = config or LoadConfig()
    tokenizer = TokenizerRequest(
        identifier=PLAN_TOKENIZER_ID,
        requested_revision=PLAN_TOKENIZER_REQUEST,
    )
    common = {
        "model_id": PLAN_MODEL_ID,
        "requested_revision": PLAN_MODEL_REQUEST,
        "tokenizer": tokenizer,
    }
    if source_kind is SourceKind.HUGGINGFACE:
        source = HuggingFaceSource(
            download_policy=effective_config.download_policy,
            allow_downloads=effective_config.allow_downloads,
            **common,
        )
    elif source_kind is SourceKind.LOCAL:
        source = LocalSource(path=path, **common)
    elif source_kind is SourceKind.INSTANCE:
        source = InstanceSource(**common)
    else:
        source = CustomLoaderSource(
            loader_reference="tests.test_cli_scan:fixture_loader",
            **common,
        )
    resolution = None
    if resolved:
        resolution = ResolvedSource(
            source_type=source_kind,
            model_id=PLAN_MODEL_ID,
            requested_model_revision=PLAN_MODEL_REQUEST,
            resolved_model_revision=PLAN_MODEL_COMMIT,
            resolved_model_revision_evidence=ImmutableRevisionEvidence(
                kind=RevisionEvidenceKind.GIT_COMMIT,
                digest=PLAN_MODEL_COMMIT,
                evidence_source="cli-test-resolver",
            ),
            requested_tokenizer_revision=PLAN_TOKENIZER_REQUEST,
            resolved_tokenizer_revision=PLAN_TOKENIZER_COMMIT,
            resolved_tokenizer_revision_evidence=ImmutableRevisionEvidence(
                kind=RevisionEvidenceKind.GIT_COMMIT,
                digest=PLAN_TOKENIZER_COMMIT,
                evidence_source="cli-test-tokenizer-resolver",
            ),
        )
    return LoadingPlan(source=source, config=effective_config, resolution=resolution)


def _loading_manifest(plan: LoadingPlan) -> ModelManifest:
    assert plan.resolution is not None
    return ModelManifest(
        model_key=make_model_key(
            plan.source.model_id,
            plan.resolution.resolved_model_revision,
        ),
        architecture="synthetic_moe",
        revision=plan.resolution.resolved_model_revision,
        config_hash=make_config_hash(asdict(SyntheticConfig())),
        tokenizer=TokenizerIdentity(
            identifier=PLAN_TOKENIZER_ID,
            revision=PLAN_TOKENIZER_COMMIT,
        ),
        dtype=DType.FLOAT32,
        device_map={"": "cpu"},
        provenance=Provenance(source="cli-test", tool_version="test"),
    )


def _write_plan(path: Path, plan: LoadingPlan) -> None:
    path.write_text(plan.to_json(), encoding="utf-8")


def _fake_plan_bridge(
    monkeypatch: pytest.MonkeyPatch,
    plan: LoadingPlan,
) -> tuple[list[LoadingPlan], DiscoveryReport]:
    received: list[LoadingPlan] = []
    report = discover_scan(SyntheticMoE(), _loading_manifest(plan))

    def fake_load_and_scan(received_plan: LoadingPlan) -> DiscoveryReport:
        received.append(received_plan)
        return report

    monkeypatch.setattr(runtime_module, "load_and_scan", fake_load_and_scan)
    return received, report


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


@pytest.mark.parametrize("source_kind", [SourceKind.HUGGINGFACE, SourceKind.LOCAL])
def test_loading_plan_scan_delegates_the_validated_plan_and_emits_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    source_kind: SourceKind,
) -> None:
    plan = _loading_plan(source_kind, path=str(tmp_path / "declared-model"))
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    received, report = _fake_plan_bridge(monkeypatch, plan)

    assert main(["scan", "--loading-plan", str(plan_path)]) == 0
    captured = capsys.readouterr()

    assert captured.err == ""
    assert captured.out == report_payload(report)
    assert len(received) == 1
    assert received[0] == plan
    assert received[0].source == plan.source
    assert received[0].plan_id == plan.plan_id


def test_explicit_online_loading_plan_warnings_are_stderr_before_dispatch_and_not_report_input(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = LoadConfig(
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
        trust_remote_code=True,
        remote_code_acknowledged=True,
    )
    plan = _loading_plan(config=config)
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    report = discover_scan(SyntheticMoE(), _loading_manifest(plan))
    events: list[tuple[str, str]] = []
    stream = io.StringIO()

    def fake_load_and_scan(received_plan: LoadingPlan) -> DiscoveryReport:
        events.append(("dispatch", stream.getvalue()))
        assert received_plan == plan
        return report

    monkeypatch.setattr(runtime_module, "load_and_scan", fake_load_and_scan)
    monkeypatch.setattr(sys, "stderr", stream)

    assert main(["scan", "--loading-plan", str(plan_path)]) == 0
    captured = capsys.readouterr()

    assert events and events[0][1]
    assert "trust_remote_code" in events[0][1]
    assert "model downloads are explicitly allowed" in events[0][1]
    assert "Hugging Face source permits network downloads" in events[0][1]
    assert captured.out == report_payload(report)
    assert "plan.json" not in captured.out
    assert "model downloads are explicitly allowed" not in captured.out
    assert "Hugging Face source permits network downloads" not in captured.out


def test_loading_plan_argument_contract_rejects_mutual_exclusion_and_force_without_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _loading_plan())
    called = False

    def forbidden(_plan: LoadingPlan) -> DiscoveryReport:
        nonlocal called
        called = True
        raise AssertionError("runtime bridge must not be called")

    monkeypatch.setattr(runtime_module, "load_and_scan", forbidden)

    assert main(["scan", PHASE0_FIXTURE_SOURCE, "--loading-plan", str(plan_path)]) == 2
    mutual = capsys.readouterr()
    assert mutual.out == ""
    assert "mutually exclusive" in mutual.err
    assert called is False

    assert main(["scan", "--loading-plan", str(plan_path), "--force"]) == 2
    forced = capsys.readouterr()
    assert forced.out == ""
    assert "requires --output PATH" in forced.err
    assert called is False


@pytest.mark.parametrize(
    "case",
    ["missing", "unreadable", "utf8", "json", "schema", "plan_id", "credential"],
)
def test_invalid_loading_plan_fails_before_dispatch_without_echo_or_publication(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    plan_path = tmp_path / "input-plan.json"
    output_path = tmp_path / "report.json"
    original = b"preserve existing output\n"
    output_path.write_bytes(original)
    plan = _loading_plan()
    if case == "missing":
        input_path = tmp_path / "does-not-exist.json"
    elif case == "unreadable":
        input_path = tmp_path / "plan-directory"
        input_path.mkdir()
    else:
        input_path = plan_path
        if case == "utf8":
            input_path.write_bytes(b"\xff\xfe\xfa")
        elif case == "json":
            input_path.write_text("{not json", encoding="utf-8")
        elif case == "schema":
            input_path.write_text('{"schema_version":"1.0"}', encoding="utf-8")
        elif case == "plan_id":
            payload = json.loads(plan.to_json())
            payload["plan_id"] = "loadplan:" + "0" * 64
            input_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            payload = json.loads(plan.to_json())
            payload.pop("plan_id")
            payload["config"]["loader_options"] = {
                "Authorization": "TOP_SECRET_VALUE",
            }
            input_path.write_text(json.dumps(payload), encoding="utf-8")

    called = False

    def forbidden(_plan: LoadingPlan) -> DiscoveryReport:
        nonlocal called
        called = True
        raise AssertionError("runtime bridge must not be called")

    monkeypatch.setattr(runtime_module, "load_and_scan", forbidden)
    before_optional = {
        name
        for name in ("torch", "transformers", "accelerate", "safetensors")
        if name in sys.modules
    }

    assert (
        main(
            [
                "scan",
                "--loading-plan",
                str(input_path),
                "--output",
                str(output_path),
                "--force",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert called is False
    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []
    assert "TOP_SECRET_VALUE" not in captured.err
    if case == "credential":
        assert "Authorization" not in captured.err
    after_optional = {
        name
        for name in ("torch", "transformers", "accelerate", "safetensors")
        if name in sys.modules
    }
    assert after_optional == before_optional


@pytest.mark.parametrize("source_kind", [SourceKind.INSTANCE, SourceKind.CUSTOM])
def test_loading_plan_scan_rejects_instance_and_custom_before_runtime_dispatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    source_kind: SourceKind,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _loading_plan(source_kind))
    called = False

    def forbidden(_plan: LoadingPlan) -> DiscoveryReport:
        nonlocal called
        called = True
        raise AssertionError("unsupported source was dispatched")

    monkeypatch.setattr(runtime_module, "load_and_scan", forbidden)

    assert main(["scan", "--loading-plan", str(plan_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "only HuggingFaceSource and LocalSource" in captured.err
    assert called is False


@pytest.mark.parametrize("source_kind", [SourceKind.HUGGINGFACE, SourceKind.LOCAL])
def test_loading_plan_scan_rejects_unresolved_sources_before_runtime_dispatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    source_kind: SourceKind,
) -> None:
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _loading_plan(source_kind, resolved=False))
    called = False

    def forbidden(_plan: LoadingPlan) -> DiscoveryReport:
        nonlocal called
        called = True
        raise AssertionError("unresolved source was dispatched")

    monkeypatch.setattr(runtime_module, "load_and_scan", forbidden)

    assert main(["scan", "--loading-plan", str(plan_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "immutable model and tokenizer resolution" in captured.err
    assert called is False


def test_loading_plan_runtime_cleanup_failure_preserves_existing_output_and_temp_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _loading_plan()
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "report.json"
    _write_plan(plan_path, plan)
    original = b"existing report\n"
    output_path.write_bytes(original)

    def fail_cleanup(_plan: LoadingPlan) -> DiscoveryReport:
        raise RuntimeCleanupError((OSError("private cleanup detail"),))

    monkeypatch.setattr(runtime_module, "load_and_scan", fail_cleanup)

    assert (
        main(
            [
                "scan",
                "--loading-plan",
                str(plan_path),
                "--output",
                str(output_path),
                "--force",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []
    assert "private cleanup detail" not in captured.err


def test_loading_plan_runtime_value_error_is_generic_and_does_not_publish(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _loading_plan()
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "report.json"
    _write_plan(plan_path, plan)
    original = b"keep this report\n"
    output_path.write_bytes(original)

    def fail(_plan: LoadingPlan) -> DiscoveryReport:
        raise ValueError("TOP_SECRET_RUNTIME_VALUE")

    monkeypatch.setattr(runtime_module, "load_and_scan", fail)

    assert (
        main(
            [
                "scan",
                "--loading-plan",
                str(plan_path),
                "--output",
                str(output_path),
                "--force",
            ]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "moeatlas scan: loading or report generation failed\n"
    assert "TOP_SECRET_RUNTIME_VALUE" not in captured.err
    assert output_path.read_bytes() == original
    assert list(tmp_path.glob(f".{output_path.name}.*.tmp")) == []


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_loading_plan_control_flow_exceptions_propagate_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[BaseException],
) -> None:
    plan = _loading_plan()
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, plan)
    original = error_type("control-flow")

    def fail(_plan: LoadingPlan) -> DiscoveryReport:
        raise original

    monkeypatch.setattr(runtime_module, "load_and_scan", fail)

    with pytest.raises(error_type) as caught:
        main(["scan", "--loading-plan", str(plan_path)])
    assert caught.value is original


def test_loading_plan_scan_does_not_touch_network_caches_or_extra_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_names = (
        "HF_HOME",
        "TRANSFORMERS_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "TORCH_HOME",
    )
    cache_dirs: list[Path] = []
    for index, name in enumerate(cache_names):
        cache_dir = tmp_path / f"cache-{index}"
        cache_dir.mkdir()
        cache_dirs.append(cache_dir)
        monkeypatch.setenv(name, str(cache_dir))

    def network_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("network access is outside the CLI plan boundary")

    monkeypatch.setattr(socket, "create_connection", network_forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", network_forbidden)

    plan = _loading_plan()
    plan_path = tmp_path / "plan.json"
    output_path = tmp_path / "report.json"
    _write_plan(plan_path, plan)
    received, _report = _fake_plan_bridge(monkeypatch, plan)

    def entries() -> set[Path]:
        return {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    expected_before = {path.relative_to(tmp_path) for path in cache_dirs}
    expected_before.add(plan_path.relative_to(tmp_path))
    assert entries() == expected_before

    assert main(["scan", "--loading-plan", str(plan_path), "--output", str(output_path)]) == 0
    captured = capsys.readouterr()

    assert captured.out == ""
    assert len(received) == 1
    expected_after = expected_before | {output_path.relative_to(tmp_path)}
    assert entries() == expected_after
    assert not list(tmp_path.glob("**/*.tmp"))
    assert not list(tmp_path.glob("**/.report.json.*"))
    assert output_path.is_file()
