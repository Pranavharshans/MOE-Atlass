"""Model-free tests for pre-weight Hugging Face runtime qualification."""

from __future__ import annotations

import json

import pytest

import moeatlas.runtime.qualification as qualification_module
from moeatlas.loading import (
    DownloadPolicy,
    HuggingFaceSource,
    ImmutableRevisionEvidence,
    LoadConfig,
    LoadingPlan,
    ResolvedSource,
    RevisionEvidenceKind,
    SourceKind,
)
from moeatlas.runtime import RuntimeQualificationError, qualify_huggingface_runtime

MODEL_ID = "org/model"
REVISION = "a" * 40


def _plan(*, trust_remote_code: bool = False) -> LoadingPlan:
    source = HuggingFaceSource(
        model_id=MODEL_ID,
        requested_revision="main",
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    config = LoadConfig(
        trust_remote_code=trust_remote_code,
        remote_code_acknowledged=trust_remote_code,
        download_policy=DownloadPolicy.ALLOW_DOWNLOADS,
        allow_downloads=True,
    )
    resolution = ResolvedSource(
        source_type=SourceKind.HUGGINGFACE,
        model_id=MODEL_ID,
        requested_model_revision="main",
        resolved_model_revision=REVISION,
        resolved_model_revision_evidence=ImmutableRevisionEvidence(
            kind=RevisionEvidenceKind.GIT_COMMIT,
            digest=REVISION,
            evidence_source="test resolver",
        ),
    )
    return LoadingPlan(source=source, config=config, resolution=resolution)


def _fetch(files: dict[str, bytes]):
    def fetch(model_id: str, revision: str, filename: str, allow_network: bool):
        assert model_id == MODEL_ID
        assert revision == REVISION
        assert allow_network is True
        return files.get(filename)

    return fetch


@pytest.fixture(autouse=True)
def installed_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    real_version = qualification_module.importlib.metadata.version

    def version(name: str) -> str:
        if name == "transformers":
            return "5.15.1"
        return real_version(name)

    monkeypatch.setattr(qualification_module.importlib.metadata, "version", version)


def test_standard_model_is_ready_without_importing_remote_code() -> None:
    result = qualify_huggingface_runtime(
        _plan(),
        fetch_file=_fetch({"config.json": json.dumps({"model_type": "test"}).encode()}),
    )

    assert result.ready is True
    assert result.remote_code_required is False
    assert result.required_imports == ()
    assert result.environment_id.startswith("runtime:")


def test_missing_remote_symbol_is_rejected_before_weight_loading() -> None:
    files = {
        "config.json": json.dumps(
            {"auto_map": {"AutoModelForCausalLM": "modeling_test.TestModel"}}
        ).encode(),
        "modeling_test.py": b"from json import definitely_missing_symbol\n",
    }
    result = qualify_huggingface_runtime(
        _plan(trust_remote_code=True), fetch_file=_fetch(files)
    )

    assert result.ready is False
    assert result.missing_imports == ("json:definitely_missing_symbol",)
    assert result.inspected_remote_files == ("modeling_test.py",)


def test_remote_code_requires_explicit_permission() -> None:
    files = {
        "config.json": json.dumps(
            {"auto_map": {"AutoModelForCausalLM": "modeling_test.TestModel"}}
        ).encode(),
        "modeling_test.py": b"import json\n",
    }
    result = qualify_huggingface_runtime(_plan(), fetch_file=_fetch(files))

    assert result.ready is False
    assert any("permission is disabled" in warning for warning in result.warnings)


def test_unavailable_remote_source_is_not_claimed_ready() -> None:
    files = {
        "config.json": json.dumps(
            {"auto_map": {"AutoModelForCausalLM": "modeling_test.TestModel"}}
        ).encode(),
    }
    result = qualify_huggingface_runtime(
        _plan(trust_remote_code=True), fetch_file=_fetch(files)
    )

    assert result.ready is False
    assert result.uninspectable_remote_files == ("modeling_test.py",)


def test_missing_requirement_produces_an_install_plan() -> None:
    files = {
        "config.json": b"{}",
        "requirements.txt": b"package-that-cannot-exist-928374>=1.0\n",
    }
    result = qualify_huggingface_runtime(_plan(), fetch_file=_fetch(files))

    assert result.ready is False
    assert result.missing_packages == ("package-that-cannot-exist-928374>=1.0",)
    assert result.install_plan == result.missing_packages


def test_environment_identity_is_stable_and_revision_bound() -> None:
    fetch = _fetch({"config.json": b"{}"})
    first = qualify_huggingface_runtime(_plan(), fetch_file=fetch)
    second = qualify_huggingface_runtime(_plan(), fetch_file=fetch)

    assert first.environment_id == second.environment_id


def test_unavailable_config_fails_qualification() -> None:
    with pytest.raises(RuntimeQualificationError, match="config is unavailable"):
        qualify_huggingface_runtime(_plan(), fetch_file=_fetch({}))
