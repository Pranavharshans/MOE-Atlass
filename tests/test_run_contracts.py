"""Model-free tests for run identity and provenance contracts."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from moeatlas.loading import QuantizationPolicy
from moeatlas.runs import (
    RUN_SPEC_SCHEMA_VERSION,
    AdapterProvenance,
    ChatMessage,
    DataProvenance,
    DatasetFormat,
    DatasetInputSpec,
    ExecutionEnvironment,
    GenerationConfig,
    InterventionLineage,
    ModelProvenance,
    PrivacyPolicy,
    ProbeProvenance,
    PromptInputSpec,
    RunInputKind,
    RunMode,
    RunSpecification,
    TokenTextPolicy,
    make_run_key,
    parse_run_key,
)

ROOT = Path(__file__).resolve().parents[1]

_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def model_provenance(**overrides: object) -> ModelProvenance:
    values: dict[str, object] = {
        "loading_plan_id": f"loadplan:{_DIGEST_A}",
        "model_id": "org/model",
        "model_revision": "abc123",
        "config_hash": f"sha256:{_DIGEST_B}",
        "tokenizer_revision": "tok-1",
        "quantization": QuantizationPolicy.NONE,
    }
    values.update(overrides)
    return ModelProvenance(**values)  # type: ignore[arg-type]


def prompt_input(**overrides: object) -> PromptInputSpec:
    values: dict[str, object] = {
        "messages": (ChatMessage(role="user", content="hello"),),
    }
    values.update(overrides)
    return PromptInputSpec(**values)  # type: ignore[arg-type]


def dataset_input(**overrides: object) -> DatasetInputSpec:
    values: dict[str, object] = {
        "format": DatasetFormat.JSONL,
        "location": "data/rows.jsonl",
        "row_count": 10,
        "batch_size": 2,
        "seed": 3,
    }
    values.update(overrides)
    return DatasetInputSpec(**values)  # type: ignore[arg-type]


def data_provenance(**overrides: object) -> DataProvenance:
    values: dict[str, object] = {
        "input": prompt_input(),
        "task_labels": ("math",),
    }
    values.update(overrides)
    return DataProvenance(**values)  # type: ignore[arg-type]


def probe_provenance(**overrides: object) -> ProbeProvenance:
    values: dict[str, object] = {
        "probe_plan_id": f"plan:{_DIGEST_A}",
        "capture_level": 1,
    }
    values.update(overrides)
    return ProbeProvenance(**values)  # type: ignore[arg-type]


def run_specification(**overrides: object) -> RunSpecification:
    values: dict[str, object] = {
        "model": model_provenance(),
        "data": data_provenance(),
        "generation": GenerationConfig(seed=7, temperature=0.7),
        "tags": ("b", "a"),
        "created_at": "2026-08-21T00:00:00Z",
    }
    values.update(overrides)
    return RunSpecification(**values)  # type: ignore[arg-type]


def test_run_key_is_content_addressed_and_metadata_insensitive() -> None:
    spec = run_specification()
    assert spec.run_key == make_run_key(spec._identity_payload())
    parse_run_key(spec.run_key)
    for metadata in (
        {"tags": ("different",)},
        {"created_at": "1999-01-01T00:00:00Z"},
        {"workspace": "elsewhere"},
        {"execution": ExecutionEnvironment(python_version="3.11")},
    ):
        assert run_specification(**metadata).run_key == spec.run_key
    assert run_specification(replication=1).run_key != spec.run_key
    assert run_specification(
        generation=GenerationConfig(seed=8, temperature=0.7)
    ).run_key != spec.run_key


def test_run_key_is_stable_across_hash_seeds() -> None:
    script = (
        "from tests.test_run_contracts import run_specification\n"
        "print(run_specification().run_key)\n"
    )
    keys = set()
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{ROOT}{os.pathsep}{ROOT / 'src'}{os.pathsep}" + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    for seed in ("1", "17", "random"):
        child = dict(env)
        child["PYTHONHASHSEED"] = seed
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=child,
            timeout=120,
            check=True,
        )
        keys.add(completed.stdout.strip())
    assert len(keys) == 1


def test_specification_round_trips_through_json() -> None:
    spec = run_specification(
        probe=probe_provenance(intervention_opt_in=False),
        adapter=AdapterProvenance(adapter="mixtral", adapter_version="1.0"),
        privacy=PrivacyPolicy(token_text=TokenTextPolicy.STORED),
        intervention=None,
    )
    revived = RunSpecification.from_json(spec.to_json())
    assert revived == spec
    assert revived.run_key == spec.run_key
    assert RunSpecification.manifest_type == "run_specification"
    payload = spec.to_dict()
    assert payload["schema_version"] == RUN_SPEC_SCHEMA_VERSION
    assert payload["model"]["quantization"] == "none"
    assert payload["privacy"]["token_text"] == "stored"


def test_wrong_supplied_run_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="run_key does not match"):
        RunSpecification.model_validate(
            {**run_specification().to_dict(), "run_key": f"run:{_DIGEST_B}"}
        )


def test_identity_groups_reject_invalid_shapes() -> None:
    with pytest.raises(ValidationError, match="loadplan"):
        model_provenance(loading_plan_id="plan:whatever")
    with pytest.raises(ValidationError, match="config_hash"):
        model_provenance(config_hash="deadbeef")
    with pytest.raises(ValidationError, match="model identifier"):
        model_provenance(model_id="/absolute/path")
    with pytest.raises(ValidationError, match="plan:<64 lowercase hex>"):
        probe_provenance(probe_plan_id=f"loadplan:{_DIGEST_A}")
    with pytest.raises(ValidationError, match="capture_level"):
        probe_provenance(capture_level=6)


def test_adapter_provenance_requires_paired_identity() -> None:
    with pytest.raises(ValidationError, match="together"):
        AdapterProvenance(adapter="solo")
    with pytest.raises(ValidationError, match="inspection_fingerprint"):
        AdapterProvenance(adapter="a", adapter_version="1", inspection_fingerprint="nope")
    paired = AdapterProvenance()
    assert paired.adapter is None and paired.adapter_version is None


def test_prompt_input_requires_exactly_one_form() -> None:
    with pytest.raises(ValidationError, match="exactly one of chat messages or raw text"):
        prompt_input(messages=(), text=None)
    with pytest.raises(ValidationError, match="exactly one of chat messages or raw text"):
        prompt_input(text="both")
    assert prompt_input(messages=(), text="only text").messages == ()
    assert prompt_input().text is None


def test_dataset_descriptor_validates_bounds_and_freezes_mapping() -> None:
    descriptor = dataset_input(
        column_mapping={"prompt": "b", "label": "a"},
        mode=RunMode.TEACHER_FORCED,
        content_digest=f"sha256:{_DIGEST_B}",
    )
    assert descriptor.input_kind is RunInputKind.DATASET
    assert descriptor.column_mapping == {"label": "a", "prompt": "b"}
    with pytest.raises(TypeError, match="immutable"):
        descriptor.column_mapping["prompt"] = "c"  # type: ignore[index]
    with pytest.raises(ValidationError, match="content_digest"):
        dataset_input(content_digest="sha256:short")
    with pytest.raises(ValidationError, match="dataset location"):
        dataset_input(location="")
    with pytest.raises(ValidationError, match="seed"):
        dataset_input(seed=-1)
    with pytest.raises(ValidationError, match="allow_downloads"):
        dataset_input(format=DatasetFormat.JSONL, allow_downloads=True)
    with pytest.raises(ValidationError, match="config_name"):
        dataset_input(format=DatasetFormat.JSONL, config_name="default")


def test_data_fingerprint_is_deterministic_and_sensitive() -> None:
    base = data_provenance()
    assert base.fingerprint.startswith("data:")
    assert base.fingerprint == data_provenance().fingerprint
    assert data_provenance(task_labels=("other",)).fingerprint != base.fingerprint
    reordered = data_provenance(
        input=prompt_input(
            messages=(ChatMessage(role="user", content="hello"),)
        ),
        task_labels=("math",),
    )
    assert reordered.fingerprint == base.fingerprint


def test_generation_config_bounds() -> None:
    with pytest.raises(ValidationError, match="temperature"):
        GenerationConfig(temperature=0.0)
    with pytest.raises(ValidationError, match="top_p"):
        GenerationConfig(top_p=1.5)
    with pytest.raises(ValidationError, match="stop sequence"):
        GenerationConfig(stop_sequences=("dup", "dup"))
    config = GenerationConfig(max_new_tokens=16, do_sample=True)
    round_tripped = GenerationConfig.model_validate(config.model_dump(mode="json"))
    assert round_tripped == config


def test_intervention_lineage_requires_probe_opt_in() -> None:
    lineage = InterventionLineage(
        baseline_run_key=f"run:{_DIGEST_B}",
        recipe_fingerprint=f"sha256:{_DIGEST_B}",
        operation="ablate",
        targets=("component:a", "component:b"),
    )
    assert lineage.targets == ("component:a", "component:b")
    with pytest.raises(ValidationError, match="requires a probe plan"):
        run_specification(intervention=lineage)
    with pytest.raises(ValidationError, match="intervention_opt_in"):
        run_specification(intervention=lineage, probe=probe_provenance())
    bound = run_specification(
        intervention=lineage,
        probe=probe_provenance(intervention_opt_in=True, capture_level=5),
    )
    assert bound.intervention is not None
    assert bound.run_key.startswith("run:")


def test_privacy_defaults_to_redaction() -> None:
    policy = PrivacyPolicy()
    assert policy.token_text is TokenTextPolicy.REDACTED
    assert policy.retain_raw_payloads is False
    assert policy.allow_export is True


def test_execution_environment_is_metadata_and_frozen() -> None:
    environment = ExecutionEnvironment(
        python_version="3.11",
        device_map="auto",
        device_metadata={"gpu": 2},
    )
    with pytest.raises(TypeError, match="immutable"):
        environment.device_metadata["gpu"] = 4  # type: ignore[index]
    assert environment.pytorch_version is None


def test_runs_package_import_boundary() -> None:
    allowed = {
        "specs.py": {"__future__", ".core", "..core", ".loading", "..loading"},
        "lifecycle.py": {"__future__", ".core", "..core", ".specs"},
        "__init__.py": None,
    }
    package_dir = ROOT / "src" / "moeatlas" / "runs"
    for name, permitted in allowed.items():
        source = (package_dir / name).read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add("." * node.level + (node.module or ""))
        forbidden = ("torch", "transformers", "duckdb")
        assert not any(name.split(".")[0] in forbidden for name in imported)
        if permitted is not None:
            local = {name for name in imported if name.startswith(".")}
            unexpected = local - permitted
            assert not unexpected, f"{name} imports outside the runs boundary: {unexpected}"


def test_runs_import_without_model_stack() -> None:
    script = "\n".join(
        (
            "import sys",
            "for name in ('torch', 'transformers', 'duckdb', 'safetensors',",
            "             'moeatlas.runtime', 'moeatlas.store'):",
            "    sys.modules[name] = None",
            "import moeatlas.runs as runs",
            "assert runs.RunSpecification is not None",
            "print('runs-import-ok')",
        )
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "runs-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_contracts() -> None:
    assert not any(
        name in sys.modules for name in ("torch", "transformers", "safetensors")
    )
