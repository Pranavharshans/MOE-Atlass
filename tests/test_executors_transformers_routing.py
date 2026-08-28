"""Contract tests for the built-in transformers-routing executor.

All doubles are torch-free fake model/tokenizer pairs composed through the
real ``load_instance`` seam; the lazy HF loader boundary is monkeypatched so
no optional dependency is ever imported. The CLI path must resolve the
built-in by name alone — no entry points installed.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest

import moeatlas.cli as cli_module
from moeatlas.cli import main
from moeatlas.core import DType  # noqa: F401
from moeatlas.executors import (
    EXECUTOR_NAME,
    build_builtin_executor,
    builtin_executor_names,
)
from moeatlas.executors.transformers_routing import (
    TransformersRoutingExecutor,
    _move_model_inputs,
    _safe_validation_error,
)
from moeatlas.runtime.contracts import LoadedModel
from moeatlas.services import initialize_workspace, query_runs
from moeatlas.services.run_engine import RowFailure
from moeatlas.store import list_routing_shards

from .test_cli_scan import SourceKind, _loading_manifest, _loading_plan, _write_plan
from .test_runtime_generic_capture import _flat_logits, _HookedModel

# ---------------------------------------------------------------------------
# Fake runtime objects
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def __call__(self, prompt: str, **kwargs: object) -> dict[str, list[list[int]]]:
        del kwargs
        ids = [(index * 7 + 3) % 97 + 1 for index in range(max(len(prompt), 2))]
        return {"input_ids": [ids], "attention_mask": [[1] * len(ids)]}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [f"t{value}" for value in ids]

    def decode(self, ids: list[int], **kwargs: object) -> str:
        del kwargs
        return " ".join(f"t{value}" for value in ids)


class _PreHookHandle:
    def __init__(self, owner: _GeneratingModel, callback: object) -> None:
        self.owner = owner
        self.callback = callback

    def remove(self) -> None:
        if self.callback in self.owner.pre_callbacks:
            self.owner.pre_callbacks.remove(self.callback)


class _GeneratingModel(_HookedModel):
    """Torch-free generate loop that exposes cached one-token decode calls."""

    def __init__(self) -> None:
        super().__init__(_flat_logits(2))
        self.pre_callbacks: list[object] = []

    def register_forward_pre_hook(
        self, callback: object, *, with_kwargs: bool = False
    ) -> _PreHookHandle:
        assert with_kwargs is True
        self.pre_callbacks.append(callback)
        return _PreHookHandle(self, callback)

    def __call__(self, **kwargs: object) -> object:
        for callback in tuple(self.pre_callbacks):
            callback(self, (), kwargs)
        ids = kwargs["input_ids"]
        count = len(ids[0])  # type: ignore[index]
        for path in self.fire_paths:
            self.payload_by_path[path] = _flat_logits(count)
        return super().__call__(**kwargs)

    def generate(
        self,
        *,
        input_ids: list[list[int]],
        max_new_tokens: int,
        do_sample: bool,
        **kwargs: object,
    ) -> list[list[int]]:
        assert do_sample is False
        prompt = list(input_ids[0])
        current = [list(prompt)]
        generated: list[int] = []
        for index in range(max_new_tokens):
            self(input_ids=current, **kwargs)
            token_id = 80 + index
            generated.append(token_id)
            current = [[token_id]]
        return [prompt + generated]


def _loaded(plan, model=None, tokenizer=None):
    return LoadedModel(
        model=model if model is not None else _HookedModel(_flat_logits(2)),
        tokenizer=tokenizer if tokenizer is not None else _FakeTokenizer(),
        plan=plan,
        manifest=_loading_manifest(plan),
        warnings=(),
    )


@pytest.fixture()
def fake_loader(monkeypatch: pytest.MonkeyPatch):
    def install(**kwargs):
        holder["loaded"] = _loaded(_loading_plan(), **kwargs)

    holder: dict[str, object] = {}

    def fake_load_huggingface(plan):
        assert plan.source.source_type == "huggingface"
        return holder.get("loaded") or _loaded(plan)

    import moeatlas.runtime.model_loader as model_loader

    monkeypatch.setattr(model_loader, "load_huggingface", fake_load_huggingface)
    return holder


# ---------------------------------------------------------------------------
# Registry resolution
# ---------------------------------------------------------------------------


def test_builtin_registry_resolves_by_name(tmp_path: Path) -> None:
    plan = _loading_plan()
    executor = build_builtin_executor(EXECUTOR_NAME, plan)
    assert isinstance(executor, TransformersRoutingExecutor)
    assert builtin_executor_names() == ("transformers-routing",)


def test_unknown_names_fall_through_to_entry_points() -> None:
    plan = _loading_plan()
    assert build_builtin_executor("no-such-executor", plan) is None
    with pytest.raises(TypeError):
        build_builtin_executor(None, plan)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_builtin_executor("transformers-routing", "not-a-plan")  # type: ignore[arg-type]


def test_cli_load_executor_prefers_the_builtin(tmp_path: Path) -> None:
    plan = _loading_plan()
    resolved = cli_module._load_executor("transformers-routing", plan)
    assert isinstance(resolved, TransformersRoutingExecutor)
    with pytest.raises(ValueError, match="executor plugin is not registered"):
        cli_module._load_executor("still-not-registered", plan)


# ---------------------------------------------------------------------------
# Row execution failure paths
# ---------------------------------------------------------------------------


def test_unbound_run_key_is_rejected() -> None:
    plan = _loading_plan()
    executor = TransformersRoutingExecutor(plan)
    with pytest.raises(RowFailure) as excinfo:
        executor(row_index=0, batch_index=0, values={"prompt": "hi"})
    assert excinfo.value.kind == "validation"
    assert (
        str(excinfo.value)
        == "row failure [validation]: executor run key was not bound before execution"
    )


def test_missing_prompt_is_rejected() -> None:
    plan = _loading_plan()
    executor = TransformersRoutingExecutor(plan)
    executor.bind_run_key("run:" + "0" * 64)
    with pytest.raises(RowFailure) as excinfo:
        executor(row_index=0, batch_index=0, values={"text": "hi"})
    assert excinfo.value.message == "row values must carry a non-empty string 'prompt'"


def test_structured_validation_failure_retains_safe_schema_evidence_only() -> None:
    class ValidationError(Exception):
        def errors(self, **kwargs: object) -> list[dict[str, object]]:
            assert kwargs == {"include_input": False, "include_url": False}
            return [
                {
                    "loc": ("router", "top_k"),
                    "type": "greater_than_equal",
                    "msg": "Input should be greater than or equal to 1",
                    "input": "hf_secret_prompt_value",
                    "ctx": {"private": "do-not-retain"},
                }
            ]

    message = _safe_validation_error(ValidationError("unsafe raw exception"))

    assert "router.top_k" in message
    assert "greater_than_equal" in message
    assert "greater than or equal to 1" in message
    assert "hf_secret_prompt_value" not in message
    assert "do-not-retain" not in message
    assert "unsafe raw exception" not in message


def test_non_validation_structured_failure_retains_class_name_only() -> None:
    message = _safe_validation_error(RuntimeError("private prompt and model path"))

    assert message == "structured forward failed (RuntimeError)"


def test_model_inputs_move_tensor_like_values_to_model_device() -> None:
    class TensorLike:
        def __init__(self) -> None:
            self.devices: list[object] = []

        def to(self, device: object) -> TensorLike:
            self.devices.append(device)
            return self

    class DeviceModel:
        device = "cuda:0"

    input_ids = TensorLike()
    attention_mask = TensorLike()
    moved = _move_model_inputs(
        DeviceModel(), {"input_ids": input_ids, "attention_mask": attention_mask}
    )
    assert moved["input_ids"] is input_ids
    assert input_ids.devices == ["cuda:0"]
    assert attention_mask.devices == ["cuda:0"]


def test_unsupported_source_plan_is_a_dependency_failure(fake_loader) -> None:
    plan = _loading_plan(source_kind=SourceKind.INSTANCE)
    executor = TransformersRoutingExecutor(plan)
    executor.bind_run_key("run:" + "0" * 64)
    with pytest.raises(RowFailure) as excinfo:
        executor(row_index=0, batch_index=0, values={"prompt": "hi"})
    assert excinfo.value.kind == "dependency"
    assert excinfo.value.message == "loading plan source type is not supported by this executor"


def test_missing_tokenizer_is_a_dependency_failure(fake_loader) -> None:
    loaded = _loaded(_loading_plan())
    broken = dataclasses.replace(loaded, tokenizer=None)
    fake_loader["loaded"] = broken
    executor = TransformersRoutingExecutor(_loading_plan())
    executor.bind_run_key("run:" + "0" * 64)
    with pytest.raises(RowFailure) as excinfo:
        executor(row_index=0, batch_index=0, values={"prompt": "hi"})
    assert excinfo.value.kind == "dependency"
    assert excinfo.value.message == "loaded model did not resolve a tokenizer"


def test_native_forward_mode_skips_capture_and_reports_timing(fake_loader) -> None:
    executor = TransformersRoutingExecutor(_loading_plan(), capture_routing=False)
    executor.bind_run_key("run:" + "0" * 64)

    result = executor(row_index=0, batch_index=0, values={"prompt": "hi"})

    assert result["capture_routing"] is False
    assert result["expert_backend_handshake"]["status"] == "unavailable"
    assert result["expert_backend_handshake"]["restored"] is True
    assert executor._token_events == []
    timing = executor.timing_summary()
    assert timing["capture_routing"] is False
    assert timing["successful_rows"] == 1
    assert isinstance(timing["total_ms"], float)


def test_capture_routing_requires_an_exact_boolean() -> None:
    with pytest.raises(TypeError, match="capture_routing must be an exact bool"):
        TransformersRoutingExecutor(_loading_plan(), capture_routing=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown evaluation method"):
        TransformersRoutingExecutor(_loading_plan(), evaluation_method="arbitrary-code")
    with pytest.raises(TypeError, match="load_progress must be callable"):
        TransformersRoutingExecutor(_loading_plan(), load_progress=object())  # type: ignore[arg-type]


def test_generation_output_and_routing_come_from_the_same_model_calls(fake_loader) -> None:
    model = _GeneratingModel()
    fake_loader["loaded"] = _loaded(_loading_plan(), model=model)
    executor = TransformersRoutingExecutor(
        _loading_plan(),
        store_token_text=True,
        capture_expert_activity=True,
        max_new_tokens=3,
        evaluation_method="token_f1",
    )
    executor.bind_run_key("run-generation")

    result = executor(
        row_index=0,
        batch_index=0,
        values={"prompt": "ab", "reference": "t80 t82"},
    )

    assert model.calls == 3
    assert result["output_mode"] == "generated"
    assert result["output_token_count"] == 3
    assert result["output_preview"] == "t80 t81 t82"
    assert result["score_name"] == "token_f1"
    assert result["task_score"] == pytest.approx(0.8)
    assert result["evaluation_method"] == "token_f1"
    assert result["input_digest"].startswith("sha256:")
    assert isinstance(result["generation_ms"], float)
    assert result["token_count"] == 4
    assert result["prefill_token_count"] == 2
    assert result["decode_token_count"] == 2
    assert result["routing_scope"] == "actual_generation"
    assert result["routing_event_count"] == 16
    assert [event.phase.value for event in executor._token_events] == [
        "prefill",
        "prefill",
        "decode",
        "decode",
    ]
    assert "generation_routing_excludes_terminal_output_token" in result["capability_notes"]
    assert "generation_expert_activity_unavailable" in result["capability_notes"]
    assert model.pre_callbacks == []
    assert all(
        node.callbacks == [] for _path, node in model.named_modules() if hasattr(node, "callbacks")
    )


def test_capture_mismatch_surfaces_as_execution_evidence(fake_loader) -> None:
    silent_model = _HookedModel(_flat_logits(2), skip_paths=("layers.1.router",))
    fake_loader["loaded"] = _loaded(_loading_plan(), model=silent_model)
    executor = TransformersRoutingExecutor(_loading_plan())
    executor.bind_run_key("run:" + "0" * 64)
    with pytest.raises(RowFailure) as excinfo:
        executor(row_index=0, batch_index=0, values={"prompt": "hi"})
    assert excinfo.value.kind == "execution"
    assert (
        excinfo.value.message
        == "structured routing capture failed at events: routers did not fire during "
        "the forward: ['layers.1.router']"
    )


def test_loader_is_closed_when_executor_discovery_fails(fake_loader, monkeypatch) -> None:
    """A failed topology scan must not retain the model for a later retry."""

    loaded = _loaded(_loading_plan())
    fake_loader["loaded"] = loaded

    def fail_scan(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic scan failure")

    import moeatlas.executors.transformers_routing as executor_module

    monkeypatch.setattr(executor_module, "scan", fail_scan)
    executor = TransformersRoutingExecutor(_loading_plan())
    executor.bind_run_key("run:" + "0" * 64)

    with pytest.raises(RowFailure, match="model loading failed"):
        executor(row_index=0, batch_index=0, values={"prompt": "hi"})
    assert loaded.closed is True


# ---------------------------------------------------------------------------
# Happy path over the shared run service and store
# ---------------------------------------------------------------------------


def _workspace_with_rows(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    (workspace / "rows.jsonl").write_text('{"prompt": "ab"}\n{"prompt": "cd"}\n', encoding="utf-8")
    descriptor = tmp_path / "dataset.json"
    descriptor.write_text(
        '{"format": "jsonl", "location": "rows.jsonl", "batch_size": 1}',
        encoding="utf-8",
    )
    return workspace, descriptor


def test_executor_happy_path_appends_shard_and_updates_catalog(
    tmp_path: Path,
    fake_loader,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workspace, descriptor = _workspace_with_rows(tmp_path)
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, _loading_plan())
    code = main(
        [
            "run",
            str(workspace),
            "--loading-plan",
            str(plan_path),
            "--dataset",
            str(descriptor),
            "--executor",
            "transformers-routing",
            "--at",
            "2026-08-22T00:00:00Z",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "completed" in captured.out

    entries = query_runs(workspace)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.state == "completed"
    assert entry.shard_count == 1
    assert entry.token_event_count > 0
    assert entry.routing_event_count == entry.token_event_count * 4  # 2 layers x top-k 2

    shards = list_routing_shards(workspace, run_key=entry.run_key)
    assert len(shards) == 1
    assert shards[0].token_text_stored is False
    inspection_path = workspace / "inspections" / f"{entry.run_key}.json"
    assert inspection_path.is_file()
    assert "universal_routing_inspection" in inspection_path.read_text(encoding="utf-8")


def test_publish_without_events_returns_none_and_stays_idempotent(
    tmp_path: Path,
    fake_loader,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    executor = TransformersRoutingExecutor(_loading_plan())
    assert executor.publish_run_artifacts(workspace) is None
    with pytest.raises(RuntimeError, match="already published"):
        executor._published = True
        executor.publish_run_artifacts(workspace)


def test_store_token_text_opt_in_is_forwarded(
    tmp_path: Path,
    fake_loader,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    initialize_workspace(workspace)
    plan = _loading_plan()
    executor = TransformersRoutingExecutor(plan, store_token_text=True)
    executor.bind_run_key("run-1")
    executor(row_index=0, batch_index=0, values={"prompt": "ab"})
    receipt = executor.publish_run_artifacts(workspace)
    assert receipt is not None
    assert receipt.token_text_stored is True
    shards = list_routing_shards(workspace, run_key="run-1")
    assert len(shards) == 1
    entry = query_runs(workspace)[0]
    assert entry.run_key == "run-1"
    assert entry.token_event_count == 2


# ---------------------------------------------------------------------------
# Isolation guards
# ---------------------------------------------------------------------------


def test_executor_module_imports_without_model_stack() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys\n"
        "for name in ('torch', 'transformers', 'duckdb', 'safetensors'):\n"
        "    sys.modules[name] = None\n"
        "import moeatlas.executors\n"
        "import moeatlas.executors.transformers_routing\n"
        "print('executor-import-ok')\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")).rstrip(
        os.pathsep
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "executor-import-ok" in completed.stdout


def test_no_model_runtime_imported_by_executors_package() -> None:
    import moeatlas.executors.transformers_routing as module

    assert module.__name__ == "moeatlas.executors.transformers_routing"
    assert not any(name in sys.modules for name in ("torch", "transformers", "safetensors"))
