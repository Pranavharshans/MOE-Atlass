"""The built-in ``transformers-routing`` real-model executor.

This executor binds one validated :class:`~moeatlas.loading.LoadingPlan` and
drives planned prompt rows through the real model that plan resolves. It adds
no loading logic of its own: models arrive through the existing
``moeatlas.runtime`` seams (:func:`~moeatlas.runtime.load_huggingface`,
:func:`~moeatlas.runtime.load_local`), structure evidence arrives through the
existing static scanner, and routing/expert capture composes through the
generic structure-driven seams
(:func:`~moeatlas.runtime.run_structured_routing_forward` and
:func:`~moeatlas.runtime.run_structured_expert_forward`).

Every optional dependency stays lazy: importing this module never imports
``torch`` or ``transformers``; they enter only inside the runtime loaders once
a row actually executes. Row failures are evidence, not run deaths — declared
with :class:`~moeatlas.services.run_engine.RowFailure`. Publication appends
one immutable structured shard through the existing store boundary and
reconciles the workspace catalog so ``/api/runs`` sees the run without manual
steps.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from ..core import validate_stable_identifier
from ..discovery import DiscoveryReport, scan
from ..events import EVENT_SCHEMA_VERSION, ExpertEvent, RoutingEvent, TokenEvent, TokenPhase
from ..loading import HuggingFaceSource, LoadingPlan, LocalSource
from ..probe import ProbeResolutionError
from ..runtime.contracts import LoadedModel
from ..runtime.generic_capture import (
    StructuredCaptureError,
    StructuredRouterTarget,
    StructuredRoutingForwardResult,
    run_structured_expert_forward,
    run_structured_routing_forward,
    structured_router_targets,
)
from ..services.run_engine import RowFailure

EXECUTOR_RESULT_SCHEMA_VERSION = "1.0"

_TOKENIZER_CALL_FAILURE = "tokenizer call failed"
_TOKENIZER_SHAPE_FAILURE = "tokenizer encoding must be shaped exactly (1, N)"
_PROMPT_FAILURE = "row values must carry a non-empty string 'prompt'"
_TOKENIZER_MISSING_FAILURE = "loaded model did not resolve a tokenizer"
_LOAD_FAILURE = "model loading failed before execution"
_RUN_KEY_FAILURE = "executor run key was not bound before execution"


def _materialize_ids(value: object) -> list[int]:
    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        current = method() if callable(method) else current
    tolist = getattr(current, "tolist", None)
    rows = tolist() if callable(tolist) else current
    if type(rows) is list and len(rows) == 1 and type(rows[0]) is list:
        ids = rows[0]
    else:
        raise ValueError(_TOKENIZER_SHAPE_FAILURE)
    if any(type(item) is not int or isinstance(item, bool) or item < 0 for item in ids):
        raise ValueError("tokenizer input_ids must be non-negative integers")
    return ids


def _model_input_device(model: object) -> object | None:
    """Resolve the device expected by a model's first input tensor."""

    try:
        device = getattr(model, "device", None)
    except Exception:
        device = None
    if device is not None:
        return device
    try:
        parameters = getattr(model, "parameters", None)
        if callable(parameters):
            first = next(iter(parameters()))
            return getattr(first, "device", None)
    except (StopIteration, Exception):
        return None
    return None


def _move_model_inputs(model: object, encoding: dict[str, object]) -> dict[str, object]:
    """Move tensor-like tokenizer fields without importing a model stack."""

    device = _model_input_device(model)
    if device is None:
        return dict(encoding)
    moved: dict[str, object] = {}
    for key, value in encoding.items():
        to = getattr(value, "to", None)
        if callable(to):
            placed = to(device)
            moved[key] = value if placed is None else placed
        else:
            moved[key] = value
    return moved


def _publish_universal_inspection(workspace: object, run_key: str, report: object) -> None:
    """Persist one immutable universal topology beside the routing shard."""

    if not isinstance(workspace, str | Path):
        raise TypeError("workspace must be a string or Path")
    from ..adapters import build_universal_inspection

    inspection = build_universal_inspection(report)
    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    directory = root / "inspections"
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("inspection directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{run_key}.json"
    payload = inspection.to_json().encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("published universal inspection conflicts with the run")
        return
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _publish_discovery_report(workspace: object, run_key: str, report: DiscoveryReport) -> None:
    """Persist the exact static report used to derive the routing universe."""

    root = Path(workspace)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError("workspace must be an existing non-symlink directory")
    directory = root / "discoveries"
    if directory.exists() and directory.is_symlink():
        raise RuntimeError("discovery directory must not be a symlink")
    directory.mkdir(exist_ok=True)
    target = directory / f"{run_key}.json"
    payload = report.to_json().encode("utf-8")
    if target.exists():
        if target.is_symlink() or not target.is_file() or target.read_bytes() != payload:
            raise RuntimeError("published discovery report conflicts with the run")
        return
    fd, staged_name = tempfile.mkstemp(dir=str(directory), prefix=f".{run_key}.", suffix=".staging")
    staged = Path(staged_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
        os.replace(staged, target)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


class TransformersRoutingExecutor:
    """One plan-bound, single-run real-model routing executor."""

    def __init__(
        self,
        plan: LoadingPlan,
        *,
        store_token_text: bool = False,
        capture_expert_activity: bool = False,
        capture_routing: bool = True,
    ) -> None:
        if not isinstance(plan, LoadingPlan):
            raise TypeError("plan must be a validated LoadingPlan")
        if type(store_token_text) is not bool:
            raise TypeError("store_token_text must be an exact bool")
        if type(capture_expert_activity) is not bool:
            raise TypeError("capture_expert_activity must be an exact bool")
        if type(capture_routing) is not bool:
            raise TypeError("capture_routing must be an exact bool")
        self._plan = plan
        self._store_token_text = store_token_text
        self._capture_expert_activity = capture_expert_activity
        self._capture_routing = capture_routing
        self._loaded: LoadedModel | None = None
        self._report: DiscoveryReport | None = None
        self._targets: tuple[StructuredRouterTarget, ...] = ()
        self._token_events: list[TokenEvent] = []
        self._routing_events: list[RoutingEvent] = []
        self._expert_events: list[ExpertEvent] = []
        self._notes: set[str] = set()
        self._outputs: list[object] = []
        self._forward_timings_ms: list[float] = []
        self._run_key: str | None = None
        self._published = False

    @property
    def name(self) -> str:
        return "transformers-routing"

    def bind_run_key(self, run_key: str) -> None:
        """Bind the content-addressed run key before the first row executes."""

        if type(run_key) is not str:
            raise TypeError("run_key must be a string")
        validate_stable_identifier(run_key, field_name="run_key")
        if self._token_events:
            raise RuntimeError("the run key cannot be rebound after rows executed")
        self._run_key = run_key

    @property
    def capture_routing(self) -> bool:
        """Whether this executor installs routing/activity capture hooks."""

        return self._capture_routing

    def timing_summary(self) -> dict[str, Any]:
        """Return forward-only timing evidence collected by this executor.

        Timings deliberately exclude model loading, tokenization, persistence,
        and browser/server overhead.  GPU adapters synchronize around the
        forward when CUDA is available so asynchronous kernels are not
        reported as artificially cheap host calls.
        """

        values = tuple(self._forward_timings_ms)
        total = sum(values)
        return {
            "scope": "model_forward",
            "capture_routing": self._capture_routing,
            "successful_rows": len(values),
            "total_ms": total,
            "mean_ms": total / len(values) if values else None,
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
        }

    def _ensure_loaded(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        try:
            from ..runtime.model_loader import load_huggingface, load_local

            if isinstance(self._plan.source, HuggingFaceSource):
                loaded = load_huggingface(self._plan)
            elif isinstance(self._plan.source, LocalSource):
                loaded = load_local(self._plan)
            else:
                raise RowFailure(
                    "dependency",
                    "loading plan source type is not supported by this executor",
                )
            report = scan(loaded.model, loaded.manifest)
        except RowFailure:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RowFailure("dependency", _LOAD_FAILURE) from None
        self._loaded = loaded
        self._report = report
        return loaded

    @staticmethod
    def _synchronize_cuda(model: object) -> None:
        """Synchronize only when the loaded model is running on CUDA."""

        torch = sys.modules.get("torch")
        if torch is None:
            return
        device = _model_input_device(model)
        device_type = getattr(device, "type", None)
        if device_type is None:
            device_type = str(device).split(":", 1)[0] if device is not None else ""
        if device_type != "cuda":
            return
        cuda = getattr(torch, "cuda", None)
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            synchronize(device)

    def _timed_forward(self, model: object, forward: Callable[[], object]) -> object:
        """Run one forward and retain a successful model-forward duration."""

        self._synchronize_cuda(model)
        started = time.perf_counter()
        succeeded = False
        try:
            output = forward()
            succeeded = True
            return output
        finally:
            self._synchronize_cuda(model)
            if succeeded:
                self._forward_timings_ms.append((time.perf_counter() - started) * 1000.0)

    @staticmethod
    def _native_forward(model: object, encoding: Mapping[str, object]) -> object:
        """Execute one model forward without installing any capture hooks."""

        torch = sys.modules.get("torch")
        if torch is None:
            context = nullcontext()
        else:
            inference_mode = getattr(torch, "inference_mode", None)
            context = inference_mode() if callable(inference_mode) else nullcontext()
        with context:
            call = getattr(model, "__call__", None)
            if not callable(call):
                raise TypeError("loaded model is not callable")
            return call(**dict(encoding))

    def __call__(
        self, *, row_index: int, batch_index: int, values: Mapping[str, Any]
    ) -> dict[str, Any]:
        del batch_index
        if self._run_key is None:
            raise RowFailure("validation", _RUN_KEY_FAILURE)
        if not isinstance(values, Mapping) or not isinstance(values.get("prompt"), str):
            raise RowFailure("validation", _PROMPT_FAILURE)
        prompt = values["prompt"]
        if not prompt:
            raise RowFailure("validation", _PROMPT_FAILURE)

        loaded = self._ensure_loaded()
        assert self._report is not None
        tokenizer = loaded.tokenizer
        if tokenizer is None or not callable(tokenizer):
            raise RowFailure("dependency", _TOKENIZER_MISSING_FAILURE)

        sequence_id = f"row-{row_index}"
        token_events, encoding = self._encode(tokenizer, prompt, sequence_id)
        try:
            encoding = _move_model_inputs(loaded.model, encoding)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RowFailure("execution", "model input placement failed") from None
        try:
            if not self._capture_routing:
                self._timed_forward(
                    loaded.model,
                    lambda: self._native_forward(loaded.model, encoding),
                )
                return {
                    "schema_version": EXECUTOR_RESULT_SCHEMA_VERSION,
                    "capture_routing": False,
                    "prompt": prompt,
                    "sequence_id": sequence_id,
                    "token_count": len(token_events),
                }

            targets = structured_router_targets(self._report)
            max_events = len(token_events) * len(targets) * targets[0].routed_top_k
            config = getattr(loaded.model, "config", None)

            def captured_forward() -> StructuredRoutingForwardResult:
                if self._capture_expert_activity:
                    try:
                        return run_structured_expert_forward(
                            loaded.model,
                            self._report,
                            token_events,
                            dict(encoding),
                            max_events=max_events,
                            max_expert_events=max_events,
                            config=config,
                        )
                    except (ProbeResolutionError, StructuredCaptureError):
                        # Routing remains valid evidence when a family dispatches
                        # experts through an opaque fused kernel.  Retry the
                        # routing-only capture and carry the limitation as a
                        # capability note rather than claiming activations.
                        result = run_structured_routing_forward(
                            loaded.model,
                            self._report,
                            token_events,
                            dict(encoding),
                            max_events=max_events,
                            config=config,
                        )
                        return StructuredRoutingForwardResult(
                            output=result.output,
                            token_events=result.token_events,
                            routing_events=result.routing_events,
                            capability_notes=(
                                *result.capability_notes,
                                "expert_activity_unavailable",
                            ),
                        )
                return run_structured_routing_forward(
                    loaded.model,
                    self._report,
                    token_events,
                    dict(encoding),
                    max_events=max_events,
                    config=config,
                )

            result = self._timed_forward(loaded.model, captured_forward)
        except StructuredCaptureError:
            raise
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RowFailure("execution", f"structured forward failed ({type(exc).__name__})")

        self._targets = targets
        self._token_events.extend(result.token_events)
        self._routing_events.extend(result.routing_events)
        self._expert_events.extend(result.expert_events)
        self._notes.update(result.capability_notes)
        self._outputs.append(result.output)
        return {
            "schema_version": EXECUTOR_RESULT_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "prompt": prompt,
            "sequence_id": sequence_id,
            "token_count": len(result.token_events),
            "routing_event_count": len(result.routing_events),
            "capability_notes": sorted(result.capability_notes),
        }

    def _encode(
        self, tokenizer: object, prompt: str, sequence_id: str
    ) -> tuple[tuple[TokenEvent, ...], dict[str, object]]:
        assert self._run_key is not None
        try:
            encoded = tokenizer(  # type: ignore[operator]
                prompt,
                padding=False,
                truncation=False,
                return_attention_mask=True,
                return_token_type_ids=False,
                return_tensors="pt",
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RowFailure("execution", _TOKENIZER_CALL_FAILURE) from None
        try:
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise ValueError(_TOKENIZER_SHAPE_FAILURE)
            copied = {
                "input_ids": encoded["input_ids"],
                **(
                    {"attention_mask": encoded["attention_mask"]}
                    if "attention_mask" in encoded
                    else {}
                ),
            }
            ids = _materialize_ids(copied["input_ids"])
            converter = getattr(tokenizer, "convert_ids_to_tokens", None)
            if not callable(converter):
                raise ValueError("tokenizer does not expose convert_ids_to_tokens")
            pieces = converter(ids)
            if (
                type(pieces) is not list
                or len(pieces) != len(ids)
                or any(type(piece) is not str for piece in pieces)
            ):
                raise ValueError("converted token pieces must be an exact string list")
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            raise RowFailure("execution", _TOKENIZER_SHAPE_FAILURE) from None
        token_events = tuple(
            TokenEvent.model_validate(
                {
                    "run_key": self._run_key,
                    "sequence_id": sequence_id,
                    "token_pos": position,
                    "token_id": token_id,
                    "token_text": piece,
                    "phase": TokenPhase.PREFILL.value,
                }
            )
            for position, (token_id, piece) in enumerate(zip(ids, pieces, strict=True))
        )
        return token_events, copied

    def publish_run_artifacts(self, workspace: object) -> object | None:
        """Append accumulated events as one shard and reconcile the catalog.

        Returns the shard receipt, or ``None`` when no row produced events.
        """

        if self._published:
            raise RuntimeError("executor artifacts were already published")
        if not self._token_events:
            self._release()
            return None
        if self._run_key is None:
            raise RuntimeError(_RUN_KEY_FAILURE)
        self._published = True

        ordered_events = self._ordered_storage_events()
        from ..runtime.generic_capture import StructuredRoutingForwardResult

        result = StructuredRoutingForwardResult(
            output=self._outputs[-1] if self._outputs else object(),
            token_events=tuple(self._token_events),
            routing_events=ordered_events,
            expert_events=tuple(self._expert_events),
        )
        from ..store import append_structured_shard, rebuild_catalog

        receipt = append_structured_shard(
            workspace,
            result,
            store_token_text=self._store_token_text,
        )
        if self._report is None:
            raise RuntimeError("routing discovery report was not retained for publication")
        _publish_discovery_report(workspace, self._run_key, self._report)
        _publish_universal_inspection(workspace, self._run_key, self._report)
        rebuild_catalog(workspace, at=None)
        self._release()
        return receipt

    def _ordered_storage_events(self) -> tuple[RoutingEvent, ...]:
        layer_order = {
            target.layer_key: position
            for position, target in enumerate(sorted(self._targets, key=lambda t: t.layer_index))
        }
        token_positions = {token.token_key: index for index, token in enumerate(self._token_events)}
        return tuple(
            sorted(
                self._routing_events,
                key=lambda event: (
                    layer_order[event.layer_key],
                    token_positions[event.token_key],
                    event.rank,
                ),
            )
        )

    def _release(self) -> None:
        loaded = self._loaded
        self._loaded = None
        if loaded is None:
            return
        try:
            loaded.close()
        except Exception:
            pass  # publication succeeded; cleanup problems never replace it

    def close(self) -> None:
        """Release a loaded model when execution ends before publication."""

        self._release()


__all__ = ["EXECUTOR_RESULT_SCHEMA_VERSION", "TransformersRoutingExecutor"]
