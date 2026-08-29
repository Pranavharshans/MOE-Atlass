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

import sys
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from typing import Any

from ..core import ComponentKind, stable_digest, validate_stable_identifier
from ..discovery import DiscoveryReport, scan
from ..evaluation import EvaluationMethod, evaluate_text
from ..events import EVENT_SCHEMA_VERSION, ExpertEvent, RoutingEvent, TokenEvent, TokenPhase
from ..loading import HuggingFaceSource, LoadingPlan, LocalSource
from ..probe import (
    CaptureMode,
    CapturePolicy,
    HookBinding,
    HookManager,
    HookPoint,
    ProbeLevel,
    ProbePlan,
    ProbeResolutionError,
    ProbeTarget,
    ReductionPolicy,
)
from ..runtime.contracts import LoadedModel
from ..runtime.generic_capture import (
    StructuredCaptureError,
    StructuredRouterTarget,
    StructuredRoutingForwardResult,
    decode_structured_payload,
    run_structured_expert_forward,
    run_structured_routing_forward,
    structured_router_targets,
)
from ..runtime.memory import release_accelerator_memory
from ..services.run_engine import RowFailure
from .transformers_support import (
    _TOKENIZER_SHAPE_FAILURE,
    _materialize_ids,
    _model_input_device,
    _move_model_inputs,
    _publish_discovery_report,
    _publish_universal_inspection,
    _safe_validation_error,
)

EXECUTOR_RESULT_SCHEMA_VERSION = "1.0"

_TOKENIZER_CALL_FAILURE = "tokenizer call failed"
_PROMPT_FAILURE = "row values must carry a non-empty string 'prompt'"
_TOKENIZER_MISSING_FAILURE = "loaded model did not resolve a tokenizer"
_LOAD_FAILURE = "model loading failed before execution"
_RUN_KEY_FAILURE = "executor run key was not bound before execution"
_GENERATION_CAPTURE_FAILURE = (
    "model generation does not expose a compatible top-level forward pre-hook"
)


class TransformersRoutingExecutor:
    """One plan-bound, single-run real-model routing executor."""

    def __init__(
        self,
        plan: LoadingPlan,
        *,
        store_token_text: bool = False,
        capture_expert_activity: bool = False,
        capture_routing: bool = True,
        mode: str = "generation",
        max_new_tokens: int = 128,
        thinking_mode: str = "model_default",
        evaluation_method: EvaluationMethod | str = EvaluationMethod.EXACT_MATCH,
        load_progress: Callable[[str, int, int, str], None] | None = None,
    ) -> None:
        if not isinstance(plan, LoadingPlan):
            raise TypeError("plan must be a validated LoadingPlan")
        if type(store_token_text) is not bool:
            raise TypeError("store_token_text must be an exact bool")
        if type(capture_expert_activity) is not bool:
            raise TypeError("capture_expert_activity must be an exact bool")
        if type(capture_routing) is not bool:
            raise TypeError("capture_routing must be an exact bool")
        if mode not in {"generation", "teacher_forced"}:
            raise ValueError("mode must be generation or teacher_forced")
        if (
            type(max_new_tokens) is not int
            or isinstance(max_new_tokens, bool)
            or max_new_tokens <= 0
        ):
            raise TypeError("max_new_tokens must be a strict positive integer")
        if thinking_mode not in {"model_default", "disabled", "enabled"}:
            raise ValueError("thinking_mode must be model_default, disabled, or enabled")
        try:
            resolved_evaluation = (
                evaluation_method
                if isinstance(evaluation_method, EvaluationMethod)
                else EvaluationMethod(evaluation_method)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown evaluation method") from exc
        self._plan = plan
        self._store_token_text = store_token_text
        self._capture_expert_activity = capture_expert_activity
        self._capture_routing = capture_routing
        self._mode = mode
        self._max_new_tokens = max_new_tokens
        self._thinking_mode = thinking_mode
        self._evaluation_method = resolved_evaluation
        if load_progress is not None and not callable(load_progress):
            raise TypeError("load_progress must be callable")
        self._load_progress = load_progress
        self._loaded: LoadedModel | None = None
        self._report: DiscoveryReport | None = None
        self._targets: tuple[StructuredRouterTarget, ...] = ()
        self._token_events: list[TokenEvent] = []
        self._routing_events: list[RoutingEvent] = []
        self._expert_events: list[ExpertEvent] = []
        self._notes: set[str] = set()
        self._outputs: list[object] = []
        self._forward_timings_ms: list[float] = []
        self._generation_timings_ms: list[float] = []
        self._backend_handshake: dict[str, Any] | None = None
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
        generation = tuple(self._generation_timings_ms)
        generation_total = sum(generation)
        return {
            "scope": "model_forward",
            "capture_routing": self._capture_routing,
            "successful_rows": len(values),
            "total_ms": total,
            "mean_ms": total / len(values) if values else None,
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None,
            "generation_total_ms": generation_total if generation else None,
            "generation_mean_ms": generation_total / len(generation) if generation else None,
        }

    def intervention_inventory(self) -> tuple[dict[str, Any], ...]:
        """Load once and expose the structure-bound routed-expert coordinates."""

        from ..interventions import intervention_targets

        self._ensure_loaded()
        assert self._report is not None
        return tuple(target.to_dict() for target in intervention_targets(self._report))

    def run_with_intervention(
        self, recipe: object, execute: Callable[[], Any]
    ) -> tuple[Any, Any, dict[str, int]]:
        """Execute one caller-owned run under temporary expert hooks.

        The shared intervention engine owns restoration.  The loaded model is
        prepared before mutation, so every row executed by ``execute`` observes
        the same temporary hooks and the model is clean before publication.
        """

        from ..interventions import (
            InterventionRecipe,
            TransformersExpertInterventionCapability,
            run_intervention,
        )

        if type(recipe) is not InterventionRecipe:
            raise TypeError("recipe must be an InterventionRecipe")
        if not callable(execute):
            raise TypeError("execute must be callable")
        loaded = self._ensure_loaded()
        assert self._report is not None
        capability = TransformersExpertInterventionCapability(self._report)
        observed: list[Any] = []

        def observe(_module: object) -> None:
            observed.append(execute())

        outcome = run_intervention(loaded.model, recipe, capability, observe)
        if len(observed) != 1:
            raise RuntimeError("intervention execution did not return exactly one observation")
        return observed[0], outcome, capability.invocation_counts

    def _ensure_loaded(self) -> LoadedModel:
        if self._loaded is not None:
            return self._loaded
        loaded: LoadedModel | None = None
        try:
            from ..runtime.model_loader import load_huggingface, load_local

            if isinstance(self._plan.source, HuggingFaceSource):
                loaded = (
                    load_huggingface(
                        self._plan,
                        progress_callback=self._load_progress,
                    )
                    if self._load_progress is not None
                    else load_huggingface(self._plan)
                )
            elif isinstance(self._plan.source, LocalSource):
                loaded = load_local(self._plan)
            else:
                raise RowFailure(
                    "dependency",
                    "loading plan source type is not supported by this executor",
                )
            report = scan(loaded.model, loaded.manifest)
        except RowFailure:
            if loaded is not None:
                try:
                    loaded.close()
                except Exception:
                    pass
            release_accelerator_memory()
            raise
        except (KeyboardInterrupt, SystemExit):
            if loaded is not None:
                try:
                    loaded.close()
                except Exception:
                    pass
            release_accelerator_memory()
            raise
        except Exception:
            if loaded is not None:
                try:
                    loaded.close()
                except Exception:
                    pass
            release_accelerator_memory()
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

    def _run_with_backend_handshake(
        self,
        model: object,
        execute: Callable[[], Any],
    ) -> Any:
        """Exercise one identity delegate on the first successful run forward."""

        if self._backend_handshake is not None:
            return execute()
        from ..interventions import run_huggingface_expert_handshake

        result, report = run_huggingface_expert_handshake(model, execute)
        self._backend_handshake = report.to_dict()
        self._notes.add(f"expert_backend_handshake:{report.status.value}")
        return result

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

    @staticmethod
    def _prediction_ids(output: object) -> list[int]:
        """Extract bounded argmax predictions from common causal-LM outputs."""

        logits = (
            output.get("logits") if isinstance(output, Mapping) else getattr(output, "logits", None)
        )
        if logits is None and isinstance(output, tuple) and output:
            logits = output[0]
        if logits is None:
            return []
        argmax = getattr(logits, "argmax", None)
        if callable(argmax):
            try:
                return _materialize_ids(argmax(dim=-1))[-256:]
            except Exception:
                return []
        if type(logits) is list and len(logits) == 1 and type(logits[0]) is list:
            rows = logits[0]
            predictions: list[int] = []
            for row in rows[-256:]:
                if type(row) is not list or not row:
                    return []
                predictions.append(max(range(len(row)), key=row.__getitem__))
            return predictions
        return []

    def _generate_output(
        self,
        model: object,
        tokenizer: object,
        encoding: Mapping[str, object],
        *,
        prompt_token_count: int,
    ) -> tuple[list[int], str | None]:
        generate = getattr(model, "generate", None)
        if not callable(generate):
            return [], None
        torch = sys.modules.get("torch")
        inference_mode = getattr(torch, "inference_mode", None) if torch is not None else None
        context = inference_mode() if callable(inference_mode) else nullcontext()
        self._synchronize_cuda(model)
        started = time.perf_counter()
        succeeded = False
        try:
            with context:
                generated = generate(
                    **dict(encoding),
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            ids = _materialize_ids(generated)
            continuation = ids[prompt_token_count:]
            decode = getattr(tokenizer, "decode", None)
            text = (
                decode(continuation, skip_special_tokens=True)
                if continuation and callable(decode)
                else None
            )
            succeeded = True
            return continuation, text if isinstance(text, str) else None
        finally:
            self._synchronize_cuda(model)
            if succeeded:
                elapsed = (time.perf_counter() - started) * 1000.0
                self._generation_timings_ms.append(elapsed)

    def _tokens_for_generation_step(
        self,
        tokenizer: object,
        ids: list[int],
        positions: list[int],
        *,
        sequence_id: str,
        prompt_token_count: int,
    ) -> tuple[TokenEvent, ...]:
        """Build canonical token identities for one observed generation forward."""

        if self._run_key is None:
            raise StructuredCaptureError("generation", _RUN_KEY_FAILURE)
        if len(ids) != len(positions):
            raise StructuredCaptureError(
                "generation", "generation input positions do not match input token ids"
            )
        converter = getattr(tokenizer, "convert_ids_to_tokens", None)
        if not callable(converter):
            raise StructuredCaptureError(
                "generation", "tokenizer does not expose convert_ids_to_tokens"
            )
        pieces = converter(ids)
        if isinstance(pieces, str) and len(ids) == 1:
            pieces = [pieces]
        if (
            type(pieces) is not list
            or len(pieces) != len(ids)
            or any(type(piece) is not str for piece in pieces)
        ):
            raise StructuredCaptureError(
                "generation", "generated token pieces must be an exact string list"
            )
        return tuple(
            TokenEvent.model_validate(
                {
                    "run_key": self._run_key,
                    "sequence_id": sequence_id,
                    "token_pos": position,
                    "token_id": token_id,
                    "token_text": piece,
                    "phase": (
                        TokenPhase.PREFILL.value
                        if position < prompt_token_count
                        else TokenPhase.DECODE.value
                    ),
                }
            )
            for token_id, piece, position in zip(ids, pieces, positions, strict=True)
        )

    def _generate_with_routing_capture(
        self,
        model: object,
        tokenizer: object,
        encoding: Mapping[str, object],
        prompt_tokens: tuple[TokenEvent, ...],
        targets: tuple[StructuredRouterTarget, ...],
        *,
        sequence_id: str,
        config: object,
    ) -> tuple[StructuredRoutingForwardResult, list[int], str | None]:
        """Trace every model forward used by deterministic generation.

        A top-level pre-hook binds each router invocation to the exact input
        tokens of that generation step. Cached decode calls normally carry one
        token; uncached implementations may replay the full prefix, in which
        case only positions not already published are retained. The final
        generated token has no subsequent model forward and therefore has no
        routing row, which is recorded explicitly in the capability notes.
        """

        generate = getattr(model, "generate", None)
        register_pre_hook = getattr(model, "register_forward_pre_hook", None)
        if not callable(generate) or not callable(register_pre_hook):
            raise StructuredCaptureError("generation", _GENERATION_CAPTURE_FAILURE)
        prompt_ids = [token.token_id for token in prompt_tokens]
        if not prompt_ids:
            raise StructuredCaptureError("generation", "generation prompt is empty")
        top_k = targets[0].routed_top_k
        max_events = (len(prompt_ids) + self._max_new_tokens) * len(targets) * top_k
        target_paths = {target.module_path for target in targets}
        published_positions: set[int] = set()
        observed_tokens: list[TokenEvent] = []
        observed_routes: list[RoutingEvent] = []
        notes: set[str] = {"generation_routing_excludes_terminal_output_token"}
        current_tokens: tuple[TokenEvent, ...] = ()
        current_new_keys: set[str] = set()
        current_paths: set[str] = set()
        forward_index = 0

        def finish_step() -> None:
            if not current_tokens:
                return
            missing = sorted(target_paths.difference(current_paths))
            if missing:
                raise StructuredCaptureError(
                    "generation", f"routers did not fire during generation forward: {missing}"
                )

        def before_forward(
            _module: object, args: tuple[object, ...], kwargs: Mapping[str, object]
        ) -> None:
            nonlocal current_tokens, current_new_keys, current_paths, forward_index
            finish_step()
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is None:
                raise StructuredCaptureError(
                    "generation", "generation forward did not expose input_ids"
                )
            try:
                ids = _materialize_ids(input_ids)
            except Exception as exc:
                raise StructuredCaptureError(
                    "generation", "generation input_ids must be shaped exactly (1, N)"
                ) from exc
            if forward_index == 0:
                if ids != prompt_ids:
                    raise StructuredCaptureError(
                        "generation", "first generation forward does not match the prompt ids"
                    )
                positions = list(range(len(ids)))
            elif len(ids) == 1:
                positions = [max(published_positions, default=len(prompt_ids) - 1) + 1]
            else:
                positions = list(range(len(ids)))
                if ids[: len(prompt_ids)] != prompt_ids:
                    raise StructuredCaptureError(
                        "generation", "uncached generation replay does not match the prompt ids"
                    )
            current_tokens = self._tokens_for_generation_step(
                tokenizer,
                ids,
                positions,
                sequence_id=sequence_id,
                prompt_token_count=len(prompt_ids),
            )
            fresh = tuple(
                token for token in current_tokens if token.token_pos not in published_positions
            )
            if not fresh:
                raise StructuredCaptureError(
                    "generation", "generation forward did not introduce a new routed token"
                )
            current_new_keys = {token.token_key for token in fresh}
            observed_tokens.extend(fresh)
            published_positions.update(token.token_pos for token in fresh)
            current_paths = set()
            forward_index += 1

        def callback_for(target: StructuredRouterTarget):
            def callback(_module: object, _inputs: object, output: object) -> None:
                if not current_tokens:
                    raise StructuredCaptureError(
                        "generation", "router fired before the generation input was observed"
                    )
                if target.module_path in current_paths:
                    raise StructuredCaptureError(
                        "generation",
                        f"router {target.module_path!r} fired more than once in one forward",
                    )
                events, note = decode_structured_payload(
                    output,
                    target=target,
                    token_events=current_tokens,
                    config=config,
                )
                filtered = [event for event in events if event.token_key in current_new_keys]
                expected = len(current_new_keys) * target.routed_top_k
                if len(filtered) != expected:
                    raise StructuredCaptureError(
                        "generation", "generation router payload did not cover new token positions"
                    )
                observed_routes.extend(filtered)
                current_paths.add(target.module_path)
                if note:
                    notes.add(note)

            return callback

        plan = ProbePlan(
            level=ProbeLevel.ROUTING,
            hook_points=(HookPoint.FORWARD,),
            targets=tuple(
                ProbeTarget(
                    module_path=target.module_path,
                    component_key=target.component_key,
                    component_kind=ComponentKind.ROUTER,
                )
                for target in targets
            ),
            capture=CapturePolicy(mode=CaptureMode.STATS, reduction=ReductionPolicy.COUNTS),
        )
        callbacks = {
            HookBinding(target.module_path, HookPoint.FORWARD): callback_for(target)
            for target in targets
        }
        manager = HookManager(model, plan, callbacks)
        pre_handle: object | None = None
        generated: object = None
        self._synchronize_cuda(model)
        started = time.perf_counter()
        try:
            manager.__enter__()
            try:
                pre_handle = register_pre_hook(before_forward, with_kwargs=True)
            except TypeError as exc:
                raise StructuredCaptureError("generation", _GENERATION_CAPTURE_FAILURE) from exc
            if not callable(getattr(pre_handle, "remove", None)):
                raise StructuredCaptureError(
                    "generation", "generation pre-hook did not return a removable handle"
                )
            torch = sys.modules.get("torch")
            inference_mode = getattr(torch, "inference_mode", None) if torch is not None else None
            context = inference_mode() if callable(inference_mode) else nullcontext()
            with context:
                generated = generate(
                    **dict(encoding),
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )
            finish_step()
        finally:
            if pre_handle is not None:
                try:
                    pre_handle.remove()  # type: ignore[union-attr]
                except Exception:
                    pass
            try:
                manager.close()
            finally:
                self._synchronize_cuda(model)
        if forward_index == 0:
            raise StructuredCaptureError("generation", "model.generate performed no model forward")
        elapsed = (time.perf_counter() - started) * 1000.0
        self._forward_timings_ms.append(elapsed)
        self._generation_timings_ms.append(elapsed)
        if len(observed_routes) > max_events:
            raise StructuredCaptureError(
                "generation", "generation routing exceeded its event budget"
            )
        generated_ids = _materialize_ids(getattr(generated, "sequences", generated))
        continuation = generated_ids[len(prompt_ids) :]
        decode = getattr(tokenizer, "decode", None)
        text = (
            decode(continuation, skip_special_tokens=True)
            if continuation and callable(decode)
            else None
        )
        result = StructuredRoutingForwardResult(
            output=generated,
            token_events=tuple(observed_tokens),
            routing_events=tuple(observed_routes),
            capability_notes=tuple(sorted(notes)),
        )
        return result, continuation, text if isinstance(text, str) else None

    def _output_evidence(
        self,
        model: object,
        tokenizer: object,
        encoding: Mapping[str, object],
        forward_output: object,
        *,
        prompt_token_count: int,
        reference: object,
        captured_generation: tuple[list[int], str | None] | None = None,
    ) -> dict[str, Any]:
        generation_before = len(self._generation_timings_ms)
        generated_ids: list[int] = []
        generated_text: str | None = None
        if captured_generation is not None:
            generated_ids, generated_text = captured_generation
        elif self._mode == "generation":
            generated_ids, generated_text = self._generate_output(
                model,
                tokenizer,
                encoding,
                prompt_token_count=prompt_token_count,
            )
        prediction_ids = generated_ids or self._prediction_ids(forward_output)
        digest = (
            f"sha256:{stable_digest({'token_ids': prediction_ids, 'mode': self._mode})}"
            if prediction_ids
            else None
        )
        evidence: dict[str, Any] = {
            "output_digest": digest,
            "output_token_count": len(prediction_ids),
            "output_mode": (
                "generated"
                if generated_ids
                else "forward_argmax"
                if prediction_ids
                else "unavailable"
            ),
            "generation_ms": (
                self._generation_timings_ms[-1]
                if self._generation_timings_ms
                and (
                    captured_generation is not None
                    or len(self._generation_timings_ms) > generation_before
                )
                else None
            ),
            "score_name": None,
            "task_score": None,
        }
        if self._store_token_text and generated_text is not None:
            evidence["output_preview"] = generated_text[:2048]
        if reference is not None and generated_text is not None:
            evaluation = evaluate_text(
                generated_text,
                reference,
                self._evaluation_method,
            )
            evidence["score_name"] = evaluation.method.value
            evidence["task_score"] = evaluation.score
        return evidence

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
                self._run_with_backend_handshake(
                    loaded.model,
                    lambda: self._timed_forward(
                        loaded.model,
                        lambda: self._native_forward(loaded.model, encoding),
                    ),
                )
                return {
                    "schema_version": EXECUTOR_RESULT_SCHEMA_VERSION,
                    "capture_routing": False,
                    "prompt": prompt,
                    "sequence_id": sequence_id,
                    "token_count": len(token_events),
                    "expert_backend_handshake": self._backend_handshake,
                }

            targets = structured_router_targets(self._report)
            max_events = len(token_events) * len(targets) * targets[0].routed_top_k
            config = getattr(loaded.model, "config", None)
            captured_generation: tuple[list[int], str | None] | None = None

            if self._mode == "generation" and callable(getattr(loaded.model, "generate", None)):
                result, generated_ids, generated_text = self._run_with_backend_handshake(
                    loaded.model,
                    lambda: self._generate_with_routing_capture(
                        loaded.model,
                        tokenizer,
                        encoding,
                        token_events,
                        targets,
                        sequence_id=sequence_id,
                        config=config,
                    ),
                )
                captured_generation = (generated_ids, generated_text)
                if self._capture_expert_activity:
                    result = StructuredRoutingForwardResult(
                        output=result.output,
                        token_events=result.token_events,
                        routing_events=result.routing_events,
                        capability_notes=tuple(
                            sorted(
                                {
                                    *result.capability_notes,
                                    "generation_expert_activity_unavailable",
                                }
                            )
                        ),
                    )
            else:

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

                result = self._run_with_backend_handshake(
                    loaded.model,
                    lambda: self._timed_forward(loaded.model, captured_forward),
                )
        except StructuredCaptureError as exc:
            # StructuredCaptureError messages are deliberately bounded and
            # stage-labelled. Preserve that safe diagnostic as row evidence
            # instead of letting the run engine collapse it to the exception
            # class name alone.
            raise RowFailure("execution", str(exc)) from None
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise RowFailure("execution", _safe_validation_error(exc)) from None

        self._targets = targets
        self._token_events.extend(result.token_events)
        self._routing_events.extend(result.routing_events)
        self._expert_events.extend(result.expert_events)
        self._notes.update(result.capability_notes)
        self._outputs.append(result.output)
        output_evidence = self._output_evidence(
            loaded.model,
            tokenizer,
            encoding,
            result.output,
            prompt_token_count=len(result.token_events),
            reference=values.get("reference"),
            captured_generation=captured_generation,
        )
        input_digest = "sha256:" + stable_digest(
            {
                "evaluation_method": self._evaluation_method.value,
                "max_new_tokens": self._max_new_tokens,
                "mode": self._mode,
                "thinking_mode": self._thinking_mode,
                "reference": str(values["reference"]) if "reference" in values else None,
                "token_ids": [token.token_id for token in token_events],
            }
        )
        return {
            "schema_version": EXECUTOR_RESULT_SCHEMA_VERSION,
            "event_schema_version": EVENT_SCHEMA_VERSION,
            "prompt": prompt,
            "sequence_id": sequence_id,
            "token_count": len(result.token_events),
            "prefill_token_count": sum(
                token.phase is TokenPhase.PREFILL for token in result.token_events
            ),
            "decode_token_count": sum(
                token.phase is TokenPhase.DECODE for token in result.token_events
            ),
            "routing_scope": (
                "actual_generation" if captured_generation is not None else "single_forward"
            ),
            "input_digest": input_digest,
            "evaluation_method": self._evaluation_method.value,
            "routing_event_count": len(result.routing_events),
            "capability_notes": sorted(result.capability_notes),
            "forward_ms": self._forward_timings_ms[-1],
            "expert_backend_handshake": self._backend_handshake,
            **output_evidence,
        }

    def _encode(
        self, tokenizer: object, prompt: str, sequence_id: str
    ) -> tuple[tuple[TokenEvent, ...], dict[str, object]]:
        assert self._run_key is not None
        try:
            rendered_prompt = prompt
            add_special_tokens = True
            if self._mode == "generation" and self._thinking_mode != "model_default":
                apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
                if not callable(apply_chat_template):
                    raise ValueError("tokenizer does not support explicit thinking control")
                rendered_prompt = apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self._thinking_mode == "enabled",
                )
                if not isinstance(rendered_prompt, str) or not rendered_prompt:
                    raise ValueError("chat template did not return rendered text")
                add_special_tokens = False
            encoded = tokenizer(  # type: ignore[operator]
                rendered_prompt,
                add_special_tokens=add_special_tokens,
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
        from ..services import reconcile_run_shards
        from ..store import append_structured_shard

        receipt = append_structured_shard(
            workspace,
            result,
            store_token_text=self._store_token_text,
        )
        if self._report is None:
            raise RuntimeError("routing discovery report was not retained for publication")
        _publish_discovery_report(workspace, self._run_key, self._report)
        _publish_universal_inspection(workspace, self._run_key, self._report)
        reconcile_run_shards(workspace, self._run_key, at=None)
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
        finally:
            release_accelerator_memory()

    def close(self) -> None:
        """Release a loaded model when execution ends before publication."""

        self._release()


__all__ = ["EXECUTOR_RESULT_SCHEMA_VERSION", "TransformersRoutingExecutor"]
