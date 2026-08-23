"""One bounded plain-text Mixtral prompt-prefill execution seam."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from ..adapters import AdapterInspection, build_routing_probe_plan
from ..core import ModelManifest, validate_stable_identifier
from ..events import TokenEvent
from ..probe import ProbePlan
from .contracts import LoadedModel
from .routing_forward import RoutingForwardResult, run_mixtral_routing_forward

_STAGES = frozenset({"tokenize", "encoding"})
_ADAPTER_NAME = "huggingface-mixtral-static"
_ADAPTER_VERSION = "1.0"
_ADAPTER_FAMILIES = ("mixtral",)


class MixtralPromptPrefillError(RuntimeError):
    """Fixed-stage error for tokenizer and encoding work only."""

    def __init__(self, stage: Literal["tokenize", "encoding"]) -> None:
        if stage not in _STAGES:
            raise ValueError("mixtral prompt prefill error stage is not supported")
        self.stage = stage
        super().__init__(f"Mixtral prompt prefill failed at {stage}")


def _stage_error(
    stage: Literal["tokenize", "encoding"], cause: BaseException
) -> MixtralPromptPrefillError:
    error = MixtralPromptPrefillError(stage)
    error.__cause__ = cause
    return error


def _positive_budget(value: object, name: str) -> int:
    if type(value) is not int or isinstance(value, bool):
        raise TypeError(f"{name} must be a strict positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a strict positive integer")
    return value


def _fresh_loaded_manifest(loaded: LoadedModel) -> ModelManifest:
    if type(loaded.manifest) is not ModelManifest:
        raise TypeError("loaded.manifest must be an exact ModelManifest")
    fresh = ModelManifest.model_validate(loaded.manifest.model_dump(mode="json"))
    if type(fresh) is not ModelManifest or fresh is loaded.manifest:
        raise TypeError("loaded manifest revalidation returned an unexpected type")
    if fresh.model_dump(mode="json") != loaded.manifest.model_dump(mode="json"):
        raise ValueError("loaded manifest JSON changed during revalidation")
    return fresh


def _fresh_inspection(value: object) -> AdapterInspection:
    if type(value) is not AdapterInspection:
        raise TypeError("inspection must be an exact AdapterInspection")
    fresh = AdapterInspection.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not AdapterInspection or fresh is value:
        raise TypeError("inspection revalidation returned an unexpected type")
    if fresh.model_dump(mode="json") != value.model_dump(mode="json"):
        raise ValueError("inspection JSON changed during revalidation")
    return fresh


def _fresh_plan(value: object) -> ProbePlan:
    if type(value) is not ProbePlan:
        raise TypeError("plan must be an exact ProbePlan")
    fresh = ProbePlan.model_validate(value.model_dump(mode="json"))
    if type(fresh) is not ProbePlan or fresh is value:
        raise TypeError("plan revalidation returned an unexpected type")
    if fresh.model_dump(mode="json") != value.model_dump(mode="json"):
        raise ValueError("plan JSON changed during revalidation")
    return fresh


def _preflight(
    loaded: object,
    inspection: object,
    plan: object,
    prompt: object,
    *,
    run_key: object,
    sequence_id: object,
    add_special_tokens: object,
    max_prompt_chars: object,
    max_tokens: object,
    max_events: object,
) -> tuple[
    LoadedModel,
    ModelManifest,
    AdapterInspection,
    ProbePlan,
    object,
    object,
    int,
    int,
    int,
]:
    if type(loaded) is not LoadedModel:
        raise TypeError("loaded must be an exact LoadedModel")
    if loaded.closed:
        raise ValueError("loaded must be open")
    if loaded.model is None or not callable(loaded.model):
        raise ValueError("loaded.model must be a non-None callable")
    if loaded.tokenizer is None or not callable(loaded.tokenizer):
        raise ValueError("loaded.tokenizer must be a non-None callable")
    manifest = _fresh_loaded_manifest(loaded)
    fresh_inspection = _fresh_inspection(inspection)
    fresh_plan = _fresh_plan(plan)
    if (
        fresh_inspection.descriptor.name != _ADAPTER_NAME
        or fresh_inspection.descriptor.version != _ADAPTER_VERSION
        or fresh_inspection.descriptor.architecture_families != _ADAPTER_FAMILIES
    ):
        raise ValueError("inspection descriptor is not the exact Mixtral static descriptor")
    if manifest.model_dump(mode="json") != fresh_inspection.report.model_manifest.model_dump(
        mode="json"
    ):
        raise ValueError("inspection report manifest does not equal loaded manifest")
    canonical_plan = build_routing_probe_plan(fresh_inspection)
    if (
        fresh_plan != canonical_plan
        or fresh_plan.model_dump(mode="json") != canonical_plan.model_dump(mode="json")
        or fresh_plan.to_json() != canonical_plan.to_json()
        or fresh_plan.plan_id != canonical_plan.plan_id
    ):
        raise ValueError("supplied plan is not the canonical routing plan")
    if type(prompt) is not str:
        raise TypeError("prompt must be an exact string")
    if not prompt:
        raise ValueError("prompt must be non-empty")
    if type(run_key) is not str:
        raise TypeError("run_key must be an exact string")
    if type(sequence_id) is not str:
        raise TypeError("sequence_id must be an exact string")
    validate_stable_identifier(run_key, field_name="run_key")
    validate_stable_identifier(sequence_id, field_name="sequence_id")
    if type(add_special_tokens) is not bool:
        raise TypeError("add_special_tokens must be an exact bool")
    prompt_budget = _positive_budget(max_prompt_chars, "max_prompt_chars")
    token_budget = _positive_budget(max_tokens, "max_tokens")
    event_budget = _positive_budget(max_events, "max_events")
    if len(prompt) > prompt_budget:
        raise ValueError("prompt exceeds max_prompt_chars")
    fresh_loaded_tokenizer = loaded.tokenizer
    routed_top_k = fresh_inspection.report.facts.routed_top_k
    if type(routed_top_k) is not int or isinstance(routed_top_k, bool) or routed_top_k <= 0:
        raise ValueError("inspection routed_top_k must be a strict positive integer")
    return (
        loaded,
        manifest,
        fresh_inspection,
        fresh_plan,
        fresh_loaded_tokenizer,
        prompt_budget,
        token_budget,
        event_budget,
        routed_top_k,
    )


def _shape(value: object) -> tuple[int, int]:
    shape = tuple(value.shape)  # type: ignore[attr-defined]
    if len(shape) != 2 or any(type(item) is not int or isinstance(item, bool) for item in shape):
        raise ValueError("encoded tensor shape must be exact (1, N)")
    if shape[0] != 1 or shape[1] <= 0:
        raise ValueError("encoded tensor shape must be exact (1, N)")
    return shape


def _materialize(value: object) -> object:
    detached = value.detach()  # type: ignore[attr-defined]
    cpu = detached.cpu()
    return cpu.tolist()


def _resolve_converter(tokenizer: object) -> object:
    """Resolve the tokenizer converter without invoking tokenization."""
    try:
        converter = getattr(tokenizer, "convert_ids_to_tokens")
    except (KeyboardInterrupt, SystemExit):
        raise
    except MixtralPromptPrefillError as exc:
        raise _stage_error("encoding", exc)
    except Exception as exc:
        raise _stage_error("encoding", exc)
    if not callable(converter):
        raise _stage_error("encoding", TypeError("convert_ids_to_tokens must be callable"))
    return converter


def _encode(
    tokenizer: object,
    prompt: str,
    *,
    add_special_tokens: bool,
    max_tokens: int,
    max_events: int,
    target_count: int,
    routed_top_k: int,
    converter: object,
    run_key: str,
    sequence_id: str,
) -> tuple[tuple[TokenEvent, ...], dict[str, object]]:
    try:
        encoded = tokenizer(  # type: ignore[operator]
            prompt,
            add_special_tokens=add_special_tokens,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _stage_error("tokenize", exc)
    try:
        if not isinstance(encoded, Mapping) or set(encoded) != {"input_ids", "attention_mask"}:
            raise ValueError("tokenizer output keys must be exactly input_ids and attention_mask")
        copied_encoding = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        input_shape = _shape(copied_encoding["input_ids"])
        mask_shape = _shape(copied_encoding["attention_mask"])
        if mask_shape != input_shape:
            raise ValueError("input_ids and attention_mask shapes must match")
        token_count = input_shape[1]
        if token_count > max_tokens:
            raise ValueError("encoded token count exceeds max_tokens")
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _stage_error("encoding", exc)
    expected_events = token_count * target_count * routed_top_k
    if max_events < expected_events:
        raise ValueError("max_events is insufficient for complete routing capture")
    try:
        ids_rows = _materialize(copied_encoding["input_ids"])
        mask_rows = _materialize(copied_encoding["attention_mask"])
        if (
            type(ids_rows) is not list
            or len(ids_rows) != 1
            or type(ids_rows[0]) is not list
            or len(ids_rows[0]) != token_count
            or type(mask_rows) is not list
            or len(mask_rows) != 1
            or type(mask_rows[0]) is not list
            or len(mask_rows[0]) != token_count
        ):
            raise ValueError("materialized tensors must be exact nested lists")
        ids = ids_rows[0]
        masks = mask_rows[0]
        if any(type(value) is not int or isinstance(value, bool) or value < 0 for value in ids):
            raise ValueError("input_ids must contain non-negative strict integers")
        if any(type(value) is not int or isinstance(value, bool) or value != 1 for value in masks):
            raise ValueError("attention_mask must contain exact one values")
        pieces = converter(ids)  # type: ignore[operator]
        if (
            type(pieces) is not list
            or len(pieces) != token_count
            or any(type(piece) is not str for piece in pieces)
        ):
            raise ValueError("converted token pieces must be an exact string list")
        events = tuple(
            TokenEvent.model_validate(
                {
                    "run_key": run_key,
                    "sequence_id": sequence_id,
                    "token_pos": position,
                    "token_id": token_id,
                    "token_text": piece,
                    "phase": "prefill",
                }
            )
            for position, (token_id, piece) in enumerate(zip(ids, pieces, strict=True))
        )
        return events, copied_encoding
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _stage_error("encoding", exc)


def run_mixtral_prompt_prefill(
    loaded: LoadedModel,
    inspection: AdapterInspection,
    plan: ProbePlan,
    prompt: str,
    *,
    run_key: str,
    sequence_id: str,
    add_special_tokens: bool,
    max_prompt_chars: int,
    max_tokens: int,
    max_events: int,
) -> RoutingForwardResult:
    """Tokenize one plain-text prompt and delegate one bounded passive prefill."""

    (
        fresh_loaded,
        _manifest,
        fresh_inspection,
        fresh_plan,
        tokenizer,
        _prompt_budget,
        token_budget,
        event_budget,
        routed_top_k,
    ) = _preflight(
        loaded,
        inspection,
        plan,
        prompt,
        run_key=run_key,
        sequence_id=sequence_id,
        add_special_tokens=add_special_tokens,
        max_prompt_chars=max_prompt_chars,
        max_tokens=max_tokens,
        max_events=max_events,
    )
    target_count = len(fresh_plan.targets)
    converter = _resolve_converter(tokenizer)
    token_events, copied_encoding = _encode(
        tokenizer,
        prompt,
        add_special_tokens=add_special_tokens,
        max_tokens=token_budget,
        max_events=event_budget,
        target_count=target_count,
        routed_top_k=routed_top_k,
        converter=converter,
        run_key=run_key,
        sequence_id=sequence_id,
    )

    return run_mixtral_routing_forward(
        fresh_loaded.model,
        fresh_inspection,
        fresh_plan,
        token_events,
        copied_encoding,
        max_events=event_budget,
    )


__all__ = [
    "MixtralPromptPrefillError",
    "run_mixtral_prompt_prefill",
]
