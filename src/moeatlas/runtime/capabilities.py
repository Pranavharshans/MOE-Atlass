"""Model-neutral routing decode capability contracts.

Adapters own the family-specific knowledge of how raw router payloads are
shaped and decoded.  The shared runtime consumes only this contract: an
adapter-declared payload-shape vocabulary, adapter-declared score semantics,
one decode entry point producing canonical :class:`~moeatlas.events.RoutingEvent`
rows, and shared postcondition validation that never inspects adapter names,
module-path conventions, or payload type identities.

The historical Mixtral (``tuple_logits``) and Qwen3.5
(``tuple_scores_indices``) shapes are ordinary values of the declared
vocabulary; unknown families declare their own — mapping-keyed arrays,
assignment-only indices, or anything else an adapter can decode — without a
central branch.  Native expert identifiers may be sparse or unordered; the
:func:`native_id_map` helper resolves them through the published
:class:`~moeatlas.adapters.RoutingUniverse`.

This module performs no model loading, no network access, and imports no
model-runtime dependency.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..adapters import LayerRoutingUniverse, RoutingUniverse
from ..core import parse_token_key
from ..events import RoutingEvent

if TYPE_CHECKING:
    from .routing import RoutingCaptureTarget

ROUTING_DECODE_CAPABILITY_SCHEMA_VERSION = "1.0"

_DECODE_STAGES = frozenset({"dependency", "decode", "postcondition"})


class RoutingDecodeError(ValueError):
    """Safe fixed-stage failure for capability-driven routing decode."""

    def __init__(
        self,
        stage: str,
        message: str | None = None,
        *,
        cause: BaseException | None = None,
    ) -> None:
        if type(stage) is not str or stage not in _DECODE_STAGES:
            raise ValueError(f"unsupported routing decode stage: {stage!r}")
        self.stage = stage
        text = f"routing decode failed at {stage}"
        if message is not None:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


class RouterPayloadShape(str, Enum):
    """Adapter-declared vocabulary of raw router payload layouts."""

    LOGITS_TUPLE = "tuple_logits"
    SCORES_INDICES_TUPLE = "tuple_scores_indices"
    INDICES_WEIGHTS_LOGITS_TUPLE = "tuple_indices_weights_logits"
    DICT_ARRAYS = "dict_arrays"
    ASSIGNMENT_INDICES = "assignment_indices"

    def __str__(self) -> str:
        return self.value


class ScoreSemantics(str, Enum):
    """What the optional score columns on decoded events mean.

    ``NONE`` is assignment-only evidence: no logit or probability claims are
    made, and every row pins ``weight`` to exactly ``1.0`` as the explicit
    "selected, unweighted" marker required by the event contract.
    """

    LOGITS = "logits"
    PROBABILITIES = "probabilities"
    WEIGHTS = "weights"
    NONE = "none"

    def __str__(self) -> str:
        return self.value


class TokenSequencePolicy(str, Enum):
    """How strictly one forward's token identities are validated.

    ``SHARED_RUN`` requires unique keys sharing one run and phase.
    ``CANONICAL_SEQUENCE`` additionally requires one shared sequence in the
    contiguous canonical position order ``0..N-1`` — the shape batched
    whole-sequence payloads depend on.
    """

    SHARED_RUN = "shared_run"
    CANONICAL_SEQUENCE = "canonical_sequence"

    def __str__(self) -> str:
        return self.value


@runtime_checkable
class RoutingHookDecoder(Protocol):
    """One adapter's declared router-hook decode seam for forward capture.

    A conforming decoder declares the payload vocabulary it interprets and
    the score semantics of the rows it emits, then decodes one opaque router
    hook invocation — ``(module, inputs, output)`` for a single resolved
    :class:`~moeatlas.runtime.RoutingCaptureTarget` — into canonical
    :class:`~moeatlas.events.RoutingEvent` rows.  Family knowledge lives
    entirely in the decoder; the shared session and runner never inspect
    adapter names or payload type identities.
    """

    @property
    def payload_shape(self) -> RouterPayloadShape: ...

    @property
    def score_semantics(self) -> ScoreSemantics: ...

    def __call__(
        self,
        context: RoutingCaptureTarget,
        module: object,
        inputs: tuple[object, ...],
        output: object,
    ) -> tuple[RoutingEvent, ...]: ...


@runtime_checkable
class RoutingDecodeCapability(Protocol):
    """One adapter's family-specific router-output decoding seam."""

    @property
    def payload_shape(self) -> RouterPayloadShape: ...

    @property
    def score_semantics(self) -> ScoreSemantics: ...

    def decode(
        self,
        payload: object,
        *,
        universe: RoutingUniverse,
        token_key: str,
    ) -> tuple[RoutingEvent, ...]: ...


def native_id_map(layer: object) -> dict[int, str]:
    """Map native expert identifiers to canonical keys for one layer.

    The mapping follows the published parallel arrays exactly: native
    identifiers may be sparse or unordered, and each maps to the canonically
    sorted ``expert_keys`` entry at the same position.  Layers that do not
    publish native identifiers have no such map.
    """

    if type(layer) is not LayerRoutingUniverse:
        raise RoutingDecodeError(
            "dependency",
            f"layer must be an exact LayerRoutingUniverse, got {type(layer).__name__}",
        )
    if layer.expert_indices is None:
        raise RoutingDecodeError(
            "decode",
            f"layer {layer.layer_index} does not declare native expert indices",
        )
    return dict(zip(layer.expert_indices, layer.expert_keys, strict=True))


def validate_decoded_routing(
    events: object,
    *,
    universe: RoutingUniverse,
    token_key: object,
    score_semantics: object,
    require_all_layers: bool = True,
) -> tuple[RoutingEvent, ...]:
    """Validate decoded rows against the shared, family-neutral postconditions.

    Every row must name the expected token, resolve to a universe layer, and
    carry the layer's complete rank schedule ``0..top_k-1`` with unique
    experts drawn from that layer's published universe.  Score columns must
    agree with the declared semantics: logit rows carry finite logits,
    probability rows carry probabilities in the unit interval (already
    enforced by the event contract itself), and assignment-only rows make no
    score claims at all, pinning ``weight`` to the unweighted ``1.0`` marker.
    """

    semantics = _exact_enum(score_semantics, ScoreSemantics, "score_semantics")
    key = _exact_token_key(token_key)
    if type(events) is not tuple or any(type(item) is not RoutingEvent for item in events):
        raise RoutingDecodeError(
            "postcondition", "decoded events must be a tuple of exact RoutingEvent values"
        )
    try:
        fresh = tuple(
            RoutingEvent.model_validate(item.model_dump(mode="json")) for item in events
        )
    except Exception as exc:
        raise RoutingDecodeError(
            "postcondition", "decoded events failed revalidation", cause=exc
        ) from exc
    layer_by_key = {layer.moe_layer_key: layer for layer in universe.layers}
    layer_by_index = {layer.layer_index: layer for layer in universe.layers}
    seen_layers: dict[int, list[RoutingEvent]] = {}
    for event in fresh:
        if event.token_key != key:
            raise RoutingDecodeError(
                "postcondition", "decoded event token does not match the requested token"
            )
        layer = layer_by_key.get(event.layer_key)
        if layer is None:
            raise RoutingDecodeError(
                "postcondition",
                f"decoded event layer {event.layer_key!r} is outside the universe",
            )
        if event.selected is not True:
            raise RoutingDecodeError(
                "postcondition", "every decoded routing row must be selected"
            )
        seen_layers.setdefault(layer.layer_index, []).append(event)
        _check_score_columns(event, semantics)
    for index in sorted(seen_layers):
        layer = layer_by_index[index]
        layer_events = seen_layers[index]
        ranks = sorted(event.rank for event in layer_events)
        if ranks != list(range(layer.routed_top_k)):
            raise RoutingDecodeError(
                "postcondition",
                f"layer {index} ranks are not exactly 0..{layer.routed_top_k - 1}",
            )
        ordered = sorted(layer_events, key=lambda event: event.rank)
        experts = [event.expert_key for event in ordered]
        if len(set(experts)) != len(experts):
            raise RoutingDecodeError(
                "postcondition", f"layer {index} selects the same expert twice"
            )
        allowed = set(layer.expert_keys)
        if any(expert not in allowed for expert in experts):
            raise RoutingDecodeError(
                "postcondition",
                f"layer {index} selects an expert outside its published universe",
            )
    if require_all_layers:
        missing = sorted(set(universe.layer_indices) - set(seen_layers))
        if missing:
            raise RoutingDecodeError(
                "postcondition",
                f"universe layers {missing} have no decoded rows",
            )
    return fresh


def validate_observed_routing(
    events: object,
    *,
    universe: RoutingUniverse,
    token_keys: object,
    score_semantics: object,
) -> tuple[RoutingEvent, ...]:
    """Validate one complete forward capture against shared postconditions.

    This is the multi-token form of :func:`validate_decoded_routing`: every
    captured token must carry every universe layer's complete rank schedule
    ``0..top_k-1`` with unique experts drawn from that layer's published
    universe, and every score column must agree with the decoder-declared
    semantics.  It is the family-neutral completeness gate that a forward
    runner applies to whatever any conforming hook decoder produced.
    """

    if type(universe) is not RoutingUniverse:
        raise RoutingDecodeError(
            "dependency",
            f"universe must be an exact RoutingUniverse, got {type(universe).__name__}",
        )
    semantics = _exact_enum(score_semantics, ScoreSemantics, "score_semantics")
    if type(token_keys) is not tuple or any(type(key) is not str for key in token_keys):
        raise RoutingDecodeError("dependency", "token_keys must be a tuple of exact strings")
    for key in token_keys:
        parse_token_key(key)
    keys = tuple(token_keys)
    if len(set(keys)) != len(keys):
        raise RoutingDecodeError("dependency", "token_keys must be unique")
    if type(events) is not tuple or any(type(item) is not RoutingEvent for item in events):
        raise RoutingDecodeError(
            "postcondition", "observed events must be a tuple of exact RoutingEvent values"
        )
    try:
        fresh = tuple(
            RoutingEvent.model_validate(item.model_dump(mode="json")) for item in events
        )
    except Exception as exc:
        raise RoutingDecodeError(
            "postcondition", "observed events failed revalidation", cause=exc
        ) from exc
    layer_by_key = {layer.moe_layer_key: layer for layer in universe.layers}
    key_set = set(keys)
    groups: dict[tuple[str, int], list[RoutingEvent]] = {}
    for event in fresh:
        if event.token_key not in key_set:
            raise RoutingDecodeError(
                "postcondition",
                "observed event references a token outside the capture",
            )
        layer = layer_by_key.get(event.layer_key)
        if layer is None:
            raise RoutingDecodeError(
                "postcondition",
                f"observed event layer {event.layer_key!r} is outside the universe",
            )
        if event.selected is not True:
            raise RoutingDecodeError(
                "postcondition", "every observed routing row must be selected"
            )
        groups.setdefault((event.token_key, layer.layer_index), []).append(event)
        _check_score_columns(event, semantics)
    for key in keys:
        for layer in universe.layers:
            group = groups.get((key, layer.layer_index))
            if group is None:
                raise RoutingDecodeError(
                    "postcondition",
                    f"layer {layer.layer_index} has no observed rows for a captured token",
                )
            ranks = sorted(event.rank for event in group)
            if ranks != list(range(layer.routed_top_k)):
                raise RoutingDecodeError(
                    "postcondition",
                    f"layer {layer.layer_index} ranks are not exactly 0..{layer.routed_top_k - 1}",
                )
            ordered = sorted(group, key=lambda event: event.rank)
            experts = [event.expert_key for event in ordered]
            if len(set(experts)) != len(experts):
                raise RoutingDecodeError(
                    "postcondition", f"layer {layer.layer_index} selects the same expert twice"
                )
            allowed = set(layer.expert_keys)
            if any(expert not in allowed for expert in experts):
                raise RoutingDecodeError(
                    "postcondition",
                    f"layer {layer.layer_index} selects an expert outside its published universe",
                )
    return fresh


def _check_score_columns(event: RoutingEvent, semantics: ScoreSemantics) -> None:
    if semantics is ScoreSemantics.NONE:
        if event.router_logit is not None or event.probability is not None:
            raise RoutingDecodeError(
                "postcondition",
                "assignment-only semantics forbid logit and probability claims",
            )
        if event.weight != 1.0:
            raise RoutingDecodeError(
                "postcondition",
                "assignment-only semantics pin weight to the unweighted 1.0 marker",
            )
        return
    if semantics is ScoreSemantics.LOGITS and event.router_logit is None:
        raise RoutingDecodeError(
            "postcondition", "logit semantics require a finite router_logit"
        )
    if semantics is ScoreSemantics.PROBABILITIES and event.probability is None:
        raise RoutingDecodeError(
            "postcondition", "probability semantics require a finite probability"
        )
    if semantics is ScoreSemantics.WEIGHTS and event.weight is None:
        raise RoutingDecodeError(
            "postcondition", "weight semantics require a finite weight"
        )


def _exact_enum(value: object, enum_type: type[Enum], field_name: str):
    if type(value) is not enum_type or not isinstance(value, enum_type):
        raise RoutingDecodeError(
            "dependency", f"{field_name} must be an exact {enum_type.__name__}"
        )
    return value


def _exact_token_key(value: object) -> str:
    if type(value) is not str:
        raise RoutingDecodeError("dependency", "token_key must be an exact string")
    parse_token_key(value)
    return value
