"""Router-margin routing-confidence summaries (PRD §11.1).

The router margin is the difference between a token's two top selected-route
scores at one layer — a routing-confidence lens that never re-weights or
re-ranks anything. Callers supply, per layer, each token's selected-route
scores in rank order (logits or probabilities alike; any finite scores);
the analysis derives per-token margins where two scored ranks exist and
summarizes their mean with explicit defined/total token counts.

Tokens with fewer than two scored ranks contribute no margin — absence is
evidence, never inferred. The layer stays pure: no storage reads, clocks,
randomness, or model knowledge, and no claim that a margin explains
specialization or causality.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

ROUTER_MARGIN_SCHEMA_VERSION = "1.0"
"""Schema version of the router-margin contracts."""

_ROUTER_MARGIN_ARTIFACT_TYPE = "moeatlas.router_margin"

_ERROR_STAGES = frozenset({"contract", "budget"})


class RouterMarginError(RuntimeError):
    """Safe fixed-stage failure for router-margin handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("router margin error stage is not supported")
        self.stage = stage
        text = f"router margin failed at {stage}"
        if message:
            text = f"{text}: {message}"
        super().__init__(text)
        if cause is not None:
            self.__cause__ = cause


def _strict_key_tuple(value: object, field_name: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of strings")
    for entry in value:
        if type(entry) is not str:
            raise TypeError(f"{field_name} entries must be strings")
    keys = tuple(value)
    if not keys or list(keys) != sorted(set(keys)):
        raise ValueError(f"{field_name} must be non-empty, unique, and sorted")
    return keys


def _strict_score_row(value: object, field_name: str) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of finite score tuples")
    for scores in value:
        if type(scores) is not tuple:
            raise TypeError(f"{field_name} entries must be tuples of scores")
        for score in scores:
            if type(score) is not float and type(score) is not int:
                raise TypeError(f"{field_name} scores must be numbers")
            if score != score or score in (float("inf"), float("-inf")):
                raise ValueError(f"{field_name} scores must be finite")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class RouterMarginSamples:
    """Per-layer selected-route scores in rank order, one entry per token.

    ``token_scores[layer]`` holds one inner tuple per token: that token's
    selected-route scores ordered best-first. Empty inner tuples are legal
    and mean "no scored ranks for this token" — their margin stays undefined
    downstream instead of being inferred.
    """

    layer_keys: tuple[str, ...]
    token_scores: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        _strict_key_tuple(self.layer_keys, "layer_keys")
        if type(self.token_scores) is not tuple:
            raise TypeError("token_scores must be a tuple of per-layer rows")
        if len(self.token_scores) != len(self.layer_keys):
            raise ValueError("token_scores must hold exactly one row per layer")
        for index, layer in enumerate(self.layer_keys):
            _strict_score_row(self.token_scores[index], f"token_scores[{layer!r}]")

    @property
    def cell_count(self) -> int:
        return sum(len(row) for row in self.token_scores)


@dataclass(frozen=True, slots=True)
class RouterMarginSummary:
    """Per-layer margin summaries with explicit defined/total token counts.

    ``mean_margin_rows`` hold the mean top1-minus-top2 margin over tokens
    with at least two scored ranks, or ``null`` when no token in the layer
    has a defined margin. ``margin_token_rows`` count the tokens that
    contributed; ``token_rows`` count all supplied tokens.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    mean_margin_rows: tuple[float | None, ...]
    margin_token_rows: tuple[int, ...]
    token_rows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _ROUTER_MARGIN_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "mean_margin_rows": list(self.mean_margin_rows),
            "margin_token_rows": list(self.margin_token_rows),
            "token_rows": list(self.token_rows),
        }

    def to_json(self) -> str:
        """Serialize this summary with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> RouterMarginSummary:
        """Validate one canonical JSON document into an exact summary value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("router margin document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("router margin document must be a JSON object")
        if (
            document.get("artifact_type") != _ROUTER_MARGIN_ARTIFACT_TYPE
            or document.get("schema_version") != ROUTER_MARGIN_SCHEMA_VERSION
        ):
            raise ValueError("document is not a router margin artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                mean_margin_rows=tuple(document["mean_margin_rows"]),
                margin_token_rows=tuple(document["margin_token_rows"]),
                token_rows=tuple(document["token_rows"]),
            )
        except KeyError as exc:
            raise ValueError("router margin document is missing fields") from exc


def analyze_router_margin(
    samples: RouterMarginSamples, *, max_tokens: int = 1_000_000
) -> RouterMarginSummary:
    """Derive per-layer router-margin summaries from ranked score samples."""

    if type(samples) is not RouterMarginSamples:
        raise TypeError("samples must be a RouterMarginSamples")
    if type(max_tokens) is not int or isinstance(max_tokens, bool):
        raise TypeError("max_tokens must be an integer")
    if max_tokens <= 0:
        raise RouterMarginError("budget", "max_tokens must be strictly positive")
    if samples.cell_count > max_tokens:
        raise RouterMarginError(
            "budget",
            f"sample table has {samples.cell_count} tokens; budget is {max_tokens}",
        )
    mean_rows: list[float | None] = []
    margin_counts: list[int] = []
    token_counts: list[int] = []
    for index, layer in enumerate(samples.layer_keys):
        rows = samples.token_scores[index]
        margins = [
            scores[0] - scores[1] for scores in rows if len(scores) >= 2
        ]
        mean_rows.append(sum(margins) / len(margins) if margins else None)
        margin_counts.append(len(margins))
        token_counts.append(len(rows))
    return RouterMarginSummary(
        schema_version=ROUTER_MARGIN_SCHEMA_VERSION,
        layer_keys=samples.layer_keys,
        mean_margin_rows=tuple(mean_rows),
        margin_token_rows=tuple(margin_counts),
        token_rows=tuple(token_counts),
    )
