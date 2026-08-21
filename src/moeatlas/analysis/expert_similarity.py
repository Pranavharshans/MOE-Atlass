"""Expert weight/representation similarity summaries (PRD §11.3).

Compares experts within a layer by the cosine similarity of caller-supplied
vectors — expert weight summaries or selected representation summaries
alike; the contract is agnostic to provenance. Per layer the analysis
derives the symmetric similarity matrix with exact ``1.0`` diagonals and
explicit ``null`` entries wherever an expert's vector has zero norm:
undefined geometry is evidence, never inferred.

The layer stays pure: no storage reads, clocks, randomness, or model
knowledge. Similarity describes geometry only — it never implies
specialization or causality.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

EXPERT_SIMILARITY_SCHEMA_VERSION = "1.0"
"""Schema version of the expert-similarity contracts."""

_EXPERT_SIMILARITY_ARTIFACT_TYPE = "moeatlas.expert_similarity"

_ERROR_STAGES = frozenset({"contract", "budget"})


class ExpertSimilarityError(RuntimeError):
    """Safe fixed-stage failure for expert-similarity handling."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _ERROR_STAGES:
            raise ValueError("expert similarity error stage is not supported")
        self.stage = stage
        text = f"expert similarity failed at {stage}"
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


def _strict_vector_row(value: object, width: int, field_name: str) -> tuple[tuple[float, ...], ...]:
    if type(value) is not tuple:
        raise TypeError(f"{field_name} must be a tuple of vector tuples")
    if len(value) != width:
        raise ValueError(f"{field_name} must hold exactly one vector per expert")
    length: int | None = None
    for vector in value:
        if type(vector) is not tuple:
            raise TypeError(f"{field_name} vectors must be tuples of numbers")
        if length is None:
            length = len(vector)
        elif len(vector) != length:
            raise ValueError(f"{field_name} vectors must share one length")
        for entry in vector:
            if type(entry) is not float and type(entry) is not int:
                raise TypeError(f"{field_name} entries must be numbers")
            if entry != entry or entry in (float("inf"), float("-inf")):
                raise ValueError(f"{field_name} entries must be finite")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class ExpertVectors:
    """Per-layer expert vectors over a shared within-layer length.

    ``vectors[layer]`` holds exactly one finite vector per expert key; every
    vector in one layer must share its length. Zero vectors are legal and
    mark the expert's geometry as undefined downstream.
    """

    layer_keys: tuple[str, ...]
    expert_keys: tuple[tuple[str, ...], ...]
    vectors: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        _strict_key_tuple(self.layer_keys, "layer_keys")
        if type(self.expert_keys) is not tuple:
            raise TypeError("expert_keys must be a tuple of per-layer key tuples")
        for row in self.expert_keys:
            _strict_key_tuple(row, "expert_keys")
        if len(self.expert_keys) != len(self.layer_keys):
            raise ValueError("expert_keys must hold exactly one row per layer")
        if type(self.vectors) is not tuple:
            raise TypeError("vectors must be a tuple of per-layer rows")
        if len(self.vectors) != len(self.layer_keys):
            raise ValueError("vectors must hold exactly one row per layer")
        for index, layer in enumerate(self.layer_keys):
            _strict_vector_row(
                self.vectors[index],
                len(self.expert_keys[index]),
                f"vectors[{layer!r}]",
            )

    @property
    def cell_count(self) -> int:
        return sum(len(row) for row in self.expert_keys)


@dataclass(frozen=True, slots=True)
class ExpertSimilarity:
    """Per-layer cosine-similarity matrices with explicit undefined experts.

    ``similarity_rows[layer][i][j]`` is the cosine similarity between
    experts i and j: ``1.0`` on diagonals of nonzero vectors, ``null`` in
    every cell touching a zero-norm expert. ``undefined_expert_rows`` count
    zero-norm experts per layer.
    """

    schema_version: str
    layer_keys: tuple[str, ...]
    similarity_rows: tuple[tuple[tuple[float | None, ...], ...], ...]
    undefined_expert_rows: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": _EXPERT_SIMILARITY_ARTIFACT_TYPE,
            "schema_version": self.schema_version,
            "layer_keys": list(self.layer_keys),
            "similarity_rows": [
                [list(row) for row in matrix] for matrix in self.similarity_rows
            ],
            "undefined_expert_rows": list(self.undefined_expert_rows),
        }

    def to_json(self) -> str:
        """Serialize this result with deterministic key order and no whitespace."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, payload: str | bytes | bytearray) -> ExpertSimilarity:
        """Validate one canonical JSON document into an exact result value."""

        try:
            document = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("expert similarity document is not valid JSON") from exc
        if type(document) is not dict:
            raise ValueError("expert similarity document must be a JSON object")
        if (
            document.get("artifact_type") != _EXPERT_SIMILARITY_ARTIFACT_TYPE
            or document.get("schema_version") != EXPERT_SIMILARITY_SCHEMA_VERSION
        ):
            raise ValueError("document is not an expert similarity artifact")
        try:
            return cls(
                schema_version=document["schema_version"],
                layer_keys=tuple(document["layer_keys"]),
                similarity_rows=tuple(
                    tuple(tuple(row) for row in matrix)
                    for matrix in document["similarity_rows"]
                ),
                undefined_expert_rows=tuple(document["undefined_expert_rows"]),
            )
        except KeyError as exc:
            raise ValueError("expert similarity document is missing fields") from exc


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    similarity = dot / (left_norm * right_norm)
    # Clamp float noise so the documented [-1, 1] bounds hold exactly.
    return min(1.0, max(-1.0, similarity))


def analyze_expert_similarity(
    vectors: ExpertVectors, *, max_cells: int = 1_000_000
) -> ExpertSimilarity:
    """Derive per-layer cosine-similarity matrices from expert vectors."""

    if type(vectors) is not ExpertVectors:
        raise TypeError("vectors must be an ExpertVectors")
    if type(max_cells) is not int or isinstance(max_cells, bool):
        raise TypeError("max_cells must be an integer")
    if max_cells <= 0:
        raise ExpertSimilarityError("budget", "max_cells must be strictly positive")
    if vectors.cell_count > max_cells:
        raise ExpertSimilarityError(
            "budget",
            f"vector tables have {vectors.cell_count} cells; budget is {max_cells}",
        )
    similarity_rows: list[tuple[tuple[float | None, ...], ...]] = []
    undefined_rows: list[int] = []
    for index, layer in enumerate(vectors.layer_keys):
        row = vectors.vectors[index]
        norms = [math.sqrt(sum(entry * entry for entry in vector)) for vector in row]
        undefined = sum(1 for norm in norms if norm == 0.0)
        undefined_rows.append(undefined)
        matrix: list[tuple[float | None, ...]] = []
        for i, vector_i in enumerate(row):
            cells: list[float | None] = []
            for j, vector_j in enumerate(row):
                if norms[i] == 0.0 or norms[j] == 0.0:
                    cells.append(None)
                else:
                    cells.append(_cosine(vector_i, vector_j))
            matrix.append(tuple(cells))
        similarity_rows.append(tuple(matrix))
    return ExpertSimilarity(
        schema_version=EXPERT_SIMILARITY_SCHEMA_VERSION,
        layer_keys=vectors.layer_keys,
        similarity_rows=tuple(similarity_rows),
        undefined_expert_rows=tuple(undefined_rows),
    )
