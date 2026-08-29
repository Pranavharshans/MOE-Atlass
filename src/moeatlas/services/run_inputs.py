"""Input preparation: descriptors and prompt specs become engine-ready rows.

This module closes the input-preparation half of the run engine. It turns
either input descriptor from ``moeatlas.runs.specs`` into the exact shapes
``execute_row_schedule`` consumes — a ``{row_index: values}`` mapping plus a
deterministic batch schedule — so the execution core never branches on input
kind. Prompt specs become exactly one row (raw ``{"prompt": text}`` or
``{"messages": [{"role", "content"}, ...]}``); dataset descriptors compose
the bounded reader with task-role projection and the descriptor's own
sample/batch/shuffle/seed settings.

Preparation is deterministic and bounded: canonical-JSON byte budgets for
prompt rows, the reader's row/byte/file budgets for datasets, SHA-256-keyed
schedules, and no clocks, randomness, or model dependencies. A Hub dataset
may perform network I/O only when its descriptor explicitly opts into it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..runs.specs import DatasetInputSpec, PromptInputSpec
from .datasets import plan_dataset_batches, project_dataset_rows, read_dataset_rows

RUN_INPUTS_SCHEMA_VERSION = "1.0"
"""Schema version of the input-preparation contracts."""

_DEFAULT_MAX_INPUT_BYTES = 65_536

_STAGES = frozenset({"spec", "format", "budget"})
_MULTIPLE_CHOICE_LABELS = ("A", "B", "C", "D")


class RunInputError(RuntimeError):
    """Safe fixed-stage failure for run input preparation."""

    def __init__(
        self, stage: str, message: str | None = None, *, cause: BaseException | None = None
    ) -> None:
        if stage not in _STAGES:
            raise ValueError("run input error stage is not supported")
        self.stage = stage
        if message is None:
            super().__init__(f"run input preparation failed at {stage}")
        else:
            super().__init__(f"run input preparation failed at {stage}: {message}")
        if cause is not None:
            self.__cause__ = cause


def _canonical_bytes(values: Any) -> bytes:
    try:
        payload = json.dumps(
            values,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise RunInputError("format", "prepared row is not canonically encodable", cause=exc)
    return payload.encode("utf-8")


def prepare_prompt_rows(
    spec: PromptInputSpec, *, max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES
) -> dict[int, dict[str, Any]]:
    """Prepare one prompt spec as exactly one engine-ready values mapping.

    Raw text becomes ``{0: {"prompt": text}}``; chat messages become
    ``{0: {"messages": [{"role": ..., "content": ...}, ...]}}`` in declared
    order. The canonical encoding must fit ``max_input_bytes``.
    """

    if not isinstance(spec, PromptInputSpec):
        raise TypeError("spec must be a PromptInputSpec")
    if (
        type(max_input_bytes) is not int
        or isinstance(max_input_bytes, bool)
        or max_input_bytes <= 0
    ):
        raise TypeError("max_input_bytes must be a strict positive integer")
    if spec.text is not None:
        values: dict[str, Any] = {"prompt": spec.text}
    else:
        values = {
            "messages": [
                {"role": message.role, "content": message.content} for message in spec.messages
            ]
        }
    encoded = _canonical_bytes(values)
    if len(encoded) > max_input_bytes:
        raise RunInputError("budget", f"prompt input exceeds the {max_input_bytes} byte budget")
    return {0: values}


def plan_input_batches(
    spec: PromptInputSpec | DatasetInputSpec, total_rows: int
) -> tuple[tuple[int, ...], ...]:
    """Derive the deterministic batch schedule an input implies.

    Prompt inputs are the single-row schedule ``((0,),)``; dataset inputs
    apply their descriptor's ``sample_cap``/``batch_size``/``shuffle``/``seed``
    through ``plan_dataset_batches``.
    """

    if isinstance(spec, PromptInputSpec):
        if type(total_rows) is not int or isinstance(total_rows, bool) or total_rows != 1:
            raise ValueError("prompt inputs plan exactly one row")
        return ((0,),)
    if not isinstance(spec, DatasetInputSpec):
        raise TypeError("spec must be a PromptInputSpec or DatasetInputSpec")
    return plan_dataset_batches(
        total_rows,
        batch_size=spec.batch_size,
        sample_cap=spec.sample_cap,
        shuffle=spec.shuffle,
        seed=spec.seed,
    )


def prepare_input_rows(
    spec: PromptInputSpec | DatasetInputSpec,
    *,
    base_directory: str | Path | None = None,
    max_input_bytes: int = _DEFAULT_MAX_INPUT_BYTES,
    max_rows: int = 10_000,
    max_row_bytes: int = 65_536,
    max_file_bytes: int = 100_000_000,
    duckdb: Any = None,
) -> dict[int, dict[str, Any]]:
    """Prepare either input kind as the engine's ``{row_index: values}`` mapping.

    Prompt specs produce exactly one entry; dataset descriptors are read
    through the bounded reader and, when a column mapping is declared,
    projected onto their task roles while preserving read-order indices.
    """

    if isinstance(spec, PromptInputSpec):
        return prepare_prompt_rows(spec, max_input_bytes=max_input_bytes)
    if not isinstance(spec, DatasetInputSpec):
        raise TypeError("spec must be a PromptInputSpec or DatasetInputSpec")
    rows = read_dataset_rows(
        spec,
        base_directory=base_directory,
        max_rows=max_rows,
        max_row_bytes=max_row_bytes,
        max_file_bytes=max_file_bytes,
        duckdb=duckdb,
    )
    if spec.prompt_format == "mmlu_multiple_choice":
        prompt_column = spec.column_mapping.get("prompt")
        reference_column = spec.column_mapping.get("reference")
        choices_column = spec.choices_column
        if prompt_column is None or reference_column is None or choices_column is None:
            raise RunInputError(
                "spec", "MMLU formatting requires prompt, reference, and choices columns"
            )
        prepared: dict[int, dict[str, Any]] = {}
        for row in rows:
            question = row.values.get(prompt_column)
            choices = row.values.get(choices_column)
            answer = row.values.get(reference_column)
            if not isinstance(question, str) or not question.strip():
                raise RunInputError("format", "MMLU question must be a non-empty string")
            if (
                not isinstance(choices, list | tuple)
                or len(choices) != 4
                or any(not isinstance(choice, str) for choice in choices)
            ):
                raise RunInputError("format", "MMLU choices must contain exactly four strings")
            if type(answer) is int and not isinstance(answer, bool) and 0 <= answer < 4:
                reference = _MULTIPLE_CHOICE_LABELS[answer]
            elif isinstance(answer, str) and answer.strip().upper() in _MULTIPLE_CHOICE_LABELS:
                reference = answer.strip().upper()
            else:
                raise RunInputError("format", "MMLU answer must be an index from 0 to 3")
            rendered_choices = "\n".join(
                f"{label}. {choice}"
                for label, choice in zip(_MULTIPLE_CHOICE_LABELS, choices, strict=True)
            )
            prepared[row.index] = {
                "prompt": (
                    f"{question.strip()}\n\n{rendered_choices}\n\nAnswer with only A, B, C, or D."
                ),
                "reference": reference,
            }
        return prepared
    if spec.column_mapping:
        projected = project_dataset_rows(rows, spec.column_mapping)
        return {row.index: values for row, values in zip(rows, projected)}
    return {row.index: dict(row.values) for row in rows}
