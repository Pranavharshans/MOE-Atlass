"""Build a local synthetic workspace and evaluate retention on it.

This example exercises only public, family-neutral services: it registers
two synthetic runs into a fresh workspace, queries the registry, and
evaluates a retention policy that keeps only the newest run. Nothing
downloads a model, touches the network, or requires a GPU — every digest
below is synthetic content derived from this script's own constants.

Usage::

    python examples/synthetic_workspace.py TARGET_DIRECTORY
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

from moeatlas.runs.specs import DataProvenance, ModelProvenance, PromptInputSpec, RunSpecification
from moeatlas.services import (
    RetentionPolicy,
    evaluate_retention,
    initialize_workspace,
    open_workspace,
    query_runs,
    register_run,
)


def _digest(label: str) -> str:
    """Deterministic 64-hex digest from a label (synthetic content only)."""

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _synthetic_specification(prompt_text: str) -> RunSpecification:
    plan_digest = _digest(f"plan:{prompt_text}")
    specification = RunSpecification(
        model=ModelProvenance(
            loading_plan_id=f"loadplan:{plan_digest}",
            model_id="example/synthetic-moe",
            model_revision=_digest(f"revision:{prompt_text}")[:12],
            tokenizer_revision=_digest("tokenizer")[:12],
        ),
        data=DataProvenance(input=PromptInputSpec(text=prompt_text)),
        created_by="examples/synthetic_workspace",
    )
    return specification


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    if target.exists():
        print(f"refusing to touch an existing directory: {target}")
        return 2
    target.mkdir(parents=True)

    initialize_workspace(target)
    snapshot = open_workspace(target)
    print(f"workspace: {snapshot.path}")

    registered: list[str] = []
    for index, prompt_text in enumerate(("first synthetic prompt", "second synthetic prompt")):
        specification = _synthetic_specification(prompt_text)
        entry = register_run(
            target,
            specification,
            at=f"2026-08-21T00:00:0{index}Z",
        )
        registered.append(entry.run_key)
        print(f"registered: {entry.run_key} state={entry.state}")

    entries = query_runs(target)
    print(f"registry holds {len(entries)} runs")

    policy = RetentionPolicy(max_runs=1)
    report = evaluate_retention(entries, policy)
    print(f"retained: {len(report.retained_keys)} expired: {len(report.expired_keys)}")
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
