# Pull request

## What and why

<!-- One paragraph: what changes, which PRD slice/roadmap row it serves. -->

## Evidence

- [ ] Feature-targeted tests added or updated (synthetic fixtures only)
- [ ] Full local gate green: `uv run --locked pytest -q`
- [ ] `uv run --locked ruff check src tests` clean
- [ ] `uv run --locked python -m unittest discover -s tests -t .` OK
- [ ] `uv build --no-sources` succeeds
- [ ] No model, tokenizer, dataset, cache, or unrelated files included
- [ ] `docs/roadmap.md` / `docs/model-validation-ledger.md` updated honestly
      (deferred rows stay deferred; model-free evidence is not claimed as
      checkpoint/GPU evidence)

## Notes for reviewers

<!-- Anything surprising: contract changes, alias behavior, error text. -->
