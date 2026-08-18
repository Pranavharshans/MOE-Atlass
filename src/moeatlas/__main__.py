"""Allow ``python -m moeatlas`` to invoke the CLI."""

from .cli import main

if __name__ == "__main__":  # pragma: no cover - exercised through subprocesses
    raise SystemExit(main())
