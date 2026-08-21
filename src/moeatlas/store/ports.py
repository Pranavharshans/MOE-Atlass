"""Model-neutral storage ports so services/CLI/server depend on protocols,
not the concrete module; duckdb is imported lazily at call time through
routing_shards, never at module import.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..runtime.routing_forward import RoutingForwardResult
from . import routing_shards as _routing_shards
from .routing_shards import (
    MixtralRoutingRunInventory,
    RoutingShardAssignmentQuery,
    RoutingShardReceipt,
    append_routing_shard,
    list_routing_runs,
    list_routing_shards,
    query_routing_run_assignments,
)


@runtime_checkable
class RoutingRunReader(Protocol):
    """Protocol for reading routing-run data without depending on the concrete store."""

    def list_runs(
        self,
        *,
        max_runs: int,
        max_shards: int,
        max_event_rows: int,
        max_source_bytes: int,
    ) -> MixtralRoutingRunInventory:
        """List routing runs within the given budget bounds."""
        ...

    def list_shards(self, *, run_key: str) -> tuple[RoutingShardReceipt, ...]:
        """List all validated immutable routing shards for one run."""
        ...

    def query_assignments(
        self,
        *,
        run_key: str,
        layer_keys: tuple[str, ...],
        expert_keys: tuple[tuple[str, ...], ...],
        routed_top_k: int,
        max_routing_rows: int,
        max_source_bytes: int,
    ) -> tuple[RoutingShardAssignmentQuery, ...]:
        """Return validated per-shard assignment summaries for one run."""
        ...


@runtime_checkable
class RoutingShardAppender(Protocol):
    """Protocol for appending routing shards without depending on the concrete store."""

    def append(
        self,
        result: RoutingForwardResult,
        *,
        store_token_text: bool = False,
    ) -> RoutingShardReceipt:
        """Append one complete routing result as an immutable content-addressed shard."""
        ...


@dataclass(frozen=True, slots=True)
class DuckDBRoutingShardStore:
    """DuckDB-backed routing shard store implementing both
    RoutingRunReader and RoutingShardAppender.

    Delegates all operations to the routing_shards module, which lazily
    imports duckdb only at call time.
    """

    workspace: Path

    @classmethod
    def bind(cls, workspace: str | Path) -> DuckDBRoutingShardStore:
        """Bind to an existing workspace directory.

        Args:
            workspace: Filesystem path to an existing routing workspace.

        Returns:
            A new DuckDBRoutingShardStore bound to the given workspace.

        Raises:
            TypeError: If workspace is not a str or Path.
        """
        if not isinstance(workspace, str | Path):
            raise TypeError(
                f"workspace must be a string or pathlib.Path, not {type(workspace).__name__}"
            )
        return cls(workspace=Path(workspace))

    def list_runs(
        self,
        *,
        max_runs: int,
        max_shards: int,
        max_event_rows: int,
        max_source_bytes: int,
    ) -> MixtralRoutingRunInventory:
        """List routing runs within the given budget bounds.

        Delegates to :func:`list_routing_runs`.
        """
        return list_routing_runs(
            self.workspace,
            max_runs=max_runs,
            max_shards=max_shards,
            max_event_rows=max_event_rows,
            max_source_bytes=max_source_bytes,
        )

    def list_shards(self, *, run_key: str) -> tuple[RoutingShardReceipt, ...]:
        """List all validated immutable routing shards for one run.

        Delegates to :func:`list_routing_shards`.
        """
        return list_routing_shards(self.workspace, run_key=run_key)

    def query_assignments(
        self,
        *,
        run_key: str,
        layer_keys: tuple[str, ...],
        expert_keys: tuple[tuple[str, ...], ...],
        routed_top_k: int,
        max_routing_rows: int,
        max_source_bytes: int,
    ) -> tuple[RoutingShardAssignmentQuery, ...]:
        """Return validated per-shard assignment summaries for one run.

        Opens and closes the bounded in-memory query connection around one
        delegated :func:`query_routing_run_assignments` call.
        """
        duckdb = _routing_shards._load_duckdb()
        connection = duckdb.connect(database=":memory:")
        try:
            return query_routing_run_assignments(
                self.workspace,
                run_key=run_key,
                layer_keys=layer_keys,
                expert_keys=expert_keys,
                routed_top_k=routed_top_k,
                max_routing_rows=max_routing_rows,
                max_source_bytes=max_source_bytes,
                duckdb=duckdb,
                connection=connection,
            )
        finally:
            connection.close()

    def append(
        self,
        result: RoutingForwardResult,
        *,
        store_token_text: bool = False,
    ) -> RoutingShardReceipt:
        """Append one complete routing result as an immutable content-addressed shard.

        Delegates to :func:`append_routing_shard`.
        """
        return append_routing_shard(
            self.workspace, result, store_token_text=store_token_text
        )


def reader_from_workspace(workspace: str | Path) -> RoutingRunReader:
    """Create a RoutingRunReader bound to the given workspace.

    Args:
        workspace: Filesystem path to an existing routing workspace.

    Returns:
        A DuckDBRoutingShardStore satisfying the RoutingRunReader protocol.
    """
    return DuckDBRoutingShardStore.bind(workspace)


__all__ = [
    "DuckDBRoutingShardStore",
    "RoutingRunReader",
    "RoutingShardAppender",
    "reader_from_workspace",
]