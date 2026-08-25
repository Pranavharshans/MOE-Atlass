"""Model-neutral run identity, provenance, and lifecycle contracts.

These contracts describe run intent and lifecycle state only. They never load
a model, read a dataset, contact the network, execute a forward pass, or touch
storage. The prompt/dataset execution engine (later slice) drives these
contracts; the workspace/catalog persists them.
"""

from __future__ import annotations

from .lifecycle import (
    RUN_LIFECYCLE_SCHEMA_VERSION,
    RunAction,
    RunCancellation,
    RunFailure,
    RunLifecycleError,
    RunProgress,
    RunRecord,
    RunState,
    advance_progress,
    apply,
    can_transition,
    transition,
)
from .specs import (
    RUN_SPEC_SCHEMA_VERSION,
    AdapterProvenance,
    ChatMessage,
    DataProvenance,
    DatasetFormat,
    DatasetInputSpec,
    ExecutionEnvironment,
    GenerationConfig,
    InterventionLineage,
    ModelProvenance,
    PrivacyPolicy,
    ProbeProvenance,
    PromptInputSpec,
    RunInputKind,
    RunMode,
    RunSpecification,
    TokenTextPolicy,
    make_run_key,
    parse_probe_plan_id,
    parse_run_key,
    validate_run_name,
)

__all__ = [
    "RUN_LIFECYCLE_SCHEMA_VERSION",
    "RUN_SPEC_SCHEMA_VERSION",
    "AdapterProvenance",
    "ChatMessage",
    "DataProvenance",
    "DatasetFormat",
    "DatasetInputSpec",
    "ExecutionEnvironment",
    "GenerationConfig",
    "InterventionLineage",
    "ModelProvenance",
    "ProbeProvenance",
    "PrivacyPolicy",
    "PromptInputSpec",
    "RunAction",
    "RunCancellation",
    "RunFailure",
    "RunInputKind",
    "RunLifecycleError",
    "RunMode",
    "RunProgress",
    "RunRecord",
    "RunSpecification",
    "RunState",
    "TokenTextPolicy",
    "advance_progress",
    "apply",
    "can_transition",
    "make_run_key",
    "parse_probe_plan_id",
    "parse_run_key",
    "transition",
    "validate_run_name",
]
