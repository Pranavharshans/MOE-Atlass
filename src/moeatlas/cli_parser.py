"""Argument parser construction for the MoEAtlas CLI."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping

Handler = Callable[[argparse.Namespace], int]
_COMPARE_METRICS = ("count_deltas", "share_deltas", "ratio_deltas")


def build_parser(*, version: str, handlers: Mapping[str, Handler]) -> argparse.ArgumentParser:
    """Create the command-line parser without performing runtime imports."""

    parser = argparse.ArgumentParser(
        prog="moeatlas",
        description=(
            "Map, inspect, and understand Mixture-of-Experts models. "
            "Direct Phase 0 scanning supports only the explicit synthetic fixture; "
            "resolved plan-file scans use the runtime bridge."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version}",
    )

    subparsers = parser.add_subparsers(dest="command")
    doctor = subparsers.add_parser(
        "doctor",
        help="report model-free package and deferred validation status",
    )
    doctor.add_argument(
        "--json",
        action="store_true",
        help="emit the diagnostic report as JSON",
    )
    doctor.set_defaults(handler=handlers["doctor"])

    scan = subparsers.add_parser(
        "scan",
        help="emit a deterministic semantic discovery report",
        description=(
            "Run a deterministic semantic scanner. MODEL=fixture:synthetic is "
            "model-free; --loading-plan accepts one strictly validated HF/local "
            "LoadingPlan JSON document and delegates to the resolved runtime."
        ),
        epilog=(
            "Examples:\n"
            "  moeatlas scan fixture:synthetic\n"
            "  moeatlas scan fixture:synthetic --output report.json\n"
            "  moeatlas scan --loading-plan plan.json --output report.json\n\n"
            "MODEL and --loading-plan are mutually exclusive. A plan file must "
            "contain a resolved HuggingFaceSource or LocalSource; no model ID or "
            "local path is inferred by the CLI. Real checkpoint validation remains "
            "deferred to MV-01/MV-02 and the final VM."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scan.add_argument(
        "model",
        nargs="?",
        metavar="MODEL",
        help="source name; only fixture:synthetic is supported directly",
    )
    scan.add_argument(
        "--loading-plan",
        metavar="PLAN.json",
        help="read one validated HF/local LoadingPlan JSON document",
    )
    scan.add_argument(
        "--output",
        metavar="PATH",
        help="write the report atomically to PATH instead of stdout",
    )
    scan.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output file",
    )
    scan.set_defaults(handler=handlers["scan"], _scan_parser=scan)

    heatmap = subparsers.add_parser(
        "heatmap",
        help="aggregate one stored routing run and write a static HTML heatmap",
        description=(
            "Aggregate one complete, inspection-bound routing run and "
            "write a deterministic static HTML heatmap. All budgets are required "
            "canonical positive decimal integers."
        ),
        epilog=(
            "The inspection is a caller-created AdapterInspection.to_json() "
            "document or a universal structure inspection derived from a "
            "[STRUCTURE] discovery report (manifest_type "
            "\"universal_routing_inspection\"); all four budgets are required "
            "canonical positive decimals; --output must use the exact lowercase "
            ".html suffix. Existing output requires --force and publication "
            "reuses write_report_atomic(). The optional DuckDB store extra is "
            "required for committed routing shards; no model, browser, network, "
            "cache, or generation path is used."
        ),
    )
    heatmap.add_argument("workspace", metavar="WORKSPACE")
    heatmap.add_argument("--inspection", required=True, metavar="INSPECTION.json")
    heatmap.add_argument("--run-key", required=True, metavar="RUN_KEY")
    heatmap.add_argument(
        "--metric",
        required=True,
        choices=("assignment_counts", "assignment_shares", "load_ratios"),
    )
    heatmap.add_argument("--max-inspection-bytes", required=True, metavar="N")
    heatmap.add_argument("--max-routing-rows", required=True, metavar="N")
    heatmap.add_argument("--max-source-bytes", required=True, metavar="N")
    heatmap.add_argument("--max-matrix-cells", required=True, metavar="N")
    heatmap.add_argument("--output", required=True, metavar="OUTPUT.html")
    heatmap.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output file",
    )
    heatmap.set_defaults(handler=handlers["heatmap"], _heatmap_parser=heatmap)

    routing_runs = subparsers.add_parser(
        "routing-runs",
        help="inventory committed routing runs",
        description=(
            "Inventory immutable, inspection-free routing shards. All four "
            "budgets are required canonical positive decimal integers."
        ),
        epilog=(
            "The inventory is read-only and bounded by max-runs, max-shards, "
            "max-event-rows, and max-source-bytes. No model, tokenizer, network, "
            "cache, catalog, or generation path is used. --output uses the exact "
            "lowercase .json suffix and write_report_atomic()."
        ),
    )
    routing_runs.add_argument("workspace", metavar="WORKSPACE")
    routing_runs.add_argument("--max-runs", required=True, metavar="N")
    routing_runs.add_argument("--max-shards", required=True, metavar="N")
    routing_runs.add_argument("--max-event-rows", required=True, metavar="N")
    routing_runs.add_argument("--max-source-bytes", required=True, metavar="N")
    routing_runs.add_argument("--output", metavar="INVENTORY.json")
    routing_runs.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output file",
    )
    routing_runs.set_defaults(handler=handlers["routing_runs"], _routing_runs_parser=routing_runs)

    compare = subparsers.add_parser(
        "compare",
        help="compare two stored routing runs and write a static HTML delta report",
        description=(
            "Aggregate two complete, inspection-bound routing runs over one "
            "identical universe and write a deterministic static HTML comparison. "
            "All budgets are required canonical positive decimal integers."
        ),
        epilog=(
            "The inspection is a caller-created AdapterInspection.to_json() "
            "document or a universal structure inspection derived from a "
            "[STRUCTURE] discovery report (manifest_type "
            "\"universal_routing_inspection\"); all four budgets are required "
            "canonical positive decimals; the two run keys must differ; "
            "--output must use the exact lowercase .html suffix. Existing "
            "output requires --force and publication reuses "
            "write_report_atomic(). The optional DuckDB store extra is required "
            "for committed routing shards; no model, browser, network, cache, "
            "or generation path is used."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare.add_argument("workspace", metavar="WORKSPACE")
    compare.add_argument("--inspection", required=True, metavar="INSPECTION.json")
    compare.add_argument("--baseline-run-key", required=True, metavar="RUN_KEY")
    compare.add_argument("--comparison-run-key", required=True, metavar="RUN_KEY")
    compare.add_argument("--metric", required=True, choices=_COMPARE_METRICS)
    compare.add_argument("--max-inspection-bytes", required=True, metavar="N")
    compare.add_argument("--max-routing-rows", required=True, metavar="N")
    compare.add_argument("--max-source-bytes", required=True, metavar="N")
    compare.add_argument("--max-matrix-cells", required=True, metavar="N")
    compare.add_argument("--output", required=True, metavar="OUTPUT.html")
    compare.add_argument(
        "--force",
        action="store_true",
        help="replace an existing --output file",
    )
    compare.set_defaults(handler=handlers["compare"], _compare_parser=compare)

    adapters = subparsers.add_parser(
        "adapters",
        help="inspect the adapter plugin registry",
        description=(
            "List built-in adapters and third-party plugins published under "
            "the moeatlas.adapters entry-point group through one contract."
        ),
    )
    adapters_subparsers = adapters.add_subparsers(dest="adapters_command")
    adapters_list = adapters_subparsers.add_parser(
        "list",
        help="list registered adapter plugins with provenance and status",
        description=(
            "List every registered adapter plugin with provenance, trust "
            "policy status, and declared architecture families. Discovery "
            "is metadata-only; no model is loaded and nothing is downloaded."
        ),
        epilog=(
            "Policy flags compose: --builtin-only restricts trusted sources, "
            "--enable allow-lists specific plugins, --disable force-disables "
            "names, and --family keeps only enabled records serving one "
            "architecture family. A name passed to both --enable and "
            "--disable is rejected."
        ),
    )
    adapters_list.add_argument(
        "--json",
        action="store_true",
        help="emit the canonical moeatlas.adapter_registry document",
    )
    adapters_list.add_argument(
        "--builtin-only",
        action="store_true",
        help="treat entry-point plugins as untrusted (listed as disabled)",
    )
    adapters_list.add_argument(
        "--enable",
        action="append",
        metavar="NAME",
        default=[],
        help="allow-list one plugin name (repeatable)",
    )
    adapters_list.add_argument(
        "--disable",
        action="append",
        metavar="NAME",
        default=[],
        help="force-disable one plugin name (repeatable)",
    )
    adapters_list.add_argument(
        "--family",
        metavar="FAMILY",
        help="keep only enabled records serving this architecture family",
    )
    adapters_list.set_defaults(handler=handlers["adapters_list"])

    run = subparsers.add_parser(
        "run",
        help="execute one headless run through the shared run service",
        description=(
            "Build a content-addressed RunSpecification from one validated "
            "loading plan plus exactly one input form (--prompt TEXT or "
            "--dataset DESCRIPTOR.json) and execute it through the shared "
            "run service with an executor plugin from the moeatlas.executors "
            "entry-point group."
        ),
        epilog=(
            "The executor is mandatory: the built-in routing executor drives "
            "a real resolved model, and additional executors publish as "
            "plugins in the moeatlas.executors entry-point group. No command "
            "downloads a model implicitly. Dataset locations resolve "
            "relative to the workspace directory. Timestamps come only from "
            "--at; the CLI never reads a clock."
        ),
    )
    run.add_argument("workspace", metavar="WORKSPACE")
    run.add_argument("--loading-plan", required=True, metavar="PLAN.json")
    run.add_argument("--prompt", metavar="TEXT")
    run.add_argument("--dataset", metavar="DESCRIPTOR.json")
    run.add_argument("--executor", required=True, metavar="NAME")
    run.add_argument("--at", metavar="TIMESTAMP", help="caller-supplied timestamp")
    run.add_argument("--created-by", metavar="LABEL")
    run.add_argument("--checkpoint-directory", metavar="DIR")
    run.add_argument("--resume-from", metavar="CHECKPOINT.json")
    run.add_argument("--max-input-bytes", metavar="N")
    run.add_argument("--max-rows", metavar="N")
    run.add_argument("--max-row-bytes", metavar="N")
    run.add_argument("--max-result-bytes", metavar="N")
    run.set_defaults(handler=handlers["run"])

    export = subparsers.add_parser(
        "export",
        help="export one run's committed evidence as a canonical bundle",
        description=(
            "Export every committed shard of one routing run as a bounded, "
            "tamper-evident bundle directory with a manifest written last."
        ),
        epilog=(
            "The destination must be nonexistent or an empty real directory. "
            "Two exports of the same committed run state are byte-identical. "
            "No model, tokenizer, network, cache, or generation path is used."
        ),
    )
    export.add_argument("workspace", metavar="WORKSPACE")
    export.add_argument("run_key", metavar="RUN_KEY")
    export.add_argument(
        "--format",
        default="bundle",
        choices=("bundle",),
        help="evidence format (only the canonical bundle exists today)",
    )
    export.add_argument("--output", required=True, metavar="DEST_DIR")
    export.add_argument("--max-event-rows", metavar="N")
    export.add_argument("--max-file-bytes", metavar="N")
    export.set_defaults(handler=handlers["export"])

    ui = subparsers.add_parser(
        "ui",
        help="launch the local read-only server for one workspace",
        description=(
            "Serve the local read-only API (health, workspace snapshot, "
            "bounded run listings, adapter registry) for one workspace."
        ),
        epilog=(
            "The server binds to the loopback interface by default and never "
            "loads a model, downloads anything, or writes to storage. Remote "
            "binding requires an explicit opt-in."
        ),
    )
    ui.add_argument("workspace", metavar="WORKSPACE")
    ui.add_argument("--host", default="127.0.0.1", metavar="HOST")
    ui.add_argument("--port", default="8000", metavar="N")
    ui.add_argument(
        "--allow-remote",
        action="store_true",
        help="explicitly allow binding to a non-loopback interface",
    )
    ui.set_defaults(handler=handlers["ui"])
    return parser


__all__ = ["build_parser"]
