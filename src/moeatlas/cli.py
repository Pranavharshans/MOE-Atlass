"""Command-line entry point for direct Phase 0 and resolved plan scans."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import PRODUCT_NAME, __version__
from .diagnostics import collect_doctor_report
from .loading import HuggingFaceSource, LoadingPlan, LocalSource
from .scan import ScanOutputError, ScanSourceError, report_payload, scan_source, write_report_atomic


class _LoadingPlanInputError(ValueError):
    """Raised when a plan file cannot be safely accepted by the CLI."""


class _HeatmapInputError(ValueError):
    """Raised when a heatmap input cannot be safely accepted by the CLI."""


_CANONICAL_DECIMAL = re.compile(r"^[1-9][0-9]*$")
_HEATMAP_LOAD_STAGES = frozenset({"inspection", "budget", "source", "query"})
_HEATMAP_SHARD_STAGES = frozenset(
    {"dependency", "workspace", "write", "publish", "reopen", "conflict"}
)
_INVENTORY_STAGES = frozenset({"budget", "index"})
_COMPARE_METRICS = ("count_deltas", "share_deltas", "ratio_deltas")


def build_parser() -> argparse.ArgumentParser:
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
        version=f"%(prog)s {__version__}",
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
    doctor.set_defaults(handler=_handle_doctor)

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
    scan.set_defaults(handler=_handle_scan, _scan_parser=scan)

    heatmap = subparsers.add_parser(
        "heatmap",
        help="aggregate one stored routing run and write a static HTML heatmap",
        description=(
            "Aggregate one complete, inspection-bound routing run and "
            "write a deterministic static HTML heatmap. All budgets are required "
            "canonical positive decimal integers."
        ),
        epilog=(
            "The inspection is a caller-created AdapterInspection.to_json() document; "
            "all four budgets are required canonical positive decimals; --output must "
            "use the exact lowercase .html suffix. Existing output requires --force "
            "and publication reuses write_report_atomic(). The optional DuckDB store "
            "extra is required for committed routing shards; no model, browser, "
            "network, cache, or generation path is used."
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
    heatmap.set_defaults(handler=_handle_heatmap, _heatmap_parser=heatmap)

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
    routing_runs.set_defaults(handler=_handle_routing_runs, _routing_runs_parser=routing_runs)

    compare = subparsers.add_parser(
        "compare",
        help="compare two stored routing runs and write a static HTML delta report",
        description=(
            "Aggregate two complete, inspection-bound routing runs over one "
            "identical universe and write a deterministic static HTML comparison. "
            "All budgets are required canonical positive decimal integers."
        ),
        epilog=(
            "The inspection is a caller-created AdapterInspection.to_json() document; "
            "all four budgets are required canonical positive decimals; the two run "
            "keys must differ; --output must use the exact lowercase .html suffix. "
            "Existing output requires --force and publication reuses "
            "write_report_atomic(). The optional DuckDB store extra is required for "
            "committed routing shards; no model, browser, network, cache, or "
            "generation path is used."
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
    compare.set_defaults(handler=_handle_compare, _compare_parser=compare)

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
    adapters_list.set_defaults(handler=_handle_adapters_list)

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
            "The executor is mandatory and never built in: real model "
            "execution belongs to adapter plugins, and no command downloads "
            "a model implicitly. Dataset locations resolve relative to the "
            "workspace directory. Timestamps come only from --at; the CLI "
            "never reads a clock."
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
    run.set_defaults(handler=_handle_run)

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
    export.set_defaults(handler=_handle_export)

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
    ui.set_defaults(handler=_handle_ui)
    return parser


def _handle_doctor(args: argparse.Namespace) -> int:
    report = collect_doctor_report()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    print(f"{PRODUCT_NAME} {report['package_version']}")
    python_info = report["python"]
    support = "supported" if python_info["supported"] else "unsupported"
    print(f"Python: {python_info['version']} ({support}; {python_info['implementation']})")
    print("Optional runtime packages (presence only; no imports performed):")
    for name, info in report["optional_runtime_packages"].items():
        status = "available" if info["available"] else "not installed"
        print(f"  - {name}: {status}")

    validation = report["validation"]["model_and_gpu"]
    print(f"Model/GPU validation: {validation['status']}")
    print(f"  {validation['reason']}")
    return 0


def _read_loading_plan(path: str) -> LoadingPlan:
    """Read and strictly validate one plan without exposing input details."""

    try:
        payload = Path(path).read_bytes()
    except (OSError, ValueError) as exc:
        raise _LoadingPlanInputError("could not read loading plan") from exc
    try:
        return LoadingPlan.from_json(payload)
    except Exception as exc:
        raise _LoadingPlanInputError("loading plan is not a valid LoadingPlan document") from exc


def _preflight_loading_plan(plan: LoadingPlan) -> None:
    """Reject unsupported or unresolved plans before optional runtime dispatch."""

    if not isinstance(plan.source, HuggingFaceSource | LocalSource):
        raise _LoadingPlanInputError(
            "loading-plan scan supports only HuggingFaceSource and LocalSource"
        )
    if plan.resolution is None:
        raise _LoadingPlanInputError(
            "loading plan must contain immutable model and tokenizer resolution evidence"
        )
    if plan.source.tokenizer is None:
        raise _LoadingPlanInputError(
            "loading plan must contain a tokenizer request and immutable tokenizer resolution"
        )
    if (
        plan.resolution.resolved_tokenizer_revision is None
        or plan.resolution.resolved_tokenizer_revision_evidence is None
    ):
        raise _LoadingPlanInputError(
            "loading plan must contain immutable tokenizer resolution evidence"
        )


def _load_and_scan_plan(plan: LoadingPlan):
    """Import the public runtime bridge only after plan preflight succeeds."""

    from .runtime import load_and_scan

    return load_and_scan(plan)


def _handle_scan(args: argparse.Namespace) -> int:
    if args.force and args.output is None:
        print("moeatlas scan: --force requires --output PATH", file=sys.stderr)
        return 2

    if args.model is None and args.loading_plan is None:
        args._scan_parser.error("one of MODEL or --loading-plan is required")
    if args.model is not None and args.loading_plan is not None:
        print("moeatlas scan: MODEL and --loading-plan are mutually exclusive", file=sys.stderr)
        return 2

    try:
        if args.loading_plan is not None:
            plan = _read_loading_plan(args.loading_plan)
            _preflight_loading_plan(plan)
            for warning in plan.security_warnings:
                print(f"moeatlas scan: warning: {warning}", file=sys.stderr)
            report = _load_and_scan_plan(plan)
        else:
            report = scan_source(args.model)
        payload = report_payload(report)
        if args.output is None:
            sys.stdout.write(payload)
            return 0
        output = write_report_atomic(payload, args.output, force=args.force)
    except _LoadingPlanInputError as exc:
        print(f"moeatlas scan: {exc}", file=sys.stderr)
        return 2
    except (ScanSourceError, ScanOutputError) as exc:
        print(f"moeatlas scan: {exc}", file=sys.stderr)
        return 2
    except Exception:
        print("moeatlas scan: loading or report generation failed", file=sys.stderr)
        return 2

    print(f"saved scan report to {output}", file=sys.stderr)
    return 0


def _parse_heatmap_budget(raw: object, field_name: str) -> int:
    """Parse one required positive canonical decimal without accepting variants."""

    if type(raw) is not str or _CANONICAL_DECIMAL.fullmatch(raw) is None:
        raise _HeatmapInputError(
            f"--{field_name.replace('_', '-')} must be a canonical positive decimal integer"
        )
    try:
        value = int(raw, 10)
    except (ValueError, OverflowError) as exc:
        raise _HeatmapInputError(
            f"--{field_name.replace('_', '-')} must be a canonical positive decimal integer"
        ) from exc
    if value > sys.maxsize:
        raise _HeatmapInputError(f"--{field_name.replace('_', '-')} is too large for this platform")
    return value


def _preflight_heatmap_output(raw_output: object, *, force: object) -> Path:
    """Validate output destination before reading input or importing analysis/store."""

    if type(raw_output) is not str or not raw_output:
        raise _HeatmapInputError("--output must be a non-empty path")
    if type(force) is not bool:
        raise _HeatmapInputError("--force must be a boolean")
    try:
        target = Path(raw_output)
        if target.suffix != ".html":
            raise _HeatmapInputError("output must have the exact .html suffix")
        if os.path.lexists(target):
            if target.is_dir():
                raise _HeatmapInputError("output path is a directory")
            if not force:
                raise _HeatmapInputError("output already exists; pass --force to replace it")
        parent = target.parent
        if not parent.exists():
            raise _HeatmapInputError("output parent does not exist")
        if not parent.is_dir():
            raise _HeatmapInputError("output parent is not a directory")
    except _HeatmapInputError:
        raise
    except (OSError, ValueError) as exc:
        raise _HeatmapInputError("output path is not usable") from exc
    return target


def _read_heatmap_inspection(raw_path: object, max_bytes: int):
    """Read at most the inspection budget from one non-symlink JSON file."""

    if type(raw_path) is not str or not raw_path:
        raise _HeatmapInputError("--inspection must be a non-empty path")
    try:
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise _HeatmapInputError("inspection must be a regular non-symlink file")
        initial_size = path.stat().st_size
        if initial_size > max_bytes:
            raise _HeatmapInputError("inspection exceeds --max-inspection-bytes")
        with path.open("rb") as stream:
            payload = stream.read(max_bytes + 1)
        final_size = path.stat().st_size
    except _HeatmapInputError:
        raise
    except (OSError, ValueError, OverflowError) as exc:
        raise _HeatmapInputError("inspection could not be read") from exc
    if len(payload) > max_bytes or final_size > max_bytes:
        raise _HeatmapInputError("inspection exceeds --max-inspection-bytes")

    try:
        from .adapters import AdapterInspection

        return AdapterInspection.from_json(payload)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise _HeatmapInputError("inspection is not a valid AdapterInspection document") from exc


def _run_heatmap_analysis(
    workspace: str | Path,
    inspection: object,
    *,
    run_key: str,
    metric: str,
    max_routing_rows: int,
    max_source_bytes: int,
    max_matrix_cells: int,
) -> str:
    """Lazy, exactly-once aggregation and rendering for the heatmap command."""

    from .analysis import aggregate_routing_load, render_routing_load_heatmap

    matrix = aggregate_routing_load(
        workspace,
        inspection,
        run_key=run_key,
        max_routing_rows=max_routing_rows,
        max_source_bytes=max_source_bytes,
        max_matrix_cells=max_matrix_cells,
    )
    return render_routing_load_heatmap(
        matrix,
        metric=metric,
        max_cells=max_matrix_cells,
    )


def _safe_heatmap_failure(exc: Exception) -> str:
    """Return only exact fixed-stage messages from accepted analysis/store errors."""

    from .analysis import RoutingLoadError
    from .store import RoutingShardError

    if type(exc) is RoutingLoadError and exc.stage in _HEATMAP_LOAD_STAGES:
        message = f"routing load aggregation failed at {exc.stage}"
        if str(exc) == message:
            return message
    if type(exc) is RoutingShardError and exc.stage in _HEATMAP_SHARD_STAGES:
        message = f"routing shard failed at {exc.stage}"
        if str(exc) == message:
            return message
    return "heatmap generation failed"


def _handle_heatmap(args: argparse.Namespace) -> int:
    try:
        budgets = {
            name: _parse_heatmap_budget(getattr(args, name), name)
            for name in (
                "max_inspection_bytes",
                "max_routing_rows",
                "max_source_bytes",
                "max_matrix_cells",
            )
        }
        output_path = _preflight_heatmap_output(args.output, force=args.force)
        inspection = _read_heatmap_inspection(args.inspection, budgets["max_inspection_bytes"])
        payload = _run_heatmap_analysis(
            args.workspace,
            inspection,
            run_key=args.run_key,
            metric=args.metric,
            max_routing_rows=budgets["max_routing_rows"],
            max_source_bytes=budgets["max_source_bytes"],
            max_matrix_cells=budgets["max_matrix_cells"],
        )
        output = write_report_atomic(payload, output_path, force=args.force)
    except _HeatmapInputError as exc:
        print(f"moeatlas heatmap: {exc}", file=sys.stderr)
        return 2
    except ScanOutputError as exc:
        print(f"moeatlas heatmap: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"moeatlas heatmap: {_safe_heatmap_failure(exc)}", file=sys.stderr)
        return 2

    print(f"saved routing heatmap to {output}", file=sys.stderr)
    return 0


def _run_compare_analysis(
    workspace: str | Path,
    inspection: object,
    *,
    baseline_run_key: str,
    comparison_run_key: str,
    metric: str,
    max_routing_rows: int,
    max_source_bytes: int,
    max_matrix_cells: int,
) -> str:
    """Lazy, exactly-once aggregation, comparison, and rendering."""

    from .analysis import (
        aggregate_routing_load,
        compare_routing_load,
        render_routing_load_comparison,
    )

    aggregate = aggregate_routing_load
    baseline = aggregate(
        workspace,
        inspection,
        run_key=baseline_run_key,
        max_routing_rows=max_routing_rows,
        max_source_bytes=max_source_bytes,
        max_matrix_cells=max_matrix_cells,
    )
    comparison = aggregate(
        workspace,
        inspection,
        run_key=comparison_run_key,
        max_routing_rows=max_routing_rows,
        max_source_bytes=max_source_bytes,
        max_matrix_cells=max_matrix_cells,
    )
    value = compare_routing_load(baseline, comparison, max_cells=max_matrix_cells)
    return render_routing_load_comparison(
        value,
        metric=metric,
        max_cells=max_matrix_cells,
    )


def _safe_compare_failure(exc: Exception) -> str:
    """Return only exact fixed-stage messages from accepted analysis/store errors."""

    from .analysis import RoutingLoadError
    from .store import RoutingShardError

    if type(exc) is RoutingLoadError and exc.stage in _HEATMAP_LOAD_STAGES:
        message = f"routing load aggregation failed at {exc.stage}"
        if str(exc) == message:
            return message
    if type(exc) is RoutingShardError and exc.stage in _HEATMAP_SHARD_STAGES:
        message = f"routing shard failed at {exc.stage}"
        if str(exc) == message:
            return message
    return "routing comparison failed"


def _handle_compare(args: argparse.Namespace) -> int:
    try:
        if args.baseline_run_key == args.comparison_run_key:
            raise _HeatmapInputError("baseline and comparison run keys must differ")
        budgets = {
            name: _parse_heatmap_budget(getattr(args, name), name)
            for name in (
                "max_inspection_bytes",
                "max_routing_rows",
                "max_source_bytes",
                "max_matrix_cells",
            )
        }
        output_path = _preflight_heatmap_output(args.output, force=args.force)
        inspection = _read_heatmap_inspection(args.inspection, budgets["max_inspection_bytes"])
        payload = _run_compare_analysis(
            args.workspace,
            inspection,
            baseline_run_key=args.baseline_run_key,
            comparison_run_key=args.comparison_run_key,
            metric=args.metric,
            max_routing_rows=budgets["max_routing_rows"],
            max_source_bytes=budgets["max_source_bytes"],
            max_matrix_cells=budgets["max_matrix_cells"],
        )
        output = write_report_atomic(payload, output_path, force=args.force)
    except _HeatmapInputError as exc:
        print(f"moeatlas compare: {exc}", file=sys.stderr)
        return 2
    except ScanOutputError as exc:
        print(f"moeatlas compare: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"moeatlas compare: {_safe_compare_failure(exc)}", file=sys.stderr)
        return 2

    print(f"saved routing comparison to {output}", file=sys.stderr)
    return 0


def _parse_inventory_budget(raw: object, field_name: str) -> int:
    if type(raw) is not str or _CANONICAL_DECIMAL.fullmatch(raw) is None:
        raise _HeatmapInputError(
            f"--{field_name.replace('_', '-')} must be a canonical positive decimal integer"
        )
    try:
        value = int(raw, 10)
    except (ValueError, OverflowError) as exc:
        raise _HeatmapInputError(
            f"--{field_name.replace('_', '-')} must be a canonical positive decimal integer"
        ) from exc
    if value > sys.maxsize:
        raise _HeatmapInputError(f"--{field_name.replace('_', '-')} is too large for this platform")
    return value


def _preflight_routing_runs_output(raw_output: object, *, force: object) -> Path | None:
    if raw_output is None:
        if force:
            raise _HeatmapInputError("--force requires --output PATH")
        return None
    if type(raw_output) is not str or not raw_output:
        raise _HeatmapInputError("--output must be a non-empty path")
    if type(force) is not bool:
        raise _HeatmapInputError("--force must be a boolean")
    try:
        target = Path(raw_output)
        if target.suffix != ".json":
            raise _HeatmapInputError("output must have the exact .json suffix")
        if os.path.lexists(target):
            if target.is_dir():
                raise _HeatmapInputError("output path is a directory")
            if not force:
                raise _HeatmapInputError("output already exists; pass --force to replace it")
        parent = target.parent
        if not parent.exists():
            raise _HeatmapInputError("output parent does not exist")
        if not parent.is_dir():
            raise _HeatmapInputError("output parent is not a directory")
    except _HeatmapInputError:
        raise
    except (OSError, ValueError) as exc:
        raise _HeatmapInputError("output path is not usable") from exc
    return target


def _run_routing_run_inventory(
    workspace: str,
    *,
    max_runs: int,
    max_shards: int,
    max_event_rows: int,
    max_source_bytes: int,
):
    from .store import list_mixtral_routing_runs

    return list_mixtral_routing_runs(
        workspace,
        max_runs=max_runs,
        max_shards=max_shards,
        max_event_rows=max_event_rows,
        max_source_bytes=max_source_bytes,
    )


def _write_routing_runs_report(payload: str, output: Path, *, force: bool) -> Path:
    """Delegate inventory publication to the existing atomic report writer."""

    writer = write_report_atomic
    return writer(payload, output, force=force)


def _safe_routing_runs_failure(exc: Exception) -> str:
    from .store import RoutingRunInventoryError, RoutingShardError

    if type(exc) is RoutingRunInventoryError and exc.stage in _INVENTORY_STAGES:
        message = f"routing run inventory failed at {exc.stage}"
        if str(exc) == message:
            return message
    if type(exc) is RoutingShardError and exc.stage in _HEATMAP_SHARD_STAGES:
        message = f"routing shard failed at {exc.stage}"
        if str(exc) == message:
            return message
    return "routing run inventory failed"


def _handle_routing_runs(args: argparse.Namespace) -> int:
    try:
        budgets = {
            name: _parse_inventory_budget(getattr(args, name), name)
            for name in ("max_runs", "max_shards", "max_event_rows", "max_source_bytes")
        }
        output_path = _preflight_routing_runs_output(args.output, force=args.force)
        inventory = _run_routing_run_inventory(
            args.workspace,
            max_runs=budgets["max_runs"],
            max_shards=budgets["max_shards"],
            max_event_rows=budgets["max_event_rows"],
            max_source_bytes=budgets["max_source_bytes"],
        )
        payload = inventory.to_json() + "\n"
        if output_path is None:
            sys.stdout.write(payload)
            return 0
        output = _write_routing_runs_report(payload, output_path, force=args.force)
    except _HeatmapInputError as exc:
        print(f"moeatlas routing-runs: {exc}", file=sys.stderr)
        return 2
    except ScanOutputError as exc:
        print(f"moeatlas routing-runs: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"moeatlas routing-runs: {_safe_routing_runs_failure(exc)}", file=sys.stderr)
        return 2

    print(f"saved routing run inventory to {output}", file=sys.stderr)
    return 0


def _handle_adapters_list(args: argparse.Namespace) -> int:
    try:
        from .adapters import (
            AdapterRegistryPolicy,
            collect_adapter_registry,
            match_adapters_for_family,
        )

        policy = AdapterRegistryPolicy(
            trusted_sources=("builtin",) if args.builtin_only else ("builtin", "entry_point"),
            enabled_names=tuple(sorted(set(args.enable))) if args.enable else None,
            disabled_names=tuple(sorted(set(args.disable))),
        )
        report = collect_adapter_registry(policy=policy)
        if args.family is not None:
            matching = match_adapters_for_family(report.entries, family=args.family)
            report = type(report)(
                schema_version=report.schema_version,
                entries=matching,
                collisions=report.collisions,
                failures=report.failures,
            )
    except (TypeError, ValueError):
        print(
            "moeatlas adapters list: invalid adapter policy or family filter",
            file=sys.stderr,
        )
        return 2

    if args.json:
        sys.stdout.write(report.to_json() + "\n")
        return 0

    enabled = sum(1 for entry in report.entries if entry.status == "enabled")
    print(
        f"adapter plugins: {len(report.entries)} listed "
        f"({enabled} enabled, {len(report.entries) - enabled} disabled)"
    )
    for entry in report.entries:
        record = entry.record
        distribution = record.distribution if record.distribution is not None else "-"
        families = ",".join(record.architecture_families)
        print(
            f"{record.name} {record.version} {entry.status} {record.source} "
            f"{distribution} families={families}"
        )
    for name, kept, suppressed in report.collisions:
        print(
            f"collision: {name} kept {kept}; suppressed {suppressed}",
            file=sys.stderr,
        )
    for value, reason in report.failures:
        print(f"plugin failure: {value}: {reason}", file=sys.stderr)
    return 0


def _handle_adapters_list(args: argparse.Namespace) -> int:
    try:
        from .adapters import (
            AdapterRegistryPolicy,
            collect_adapter_registry,
            match_adapters_for_family,
        )

        policy = AdapterRegistryPolicy(
            trusted_sources=("builtin",) if args.builtin_only else ("builtin", "entry_point"),
            enabled_names=tuple(sorted(set(args.enable))) if args.enable else None,
            disabled_names=tuple(sorted(set(args.disable))),
        )
        report = collect_adapter_registry(policy=policy)
        if args.family is not None:
            matching = match_adapters_for_family(report.entries, family=args.family)
            report = type(report)(
                schema_version=report.schema_version,
                entries=matching,
                collisions=report.collisions,
                failures=report.failures,
            )
    except (TypeError, ValueError):
        print(
            "moeatlas adapters list: invalid adapter policy or family filter",
            file=sys.stderr,
        )
        return 2

    if args.json:
        sys.stdout.write(report.to_json() + "\n")
        return 0

    enabled = sum(1 for entry in report.entries if entry.status == "enabled")
    print(
        f"adapter plugins: {len(report.entries)} listed "
        f"({enabled} enabled, {len(report.entries) - enabled} disabled)"
    )
    for entry in report.entries:
        record = entry.record
        distribution = record.distribution if record.distribution is not None else "-"
        families = ",".join(record.architecture_families)
        print(
            f"{record.name} {record.version} {entry.status} {record.source} "
            f"{distribution} families={families}"
        )
    for name, kept, suppressed in report.collisions:
        print(
            f"collision: {name} kept {kept}; suppressed {suppressed}",
            file=sys.stderr,
        )
    for value, reason in report.failures:
        print(f"plugin failure: {value}: {reason}", file=sys.stderr)
    return 0


_RUN_SERVICE_STAGES = frozenset({"checkpoint", "lifecycle"})
_EXPORT_BUNDLE_STAGES = frozenset(
    {"dependency", "workspace", "source", "format", "budget", "write", "publish", "conflict"}
)
_EXECUTOR_ENTRY_POINT_GROUP = "moeatlas.executors"


class _RunInputError(ValueError):
    """Raised when run inputs cannot be safely accepted by the CLI."""


def _load_executor(name: str):
    """Resolve one executor plugin from the moeatlas.executors group."""

    from importlib.metadata import entry_points

    if not name or name != name.strip():
        raise _RunInputError("--executor must be a non-empty name")
    try:
        candidates = entry_points(group=_EXECUTOR_ENTRY_POINT_GROUP)
    except Exception as exc:
        raise _RunInputError("executor registry could not be read") from exc
    for candidate in candidates:
        if candidate.name != name:
            continue
        try:
            executor = candidate.load()
        except Exception as exc:
            raise _RunInputError("executor plugin failed to load") from exc
        if not callable(executor):
            raise _RunInputError("executor plugin is not callable")
        return executor
    raise _RunInputError("executor plugin is not registered")


def _read_run_loading_plan(path: str):
    try:
        payload = Path(path).read_bytes()
    except (OSError, ValueError) as exc:
        raise _RunInputError("could not read loading plan") from exc
    try:
        from .loading import LoadingPlan

        return LoadingPlan.from_json(payload)
    except Exception as exc:
        raise _RunInputError("loading plan is not a valid LoadingPlan document") from exc


def _build_run_input(args: argparse.Namespace):
    if (args.prompt is None) == (args.dataset is None):
        raise _RunInputError("exactly one of --prompt or --dataset is required")
    try:
        from .runs.specs import DatasetInputSpec, PromptInputSpec

        if args.prompt is not None:
            return PromptInputSpec(text=args.prompt)
        payload = Path(args.dataset).read_bytes()
        return DatasetInputSpec.model_validate_json(payload)
    except _RunInputError:
        raise
    except (OSError, ValueError) as exc:
        raise _RunInputError("dataset descriptor is not a valid document") from exc


def _safe_run_failure(exc: Exception) -> str:
    from .services import RunServiceError

    if type(exc) is RunServiceError and exc.stage in _RUN_SERVICE_STAGES:
        message = f"run service failed at {exc.stage}"
        if str(exc) == message:
            return message
    return "run execution failed"


def _handle_run(args: argparse.Namespace) -> int:
    try:
        budgets = {
            name: _parse_heatmap_budget(getattr(args, name), name)
            for name in ("max_input_bytes", "max_rows", "max_row_bytes", "max_result_bytes")
            if getattr(args, name) is not None
        }
        plan = _read_run_loading_plan(args.loading_plan)
        _preflight_loading_plan(plan)
        input_spec = _build_run_input(args)
        executor = _load_executor(args.executor)

        from .runs.specs import DataProvenance, ModelProvenance, RunSpecification
        from .services import execute_specification, publish_run_report
        from .services.workspace import open_workspace

        open_workspace(args.workspace)
        specification = RunSpecification(
            model=ModelProvenance(
                loading_plan_id=plan.plan_id,
                model_id=plan.resolution.model_id,
                model_revision=plan.resolution.resolved_model_revision,
                tokenizer_revision=plan.resolution.resolved_tokenizer_revision,
            ),
            data=DataProvenance(input=input_spec),
            created_by=args.created_by,
        )
        report = execute_specification(
            specification,
            executor=executor,
            base_directory=args.workspace,
            at=args.at,
            checkpoint_directory=args.checkpoint_directory,
            resume_from=args.resume_from,
            **budgets,
        )
        publish_run_report(args.workspace, report, at=args.at)
    except _RunInputError as exc:
        print(f"moeatlas run: {exc}", file=sys.stderr)
        return 2
    except _HeatmapInputError as exc:
        print(f"moeatlas run: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"moeatlas run: {_safe_run_failure(exc)}", file=sys.stderr)
        return 2

    final = report.final_record
    progress = final.progress
    completed = progress.completed_units if progress is not None else 0
    total = progress.total_units if progress is not None else 0
    print(f"run {report.run_key} {final.state.value}")
    print(f"rows: {completed} of {total} units; resumed from batch {report.resumed_from_batch}")
    if report.checkpoint_path is not None:
        print(f"checkpoint: {report.checkpoint_path}")
    return 0


def _safe_export_failure(exc: Exception) -> str:
    from .store import RunBundleError

    if type(exc) is RunBundleError and exc.stage in _EXPORT_BUNDLE_STAGES:
        message = f"run bundle failed at {exc.stage}"
        if str(exc) == message:
            return message
    return "run export failed"


def _handle_export(args: argparse.Namespace) -> int:
    try:
        budgets = {
            name: _parse_heatmap_budget(getattr(args, name), name)
            for name in ("max_event_rows", "max_file_bytes")
            if getattr(args, name) is not None
        }
        from .store import export_run_bundle

        receipt = export_run_bundle(
            args.workspace,
            args.output,
            run_key=args.run_key,
            **budgets,
        )
    except _HeatmapInputError as exc:
        print(f"moeatlas export: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        print(f"moeatlas export: {_safe_export_failure(exc)}", file=sys.stderr)
        return 2

    print(
        f"exported run {receipt.run_key} to {args.output} "
        f"({receipt.shard_count} shards, {receipt.token_count} token events, "
        f"{receipt.routing_count} routing events)"
    )
    print(f"manifest sha256: {receipt.manifest_sha256}")
    return 0


def _run_ui_server(app: object, *, host: str, port: int) -> None:
    """Serve the app until interrupted; isolated for testing."""

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="warning")


def _handle_ui(args: argparse.Namespace) -> int:
    host = args.host
    if not (
        host == "localhost"
        or host.startswith("127.")
        or host == "::1"
        or host == "[::1]"
    ):
        if not args.allow_remote:
            print(
                "moeatlas ui: non-loopback hosts require --allow-remote",
                file=sys.stderr,
            )
            return 2
    if type(args.port) is not str or _CANONICAL_DECIMAL.fullmatch(args.port) is None:
        print("moeatlas ui: --port must be a canonical positive decimal integer", file=sys.stderr)
        return 2
    port = int(args.port, 10)
    if port > 65535:
        print("moeatlas ui: --port must be at most 65535", file=sys.stderr)
        return 2
    try:
        from .server import ServerDependencyError, create_app

        app = create_app(args.workspace)
    except ServerDependencyError:
        print(
            "moeatlas ui: server dependency 'fastapi' is not installed; "
            "install moeatlas with the 'server' extra",
            file=sys.stderr,
        )
        return 2
    try:
        _run_ui_server(app, host=host, port=port)
    except KeyboardInterrupt:
        pass
    print(f"served workspace {args.workspace} at http://{host}:{port}", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args: Any = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args))
