"""Command-line entry point for direct Phase 0 and resolved plan scans."""

from __future__ import annotations

import argparse
import json
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""

    parser = build_parser()
    args: Any = parser.parse_args(argv)
    if not hasattr(args, "handler"):
        parser.print_help()
        return 0
    return int(args.handler(args))
