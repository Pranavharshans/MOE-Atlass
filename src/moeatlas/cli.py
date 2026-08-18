"""Command-line entry point for the MoEAtlas foundation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from . import PRODUCT_NAME, __version__
from .diagnostics import collect_doctor_report
from .scan import ScanOutputError, ScanSourceError, report_payload, scan_source, write_report_atomic


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser without performing runtime imports."""

    parser = argparse.ArgumentParser(
        prog="moeatlas",
        description=(
            "Map, inspect, and understand Mixture-of-Experts models. "
            "Phase 0 scanning supports only the explicit synthetic fixture."
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
        help="emit a deterministic Phase 0 semantic discovery report",
        description=(
            "Run the model-free Phase 0 semantic scanner. The only supported "
            "source is MODEL=fixture:synthetic. HF/local model loading is deferred "
            "to Phase 1 (MV-01/MV-02)."
        ),
        epilog=(
            "Examples:\n"
            "  moeatlas scan fixture:synthetic\n"
            "  moeatlas scan fixture:synthetic --output report.json\n\n"
            "Real Hugging Face and local checkpoint sources are not inspected or "
            "downloaded in Phase 0."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    scan.add_argument(
        "model",
        metavar="MODEL",
        help="Phase 0 source; currently only fixture:synthetic is supported",
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
    scan.set_defaults(handler=_handle_scan)
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


def _handle_scan(args: argparse.Namespace) -> int:
    if args.force and args.output is None:
        print("moeatlas scan: --force requires --output PATH", file=sys.stderr)
        return 2

    try:
        payload = report_payload(scan_source(args.model))
        if args.output is None:
            sys.stdout.write(payload)
            return 0
        output = write_report_atomic(payload, args.output, force=args.force)
    except (ScanSourceError, ScanOutputError, ValueError) as exc:
        print(f"moeatlas scan: {exc}", file=sys.stderr)
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
