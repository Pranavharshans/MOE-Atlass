from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

import moeatlas.analysis.bundle as bundle_module
from moeatlas.analysis import (
    ANALYSIS_BUNDLE_SCHEMA_VERSION,
    AnalysisBundleEntry,
    AnalysisBundleReceipt,
    RoutingLoadComparison,
    RoutingLoadMatrix,
    RoutingLoadSummary,
    compare_routing_load,
    summarize_routing_load,
    write_analysis_bundle,
)

from .test_analysis_routing_compare import _comparison_matrix, _matrix


@pytest.fixture()
def artifacts():
    matrix = _matrix()
    comparison = compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8)
    summary = summarize_routing_load(matrix, max_cells=8)
    return matrix, comparison, summary


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def test_happy_path_writes_exactly_four_canonical_files(tmp_path: Path, artifacts) -> None:
    matrix, comparison, summary = artifacts
    destination = tmp_path / "bundle"
    receipt = write_analysis_bundle(
        destination, matrix=matrix, comparison=comparison, summary=summary
    )
    names = sorted(path.name for path in destination.iterdir())
    assert names == [
        "manifest.json",
        "routing_load_comparison.json",
        "routing_load_matrix.json",
        "routing_load_summary.json",
    ]
    assert receipt.schema_version == ANALYSIS_BUNDLE_SCHEMA_VERSION == "1.0"
    assert [entry.name for entry in receipt.entries] == [
        "routing_load_comparison.json",
        "routing_load_matrix.json",
        "routing_load_summary.json",
    ]
    for entry in receipt.entries:
        document = _read_bytes(destination / entry.name)
        assert entry.bytes == len(document)
        assert entry.sha256 == "sha256:" + hashlib.sha256(document).hexdigest()
    assert receipt.total_bytes == sum(entry.bytes for entry in receipt.entries)


def test_manifest_is_canonical_and_matches_receipt(tmp_path: Path, artifacts) -> None:
    _, _, _ = artifacts
    matrix = _matrix()
    summary = summarize_routing_load(matrix, max_cells=8)
    destination = tmp_path / "bundle"
    receipt = write_analysis_bundle(destination, matrix=matrix, summary=summary)
    manifest = json.loads(_read_bytes(destination / "manifest.json"))
    assert manifest["artifact_type"] == "moeatlas.analysis_bundle"
    assert manifest["schema_version"] == "1.0"
    assert manifest["entry_count"] == 2
    assert manifest["total_bytes"] == receipt.total_bytes
    assert manifest["entries"] == [
        {"name": entry.name, "sha256": entry.sha256, "bytes": entry.bytes}
        for entry in receipt.entries
    ]
    raw = _read_bytes(destination / "manifest.json")
    assert b'": ' not in raw and b'", "' not in raw


def test_bundle_artifacts_round_trip_through_from_json(tmp_path: Path, artifacts) -> None:
    matrix, comparison, summary = artifacts
    destination = tmp_path / "bundle"
    write_analysis_bundle(destination, matrix=matrix, comparison=comparison, summary=summary)
    assert RoutingLoadMatrix.from_json(
        _read_bytes(destination / "routing_load_matrix.json")
    ) == matrix
    assert RoutingLoadComparison.from_json(
        _read_bytes(destination / "routing_load_comparison.json")
    ) == comparison
    assert RoutingLoadSummary.from_json(
        _read_bytes(destination / "routing_load_summary.json")
    ) == summary


def test_equal_values_produce_byte_identical_bundles(tmp_path: Path, artifacts) -> None:
    matrix, _, _ = artifacts
    summary = summarize_routing_load(matrix, max_cells=8)
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_analysis_bundle(first, matrix=matrix, summary=summary)
    write_analysis_bundle(second, matrix=matrix, summary=summary)
    for name in (
        "manifest.json",
        "routing_load_matrix.json",
        "routing_load_summary.json",
    ):
        assert _read_bytes(first / name) == _read_bytes(second / name)


@pytest.mark.parametrize("solo", ["matrix", "comparison", "summary"])
def test_single_artifact_bundles_are_valid(tmp_path: Path, solo: str, artifacts) -> None:
    matrix, comparison, summary = artifacts
    destination = tmp_path / f"bundle-{solo}"
    if solo == "matrix":
        receipt = write_analysis_bundle(destination, matrix=matrix)
    elif solo == "comparison":
        receipt = write_analysis_bundle(destination, comparison=comparison)
    else:
        receipt = write_analysis_bundle(destination, summary=summary)
    assert len(receipt.entries) == 1
    assert (destination / "manifest.json").is_file()


def test_matrix_and_summary_subset_bundle(tmp_path: Path, artifacts) -> None:
    matrix, _, summary = artifacts
    receipt = write_analysis_bundle(tmp_path / "bundle", matrix=matrix, summary=summary)
    assert [entry.name for entry in receipt.entries] == [
        "routing_load_matrix.json",
        "routing_load_summary.json",
    ]


def test_empty_bundle_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_analysis_bundle(tmp_path / "bundle")


def test_identity_mismatch_is_rejected(tmp_path: Path, artifacts) -> None:
    matrix, comparison, _ = artifacts
    for tampered in (
        dataclasses.replace(comparison, model_key="model:acme/other@r1"),
        dataclasses.replace(comparison, inspection_digest="sha256:" + "2" * 64),
        dataclasses.replace(comparison, layout="packed"),
    ):
        with pytest.raises(ValueError) as excinfo:
            write_analysis_bundle(tmp_path / "bundle", matrix=matrix, comparison=tampered)
        assert str(excinfo.value) == (
            "bundle artifacts do not share one model/adapter/inspection identity"
        )


def test_destination_validation(tmp_path: Path, artifacts) -> None:
    matrix, _, _ = artifacts
    with pytest.raises(ValueError):
        write_analysis_bundle(tmp_path / "missing-parent" / "bundle", matrix=matrix)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        write_analysis_bundle(occupied, matrix=matrix)
    file_destination = tmp_path / "a-file"
    file_destination.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        write_analysis_bundle(file_destination, matrix=matrix)
    with pytest.raises(TypeError):
        write_analysis_bundle(None, matrix=matrix)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        write_analysis_bundle(7, matrix=matrix)  # type: ignore[arg-type]


def test_empty_existing_directory_is_accepted_without_recreation(
    tmp_path: Path, artifacts
) -> None:
    matrix, _, _ = artifacts
    destination = tmp_path / "premade"
    destination.mkdir()
    receipt = write_analysis_bundle(str(destination), matrix=matrix)
    assert len(receipt.entries) == 1
    assert destination.is_dir()


class _MatrixSubclass(RoutingLoadMatrix):
    pass


def test_exact_type_checks_reject_subclasses_and_wrong_slots(
    tmp_path: Path, artifacts
) -> None:
    matrix, comparison, summary = artifacts
    subclassed = _MatrixSubclass(
        schema_version=matrix.schema_version,
        store_schema_version=matrix.store_schema_version,
        event_schema_version=matrix.event_schema_version,
        run_key=matrix.run_key,
        model_key=matrix.model_key,
        adapter_name=matrix.adapter_name,
        adapter_version=matrix.adapter_version,
        inspection_digest=matrix.inspection_digest,
        layout=matrix.layout,
        shard_keys=matrix.shard_keys,
        token_count=matrix.token_count,
        assignment_count=matrix.assignment_count,
        routed_top_k=matrix.routed_top_k,
        layer_keys=matrix.layer_keys,
        layer_indices=matrix.layer_indices,
        expert_keys=matrix.expert_keys,
        assignment_counts=matrix.assignment_counts,
        assignment_shares=matrix.assignment_shares,
        load_ratios=matrix.load_ratios,
    )
    with pytest.raises(TypeError):
        write_analysis_bundle(tmp_path / "one", matrix=subclassed)
    with pytest.raises(TypeError):
        write_analysis_bundle(tmp_path / "two", comparison=matrix)
    with pytest.raises(TypeError):
        write_analysis_bundle(tmp_path / "three", summary=comparison)
    del summary


def test_failure_during_publication_leaves_no_partial_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifacts
) -> None:
    matrix, _, summary = artifacts
    destination = tmp_path / "bundle"
    original_replace = os.replace
    calls: list[str] = []

    def failing_replace(src: Any, dst: Any) -> None:
        calls.append(Path(dst).name)
        if len(calls) == 2:
            raise OSError("injected publication failure")
        original_replace(src, dst)

    monkeypatch.setattr(bundle_module.os, "replace", failing_replace)
    with pytest.raises(OSError):
        write_analysis_bundle(destination, matrix=matrix, summary=summary)
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_keyboard_interrupt_propagates_after_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifacts
) -> None:
    matrix, _, _ = artifacts
    destination = tmp_path / "bundle"

    def interrupting_replace(src: Any, dst: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(bundle_module.os, "replace", interrupting_replace)
    with pytest.raises(KeyboardInterrupt):
        write_analysis_bundle(destination, matrix=matrix)
    assert not destination.exists()


def test_receipt_tampering_is_rejected(artifacts) -> None:
    matrix, _, _ = artifacts
    destination_entry = AnalysisBundleEntry(
        name="routing_load_matrix.json",
        sha256="sha256:" + "a" * 64,
        bytes=10,
    )
    receipt = AnalysisBundleReceipt(
        schema_version="1.0",
        entries=(destination_entry,),
        total_bytes=10,
    )
    assert receipt.total_bytes == 10
    other = AnalysisBundleEntry(name="summary.json", sha256="sha256:" + "b" * 64, bytes=5)
    with pytest.raises(ValueError):
        dataclasses.replace(receipt, schema_version="9.9")
    with pytest.raises(ValueError):
        AnalysisBundleReceipt(
            schema_version="1.0",
            entries=(
                destination_entry,
                AnalysisBundleEntry(
                    name="routing_load_matrix.json",
                    sha256="sha256:" + "c" * 64,
                    bytes=5,
                ),
            ),
            total_bytes=15,
        )
    with pytest.raises(ValueError):
        AnalysisBundleReceipt(
            schema_version="1.0",
            entries=(
                other,
                destination_entry,
            ),
            total_bytes=15,
        )
    with pytest.raises(ValueError):
        AnalysisBundleReceipt(
            schema_version="1.0",
            entries=(
                destination_entry,
                AnalysisBundleEntry(name="x.json", sha256="nope", bytes=5),
            ),
            total_bytes=15,
        )
    with pytest.raises(ValueError):
        AnalysisBundleReceipt(
            schema_version="1.0",
            entries=(
                destination_entry,
                AnalysisBundleEntry(name="x.json", sha256="sha256:" + "d" * 64, bytes=0),
            ),
            total_bytes=10,
        )
    with pytest.raises(ValueError):
        AnalysisBundleReceipt(
            schema_version="1.0",
            entries=(destination_entry,),
            total_bytes=11,
        )
    with pytest.raises((TypeError, ValueError)):
        AnalysisBundleReceipt(schema_version="1.0", entries=(), total_bytes=0)


def test_module_source_stays_model_free_and_offline() -> None:
    source = Path("src/moeatlas/analysis/bundle.py").read_text()
    forbidden = {
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
        "polars",
        "duckdb",
        "socket",
        "urllib",
        "httpx",
        "requests",
        "importlib",
        "shutil",
        "tempfile",
    }
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in forbidden
    assert "token_text" not in source
    assert "urlopen" not in source
    assert "http://" not in source
