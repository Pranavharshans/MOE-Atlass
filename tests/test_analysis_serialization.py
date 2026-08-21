from __future__ import annotations

import ast
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from moeatlas.analysis import (
    RoutingLoadComparison,
    RoutingLoadMatrix,
    RoutingLoadSummary,
    compare_routing_load,
    summarize_routing_load,
)

from .test_analysis_routing_compare import _comparison_matrix, _matrix

_MATRIX_MARKER = "moeatlas.routing_load_matrix"
_COMPARE_MARKER = "moeatlas.routing_load_comparison"
_SUMMARY_MARKER = "moeatlas.routing_load_summary"

_CASES: list[tuple[str, Callable[[], object], type[object], str]] = [
    ("matrix", _matrix, RoutingLoadMatrix, _MATRIX_MARKER),
    (
        "comparison",
        lambda: compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8),
        RoutingLoadComparison,
        _COMPARE_MARKER,
    ),
    (
        "summary",
        lambda: summarize_routing_load(_matrix(), max_cells=8),
        RoutingLoadSummary,
        _SUMMARY_MARKER,
    ),
]

_CASE_IDS = [name for name, _, _, _ in _CASES]


def _case(name: str) -> tuple[Callable[[], object], type[object], str]:
    for candidate, builder, cls, marker in _CASES:
        if candidate == name:
            return builder, cls, marker
    raise AssertionError(f"unknown artifact case {name}")


def _canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _document(artifact: object) -> dict[str, object]:
    parsed = json.loads(artifact.to_json())
    assert type(parsed) is dict
    return parsed


def _mutated(artifact: object, **updates: object) -> bytes:
    document = _document(artifact)
    document.update(updates)
    return _canonical_bytes(document)


def _assert_primitives_only(value: object) -> None:
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():
            _assert_primitives_only(item)
    elif type(value) is list:
        for item in value:
            _assert_primitives_only(item)
    else:
        assert type(value) in (str, int, float)


@pytest.mark.parametrize("name", _CASE_IDS)
def test_round_trip_equality_and_byte_determinism(name: str) -> None:
    builder, cls, _ = _case(name)
    first = builder()
    second = builder()
    assert first == second
    assert first.to_json() == second.to_json()
    restored = cls.from_json(first.to_json())
    assert restored == first
    assert restored.to_json() == first.to_json()


@pytest.mark.parametrize("name", _CASE_IDS)
def test_to_dict_contains_only_json_primitives(name: str) -> None:
    builder, _, _ = _case(name)
    exported = builder().to_dict()
    assert type(exported) is dict
    _assert_primitives_only(exported)


@pytest.mark.parametrize("name", _CASE_IDS)
def test_to_json_is_canonical_compact_sorted_form(name: str) -> None:
    builder, _, _ = _case(name)
    text = builder().to_json()
    assert '"schema_version":"1.0"' in text
    assert ": " not in text
    assert ", " not in text
    reparsed = json.loads(text)
    assert text == json.dumps(
        reparsed, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    )


@pytest.mark.parametrize("name", _CASE_IDS)
def test_artifact_type_marker_is_present_in_dict_and_json(name: str) -> None:
    builder, _, marker = _case(name)
    artifact = builder()
    assert artifact.to_dict()["artifact_type"] == marker
    assert f'"artifact_type":"{marker}"' in artifact.to_json()


@pytest.mark.parametrize(
    "source,target",
    [
        ("matrix", "comparison"),
        ("matrix", "summary"),
        ("comparison", "matrix"),
        ("comparison", "summary"),
        ("summary", "matrix"),
        ("summary", "comparison"),
    ],
)
def test_cross_type_documents_are_rejected(source: str, target: str) -> None:
    source_builder, _, _ = _case(source)
    _, target_cls, target_marker = _case(target)
    payload = source_builder().to_json()
    with pytest.raises(ValueError, match="document is not a"):
        target_cls.from_json(payload)


@pytest.mark.parametrize("name", _CASE_IDS)
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        "not json",
        b"null",
        b"[]",
        b"{}",
        b'{"artifact_type":"moeatlas.something_else","schema_version":"1.0"}',
    ],
)
def test_malformed_payloads_are_rejected(name: str, payload: str | bytes) -> None:
    _, cls, _ = _case(name)
    with pytest.raises(ValueError):
        cls.from_json(payload)


@pytest.mark.parametrize("name", _CASE_IDS)
def test_wrong_schema_version_is_rejected(name: str) -> None:
    builder, cls, marker = _case(name)
    payload = _mutated(builder(), schema_version="9.9")
    expected = f"document is not a {marker.removeprefix('moeatlas.').replace('_', ' ')} artifact"
    with pytest.raises(ValueError, match=expected.replace(".", r"\.")):
        cls.from_json(payload)


@pytest.mark.parametrize(
    "name,key",
    [
        ("matrix", "run_key"),
        ("comparison", "baseline_run_key"),
        ("summary", "dead_expert_fraction"),
    ],
)
def test_missing_field_is_rejected(name: str, key: str) -> None:
    builder, cls, _ = _case(name)
    document = _document(builder())
    del document[key]
    with pytest.raises(ValueError, match="missing fields"):
        cls.from_json(_canonical_bytes(document))


@pytest.mark.parametrize("name", _CASE_IDS)
def test_unknown_extra_fields_are_tolerated(name: str) -> None:
    builder, cls, _ = _case(name)
    artifact = builder()
    document = _document(artifact)
    document["unexpected_extra_field"] = {"nested": [1, 2]}
    restored = cls.from_json(_canonical_bytes(document))
    assert restored == artifact


@pytest.mark.parametrize(
    "mutation",
    [
        {"shard_keys": "shard:not-a-list"},
        {"layer_indices": [True, 1]},
        {"assignment_counts": [[1.0, 2.0, 0.0, 2.0], [1, 1, 1, 1]]},
        {"assignment_shares": [[0, 0.5, 0.0, 0.5], [0.25, 0.25, 0.25, 0.25]]},
        {"expert_keys": ["component:not-a-row", ["component:x"]]},
    ],
)
def test_matrix_rejects_type_violations_inside_arrays(mutation: dict[str, object]) -> None:
    payload = _mutated(_matrix(), **mutation)
    with pytest.raises(ValueError):
        RoutingLoadMatrix.from_json(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"dead_expert_count": True},
        {"layer_entropies": ["high", 1.0]},
    ],
)
def test_summary_rejects_type_violations_inside_arrays(mutation: dict[str, object]) -> None:
    summary = summarize_routing_load(_matrix(), max_cells=8)
    payload = _mutated(summary, **mutation)
    with pytest.raises(ValueError):
        RoutingLoadSummary.from_json(payload)


@pytest.mark.parametrize(
    "mutation",
    [
        {"count_deltas": [[1.0, -1.0, 0.0, 0.0], [1, -1, 0, 0]]},
        {"baseline_shard_keys": "shard:not-a-list"},
    ],
)
def test_comparison_rejects_type_violations_inside_arrays(mutation: dict[str, object]) -> None:
    comparison = compare_routing_load(_matrix(), _comparison_matrix(), max_cells=8)
    payload = _mutated(comparison, **mutation)
    with pytest.raises(ValueError):
        RoutingLoadComparison.from_json(payload)


@pytest.mark.parametrize(
    "name,mutation",
    [
        ("matrix", {"shard_keys": ["shard:" + "9" * 64, "shard:" + "1" * 64]}),
        ("summary", {"shard_keys": ["shard:" + "9" * 64, "shard:" + "1" * 64]}),
        ("comparison", {"inspection_digest": "sha256:" + "z" * 64}),
        ("comparison", {"baseline_run_key": "run-baseline", "comparison_run_key": "run-baseline"}),
        ("matrix", {"token_count": True}),
    ],
)
def test_content_violations_surface_value_errors_through_post_init(
    name: str, mutation: dict[str, object]
) -> None:
    builder, cls, _ = _case(name)
    payload = _mutated(builder(), **mutation)
    with pytest.raises(ValueError):
        cls.from_json(payload)


def test_infinity_share_is_rejected_as_non_finite_on_import() -> None:
    document = _document(_matrix())
    document["assignment_shares"][0][0] = float("inf")
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"))
    assert "Infinity" in payload
    with pytest.raises(TypeError):
        RoutingLoadMatrix.from_json(payload)


@pytest.mark.parametrize("name", _CASE_IDS)
@pytest.mark.parametrize(
    "wrap",
    [
        ("str", lambda text: text),
        ("bytes", lambda text: text.encode()),
        ("bytearray", lambda text: bytearray(text.encode())),
    ],
)
def test_from_json_accepts_str_bytes_and_bytearray(
    name: str, wrap: tuple[str, Callable[[str], object]]
) -> None:
    builder, cls, _ = _case(name)
    artifact = builder()
    assert cls.from_json(wrap[1](artifact.to_json())) == artifact


_SERIALIZATION_MODULES = (
    Path("src/moeatlas/analysis/routing_load.py"),
    Path("src/moeatlas/analysis/routing_compare.py"),
    Path("src/moeatlas/analysis/routing_summary.py"),
)

_ALLOWED_IMPORT_ROOTS = frozenset(
    {"__future__", "json", "math", "re", "dataclasses", "pathlib", "typing"}
)

_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "os",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
        "httpx",
        "requests",
        "tempfile",
        "importlib",
        "sys",
    }
)


def test_serialization_modules_stay_pure_offline_and_filesystem_free() -> None:
    for source_path in _SERIALIZATION_MODULES:
        source = source_path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {alias.name.split(".")[0] for alias in node.names}
                assert not roots & _FORBIDDEN_IMPORT_ROOTS
                assert roots <= _ALLOWED_IMPORT_ROOTS
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in _FORBIDDEN_IMPORT_ROOTS
                if node.level == 0:
                    assert root in _ALLOWED_IMPORT_ROOTS
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in {
                    "write_text",
                    "write_bytes",
                    "mkdir",
                    "unlink",
                    "rename",
                }:
                    raise AssertionError(f"{source_path} must not mutate the filesystem")
                if isinstance(node.func, ast.Name) and node.func.id == "open":
                    raise AssertionError(f"{source_path} must not touch the filesystem")
        assert "import os" not in source
        assert "shutil" not in source
        assert "socket" not in source
        assert "open(" not in source
