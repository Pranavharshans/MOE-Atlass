from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectFilesTests(unittest.TestCase):
    def test_packaging_and_deferred_validation_docs_are_present(self) -> None:
        required_files = (
            ROOT / "pyproject.toml",
            ROOT / "LICENSE",
            ROOT / "docs" / "architecture.md",
            ROOT / "docs" / "development.md",
            ROOT / "docs" / "model-validation-ledger.md",
            ROOT / "docs" / "schemas.md",
            ROOT / "docs" / "discovery.md",
            ROOT / "docs" / "probe.md",
            ROOT / "docs" / "events.md",
            ROOT / "docs" / "cli.md",
            ROOT / "docs" / "loading.md",
            ROOT / "docs" / "runtime.md",
        )
        for path in required_files:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

    def test_validation_ledger_keeps_model_execution_deferred(self) -> None:
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        self.assertIn("Status: deferred", ledger)
        self.assertIn("No model files are downloaded", ledger)
        self.assertIn("final VM", ledger)

    def test_schema_docs_describe_version_and_capability_policy(self) -> None:
        schemas = (ROOT / "docs" / "schemas.md").read_text()
        self.assertIn("schema version", schemas)
        self.assertIn("verified=True", schemas)
        self.assertIn("parse_model_key()", schemas)
        self.assertIn("recomputes that digest", schemas)
        for label in ("FULL", "ROUTING", "MODULE", "STRUCTURE", "EXPERIMENTAL", "UNSUPPORTED"):
            with self.subTest(label=label):
                self.assertIn(f"`{label}`", schemas)

    def test_discovery_docs_describe_static_boundary(self) -> None:
        discovery = (ROOT / "docs" / "discovery.md").read_text()
        for term in ("named_modules()", "confidence", "STRUCTURE", "dry-run", "MV-02"):
            with self.subTest(term=term):
                self.assertIn(term, discovery)

    def test_probe_docs_describe_bounded_lifecycle(self) -> None:
        probe = (ROOT / "docs" / "probe.md").read_text()
        for term in ("ProbePlan", "HookManager", "raw_opt_in", "reverse", "MV-04"):
            with self.subTest(term=term):
                self.assertIn(term, probe)

    def test_event_docs_describe_normalized_boundary(self) -> None:
        events = (ROOT / "docs" / "events.md").read_text()
        for term in (
            "TokenEvent",
            "RoutingEvent",
            "ExpertEvent",
            "token:<64 lowercase hex",
            "zero-based",
            "latency_ms",
            "Parquet/DuckDB",
        ):
            with self.subTest(term=term):
                self.assertIn(term, events)

    def test_cli_docs_describe_phase_zero_scan_boundary(self) -> None:
        cli = (ROOT / "docs" / "cli.md").read_text()
        for term in (
            "fixture:synthetic",
            "DiscoveryReport",
            "--force",
            "atomically",
            "MV-01/MV-02",
            "does not inspect local paths",
        ):
            with self.subTest(term=term):
                self.assertIn(term, cli)

    def test_loading_docs_describe_schema_only_boundary(self) -> None:
        loading = (ROOT / "docs" / "loading.md").read_text()
        for term in (
            "HuggingFaceSource",
            "LocalSource",
            "InstanceSource",
            "CustomLoaderSource",
            "trust_remote_code=False",
            "plan_id",
            "ImmutableRevisionEvidence",
            "MV-01/MV-02",
        ):
            with self.subTest(term=term):
                self.assertIn(term, loading)

    def test_runtime_docs_describe_owned_execution_boundary(self) -> None:
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        for term in (
            "RuntimeArtifacts",
            "InstanceSource",
            "CustomLoaderSource",
            "load_huggingface()",
            "load_local()",
            "local_files_only=True",
            "Accelerate",
            "execute_user_code=True",
            "PendingRuntimeCleanup",
            "named_modules()",
            "MV-01",
        ):
            with self.subTest(term=term):
                self.assertIn(term, runtime)


if __name__ == "__main__":
    unittest.main()
