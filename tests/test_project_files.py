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
            ROOT / "docs" / "adapters.md",
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
        for term in (
            "named_modules()",
            "confidence",
            "STRUCTURE",
            "dry-run",
            "MV-02",
            "fixture:synthetic",
            "--loading-plan",
            "Hugging Face",
        ):
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
            "--loading-plan",
            "LoadingPlan",
            "load_and_scan()",
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
            "load_and_scan(plan)",
            "discovery.scan()",
            "STRUCTURE",
            "local_files_only=True",
            "Accelerate",
            "execute_user_code=True",
            "PendingRuntimeCleanup",
            "named_modules()",
            "MV-01",
        ):
            with self.subTest(term=term):
                self.assertIn(term, runtime)

    def test_runtime_routing_docs_and_exports_describe_passive_boundary(self) -> None:
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        probe = (ROOT / "docs" / "probe.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runtime,
                (
                    "RoutingCaptureSession",
                    "RoutingCaptureTarget",
                    "max_events",
                    "exact opaque `(module, inputs,",
                    "detaching tensors",
                    "exact primary exception",
                    "caller-owned model",
                    "MV-03",
                    "MV-04",
                    "MV-05",
                ),
            ),
            (
                probe,
                (
                    "RoutingCaptureSession",
                    "RoutingCaptureTarget",
                    "retained events",
                    "decode",
                    "control-flow",
                    "forward-hook API",
                    "Mixtral v4.50.0",
                    "Qwen3-MoE v4.57.1",
                ),
            ),
            (architecture, ("RoutingCaptureSession", "retained-event quota")),
            (ledger, ("Feature 16", "RoutingCaptureSession", "MV-03")),
            (readme, ("RoutingCaptureSession", "RoutingEvent", "certification")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        runtime_exports = (ROOT / "src" / "moeatlas" / "runtime" / "__init__.py").read_text()
        routing_source = (ROOT / "src" / "moeatlas" / "runtime" / "routing.py").read_text()
        for term in ("RoutingCaptureError", "RoutingCaptureSession", "RoutingCaptureTarget"):
            self.assertIn(term, runtime_exports)
            self.assertIn(term, routing_source)
        self.assertTrue((ROOT / "tests" / "test_runtime_routing.py").is_file())

    def test_adapter_docs_describe_static_structure_boundary(self) -> None:
        adapters = (ROOT / "docs" / "adapters.md").read_text()
        for term in (
            "AdapterDescriptor",
            "AdapterDetection",
            "AdapterInspection",
            "MixtralStaticAdapter()",
            "block_sparse_moe",
            "packed",
            "gate_up_proj",
            "down_proj",
            "legacy_indexed",
            "structural attributes",
            "not independently hookable",
            "routing certification",
            "STRUCTURE",
            "STATIC_STRUCTURE",
            "KeyboardInterrupt",
            "SystemExit",
        ):
            with self.subTest(term=term):
                self.assertIn(term, adapters)

    def test_qwen3_adapter_docs_describe_bounded_reference_layouts(self) -> None:
        adapters = (ROOT / "docs" / "adapters.md").read_text()
        discovery = (ROOT / "docs" / "discovery.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                adapters,
                (
                    "Qwen3MoeStaticAdapter()",
                    "qwen3_moe",
                    "legacy_indexed",
                    "mlp_only_layers",
                    "decoder_sparse_step",
                    "Qwen2",
                    "Qwen3.5",
                    "64f30450dbfd1d02f610ad7080535cb906637fb9",
                    "v4.51.3",
                    "v4.57.1",
                    "v5.0.0",
                ),
            ),
            (
                discovery,
                (
                    "Qwen3MoeStaticAdapter()",
                    "qwen3_moe",
                    "legacy_indexed",
                    "mlp_only_layers",
                    "decoder_sparse_step",
                    "Qwen2",
                    "Qwen3.5",
                ),
            ),
            (
                ledger,
                (
                    "Qwen3MoeStaticAdapter()",
                    "legacy_indexed",
                    "Qwen2",
                    "Qwen3.5",
                    "STRUCTURE",
                ),
            ),
            (
                readme,
                (
                    "Qwen3MoeStaticAdapter()",
                    "legacy_indexed",
                    "Qwen2/Qwen3.5",
                    "STRUCTURE-only",
                ),
            ),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        for term in ("Qwen3MoeStaticAdapter", "qwen3_moe"):
            with self.subTest(term=term):
                self.assertIn(
                    term,
                    (ROOT / "src" / "moeatlas" / "adapters" / "__init__.py").read_text(),
                )
        self.assertTrue((ROOT / "src" / "moeatlas" / "adapters" / "qwen3_moe.py").is_file())

    def test_adapter_probe_planning_docs_and_exports_are_present(self) -> None:
        adapters = (ROOT / "docs" / "adapters.md").read_text()
        probe = (ROOT / "docs" / "probe.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                adapters,
                (
                    "build_routing_probe_plan(inspection)",
                    "family-neutral",
                    "ComponentKind.ROUTER",
                    "TOP_K",
                    "forward-hook API",
                    "no execution, event, or storage bound",
                    "payload conventions",
                    "MV-03",
                    "Mixtral v4.50.0 source",
                    "Qwen3-MoE v4.57.1 source",
                    "Mixtral source at `64f30450dbfd1d02f610ad7080535cb906637fb9`",
                    "Qwen3-MoE source at the same pinned commit",
                ),
            ),
            (
                probe,
                (
                    "build_routing_probe_plan()",
                    "AdapterInspection",
                    "ROUTING",
                    "Mixtral source",
                    "Qwen3-MoE source",
                    "no execution, event, or storage bound",
                    "payload conventions",
                    "MV-03",
                    "Mixtral v4.50.0 source",
                    "Qwen3-MoE v4.57.1 source",
                    "Mixtral source at `64f30450dbfd1d02f610ad7080535cb906637fb9`",
                    "Qwen3-MoE source at the same pinned commit",
                ),
            ),
            (architecture, ("family-neutral", "ROUTING")),
            (ledger, ("Feature 15", "adapter-inspection-to-routing-plan", "MV-03")),
            (readme, ("AdapterInspection", "family-neutral reduced")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "adapters" / "planning.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "adapters" / "__init__.py").read_text()
        for term in ("AdapterProbePlanError", "build_routing_probe_plan"):
            self.assertIn(term, source)
            self.assertIn(term, exports)
        self.assertTrue((ROOT / "tests" / "test_adapter_probe_planning.py").is_file())

    def test_mixtral_routing_decoder_docs_and_exports_are_present(self) -> None:
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        adapters = (ROOT / "docs" / "adapters.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runtime,
                (
                    "MixtralRoutingDecoder",
                    "one-forward caller",
                    "legacy_indexed",
                    "packed",
                    "TokenEvent",
                    "not a tokenizer",
                    "not a runner",
                    "EXPERIMENTAL",
                    "MV-03",
                    "MV-08",
                ),
            ),
            (
                adapters,
                (
                    "MixtralRoutingDecoder",
                    "one-forward",
                    "legacy_indexed",
                    "packed",
                    "softmax/top-k renormalization",
                    "EXPERIMENTAL",
                    "routing certification",
                    "MV-03",
                    "MV-08",
                ),
            ),
            (architecture, ("MixtralRoutingDecoder", "one-forward", "MV-03", "MV-08")),
            (ledger, ("Feature 17", "MixtralRoutingDecoder", "MV-03", "MV-08")),
            (readme, ("MixtralRoutingDecoder", "EXPERIMENTAL", "MV-03", "MV-08")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "runtime" / "mixtral_routing.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "runtime" / "__init__.py").read_text()
        for term in ("MixtralRoutingDecoder", "RoutingCaptureTarget", "TokenEvent", "RoutingEvent"):
            self.assertIn(term, source)
        self.assertIn("MixtralRoutingDecoder", exports)
        self.assertTrue((ROOT / "tests" / "test_mixtral_routing_decoder.py").is_file())

    def test_mixtral_routing_forward_docs_and_exports_are_present(self) -> None:
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runtime,
                (
                    "run_mixtral_routing_forward",
                    "single-forward prerequisite",
                    "caller-tokenized",
                    "row order",
                    "padding",
                    "one common run and phase",
                    "exactly once",
                    "session.close()",
                    "exactly once internally",
                    "exact primary exception",
                    "PendingRuntimeCleanup",
                    "pending_cleanup",
                    "pending_runtime_cleanup",
                    "complete-event budget",
                    "output identity",
                    "not a tokenizer",
                    "generation runner",
                    "storage sink",
                    "EXPERIMENTAL",
                    "MV-03",
                    "MV-08",
                ),
            ),
            (
                architecture,
                (
                    "run_mixtral_routing_forward",
                    "caller-tokenized",
                    "exactly once",
                    "internal `session.close()` retry",
                    "exact primary exception",
                    "caller-owned `PendingRuntimeCleanup` handle",
                    "complete-event budget",
                    "EXPERIMENTAL",
                    "MV-03",
                    "MV-08",
                ),
            ),
            (ledger, ("Feature 18", "run_mixtral_routing_forward", "MV-03", "MV-08")),
            (
                readme,
                ("run_mixtral_routing_forward", "caller-tokenized", "frozen", "one model forward"),
            ),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "runtime" / "routing_forward.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "runtime" / "__init__.py").read_text()
        for term in ("MixtralRoutingForwardResult", "run_mixtral_routing_forward"):
            self.assertIn(term, source)
            self.assertIn(term, exports)
        self.assertTrue((ROOT / "tests" / "test_runtime_routing_forward.py").is_file())


if __name__ == "__main__":
    unittest.main()
