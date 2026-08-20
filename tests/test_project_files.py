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
            ROOT / "docs" / "storage.md",
            ROOT / "docs" / "analysis.md",
            ROOT / "docs" / "visualization.md",
            ROOT / "src" / "moeatlas" / "analysis" / "__init__.py",
            ROOT / "src" / "moeatlas" / "analysis" / "routing_load.py",
            ROOT / "src" / "moeatlas" / "store" / "__init__.py",
            ROOT / "src" / "moeatlas" / "store" / "routing_shards.py",
            ROOT / "tests" / "test_store_routing_shards.py",
            ROOT / "tests" / "test_store_routing_run_inventory.py",
            ROOT / "tests" / "test_analysis_routing_load.py",
            ROOT / "tests" / "test_analysis_routing_heatmap.py",
            ROOT / "tests" / "test_cli_heatmap.py",
            ROOT / "tests" / "test_cli_routing_runs.py",
            ROOT / "src" / "moeatlas" / "runtime" / "prompt_prefill.py",
            ROOT / "tests" / "test_runtime_prompt_prefill.py",
            ROOT / "src" / "moeatlas" / "adapters" / "qwen3_5_moe.py",
            ROOT / "tests" / "fixtures" / "qwen3_5_moe.py",
            ROOT / "tests" / "test_qwen3_5_moe_adapter.py",
        )
        for path in required_files:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

    def test_validation_ledger_keeps_model_execution_deferred(self) -> None:
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        self.assertIn("Status: deferred", ledger)
        self.assertIn("No model files are downloaded", ledger)
        self.assertIn("final VM", ledger)

    def test_qwen35_acceptance_anchors_are_present(self) -> None:
        adapter_docs = (ROOT / "docs" / "adapters.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document in (adapter_docs, architecture, ledger, readme):
            self.assertIn("qwen3_5_moe", document)
            self.assertIn("router_logits", document)
            self.assertIn("router_scores", document)
            self.assertIn("router_indices", document)
            self.assertIn("Feature 26", document)
            self.assertIn("final VM", document)
        self.assertIn(
            "https://github.com/huggingface/transformers/blob/v5.14.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
            adapter_docs,
        )
        self.assertIn("Qwen3.5-35B-A3B", ledger)
        self.assertIn("Mixtral is the reference", architecture)

    def test_prompt_prefill_docs_and_surface_are_present(self) -> None:
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        source = (ROOT / "src" / "moeatlas" / "runtime" / "__init__.py").read_text()
        for text, terms in (
            (
                runtime,
                (
                    "run_mixtral_prompt_prefill",
                    "max_prompt_chars",
                    "tokenize",
                    "encoding",
                    "pending-handle",
                    "no progress stream",
                ),
            ),
            (
                architecture,
                (
                    "Feature 24",
                    "server endpoint",
                    "wire/view-model contract",
                    "Feature 19 append",
                ),
            ),
            (
                ledger,
                (
                    "Feature 24 prompt prefill",
                    "Status: deferred",
                    "115 focused cases",
                    "Prompts, paths",
                ),
            ),
            (
                readme,
                (
                    "append_mixtral_routing_shard",
                    "list_mixtral_routing_runs",
                    "aggregate_mixtral_routing_load",
                    "render_mixtral_routing_load_heatmap",
                ),
            ),
            (source, ("MixtralPromptPrefillError", "run_mixtral_prompt_prefill")),
        ):
            for term in terms:
                self.assertIn(term, text)

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

    def test_cli_heatmap_docs_and_model_free_surface_are_present(self) -> None:
        cli = (ROOT / "docs" / "cli.md").read_text()
        visualization = (ROOT / "docs" / "visualization.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                cli,
                (
                    "moeatlas heatmap WORKSPACE",
                    "--inspection",
                    "--max-inspection-bytes",
                    "--max-routing-rows",
                    "--max-source-bytes",
                    "--max-matrix-cells",
                    "canonical positive decimal",
                    "non-symlink",
                    "exactly once",
                    "write_report_atomic",
                    ".html",
                    "--force",
                    "store",
                    "saved routing heatmap to ",
                    "temporary file",
                    "KeyboardInterrupt",
                    "SystemExit",
                ),
            ),
            (
                visualization,
                (
                    "Feature 22",
                    "moeatlas heatmap",
                    "non-symlink",
                    "exactly once",
                    "inspection.to_json()",
                    ".html",
                    "DuckDB `store` extra",
                    "EXPERIMENTAL",
                ),
            ),
            (architecture, ("Feature 22", "moeatlas heatmap WORKSPACE", "exactly once")),
            (ledger, ("Feature 22", "canonical decimal", "non-symlink", "exactly-once")),
            (readme, ("moeatlas heatmap WORKSPACE", "atomic publication")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "cli.py").read_text()
        for term in (
            "_parse_heatmap_budget",
            "_preflight_heatmap_output",
            "_read_heatmap_inspection",
            "_run_heatmap_analysis",
            "write_report_atomic",
        ):
            self.assertIn(term, source)

    def test_routing_run_inventory_docs_and_surface_are_present(self) -> None:
        storage = (ROOT / "docs" / "storage.md").read_text()
        cli = (ROOT / "docs" / "cli.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                storage,
                (
                    "Feature 23",
                    "list_mixtral_routing_runs",
                    "max_event_rows",
                    "max_source_bytes",
                    "mixtral_routing_run_inventory",
                    "redacted",
                    "stored",
                    "mixed",
                    "latest run",
                    "ST-04",
                ),
            ),
            (
                cli,
                (
                    "moeatlas routing-runs WORKSPACE",
                    "--max-runs",
                    "--max-shards",
                    "--max-event-rows",
                    "--max-source-bytes",
                    ".json",
                    "saved routing run inventory to ",
                    "write_report_atomic",
                    "--force",
                ),
            ),
            (architecture, ("Feature 23", "routing-run inventory", "latest-run")),
            (ledger, ("Feature 23", "declared/actual event budgets", "atomic JSON CLI")),
            (readme, ("moeatlas routing-runs WORKSPACE", "run registry")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "store" / "routing_shards.py").read_text()
        for term in (
            "ROUTING_RUN_INVENTORY_SCHEMA_VERSION",
            "RoutingRunInventoryError",
            "MixtralRoutingRunSummary",
            "MixtralRoutingRunInventory",
            "list_mixtral_routing_runs",
        ):
            self.assertIn(term, source)
        cli_source = (ROOT / "src" / "moeatlas" / "cli.py").read_text()
        for term in (
            "_preflight_routing_runs_output",
            "_run_routing_run_inventory",
            "routing-runs",
        ):
            self.assertIn(term, cli_source)

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

    def test_routing_shard_storage_docs_and_surface_are_present(self) -> None:
        storage = (ROOT / "docs" / "storage.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                storage,
                (
                    "append_mixtral_routing_shard",
                    "list_mixtral_routing_shards",
                    "Feature 18",
                    "tokens.parquet",
                    "routing.parquet",
                    "ZSTD",
                    "content-addressed",
                    "redaction",
                    "manifest_type",
                    "store_schema_version",
                    "writer_name",
                    "sha256:<64hex>",
                    "opaque Feature 18 output",
                    "event_index",
                    "token_text",
                    "token_text_stored",
                    "layer_key",
                    "router_logit",
                    "probability",
                    "weight",
                    "selected",
                    "nullable",
                    "not a full workspace/catalog",
                    "dependency",
                    "reopen",
                ),
            ),
            (architecture, ("Feature 19", "ST-01", "ST-04", "MV-01 through MV-08")),
            (ledger, ("Feature 19", "ST-01", "ST-04", "does not change MV-01/MV-08")),
            (readme, ("routing-shard", "storage", "workspace/catalog")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "store" / "routing_shards.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "store" / "__init__.py").read_text()
        for term in (
            "STORE_SCHEMA_VERSION",
            "RoutingShardError",
            "RoutingShardReceipt",
            "append_mixtral_routing_shard",
            "list_mixtral_routing_shards",
        ):
            self.assertIn(term, source)
            self.assertIn(term, exports)

    def test_routing_load_analysis_docs_and_surface_are_present(self) -> None:
        analysis = (ROOT / "docs" / "analysis.md").read_text()
        storage = (ROOT / "docs" / "storage.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis,
                (
                    "Feature 20",
                    "aggregate_mixtral_routing_load",
                    "MixtralRoutingLoadMatrix",
                    "legacy_indexed",
                    "packed",
                    "max_routing_rows",
                    "max_source_bytes",
                    "max_matrix_cells",
                    "assignment_counts",
                    "assignment_shares",
                    "load_ratios",
                    "zero-count experts",
                    "not a catalog",
                    "ST-04",
                    "MV-01 through MV-08",
                ),
            ),
            (storage, ("Feature 20", "does not alter shard bytes")),
            (architecture, ("Feature 20", "inspection-published", "ST-04")),
            (ledger, ("Feature 20", "zero-count experts", "ST-04 remains deferred")),
            (readme, ("aggregate_mixtral_routing_load", "analysis")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "analysis" / "routing_load.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "ROUTING_LOAD_SCHEMA_VERSION",
            "RoutingLoadError",
            "MixtralRoutingLoadMatrix",
            "aggregate_mixtral_routing_load",
        ):
            self.assertIn(term, source)
            self.assertIn(term, exports)

    def test_routing_heatmap_visualization_docs_and_surface_are_present(self) -> None:
        visualization = (ROOT / "docs" / "visualization.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                visualization,
                (
                    "Feature 21",
                    "render_mixtral_routing_load_heatmap",
                    "assignment_counts",
                    "assignment_shares",
                    "load_ratios",
                    "max_cells",
                    "zero-count experts",
                    "heat-0",
                    "heat-8",
                    "1 + min(7, int((v / m) * 8))",
                    "Routing load only. Selection frequency is association evidence, not expert "
                    "specialization or causal effect.",
                    "details",
                    "shard count",
                    "Content-Security-Policy",
                    "JavaScript",
                    "artifact.write_text",
                    "webbrowser.open",
                    "permanent",
                    "React",
                    "not a replacement",
                    "EXPERIMENTAL",
                    "MV-01",
                ),
            ),
            (architecture, ("Feature 21", "static HTML heatmap", "heat bins")),
            (ledger, ("Feature 21", "heat-0..8", "MV-01 through MV-08")),
            (readme, ("render_mixtral_routing_load_heatmap", "visualization")),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "analysis" / "routing_heatmap.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in ("ROUTING_HEATMAP_SCHEMA_VERSION", "render_mixtral_routing_load_heatmap"):
            self.assertIn(term, source)
            self.assertIn(term, exports)


if __name__ == "__main__":
    unittest.main()
