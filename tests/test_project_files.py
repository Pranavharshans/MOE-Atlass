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
            ROOT / "docs" / "roadmap.md",
            ROOT / "docs" / "schemas.md",
            ROOT / "docs" / "discovery.md",
            ROOT / "docs" / "probe.md",
            ROOT / "docs" / "events.md",
            ROOT / "docs" / "cli.md",
            ROOT / "docs" / "loading.md",
            ROOT / "docs" / "runs.md",
            ROOT / "docs" / "workspace.md",
            ROOT / "docs" / "runtime.md",
            ROOT / "docs" / "adapters.md",
            ROOT / "docs" / "storage.md",
            ROOT / "docs" / "analysis.md",
            ROOT / "docs" / "visualization.md",
            ROOT / "src" / "moeatlas" / "analysis" / "__init__.py",
            ROOT / "src" / "moeatlas" / "analysis" / "routing_load.py",
            ROOT / "src" / "moeatlas" / "store" / "__init__.py",
            ROOT / "src" / "moeatlas" / "store" / "routing_shards.py",
            ROOT / "src" / "moeatlas" / "store" / "catalog.py",
            ROOT / "src" / "moeatlas" / "store" / "ports.py",
            ROOT / "src" / "moeatlas" / "store" / "run_export.py",
            ROOT / "src" / "moeatlas" / "store" / "run_tables.py",
            ROOT / "src" / "moeatlas" / "services" / "__init__.py",
            ROOT / "src" / "moeatlas" / "services" / "workspace.py",
            ROOT / "src" / "moeatlas" / "services" / "datasets.py",
            ROOT / "src" / "moeatlas" / "services" / "run_engine.py",
            ROOT / "src" / "moeatlas" / "services" / "run_inputs.py",
            ROOT / "src" / "moeatlas" / "services" / "run_service.py",
            ROOT / "src" / "moeatlas" / "analysis" / "task_association.py",
            ROOT / "src" / "moeatlas" / "analysis" / "evidence_cards.py",
            ROOT / "src" / "moeatlas" / "analysis" / "routing_agreement.py",
            ROOT / "src" / "moeatlas" / "analysis" / "association_stability.py",
            ROOT / "src" / "moeatlas" / "analysis" / "router_margin.py",
            ROOT / "src" / "moeatlas" / "analysis" / "route_churn.py",
            ROOT / "src" / "moeatlas" / "analysis" / "corouting.py",
            ROOT / "src" / "moeatlas" / "analysis" / "expert_similarity.py",
            ROOT / "src" / "moeatlas" / "adapters" / "registry.py",
            ROOT / "tests" / "test_store_catalog.py",
            ROOT / "tests" / "test_store_ports.py",
            ROOT / "tests" / "test_store_assignment_queries.py",
            ROOT / "tests" / "test_store_run_export.py",
            ROOT / "tests" / "test_store_run_tables.py",
            ROOT / "tests" / "test_services_workspace.py",
            ROOT / "tests" / "test_services_datasets.py",
            ROOT / "tests" / "test_services_run_engine.py",
            ROOT / "tests" / "test_services_run_inputs.py",
            ROOT / "tests" / "test_services_run_service.py",
            ROOT / "src" / "moeatlas" / "services" / "retention.py",
            ROOT / "tests" / "test_services_retention.py",
            ROOT / "tests" / "test_analysis_task_association.py",
            ROOT / "tests" / "test_analysis_evidence_cards.py",
            ROOT / "tests" / "test_analysis_routing_agreement.py",
            ROOT / "tests" / "test_analysis_association_stability.py",
            ROOT / "tests" / "test_analysis_router_margin.py",
            ROOT / "tests" / "test_analysis_route_churn.py",
            ROOT / "tests" / "test_analysis_corouting.py",
            ROOT / "tests" / "test_analysis_expert_similarity.py",
            ROOT / "tests" / "test_adapters_registry.py",
            ROOT / "tests" / "test_cli_adapters.py",
            ROOT / "src" / "moeatlas" / "server" / "__init__.py",
            ROOT / "src" / "moeatlas" / "server" / "dto.py",
            ROOT / "src" / "moeatlas" / "server" / "app.py",
            ROOT / "docs" / "server.md",
            ROOT / "src" / "moeatlas" / "interventions" / "__init__.py",
            ROOT / "src" / "moeatlas" / "interventions" / "recipes.py",
            ROOT / "src" / "moeatlas" / "interventions" / "engine.py",
            ROOT / "docs" / "interventions.md",
            ROOT / "tests" / "test_interventions_recipes.py",
            ROOT / "tests" / "test_interventions_engine.py",
            ROOT / "src" / "moeatlas" / "analysis" / "causal_evidence.py",
            ROOT / "tests" / "test_analysis_causal_evidence.py",
            ROOT / "tests" / "test_cli_run.py",
            ROOT / "tests" / "test_cli_export.py",
            ROOT / "tests" / "test_server_app.py",
            ROOT / "tests" / "test_cli_ui.py",
            ROOT / "tests" / "test_store_routing_shards.py",
            ROOT / "tests" / "test_store_routing_run_inventory.py",
            ROOT / "tests" / "test_analysis_routing_load.py",
            ROOT / "tests" / "test_analysis_routing_heatmap.py",
            ROOT / "tests" / "test_cli_heatmap.py",
            ROOT / "tests" / "test_cli_routing_runs.py",
            ROOT / "src" / "moeatlas" / "runtime" / "prompt_prefill.py",
            ROOT / "tests" / "test_runtime_prompt_prefill.py",
            ROOT / "src" / "moeatlas" / "adapters" / "universe.py",
            ROOT / "src" / "moeatlas" / "runtime" / "capabilities.py",
            ROOT / "tests" / "test_adapters_universe.py",
            ROOT / "tests" / "test_runtime_capabilities.py",
            ROOT / "tests" / "test_runtime_forward_capabilities.py",
            ROOT / "src" / "moeatlas" / "adapters" / "qwen3_5_moe.py",
            ROOT / "tests" / "fixtures" / "qwen3_5_moe.py",
            ROOT / "tests" / "test_qwen3_5_moe_adapter.py",
            ROOT / "src" / "moeatlas" / "runtime" / "qwen3_5_routing.py",
            ROOT / "tests" / "test_qwen3_5_routing_decoder.py",
            ROOT / "src" / "moeatlas" / "runtime" / "routing_forward.py",
            ROOT / "tests" / "test_runtime_routing_forward.py",
            ROOT / "tests" / "test_qwen3_5_routing_forward.py",
        )
        for path in required_files:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)

    def test_validation_ledger_keeps_model_execution_deferred(self) -> None:
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        self.assertIn("Status: deferred", ledger)
        self.assertIn("No model files are downloaded", ledger)
        self.assertIn("final VM", ledger)

    def test_roadmap_preserves_authority_status_and_no_download_policy(self) -> None:
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        for term in (
            "Status: in progress",
            "MoEAtlas PRD",
            "Architecture",
            "model-validation ledger",
            "model-free complete",
            "VM/GPU deferred",
            "kickbacks-v2.vsix",
            "append_routing_shard",
            "public structural result protocol",
        ):
            with self.subTest(term=term):
                self.assertIn(term, roadmap)

    def test_run_contracts_docs_preserve_identity_and_lifecycle_anchors(self) -> None:
        runs_doc = (ROOT / "docs" / "runs.md").read_text()
        for term in (
            "RunSpecification",
            "run:<64 lowercase hex>",
            "RunRecord",
            "RunLifecycleError",
            "intervention_opt_in",
            "redacted",
            "retry",
            "model-validation-ledger.md",
        ):
            with self.subTest(term=term):
                self.assertIn(term, runs_doc)

    def test_workspace_services_docs_preserve_catalog_and_port_anchors(self) -> None:
        workspace = (ROOT / "docs" / "workspace.md").read_text()
        for term in (
            "WorkspaceCatalog",
            "RunRegistryEntry",
            "initialize_catalog",
            "rebuild_catalog",
            "CatalogRebuildReceipt",
            "RoutingRunReader",
            "DuckDBRoutingShardStore",
            ".moeatlas/catalog.json",
            "schema_version",
            "not initialized",
            "atomic",
            "ST-01",
            "model-validation-ledger.md",
        ):
            with self.subTest(term=term):
                self.assertIn(term, workspace)
        services_source = (
            ROOT / "src" / "moeatlas" / "services" / "workspace.py"
        ).read_text()
        for term in (
            "initialize_workspace",
            "open_workspace",
            "register_run",
            "record_run_record",
            "sync_runs_from_shards",
            "query_runs",
        ):
            self.assertIn(term, services_source)

    def test_routing_universe_docs_preserve_contract_anchors(self) -> None:
        adapters_docs = (ROOT / "docs" / "adapters.md").read_text()
        for term in (
            "RoutingUniverse",
            "publish_routing_universe",
            "project_rectangular_universe",
            "expert_indices",
            "routed_top_k",
            "shared_expert_keys",
            "legacy_indexed",
            "family-blind",
            "non-rectangular",
            "declared_universe",
            "final VM",
        ):
            with self.subTest(term=term):
                self.assertIn(term, adapters_docs)
        analysis_docs = (ROOT / "docs" / "analysis.md").read_text()
        for term in (
            "declared_universe",
            "RoutingUniverse",
            "project_rectangular_universe",
            "checked, named gate",
        ):
            self.assertIn(term, analysis_docs)
        runtime_docs = (ROOT / "docs" / "runtime.md").read_text()
        for term in (
            "RoutingDecodeCapability",
            "RouterPayloadShape",
            "ScoreSemantics",
            "validate_decoded_routing",
            "native_id_map",
            "RoutingDecodeError",
            "assignment_indices",
            "run_routing_forward",
            "RoutingHookDecoder",
            "TokenSequencePolicy",
            "validate_observed_routing",
            "rectangular projection",
            "no central branching",
            "family-neutral",
            "final VM",
        ):
            with self.subTest(term=term):
                self.assertIn(term, runtime_docs)
        universe_source = (
            ROOT / "src" / "moeatlas" / "adapters" / "universe.py"
        ).read_text()
        for term in (
            "ROUTING_UNIVERSE_SCHEMA_VERSION",
            "LayerRoutingUniverse",
            "RectangularProjection",
            "RoutingUniverseError",
            "manifest_type",
            "routing_universe",
        ):
            self.assertIn(term, universe_source)

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

    def test_shared_expert_capture_rule_is_model_neutral(self) -> None:
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        runtime = (ROOT / "docs" / "runtime.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document in (architecture, runtime, readme):
            self.assertIn("shared", document.lower())
        self.assertIn("model-neutral", architecture)
        self.assertIn("expert_keys", runtime)
        self.assertIn("non-routed metadata", readme)

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
        for term in (
            "RoutingForwardResult",
            "MixtralRoutingForwardResult",
            "run_mixtral_routing_forward",
            "run_qwen3_5_routing_forward",
        ):
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

    def test_run_export_bundle_docs_and_surface_are_present(self) -> None:
        storage = (ROOT / "docs" / "storage.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                storage,
                (
                    "routing_run_export",
                    "export_run_bundle",
                    "verify_run_bundle",
                    "import_run_bundle",
                    "max_event_rows",
                    "max_file_bytes",
                    "tokens.jsonl",
                    "routing.jsonl",
                    "sha256:<64hex>",
                    "token_text_stored",
                    "event_index",
                    "canonically encoded",
                    "byte-identical bundles",
                    "forged digests",
                    "created=False",
                    "RunBundleError",
                    "conflict",
                    "EXPERIMENTAL",
                ),
            ),
            (architecture, ("run-evidence export bundle",)),
            (ledger, ("does not change MV-01/MV-08",)),
            (readme, ("export_run_bundle",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "store" / "run_export.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "store" / "__init__.py").read_text()
        for term in (
            "RUN_EXPORT_SCHEMA_VERSION",
            "BUNDLE_MANIFEST_TYPE",
            "RunBundleError",
            "RunBundleReceipt",
            "export_run_bundle",
            "verify_run_bundle",
            "import_run_bundle",
        ):
            self.assertIn(term, source)
            self.assertIn(term, exports)
        self.assertTrue((ROOT / "tests" / "test_store_run_export.py").is_file())

    def test_assignment_query_seam_docs_and_surface_are_present(self) -> None:
        storage = (ROOT / "docs" / "storage.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        workspace_doc = (ROOT / "docs" / "workspace.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                storage,
                (
                    "query_routing_run_assignments",
                    "RoutingShardAssignmentQuery",
                    "RoutingRunQueryError",
                    "RoutingRunInventoryError",
                    "query_assignments",
                    "canonical order",
                    "conflicts",
                ),
            ),
            (architecture, ("assignment-query seam",)),
            (workspace_doc, ("query_assignments",)),
            (roadmap, ("query_routing_run_assignments",)),
            (ledger, ("Routing-run assignment query seam",)),
            (readme, ("query_routing_run_assignments",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "store" / "routing_shards.py").read_text()
        ports_source = (ROOT / "src" / "moeatlas" / "store" / "ports.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "store" / "__init__.py").read_text()
        analysis = (ROOT / "src" / "moeatlas" / "analysis" / "routing_load.py").read_text()
        for term in ("RoutingShardAssignmentQuery", "RoutingRunQueryError"):
            self.assertIn(term, source)
            self.assertIn(term, exports)
        for term in (
            "RoutingShardAssignmentQuery",
            "query_assignments",
            "query_routing_run_assignments",
        ):
            self.assertIn(term, ports_source)
        self.assertIn("_storage.query_routing_run_assignments(", analysis)
        # Analysis must not reach into concrete shard internals any more.
        for private in (
            "_validate_sources",
            "_validate_routing_load_source",
            "_validate_file_metadata",
            "_read_shard_manifest",
            "_existing_run_parent",
            "_validate_workspace",
        ):
            with self.subTest(private=private):
                self.assertNotIn(f"_storage.{private}", analysis)
        self.assertTrue((ROOT / "tests" / "test_store_assignment_queries.py").is_file())

    def test_tabular_run_export_docs_and_surface_are_present(self) -> None:
        storage = (ROOT / "docs" / "storage.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        workspace_doc = (ROOT / "docs" / "workspace.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                storage,
                (
                    "export_run_tables",
                    "verify_run_tables",
                    "RunTableError",
                    "RunTableReceipt",
                    "byte-deterministic",
                    "one-way",
                    "export-staging",
                ),
            ),
            (architecture, ("export_run_tables",)),
            (workspace_doc, ("export_run_tables",)),
            (roadmap, ("export_run_tables",)),
            (ledger, ("Tabular run exports",)),
            (readme, ("export_run_tables",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "store" / "run_tables.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "store" / "__init__.py").read_text()
        for term in (
            "RUN_TABLES_SCHEMA_VERSION",
            "TABLES_MANIFEST_TYPE",
            "RunTableError",
            "RunTableFileEntry",
            "RunTableReceipt",
            "export_run_tables",
            "verify_run_tables",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The tabular projection stays one-way: no import path back into shards.
        self.assertNotIn("import_run_bundle", source)
        self.assertNotIn("append_routing_shard", source)
        self.assertTrue((ROOT / "tests" / "test_store_run_tables.py").is_file())

    def test_dataset_reader_docs_and_surface_are_present(self) -> None:
        runs_doc = (ROOT / "docs" / "runs.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runs_doc,
                (
                    "Bounded dataset reading",
                    "read_dataset_rows",
                    "DatasetRow",
                    "DatasetReadError",
                    "plan_dataset_batches",
                    "project_dataset_rows",
                    "hf_datasets",
                    "Descriptors never fetch",
                ),
            ),
            (architecture, ("moeatlas.services.datasets",)),
            (roadmap, ("read_dataset_rows",)),
            (ledger, ("Bounded dataset reading service",)),
            (readme, ("read_dataset_rows",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "services" / "datasets.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "services" / "__init__.py").read_text()
        for term in (
            "DATASET_READER_SCHEMA_VERSION",
            "DATASET_COLUMN_ROLES",
            "DatasetReadError",
            "DatasetRow",
            "read_dataset_rows",
            "plan_dataset_batches",
            "project_dataset_rows",
            "resolve_dataset_location",
            "validate_column_mapping",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # Descriptors never fetch data: no network clients anywhere in the reader.
        for forbidden in ("urllib", "requests", "httpx", "huggingface", "socket"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_services_datasets.py").is_file())

    def test_run_engine_docs_and_surface_are_present(self) -> None:
        runs_doc = (ROOT / "docs" / "runs.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runs_doc,
                (
                    "Deterministic execution core",
                    "execute_row_schedule",
                    "ExecutionOutcome",
                    "RowFailure",
                    "should_cancel",
                    "cancelled_before_row",
                    "executing",
                ),
            ),
            (architecture, ("moeatlas.services.run_engine",)),
            (roadmap, ("execute_row_schedule",)),
            (ledger, ("Deterministic run-engine execution core",)),
            (readme, ("execute_row_schedule",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "services" / "run_engine.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "services" / "__init__.py").read_text()
        for term in (
            "RUN_ENGINE_SCHEMA_VERSION",
            "ROW_FAILURE_KINDS",
            "EXECUTION_PROGRESS_STAGE",
            "RunEngineError",
            "RowFailure",
            "RowResult",
            "RowRecord",
            "ExecutionOutcome",
            "execute_row_schedule",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The core stays deterministic and family-blind: no clocks, randomness,
        # network, or model dependencies.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "urllib",
            "requests",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_services_run_engine.py").is_file())

    def test_run_input_preparation_docs_and_surface_are_present(self) -> None:
        runs_doc = (ROOT / "docs" / "runs.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runs_doc,
                (
                    "Input preparation",
                    "prepare_input_rows",
                    "plan_input_batches",
                    "RunInputError",
                    "never branches on input kind",
                ),
            ),
            (architecture, ("moeatlas.services.run_inputs",)),
            (roadmap, ("prepare_input_rows",)),
            (ledger, ("Run input preparation service",)),
            (readme, ("prepare_input_rows",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "services" / "run_inputs.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "services" / "__init__.py").read_text()
        for term in (
            "RUN_INPUTS_SCHEMA_VERSION",
            "RunInputError",
            "prepare_prompt_rows",
            "plan_input_batches",
            "prepare_input_rows",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # Preparation stays deterministic and family-blind.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_services_run_inputs.py").is_file())

    def test_run_service_docs_and_surface_are_present(self) -> None:
        runs_doc = (ROOT / "docs" / "runs.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                runs_doc,
                (
                    "Headless run service",
                    "execute_specification",
                    "RunExecutionReport",
                    "run_checkpoint",
                    "load_checkpoint",
                    "resume_from",
                    "publish_run_report",
                    "never reads a clock",
                ),
            ),
            (architecture, ("moeatlas.services.run_service",)),
            (roadmap, ("execute_specification",)),
            (ledger, ("Headless run-engine service surface",)),
            (readme, ("execute_specification",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "services" / "run_service.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "services" / "__init__.py").read_text()
        for term in (
            "RUN_SERVICE_SCHEMA_VERSION",
            "CHECKPOINT_SCHEMA_VERSION",
            "RunServiceError",
            "RunCheckpoint",
            "build_initial_record",
            "derive_run_failure",
            "execute_specification",
            "load_checkpoint",
            "publish_run_report",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The service stays deterministic, family-blind, and network-free.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "urllib",
            "requests",
            "httpx",
            "socket",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_services_run_service.py").is_file())

    def test_task_association_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Task association metrics",
                    "TaskExpertCounts",
                    "analyze_task_association",
                    "TaskAssociationMatrix",
                    "moeatlas.task_association",
                    "never specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.task_association",)),
            (roadmap, ("analyze_task_association",)),
            (ledger, ("Task association metrics",)),
            (readme, ("analyze_task_association",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "task_association.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "TASK_ASSOCIATION_SCHEMA_VERSION",
            "TaskAssociationError",
            "TaskAssociationMatrix",
            "TaskExpertCounts",
            "analyze_task_association",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The math layer stays pure: no storage reads, clocks, or randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_task_association.py").is_file())

    def test_evidence_card_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Evidence Cards",
                    "EvidenceCard",
                    "EVIDENCE_TIERS",
                    "moeatlas.evidence_card",
                    "not measured",
                ),
            ),
            (architecture, ("moeatlas.analysis.evidence_cards",)),
            (roadmap, ("EvidenceCard",)),
            (ledger, ("Evidence Cards",)),
            (readme, ("EvidenceCard()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "evidence_cards.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "EVIDENCE_CARD_SCHEMA_VERSION",
            "EVIDENCE_TIERS",
            "EvidenceCardError",
            "EvidenceCard",
            "TaskAssociationSection",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The card layer stays pure: no storage reads, clocks, or randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_evidence_cards.py").is_file())

    def test_routing_agreement_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Prompt-vs-rollout agreement",
                    "PromptRolloutCounts",
                    "analyze_routing_agreement",
                    "RoutingAgreement",
                    "moeatlas.routing_agreement",
                    "never specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.routing_agreement",)),
            (roadmap, ("analyze_routing_agreement",)),
            (ledger, ("Prompt-vs-rollout routing agreement",)),
            (readme, ("analyze_routing_agreement()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "routing_agreement.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "ROUTING_AGREEMENT_SCHEMA_VERSION",
            "RoutingAgreementError",
            "PromptRolloutCounts",
            "RoutingAgreement",
            "analyze_routing_agreement",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The agreement layer stays pure: no storage reads, clocks, randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue(
            (ROOT / "tests" / "test_analysis_routing_agreement.py").is_file()
        )

    def test_association_stability_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Cross-run association stability",
                    "analyze_association_stability",
                    "AssociationStability",
                    "moeatlas.association_stability",
                    "never specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.association_stability",)),
            (roadmap, ("analyze_association_stability",)),
            (ledger, ("Cross-run association stability",)),
            (readme, ("analyze_association_stability()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "association_stability.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "ASSOCIATION_STABILITY_SCHEMA_VERSION",
            "AssociationStabilityError",
            "AssociationStability",
            "analyze_association_stability",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The stability layer stays pure: no storage reads, clocks, randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue(
            (ROOT / "tests" / "test_analysis_association_stability.py").is_file()
        )

    def test_router_margin_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Router margin",
                    "RouterMarginSamples",
                    "analyze_router_margin",
                    "RouterMarginSummary",
                    "moeatlas.router_margin",
                    "never specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.router_margin",)),
            (roadmap, ("analyze_router_margin",)),
            (ledger, ("Router margin",)),
            (readme, ("analyze_router_margin()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "router_margin.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "ROUTER_MARGIN_SCHEMA_VERSION",
            "RouterMarginError",
            "RouterMarginSamples",
            "RouterMarginSummary",
            "analyze_router_margin",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The margin layer stays pure: no storage reads, clocks, randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_router_margin.py").is_file())

    def test_route_churn_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Route churn",
                    "RouteChurnSequences",
                    "analyze_route_churn",
                    "RouteChurnSummary",
                    "moeatlas.route_churn",
                    "never specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.route_churn",)),
            (roadmap, ("analyze_route_churn",)),
            (ledger, ("Route churn",)),
            (readme, ("analyze_route_churn()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "route_churn.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "ROUTE_CHURN_SCHEMA_VERSION",
            "RouteChurnError",
            "RouteChurnSequences",
            "RouteChurnSummary",
            "analyze_route_churn",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The churn layer stays pure: no storage reads, clocks, randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_route_churn.py").is_file())

    def test_corouting_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Co-routing graphs",
                    "ExpertCoRoutingCounts",
                    "summarize_co_routing",
                    "CoRoutingGraph",
                    "moeatlas.corouting",
                    "never implies specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.corouting",)),
            (roadmap, ("summarize_co_routing",)),
            (ledger, ("Co-routing graphs",)),
            (readme, ("summarize_co_routing()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "corouting.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "COROUTING_SCHEMA_VERSION",
            "CoRoutingError",
            "CoRoutingGraph",
            "ExpertCoRoutingCounts",
            "summarize_co_routing",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The co-routing layer stays pure: no storage, clocks, or randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_corouting.py").is_file())

    def test_expert_similarity_docs_and_surface_are_present(self) -> None:
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                (
                    "Expert similarity",
                    "ExpertVectors",
                    "analyze_expert_similarity",
                    "ExpertSimilarity",
                    "moeatlas.expert_similarity",
                    "it never implies specialization or causality",
                ),
            ),
            (architecture, ("moeatlas.analysis.expert_similarity",)),
            (roadmap, ("analyze_expert_similarity",)),
            (ledger, ("Expert similarity",)),
            (readme, ("analyze_expert_similarity()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (
            ROOT / "src" / "moeatlas" / "analysis" / "expert_similarity.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "analysis" / "__init__.py").read_text()
        for term in (
            "EXPERT_SIMILARITY_SCHEMA_VERSION",
            "ExpertSimilarityError",
            "ExpertSimilarity",
            "ExpertVectors",
            "analyze_expert_similarity",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The similarity layer stays pure: no storage, clocks, or randomness.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        self.assertTrue((ROOT / "tests" / "test_analysis_expert_similarity.py").is_file())

    def test_adapter_registry_docs_and_surface_are_present(self) -> None:
        adapters_doc = (ROOT / "docs" / "adapters.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                adapters_doc,
                (
                    "Adapter plugin registry",
                    "AdapterPluginRecord",
                    "collect_adapter_registry()",
                    "AdapterRegistryPolicy",
                    "moeatlas.adapter_registry",
                    "match_adapters_for_family()",
                ),
            ),
            (architecture, ("entry-point plugin registry",)),
            (roadmap, ("collect_adapter_registry()",)),
            (ledger, ("Adapter plugin registry",)),
            (readme, ("collect_adapter_registry()",)),
            (
                (ROOT / "docs" / "cli.md").read_text(),
                (
                    "moeatlas adapters list",
                    "moeatlas.adapter_registry",
                    "moeatlas run WORKSPACE",
                    "--executor NAME",
                    "moeatlas export WORKSPACE RUN_KEY",
                    "manifest sha256",
                ),
            ),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        source = (ROOT / "src" / "moeatlas" / "adapters" / "registry.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "adapters" / "__init__.py").read_text()
        for term in (
            "ADAPTER_REGISTRY_SCHEMA_VERSION",
            "ENTRY_POINT_GROUP",
            "AdapterPluginRecord",
            "AdapterRegistryEntry",
            "AdapterRegistryError",
            "AdapterRegistryPolicy",
            "AdapterRegistryReport",
            "apply_registry_policy",
            "builtin_adapter_records",
            "collect_adapter_registry",
            "discover_entry_point_records",
            "match_adapters_for_family",
        ):
            with self.subTest(term=term):
                self.assertIn(term, source)
                self.assertIn(term, exports)
        # The registry stays model-free: no storage, clocks, randomness, or
        # model/network imports.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "duckdb",
            "urllib",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)
        cli_source = (ROOT / "src" / "moeatlas" / "cli.py").read_text()
        for term in ("adapters", "adapters_list", "_handle_adapters_list"):
            with self.subTest(term=term):
                self.assertIn(term, cli_source)
        self.assertTrue((ROOT / "tests" / "test_adapters_registry.py").is_file())
        self.assertTrue((ROOT / "tests" / "test_cli_adapters.py").is_file())
        self.assertTrue((ROOT / "tests" / "test_cli_run.py").is_file())
        self.assertTrue((ROOT / "tests" / "test_cli_export.py").is_file())

    def test_server_docs_and_surface_are_present(self) -> None:
        server_doc = (ROOT / "docs" / "server.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                server_doc,
                (
                    "create_app()",
                    "/healthz",
                    "/api/workspace",
                    "/api/runs",
                    "/api/adapters",
                    "moeatlas ui WORKSPACE",
                    "--allow-remote",
                    "workspace is not initialized",
                    "deferred release-engineering evidence",
                ),
            ),
            (roadmap, ("create_app()", "moeatlas ui WORKSPACE")),
            (ledger, ("Local server and UI launch",)),
            (readme, ("moeatlas.server.create_app()",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        app_source = (ROOT / "src" / "moeatlas" / "server" / "app.py").read_text()
        dto_source = (ROOT / "src" / "moeatlas" / "server" / "dto.py").read_text()
        exports = (ROOT / "src" / "moeatlas" / "server" / "__init__.py").read_text()
        for term in (
            "SERVER_SCHEMA_VERSION",
            "ServerDependencyError",
            "create_app",
        ):
            with self.subTest(term=term):
                self.assertIn(term, app_source)
                self.assertIn(term, exports)
        for term in (
            "HealthResponse",
            "WorkspaceResponse",
            "RunEntryResponse",
            "RunsResponse",
            "AdapterEntryResponse",
            "AdaptersResponse",
        ):
            with self.subTest(term=term):
                self.assertIn(term, dto_source)
                self.assertIn(term, exports)
        # The server stays read-only and model-free: no storage writes,
        # clocks, randomness, or model imports.
        for forbidden in (
            "import time",
            "import random",
            "datetime",
            "torch",
            "transformers",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, app_source)
                self.assertNotIn(forbidden, dto_source)
        cli_source = (ROOT / "src" / "moeatlas" / "cli.py").read_text()
        for term in ("_handle_ui", "_run_ui_server", "--allow-remote"):
            with self.subTest(term=term):
                self.assertIn(term, cli_source)
        self.assertTrue((ROOT / "tests" / "test_server_app.py").is_file())
        self.assertTrue((ROOT / "tests" / "test_cli_ui.py").is_file())

    def test_intervention_docs_and_surface_are_present(self) -> None:
        interventions_doc = (ROOT / "docs" / "interventions.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        architecture = (ROOT / "docs" / "architecture.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        readme = (ROOT / "readme.md").read_text()
        for document, terms in (
            (
                interventions_doc,
                (
                    "run_intervention()",
                    "InterventionRecipe",
                    "InterventionBudget",
                    "InterventionCapability",
                    "moeatlas.intervention_recipe",
                    "moeatlas.intervention_budget",
                    "moeatlas.intervention_outcome",
                    "sha256:<64 hex>",
                    "restore",
                ),
            ),
            (roadmap, ("run_intervention()", "InterventionRecipe")),
            (architecture, ("moeatlas.interventions",)),
            (ledger, ("Intervention mechanics",)),
            (readme, ("moeatlas.interventions",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        recipes_source = (
            ROOT / "src" / "moeatlas" / "interventions" / "recipes.py"
        ).read_text()
        engine_source = (
            ROOT / "src" / "moeatlas" / "interventions" / "engine.py"
        ).read_text()
        exports = (ROOT / "src" / "moeatlas" / "interventions" / "__init__.py").read_text()
        for term in (
            "InterventionOperation",
            "InterventionRecipe",
            "InterventionBudget",
            "INTERVENTION_SCHEMA_VERSION",
        ):
            with self.subTest(term=term):
                self.assertIn(term, recipes_source)
                self.assertIn(term, exports)
        for term in (
            "run_intervention",
            "InterventionOutcome",
            "InterventionEngineError",
            "InterventionCapability",
            "INTERVENTION_ENGINE_SCHEMA_VERSION",
        ):
            with self.subTest(term=term):
                self.assertIn(term, engine_source)
                self.assertIn(term, exports)
        # The interventions package stays family-blind and model-free.
        for forbidden in ("torch", "transformers", "duckdb", "urllib"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, recipes_source)
                self.assertNotIn(forbidden, engine_source)
        analysis_doc = (ROOT / "docs" / "analysis.md").read_text()
        for document, terms in (
            (
                analysis_doc,
                ("analyze_causal_evidence", "moeatlas.causal_evidence", "CausalPair"),
            ),
            (roadmap, ("analyze_causal_evidence", "moeatlas.causal_evidence")),
            (ledger, ("Causal evidence summaries",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        causal_source = (
            ROOT / "src" / "moeatlas" / "analysis" / "causal_evidence.py"
        ).read_text()
        analysis_exports = (
            ROOT / "src" / "moeatlas" / "analysis" / "__init__.py"
        ).read_text()
        for term in (
            "CAUSAL_EVIDENCE_SCHEMA_VERSION",
            "CausalEvidence",
            "CausalPair",
            "analyze_causal_evidence",
        ):
            with self.subTest(term=term):
                self.assertIn(term, causal_source)
                self.assertIn(term, analysis_exports)
        # The causal-evidence layer stays pure: no clocks, randomness,
        # storage, or model imports.
        for forbidden in ("import time", "import random", "datetime", "duckdb",
                          "urllib", "torch", "transformers"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, causal_source)

    def test_retention_docs_and_surface_are_present(self) -> None:
        workspace_doc = (ROOT / "docs" / "workspace.md").read_text()
        roadmap = (ROOT / "docs" / "roadmap.md").read_text()
        ledger = (ROOT / "docs" / "model-validation-ledger.md").read_text()
        for document, terms in (
            (
                workspace_doc,
                (
                    "RetentionPolicy",
                    "evaluate_retention(entries, policy)",
                    "moeatlas.retention_report",
                    "evaluation, not deletion",
                ),
            ),
            (roadmap, ("evaluate_retention()", "moeatlas.retention_report")),
            (ledger, ("Retention evaluation",)),
        ):
            for term in terms:
                with self.subTest(term=term):
                    self.assertIn(term, document)
        retention_source = (
            ROOT / "src" / "moeatlas" / "services" / "retention.py"
        ).read_text()
        services_exports = (
            ROOT / "src" / "moeatlas" / "services" / "__init__.py"
        ).read_text()
        for term in (
            "RETENTION_SCHEMA_VERSION",
            "RetentionError",
            "RetentionPolicy",
            "RetentionReport",
            "evaluate_retention",
        ):
            with self.subTest(term=term):
                self.assertIn(term, retention_source)
                self.assertIn(term, services_exports)
        # Retention evaluation never deletes: no catalog writes, no
        # unlinking, and no model or storage-engine imports.
        for forbidden in ("unlink", "rmtree", "shutil", "duckdb", "torch",
                          "transformers"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, retention_source)

    def test_release_engineering_files_are_present(self) -> None:
        required = (
            ROOT / "SECURITY.md",
            ROOT / "CODE_OF_CONDUCT.md",
            ROOT / "CHANGELOG.md",
            ROOT / ".github" / "workflows" / "ci.yml",
            ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "bug_report.md",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "feature_request.md",
            ROOT / ".github" / "ISSUE_TEMPLATE" / "config.yml",
            ROOT / "examples" / "synthetic_workspace.py",
            ROOT / "src" / "moeatlas" / "benchmarks.py",
            ROOT / "tests" / "test_benchmarks.py",
            ROOT / "tests" / "test_examples.py",
        )
        for path in required:
            with self.subTest(path=path):
                self.assertTrue(path.is_file(), path)
        security = (ROOT / "SECURITY.md").read_text()
        for term in ("Report a vulnerability", "There is none"):
            with self.subTest(term=term):
                self.assertIn(term, security)
        changelog = (ROOT / "CHANGELOG.md").read_text()
        for term in ("Keep a Changelog", "Semantic Versioning", "[Unreleased]"):
            with self.subTest(term=term):
                self.assertIn(term, changelog)
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
        for term in (
            '"3.11"',
            '"3.12"',
            '"3.13"',
            "uv sync --locked --extra dev",
            "pytest -q",
            "ruff check src tests",
            "unittest discover -s tests -t .",
            "uv build --no-sources",
        ):
            with self.subTest(term=term):
                self.assertIn(term, ci)
        pull_request = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
        for term in ("uv run --locked pytest -q", "deferred rows stay deferred"):
            with self.subTest(term=term):
                self.assertIn(term, pull_request)
        example_source = (ROOT / "examples" / "synthetic_workspace.py").read_text()
        for term in (
            "initialize_workspace",
            "register_run",
            "evaluate_retention",
            "downloads a model, touches the network, or requires a GPU",
        ):
            with self.subTest(term=term):
                self.assertIn(term, example_source)

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
