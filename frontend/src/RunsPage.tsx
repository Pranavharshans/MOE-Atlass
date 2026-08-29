import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, GitBranch, Lightning, Pulse, ShieldCheck, XCircle } from "@phosphor-icons/react";
import { postJson, useJob } from "./jobs";
import {
  FailureDiagnostics,
  MetricCard,
  OperationCapabilityList,
  StatusDot,
  type InterventionCapability,
} from "./ui";

type RunEntry = {
  run_key: string;
  run_name?: string | null;
  state?: string | null;
  token_event_count?: number;
  routing_event_count?: number;
};
type RunGroupChild = { slug: string; child_run_name: string; run_key?: string | null; state: string; dataset_id: string; dataset_config?: string | null };
type RunGroup = { group_name: string; state: string; children: RunGroupChild[] };

type RunSummary = {
  status: string;
  reason?: string | null;
  adapter_name?: string | null;
  adapter_version?: string | null;
  token_count?: number | null;
  assignment_count?: number | null;
  layer_count?: number | null;
  expert_count?: number | null;
  routed_top_k?: number | null;
  inspection_digest?: string | null;
};

type ActivitySummary = {
  active_expert_cells?: number;
  inactive_expert_cells?: number;
  total_event_count?: number;
  layers?: Array<{ layer_key: string; event_counts: number[] }>;
  candidate_ranking?: {
    ranked_cell_count: number;
    incomplete_cell_count: number;
    evidence_complete: boolean;
    high_observed: ExpertCandidate[];
    low_observed: ExpertCandidate[];
    claim_boundary: string;
  };
};

type ExpertCandidate = {
  layer_index: number;
  expert_index: number;
  expert_key: string;
  routing_share: number;
  contribution_variance: number;
  total_contribution: number;
};

type ActivityResponse = { status: string; reason?: string | null; summary?: ActivitySummary | null };
type ArchitectureResponse = { status: string; reason?: string | null; report?: Record<string, unknown> | null };
type RoutingSimilarity = {
  top_n: number;
  baseline_token_count: number;
  comparison_token_count: number;
  mean_js_divergence: number;
  mean_spearman?: number | null;
  mean_top_n_jaccard: number;
  undefined_spearman_layers: number;
};
type RoutingSimilarityResponse = {
  status: "available" | "unavailable";
  reason?: string | null;
  report?: RoutingSimilarity | null;
};
type InterventionTarget = {
  label: string;
  layer_index: number;
  expert_index: number;
  layer_key: string;
  expert_key: string;
};
type InterventionTargetsResponse = {
  status: "available" | "unsupported";
  reason?: string | null;
  targets: InterventionTarget[];
  capability?: InterventionCapability | null;
};
type InterventionEvidence = {
  baseline_run_key: string;
  intervention_run_key: string;
  recipe_fingerprint: string;
  restoration_status: string;
  all_targets_exercised: boolean;
  row_count: number;
  changed_output_rows: number;
  changed_output_fraction?: number | null;
  score_name?: string | null;
  baseline_task_score?: number | null;
  intervention_task_score?: number | null;
  task_score_delta?: number | null;
  baseline_mean_latency_ms?: number | null;
  intervention_mean_latency_ms?: number | null;
  latency_delta_percent?: number | null;
  target_invocation_counts: Record<string, number>;
};
type InterventionStudy = {
  study_id: string;
  claim_status: "inconclusive" | "replicated" | "controlled";
  claim_reason: string;
  replication_count: number;
  control_count: number;
  score_name?: string | null;
  task_effect: {
    mean?: number | null;
    stdev?: number | null;
    confidence_interval_95?: [number, number] | null;
    direction_consistency?: number | null;
  };
};

const RUN_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/;

function runLabel(entry: RunEntry | undefined): string {
  return entry?.run_name ?? (entry ? `legacy-${entry.run_key.slice(-8)}` : "unknown-run");
}

export function RunsPage() {
  const [state, setState] = useState<"loading" | "ready" | "unavailable">("loading");
  const [entries, setEntries] = useState<RunEntry[]>([]);
  const [runGroups, setRunGroups] = useState<RunGroup[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [summaryState, setSummaryState] = useState<"idle" | "loading" | "ready" | "unavailable">("idle");
  const [metric, setMetric] = useState<"assignment_counts" | "assignment_shares" | "load_ratios">("assignment_counts");
  const [activity, setActivity] = useState<ActivityResponse | null>(null);
  const [architecture, setArchitecture] = useState<ArchitectureResponse | null>(null);
  const [comparisonRun, setComparisonRun] = useState("");
  const [comparisonMetric, setComparisonMetric] = useState<"count_deltas" | "share_deltas" | "ratio_deltas">("count_deltas");
  const [routingSimilarity, setRoutingSimilarity] = useState<RoutingSimilarityResponse | null>(null);
  const [interventionTargets, setInterventionTargets] = useState<InterventionTargetsResponse | null>(null);
  const [selectedTargets, setSelectedTargets] = useState<string[]>([]);
  const [targetSearch, setTargetSearch] = useState("");
  const [interventionOperation, setInterventionOperation] = useState<"ablate" | "scale">("ablate");
  const [interventionRunName, setInterventionRunName] = useState("");
  const [scaleFactor, setScaleFactor] = useState("0.5");
  const [interventionJobId, setInterventionJobId] = useState<string | null>(null);
  const [interventionStatus, setInterventionStatus] = useState<string | null>(null);
  const [interventionEvidence, setInterventionEvidence] = useState<InterventionEvidence | null>(null);
  const [studyCandidates, setStudyCandidates] = useState<InterventionEvidence[]>([]);
  const [studyRuns, setStudyRuns] = useState<string[]>([]);
  const [controlRuns, setControlRuns] = useState<string[]>([]);
  const [studyStatus, setStudyStatus] = useState<string | null>(null);
  const [study, setStudy] = useState<InterventionStudy | null>(null);
  const heatmapFrame = useRef<HTMLIFrameElement | null>(null);
  const interventionJob = useJob(interventionJobId);
  const visibleInterventionTargets = useMemo(() => {
    const query = targetSearch.trim().toLowerCase();
    const targets = interventionTargets?.targets ?? [];
    if (!query) return targets;
    return targets.filter((target) => {
      const searchable = `l${target.layer_index} e${target.expert_index} ${target.layer_index} ${target.expert_index} ${target.label}`.toLowerCase();
      return searchable.includes(query) || selectedTargets.includes(target.label);
    });
  }, [interventionTargets, selectedTargets, targetSearch]);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/runs", { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("unavailable");
        return response.json() as Promise<{ entries?: RunEntry[] }>;
      })
      .then((document) => {
        const nextEntries = Array.isArray(document.entries) ? document.entries : [];
        setEntries(nextEntries);
        setSelectedRun((current) => current || nextEntries[0]?.run_key || "");
        setComparisonRun((current) => current || nextEntries[1]?.run_key || "");
        setState("ready");
      })
      .catch((cause) => { if (cause instanceof DOMException && cause.name === "AbortError") return; setState("unavailable"); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/run-groups", { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((response) => response.ok ? response.json() as Promise<{ groups?: RunGroup[] }> : Promise.reject(new Error("unavailable")))
      .then((document) => setRunGroups(Array.isArray(document.groups) ? document.groups : []))
      .catch((cause) => { if (!(cause instanceof DOMException && cause.name === "AbortError")) setRunGroups([]); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selectedRun) {
      setSummary(null);
      setSummaryState("idle");
      return undefined;
    }
    const controller = new AbortController();
    setSummaryState("loading");
    fetch(`/api/runs/${encodeURIComponent(selectedRun)}/summary`, { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("unavailable");
        return response.json() as Promise<RunSummary>;
      })
      .then((document) => { setSummary(document); setSummaryState("ready"); })
      .catch((cause) => { if (cause instanceof DOMException && cause.name === "AbortError") return; setSummary(null); setSummaryState("unavailable"); });
    return () => controller.abort();
  }, [selectedRun]);

  useEffect(() => {
    if (!selectedRun || !comparisonRun || selectedRun === comparisonRun) {
      setRoutingSimilarity(null);
      return undefined;
    }
    const controller = new AbortController();
    const query = new URLSearchParams({
      baseline_run_key: comparisonRun,
      comparison_run_key: selectedRun,
      top_n: "5",
    });
    fetch(`/api/compare/similarity?${query.toString()}`, {
      headers: { Accept: "application/json" },
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error("unavailable");
        return response.json() as Promise<RoutingSimilarityResponse>;
      })
      .then(setRoutingSimilarity)
      .catch((cause) => {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setRoutingSimilarity({
          status: "unavailable",
          reason: "These runs cannot be compared over one shared routing topology.",
        });
      });
    return () => controller.abort();
  }, [comparisonRun, selectedRun]);

  function syncHeatmapSelection(targets: string[] = selectedTargets) {
    const document = heatmapFrame.current?.contentDocument;
    if (!document) return;
    const selected = new Set(targets);
    document.querySelectorAll<HTMLElement>("[data-target]").forEach((cell) => {
      const active = selected.has(cell.dataset.target ?? "");
      cell.classList.toggle("is-selected", active);
      cell.setAttribute("aria-checked", active ? "true" : "false");
    });
  }

  function prepareCandidate(candidate: ExpertCandidate, lane: "high" | "low") {
    const target = interventionTargets?.targets.find(
      (item) => item.expert_key === candidate.expert_key,
    );
    if (!target) {
      setInterventionStatus("This observed expert is not independently controllable in the current runtime.");
      return;
    }
    setSelectedTargets([target.label]);
    setInterventionStatus(
      `Prepared the ${lane}-observed L${candidate.layer_index} × E${candidate.expert_index} candidate. Run the paired intervention to test causality.`,
    );
  }

  function bindHeatmapTargets() {
    const document = heatmapFrame.current?.contentDocument;
    if (!document) return;
    document.querySelectorAll<HTMLElement>("[data-target]").forEach((cell) => {
      const toggle = () => {
        const target = cell.dataset.target;
        if (!target) return;
        const supported = interventionTargets?.targets.some((item) => item.label === target);
        if (!supported) {
          setInterventionStatus(
            interventionTargets?.reason ?? "This model does not expose that expert for live intervention.",
          );
          return;
        }
        setSelectedTargets((current) => (
          current.includes(target)
            ? current.filter((item) => item !== target)
            : [...current, target].sort()
        ));
        setInterventionStatus(null);
      };
      cell.onclick = toggle;
      cell.onkeydown = (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          toggle();
        }
      };
    });
    syncHeatmapSelection();
  }

  useEffect(() => {
    bindHeatmapTargets();
  }, [interventionTargets, metric, selectedRun]);

  useEffect(() => {
    syncHeatmapSelection();
  }, [selectedTargets]);

  useEffect(() => {
    if (!selectedRun) {
      setInterventionTargets(null);
      setInterventionEvidence(null);
      return undefined;
    }
    const controller = new AbortController();
    Promise.all([
      fetch(`/api/runs/${encodeURIComponent(selectedRun)}/intervention-targets`, { headers: { Accept: "application/json" }, signal: controller.signal }).then((response) => response.json() as Promise<InterventionTargetsResponse>),
      fetch(`/api/runs/${encodeURIComponent(selectedRun)}/intervention`, { headers: { Accept: "application/json" }, signal: controller.signal }).then((response) => response.json() as Promise<{ status: string; evidence?: InterventionEvidence | null }>),
    ]).then(([targets, evidence]) => {
      const nextEvidence = evidence.status === "available" ? evidence.evidence ?? null : null;
      setInterventionTargets(targets);
      setSelectedTargets([]);
      setTargetSearch("");
      setInterventionEvidence(nextEvidence);
      setStudyRuns(nextEvidence ? [selectedRun] : []);
      setControlRuns([]);
      setStudy(null);
      setStudyStatus(null);
    }).catch((cause) => {
      if (cause instanceof DOMException && cause.name === "AbortError") return;
      setInterventionTargets(null);
      setInterventionEvidence(null);
    });
    return () => controller.abort();
  }, [selectedRun]);

  useEffect(() => {
    if (interventionJob?.state !== "completed") return;
    const runKey = interventionJob.result?.run_key;
    if (typeof runKey !== "string") return;
    if (selectedRun === runKey) return;
    const controller = new AbortController();
    fetch("/api/runs", { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((response) => response.json() as Promise<{ entries?: RunEntry[] }>)
      .then((document) => {
        const nextEntries = Array.isArray(document.entries) ? document.entries : [];
        setEntries(nextEntries);
        setComparisonRun(selectedRun);
        setSelectedRun(runKey);
        setInterventionStatus("Intervention completed; paired evidence is ready.");
      })
      .catch(() => setInterventionStatus("Intervention completed; refresh to inspect it."));
    return () => controller.abort();
  }, [interventionJob?.state, interventionJob?.result, selectedRun]);

  useEffect(() => {
    if (!interventionEvidence || !entries.length) {
      setStudyCandidates([]);
      return undefined;
    }
    const controller = new AbortController();
    void Promise.all(entries.map(async (entry) => {
      try {
        const response = await fetch(
          `/api/runs/${encodeURIComponent(entry.run_key)}/intervention`,
          { headers: { Accept: "application/json" }, signal: controller.signal },
        );
        if (!response.ok) return null;
        const document = await response.json() as {
          status: string;
          evidence?: InterventionEvidence | null;
        };
        return document.status === "available" ? document.evidence ?? null : null;
      } catch {
        return null;
      }
    })).then((documents) => {
      if (!controller.signal.aborted) {
        setStudyCandidates(documents.filter((item): item is InterventionEvidence => item !== null));
      }
    });
    return () => controller.abort();
  }, [entries, interventionEvidence]);

  useEffect(() => {
    if (!selectedRun) {
      setActivity(null);
      setArchitecture(null);
      return undefined;
    }
    const controller = new AbortController();
    Promise.all([
      fetch(`/api/runs/${encodeURIComponent(selectedRun)}/activity`, { headers: { Accept: "application/json" }, signal: controller.signal }).then((response) => response.json() as Promise<ActivityResponse>),
      fetch(`/api/runs/${encodeURIComponent(selectedRun)}/architecture`, { headers: { Accept: "application/json" }, signal: controller.signal }).then((response) => response.json() as Promise<ArchitectureResponse>),
    ]).then(([nextActivity, nextArchitecture]) => { setActivity(nextActivity); setArchitecture(nextArchitecture); }).catch((cause) => { if (cause instanceof DOMException && cause.name === "AbortError") return; setActivity(null); setArchitecture(null); });
    return () => controller.abort();
  }, [selectedRun]);

  async function startIntervention() {
    if (!RUN_NAME_PATTERN.test(interventionRunName.trim())) {
      setInterventionStatus("Enter a unique run name using letters, numbers, dots, underscores, or hyphens.");
      return;
    }
    if (!selectedTargets.length) {
      setInterventionStatus("Select at least one layer × expert target.");
      return;
    }
    if (interventionOperation === "scale" && (!Number.isFinite(Number(scaleFactor)) || Number(scaleFactor) < 0)) {
      setInterventionStatus("Scale must be a finite number greater than or equal to zero.");
      return;
    }
    try {
      setInterventionStatus("Starting an exact baseline-derived run…");
      const response = await postJson<{ job_id: string }>("/api/interventions/start", {
        run_name: interventionRunName.trim(),
        baseline_run_key: selectedRun,
        operation: interventionOperation,
        targets: [...selectedTargets].sort(),
        factor: interventionOperation === "scale" ? Number(scaleFactor) : null,
      });
      setInterventionJobId(response.job_id);
    } catch (cause) {
      setInterventionStatus(cause instanceof Error ? cause.message : "This baseline cannot run that intervention.");
    }
  }

  async function createStudy() {
    if (studyRuns.length < 2) {
      setStudyStatus("Select at least two repeated intervention runs.");
      return;
    }
    try {
      setStudyStatus("Checking repeated runs and negative controls…");
      const response = await postJson<{ study_id: string; study: InterventionStudy }>(
        "/api/intervention-studies",
        {
          intervention_run_keys: [...studyRuns].sort(),
          control_run_keys: [...controlRuns].sort(),
        },
      );
      setStudy(response.study);
      setStudyStatus("Study published with immutable run evidence.");
    } catch {
      setStudy(null);
      setStudyStatus("These runs do not share one recipe and evaluator, or evidence is incomplete.");
    }
  }

  const selectedEntry = entries.find((entry) => entry.run_key === selectedRun);
  const selectedControlRecipe = studyCandidates.find((candidate) => (
    controlRuns.includes(candidate.intervention_run_key)
  ))?.recipe_fingerprint;
  const scaleCapability = interventionTargets?.capability?.operation_capabilities?.find(
    (operation) => operation.operation === "scale_contribution",
  );
  const supportsScaling = scaleCapability?.status === "available";
  useEffect(() => {
    if (!supportsScaling && interventionOperation === "scale") {
      setInterventionOperation("ablate");
    }
  }, [interventionOperation, supportsScaling]);
  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Runs / Inspect</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white">Trace inventory.</h1><p className="mt-3 max-w-[58ch] text-sm leading-6 text-muted">Routing heatmaps, validation status, and activation artifacts belong to a completed run. The UI never fills missing evidence with a visual guess.</p></div><div className="research-header-meta"><StatusDot tone={state === "unavailable" ? "warn" : "good"} /><span>{state === "loading" ? "Reading workspace…" : state === "unavailable" ? "Workspace offline" : `${entries.length} registered`}</span></div></header>
      {runGroups.length ? <section className="research-card"><div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Dataset groups</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Managed master runs</h2></div><span className="source-card-type">{runGroups.length} groups</span></div><div className="mt-5 grid gap-3">{runGroups.map((group) => <div key={group.group_name} className="rounded-xl border border-line bg-ink/50 p-4"><div className="flex flex-wrap items-center justify-between gap-3"><div><p className="font-mono text-xs text-white">{group.group_name}</p><p className="mt-1 text-[0.68rem] text-muted">{group.children.filter((child) => child.state === "completed").length}/{group.children.length} datasets completed</p></div><span className="runtime-pill"><StatusDot tone={group.state === "completed" ? "good" : group.state === "failed" || group.state === "partial" ? "warn" : "quiet"} />{group.state}</span></div><div className="mt-3 flex flex-wrap gap-2">{group.children.map((child) => <button key={child.slug} type="button" className="button-secondary" disabled={!child.run_key} onClick={() => child.run_key && setSelectedRun(child.run_key)}>{child.dataset_config ? `${child.dataset_id} · ${child.dataset_config}` : child.dataset_id} · {child.state}</button>)}</div></div>)}</div></section> : null}
      {state === "loading" ? <section className="empty-surface"><p className="text-sm text-muted">Reading run catalog…</p></section> : state === "unavailable" ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">Workspace unavailable.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Make the MoEAtlas server available before asking for stored traces.</p></section> : entries.length === 0 ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">No published traces.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Stage a run and let the executor publish its immutable shards. Heatmap cells appear only after routing events are validated.</p></section> : (
        <>
          <div className="run-selector-row"><label className="field-label" htmlFor="run-selector">Run<select id="run-selector" className="input-control mt-2" value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>{entries.map((entry) => <option key={entry.run_key} value={entry.run_key}>{runLabel(entry)} · {entry.state ?? "unknown"}</option>)}</select><span className="mt-2 block font-mono text-[0.58rem] text-muted">{selectedRun}</span></label><div className="runtime-pill"><StatusDot tone={selectedEntry?.state === "completed" ? "good" : "warn"} />{selectedEntry?.state ?? "unknown"}</div><div className="ml-auto flex flex-wrap gap-2"><a className="button-secondary" href={`/api/runs/${encodeURIComponent(selectedRun)}/export?format=bundle`} download>Export bundle</a><a className="button-secondary" href={`/api/runs/${encodeURIComponent(selectedRun)}/export?format=csv`} download>CSV</a></div></div>
          <div className="metric-grid"><MetricCard label="Tokens" value={summary?.token_count == null ? "—" : String(summary.token_count)} detail="validated token rows" /><MetricCard label="Assignments" value={summary?.assignment_count == null ? "—" : String(summary.assignment_count)} detail="selected expert routes" /><MetricCard label="Layers" value={summary?.layer_count == null ? "—" : String(summary.layer_count)} detail="published routing layers" /><MetricCard label="Experts" value={summary?.expert_count == null ? "—" : String(summary.expert_count)} detail="routed expert universe" /></div>
          {interventionEvidence ? <section className="research-card">
            <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-signal">Paired causal evidence</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Baseline versus intervention</h2></div><span className="source-card-type">{interventionEvidence.all_targets_exercised ? "targets exercised" : "target not exercised"}</span></div>
            <div className="metric-grid mt-5"><MetricCard label="Outputs changed" value={interventionEvidence.changed_output_fraction == null ? "—" : `${(interventionEvidence.changed_output_fraction * 100).toFixed(1)}%`} detail={`${interventionEvidence.changed_output_rows}/${interventionEvidence.row_count} paired rows`} /><MetricCard label="Task score delta" value={interventionEvidence.task_score_delta == null ? "unavailable" : interventionEvidence.task_score_delta.toFixed(4)} detail={interventionEvidence.score_name ?? "add a reference column for scoring"} /><MetricCard label="Latency delta" value={interventionEvidence.latency_delta_percent == null ? "—" : `${interventionEvidence.latency_delta_percent.toFixed(1)}%`} detail="same run settings" /><MetricCard label="Restoration" value={interventionEvidence.restoration_status} detail="temporary hooks removed" /></div>
            {!interventionEvidence.all_targets_exercised ? <p className="mt-4 rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal">At least one selected expert was never called by these rows. This run does not establish a causal effect for that target.</p> : null}
            <div className="mt-4 flex flex-wrap gap-2"><a className="button-secondary" target="_blank" rel="noreferrer" href={`/api/compare/heatmap?baseline_run_key=${encodeURIComponent(interventionEvidence.baseline_run_key)}&comparison_run_key=${encodeURIComponent(interventionEvidence.intervention_run_key)}&metric=count_deltas`}>Open routing delta</a></div>
            <div className="mt-5 border-t border-line pt-5">
              <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="label-caps text-[0.57rem] text-muted">Replication study</p><p className="mt-2 max-w-[62ch] text-xs leading-5 text-muted">Select repeated runs of the same intervention. Add runs against unrelated experts as negative controls.</p></div>{study ? <span className="source-card-type">{study.claim_status}</span> : null}</div>
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <label className="field-label" htmlFor="study-runs">Repeated intervention runs<select id="study-runs" className="input-control mt-2 min-h-32" multiple value={studyRuns} onChange={(event) => setStudyRuns(Array.from(event.target.selectedOptions, (option) => option.value))}>{studyCandidates.filter((candidate) => candidate.recipe_fingerprint === interventionEvidence.recipe_fingerprint).map((candidate) => <option key={candidate.intervention_run_key} value={candidate.intervention_run_key}>{runLabel(entries.find((entry) => entry.run_key === candidate.intervention_run_key))}</option>)}</select></label>
                <label className="field-label" htmlFor="control-runs">Negative-control runs<select id="control-runs" className="input-control mt-2 min-h-32" multiple value={controlRuns} onChange={(event) => setControlRuns(Array.from(event.target.selectedOptions, (option) => option.value))}>{studyCandidates.filter((candidate) => candidate.recipe_fingerprint !== interventionEvidence.recipe_fingerprint && (!selectedControlRecipe || candidate.recipe_fingerprint === selectedControlRecipe) && !studyRuns.includes(candidate.intervention_run_key)).map((candidate) => <option key={candidate.intervention_run_key} value={candidate.intervention_run_key}>{runLabel(entries.find((entry) => entry.run_key === candidate.intervention_run_key))}</option>)}</select></label>
              </div>
              <button type="button" className="button-secondary mt-4" onClick={() => void createStudy()}>Build replicated study</button>
              {studyStatus ? <p className="mt-3 text-xs leading-5 text-muted" role="status">{studyStatus}</p> : null}
              {study ? <div className="metric-grid mt-4"><MetricCard label="Mean task effect" value={study.task_effect.mean == null ? "—" : study.task_effect.mean.toFixed(4)} detail={study.score_name ?? "task evaluator"} /><MetricCard label="95% interval" value={study.task_effect.confidence_interval_95 ? `${study.task_effect.confidence_interval_95[0].toFixed(3)} to ${study.task_effect.confidence_interval_95[1].toFixed(3)}` : "—"} detail="replication uncertainty" /><MetricCard label="Consistency" value={study.task_effect.direction_consistency == null ? "—" : `${(study.task_effect.direction_consistency * 100).toFixed(0)}%`} detail={`${study.replication_count} repeated runs`} /><MetricCard label="Controls" value={String(study.control_count)} detail={study.claim_reason} /></div> : null}
            </div>
          </section> : null}
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
            <section className="research-card"><div className="flex flex-wrap items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Routing load</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Layer × expert heatmap</h2><p className="mt-2 text-xs text-muted">Click cells to prepare an exact intervention target. The entire matrix remains in view.</p></div><div className="flex items-center gap-2"><label className="sr-only" htmlFor="heatmap-metric">Heatmap metric</label><select id="heatmap-metric" className="input-control input-control-compact" value={metric} onChange={(event) => setMetric(event.target.value as typeof metric)}><option value="assignment_counts">Counts</option><option value="assignment_shares">Shares</option><option value="load_ratios">Load ratios</option></select><span className="source-card-type">{summaryState === "loading" ? "loading" : summary?.status ?? "unavailable"}</span></div></div>{summary?.status === "available" && selectedRun ? <iframe ref={heatmapFrame} onLoad={bindHeatmapTargets} className="heatmap-frame-react mt-5" title={`Routing heatmap for ${selectedRun}`} src={`/api/runs/${encodeURIComponent(selectedRun)}/heatmap?metric=${metric}&view=compact`} /> : <div className="empty-heatmap mt-5"><GitBranch size={20} className="text-muted" /><p className="mt-3 text-sm text-white">No published matrix.</p><p className="mt-1 max-w-[32ch] text-center text-xs leading-5 text-muted">{summary?.reason ?? "A validated routing inspection is required before rendering heat."}</p></div>}
              {entries.length > 1 ? <div className="mt-5 border-t border-line pt-4">
                <div className="flex flex-wrap items-end gap-3">
                  <label className="field-label min-w-[15rem]" htmlFor="comparison-run">Compare against
                    <select id="comparison-run" className="input-control mt-2" value={comparisonRun} onChange={(event) => setComparisonRun(event.target.value)}><option value="">Select baseline</option>{entries.filter((entry) => entry.run_key !== selectedRun).map((entry) => <option key={entry.run_key} value={entry.run_key}>{runLabel(entry)}</option>)}</select>
                  </label>
                  <label className="field-label" htmlFor="comparison-metric">Delta
                    <select id="comparison-metric" className="input-control mt-2" value={comparisonMetric} onChange={(event) => setComparisonMetric(event.target.value as typeof comparisonMetric)}><option value="count_deltas">Counts</option><option value="share_deltas">Shares</option><option value="ratio_deltas">Ratios</option></select>
                  </label>
                  {comparisonRun ? <a className="button-secondary" target="_blank" rel="noreferrer" href={`/api/compare/heatmap?baseline_run_key=${encodeURIComponent(comparisonRun)}&comparison_run_key=${encodeURIComponent(selectedRun)}&metric=${comparisonMetric}`}>Open comparison</a> : null}
                </div>
                {routingSimilarity?.status === "available" && routingSimilarity.report ? <div className="metric-grid mt-4">
                  <MetricCard label="Distribution agreement" value={`${((1 - routingSimilarity.report.mean_js_divergence) * 100).toFixed(1)}%`} detail="1 − mean JS divergence" />
                  <MetricCard label="Rank correlation" value={routingSimilarity.report.mean_spearman == null ? "undefined" : routingSimilarity.report.mean_spearman.toFixed(3)} detail={routingSimilarity.report.undefined_spearman_layers ? `${routingSimilarity.report.undefined_spearman_layers} uniform layers omitted` : "mean Spearman correlation"} />
                  <MetricCard label={`Top-${routingSimilarity.report.top_n} overlap`} value={`${(routingSimilarity.report.mean_top_n_jaccard * 100).toFixed(1)}%`} detail="mean Jaccard overlap" />
                  <MetricCard label="Sample sizes" value={`${routingSimilarity.report.baseline_token_count} / ${routingSimilarity.report.comparison_token_count}`} detail="normalized before comparison" />
                </div> : routingSimilarity?.status === "unavailable" ? <p className="mt-4 text-xs leading-5 text-muted">{routingSimilarity.reason}</p> : null}
                {routingSimilarity?.status === "available" ? <p className="mt-3 text-[0.65rem] leading-5 text-muted">Similarity is association evidence. It does not identify a specialized expert or establish causality.</p> : null}
              </div> : null}
            </section>
            <aside className="space-y-5">
              <section className="research-card research-card-dark"><div className="flex items-center gap-2"><ShieldCheck size={16} className="text-cyan" /><p className="label-caps text-[0.59rem] text-muted">Evidence boundary</p></div><dl className="contract-list mt-5"><div><dt>Routing</dt><dd>{summary?.status === "available" ? "validated" : "unavailable"}</dd></div><div><dt>Causal</dt><dd>{study?.claim_status ?? (interventionEvidence ? "paired once" : "not tested")}</dd></div><div><dt>Adapter</dt><dd>{summary?.adapter_name ?? "—"}</dd></div><div><dt>Top-k</dt><dd>{summary?.routed_top_k == null ? "—" : summary.routed_top_k}</dd></div><div><dt>Digest</dt><dd>{summary?.inspection_digest ? summary.inspection_digest.slice(0, 18) + "…" : "—"}</dd></div></dl><p className="mt-3 text-[0.65rem] leading-5 text-muted">Evidence applies only to this pinned model revision, dataset revision, and run settings. It is not universal model certification.</p></section>
              <section className="research-card research-card-dark"><div className="flex items-center gap-2"><Pulse size={16} className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Expert activity</p></div>{activity?.status === "available" && activity.summary ? <><div className="mt-4 grid grid-cols-2 gap-3"><MetricCard label="Active cells" value={String(activity.summary.active_expert_cells ?? "—")} detail="experts with events"/><MetricCard label="Events" value={String(activity.summary.total_event_count ?? "—")} detail="validated expert events"/></div><p className="mt-4 text-xs leading-5 text-muted">Mean, variance, and peak contribution norms come from persisted expert events. Empty cells remain explicit zeros.</p>{activity.summary.candidate_ranking?.ranked_cell_count ? <div className="mt-4 border-t border-line pt-4"><p className="label-caps text-[0.56rem] text-muted">Intervention candidates</p><div className="mt-3 grid gap-2">{activity.summary.candidate_ranking.high_observed[0] ? <button type="button" className="button-secondary w-full justify-between" onClick={() => prepareCandidate(activity.summary!.candidate_ranking!.high_observed[0], "high")}>Test high-observed L{activity.summary.candidate_ranking.high_observed[0].layer_index} × E{activity.summary.candidate_ranking.high_observed[0].expert_index}<ArrowRight size={14}/></button> : null}{activity.summary.candidate_ranking.low_observed[0] ? <button type="button" className="button-secondary w-full justify-between" onClick={() => prepareCandidate(activity.summary!.candidate_ranking!.low_observed[0], "low")}>Test low-observed L{activity.summary.candidate_ranking.low_observed[0].layer_index} × E{activity.summary.candidate_ranking.low_observed[0].expert_index}<ArrowRight size={14}/></button> : null}</div><p className="mt-3 text-[0.65rem] leading-5 text-muted">Descriptive ranking only. The paired intervention establishes whether a candidate affects this task.</p></div> : null}</> : <p className="mt-4 text-xs leading-5 text-muted">{activity?.reason ?? "Loading activation evidence…"}</p>}</section>
              <section className="research-card research-card-dark"><div className="flex items-center gap-2"><GitBranch size={16} className="text-cyan" /><p className="label-caps text-[0.59rem] text-muted">Architecture</p></div>{architecture?.status === "available" && architecture.report ? <><dl className="contract-list mt-4"><div><dt>Families</dt><dd>{Array.isArray(architecture.report.architecture_families) ? architecture.report.architecture_families.join(", ") : "generic"}</dd></div><div><dt>Components</dt><dd>{Array.isArray(architecture.report.components) ? architecture.report.components.length : "—"}</dd></div><div><dt>Warnings</dt><dd>{Array.isArray(architecture.report.warnings) ? architecture.report.warnings.length : "0"}</dd></div></dl><p className="mt-3 text-xs leading-5 text-muted">{architecture.report.model_key ? `Manifest ${String(architecture.report.model_key)}` : "Persisted discovery report"}</p></> : <p className="mt-4 text-xs leading-5 text-muted">{architecture?.reason ?? "Architecture evidence is not published for this run yet."}</p>}</section>
              <section className="research-card research-card-dark">
                <div className="flex items-center gap-2"><Lightning size={16} className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Causal intervention</p></div>
                {interventionTargets?.capability?.operation_capabilities?.length ? <div className="mt-4 border-y border-line"><OperationCapabilityList operations={interventionTargets.capability.operation_capabilities} compact /></div> : null}
                {interventionEvidence ? <p className="mt-4 text-xs leading-5 text-muted">This is a derived intervention run. Select its recorded baseline to prepare another intervention.</p> : interventionTargets?.status === "available" ? <>
                  <label className="field-label mt-4 block" htmlFor="intervention-target-search">Find a target<input id="intervention-target-search" className="input-control mt-2" type="search" value={targetSearch} onChange={(event) => setTargetSearch(event.target.value)} placeholder="L9, E188, or 9 188" autoComplete="off" /></label>
                  <label className="field-label mt-4" htmlFor="intervention-targets">Layer × expert targets
                    <select id="intervention-targets" className="input-control mt-2 min-h-40" multiple value={selectedTargets} onChange={(event) => setSelectedTargets(Array.from(event.target.selectedOptions, (option) => option.value))}>
                      {visibleInterventionTargets.map((target) => <option key={target.label} value={target.label}>L{target.layer_index} × E{target.expert_index}</option>)}
                    </select>
                  </label>
                  <p className="mt-2 text-[0.65rem] leading-5 text-muted">Showing {visibleInterventionTargets.length} of {interventionTargets.targets.length} controllable experts. Packed experts are validated against the live backend when the run starts. Hold Ctrl or Command to select several.</p>
                  <label className="field-label mt-3 block" htmlFor="intervention-run-name">New run name<input id="intervention-run-name" className="input-control mt-2" value={interventionRunName} onChange={(event) => setInterventionRunName(event.target.value)} placeholder={`${runLabel(selectedEntry)}-ablation`} autoComplete="off" /></label>
                  <label className="field-label mt-3" htmlFor="intervention-operation">Operation
                    <select id="intervention-operation" className="input-control mt-2" value={interventionOperation} onChange={(event) => setInterventionOperation(event.target.value as typeof interventionOperation)}><option value="ablate">Disable contribution</option>{supportsScaling ? <option value="scale">Scale contribution</option> : null}</select>
                  </label>
                  {interventionOperation === "scale" ? <label className="field-label mt-3" htmlFor="scale-factor">Scale factor<input id="scale-factor" className="input-control mt-2" type="number" min="0" step="0.1" value={scaleFactor} onChange={(event) => setScaleFactor(event.target.value)} /></label> : null}
                  <button type="button" className="button-primary mt-4 w-full justify-between" disabled={interventionJob?.state === "queued" || interventionJob?.state === "running"} onClick={() => void startIntervention()}>{interventionJob?.state === "queued" || interventionJob?.state === "running" ? "Intervention running…" : "Run intervention"}<ArrowRight size={15}/></button>
                  {interventionJobId && (interventionJob?.state === "queued" || interventionJob?.state === "running") ? <button type="button" className="button-secondary mt-2 w-full justify-between" onClick={() => void postJson(`/api/jobs/${encodeURIComponent(interventionJobId)}/cancel`, {})}>Cancel intervention<XCircle size={15}/></button> : null}
                </> : <p className="mt-4 text-xs leading-5 text-muted">{interventionTargets?.reason ?? "Reading intervention targets…"}</p>}
                {interventionStatus ? <p className="mt-3 text-xs leading-5 text-muted" role="status">{interventionStatus}</p> : null}
                {interventionJob ? <p className="mt-2 font-mono text-[0.62rem] text-muted">{interventionJob.progress.message}</p> : null}
                <FailureDiagnostics job={interventionJob} />
              </section>
            </aside>
          </div>
        </>
      )}
    </div>
  );
}
