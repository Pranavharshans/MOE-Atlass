import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowRight,
  Check,
  CheckCircle,
  Cpu,
  Database,
  GitBranch,
  Lightning,
  Plus,
  Pulse,
  ShieldCheck,
  WifiHigh,
  XCircle,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";
import { RunsPage } from "./RunsPage";
import { postJson, useJob, type JobSnapshot } from "./jobs";
import {
  AppMark,
  FailureDiagnostics,
  GateRow,
  MetricCard,
  OperationCapabilityList,
  StatusDot,
  type InterventionCapability,
} from "./ui";

type NavigationItem = "analysis" | "discovery" | "run" | "runs";
type HubKind = "model" | "dataset";
type SearchState = "idle" | "loading" | "ready" | "unavailable";

type SourceDraft = {
  modelId: string;
  modelRevision: string;
  datasetId: string;
  datasetRevision: string;
  datasetConfig: string;
  datasetSplit: string;
  promptColumn: string;
  referenceColumn: string;
  device: string;
  dtype: "preserve" | "float32" | "float16" | "bfloat16";
  trustRemoteCode: boolean;
};

type RunDraft = {
  runName: string;
  mode: "generation" | "teacher_forced";
  evaluationMethod: "normalized_exact_match" | "token_f1" | "contains_reference" | "multiple_choice_accuracy" | "numeric_match";
  sampleCap: string;
  datasetSeed: string;
  batchSize: string;
  maxNewTokens: string;
  tokenTextPolicy: "redacted" | "stored";
  allowExport: boolean;
  measureCaptureOverhead: boolean;
};

type HubSuggestion = {
  identifier: string;
  author?: string | null;
  downloads?: number | null;
  likes?: number | null;
  pipeline_tag?: string | null;
  library_name?: string | null;
};

const DEFAULT_SOURCES: SourceDraft = {
  modelId: "",
  modelRevision: "main",
  datasetId: "",
  datasetRevision: "main",
  datasetConfig: "",
  datasetSplit: "train",
  promptColumn: "prompt",
  referenceColumn: "",
  device: "auto",
  dtype: "preserve",
  trustRemoteCode: false,
};

const DEFAULT_RUN: RunDraft = {
  runName: "",
  mode: "generation",
  evaluationMethod: "normalized_exact_match",
  sampleCap: "128",
  datasetSeed: "20260828",
  batchSize: "1",
  maxNewTokens: "128",
  tokenTextPolicy: "redacted",
  allowExport: true,
  measureCaptureOverhead: false,
};

const NAVIGATION: Array<{ id: NavigationItem; label: string; icon: Icon }> = [
  { id: "analysis", label: "Analysis", icon: Plus },
  { id: "discovery", label: "Discover", icon: GitBranch },
  { id: "run", label: "Run", icon: Lightning },
  { id: "runs", label: "Runs", icon: Pulse },
];

function readStored<T>(key: string, fallback: T): T {
  try {
    const value = window.localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}

function validateHubId(value: string, label: string): string | null {
  const normalized = value.trim();
  if (!normalized) return `${label} is required.`;
  if (/\s/.test(normalized) || normalized.startsWith("/") || normalized.endsWith("/")) {
    return "Use a Hugging Face namespace/repository ID without spaces.";
  }
  const parts = normalized.split("/");
  if (parts.length !== 2 || parts.some((part) => !part)) {
    return "Use the Hugging Face form namespace/repository.";
  }
  return null;
}

function validateRunName(value: string): string | null {
  if (!value.trim()) return "Run name is required.";
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$/.test(value.trim())) {
    return "Use 1–80 letters, numbers, dots, underscores, or hyphens; start with a letter or number.";
  }
  return null;
}

function formatCount(value: number | null | undefined): string | null {
  if (typeof value !== "number") return null;
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function SourceSearchField({
  kind,
  inputId,
  value,
  onChange,
  placeholder,
  error,
}: {
  kind: HubKind;
  inputId: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
  error: string | null;
}) {
  const [searchState, setSearchState] = useState<SearchState>("idle");
  const [suggestions, setSuggestions] = useState<HubSuggestion[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const query = value.trim();
    if (query.length < 2) {
      setSearchState("idle");
      setSuggestions([]);
      return undefined;
    }
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setSearchState("loading");
      try {
        const params = new URLSearchParams({ kind, q: query, limit: "6" });
        const response = await fetch(`/api/hub/search?${params.toString()}`, {
          headers: { Accept: "application/json" },
          signal: controller.signal,
        });
        if (!response.ok) throw new Error("search unavailable");
        const document = (await response.json()) as { entries?: HubSuggestion[] };
        setSuggestions(Array.isArray(document.entries) ? document.entries : []);
        setSearchState("ready");
        setOpen(true);
      } catch (cause) {
        if (cause instanceof DOMException && cause.name === "AbortError") return;
        setSuggestions([]);
        setSearchState("unavailable");
      }
    }, 260);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [kind, value]);

  const searchLabel =
    searchState === "loading"
      ? "Searching public Hub metadata…"
      : searchState === "unavailable"
        ? "Search unavailable — paste the exact ID."
        : searchState === "ready" && suggestions.length === 0
          ? "No public matches — the exact ID can still be used."
          : "Suggestions are public metadata only.";

  return (
    <div className="relative">
      <div className={`source-input-shell ${error ? "source-input-error" : ""}`}>
        <span className="source-input-prefix">hf.co/</span>
        <input
          id={inputId}
          value={value}
          onChange={(event) => {
            onChange(event.target.value);
            setOpen(true);
          }}
          onFocus={() => setOpen(suggestions.length > 0)}
          onBlur={() => window.setTimeout(() => setOpen(false), 120)}
          placeholder={placeholder}
          aria-label={`${kind} Hugging Face repository ID`}
          aria-invalid={Boolean(error)}
          aria-expanded={open && suggestions.length > 0}
          autoComplete="off"
          spellCheck={false}
        />
        {value.trim() && !error ? <CheckCircle size={17} weight="fill" className="source-input-valid" /> : null}
      </div>
      <p className={`mt-2 text-[0.68rem] leading-5 ${error ? "text-signal" : "text-muted"}`}>
        {error ?? searchLabel}
      </p>
      {open && suggestions.length > 0 ? (
        <ul className="source-suggestions" role="listbox" aria-label={`${kind} Hub suggestions`}>
          {suggestions.map((suggestion) => {
            const counts = formatCount(suggestion.downloads);
            return (
              <li key={suggestion.identifier} role="option" aria-selected="false">
                <button
                  type="button"
                  className="source-suggestion"
                  onMouseDown={(event) => event.preventDefault()}
                  onClick={() => {
                    onChange(suggestion.identifier);
                    setOpen(false);
                  }}
                >
                  <span className="min-w-0">
                    <span className="block truncate font-mono text-xs text-white">{suggestion.identifier}</span>
                    <span className="mt-1 block truncate text-[0.66rem] text-muted">
                      {[suggestion.pipeline_tag, suggestion.library_name, counts ? `${counts} downloads` : null].filter(Boolean).join(" · ") || "Public Hub repository"}
                    </span>
                  </span>
                  <ArrowRight size={14} className="shrink-0 text-cyan" />
                </button>
              </li>
            );
          })}
        </ul>
      ) : null}
    </div>
  );
}

function SourceCard({
  kind,
  value,
  onChange,
  revision,
  onRevisionChange,
  config,
  onConfigChange,
  split,
  onSplitChange,
  promptColumn,
  onPromptColumnChange,
  referenceColumn,
  onReferenceColumnChange,
  error,
}: {
  kind: HubKind;
  value: string;
  onChange: (value: string) => void;
  revision?: string;
  onRevisionChange?: (value: string) => void;
  config?: string;
  onConfigChange?: (value: string) => void;
  split?: string;
  onSplitChange?: (value: string) => void;
  promptColumn?: string;
  onPromptColumnChange?: (value: string) => void;
  referenceColumn?: string;
  onReferenceColumnChange?: (value: string) => void;
  error: string | null;
}) {
  const isModel = kind === "model";
  return (
    <section className="research-card">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div className="source-card-icon"><span>{isModel ? "M" : "D"}</span></div>
          <div>
            <p className="label-caps text-[0.59rem] text-signal">{isModel ? "01 / Model" : "02 / Dataset"}</p>
            <h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">
              {isModel ? "Model source" : "Dataset source"}
            </h2>
          </div>
        </div>
        <span className="source-card-type">Hugging Face</span>
      </div>
      <div className="mt-6">
        <label className="field-label" htmlFor={`${kind}-id`}>Repository ID</label>
        <SourceSearchField inputId={`${kind}-id`} kind={kind} value={value} onChange={onChange} placeholder={isModel ? "inclusionAI/Ling-3.0-tiny" : "HuggingFaceH4/ultrachat_200k"} error={error} />
      </div>
      <div className="mt-5 grid gap-4 sm:grid-cols-2">
        <label className="field-label" htmlFor={`${kind}-revision`}>
          Revision <span className="field-optional">optional</span>
          <input id={`${kind}-revision`} className="input-control mt-2" value={revision ?? ""} onChange={(event) => onRevisionChange?.(event.target.value)} placeholder="main" spellCheck={false} />
        </label>
        {isModel ? (
          <div className="field-note">
            <span className="label-caps text-[0.56rem] text-muted">Resolution</span>
            <span className="mt-2 flex items-center gap-2 text-xs text-muted"><ShieldCheck size={15} className="text-cyan" />Revision is pinned during discovery.</span>
          </div>
        ) : (
          <label className="field-label" htmlFor="dataset-config">
            Config <span className="field-optional">optional</span>
            <input id="dataset-config" className="input-control mt-2" value={config ?? ""} onChange={(event) => onConfigChange?.(event.target.value)} placeholder="default" spellCheck={false} />
          </label>
        )}
      </div>
      {!isModel ? (
        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <label className="field-label" htmlFor="dataset-split">
            Split
            <input id="dataset-split" className="input-control mt-2" value={split ?? ""} onChange={(event) => onSplitChange?.(event.target.value)} placeholder="train" spellCheck={false} />
          </label>
          <div className="field-note">
            <span className="label-caps text-[0.56rem] text-muted">Read policy</span>
            <span className="mt-2 flex items-center gap-2 text-xs text-muted"><Database size={15} className="text-cyan" />Bounded rows and explicit provenance.</span>
          </div>
          <label className="field-label" htmlFor="dataset-prompt-column">
            Prompt column
            <input id="dataset-prompt-column" className="input-control mt-2" value={promptColumn ?? "prompt"} onChange={(event) => onPromptColumnChange?.(event.target.value)} placeholder="prompt or text" spellCheck={false} />
          </label>
          <label className="field-label" htmlFor="dataset-reference-column">
            Reference column <span className="field-optional">optional</span>
            <input id="dataset-reference-column" className="input-control mt-2" value={referenceColumn ?? ""} onChange={(event) => onReferenceColumnChange?.(event.target.value)} placeholder="answer or label" spellCheck={false} />
          </label>
        </div>
      ) : null}
    </section>
  );
}

function AnalysisPage({ onNavigate }: { onNavigate: (item: NavigationItem) => void }) {
  const [sources, setSources] = useState<SourceDraft>(() => readStored("moeatlas-analysis-sources", DEFAULT_SOURCES));
  const [queued, setQueued] = useState(false);
  const [starting, setStarting] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const modelError = useMemo(() => validateHubId(sources.modelId, "Model ID"), [sources.modelId]);
  const datasetError = useMemo(() => validateHubId(sources.datasetId, "Dataset ID"), [sources.datasetId]);
  const ready = !modelError && !datasetError;

  function update(field: keyof SourceDraft, value: string | boolean) {
    setSources((current) => ({ ...current, [field]: value } as SourceDraft));
    setQueued(false);
  }

  async function queueDiscovery() {
    if (!ready) return;
    setStarting(true);
    setRequestError(null);
    window.localStorage.setItem("moeatlas-analysis-sources", JSON.stringify(sources));
    try {
      const created = await postJson<{ job_id: string }>("/api/discovery", {
        model_id: sources.modelId.trim(),
        model_revision: sources.modelRevision.trim() || "main",
        device: sources.device,
        dtype: sources.dtype,
        trust_remote_code: sources.trustRemoteCode,
        allow_downloads: true,
      });
      window.localStorage.setItem("moeatlas-discovery-job", created.job_id);
      setQueued(true);
      onNavigate("discovery");
    } catch {
      setRequestError("The discovery job could not be started. Check that the local server is running.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className="space-y-6">
      <header className="research-header">
        <div>
          <p className="label-caps text-[0.61rem] text-signal">Analysis / New</p>
          <h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white sm:text-5xl">Define a routing capture.</h1>
          <p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted">Bind one model revision to one dataset. Discovery will inspect the resolved runtime before any token-level capture starts.</p>
        </div>
        <div className="research-header-meta"><StatusDot tone={ready ? "good" : "quiet"} /><span>{ready ? "Ready for discovery" : "Awaiting sources"}</span></div>
      </header>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
        <main className="space-y-5">
          <div className="grid gap-5 lg:grid-cols-2">
            <SourceCard kind="model" value={sources.modelId} onChange={(value) => update("modelId", value)} revision={sources.modelRevision} onRevisionChange={(value) => update("modelRevision", value)} error={modelError} />
            <SourceCard kind="dataset" value={sources.datasetId} onChange={(value) => update("datasetId", value)} revision={sources.datasetRevision} onRevisionChange={(value) => update("datasetRevision", value)} config={sources.datasetConfig} onConfigChange={(value) => update("datasetConfig", value)} split={sources.datasetSplit} onSplitChange={(value) => update("datasetSplit", value)} promptColumn={sources.promptColumn} onPromptColumnChange={(value) => update("promptColumn", value)} referenceColumn={sources.referenceColumn} onReferenceColumnChange={(value) => update("referenceColumn", value)} error={datasetError} />
          </div>
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Runtime policy</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">How the model is loaded</h2></div><Cpu size={19} className="text-cyan" /></div>
            <div className="mt-5 grid gap-4 sm:grid-cols-3">
              <label className="field-label" htmlFor="runtime-device">Device<select id="runtime-device" className="input-control mt-2" value={sources.device} onChange={(event) => update("device", event.target.value)}><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="cpu">CPU</option><option value="mps">MPS</option></select></label>
              <label className="field-label" htmlFor="runtime-dtype">Dtype<select id="runtime-dtype" className="input-control mt-2" value={sources.dtype} onChange={(event) => update("dtype", event.target.value as SourceDraft["dtype"])}><option value="preserve">Preserve</option><option value="bfloat16">bfloat16</option><option value="float16">float16</option><option value="float32">float32</option></select></label>
              <label className="toggle-row self-end"><input className="check-control" type="checkbox" checked={sources.trustRemoteCode} onChange={(event) => update("trustRemoteCode", event.target.checked)} /><span><span className="block text-xs font-medium text-white">Trust remote code</span><span className="mt-1 block text-[0.68rem] leading-5 text-muted">Required by some custom architectures; opt in deliberately.</span></span></label>
            </div>
          </section>
        </main>

        <aside className="space-y-5">
          <section className="research-card research-card-dark">
            <div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Run contract</p><GitBranch size={16} className="text-cyan" /></div>
            <dl className="contract-list mt-5">
              <div><dt>Model</dt><dd>{sources.modelId.trim() || "—"}</dd></div>
              <div><dt>Revision</dt><dd>{sources.modelRevision.trim() || "main"}</dd></div>
              <div><dt>Dataset</dt><dd>{sources.datasetId.trim() || "—"}</dd></div>
              <div><dt>Data rev.</dt><dd>{sources.datasetRevision.trim() || "main"}</dd></div>
              <div><dt>Split</dt><dd>{sources.datasetSplit.trim() || "train"}</dd></div>
              <div><dt>Execution</dt><dd>bound server</dd></div>
            </dl>
          </section>
          <section className="research-card research-card-dark">
            <div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Evidence policy</p></div>
            <ul className="mt-4 space-y-3 text-xs leading-5 text-muted">
              <li className="flex gap-2"><Check size={14} className="mt-0.5 shrink-0 text-cyan" />Model revision and config are recorded.</li>
              <li className="flex gap-2"><Check size={14} className="mt-0.5 shrink-0 text-cyan" />Dataset split and provenance stay bound to the run.</li>
              <li className="flex gap-2"><Check size={14} className="mt-0.5 shrink-0 text-cyan" />Search is optional; exact IDs never depend on it.</li>
            </ul>
          </section>
          <button type="button" className="button-primary w-full justify-between" disabled={!ready || starting} onClick={() => void queueDiscovery()}>
            {starting ? "Starting discovery…" : queued ? "Discovery running" : "Start discovery"}<ArrowRight size={16} weight="bold" />
          </button>
          {requestError ? <p className="rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal" role="alert">{requestError}</p> : null}
          {queued ? <div className="space-y-2" role="status"><p className="rounded-xl border border-cyan/25 bg-cyan/[0.06] p-3 text-xs leading-5 text-cyan">A real discovery job is running on the bound server. Open the evidence surface for progress.</p><button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("discovery")}>Open discovery <ArrowRight size={16} /></button></div> : null}
        </aside>
      </div>
    </div>
  );
}

function DiscoveryPage({ onNavigate }: { onNavigate: (item: NavigationItem) => void }) {
  const [sources] = useState<SourceDraft>(() => readStored("moeatlas-analysis-sources", DEFAULT_SOURCES));
  const [jobId, setJobId] = useState<string | null>(() => readStored<string | null>("moeatlas-discovery-job", null));
  const [starting, setStarting] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const job = useJob(jobId);
  const result = (job?.result ?? {}) as Record<string, unknown>;
  const report = (result.report ?? null) as Record<string, unknown> | null;
  const facts = (report?.facts ?? {}) as Record<string, unknown>;
  const captureSupport = (result.capture_support ?? {}) as Record<string, unknown>;
  const interventionCapability = (result.intervention_capability ?? null) as InterventionCapability | null;
  const operationCapabilities = Array.isArray(interventionCapability?.operation_capabilities) ? interventionCapability.operation_capabilities : [];
  const hasContract = !validateHubId(sources.modelId, "Model ID") && !validateHubId(sources.datasetId, "Dataset ID");

  if (!hasContract) {
    return (
      <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><GitBranch size={21} /></div><h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.04em] text-white">No discovery contract.</h1><p className="mt-3 max-w-[43ch] text-center text-sm leading-6 text-muted">Bind a model and dataset on the analysis surface before asking the runtime to inspect anything.</p><button type="button" className="button-primary mt-6" onClick={() => onNavigate("analysis")}>Back to analysis <ArrowRight size={16} weight="bold" /></button></section>
    );
  }

  async function startDiscovery() {
    if (!hasContract || starting) return;
    setStarting(true);
    setRequestError(null);
    try {
      const created = await postJson<{ job_id: string }>("/api/discovery", {
        model_id: sources.modelId.trim(),
        model_revision: sources.modelRevision.trim() || "main",
        device: sources.device,
        dtype: sources.dtype,
        trust_remote_code: sources.trustRemoteCode,
        allow_downloads: true,
      });
      window.localStorage.setItem("moeatlas-discovery-job", created.job_id);
      setJobId(created.job_id);
    } catch {
      setRequestError("Discovery could not start. Make sure the server is running in this runtime.");
    } finally {
      setStarting(false);
    }
  }

  const running = job?.state === "queued" || job?.state === "running";
  const available = job?.state === "completed" && result.status === "available";

  return (
    <div className="space-y-6">
      <header className="research-header">
        <div><p className="label-caps text-[0.61rem] text-signal">Discovery / Preflight</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white sm:text-5xl">Read the runtime before the run.</h1><p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted">This envelope shows what the bound server inspected. A staged contract is not model evidence.</p></div>
        <div className="research-header-meta"><StatusDot tone={available ? "good" : job?.state === "failed" ? "warn" : "quiet"} /><span>{available ? "Architecture discovered" : running ? `${job?.progress.stage ?? "running"} · ${job?.progress.completed ?? 0}/${job?.progress.total ?? "?"}` : job?.state === "failed" ? "Discovery failed" : "Ready to inspect"}</span></div>
      </header>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
        <main className="space-y-5">
          <section className="research-card">
            <div className="flex items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Bound identity</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Provenance envelope</h2></div><ShieldCheck size={19} className="text-cyan" /></div>
            <dl className="contract-grid mt-6">
              <div><dt>Model ID</dt><dd>{sources.modelId}</dd></div>
              <div><dt>Model revision</dt><dd>{sources.modelRevision.trim() || "main"}</dd></div>
              <div><dt>Dataset ID</dt><dd>{sources.datasetId}</dd></div>
              <div><dt>Dataset revision</dt><dd>{sources.datasetRevision?.trim() || "main"}</dd></div>
              <div><dt>Dataset config</dt><dd>{sources.datasetConfig.trim() || "default"}</dd></div>
              <div><dt>Dataset split</dt><dd>{sources.datasetSplit.trim() || "train"}</dd></div>
              <div><dt>Prompt column</dt><dd>{sources.promptColumn}</dd></div>
              <div><dt>Execution</dt><dd>bound server</dd></div>
            </dl>
          </section>
          <section className="research-card">
            <div className="flex items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Inspection gates</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">What the server must prove</h2></div><GitBranch size={19} className="text-cyan" /></div>
            <div className="mt-5 divide-y divide-line"><GateRow label="Model configuration" detail={available ? "READY" : running ? "RUNNING" : "PENDING"} tone={available ? "good" : running ? "quiet" : "quiet"} /><GateRow label="MoE topology" detail={available ? `${facts.expert_count ?? "?"} experts · ${facts.routed_top_k ?? "?"}-way` : running ? "SCANNING" : "PENDING"} tone={available ? "good" : "quiet"} /><GateRow label="Capture support" detail={available ? String(captureSupport.grade ?? "topology_only").replaceAll("_", " ").toUpperCase() : "DEFERRED"} tone={captureSupport.routing_capture === "candidate" ? "warn" : available ? "quiet" : "quiet"} /><GateRow label="Router payload" detail={captureSupport.routing_capture === "candidate" ? "UNPROVEN · RUN REQUIRED" : "UNAVAILABLE"} tone={captureSupport.routing_capture === "candidate" ? "warn" : "quiet"} /><GateRow label="Dataset schema" detail="validated at run" /><GateRow label="Immutable revision evidence" detail={typeof result.resolved_revision === "string" ? result.resolved_revision.slice(0, 12) + "…" : "PENDING"} tone={typeof result.resolved_revision === "string" ? "good" : "quiet"} /></div>
          </section>
          {available && interventionCapability ? <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Capability matrix</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">What this runtime can do now</h2><p className="mt-2 text-xs leading-5 text-muted">Each verdict names one exact operation. Detecting a backend does not automatically make every intervention safe.</p></div><Lightning size={19} className="shrink-0 text-signal" /></div>
            <dl className="contract-grid mt-5">
              <div><dt>Expert storage</dt><dd>{interventionCapability.weight_layout.replaceAll("_", " ")}</dd></div>
              <div><dt>Expert backend</dt><dd>{interventionCapability.execution_backend ?? "unresolved"}</dd></div>
            </dl>
            <div className="mt-4"><OperationCapabilityList operations={operationCapabilities} /></div>
          </section> : null}
          {available ? <section className="research-card"><div className="flex items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Discovered architecture</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Evidence, not a family allowlist.</h2></div><ShieldCheck size={19} className="text-cyan" /></div><div className="mt-5 grid gap-3 sm:grid-cols-3"><MetricCard label="Experts" value={String(facts.expert_count ?? "—")} detail={String(facts.expert_count_source ?? "scanner")}/><MetricCard label="Top-k" value={String(facts.routed_top_k ?? "—")} detail={String(facts.routed_top_k_source ?? "scanner")}/><MetricCard label="Hook targets" value={String(captureSupport.router_target_count ?? "—")} detail="static router candidates"/></div><p className="mt-4 text-xs leading-5 text-muted">The scan identifies topology and hook candidates generically. Candidate means structurally addressable—not captured. A real forward must still validate the router payload, complete top-k assignments, and any expert activity.</p></section> : null}
        </main>
        <aside className="space-y-5">
          <section className="research-card research-card-dark"><div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Resource envelope</p><Database size={16} className="text-cyan" /></div><dl className="contract-list mt-5"><div><dt>Weights</dt><dd>not measured</dd></div><div><dt>Accelerator</dt><dd>server runtime</dd></div><div><dt>Rows</dt><dd>bounded later</dd></div><div><dt>Capture</dt><dd>off</dd></div></dl></section>
          <section className="research-card research-card-dark"><div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Evidence rule</p></div><p className="mt-4 text-xs leading-5 text-muted">Discovery can report topology and router seams. It must not label routing as captured until a real forward produces validated events.</p></section>
          <button type="button" className="button-primary w-full justify-between" disabled={running || starting} onClick={() => void startDiscovery()}>{starting ? "Starting…" : running ? "Discovery running…" : available ? "Re-run discovery" : "Run live discovery"}<ArrowRight size={16} weight="bold" /></button>
          {jobId && running ? <button type="button" className="button-secondary w-full justify-between" onClick={() => void postJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {})}>Cancel job <XCircle size={16} /></button> : null}
          {available ? <button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("run")}>Configure capture <ArrowRight size={16} /></button> : null}
          {job?.progress.message ? <p className="rounded-xl border border-line bg-white/[0.03] p-3 text-xs leading-5 text-muted" role="status">{job.progress.message}</p> : null}
          {requestError || job?.error ? <p className="rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal" role="alert">{requestError ?? job?.error}</p> : null}
          <FailureDiagnostics job={job} />
        </aside>
      </div>
    </div>
  );
}

function validatePositiveSetting(value: string, label: string, maximum: number): string | null {
  if (!value.trim()) return `${label} is required.`;
  if (!/^\d+$/.test(value.trim()) || Number(value) < 1 || Number(value) > maximum) return `${label} must be an integer between 1 and ${maximum}.`;
  return null;
}

function validateDatasetSeed(value: string): string | null {
  if (!/^\d+$/.test(value.trim()) || Number(value) > 2_147_483_647) {
    return "Dataset seed must be an integer between 0 and 2147483647.";
  }
  return null;
}

function RunConfigPage({ onNavigate }: { onNavigate: (item: NavigationItem) => void }) {
  const [sources] = useState<SourceDraft>(() => ({ ...DEFAULT_SOURCES, ...readStored<Partial<SourceDraft>>("moeatlas-analysis-sources", {}) }));
  const [run, setRun] = useState<RunDraft>(() => ({ ...DEFAULT_RUN, ...readStored<Partial<RunDraft>>("moeatlas-run", {}) }));
  const [jobId, setJobId] = useState<string | null>(() => readStored<string | null>("moeatlas-run-job", null));
  const [starting, setStarting] = useState(false);
  const [requestError, setRequestError] = useState<string | null>(null);
  const job = useJob(jobId);
  const modelError = validateHubId(sources.modelId, "Model ID");
  const datasetError = validateHubId(sources.datasetId, "Dataset ID");
  const sampleError = validatePositiveSetting(run.sampleCap, "Sample cap", 1_000_000);
  const seedError = validateDatasetSeed(run.datasetSeed);
  const batchError = validatePositiveSetting(run.batchSize, "Batch size", 4096);
  const tokenError = validatePositiveSetting(run.maxNewTokens, "Max new tokens", 1_000_000);
  const runNameError = validateRunName(run.runName);
  const ready = !modelError && !datasetError && !sampleError && !seedError && !batchError && !tokenError && !runNameError;
  const running = job?.state === "queued" || job?.state === "running";
  const overhead = (job?.result?.capture_overhead ?? null) as Record<string, unknown> | null;
  const overheadRunning = running && job?.progress.stage === "overhead";

  function update(field: keyof RunDraft, value: string | boolean) {
    setRun((current) => ({ ...current, [field]: value } as RunDraft));
  }

  async function startRun(resumeJobId: string | null = null) {
    if (!ready || starting) return;
    setStarting(true);
    setRequestError(null);
    window.localStorage.setItem("moeatlas-run", JSON.stringify(run));
    try {
      const created = await postJson<{ job_id: string }>("/api/runs/start", {
        run_name: run.runName.trim(),
        model_id: sources.modelId.trim(),
        model_revision: sources.modelRevision.trim() || "main",
        dataset_id: sources.datasetId.trim(),
        dataset_revision: sources.datasetRevision.trim() || "main",
        dataset_config: sources.datasetConfig.trim() || null,
        dataset_split: sources.datasetSplit.trim() || "train",
        prompt_column: sources.promptColumn.trim() || "prompt",
        reference_column: sources.referenceColumn.trim() || null,
        evaluation_method: run.evaluationMethod,
        sample_cap: Number(run.sampleCap),
        dataset_seed: Number(run.datasetSeed),
        batch_size: Number(run.batchSize),
        max_new_tokens: Number(run.maxNewTokens),
        token_text_policy: run.tokenTextPolicy,
        allow_export: run.allowExport,
        mode: run.mode,
        device: sources.device,
        dtype: sources.dtype,
        trust_remote_code: sources.trustRemoteCode,
        allow_downloads: true,
        capture_expert_activity: true,
        measure_capture_overhead: run.measureCaptureOverhead,
        resume_job_id: resumeJobId,
      });
      window.localStorage.setItem("moeatlas-run-job", created.job_id);
      setJobId(created.job_id);
    } catch (cause) {
      setRequestError(cause instanceof Error ? cause.message : "The capture job could not be started.");
    } finally {
      setStarting(false);
    }
  }

  async function skipOverhead() {
    if (!jobId || !overheadRunning) return;
    try {
      await postJson(`/api/jobs/${encodeURIComponent(jobId)}/skip-overhead`, {});
    } catch {
      setRequestError("The optional overhead pass could not be skipped.");
    }
  }

  const canResume = job?.state === "cancelled" && typeof job.result?.checkpoint_path === "string";

  if (modelError || datasetError) {
    return (
      <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Lightning size={21} /></div><h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.04em] text-white">No run inputs.</h1><p className="mt-3 max-w-[43ch] text-center text-sm leading-6 text-muted">Return to analysis and bind a model and dataset before configuring execution.</p><button type="button" className="button-primary mt-6" onClick={() => onNavigate("analysis")}>Back to analysis <ArrowRight size={16} weight="bold" /></button></section>
    );
  }

  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Run / Capture</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white sm:text-5xl">Set the capture budget.</h1><p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted">This starts a real server-side model run over the selected dataset. Progress, checkpoints, routing, and activation evidence remain tied to the resulting run key.</p></div><div className="research-header-meta"><StatusDot tone={job?.state === "failed" ? "warn" : ready ? "good" : "quiet"} /><span>{job?.state === "running" ? job.progress.stage === "overhead" ? `Native baseline · ${job.progress.completed}/${job.progress.total ?? "?"}` : `${job.progress.stage} · ${job.progress.completed}/${job.progress.total ?? "?"}` : job?.state === "completed" ? "Capture complete" : job?.state === "cancelled" ? "Capture cancelled" : ready ? "Ready to run" : "Invalid budget"}</span></div></header>
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
        <main className="space-y-5">
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Run identity</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Give this run a readable name</h2></div><GitBranch size={19} className="text-cyan" /></div>
            <label className="field-label mt-6 block" htmlFor="run-name">Run name<input id="run-name" className={`input-control mt-2 ${runNameError ? "input-control-error" : ""}`} value={run.runName} onChange={(event) => update("runName", event.target.value)} placeholder="v4-cybersecurity-baseline" autoComplete="off" /></label>
            <p className={`field-hint mt-2 ${runNameError ? "field-hint-error" : ""}`}>{runNameError ?? "This appears in Runs and becomes the folder name under workspace/runs/."}</p>
          </section>
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Execution budget</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Rows and generation</h2></div><Lightning size={19} className="text-signal" /></div>
            <div className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <label className="field-label" htmlFor="sample-cap">Sample cap<input id="sample-cap" className={`input-control mt-2 ${sampleError ? "input-control-error" : ""}`} value={run.sampleCap} onChange={(event) => update("sampleCap", event.target.value)} inputMode="numeric" /></label>
              <label className="field-label" htmlFor="dataset-seed">Dataset seed<input id="dataset-seed" className={`input-control mt-2 ${seedError ? "input-control-error" : ""}`} value={run.datasetSeed} onChange={(event) => update("datasetSeed", event.target.value)} inputMode="numeric" /></label>
              <label className="field-label" htmlFor="batch-size">Batch size<input id="batch-size" className={`input-control mt-2 ${batchError ? "input-control-error" : ""}`} value={run.batchSize} onChange={(event) => update("batchSize", event.target.value)} inputMode="numeric" /></label>
              <label className="field-label" htmlFor="max-new-tokens">Max new tokens<input id="max-new-tokens" className={`input-control mt-2 ${tokenError ? "input-control-error" : ""}`} value={run.maxNewTokens} onChange={(event) => update("maxNewTokens", event.target.value)} inputMode="numeric" /></label>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4"><p className={`field-hint ${sampleError ? "field-hint-error" : ""}`}>{sampleError ?? "Rows are bounded before execution."}</p><p className={`field-hint ${seedError ? "field-hint-error" : ""}`}>{seedError ?? "The same revision and seed select the same rows."}</p><p className={`field-hint ${batchError ? "field-hint-error" : ""}`}>{batchError ?? "Schedule remains deterministic."}</p><p className={`field-hint ${tokenError ? "field-hint-error" : ""}`}>{tokenError ?? "Generation budget is recorded."}</p></div>
            <div className="mt-6"><span className="field-label">Run mode</span><div className="mt-2 inline-flex rounded-xl border border-line bg-ink p-1" role="group" aria-label="Run mode"><button type="button" className={`runner-tab ${run.mode === "generation" ? "runner-tab-active" : ""}`} aria-pressed={run.mode === "generation"} onClick={() => update("mode", "generation")}>Generation</button><button type="button" className={`runner-tab ${run.mode === "teacher_forced" ? "runner-tab-active" : ""}`} aria-pressed={run.mode === "teacher_forced"} onClick={() => update("mode", "teacher_forced")}>Teacher-forced</button></div></div>
            <label className="field-label mt-6 block" htmlFor="evaluation-method">Task evaluator<select id="evaluation-method" className="input-control mt-2" value={run.evaluationMethod} onChange={(event) => update("evaluationMethod", event.target.value)}><option value="normalized_exact_match">Exact text match</option><option value="token_f1">Token F1</option><option value="contains_reference">Contains reference</option><option value="multiple_choice_accuracy">Multiple-choice accuracy</option><option value="numeric_match">Numeric match</option></select></label>
            <p className="field-hint mt-2">Scoring is available when the analysis contract maps a reference column.</p>
          </section>
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Evidence and privacy</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">What may be retained</h2></div><ShieldCheck size={19} className="text-cyan" /></div>
            <div className="mt-6"><span className="field-label">Token text</span><div className="mt-2 inline-flex rounded-xl border border-line bg-ink p-1" role="group" aria-label="Token text policy"><button type="button" className={`runner-tab ${run.tokenTextPolicy === "redacted" ? "runner-tab-active" : ""}`} aria-pressed={run.tokenTextPolicy === "redacted"} onClick={() => update("tokenTextPolicy", "redacted")}>Redacted (default)</button><button type="button" className={`runner-tab ${run.tokenTextPolicy === "stored" ? "runner-tab-active" : ""}`} aria-pressed={run.tokenTextPolicy === "stored"} onClick={() => update("tokenTextPolicy", "stored")}>Store token text</button></div></div>
            <div className="mt-5"><label className="toggle-row"><input className="check-control" type="checkbox" checked={run.allowExport} onChange={(event) => update("allowExport", event.target.checked)} /><span><span className="block text-xs font-medium text-white">Allow artifact export</span><span className="mt-1 block text-[0.68rem] leading-5 text-muted">Persist this decision with the run; disabled exports cannot be re-enabled from the UI.</span></span></label></div>
          </section>
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Optional benchmark lane</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Measure capture overhead</h2></div><Pulse size={19} className="text-cyan" /></div>
            <label className="toggle-row mt-5"><input className="check-control" type="checkbox" checked={run.measureCaptureOverhead} disabled={running || starting} onChange={(event) => update("measureCaptureOverhead", event.target.checked)} /><span><span className="block text-xs font-medium text-white">Run native baseline first</span><span className="mt-1 block text-[0.68rem] leading-5 text-muted">Adds a forward-only pass with routing capture disabled. It is off by default and can be skipped while running.</span></span></label>
          </section>
          <section className="research-card"><div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Worker boundary</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Execution handoff</h2></div><WifiHigh size={19} className="text-cyan" /></div><div className="mt-5 flex flex-wrap items-center gap-3"><span className="runtime-pill"><StatusDot />Bound server</span><span className="text-xs text-muted">The server resolves its own accelerator and model cache.</span></div><p className="mt-4 text-xs leading-5 text-muted">Use the same UI on a local machine or inside a provider VM. Only the server process needs access to the model and dataset; no SSH or path selector is involved.</p></section>
        </main>
        <aside className="space-y-5"><section className="research-card research-card-dark"><div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Run contract</p><GitBranch size={16} className="text-cyan" /></div><dl className="contract-list mt-5"><div><dt>Model</dt><dd>{sources.modelId}</dd></div><div><dt>Dataset</dt><dd>{sources.datasetId}</dd></div><div><dt>Prompt</dt><dd>{sources.promptColumn}</dd></div><div><dt>Rows</dt><dd>{run.sampleCap}</dd></div><div><dt>Seed</dt><dd>{run.datasetSeed}</dd></div><div><dt>Batch</dt><dd>{run.batchSize}</dd></div><div><dt>Mode</dt><dd>{run.mode.replace("_", " ")}</dd></div><div><dt>Tokens</dt><dd>{run.tokenTextPolicy}</dd></div><div><dt>Overhead</dt><dd>{run.measureCaptureOverhead ? "optional native pass" : "off"}</dd></div></dl></section><section className="research-card research-card-dark"><div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Live state</p></div><p className="mt-4 text-xs leading-5 text-muted">{job?.progress.message ?? "The executor will resolve the model, stream bounded dataset rows, and publish immutable evidence."}</p>{job?.progress.total ? <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-cyan transition-all" style={{ width: `${Math.min(100, (job.progress.completed / job.progress.total) * 100)}%` }} /></div> : null}</section>{overhead ? <section className="research-card research-card-dark"><div className="flex items-center gap-2"><Pulse size={16} className="text-cyan" /><p className="label-caps text-[0.59rem] text-muted">Overhead result</p></div><dl className="contract-list mt-4"><div><dt>Status</dt><dd>{String(overhead.status ?? "unknown")}</dd></div><div><dt>Native forward</dt><dd>{typeof (overhead.native as Record<string, unknown> | null)?.mean_ms === "number" ? `${((overhead.native as Record<string, unknown>).mean_ms as number).toFixed(2)} ms/row` : "—"}</dd></div><div><dt>Captured forward</dt><dd>{typeof (overhead.captured as Record<string, unknown> | null)?.mean_ms === "number" ? `${((overhead.captured as Record<string, unknown>).mean_ms as number).toFixed(2)} ms/row` : "—"}</dd></div><div><dt>Delta</dt><dd>{typeof overhead.delta_percent === "number" ? `${(overhead.delta_percent as number).toFixed(2)}%` : "—"}</dd></div></dl></section> : null}<button type="button" className="button-primary w-full justify-between" disabled={!ready || starting || running} onClick={() => void startRun()}>{starting ? "Starting capture…" : running ? "Capture running…" : job?.state === "completed" ? "Capture complete" : "Start capture"}<ArrowRight size={16} weight="bold" /></button>{jobId && overheadRunning ? <button type="button" className="button-secondary w-full justify-between" onClick={() => void skipOverhead()}>Skip overhead measurement <XCircle size={16} /></button> : null}{jobId && running ? <button type="button" className="button-secondary w-full justify-between" onClick={() => void postJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {})}>{overheadRunning ? "Cancel study" : "Cancel capture"} <XCircle size={16} /></button> : null}{canResume && jobId ? <button type="button" className="button-secondary w-full justify-between" disabled={starting} onClick={() => void startRun(jobId)}>Resume from checkpoint <ArrowRight size={16} /></button> : null}{job?.state === "completed" ? <button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("runs")}>Inspect evidence <ArrowRight size={16} /></button> : null}{requestError || job?.error ? <p className="rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal" role="alert">{requestError ?? job?.error}</p> : null}<FailureDiagnostics job={job} /></aside>
      </div>
    </div>
  );
}

export function App() {
  const [active, setActive] = useState<NavigationItem>("analysis");
  const content = active === "analysis" ? <AnalysisPage onNavigate={setActive} /> : active === "discovery" ? <DiscoveryPage onNavigate={setActive} /> : active === "run" ? <RunConfigPage onNavigate={setActive} /> : <RunsPage />;

  return (
    <div className="app-shell min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1680px]">
        <aside className="hidden w-[14.5rem] shrink-0 flex-col border-r border-line px-5 py-6 lg:flex">
          <AppMark />
          <div className="mt-12"><p className="label-caps px-3 text-[0.58rem] text-muted">Observe</p><nav className="mt-3 space-y-1" aria-label="Primary navigation">{NAVIGATION.map((item) => { const Icon = item.icon; const selected = active === item.id; return <button type="button" key={item.id} className={`nav-item ${selected ? "nav-item-active" : ""}`} onClick={() => setActive(item.id)} aria-current={selected ? "page" : undefined}><Icon size={18} weight={selected ? "fill" : "regular"} />{item.label}</button>; })}</nav></div>
          <div className="mt-auto flex items-center justify-between px-1 text-[0.68rem] text-muted"><span>MoEAtlas</span><span>v0.1.0</span></div>
        </aside>
        <main className="min-w-0 flex-1 px-4 py-4 sm:px-7 sm:py-6 lg:px-10">
          <header className="mb-9 flex flex-wrap items-center justify-between gap-4"><div className="flex items-center gap-3 lg:hidden"><AppMark /></div><div className="hidden items-center gap-2 text-xs text-muted lg:flex"><span className="text-white">MoEAtlas</span><span className="text-muted/40">/</span><span>{NAVIGATION.find((item) => item.id === active)?.label}</span></div><div className="ml-auto flex items-center gap-3"><div className="runtime-pill"><StatusDot /><span>Bound server</span></div><span className="hidden font-mono text-[0.62rem] text-muted sm:inline">schema 1.0</span></div></header>
          <nav className="mb-7 flex gap-1 overflow-x-auto border-b border-line pb-2 lg:hidden" aria-label="Primary navigation">{NAVIGATION.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={`mobile-nav-item ${active === item.id ? "mobile-nav-item-active" : ""}`} onClick={() => setActive(item.id)}><Icon size={15} />{item.label}</button>; })}</nav>
          {content}
        </main>
      </div>
    </div>
  );
}
