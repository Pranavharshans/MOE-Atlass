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
  mode: "generation" | "teacher_forced";
  evaluationMethod: "normalized_exact_match" | "token_f1" | "contains_reference" | "multiple_choice_accuracy" | "numeric_match";
  sampleCap: string;
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

type JobProgress = {
  stage: string;
  completed: number;
  total?: number | null;
  message: string;
};

type JobSnapshot = {
  job_id: string;
  kind: string;
  state: "queued" | "running" | "completed" | "cancelled" | "failed";
  progress: JobProgress;
  result?: Record<string, unknown> | null;
  error?: string | null;
  diagnostics?: {
    endpoint: string;
    available: boolean;
    entry_count: number;
    truncated: boolean;
  } | null;
};

type JobDiagnosticEntry = {
  sequence: number;
  event: string;
  stage?: string | null;
  exception_type?: string | null;
  exception_message?: string | null;
  traceback?: string | null;
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
  mode: "generation",
  evaluationMethod: "normalized_exact_match",
  sampleCap: "128",
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

function useJob(jobId: string | null): JobSnapshot | null {
  const [snapshot, setSnapshot] = useState<JobSnapshot | null>(null);
  useEffect(() => {
    if (!jobId) {
      setSnapshot(null);
      return undefined;
    }
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}`, {
          headers: { Accept: "application/json" },
        });
        if (!response.ok) throw new Error("job unavailable");
        const next = (await response.json()) as JobSnapshot;
        if (!active) return;
        setSnapshot(next);
        if (next.state === "queued" || next.state === "running") {
          timer = window.setTimeout(poll, 700);
        }
      } catch {
        if (active) {
          setSnapshot((current) => current ?? {
            job_id: jobId,
            kind: "unknown",
            state: "failed",
            progress: { stage: "offline", completed: 0, message: "Job status unavailable" },
            error: "Job status unavailable",
          });
        }
      }
    };
    void poll();
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [jobId]);
  return snapshot;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error((await response.text()) || "request failed");
  return (await response.json()) as T;
}

function AppMark() {
  return (
    <div className="app-mark" aria-label="MoEAtlas">
      <div className="app-mark-code" aria-hidden="true">M</div>
      <div>
        <div className="font-display text-[1.05rem] font-semibold tracking-[-0.03em] text-white">
          MoE<span className="text-signal">Atlas</span>
        </div>
        <div className="label-caps mt-0.5 text-[0.56rem] text-muted">Routing observatory</div>
      </div>
    </div>
  );
}

function StatusDot({ tone = "good" }: { tone?: "good" | "quiet" | "warn" }) {
  return <span className={`status-dot status-dot-${tone}`} aria-hidden="true" />;
}

function FailureDiagnostics({ job }: { job: JobSnapshot | null }) {
  const [entry, setEntry] = useState<JobDiagnosticEntry | null>(null);
  const endpoint = job?.state === "failed" && job.diagnostics?.available
    ? job.diagnostics.endpoint
    : null;
  useEffect(() => {
    if (!endpoint) {
      setEntry(null);
      return undefined;
    }
    const controller = new AbortController();
    void fetch(endpoint, { headers: { Accept: "application/json" }, signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error("diagnostics unavailable");
        return response.json() as Promise<{ entries?: JobDiagnosticEntry[] }>;
      })
      .then((document) => {
        const failures = (document.entries ?? []).filter((item) => item.event === "failed");
        setEntry(failures.at(-1) ?? null);
      })
      .catch((error: unknown) => {
        if (!(error instanceof DOMException && error.name === "AbortError")) setEntry(null);
      });
    return () => controller.abort();
  }, [endpoint]);
  if (job?.state !== "failed") return null;
  if (!entry) {
    return <p className="rounded-xl border border-line bg-white/[0.03] p-3 text-xs leading-5 text-muted">No sanitized diagnostic record is available for this job.</p>;
  }
  return (
    <section className="research-card research-card-dark" aria-label="Failure diagnostics">
      <div className="flex items-center gap-2"><XCircle size={16} className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Failure evidence</p></div>
      <dl className="contract-list mt-4"><div><dt>Type</dt><dd>{entry.exception_type ?? "unknown"}</dd></div><div><dt>Stage</dt><dd>{entry.stage ?? "unknown"}</dd></div></dl>
      {entry.exception_message ? <p className="mt-3 break-words text-xs leading-5 text-signal">{entry.exception_message}</p> : null}
      {entry.traceback ? <details className="mt-3"><summary className="cursor-pointer font-mono text-[0.65rem] text-muted">Sanitized traceback</summary><pre className="mt-3 max-h-72 overflow-auto whitespace-pre-wrap break-words rounded-lg border border-line bg-black/20 p-3 font-mono text-[0.61rem] leading-5 text-muted">{entry.traceback}</pre></details> : null}
    </section>
  );
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

function GateRow({ label, detail, tone = "quiet" }: { label: string; detail: string; tone?: "good" | "quiet" | "warn" }) {
  return (
    <div className="gate-row">
      <div className="flex min-w-0 items-center gap-2"><StatusDot tone={tone} /><span className="truncate text-xs font-medium text-white">{label}</span></div>
      <span className="shrink-0 font-mono text-[0.62rem] text-muted">{detail}</span>
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
  const batchError = validatePositiveSetting(run.batchSize, "Batch size", 4096);
  const tokenError = validatePositiveSetting(run.maxNewTokens, "Max new tokens", 1_000_000);
  const ready = !modelError && !datasetError && !sampleError && !batchError && !tokenError;
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
    } catch {
      setRequestError("The capture job could not be started. Check the server and dataset prompt column.");
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
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Execution budget</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Rows and generation</h2></div><Lightning size={19} className="text-signal" /></div>
            <div className="mt-6 grid gap-4 sm:grid-cols-3">
              <label className="field-label" htmlFor="sample-cap">Sample cap<input id="sample-cap" className={`input-control mt-2 ${sampleError ? "input-control-error" : ""}`} value={run.sampleCap} onChange={(event) => update("sampleCap", event.target.value)} inputMode="numeric" /></label>
              <label className="field-label" htmlFor="batch-size">Batch size<input id="batch-size" className={`input-control mt-2 ${batchError ? "input-control-error" : ""}`} value={run.batchSize} onChange={(event) => update("batchSize", event.target.value)} inputMode="numeric" /></label>
              <label className="field-label" htmlFor="max-new-tokens">Max new tokens<input id="max-new-tokens" className={`input-control mt-2 ${tokenError ? "input-control-error" : ""}`} value={run.maxNewTokens} onChange={(event) => update("maxNewTokens", event.target.value)} inputMode="numeric" /></label>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-3"><p className={`field-hint ${sampleError ? "field-hint-error" : ""}`}>{sampleError ?? "Rows are bounded before execution."}</p><p className={`field-hint ${batchError ? "field-hint-error" : ""}`}>{batchError ?? "Schedule remains deterministic."}</p><p className={`field-hint ${tokenError ? "field-hint-error" : ""}`}>{tokenError ?? "Generation budget is recorded."}</p></div>
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
        <aside className="space-y-5"><section className="research-card research-card-dark"><div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Run contract</p><GitBranch size={16} className="text-cyan" /></div><dl className="contract-list mt-5"><div><dt>Model</dt><dd>{sources.modelId}</dd></div><div><dt>Dataset</dt><dd>{sources.datasetId}</dd></div><div><dt>Prompt</dt><dd>{sources.promptColumn}</dd></div><div><dt>Rows</dt><dd>{run.sampleCap}</dd></div><div><dt>Batch</dt><dd>{run.batchSize}</dd></div><div><dt>Mode</dt><dd>{run.mode.replace("_", " ")}</dd></div><div><dt>Tokens</dt><dd>{run.tokenTextPolicy}</dd></div><div><dt>Overhead</dt><dd>{run.measureCaptureOverhead ? "optional native pass" : "off"}</dd></div></dl></section><section className="research-card research-card-dark"><div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Live state</p></div><p className="mt-4 text-xs leading-5 text-muted">{job?.progress.message ?? "The executor will resolve the model, stream bounded dataset rows, and publish immutable evidence."}</p>{job?.progress.total ? <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-white/[0.06]"><div className="h-full rounded-full bg-cyan transition-all" style={{ width: `${Math.min(100, (job.progress.completed / job.progress.total) * 100)}%` }} /></div> : null}</section>{overhead ? <section className="research-card research-card-dark"><div className="flex items-center gap-2"><Pulse size={16} className="text-cyan" /><p className="label-caps text-[0.59rem] text-muted">Overhead result</p></div><dl className="contract-list mt-4"><div><dt>Status</dt><dd>{String(overhead.status ?? "unknown")}</dd></div><div><dt>Native forward</dt><dd>{typeof (overhead.native as Record<string, unknown> | null)?.mean_ms === "number" ? `${((overhead.native as Record<string, unknown>).mean_ms as number).toFixed(2)} ms/row` : "—"}</dd></div><div><dt>Captured forward</dt><dd>{typeof (overhead.captured as Record<string, unknown> | null)?.mean_ms === "number" ? `${((overhead.captured as Record<string, unknown>).mean_ms as number).toFixed(2)} ms/row` : "—"}</dd></div><div><dt>Delta</dt><dd>{typeof overhead.delta_percent === "number" ? `${(overhead.delta_percent as number).toFixed(2)}%` : "—"}</dd></div></dl></section> : null}<button type="button" className="button-primary w-full justify-between" disabled={!ready || starting || running} onClick={() => void startRun()}>{starting ? "Starting capture…" : running ? "Capture running…" : job?.state === "completed" ? "Capture complete" : "Start capture"}<ArrowRight size={16} weight="bold" /></button>{jobId && overheadRunning ? <button type="button" className="button-secondary w-full justify-between" onClick={() => void skipOverhead()}>Skip overhead measurement <XCircle size={16} /></button> : null}{jobId && running ? <button type="button" className="button-secondary w-full justify-between" onClick={() => void postJson(`/api/jobs/${encodeURIComponent(jobId)}/cancel`, {})}>{overheadRunning ? "Cancel study" : "Cancel capture"} <XCircle size={16} /></button> : null}{canResume && jobId ? <button type="button" className="button-secondary w-full justify-between" disabled={starting} onClick={() => void startRun(jobId)}>Resume from checkpoint <ArrowRight size={16} /></button> : null}{job?.state === "completed" ? <button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("runs")}>Inspect evidence <ArrowRight size={16} /></button> : null}{requestError || job?.error ? <p className="rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal" role="alert">{requestError ?? job?.error}</p> : null}<FailureDiagnostics job={job} /></aside>
      </div>
    </div>
  );
}

type RunEntry = {
  run_key: string;
  state?: string | null;
  token_event_count?: number;
  routing_event_count?: number;
};

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

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric-card"><p className="label-caps text-[0.56rem] text-muted">{label}</p><p className="mt-3 font-mono text-lg text-white">{value}</p><p className="mt-1 text-[0.65rem] text-muted">{detail}</p></div>;
}

function RunsPage() {
  const [state, setState] = useState<"loading" | "ready" | "unavailable">("loading");
  const [entries, setEntries] = useState<RunEntry[]>([]);
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
  const [interventionOperation, setInterventionOperation] = useState<"ablate" | "scale">("ablate");
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
        baseline_run_key: selectedRun,
        operation: interventionOperation,
        targets: [...selectedTargets].sort(),
        factor: interventionOperation === "scale" ? Number(scaleFactor) : null,
      });
      setInterventionJobId(response.job_id);
    } catch {
      setInterventionStatus("This baseline cannot run that intervention. Use a fresh completed baseline and discovered targets.");
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
  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Runs / Inspect</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white">Trace inventory.</h1><p className="mt-3 max-w-[58ch] text-sm leading-6 text-muted">Routing heatmaps, validation status, and activation artifacts belong to a completed run. The UI never fills missing evidence with a visual guess.</p></div><div className="research-header-meta"><StatusDot tone={state === "unavailable" ? "warn" : "good"} /><span>{state === "loading" ? "Reading workspace…" : state === "unavailable" ? "Workspace offline" : `${entries.length} registered`}</span></div></header>
      {state === "loading" ? <section className="empty-surface"><p className="text-sm text-muted">Reading run catalog…</p></section> : state === "unavailable" ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">Workspace unavailable.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Make the MoEAtlas server available before asking for stored traces.</p></section> : entries.length === 0 ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">No published traces.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Stage a run and let the executor publish its immutable shards. Heatmap cells appear only after routing events are validated.</p></section> : (
        <>
          <div className="run-selector-row"><label className="field-label" htmlFor="run-selector">Run key<select id="run-selector" className="input-control mt-2" value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>{entries.map((entry) => <option key={entry.run_key} value={entry.run_key}>{entry.run_key} · {entry.state ?? "unknown"}</option>)}</select></label><div className="runtime-pill"><StatusDot tone={selectedEntry?.state === "completed" ? "good" : "warn"} />{selectedEntry?.state ?? "unknown"}</div><div className="ml-auto flex flex-wrap gap-2"><a className="button-secondary" href={`/api/runs/${encodeURIComponent(selectedRun)}/export?format=bundle`} download>Export bundle</a><a className="button-secondary" href={`/api/runs/${encodeURIComponent(selectedRun)}/export?format=csv`} download>CSV</a></div></div>
          <div className="metric-grid"><MetricCard label="Tokens" value={summary?.token_count == null ? "—" : String(summary.token_count)} detail="validated token rows" /><MetricCard label="Assignments" value={summary?.assignment_count == null ? "—" : String(summary.assignment_count)} detail="selected expert routes" /><MetricCard label="Layers" value={summary?.layer_count == null ? "—" : String(summary.layer_count)} detail="published routing layers" /><MetricCard label="Experts" value={summary?.expert_count == null ? "—" : String(summary.expert_count)} detail="routed expert universe" /></div>
          {interventionEvidence ? <section className="research-card">
            <div className="flex flex-wrap items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-signal">Paired causal evidence</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Baseline versus intervention</h2></div><span className="source-card-type">{interventionEvidence.all_targets_exercised ? "targets exercised" : "target not exercised"}</span></div>
            <div className="metric-grid mt-5"><MetricCard label="Outputs changed" value={interventionEvidence.changed_output_fraction == null ? "—" : `${(interventionEvidence.changed_output_fraction * 100).toFixed(1)}%`} detail={`${interventionEvidence.changed_output_rows}/${interventionEvidence.row_count} paired rows`} /><MetricCard label="Task score delta" value={interventionEvidence.task_score_delta == null ? "unavailable" : interventionEvidence.task_score_delta.toFixed(4)} detail={interventionEvidence.score_name ?? "add a reference column for scoring"} /><MetricCard label="Latency delta" value={interventionEvidence.latency_delta_percent == null ? "—" : `${interventionEvidence.latency_delta_percent.toFixed(1)}%`} detail="same run settings" /><MetricCard label="Restoration" value={interventionEvidence.restoration_status} detail="temporary hooks removed" /></div>
            {!interventionEvidence.all_targets_exercised ? <p className="mt-4 rounded-xl border border-signal/30 bg-signal/[0.06] p-3 text-xs leading-5 text-signal">At least one selected expert was never called by these rows. This run does not establish a causal effect for that target.</p> : null}
            <div className="mt-4 flex flex-wrap gap-2"><a className="button-secondary" target="_blank" rel="noreferrer" href={`/api/compare/heatmap?baseline_run_key=${encodeURIComponent(interventionEvidence.baseline_run_key)}&comparison_run_key=${encodeURIComponent(interventionEvidence.intervention_run_key)}&metric=count_deltas`}>Open routing delta</a></div>
            <div className="mt-5 border-t border-line pt-5">
              <div className="flex flex-wrap items-start justify-between gap-3"><div><p className="label-caps text-[0.57rem] text-muted">Replication study</p><p className="mt-2 max-w-[62ch] text-xs leading-5 text-muted">Select repeated runs of the same intervention. Add runs against unrelated experts as negative controls.</p></div>{study ? <span className="source-card-type">{study.claim_status}</span> : null}</div>
              <div className="mt-4 grid gap-4 lg:grid-cols-2">
                <label className="field-label" htmlFor="study-runs">Repeated intervention runs<select id="study-runs" className="input-control mt-2 min-h-32" multiple value={studyRuns} onChange={(event) => setStudyRuns(Array.from(event.target.selectedOptions, (option) => option.value))}>{studyCandidates.filter((candidate) => candidate.recipe_fingerprint === interventionEvidence.recipe_fingerprint).map((candidate) => <option key={candidate.intervention_run_key} value={candidate.intervention_run_key}>{candidate.intervention_run_key}</option>)}</select></label>
                <label className="field-label" htmlFor="control-runs">Negative-control runs<select id="control-runs" className="input-control mt-2 min-h-32" multiple value={controlRuns} onChange={(event) => setControlRuns(Array.from(event.target.selectedOptions, (option) => option.value))}>{studyCandidates.filter((candidate) => candidate.recipe_fingerprint !== interventionEvidence.recipe_fingerprint && (!selectedControlRecipe || candidate.recipe_fingerprint === selectedControlRecipe) && !studyRuns.includes(candidate.intervention_run_key)).map((candidate) => <option key={candidate.intervention_run_key} value={candidate.intervention_run_key}>{candidate.intervention_run_key}</option>)}</select></label>
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
                    <select id="comparison-run" className="input-control mt-2" value={comparisonRun} onChange={(event) => setComparisonRun(event.target.value)}><option value="">Select baseline</option>{entries.filter((entry) => entry.run_key !== selectedRun).map((entry) => <option key={entry.run_key} value={entry.run_key}>{entry.run_key}</option>)}</select>
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
                {interventionEvidence ? <p className="mt-4 text-xs leading-5 text-muted">This is a derived intervention run. Select its recorded baseline to prepare another intervention.</p> : interventionTargets?.status === "available" ? <>
                  <label className="field-label mt-4" htmlFor="intervention-targets">Layer × expert targets
                    <select id="intervention-targets" className="input-control mt-2 min-h-40" multiple value={selectedTargets} onChange={(event) => setSelectedTargets(Array.from(event.target.selectedOptions, (option) => option.value))}>
                      {interventionTargets.targets.map((target) => <option key={target.label} value={target.label}>L{target.layer_index} × E{target.expert_index}</option>)}
                    </select>
                  </label>
                  <p className="mt-2 text-[0.65rem] leading-5 text-muted">Select one or more independently controllable experts. Hold Ctrl or Command to select several.</p>
                  <label className="field-label mt-3" htmlFor="intervention-operation">Operation
                    <select id="intervention-operation" className="input-control mt-2" value={interventionOperation} onChange={(event) => setInterventionOperation(event.target.value as typeof interventionOperation)}><option value="ablate">Disable output</option><option value="scale">Scale output</option></select>
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
