import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Check,
  CheckCircle,
  Cloud,
  Cpu,
  Database,
  GearSix,
  GitBranch,
  HardDrives,
  Lightning,
  Plus,
  Pulse,
  ShieldCheck,
  WifiHigh,
  XCircle,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

type NavigationItem = "analysis" | "discovery" | "run" | "runs" | "settings";
type RunnerMode = "local" | "remote";
type HubKind = "model" | "dataset";
type SearchState = "idle" | "loading" | "ready" | "unavailable";

type SourceDraft = {
  modelId: string;
  modelRevision: string;
  datasetId: string;
  datasetRevision: string;
  datasetConfig: string;
  datasetSplit: string;
};

type RunnerDraft = {
  mode: RunnerMode;
  endpoint: string;
};

type RunDraft = {
  mode: "generation" | "teacher_forced";
  sampleCap: string;
  batchSize: string;
  maxNewTokens: string;
  tokenTextPolicy: "redacted" | "stored";
  allowExport: boolean;
  retainRawPayloads: boolean;
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
};

const DEFAULT_RUNNER: RunnerDraft = { mode: "local", endpoint: "" };

const DEFAULT_RUN: RunDraft = {
  mode: "generation",
  sampleCap: "128",
  batchSize: "1",
  maxNewTokens: "128",
  tokenTextPolicy: "redacted",
  allowExport: true,
  retainRawPayloads: false,
};

const NAVIGATION: Array<{ id: NavigationItem; label: string; icon: Icon }> = [
  { id: "analysis", label: "Analysis", icon: Plus },
  { id: "discovery", label: "Discover", icon: GitBranch },
  { id: "run", label: "Run", icon: Lightning },
  { id: "runs", label: "Runs", icon: Pulse },
  { id: "settings", label: "Settings", icon: GearSix },
];

function readStored<T>(key: string, fallback: T): T {
  try {
    const value = window.localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
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

function validateEndpoint(value: string): string | null {
  if (!value.trim()) return null;
  try {
    const url = new URL(value.trim());
    if (url.protocol !== "http:" && url.protocol !== "https:") return "Use an HTTP or HTTPS endpoint.";
    return null;
  } catch {
    return "Use a complete HTTP or HTTPS endpoint, or leave it blank for an in-VM UI.";
  }
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
        </div>
      ) : null}
    </section>
  );
}

function RunnerBoundary({ value, onChange }: { value: RunnerDraft; onChange: (value: RunnerDraft) => void }) {
  const endpointError = value.mode === "remote" ? validateEndpoint(value.endpoint) : null;
  return (
    <section className="research-card">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="label-caps text-[0.59rem] text-cyan">Execution boundary</p>
          <h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Where inference runs</h2>
        </div>
        <StatusDot tone={value.mode === "remote" && endpointError ? "warn" : "good"} />
      </div>
      <div className="mt-5 inline-flex rounded-xl border border-line bg-ink p-1" role="group" aria-label="Execution target">
        {(["local", "remote"] as RunnerMode[]).map((mode) => (
          <button key={mode} type="button" className={`runner-tab ${value.mode === mode ? "runner-tab-active" : ""}`} aria-pressed={value.mode === mode} onClick={() => onChange({ ...value, mode })}>
            {mode === "local" ? <Cpu size={15} /> : <Cloud size={15} />}
            {mode === "local" ? "This machine" : "Provider VM"}
          </button>
        ))}
      </div>
      {value.mode === "remote" ? (
        <div className="mt-4">
          <label className="field-label" htmlFor="runner-endpoint">Runner endpoint <span className="field-optional">optional in-VM</span>
            <input id="runner-endpoint" className={`input-control mt-2 ${endpointError ? "input-control-error" : ""}`} value={value.endpoint} onChange={(event) => onChange({ ...value, endpoint: event.target.value })} placeholder="https://provider-port-or-runner" spellCheck={false} />
          </label>
          <p className={`mt-2 flex items-start gap-2 text-[0.68rem] leading-5 ${endpointError ? "text-signal" : "text-muted"}`}>
            {endpointError ? <XCircle size={15} className="mt-0.5 shrink-0" /> : <WifiHigh size={15} className="mt-0.5 shrink-0 text-cyan" />}
            {endpointError ?? "Use the provider’s port proxy, HTTPS runner, or leave blank when this console runs inside the VM. SSH is not required."}
          </p>
        </div>
      ) : (
        <p className="mt-4 flex items-start gap-2 text-[0.68rem] leading-5 text-muted"><HardDrives size={15} className="mt-0.5 shrink-0 text-cyan" />No path is needed here. The local runtime resolves its own model cache and accelerator.</p>
      )}
    </section>
  );
}

function AnalysisPage({ runner, setRunner, onNavigate }: { runner: RunnerDraft; setRunner: (value: RunnerDraft) => void; onNavigate: (item: NavigationItem) => void }) {
  const [sources, setSources] = useState<SourceDraft>(() => readStored("moeatlas-analysis-sources", DEFAULT_SOURCES));
  const [queued, setQueued] = useState(false);
  const modelError = useMemo(() => validateHubId(sources.modelId, "Model ID"), [sources.modelId]);
  const datasetError = useMemo(() => validateHubId(sources.datasetId, "Dataset ID"), [sources.datasetId]);
  const endpointError = runner.mode === "remote" ? validateEndpoint(runner.endpoint) : null;
  const ready = !modelError && !datasetError && !endpointError;

  function update(field: keyof SourceDraft, value: string) {
    setSources((current) => ({ ...current, [field]: value }));
    setQueued(false);
  }

  function queueDiscovery() {
    if (!ready) return;
    window.localStorage.setItem("moeatlas-analysis-sources", JSON.stringify(sources));
    window.localStorage.setItem("moeatlas-runner", JSON.stringify(runner));
    setQueued(true);
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
            <SourceCard kind="dataset" value={sources.datasetId} onChange={(value) => update("datasetId", value)} revision={sources.datasetRevision} onRevisionChange={(value) => update("datasetRevision", value)} config={sources.datasetConfig} onConfigChange={(value) => update("datasetConfig", value)} split={sources.datasetSplit} onSplitChange={(value) => update("datasetSplit", value)} error={datasetError} />
          </div>
          <RunnerBoundary value={runner} onChange={(value) => { setRunner(value); setQueued(false); }} />
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
              <div><dt>Target</dt><dd>{runner.mode === "local" ? "this machine" : runner.endpoint.trim() || "in-VM console"}</dd></div>
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
          <button type="button" className="button-primary w-full justify-between" disabled={!ready} onClick={queueDiscovery}>
            {queued ? "Discovery contract saved" : "Queue discovery"}<ArrowRight size={16} weight="bold" />
          </button>
          {queued ? <div className="space-y-2" role="status"><p className="rounded-xl border border-cyan/25 bg-cyan/[0.06] p-3 text-xs leading-5 text-cyan">Sources are saved locally for the next discovery step. No model or dataset has been loaded yet.</p><button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("discovery")}>Open preflight <ArrowRight size={16} /></button></div> : null}
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
  const [runner] = useState<RunnerDraft>(() => readStored("moeatlas-runner", DEFAULT_RUNNER));
  const [staged, setStaged] = useState(() => window.localStorage.getItem("moeatlas-preflight-staged") === "true");
  const hasContract = !validateHubId(sources.modelId, "Model ID") && !validateHubId(sources.datasetId, "Dataset ID");

  if (!hasContract) {
    return (
      <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><GitBranch size={21} /></div><h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.04em] text-white">No discovery contract.</h1><p className="mt-3 max-w-[43ch] text-center text-sm leading-6 text-muted">Bind a model and dataset on the analysis surface before asking the runtime to inspect anything.</p><button type="button" className="button-primary mt-6" onClick={() => onNavigate("analysis")}>Back to analysis <ArrowRight size={16} weight="bold" /></button></section>
    );
  }

  function stagePreflight() {
    window.localStorage.setItem("moeatlas-preflight-staged", "true");
    setStaged(true);
  }

  return (
    <div className="space-y-6">
      <header className="research-header">
        <div><p className="label-caps text-[0.61rem] text-signal">Discovery / Preflight</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white sm:text-5xl">Read the runtime before the run.</h1><p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted">This envelope shows what will be inspected and what still requires the selected GPU runner. A staged contract is not model evidence.</p></div>
        <div className="research-header-meta"><StatusDot tone={staged ? "good" : "quiet"} /><span>{staged ? "Preflight staged" : "Runtime inspection pending"}</span></div>
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
              <div><dt>Execution target</dt><dd>{runner.mode === "local" ? "this machine" : runner.endpoint.trim() || "in-VM console"}</dd></div>
            </dl>
          </section>
          <section className="research-card">
            <div className="flex items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Inspection gates</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">What the runner must prove</h2></div><GitBranch size={19} className="text-cyan" /></div>
            <div className="mt-5 divide-y divide-line"><GateRow label="Model configuration" detail="runtime read" /><GateRow label="MoE topology" detail="STRUCTURE pending" /><GateRow label="Router payload shape" detail="DECODE pending" /><GateRow label="Dataset schema" detail="READ pending" /><GateRow label="Immutable revision evidence" detail="RESOLVE pending" /></div>
          </section>
        </main>
        <aside className="space-y-5">
          <section className="research-card research-card-dark"><div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Resource envelope</p><Database size={16} className="text-cyan" /></div><dl className="contract-list mt-5"><div><dt>Weights</dt><dd>not measured</dd></div><div><dt>Accelerator</dt><dd>{runner.mode === "local" ? "local probe" : "provider probe"}</dd></div><div><dt>Rows</dt><dd>bounded later</dd></div><div><dt>Capture</dt><dd>off</dd></div></dl></section>
          <section className="research-card research-card-dark"><div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Evidence rule</p></div><p className="mt-4 text-xs leading-5 text-muted">Discovery can report topology and router seams. It must not label routing as captured until a real forward produces validated events.</p></section>
          <button type="button" className="button-primary w-full justify-between" onClick={stagePreflight}>{staged ? "Preflight staged" : "Stage runtime preflight"}<ArrowRight size={16} weight="bold" /></button>
          {staged ? <button type="button" className="button-secondary w-full justify-between" onClick={() => onNavigate("run")}>Configure run <ArrowRight size={16} /></button> : null}
          {staged ? <p className="rounded-xl border border-cyan/25 bg-cyan/[0.06] p-3 text-xs leading-5 text-cyan" role="status">Intent is saved locally. No model, dataset, or GPU process was started by this browser action.</p> : null}
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

function RunConfigPage({ runner, onNavigate }: { runner: RunnerDraft; onNavigate: (item: NavigationItem) => void }) {
  const [sources] = useState<SourceDraft>(() => ({ ...DEFAULT_SOURCES, ...readStored<Partial<SourceDraft>>("moeatlas-analysis-sources", {}) }));
  const [run, setRun] = useState<RunDraft>(() => ({ ...DEFAULT_RUN, ...readStored<Partial<RunDraft>>("moeatlas-run", {}) }));
  const [staged, setStaged] = useState(() => window.localStorage.getItem("moeatlas-run-staged") === "true");
  const modelError = validateHubId(sources.modelId, "Model ID");
  const datasetError = validateHubId(sources.datasetId, "Dataset ID");
  const sampleError = validatePositiveSetting(run.sampleCap, "Sample cap", 1_000_000);
  const batchError = validatePositiveSetting(run.batchSize, "Batch size", 4096);
  const tokenError = validatePositiveSetting(run.maxNewTokens, "Max new tokens", 1_000_000);
  const ready = !modelError && !datasetError && !sampleError && !batchError && !tokenError;

  function update(field: keyof RunDraft, value: string | boolean) {
    setRun((current) => ({ ...current, [field]: value } as RunDraft));
    setStaged(false);
  }

  function stageRun() {
    if (!ready) return;
    window.localStorage.setItem("moeatlas-run", JSON.stringify(run));
    window.localStorage.setItem("moeatlas-runner", JSON.stringify(runner));
    window.localStorage.setItem("moeatlas-run-staged", "true");
    setStaged(true);
  }

  if (modelError || datasetError) {
    return (
      <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Lightning size={21} /></div><h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.04em] text-white">No run inputs.</h1><p className="mt-3 max-w-[43ch] text-center text-sm leading-6 text-muted">Return to analysis and bind a model and dataset before configuring execution.</p><button type="button" className="button-primary mt-6" onClick={() => onNavigate("analysis")}>Back to analysis <ArrowRight size={16} weight="bold" /></button></section>
    );
  }

  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Run / Configure</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white sm:text-5xl">Set the capture budget.</h1><p className="mt-3 max-w-[62ch] text-sm leading-6 text-muted">Keep sampling, row limits, and privacy choices explicit. Staging this contract does not start the model or open a remote connection.</p></div><div className="research-header-meta"><StatusDot tone={ready ? "good" : "warn"} /><span>{staged ? "Run staged" : ready ? "Ready to stage" : "Invalid budget"}</span></div></header>
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
          </section>
          <section className="research-card">
            <div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Evidence and privacy</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">What may be retained</h2></div><ShieldCheck size={19} className="text-cyan" /></div>
            <div className="mt-6"><span className="field-label">Token text</span><div className="mt-2 inline-flex rounded-xl border border-line bg-ink p-1" role="group" aria-label="Token text policy"><button type="button" className={`runner-tab ${run.tokenTextPolicy === "redacted" ? "runner-tab-active" : ""}`} aria-pressed={run.tokenTextPolicy === "redacted"} onClick={() => update("tokenTextPolicy", "redacted")}>Redacted (default)</button><button type="button" className={`runner-tab ${run.tokenTextPolicy === "stored" ? "runner-tab-active" : ""}`} aria-pressed={run.tokenTextPolicy === "stored"} onClick={() => update("tokenTextPolicy", "stored")}>Store token text</button></div></div>
            <div className="mt-5 space-y-2"><label className="toggle-row"><input className="check-control" type="checkbox" checked={run.allowExport} onChange={(event) => update("allowExport", event.target.checked)} /><span><span className="block text-xs font-medium text-white">Allow artifact export</span><span className="mt-1 block text-[0.68rem] leading-5 text-muted">Keep export available for validated run artifacts.</span></span></label><label className="toggle-row"><input className="check-control" type="checkbox" checked={run.retainRawPayloads} onChange={(event) => update("retainRawPayloads", event.target.checked)} /><span><span className="block text-xs font-medium text-white">Retain raw router payloads</span><span className="mt-1 block text-[0.68rem] leading-5 text-muted">Opt in only when payload inspection is necessary.</span></span></label></div>
          </section>
          <section className="research-card"><div className="flex items-start justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Worker boundary</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Execution handoff</h2></div><WifiHigh size={19} className="text-cyan" /></div><div className="mt-5 flex flex-wrap items-center gap-3"><span className="runtime-pill"><StatusDot />{runner.mode === "local" ? "This machine" : "Provider VM"}</span>{runner.mode === "remote" ? <span className="font-mono text-[0.65rem] text-muted">{runner.endpoint.trim() || "in-VM console"}</span> : <span className="text-xs text-muted">Local accelerator is resolved by the runtime.</span>}</div><p className="mt-4 text-xs leading-5 text-muted">Remote execution uses the provider’s exposed HTTP path or an in-VM console. There is no browser-side SSH feature.</p></section>
        </main>
        <aside className="space-y-5"><section className="research-card research-card-dark"><div className="flex items-center justify-between"><p className="label-caps text-[0.59rem] text-muted">Run contract</p><GitBranch size={16} className="text-cyan" /></div><dl className="contract-list mt-5"><div><dt>Model</dt><dd>{sources.modelId}</dd></div><div><dt>Dataset</dt><dd>{sources.datasetId}</dd></div><div><dt>Rows</dt><dd>{run.sampleCap}</dd></div><div><dt>Batch</dt><dd>{run.batchSize}</dd></div><div><dt>Mode</dt><dd>{run.mode.replace("_", " ")}</dd></div><div><dt>Tokens</dt><dd>{run.tokenTextPolicy}</dd></div></dl></section><section className="research-card research-card-dark"><div className="flex items-center gap-2"><Lightning size={15} weight="fill" className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">State boundary</p></div><p className="mt-4 text-xs leading-5 text-muted">A staged contract is durable UI intent. Live progress starts only when the executor accepts it and publishes run records.</p></section><button type="button" className="button-primary w-full justify-between" disabled={!ready} onClick={stageRun}>{staged ? "Run contract staged" : "Stage run contract"}<ArrowRight size={16} weight="bold" /></button>{staged ? <p className="rounded-xl border border-cyan/25 bg-cyan/[0.06] p-3 text-xs leading-5 text-cyan" role="status">The configuration is saved locally. No model or dataset has been loaded by this browser action.</p> : null}</aside>
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

function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric-card"><p className="label-caps text-[0.56rem] text-muted">{label}</p><p className="mt-3 font-mono text-lg text-white">{value}</p><p className="mt-1 text-[0.65rem] text-muted">{detail}</p></div>;
}

function RunsPage() {
  const [state, setState] = useState<"loading" | "ready" | "unavailable">("loading");
  const [entries, setEntries] = useState<RunEntry[]>([]);
  const [selectedRun, setSelectedRun] = useState("");
  const [summary, setSummary] = useState<RunSummary | null>(null);
  const [summaryState, setSummaryState] = useState<"idle" | "loading" | "ready" | "unavailable">("idle");

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

  const selectedEntry = entries.find((entry) => entry.run_key === selectedRun);
  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Runs / Inspect</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white">Trace inventory.</h1><p className="mt-3 max-w-[58ch] text-sm leading-6 text-muted">Routing heatmaps, validation status, and activation artifacts belong to a completed run. The UI never fills missing evidence with a visual guess.</p></div><div className="research-header-meta"><StatusDot tone={state === "unavailable" ? "warn" : "good"} /><span>{state === "loading" ? "Reading workspace…" : state === "unavailable" ? "Workspace offline" : `${entries.length} registered`}</span></div></header>
      {state === "loading" ? <section className="empty-surface"><p className="text-sm text-muted">Reading run catalog…</p></section> : state === "unavailable" ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">Workspace unavailable.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Start the local UI server or connect the provider runner before asking for stored traces.</p></section> : entries.length === 0 ? <section className="empty-surface"><div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div><h2 className="mt-5 font-display text-2xl font-semibold tracking-[-0.04em] text-white">No published traces.</h2><p className="mt-3 max-w-[42ch] text-center text-sm leading-6 text-muted">Stage a run and let the executor publish its immutable shards. Heatmap cells appear only after routing events are validated.</p></section> : (
        <>
          <div className="run-selector-row"><label className="field-label" htmlFor="run-selector">Run key<select id="run-selector" className="input-control mt-2" value={selectedRun} onChange={(event) => setSelectedRun(event.target.value)}>{entries.map((entry) => <option key={entry.run_key} value={entry.run_key}>{entry.run_key} · {entry.state ?? "unknown"}</option>)}</select></label><div className="runtime-pill"><StatusDot tone={selectedEntry?.state === "completed" ? "good" : "warn"} />{selectedEntry?.state ?? "unknown"}</div></div>
          <div className="metric-grid"><MetricCard label="Tokens" value={summary?.token_count == null ? "—" : String(summary.token_count)} detail="validated token rows" /><MetricCard label="Assignments" value={summary?.assignment_count == null ? "—" : String(summary.assignment_count)} detail="selected expert routes" /><MetricCard label="Layers" value={summary?.layer_count == null ? "—" : String(summary.layer_count)} detail="published routing layers" /><MetricCard label="Experts" value={summary?.expert_count == null ? "—" : String(summary.expert_count)} detail="routed expert universe" /></div>
          <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_19rem]">
            <section className="research-card"><div className="flex items-center justify-between gap-4"><div><p className="label-caps text-[0.59rem] text-cyan">Routing load</p><h2 className="mt-1 font-display text-xl font-semibold tracking-[-0.035em] text-white">Layer × expert heatmap</h2></div><span className="source-card-type">{summaryState === "loading" ? "loading" : summary?.status ?? "unavailable"}</span></div>{summary?.status === "available" && selectedRun ? <iframe className="heatmap-frame-react mt-5" title={`Routing heatmap for ${selectedRun}`} src={`/api/runs/${encodeURIComponent(selectedRun)}/heatmap`} /> : <div className="empty-heatmap mt-5"><GitBranch size={20} className="text-muted" /><p className="mt-3 text-sm text-white">No published matrix.</p><p className="mt-1 max-w-[32ch] text-center text-xs leading-5 text-muted">{summary?.reason ?? "A validated routing inspection is required before rendering heat."}</p></div>}</section>
            <aside className="space-y-5"><section className="research-card research-card-dark"><div className="flex items-center gap-2"><ShieldCheck size={16} className="text-cyan" /><p className="label-caps text-[0.59rem] text-muted">Validation</p></div><dl className="contract-list mt-5"><div><dt>Status</dt><dd>{summary?.status ?? "pending"}</dd></div><div><dt>Adapter</dt><dd>{summary?.adapter_name ?? "—"}</dd></div><div><dt>Top-k</dt><dd>{summary?.routed_top_k == null ? "—" : summary.routed_top_k}</dd></div><div><dt>Digest</dt><dd>{summary?.inspection_digest ? summary.inspection_digest.slice(0, 18) + "…" : "—"}</dd></div></dl></section><section className="research-card research-card-dark"><div className="flex items-center gap-2"><Pulse size={16} className="text-signal" /><p className="label-caps text-[0.59rem] text-muted">Expert activity</p></div><p className="mt-4 text-xs leading-5 text-muted">Activation summaries will appear when the run publishes expert-event evidence. Routing load alone is not an activation claim.</p></section></aside>
          </div>
        </>
      )}
    </div>
  );
}

function SettingsPage({ runner, setRunner }: { runner: RunnerDraft; setRunner: (value: RunnerDraft) => void }) {
  return (
    <div className="space-y-6">
      <header className="research-header"><div><p className="label-caps text-[0.61rem] text-signal">Settings</p><h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.055em] text-white">Runtime boundary.</h1><p className="mt-3 max-w-[58ch] text-sm leading-6 text-muted">Connection details belong to the execution boundary, not the analysis identity. Nothing here launches a VM or opens an SSH session.</p></div></header>
      <div className="max-w-3xl"><RunnerBoundary value={runner} onChange={(value) => { setRunner(value); window.localStorage.setItem("moeatlas-runner", JSON.stringify(value)); }} /></div>
    </div>
  );
}

export function App() {
  const [active, setActive] = useState<NavigationItem>("analysis");
  const [runner, setRunner] = useState<RunnerDraft>(() => readStored("moeatlas-runner", DEFAULT_RUNNER));
  const content = active === "analysis" ? <AnalysisPage runner={runner} setRunner={setRunner} onNavigate={setActive} /> : active === "discovery" ? <DiscoveryPage onNavigate={setActive} /> : active === "run" ? <RunConfigPage runner={runner} onNavigate={setActive} /> : active === "runs" ? <RunsPage /> : <SettingsPage runner={runner} setRunner={setRunner} />;

  return (
    <div className="app-shell min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1680px]">
        <aside className="hidden w-[14.5rem] shrink-0 flex-col border-r border-line px-5 py-6 lg:flex">
          <AppMark />
          <div className="mt-12"><p className="label-caps px-3 text-[0.58rem] text-muted">Observe</p><nav className="mt-3 space-y-1" aria-label="Primary navigation">{NAVIGATION.map((item) => { const Icon = item.icon; const selected = active === item.id; return <button type="button" key={item.id} className={`nav-item ${selected ? "nav-item-active" : ""}`} onClick={() => setActive(item.id)} aria-current={selected ? "page" : undefined}><Icon size={18} weight={selected ? "fill" : "regular"} />{item.label}</button>; })}</nav></div>
          <div className="mt-auto flex items-center justify-between px-1 text-[0.68rem] text-muted"><span>MoEAtlas</span><span>v0.1.0</span></div>
        </aside>
        <main className="min-w-0 flex-1 px-4 py-4 sm:px-7 sm:py-6 lg:px-10">
          <header className="mb-9 flex flex-wrap items-center justify-between gap-4"><div className="flex items-center gap-3 lg:hidden"><AppMark /></div><div className="hidden items-center gap-2 text-xs text-muted lg:flex"><span className="text-white">MoEAtlas</span><span className="text-muted/40">/</span><span>{NAVIGATION.find((item) => item.id === active)?.label}</span></div><div className="ml-auto flex items-center gap-3"><div className="runtime-pill"><StatusDot /><span>{runner.mode === "local" ? "Local runtime" : "Provider runtime"}</span></div><span className="hidden font-mono text-[0.62rem] text-muted sm:inline">schema 1.0</span></div></header>
          <nav className="mb-7 flex gap-1 overflow-x-auto border-b border-line pb-2 lg:hidden" aria-label="Primary navigation">{NAVIGATION.map((item) => { const Icon = item.icon; return <button type="button" key={item.id} className={`mobile-nav-item ${active === item.id ? "mobile-nav-item-active" : ""}`} onClick={() => setActive(item.id)}><Icon size={15} />{item.label}</button>; })}</nav>
          {content}
        </main>
      </div>
    </div>
  );
}
