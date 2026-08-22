import { useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Check,
  CheckCircle,
  Cloud,
  Command,
  Cpu,
  Database,
  GearSix,
  GitBranch,
  HardDrives,
  Lightning,
  Plus,
  Pulse,
  ShieldCheck,
  SquaresFour,
  WifiHigh,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";

type DeploymentMode = "local" | "vm";
type NavigationItem = "workspace" | "analysis" | "runs" | "settings";

type ModeOption = {
  id: DeploymentMode;
  label: string;
  eyebrow: string;
  description: string;
  details: string[];
  icon: Icon;
};

const MODE_OPTIONS: ModeOption[] = [
  {
    id: "local",
    label: "Local GPU",
    eyebrow: "On this machine",
    description: "Use CUDA or Metal on the workstation that owns the model cache.",
    details: ["Works offline", "Private by default", "Direct accelerator access"],
    icon: Cpu,
  },
  {
    id: "vm",
    label: "Remote VM",
    eyebrow: "Vast.ai or another host",
    description: "Run the console in the provider workspace or point it at an exposed runner URL.",
    details: ["Browser-terminal friendly", "Provider port or HTTPS", "Explicit connection state"],
    icon: Cloud,
  },
];

const NAVIGATION: Array<{ id: NavigationItem; label: string; icon: Icon }> = [
  { id: "workspace", label: "Workspace", icon: SquaresFour },
  { id: "analysis", label: "New analysis", icon: Plus },
  { id: "runs", label: "Runs", icon: Pulse },
  { id: "settings", label: "Settings", icon: GearSix },
];

const STAGES = ["Source", "Discover", "Run", "Inspect"];

function AppMark() {
  return (
    <div className="flex items-center gap-3" aria-label="MoEAtlas">
      <div className="relative grid size-10 place-items-center overflow-hidden rounded-xl border border-signal/40 bg-signal/10 shadow-[0_0_28px_rgba(255,142,74,0.12)]">
        <span className="absolute size-5 rounded-full border border-signal/90" />
        <span className="absolute size-8 rounded-full border border-dashed border-cyan/45" />
        <span className="size-1.5 rounded-full bg-signal shadow-[0_0_14px_rgba(255,142,74,0.95)]" />
      </div>
      <div>
        <div className="font-display text-[1.05rem] font-semibold tracking-[-0.03em] text-white">
          MoE<span className="text-signal">Atlas</span>
        </div>
        <div className="label-caps mt-0.5 text-[0.58rem] text-muted">Routing observatory</div>
      </div>
    </div>
  );
}

function StatusDot({ tone = "good" }: { tone?: "good" | "quiet" | "warn" }) {
  return <span className={`status-dot status-dot-${tone}`} aria-hidden="true" />;
}

function ModeCard({
  option,
  selected,
  onSelect,
}: {
  option: ModeOption;
  selected: boolean;
  onSelect: () => void;
}) {
  const Icon = option.icon;
  return (
    <button
      type="button"
      className={`group relative min-h-[16.5rem] rounded-2xl border p-5 text-left transition duration-200 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-signal/80 ${
        selected
          ? "border-signal/75 bg-signal/[0.08] shadow-[0_0_38px_rgba(255,142,74,0.1)]"
          : "border-line bg-panel/70 hover:-translate-y-0.5 hover:border-white/20 hover:bg-panel"
      }`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <div className="flex items-start justify-between gap-4">
        <div className={`grid size-11 place-items-center rounded-xl border ${selected ? "border-signal/45 bg-signal/15 text-signal" : "border-line-bright bg-white/[0.04] text-muted"}`}>
          <Icon size={22} weight={selected ? "fill" : "regular"} />
        </div>
        <span className={`grid size-6 place-items-center rounded-full border ${selected ? "border-signal bg-signal text-ink" : "border-line-bright text-transparent"}`}>
          <Check size={14} weight="bold" />
        </span>
      </div>
      <div className="mt-7">
        <p className="label-caps text-[0.62rem] text-muted">{option.eyebrow}</p>
        <h3 className="mt-2 font-display text-xl font-semibold tracking-[-0.03em] text-white">{option.label}</h3>
        <p className="mt-2 max-w-[27ch] text-sm leading-6 text-muted">{option.description}</p>
      </div>
      <ul className="mt-5 space-y-2 border-t border-line pt-4 text-xs text-muted">
        {option.details.map((detail) => (
          <li key={detail} className="flex items-center gap-2">
            <CheckCircle size={14} weight="fill" className={selected ? "text-signal" : "text-muted/70"} />
            {detail}
          </li>
        ))}
      </ul>
    </button>
  );
}

function StatCard({ icon: Icon, label, value, detail }: { icon: Icon; label: string; value: string; detail: string }) {
  return (
    <div className="rounded-2xl border border-line bg-panel/75 p-4">
      <div className="flex items-center justify-between">
        <span className="grid size-8 place-items-center rounded-lg border border-line-bright bg-white/[0.035] text-cyan">
          <Icon size={17} />
        </span>
        <span className="label-caps text-[0.58rem] text-muted">{label}</span>
      </div>
      <p className="mt-4 font-display text-2xl font-semibold tracking-[-0.04em] text-white">{value}</p>
      <p className="mt-1 text-xs leading-5 text-muted">{detail}</p>
    </div>
  );
}

function WorkspaceHome({ mode, setMode, onNavigate }: { mode: DeploymentMode; setMode: (mode: DeploymentMode) => void; onNavigate: (item: NavigationItem) => void }) {
  const [saved, setSaved] = useState(false);
  const selectedOption = useMemo(() => MODE_OPTIONS.find((option) => option.id === mode)!, [mode]);

  useEffect(() => {
    const savedMode = window.localStorage.getItem("moeatlas-deployment-mode");
    if (savedMode === "local" || savedMode === "vm") setMode(savedMode);
  }, [setMode]);

  function handleSave() {
    window.localStorage.setItem("moeatlas-deployment-mode", mode);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2200);
  }

  return (
    <div className="space-y-7">
      <section className="hero-grid relative overflow-hidden rounded-3xl border border-line bg-panel px-6 py-7 sm:px-9 sm:py-10">
        <div className="relative z-10 max-w-2xl">
          <div className="label-caps flex items-center gap-2 text-[0.62rem] text-signal">
            <Lightning size={13} weight="fill" />
            Instrument the invisible
          </div>
          <h1 className="mt-4 max-w-[14ch] font-display text-4xl font-semibold leading-[1.02] tracking-[-0.055em] text-white sm:text-6xl">
            See where every token goes.
          </h1>
          <p className="mt-5 max-w-[56ch] text-sm leading-7 text-muted sm:text-base">
            MoEAtlas turns a model and a dataset into an inspectable routing trace — from discovery to expert-level evidence.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <button type="button" className="button-primary" onClick={() => onNavigate("analysis")}>
              Start an analysis <ArrowRight size={16} weight="bold" />
            </button>
            <button type="button" className="button-secondary" onClick={() => onNavigate("settings")}>
              Runtime settings
            </button>
          </div>
        </div>
        <div className="hero-orbit" aria-hidden="true">
          <div className="hero-orbit-ring hero-orbit-ring-one" />
          <div className="hero-orbit-ring hero-orbit-ring-two" />
          <div className="hero-orbit-core"><span /></div>
          <div className="hero-orbit-label hero-orbit-label-top">router</div>
          <div className="hero-orbit-label hero-orbit-label-bottom">experts</div>
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-3">
        <StatCard icon={Database} label="Workspace" value="Ready" detail="Local-first artifacts and run history" />
        <StatCard icon={GitBranch} label="Capture" value="Generic" detail="Structure-driven routing seam enabled" />
        <StatCard icon={ShieldCheck} label="Evidence" value="Bounded" detail="Provenance and validation stay visible" />
      </section>

      <section className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_21rem]">
        <div className="rounded-3xl border border-line bg-panel/75 p-6 sm:p-8">
          <div className="flex flex-wrap items-end justify-between gap-4">
            <div>
              <p className="label-caps text-[0.62rem] text-muted">Step 01 / Runtime profile</p>
              <h2 className="mt-2 font-display text-2xl font-semibold tracking-[-0.04em] text-white">Where will the model run?</h2>
              <p className="mt-2 max-w-[50ch] text-sm leading-6 text-muted">Choose the machine that owns inference. You can switch profiles later without changing the analysis contract.</p>
            </div>
            <div className="profile-chip"><StatusDot /> {selectedOption.label}</div>
          </div>
          <div className="mt-7 grid gap-4 md:grid-cols-2">
            {MODE_OPTIONS.map((option) => (
              <ModeCard key={option.id} option={option} selected={mode === option.id} onSelect={() => { setMode(option.id); setSaved(false); }} />
            ))}
          </div>
          <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-5">
            <div className="flex items-center gap-2 text-xs text-muted">
              {mode === "local" ? <HardDrives size={15} className="text-cyan" /> : <WifiHigh size={15} className="text-cyan" />}
              {mode === "local" ? "No network required for cached models." : "The provider endpoint will be checked before a run."}
            </div>
            <button type="button" className="button-secondary" onClick={handleSave}>
              {saved ? <><Check size={15} weight="bold" /> Profile saved</> : "Save profile"}
            </button>
          </div>
        </div>

        <aside className="rounded-3xl border border-line bg-ink/70 p-6">
          <div className="flex items-center justify-between">
            <p className="label-caps text-[0.62rem] text-muted">Workflow</p>
            <span className="label-caps text-[0.58rem] text-signal">Local-first</span>
          </div>
          <div className="mt-6 space-y-5">
            {STAGES.map((stage, index) => (
              <div key={stage} className="flex gap-3">
                <div className={`workflow-marker ${index === 0 ? "workflow-marker-active" : ""}`}>{String(index + 1).padStart(2, "0")}</div>
                <div>
                  <p className={`text-sm font-medium ${index === 0 ? "text-white" : "text-muted"}`}>{stage}</p>
                  <p className="mt-1 text-xs leading-5 text-muted/70">{index === 0 ? "Choose a runtime profile" : index === 1 ? "Read the model’s structure" : index === 2 ? "Capture bounded evidence" : "Explore routing and activity"}</p>
                </div>
              </div>
            ))}
          </div>
          <div className="mt-7 rounded-2xl border border-cyan/20 bg-cyan/[0.06] p-4">
            <div className="flex items-start gap-3">
              <Command size={18} className="mt-0.5 shrink-0 text-cyan" />
              <p className="text-xs leading-5 text-muted">Paste exact Hugging Face IDs when you are ready. Search suggestions arrive in the next slice.</p>
            </div>
          </div>
        </aside>
      </section>
    </div>
  );
}

function EmptySurface({ title, description, action, onAction }: { title: string; description: string; action?: string; onAction?: () => void }) {
  return (
    <section className="empty-surface">
      <div className="grid size-12 place-items-center rounded-2xl border border-line-bright bg-white/[0.04] text-signal"><Pulse size={21} /></div>
      <h1 className="mt-5 font-display text-3xl font-semibold tracking-[-0.04em] text-white">{title}</h1>
      <p className="mt-3 max-w-[45ch] text-center text-sm leading-6 text-muted">{description}</p>
      {action && onAction ? <button type="button" className="button-primary mt-6" onClick={onAction}>{action} <ArrowRight size={16} weight="bold" /></button> : null}
    </section>
  );
}

function SettingsSurface({ mode, setMode }: { mode: DeploymentMode; setMode: (mode: DeploymentMode) => void }) {
  return (
    <div className="space-y-6">
      <div>
        <p className="label-caps text-[0.62rem] text-signal">Runtime</p>
        <h1 className="mt-2 font-display text-4xl font-semibold tracking-[-0.05em] text-white">Settings</h1>
        <p className="mt-3 max-w-[58ch] text-sm leading-6 text-muted">Keep the execution boundary explicit. MoEAtlas never silently moves a run from your machine to a remote GPU, and remote mode does not require local SSH.</p>
      </div>
      <section className="max-w-3xl rounded-3xl border border-line bg-panel/75 p-6 sm:p-8">
        <div className="flex items-start gap-4">
          <div className="grid size-10 place-items-center rounded-xl border border-signal/30 bg-signal/10 text-signal"><Cpu size={19} /></div>
          <div>
            <h2 className="font-display text-xl font-semibold tracking-[-0.03em] text-white">Default runtime profile</h2>
            <p className="mt-2 text-sm leading-6 text-muted">This profile controls which connection the next analysis starts with. It does not launch a VM or change an existing run.</p>
          </div>
        </div>
        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          {MODE_OPTIONS.map((option) => <ModeCard key={option.id} option={option} selected={mode === option.id} onSelect={() => { setMode(option.id); window.localStorage.setItem("moeatlas-deployment-mode", option.id); }} />)}
        </div>
      </section>
    </div>
  );
}

export function App() {
  const [active, setActive] = useState<NavigationItem>("workspace");
  const [mode, setMode] = useState<DeploymentMode>("local");

  const content = active === "workspace" ? (
    <WorkspaceHome mode={mode} setMode={setMode} onNavigate={setActive} />
  ) : active === "settings" ? (
    <SettingsSurface mode={mode} setMode={setMode} />
  ) : active === "analysis" ? (
    <EmptySurface title="Start an analysis" description="The model and dataset intake form is the next feature. Your runtime profile is ready." action="Back to workspace" onAction={() => setActive("workspace")} />
  ) : (
    <EmptySurface title="Run history" description="Completed runs, live progress, and validation evidence will collect here." action="Start an analysis" onAction={() => setActive("analysis")} />
  );

  return (
    <div className="app-shell min-h-screen">
      <div className="mx-auto flex min-h-screen max-w-[1680px]">
        <aside className="hidden w-[15.5rem] shrink-0 flex-col border-r border-line px-5 py-6 lg:flex">
          <AppMark />
          <div className="mt-12">
            <p className="label-caps px-3 text-[0.58rem] text-muted">Observe</p>
            <nav className="mt-3 space-y-1" aria-label="Primary navigation">
              {NAVIGATION.map((item) => {
                const Icon = item.icon;
                const selected = active === item.id;
                return <button type="button" key={item.id} className={`nav-item ${selected ? "nav-item-active" : ""}`} onClick={() => setActive(item.id)} aria-current={selected ? "page" : undefined}><Icon size={18} weight={selected ? "fill" : "regular"} />{item.label}</button>;
              })}
            </nav>
          </div>
          <div className="mt-auto space-y-4">
            <div className="rounded-2xl border border-line bg-panel/70 p-4">
              <div className="flex items-center gap-2"><StatusDot /><span className="text-xs font-medium text-white">Workspace ready</span></div>
              <p className="mt-2 text-xs leading-5 text-muted">Artifacts stay beside the workspace and remain inspectable.</p>
            </div>
            <div className="flex items-center justify-between px-1 text-[0.68rem] text-muted"><span>MoEAtlas</span><span>v0.1.0</span></div>
          </div>
        </aside>

        <main className="min-w-0 flex-1 px-4 py-4 sm:px-7 sm:py-6 lg:px-10">
          <header className="mb-8 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3 lg:hidden"><AppMark /></div>
            <div className="hidden items-center gap-2 text-xs text-muted lg:flex"><span className="text-white">Workspace</span><span className="text-muted/40">/</span><span>{active === "workspace" ? "Overview" : NAVIGATION.find((item) => item.id === active)?.label}</span></div>
            <div className="ml-auto flex items-center gap-2 sm:gap-3">
              <div className="runtime-pill"><StatusDot /> <span className="hidden sm:inline">{mode === "local" ? "Local GPU profile" : "Remote VM profile"}</span><span className="sm:hidden">{mode === "local" ? "Local" : "VM"}</span></div>
              <button type="button" className="icon-button" aria-label="Open command menu"><Command size={18} /></button>
            </div>
          </header>
          {content}
        </main>
      </div>
    </div>
  );
}
