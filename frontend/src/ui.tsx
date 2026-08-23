import { useEffect, useState } from "react";
import { XCircle } from "@phosphor-icons/react";
import type { JobDiagnosticEntry, JobSnapshot } from "./jobs";

export type OperationCapabilityStatus = "available" | "run_validation_required" | "not_implemented" | "unavailable";
export type OperationCapability = {
  operation: string;
  label: string;
  status: OperationCapabilityStatus;
  reason: string;
  evidence: string[];
  changes_routing: boolean;
  skips_compute?: boolean | null;
};
export type InterventionCapability = {
  live_supported: boolean;
  tier: string;
  reason: string;
  weight_layout: string;
  execution_backend?: string | null;
  fused_backend?: boolean | null;
  operation_capabilities: OperationCapability[];
};

export function AppMark() {
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

export function StatusDot({ tone = "good" }: { tone?: "good" | "quiet" | "warn" }) {
  return <span className={`status-dot status-dot-${tone}`} aria-hidden="true" />;
}

export function FailureDiagnostics({ job }: { job: JobSnapshot | null }) {
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

export function GateRow({ label, detail, tone = "quiet" }: { label: string; detail: string; tone?: "good" | "quiet" | "warn" }) {
  return (
    <div className="gate-row">
      <div className="flex min-w-0 items-center gap-2"><StatusDot tone={tone} /><span className="truncate text-xs font-medium text-white">{label}</span></div>
      <span className="shrink-0 font-mono text-[0.62rem] text-muted">{detail}</span>
    </div>
  );
}

const OPERATION_STATUS_LABELS: Record<OperationCapabilityStatus, string> = {
  available: "Available",
  run_validation_required: "Run validation",
  not_implemented: "Not implemented",
  unavailable: "Unavailable",
};

export function OperationCapabilityList({ operations, compact = false }: { operations: OperationCapability[]; compact?: boolean }) {
  return <div className="divide-y divide-line">{operations.map((operation) => {
    const available = operation.status === "available";
    const pending = operation.status === "run_validation_required";
    return <div key={operation.operation} className={compact ? "py-3" : "grid gap-2 py-4 sm:grid-cols-[13rem_minmax(0,1fr)] sm:gap-5"}>
      <div className="flex items-center justify-between gap-3">
        <span className="text-xs font-medium text-white">{operation.label}</span>
        <span className={`shrink-0 font-mono text-[0.58rem] uppercase tracking-[0.08em] ${available ? "text-cyan" : pending ? "text-signal" : "text-muted"}`}>{OPERATION_STATUS_LABELS[operation.status]}</span>
      </div>
      <div>
        <p className="text-[0.68rem] leading-5 text-muted">{operation.reason}</p>
        {!compact && operation.evidence.length ? <p className="mt-1 font-mono text-[0.58rem] leading-4 text-muted/70">{operation.evidence.join(" · ")}</p> : null}
      </div>
    </div>;
  })}</div>;
}

export function MetricCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="metric-card"><p className="label-caps text-[0.56rem] text-muted">{label}</p><p className="mt-3 font-mono text-lg text-white">{value}</p><p className="mt-1 text-[0.65rem] text-muted">{detail}</p></div>;
}
