import { useEffect, useState } from "react";

export type JobProgress = {
  stage: string;
  completed: number;
  total?: number | null;
  message: string;
};

export type JobSnapshot = {
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

export type JobDiagnosticEntry = {
  sequence: number;
  event: string;
  stage?: string | null;
  exception_type?: string | null;
  exception_message?: string | null;
  traceback?: string | null;
};

export function useJob(jobId: string | null): JobSnapshot | null {
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

export async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) throw new Error((await response.text()) || "request failed");
  return (await response.json()) as T;
}
