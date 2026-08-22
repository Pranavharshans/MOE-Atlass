/* MoEAtlas local UI — dependency-free vanilla JS (fetch + DOM only). */
"use strict";

(function () {
  const RUN_STATES = [
    "planned",
    "provisioning",
    "running",
    "finalizing",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
  ];

  const view = document.getElementById("view");
  const healthPill = document.getElementById("health-pill");

  /* ---------- helpers ---------- */

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [key, value] of Object.entries(attrs)) {
        if (key === "text") {
          node.textContent = value;
        } else if (key === "class") {
          node.className = value;
        } else if (key.startsWith("on")) {
          node.addEventListener(key.slice(2).toLowerCase(), value);
        } else if (value !== null && value !== undefined) {
          node.setAttribute(key, value);
        }
      }
    }
    if (children) {
      for (const child of children) {
        if (child) node.appendChild(child);
      }
    }
    return node;
  }

  async function fetchJson(url) {
    let response;
    try {
      response = await fetch(url, { headers: { Accept: "application/json" } });
    } catch (err) {
      throw new Error(`Network request to ${url} failed — is moeatlas ui still running?`);
    }
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        if (body && typeof body.detail === "string") detail = body.detail;
      } catch (err) {
        /* keep the HTTP-status detail */
      }
      throw new Error(`${url} → ${detail}`);
    }
    return response.json();
  }

  function errorBanner(err) {
    return el("div", { class: "error-banner", role: "alert" }, [
      el("strong", { text: "Something went wrong." }),
      el("span", {
        text:
          err && err.message
            ? err.message
            : String(err) || "Unknown error while loading data.",
      }),
    ]);
  }

  async function render(section) {
    view.replaceChildren(section);
    view.focus({ preventScroll: true });
  }

  function fmtDateTime(value) {
    if (!value) return "—";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC");
  }

  function stateBadge(state) {
    return el(
      "span",
      { class: "state-badge", "data-state": state ?? "unknown", text: state ?? "unknown" },
    );
  }

  function emptyState(message) {
    return el("div", { class: "empty-state", text: message });
  }

  /* ---------- health pill ---------- */

  async function refreshHealth() {
    healthPill.dataset.state = "unknown";
    healthPill.textContent = "checking health…";
    try {
      const doc = await fetchJson("/healthz");
      healthPill.dataset.state = "ok";
      healthPill.textContent =
        `${doc.package_name} v${doc.package_version} · validation: ${doc.model_validation_status}`;
    } catch (err) {
      healthPill.dataset.state = "down";
      healthPill.textContent = "server unreachable";
    }
  }

  /* ---------- workspace view ---------- */

  async function showWorkspace() {
    try {
      const [workspace, health] = await Promise.all([
        fetchJson("/api/workspace"),
        fetchJson("/healthz"),
      ]);
      render(
        el(null, null, [
          el("section", { class: "card" }, [
            el("h1", { text: "Workspace" }),
            el("dl", { class: "kv" }, [
              el("dt", { text: "path" }),
              el("dd", { text: workspace.workspace }),
              el("dt", { text: "package" }),
              el("dd", {
                text: `${health.package_name} ${health.package_version} (Python ${health.python_version})`,
              }),
              el("dt", { text: "model validation" }),
              el("dd", { text: health.model_validation_status + " (per validation ledger)" }),
            ]),
          ]),
          el("div", { class: "cards-grid" }, [
            statCard("Registered runs", String(workspace.run_count)),
            statCard("Server schema", "v1.0"),
            statCard("Model validation", health.model_validation_status),
          ]),
          el("section", { class: "card" }, [
            el("h2", { text: "Next steps" }),
            el("p", {}, [
              el("a", { href: "#/runs", text: "Browse runs" }),
              el("span", { text: " or open a " }),
              el("a", { href: "#/heatmap", text: "heatmap" }),
              el("span", { text: "." }),
            ]),
          ]),
        ]),
      );
    } catch (err) {
      render(errorBanner(err));
    }
  }

  function statCard(label, value) {
    return el("div", { class: "stat-card" }, [
      el("div", { class: "label", text: label }),
      el("div", { class: "value", text: value }),
    ]);
  }

  /* ---------- runs view ---------- */

  async function showRuns(stateFilter) {
    const controls = el("div", { class: "toolbar" }, [
      el("label", { for: "state-filter", text: "Filter by state" }),
      buildStateSelect(stateFilter),
    ]);

    let section;
    try {
      const query = stateFilter ? `?state=${encodeURIComponent(stateFilter)}` : "";
      const doc = await fetchJson(`/api/runs${query}`);
      const table = buildRunsTable(doc.entries);
      section = el(null, null, [
        el("h1", { text: "Runs" }),
        controls,
        doc.count === 0
          ? el("section", { class: "card" }, [
              emptyState(
                stateFilter
                  ? "No runs match this state filter yet."
                  : "No runs are registered in this workspace catalog yet.",
              ),
            ])
          : el("section", { class: "card" }, [table]),
      ]);
    } catch (err) {
      section = el(null, null, [el("h1", { text: "Runs" }), controls, errorBanner(err)]);
    }
    render(section);
  }

  function buildStateSelect(selected) {
    const select = el(
      "select",
      {
        id: "state-filter",
        onchange: () => {
          const value = select.value;
          location.hash = value ? "#/runs?state=" + encodeURIComponent(value) : "#/runs";
        },
      },
      [el("option", { value: "", text: "All states" })].concat(
        RUN_STATES.map((state) =>
          el("option", {
            value: state,
            text: state,
            ...(selected === state ? { selected: "" } : {}),
          }),
        ),
      ),
    );
    return select;
  }

  function buildRunsTable(entries) {
    const head = el("thead", null, [
      el("tr", null, ["Run key", "State", "Shards", "Tokens", "Routing events"].map((t) =>
        t === "Shards" || t === "Tokens" || t === "Routing events"
          ? el("th", { class: "num", scope: "col", text: t })
          : el("th", { scope: "col", text: t }),
      )),
    ]);
    const body = el(
      "tbody",
      null,
      entries.map((entry) =>
        el("tr", null, [
          el("td", null, [
            el("a", { href: `#/runs/${encodeURIComponent(entry.run_key)}`, text: entry.run_key }),
          ]),
          el("td", null, [stateBadge(entry.state)]),
          el("td", { class: "num", text: String(entry.shard_count) }),
          el("td", { class: "num", text: String(entry.token_event_count) }),
          el("td", { class: "num", text: String(entry.routing_event_count) }),
        ]),
      ),
    );
    return el("table", { class: "data" }, [head, body]);
  }

  /* ---------- run detail view ---------- */

  async function showRunDetail(runKey) {
    let decoded;
    try {
      decoded = decodeURIComponent(runKey);
    } catch (err) {
      decoded = runKey;
    }
    try {
      const [detail, summary] = await Promise.all([
        fetchJson(`/api/runs/${encodeURIComponent(decoded)}`),
        fetchJson(`/api/runs/${encodeURIComponent(decoded)}/summary`),
      ]);
      render(
        el(null, null, [
          el("section", { class: "card" }, [
            el("h1", { text: `Run ${detail.run_key}` }),
            stateBadge(detail.state),
            el("dl", { class: "kv", style: "margin-top:0.75rem" }, [
              el("dt", { text: "attempt" }),
              el("dd", { text: String(detail.attempt) }),
              el("dt", { text: "specification fingerprint" }),
              el("dd", { text: detail.specification_fingerprint ?? "—" }),
              el("dt", { text: "token-text policy" }),
              el("dd", { text: detail.token_text_policy ?? "—" }),
              el("dt", { text: "registered at" }),
              el("dd", { text: fmtDateTime(detail.registered_at) }),
              el("dt", { text: "updated at" }),
              el("dd", { text: fmtDateTime(detail.updated_at) }),
              el("dt", { text: "shards" }),
              el("dd", { text: String(detail.shards.length) }),
            ]),
          ]),
          buildSummaryPanel(summary),
          el("section", { class: "card" }, [
            el("h2", { text: "Heatmap" }),
            el("p", { class: "muted", text:
              "Open the published routing-load heatmap document for this run." }),
            el("button", {
              class: "action",
              type: "button",
              onclick: () => {
                location.hash = `#/heatmap/${encodeURIComponent(detail.run_key)}`;
              },
              text: "Open heatmap →",
            }),
          ]),
          el("section", { class: "card" }, [
            el("h2", { text: "Routing shards" }),
            detail.shards.length === 0
              ? emptyState("This run has no committed routing shards.")
              : buildShardsTable(detail.shards),
          ]),
        ]),
      );
    } catch (err) {
      render(el(null, null, [el("h1", { text: "Run detail" }), errorBanner(err)]));
    }
  }

  function buildSummaryPanel(summary) {
    const children = [];
    if (summary.status === "unavailable") {
      children.push(
        el("p", { class: "summary-unavailable", text:
          `Summary unavailable — ${summary.reason ?? "no reason provided"}.` }),
      );
    } else {
      children.push(
        el("pre", {
          style: "font-family:var(--mono);font-size:0.82rem;overflow:auto;margin:0",
          text: JSON.stringify(summary, null, 2),
        }),
      );
    }
    return el("section", { class: "card" }, [
      el("h2", { text: "Routing-load summary" }),
      ...children,
    ]);
  }

  function buildShardsTable(shards) {
    const head = el("thead", null, [
      el("tr", null, ["Shard key", "Path", "Tokens", "Routing events"].map((t) =>
        t === "Tokens" || t === "Routing events"
          ? el("th", { class: "num", scope: "col", text: t })
          : el("th", { scope: "col", text: t }),
      )),
    ]);
    const body = el(
      "tbody",
      null,
      shards.map((shard) =>
        el("tr", null, [
          el("td", { text: shard.shard_key }),
          el("td", { text: shard.relative_path }),
          el("td", { class: "num", text: String(shard.token_count) }),
          el("td", { class: "num", text: String(shard.routing_count) }),
        ]),
      ),
    );
    return el("table", { class: "data" }, [head, body]);
  }

  /* ---------- heatmap view ---------- */

  async function showHeatmap(runKey) {
    let decoded = null;
    if (runKey) {
      try {
        decoded = decodeURIComponent(runKey);
      } catch (err) {
        decoded = runKey;
      }
    }
    if (!decoded) {
      render(
        el(null, null, [
          el("h1", { text: "Heatmap" }),
          await heatmapPicker(),
        ]),
      );
      return;
    }
    try {
      const runs = await fetchJson("/api/runs");
      const registered = runs.entries.some((entry) => entry.run_key === decoded);
      if (!registered) {
        render(heatmapEmpty(`Run “${decoded}” is not registered in this workspace.`));
        return;
      }
    } catch (err) {
      render(errorBanner(err));
      return;
    }
    const container = el("div", null, []);
    // Honest empty state if the document is not published.
    fetch(`/api/runs/${encodeURIComponent(decoded)}/heatmap`).then(
      (response) => {
        if (response.ok) {
          container.replaceChildren(
            el("iframe", {
              class: "heatmap-frame",
              title: `Routing-load heatmap for run ${decoded}`,
              src: `/api/runs/${encodeURIComponent(decoded)}/heatmap`,
            }),
          );
        } else if (response.status === 404) {
          container.replaceChildren(
            heatmapEmpty(
              `No heatmap is published for “${decoded}” yet. Publish one with the CLI heatmap command; it appears here automatically.`,
            ),
          );
        } else {
          container.replaceChildren(
            heatmapEmpty(`The heatmap document could not be loaded (HTTP ${response.status}).`),
          );
        }
      },
      () => {
        container.replaceChildren(
          heatmapEmpty("The server did not respond while loading the heatmap document."),
        );
      },
    );
    render(
      el(null, null, [
        el("h1", { text: `Heatmap — ${decoded}` }),
        el("div", { class: "toolbar" }, [
          el("button", {
            class: "action",
            type: "button",
            onclick: () => {
              location.hash = "#/heatmap";
            },
            text: "← Choose another run",
          }),
        ]),
        container,
      ]),
    );
  }

  async function heatmapPicker() {
    try {
      const runs = await fetchJson("/api/runs");
      if (runs.count === 0) {
        return el("section", { class: "card" }, [
          emptyState("Register and complete a run first — heatmaps are rendered per run."),
        ]);
      }
      return el("section", { class: "card" }, [
        el("h2", { text: "Choose a run" }),
        buildRunsTable(runs.entries),
      ]);
    } catch (err) {
      return errorBanner(err);
    }
  }

  function heatmapEmpty(message) {
    return el("section", { class: "card" }, [emptyState(message)]);
  }

  /* ---------- router ---------- */

  function route() {
    const raw = location.hash || "#/workspace";
    const queryIndex = raw.indexOf("?");
    const path = queryIndex === -1 ? raw : raw.slice(0, queryIndex);
    const params = new URLSearchParams(queryIndex === -1 ? "" : raw.slice(queryIndex + 1));

    let match;
    if ((match = path.match(/^#\/runs\/([^/]+)$/))) {
      setNav("runs");
      showRunDetail(match[1]);
    } else if (path === "#/runs" || path.startsWith("#/runs")) {
      setNav("runs");
      showRuns(params.get("state"));
    } else if ((match = path.match(/^#\/heatmap(?:\/([^/]+))?$/))) {
      setNav("heatmap");
      showHeatmap(match ? match[1] : null);
    } else if (path === "#/workspace") {
      setNav("workspace");
      showWorkspace();
    } else {
      setNav("workspace");
      showWorkspace();
    }
  }

  function setNav(name) {
    for (const link of document.querySelectorAll("[data-nav]")) {
      if (link.dataset.nav === name) {
        link.setAttribute("aria-current", "page");
      } else {
        link.removeAttribute("aria-current");
      }
    }
  }

  /* ---------- theme toggle ---------- */

  function initTheme() {
    const root = document.documentElement;
    const stored = localStorage.getItem("moeatlas-theme");
    if (stored === "light" || stored === "dark") {
      root.dataset.theme = stored;
    }
    document.getElementById("theme-toggle").addEventListener("click", () => {
      const prefersDark =
        root.dataset.theme !== "light" &&
        (root.dataset.theme === "dark" ||
          window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = prefersDark ? "light" : "dark";
      localStorage.setItem("moeatlas-theme", root.dataset.theme);
    });
  }

  initTheme();
  window.addEventListener("hashchange", route);
  refreshHealth();
  route();
})();
