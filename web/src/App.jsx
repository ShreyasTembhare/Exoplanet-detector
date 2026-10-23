import { useCallback, useEffect, useRef, useState } from "react";
import { AuthError, api, authed, getToken, setToken } from "./api.js";
import { Bars, Chart } from "./charts.jsx";
import {
  INBOX_DEFAULTS, demotionText, fmt, fmtDuration, formatHash, inboxQuery, mergeLines, parseHash, statusLabel,
} from "./lib.js";

const STEPS = [
  { id: "live", title: "Live" },
  { id: "queue", title: "Queue" },
  { id: "inbox", title: "Inbox" },
  { id: "review", title: "Review" },
  { id: "stats", title: "Throughput" },
];

const FLAG_LABEL = {
  odd_even: "Odd/even",
  secondary: "Secondary",
  centroid: "Centroid",
  n_transits: "n transits",
};

const PAGE_SIZE = 50;

function useHashRoute() {
  const [route, setRoute] = useState(() => parseHash(window.location.hash));
  useEffect(() => {
    const onChange = () => setRoute(parseHash(window.location.hash));
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);
  const go = useCallback((view, options) => {
    const next = formatHash(view, options);
    if (window.location.hash !== next) window.location.hash = next;
    else setRoute(parseHash(next));
  }, []);
  return [route, go];
}

// Live log + progress over server-sent events, with a polling fallback.
function useLiveStream(enabled, onError) {
  const [lines, setLines] = useState([]);
  const [progress, setProgress] = useState(null);
  const [state, setState] = useState(null);
  const [connected, setConnected] = useState(false);
  const lastId = useRef(0);

  const refreshState = useCallback(async () => {
    const body = await api("/api/continuous?limit=300");
    setState(body);
    setProgress(body.progress || null);
    setLines((prev) => {
      const merged = mergeLines(prev, body.lines || []);
      if (merged.length) lastId.current = merged[merged.length - 1].id || lastId.current;
      return merged;
    });
    return body;
  }, []);

  useEffect(() => {
    if (!enabled) return undefined;
    let source = null;
    let poll = null;
    let retry = null;
    let closed = false;
    refreshState().catch(onError);

    const open = () => {
      if (closed) return;
      if (typeof window.EventSource !== "function") {
        poll = setInterval(() => refreshState().catch(onError), 3000);
        return;
      }
      source = new window.EventSource(authed(`/api/stream?after_id=${lastId.current}`));
      source.addEventListener("open", () => setConnected(true));
      source.addEventListener("lines", (event) => {
        const incoming = JSON.parse(event.data);
        setLines((prev) => mergeLines(prev, incoming));
        if (incoming.length) lastId.current = incoming[incoming.length - 1].id;
      });
      source.addEventListener("progress", (event) => setProgress(JSON.parse(event.data)));
      source.addEventListener("error", () => {
        setConnected(false);
        source.close();
        // Reopen from the newest id we have, so nothing is replayed.
        retry = setTimeout(open, 3000);
      });
    };
    open();
    const stateTimer = setInterval(() => refreshState().catch(() => {}), 15000);
    return () => {
      closed = true;
      if (source) source.close();
      clearInterval(poll);
      clearTimeout(retry);
      clearInterval(stateTimer);
    };
  }, [enabled, refreshState, onError]);

  const loadOlder = useCallback(async () => {
    const first = lines.length ? lines[0].id : null;
    if (!first) return 0;
    const body = await api(`/api/continuous/events?before_id=${first}&limit=200`);
    setLines((prev) => mergeLines(body.lines || [], prev, 5000));
    return (body.lines || []).length;
  }, [lines]);

  return { lines, progress, state, setState, connected, refreshState, loadOlder };
}

function TokenGate({ onSaved }) {
  const [value, setValue] = useState("");
  return (
    <section className="gate">
      <h1>Desk token</h1>
      <p className="lede">This desk is reachable over the network, so it asks for the token set in <code>DESK_TOKEN</code>.</p>
      <form
        className="form"
        onSubmit={(event) => {
          event.preventDefault();
          setToken(value.trim());
          onSaved();
        }}
      >
        <label className="wide">
          Token
          <input type="password" autoComplete="current-password" value={value} onChange={(e) => setValue(e.target.value)} />
        </label>
        <button type="submit" disabled={!value.trim()}>Unlock</button>
      </form>
    </section>
  );
}

function ProgressStrip({ progress, connected }) {
  const p = progress || {};
  const raw = (p.status || "idle").toLowerCase();
  return (
    <header className={`strip status-${raw}`}>
      <p className={`pill ${raw}`}>{statusLabel(raw)}</p>
      <div className="strip-metrics">
        <span><em>Target</em> {p.current_target || "—"}</span>
        <span><em>Stars</em> {p.stars_done ?? 0}</span>
        <span><em>Candidates</em> {p.candidates_found ?? 0}</span>
        {p.stars_per_hour > 0 ? <span><em>Rate</em> {fmt(p.stars_per_hour, 1)}/h</span> : null}
        {p.eta_seconds ? <span><em>ETA</em> {fmtDuration(p.eta_seconds)}</span> : null}
        <span className={connected ? "live-dot on" : "live-dot"} title={connected ? "Live stream connected" : "Reconnecting"}>
          {connected ? "live" : "offline"}
        </span>
      </div>
      <p className="strip-log" title={p.last_log_line || ""}>{p.last_log_line || p.message || "Waiting for the first log line."}</p>
    </header>
  );
}

export default function App() {
  const [route, go] = useHashRoute();
  const [health, setHealth] = useState(null);
  const [needToken, setNeedToken] = useState(false);
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [queue, setQueue] = useState(null);
  const [run, setRun] = useState(null);
  const [limit, setLimit] = useState(10);
  const [sector, setSector] = useState(1);
  const [sde, setSde] = useState(7);

  const report = useCallback((error) => {
    if (error instanceof AuthError) return;
    setNotice(error && error.message ? error.message : String(error));
  }, []);

  useEffect(() => {
    const onAuth = () => setNeedToken(true);
    window.addEventListener("desk-auth-required", onAuth);
    return () => window.removeEventListener("desk-auth-required", onAuth);
  }, []);

  const boot = useCallback(() => {
    api("/api/health")
      .then((body) => {
        setHealth(body);
        setNeedToken(Boolean(body.auth_required && !getToken()));
      })
      .catch(report);
  }, [report]);
  useEffect(boot, [boot]);

  const ready = Boolean(health) && !needToken;
  const live = useLiveStream(ready, report);

  useEffect(() => {
    if (!ready) return;
    api("/api/queue").then((body) => setQueue(body.queue)).catch(report);
    api("/api/runs/current").then((body) => setRun(body.run)).catch(report);
  }, [ready, report]);

  useEffect(() => {
    if (!ready || !run || run.status !== "running") return undefined;
    const timer = setInterval(() => {
      api("/api/runs/current").then((body) => setRun(body.run)).catch(report);
    }, 2000);
    return () => clearInterval(timer);
  }, [ready, run, report]);

  async function action(fn) {
    setNotice("");
    setBusy(true);
    try {
      return await fn();
    } catch (error) {
      report(error);
      return null;
    } finally {
      setBusy(false);
    }
  }

  const status = (live.progress && live.progress.status) || "idle";
  const candidateCount = (live.progress && live.progress.candidates_found) || 0;

  if (needToken) {
    return (
      <div className="shell single">
        <main>
          <TokenGate onSaved={() => { setNeedToken(false); boot(); }} />
        </main>
      </div>
    );
  }

  return (
    <div className="shell">
      <a className="skip" href="#main">Skip to content</a>
      <aside>
        <p className="brand">Transit desk</p>
        <p className="brand-sub">Always-on search</p>
        <nav aria-label="Sections">
          {STEPS.map((item) => (
            <a
              key={item.id}
              href={formatHash(item.id, item.id === "review" && route.id ? { id: route.id } : {})}
              className={route.view === item.id ? "step on" : "step"}
              aria-current={route.view === item.id ? "page" : undefined}
            >
              {item.title}
              {item.id === "live" && status === "running" ? <em className="pulse" aria-label="running">●</em> : null}
            </a>
          ))}
        </nav>
        <p className="rail-note">
          {health && health.checkpoint
            ? "Classifier checkpoint loaded."
            : "No classifier checkpoint. Candidates are ranked by detection strength and vetting."}
        </p>
        {health && health.mast === false ? <p className="rail-note warn">{health.mast_message}</p> : null}
        {getToken() ? (
          <button type="button" className="text-button" onClick={() => { setToken(""); setNeedToken(true); }}>
            Forget token
          </button>
        ) : null}
      </aside>
      <main id="main" tabIndex={-1}>
        <ProgressStrip progress={live.progress} connected={live.connected} />
        {notice ? (
          <div className="banner" role="alert">
            <span>{notice}</span>
            <button type="button" className="text-button" onClick={() => setNotice("")} aria-label="Dismiss">Dismiss</button>
          </div>
        ) : null}
        {route.view === "live" ? (
          <LiveView
            live={live} busy={busy} run={run} setRun={setRun} action={action}
            sector={sector} setSector={setSector} limit={limit} setLimit={setLimit} sde={sde} setSde={setSde}
          />
        ) : null}
        {route.view === "queue" ? (
          <QueueView queue={queue} setQueue={setQueue} limit={limit} setLimit={setLimit} busy={busy} action={action} />
        ) : null}
        {route.view === "inbox" ? (
          <InboxView
            params={route.params} go={go} report={report} showModel={Boolean(health && health.checkpoint)}
            refreshKey={candidateCount}
          />
        ) : null}
        {route.view === "review" ? <ReviewView id={route.id} report={report} go={go} /> : null}
        {route.view === "stats" ? <StatsView report={report} action={action} /> : null}
      </main>
    </div>
  );
}

function QueueView({ queue, setQueue, limit, setLimit, busy, action }) {
  const [tableText, setTableText] = useState("");
  const [tablePath, setTablePath] = useState("");
  async function onRank(event) {
    event.preventDefault();
    const body = await action(() => api("/api/queue/rank", {
      method: "POST",
      json: { csv: tableText, path: tablePath || null, limit: Number(limit), period_min: 0.5, period_max: 15 },
    }));
    if (body) setQueue(body);
  }
  return (
    <section>
      <h1>Queue</h1>
      <p className="lede">Rank who is next before any light curve is downloaded. Bare TIC lists are enriched for radius, CROWDSAP, and Tmag in one batched catalog query.</p>
      <form onSubmit={onRank} className="form">
        <label>
          Best N
          <input type="number" min="1" value={limit} onChange={(event) => setLimit(event.target.value)} />
        </label>
        <label className="wide">
          TIC table
          <textarea rows={4} value={tableText} onChange={(event) => setTableText(event.target.value)} placeholder="tic,radius,n_sectors,crowdsap,tmag,disposition" />
        </label>
        <label className="wide">
          File
          <input type="file" accept=".csv,.txt" onChange={async (event) => {
            const file = event.target.files && event.target.files[0];
            if (file) setTableText(await file.text());
          }} />
        </label>
        <label className="wide">
          Path on the desk machine
          <input value={tablePath} onChange={(event) => setTablePath(event.target.value)} placeholder="/path/targets.csv" />
        </label>
        <button type="submit" disabled={busy}>Rank</button>
      </form>
      {!queue || ((queue.primary || []).length === 0 && (queue.second_pass || []).length === 0) ? (
        <p className="empty">No target table yet. Rank a TIC file before download; the best N come first.</p>
      ) : (
        <>
          <p className="quiet">{queue.note}</p>
          <TargetTable title="Primary" rows={queue.primary} />
          <TargetTable title="Known hosts, masked second pass" rows={queue.second_pass} />
        </>
      )}
    </section>
  );
}

function TargetTable({ title, rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <div className="block">
      <h2>{title}</h2>
      <table>
        <thead>
          <tr><th scope="col">#</th><th scope="col">TIC</th><th scope="col">Why this star is next</th><th scope="col">SNR</th><th scope="col">Disposition</th></tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.tic_id}>
              <td className="num">{index + 1}</td>
              <td className="num">{row.tic_id}</td>
              <td>{row.why}</td>
              <td className="num">{fmt(row.expected_snr, 1)}</td>
              <td>{row.disposition || "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LiveView({ live, busy, run, setRun, action, sector, setSector, limit, setLimit, sde, setSde }) {
  const logRef = useRef(null);
  const stick = useRef(true);
  const [curveDir, setCurveDir] = useState("");
  const [curveText, setCurveText] = useState("");
  const [curvePath, setCurvePath] = useState("");
  const [ticId, setTicId] = useState("local");
  const [profile, setProfile] = useState("balanced");
  const { lines, progress, state } = live;
  const p = progress || {};
  const status = p.status || "idle";
  const work = p.work || {};
  const runBusy = busy || status === "running" || (run && run.status === "running");

  useEffect(() => {
    const node = logRef.current;
    if (node && stick.current) node.scrollTop = node.scrollHeight;
  }, [lines.length]);

  async function control(path, body) {
    const result = await action(() => api(path, { method: "POST", json: body }));
    if (result) {
      live.setState(result);
      live.refreshState().catch(() => {});
    }
  }

  async function startRun(mode) {
    const body = await action(() => api("/api/runs", {
      method: "POST",
      json: {
        mode, sector: Number(sector), limit: Number(limit), sde_threshold: Number(sde),
        period_min: 0.5, period_max: 15, nperiods: mode === "local" ? 600 : 2000,
        light_curve_csv: curveText, light_curve_path: curvePath || null, tic_id: ticId,
      },
    }));
    if (body) setRun(body.run);
  }

  return (
    <section className="live">
      <h1>Live run</h1>
      <p className="lede">
        The search keeps going when this page is closed and resumes from the same item after a restart.
        {state && state.auto_start === false ? " Auto-start is off until Start or Resume." : ""}
      </p>
      <dl className="live-metrics">
        <div><dt>Status</dt><dd className={status}>{statusLabel(status)}</dd></div>
        <div><dt>Target</dt><dd>{p.current_target || "—"}</dd></div>
        <div><dt>Sector</dt><dd className="num">{p.sector ?? "—"}</dd></div>
        <div><dt>Stars done</dt><dd className="num">{p.stars_done ?? 0}</dd></div>
        <div><dt>Candidates</dt><dd className="num">{p.candidates_found ?? 0}</dd></div>
        <div><dt>Stars / hour</dt><dd className="num">{fmt(p.stars_per_hour, 1)}</dd></div>
        <div><dt>Work done / left</dt><dd className="num">{work.done ?? 0} / {(work.pending ?? 0) + (work.leased ?? 0)}</dd></div>
        <div><dt>Failed items</dt><dd className={`num ${work.failed ? "bad" : ""}`}>{work.failed ?? 0}</dd></div>
      </dl>
      <div className="form live-controls">
        <label>
          Sector
          <input type="number" min="1" value={sector} onChange={(event) => setSector(event.target.value)} />
        </label>
        <label>
          Stars per sector
          <input type="number" min="1" value={limit} onChange={(event) => setLimit(event.target.value)} />
        </label>
        <label>
          SDE gate
          <input type="number" step="0.1" value={sde} onChange={(event) => setSde(event.target.value)} />
        </label>
        <label>
          Profile
          <select value={profile} onChange={(event) => setProfile(event.target.value)}>
            <option value="balanced">balanced</option>
            <option value="ultra_short_period">ultra short period</option>
            <option value="single_transit_long_period">single transit / long period</option>
            <option value="low_snr_m_dwarf">low SNR M dwarf</option>
          </select>
        </label>
        <label className="wide">
          Local curve directory (optional)
          <input value={curveDir} onChange={(event) => setCurveDir(event.target.value)} placeholder="/path/curves" />
        </label>
        <div className="actions">
          <button type="button" disabled={runBusy} onClick={() => control("/api/continuous/start", {
            local_dir: curveDir.trim() || null, sector: Number(sector), limit: Number(limit),
            sde_threshold: Number(sde), period_min: 0.5, period_max: 15,
            nperiods: curveDir.trim() ? 400 : 2000, strategy_profile: profile,
          })}>
            {status === "paused" || status === "stopped" ? "Resume" : "Start"}
          </button>
          <button type="button" disabled={status !== "running"} onClick={() => control("/api/continuous/pause")}>Pause</button>
          <button type="button" disabled={status !== "running" && status !== "paused"} onClick={() => control("/api/continuous/stop")}>Stop</button>
        </div>
      </div>
      <div className="log-head">
        <h2>Log</h2>
        <button type="button" className="text-button" onClick={() => live.loadOlder().catch(() => {})} disabled={!lines.length}>
          Load older
        </button>
      </div>
      <ol
        className="log"
        ref={logRef}
        onScroll={(event) => {
          const node = event.currentTarget;
          stick.current = node.scrollHeight - node.scrollTop - node.clientHeight < 24;
        }}
        tabIndex={0}
        aria-label="Search log"
      >
        {lines.length === 0 ? (
          <li className="quiet">The log is empty. When MAST is missing, the loop says so and invents no stars.</li>
        ) : lines.map((line) => (
          <li key={line.id ?? `${line.ts}-${line.message}`} className={`log-line ${line.error ? "error" : line.candidate ? "candidate" : ""}`}>
            <time dateTime={line.ts}>{line.ts}</time>
            <span>{line.message}</span>
          </li>
        ))}
      </ol>
      <p className="sr-only" aria-live="polite">{p.last_log_line || ""}</p>
      <details className="one-off">
        <summary>One-off sector or local light curve</summary>
        <div className="form">
          <button type="button" disabled={runBusy} onClick={() => startRun("sector")}>Search sector {sector} once</button>
        </div>
        <RunState run={run} />
        <div className="split">
          <h2>Local light curve</h2>
          <div className="form">
            <label>
              TIC label
              <input value={ticId} onChange={(event) => setTicId(event.target.value)} />
            </label>
            <label className="wide">
              Curve
              <textarea rows={4} value={curveText} onChange={(event) => setCurveText(event.target.value)} placeholder="time,flux" />
            </label>
            <label className="wide">
              File
              <input type="file" accept=".csv,.txt" onChange={async (event) => {
                const file = event.target.files && event.target.files[0];
                if (file) setCurveText(await file.text());
              }} />
            </label>
            <label className="wide">
              Path on the desk machine
              <input value={curvePath} onChange={(event) => setCurvePath(event.target.value)} placeholder="/path/curve.csv" />
            </label>
            <button type="button" disabled={runBusy} onClick={() => startRun("local")}>Search this curve</button>
          </div>
        </div>
      </details>
    </section>
  );
}

function RunState({ run }) {
  if (!run) return <p className="empty">No one-off search has been started.</p>;
  if (run.status === "running") return <p className="state running">Running. {run.message}</p>;
  if (run.status === "failed") {
    return (
      <div className="state failed" role="status">
        <p className="state-title">{run.needs_mast ? "MAST required" : "Search failed"}</p>
        <p>{run.error}</p>
      </div>
    );
  }
  return <p className="state done">Finished. {run.message}</p>;
}

function InboxView({ params, go, report, showModel, refreshKey }) {
  const filters = { ...INBOX_DEFAULTS, ...params };
  const [data, setData] = useState({ candidates: [], total: 0 });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const query = inboxQuery(filters, PAGE_SIZE);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api(`/api/candidates?${query}`)
      .then((body) => {
        if (!cancelled) {
          setData(body);
          setError("");
        }
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
        report(err);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [query, refreshKey, report]);

  const set = (patch) => go("inbox", { params: { ...filters, page: "0", ...patch } });
  const page = Number(filters.page) || 0;
  const pages = Math.max(1, Math.ceil(data.total / PAGE_SIZE));

  return (
    <section>
      <h1>Inbox</h1>
      <p className="lede">Sorted by score: detection strength (and the classifier when loaded) times graded vetting penalties. Known planets are labelled, not hidden.</p>
      <form className="filters" onSubmit={(event) => event.preventDefault()} role="search">
        <label>
          Search
          <input value={filters.q} onChange={(event) => set({ q: event.target.value })} placeholder="TIC id" />
        </label>
        <label>
          Label
          <select value={filters.label} onChange={(event) => set({ label: event.target.value })}>
            <option value="">All</option>
            <option value="new">New</option>
            <option value="new_on_known_host">New on known host</option>
            <option value="known">Known</option>
            <option value="unreviewed">Unreviewed</option>
            <option value="planet">Reviewed: planet</option>
            <option value="false_positive">Reviewed: false positive</option>
            <option value="unsure">Reviewed: unsure</option>
          </select>
        </label>
        <label>
          Flags
          <select value={filters.flags} onChange={(event) => set({ flags: event.target.value })}>
            <option value="">Any</option>
            <option value="none">No hard flags</option>
            <option value="any">Has a flag</option>
          </select>
        </label>
        <label>
          Min SDE
          <input type="number" step="0.5" value={filters.min_sde} onChange={(event) => set({ min_sde: event.target.value })} />
        </label>
        <label>
          Sort
          <select value={filters.sort} onChange={(event) => set({ sort: event.target.value })}>
            <option value="score">Score</option>
            <option value="sde">SDE</option>
            <option value="period_days">Period</option>
            <option value="depth_ppm">Depth</option>
            <option value="updated">Newest</option>
          </select>
        </label>
        <label>
          Order
          <select value={filters.order} onChange={(event) => set({ order: event.target.value })}>
            <option value="desc">High first</option>
            <option value="asc">Low first</option>
          </select>
        </label>
      </form>
      {error ? <p className="state failed" role="alert">Could not load candidates: {error}</p> : null}
      {!error && !loading && data.total === 0 ? (
        <p className="empty">No candidates match. Nothing has cleared the SDE gate yet, or the filters are too tight.</p>
      ) : null}
      {data.candidates.length ? (
        <table aria-busy={loading}>
          <thead>
            <tr>
              <th scope="col">TIC</th><th scope="col">Score</th><th scope="col">Period</th><th scope="col">Depth</th>
              <th scope="col">SDE</th>{showModel ? <th scope="col">Model</th> : null}<th scope="col">Flags</th>
              <th scope="col">Label</th><th scope="col">Review</th>
            </tr>
          </thead>
          <tbody>
            {data.candidates.map((row) => (
              <tr key={row.id} className={row.hard_flags.length === 0 ? "hit" : ""}>
                <td><a className="link" href={formatHash("review", { id: row.id })}>{row.tic_id}</a></td>
                <td className="num">{fmt(row.score, 3)}</td>
                <td className="num">{row.kind === "monotransit" ? "single" : `${fmt(row.period_days, 4)} d`}</td>
                <td className="num">{row.depth_ppm == null ? "—" : `${fmt(row.depth_ppm, 0)} ppm`}</td>
                <td className="num">{fmt(row.sde, 1)}</td>
                {showModel ? <td className="num">{fmt(row.prob_planet, 3)}</td> : null}
                <td>{row.hard_flags.length ? row.hard_flags.map((flag) => FLAG_LABEL[flag] || flag).join(", ") : "None"}</td>
                <td>{row.label}</td>
                <td>{row.review_label ? row.review_label.replace("_", " ") : "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {pages > 1 ? (
        <nav className="pager" aria-label="Pages">
          <button type="button" disabled={page <= 0} onClick={() => go("inbox", { params: { ...filters, page: String(page - 1) } })}>Previous</button>
          <span className="num">Page {page + 1} of {pages} · {data.total} candidates</span>
          <button type="button" disabled={page + 1 >= pages} onClick={() => go("inbox", { params: { ...filters, page: String(page + 1) } })}>Next</button>
        </nav>
      ) : null}
    </section>
  );
}

function ReviewView({ id, report, go }) {
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [exportResult, setExportResult] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!id) return undefined;
    let cancelled = false;
    setDetail(null);
    setError("");
    setExportResult(null);
    api(`/api/candidates/${encodeURIComponent(id)}`)
      .then((body) => {
        if (cancelled) return;
        setDetail(body);
        setNote((body.review && body.review.note) || "");
      })
      .catch((err) => !cancelled && setError(err.message));
    return () => { cancelled = true; };
  }, [id]);

  if (!id) {
    return (
      <section>
        <h1>Review</h1>
        <p className="empty">Open a candidate from the <a className="link" href="#/inbox">inbox</a>. Each candidate has its own link you can share.</p>
      </section>
    );
  }
  if (error) {
    return (
      <section>
        <h1>Review</h1>
        <p className="state failed" role="alert">{error}</p>
        <button type="button" className="text-button" onClick={() => go("inbox")}>Back to inbox</button>
      </section>
    );
  }
  if (!detail) return <section><h1>Review</h1><p className="quiet">Loading…</p></section>;

  const v = detail.vetting || {};
  const flags = new Set(detail.hard_flags || []);
  const oe = detail.odd_even_curve || {};
  const lc = detail.light_curve || {};
  const pg = detail.periodogram || {};

  async function label(value) {
    setSaving(true);
    try {
      await api(`/api/candidates/${encodeURIComponent(detail.id)}/label`, { method: "POST", json: { label: value, note } });
      setDetail({ ...detail, review: { label: value, note } });
    } catch (err) {
      report(err);
    } finally {
      setSaving(false);
    }
  }

  async function onExport() {
    try {
      setExportResult(await api(`/api/candidates/${encodeURIComponent(detail.id)}/export`, { method: "POST" }));
    } catch (err) {
      setExportResult(null);
      report(err);
    }
  }

  return (
    <section>
      <p className="crumb"><a className="link" href="#/inbox">Inbox</a> / TIC {detail.tic_id}</p>
      <h1>TIC {detail.tic_id}</h1>
      <p className="lede">
        {detail.label}
        {detail.kind === "monotransit" ? " · single transit" : ""}
        {detail.known_match ? ` · matches TOI ${detail.known_match.toi || ""}` : ""}
        {detail.prob_planet != null ? ` · model ${fmt(detail.prob_planet, 3)}` : " · no classifier score"}
      </p>
      <dl className="metrics">
        <div><dt>Score</dt><dd>{fmt(detail.score, 3)}</dd></div>
        <div><dt>Period</dt><dd>{detail.kind === "monotransit" ? "—" : `${fmt(detail.period_days, 5)} d`}</dd></div>
        <div><dt>Epoch (BTJD)</dt><dd>{fmt(detail.epoch_btjd, 4)}</dd></div>
        <div><dt>Depth</dt><dd>{detail.depth_ppm == null ? "—" : `${fmt(detail.depth_ppm, 0)} ppm`}</dd></div>
        <div><dt>Duration</dt><dd>{fmt(detail.duration_hours, 2)} h</dd></div>
        <div><dt>SDE</dt><dd>{fmt(detail.sde, 2)}</dd></div>
        <div><dt>Transits</dt><dd>{detail.n_transits ?? "—"}</dd></div>
        <div><dt>FPP</dt><dd>{fmt(detail.fpp, 3)}</dd></div>
      </dl>

      <h2>Light curve</h2>
      <Chart
        label="Detrended light curve with predicted transit times"
        series={[{ x: lc.time || [], y: lc.flux || [], kind: "points" }]}
        markers={lc.transits || []}
        xLabel="BTJD" yLabel="flux"
      />
      <h2>BLS periodogram</h2>
      <Chart
        label="BLS periodogram"
        series={[{ x: pg.period || [], y: pg.power || [], kind: "line" }]}
        markers={detail.period_days ? [detail.period_days] : []}
        xLabel="period (d)" yLabel="power" logX
      />
      <h2>Phase fold</h2>
      <Chart
        label="Phase-folded light curve"
        series={[{ x: (detail.phase_curve || {}).phase || [], y: (detail.phase_curve || {}).flux || [], kind: "line" }]}
        markers={[0]} xLabel="phase" yLabel="flux"
      />
      <div className="pair">
        <div>
          <h2>Odd vs even transits</h2>
          <Chart
            label="Odd and even transits overlaid"
            series={[
              { x: (oe.odd || {}).hours || [], y: (oe.odd || {}).flux || [], kind: "line", tone: "ember", label: "odd" },
              { x: (oe.even || {}).hours || [], y: (oe.even || {}).flux || [], kind: "line", tone: "moss", label: "even" },
            ]}
            xLabel="hours from mid-transit" yLabel="flux" height={180}
          />
        </div>
        <div>
          <h2>Secondary eclipse window</h2>
          <Chart
            label="Flux around phase 0.5"
            series={[{ x: (detail.secondary_curve || {}).hours || [], y: (detail.secondary_curve || {}).flux || [], kind: "line" }]}
            markers={[0]} xLabel="hours from phase 0.5" yLabel="flux" height={180}
          />
        </div>
      </div>
      {detail.has_plot ? <img className="plot" alt="Pipeline phase plot" src={authed(`/api/candidates/${encodeURIComponent(detail.id)}/plot`)} /> : null}
      {detail.has_difference_image ? (
        <>
          <h2>Difference image</h2>
          <img className="plot" alt="Difference image" src={authed(`/api/candidates/${encodeURIComponent(detail.id)}/difference-image`)} />
        </>
      ) : null}

      <h2>Vetting</h2>
      <table>
        <tbody>
          <FlagRow name="Odd/even depth ratio" on={flags.has("odd_even")} value={fmt(detail.odd_even_ratio, 2)} />
          <FlagRow name="Secondary significance (σ)" on={flags.has("secondary")} value={fmt(detail.secondary_significance, 2)} />
          <FlagRow name="Centroid offset (px)" on={flags.has("centroid")} value={fmt(detail.centroid_offset_px, 2)} />
          <FlagRow name="Transits observed" on={flags.has("n_transits")} value={detail.n_transits ?? "—"} />
          <FlagRow name="Gaia neighbour flux ratio" on={Number(v.gaia_flux_ratio) > 0.5} value={fmt(v.gaia_flux_ratio, 2)} />
          <FlagRow name="Gaia neighbours within 30″" on={false} value={v.gaia_neighbors_within_30arcsec ?? "—"} />
          <FlagRow name="TRICERATOPS FPP / NFPP" on={Number(v.fpp) > 0.5} value={`${fmt(v.fpp, 3)} / ${fmt(v.nfpp, 4)}`} />
        </tbody>
      </table>
      {detail.demotion && detail.demotion.reasons && detail.demotion.reasons.length ? (
        <p className="quiet">Score multiplied by {fmt(detail.demotion.factor, 2)}: {demotionText(detail.demotion)}.</p>
      ) : null}

      <h2>Your verdict</h2>
      <div className="verdict">
        <label className="wide">
          Note
          <input value={note} onChange={(event) => setNote(event.target.value)} placeholder="Why (optional)" />
        </label>
        <div className="actions" role="group" aria-label="Label this candidate">
          {[
            ["planet", "Planet"],
            ["false_positive", "False positive"],
            ["unsure", "Unsure"],
          ].map(([value, text]) => (
            <button
              key={value}
              type="button"
              disabled={saving}
              aria-pressed={detail.review && detail.review.label === value}
              className={detail.review && detail.review.label === value ? "chosen" : ""}
              onClick={() => label(value)}
            >
              {text}
            </button>
          ))}
        </div>
        <p className="quiet">Labels are saved in the desk state and used to fine-tune the classifier (<code>python train_classifier.py --labels</code>).</p>
      </div>

      <div className="export">
        {detail.epoch_btjd == null ? (
          <p className="state failed">No epoch on this candidate. Export is refused so the packet cannot say t0 = 0.</p>
        ) : (
          <button type="button" onClick={onExport}>Export TFOP packet</button>
        )}
        {exportResult ? (
          <p className="state done">
            Wrote epoch {fmt(exportResult.epoch_btjd, 4)} BTJD.{" "}
            <a className="link" href={authed(exportResult.download_url)} download>Download JSON</a>
          </p>
        ) : null}
      </div>
    </section>
  );
}

function FlagRow({ name, on, value }) {
  return (
    <tr>
      <th scope="row">{name}</th>
      <td className="num">{value}</td>
      <td>{on ? <span className="flag">Flag</span> : <span className="quiet">Clear</span>}</td>
    </tr>
  );
}

function StatsView({ report, action }) {
  const [stats, setStats] = useState(null);
  const [error, setError] = useState("");
  const load = useCallback(() => {
    api("/api/stats").then((body) => { setStats(body); setError(""); }).catch((err) => { setError(err.message); report(err); });
  }, [report]);
  useEffect(() => {
    load();
    const timer = setInterval(load, 20000);
    return () => clearInterval(timer);
  }, [load]);

  if (error && !stats) return <section><h1>Throughput</h1><p className="state failed" role="alert">{error}</p></section>;
  if (!stats) return <section><h1>Throughput</h1><p className="quiet">Loading…</p></section>;
  const work = stats.work || {};
  const p = stats.progress || {};

  async function retry(itemId) {
    const ok = await action(() => api(itemId ? `/api/work/${itemId}/retry` : "/api/work/retry-failed", { method: "POST" }));
    if (ok) load();
  }

  return (
    <section>
      <h1>Throughput</h1>
      <dl className="live-metrics">
        <div><dt>Stars / hour</dt><dd className="num">{fmt(p.stars_per_hour, 1)}</dd></div>
        <div><dt>Running for</dt><dd className="num">{fmtDuration(p.elapsed_seconds)}</dd></div>
        <div><dt>ETA (plan)</dt><dd className="num">{fmtDuration(p.eta_seconds)}</dd></div>
        <div><dt>Reviewed</dt><dd className="num">{stats.reviewed}</dd></div>
        <div><dt>Work pending</dt><dd className="num">{work.pending ?? 0}</dd></div>
        <div><dt>In flight</dt><dd className="num">{work.leased ?? 0}</dd></div>
        <div><dt>Done</dt><dd className="num">{work.done ?? 0}</dd></div>
        <div><dt>Failed</dt><dd className={`num ${work.failed ? "bad" : ""}`}>{work.failed ?? 0}</dd></div>
      </dl>
      <h2>Stars searched per hour (24 h)</h2>
      <Bars rows={stats.hourly} valueKey="stars" label="Stars searched per hour over the last 24 hours" />
      <h2>Candidates per hour</h2>
      <Bars rows={stats.hourly} valueKey="candidates" tone="moss" label="Candidates per hour over the last 24 hours" />
      <h2>Inbox by label</h2>
      <p className="num">
        {Object.keys(stats.candidates_by_label || {}).length
          ? Object.entries(stats.candidates_by_label).map(([k, n]) => `${k}: ${n}`).join(" · ")
          : "No candidates yet."}
      </p>
      <div className="log-head">
        <h2>Failed work items</h2>
        {stats.failed_items && stats.failed_items.length ? (
          <button type="button" className="text-button" onClick={() => retry(null)}>Retry all</button>
        ) : null}
      </div>
      {stats.failed_items && stats.failed_items.length ? (
        <table>
          <thead><tr><th scope="col">Item</th><th scope="col">Attempts</th><th scope="col">Last error</th><th scope="col" /></tr></thead>
          <tbody>
            {stats.failed_items.map((item) => (
              <tr key={item.id}>
                <td>{item.target}{item.sector ? ` (sector ${item.sector})` : ""}</td>
                <td className="num">{item.attempts}/{item.max_attempts}</td>
                <td className="error-text">{item.last_error || "—"}</td>
                <td><button type="button" className="text-button" onClick={() => retry(item.id)}>Retry</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="quiet">No failed items. Errors are retried automatically up to three times.</p>}
    </section>
  );
}
