import { useCallback, useEffect, useRef, useState } from "react";

const STEPS = [
  { id: "queue", n: "01", title: "Queue" },
  { id: "search", n: "02", title: "Search" },
  { id: "inbox", n: "03", title: "Inbox" },
  { id: "review", n: "04", title: "Review" },
];

const FLAG_LABEL = {
  odd_even: "Odd/even",
  secondary: "Secondary",
  centroid: "Centroid",
  n_transits: "n transits",
};

function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

async function api(path, options) {
  const response = await fetch(path, options);
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = { detail: text.slice(0, 240) };
    }
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : response.statusText;
    throw new Error(typeof detail === "string" ? detail : "Request failed");
  }
  return body;
}

function PhaseChart({ curve }) {
  const phase = (curve && curve.phase) || [];
  const flux = (curve && curve.flux) || [];
  if (phase.length < 2) {
    return <p className="quiet">No phase curve was saved with this candidate.</p>;
  }
  const width = 720;
  const height = 220;
  const pad = { l: 44, r: 12, t: 12, b: 28 };
  const min = Math.min(...flux);
  const max = Math.max(...flux);
  const span = Math.max(max - min, 1e-6);
  const x = (p) => pad.l + p * (width - pad.l - pad.r);
  const y = (f) => pad.t + ((max - f) / span) * (height - pad.t - pad.b);
  const d = phase.map((p, i) => `${i === 0 ? "M" : "L"}${x(p).toFixed(1)},${y(flux[i]).toFixed(1)}`).join(" ");
  return (
    <svg className="fold" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Phase-folded light curve">
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} />
      <path d={d} />
      <text x={pad.l} y={height - 8}>0</text>
      <text x={width - pad.r - 8} y={height - 8}>1</text>
      <text x={8} y={pad.t + 8}>flux</text>
    </svg>
  );
}

export default function App() {
  const [step, setStep] = useState("queue");
  const [health, setHealth] = useState(null);
  const [queue, setQueue] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [run, setRun] = useState(null);
  const [limit, setLimit] = useState(10);
  const [tableText, setTableText] = useState("");
  const [tablePath, setTablePath] = useState("");
  const [sector, setSector] = useState(15);
  const [sde, setSde] = useState(7);
  const [curveText, setCurveText] = useState("");
  const [curvePath, setCurvePath] = useState("");
  const [ticId, setTicId] = useState("local");
  const [notice, setNotice] = useState("");
  const [exportResult, setExportResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [continuous, setContinuous] = useState(null);
  const [curveDir, setCurveDir] = useState("");

  const refreshCandidates = useCallback(async () => {
    const body = await api("/api/candidates");
    setCandidates(body.candidates);
    return body.candidates;
  }, []);

  useEffect(() => {
    api("/api/health").then(setHealth).catch((error) => setNotice(error.message));
    api("/api/queue").then((body) => setQueue(body.queue)).catch(() => {});
    api("/api/runs/current").then((body) => setRun(body.run)).catch(() => {});
    api("/api/continuous").then(setContinuous).catch(() => {});
    refreshCandidates().catch((error) => setNotice(error.message));
  }, [refreshCandidates]);

  const continuousStatus = continuous ? continuous.status : "idle";

  useEffect(() => {
    if (continuousStatus !== "running" && continuousStatus !== "paused") return undefined;
    const timer = setInterval(() => {
      api("/api/continuous")
        .then((body) => {
          setContinuous(body);
          refreshCandidates().catch(() => {});
        })
        .catch((error) => setNotice(error.message));
    }, 800);
    return () => clearInterval(timer);
  }, [continuousStatus, refreshCandidates]);

  useEffect(() => {
    if (!run || run.status !== "running") return undefined;
    const timer = setInterval(() => {
      api("/api/runs/current")
        .then((body) => {
          setRun(body.run);
          if (body.run && body.run.status !== "running") refreshCandidates();
        })
        .catch((error) => setNotice(error.message));
    }, 800);
    return () => clearInterval(timer);
  }, [run, refreshCandidates]);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return undefined;
    }
    let cancelled = false;
    api(`/api/candidates/${selected}`)
      .then((body) => {
        if (!cancelled) setDetail(body);
      })
      .catch((error) => setNotice(error.message));
    return () => {
      cancelled = true;
    };
  }, [selected]);

  async function onRank(event) {
    event.preventDefault();
    setNotice("");
    setBusy(true);
    try {
      const body = await api("/api/queue/rank", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          csv: tableText,
          path: tablePath || null,
          limit: Number(limit),
          period_min: 0.5,
          period_max: 15,
        }),
      });
      setQueue(body);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function onFile(event, setter) {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    setter(await file.text());
  }

  async function start(mode) {
    setNotice("");
    setBusy(true);
    try {
      const payload = {
        mode,
        sector: Number(sector),
        limit: Number(limit),
        sde_threshold: Number(sde),
        period_min: 0.5,
        period_max: 15,
        nperiods: mode === "local" ? 600 : 2000,
        tic_csv: tableText,
        tic_path: tablePath || null,
        light_curve_csv: curveText,
        light_curve_path: curvePath || null,
        tic_id: ticId,
      };
      const body = await api("/api/runs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setRun(body.run);
      setStep("search");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function keepSearching() {
    setNotice("");
    setBusy(true);
    try {
      const body = await api("/api/continuous/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          local_dir: curveDir.trim() || null,
          sector: Number(sector),
          limit: Number(limit),
          sde_threshold: Number(sde),
          period_min: 0.5,
          period_max: 15,
          nperiods: curveDir.trim() ? 400 : 2000,
        }),
      });
      setContinuous(body);
      setStep("search");
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function pauseSearching() {
    setNotice("");
    try {
      setContinuous(await api("/api/continuous/pause", { method: "POST" }));
    } catch (error) {
      setNotice(error.message);
    }
  }

  async function stopSearching() {
    setNotice("");
    try {
      setContinuous(await api("/api/continuous/stop", { method: "POST" }));
      refreshCandidates().catch(() => {});
    } catch (error) {
      setNotice(error.message);
    }
  }

  function openCandidate(id) {
    setSelected(id);
    setExportResult(null);
    setStep("review");
  }

  async function onExport() {
    if (!selected) return;
    setNotice("");
    try {
      const body = await api(`/api/candidates/${selected}/export`, { method: "POST" });
      setExportResult(body);
    } catch (error) {
      setExportResult(null);
      setNotice(error.message);
    }
  }

  const checkpoint = health && health.checkpoint;
  const keepLive = continuousStatus === "running" || continuousStatus === "paused";
  const runStatus = keepLive ? continuousStatus : (run ? run.status : "idle");

  return (
    <div className="shell">
      <aside>
        <p className="brand">Transit desk</p>
        <p className="brand-sub">TESS search</p>
        <nav>
          {STEPS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={step === item.id ? "step on" : "step"}
              onClick={() => setStep(item.id)}
            >
              <span>{item.n}</span>
              {item.title}
              {item.id === "inbox" ? <em>{candidates.length}</em> : null}
            </button>
          ))}
        </nav>
        <p className="rail-note">
          {checkpoint ? "Classifier checkpoint loaded." : "No classifier checkpoint. An SDE pass is kept as a candidate."}
        </p>
        <details>
          <summary>Command line</summary>
          <p>
            Training, autopilot, and SSL pretrain are not on this desk.
            They stay on <code>python run.py</code>.
          </p>
        </details>
      </aside>
      <main>
        <header className="top">
          <p className={`pill ${runStatus}`}>{runStatus === "idle" ? "Idle" : runStatus}</p>
          <p className="top-note">
            {keepLive && continuous.message
              ? continuous.message
              : run && run.message
                ? run.message
                : "Rank a table, then keep searching. Nothing is added without an SDE pass."}
          </p>
        </header>
        {notice ? <p className="banner">{notice}</p> : null}
        {step === "queue" ? (
          <QueueView
            queue={queue}
            limit={limit}
            setLimit={setLimit}
            tableText={tableText}
            setTableText={setTableText}
            tablePath={tablePath}
            setTablePath={setTablePath}
            onFile={onFile}
            onRank={onRank}
            busy={busy}
          />
        ) : null}
        {step === "search" ? (
          <SearchView
            sector={sector}
            setSector={setSector}
            limit={limit}
            setLimit={setLimit}
            sde={sde}
            setSde={setSde}
            curveText={curveText}
            setCurveText={setCurveText}
            curvePath={curvePath}
            setCurvePath={setCurvePath}
            ticId={ticId}
            setTicId={setTicId}
            onFile={onFile}
            start={start}
            run={run}
            busy={busy}
            curveDir={curveDir}
            setCurveDir={setCurveDir}
            continuous={continuous}
            keepSearching={keepSearching}
            pauseSearching={pauseSearching}
            stopSearching={stopSearching}
          />
        ) : null}
        {step === "inbox" ? <InboxView candidates={candidates} onOpen={openCandidate} /> : null}
        {step === "review" ? (
          <ReviewView
            detail={detail}
            onExport={onExport}
            exportResult={exportResult}
            notice={notice}
          />
        ) : null}
      </main>
    </div>
  );
}

function QueueView({ queue, limit, setLimit, tableText, setTableText, tablePath, setTablePath, onFile, onRank, busy }) {
  return (
    <section>
      <h1>Queue</h1>
      <p className="lede">Who is next, before any light curve is downloaded.</p>
      <form onSubmit={onRank} className="form">
        <label>
          Best N
          <input type="number" min="1" value={limit} onChange={(event) => setLimit(event.target.value)} />
        </label>
        <label className="wide">
          TIC table
          <textarea
            rows={4}
            value={tableText}
            onChange={(event) => setTableText(event.target.value)}
            placeholder="tic,radius,n_sectors,crowdsap,tmag,disposition"
          />
        </label>
        <label className="wide">
          File
          <input type="file" accept=".csv,.txt" onChange={(event) => onFile(event, setTableText)} />
        </label>
        <label className="wide">
          Path on this machine
          <input value={tablePath} onChange={(event) => setTablePath(event.target.value)} placeholder="/path/targets.csv" />
        </label>
        <button type="submit" disabled={busy}>Rank</button>
      </form>
      {!queue || ((queue.primary || []).length === 0 && (queue.second_pass || []).length === 0) ? (
        <p className="empty">No target table yet. Rank a TIC file before download. The best N come first. Sky position is not a factor.</p>
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
          <tr>
            <th>#</th>
            <th>TIC</th>
            <th>Why this star is next</th>
            <th>SNR</th>
            <th>Disposition</th>
          </tr>
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

function SearchView({
  sector, setSector, limit, setLimit, sde, setSde,
  curveText, setCurveText, curvePath, setCurvePath, ticId, setTicId,
  onFile, start, run, busy,
  curveDir, setCurveDir, continuous, keepSearching, pauseSearching, stopSearching,
}) {
  const logRef = useRef(null);
  const lines = (continuous && continuous.lines) || [];
  const status = continuous ? continuous.status : "idle";
  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [lines.length]);
  return (
    <section>
      <h1>Search</h1>
      <p className="lede">Keep searching stays on the server. Closing this page does not stop it.</p>
      <div className="form">
        <label>
          Sector
          <input type="number" value={sector} onChange={(event) => setSector(event.target.value)} />
        </label>
        <label>
          Best N
          <input type="number" min="1" value={limit} onChange={(event) => setLimit(event.target.value)} />
        </label>
        <label>
          SDE gate
          <input type="number" step="0.1" value={sde} onChange={(event) => setSde(event.target.value)} />
        </label>
        <label className="wide">
          Local curve directory
          <input
            value={curveDir}
            onChange={(event) => setCurveDir(event.target.value)}
            placeholder="/path/curves"
          />
        </label>
        <div className="actions">
          <button type="button" disabled={busy || status === "running" || (run && run.status === "running")} onClick={keepSearching}>
            {status === "paused" ? "Resume" : "Keep searching"}
          </button>
          <button type="button" disabled={status !== "running"} onClick={pauseSearching}>Pause</button>
          <button type="button" disabled={status !== "running" && status !== "paused"} onClick={stopSearching}>Stop</button>
        </div>
      </div>
      <p className="quiet">
        A directory of time, flux CSVs is searched as given. Otherwise a ranked queue is walked, and if there is no queue the sectors advance. A missing MAST client is logged and no star is invented.
      </p>
      <ol className="log" ref={logRef} aria-live="polite">
        {lines.length === 0 ? (
          <li className="quiet">The log is empty.</li>
        ) : lines.map((line, index) => (
          <li key={`${line.ts}-${index}`} className={line.error ? "error" : line.candidate ? "candidate" : ""}>
            <time dateTime={line.ts}>{line.ts}</time>
            <span>{line.message}</span>
          </li>
        ))}
      </ol>
      <div className="split">
        <h2>One target</h2>
        <p className="quiet">A single sector uses MAST. One pasted curve uses the same BLS path without a download.</p>
      <div className="form">
        <button type="button" disabled={busy || status === "running" || (run && run.status === "running")} onClick={() => start("sector")}>
          Start sector search
        </button>
      </div>
      <RunState run={run} />
      <div className="split">
        <h2>Local light curve</h2>
        <p className="quiet">Time and flux. A candidate is written only if the SDE gate passes. This is not a MAST discovery.</p>
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
            <input type="file" accept=".csv,.txt" onChange={(event) => onFile(event, setCurveText)} />
          </label>
          <label className="wide">
            Path on this machine
            <input value={curvePath} onChange={(event) => setCurvePath(event.target.value)} placeholder="/path/curve.csv" />
          </label>
          <button type="button" disabled={busy || status === "running" || (run && run.status === "running")} onClick={() => start("local")}>
            Search this curve
          </button>
        </div>
      </div>
      </div>
    </section>
  );
}

function RunState({ run }) {
  if (!run) {
    return <p className="empty">No search has been started.</p>;
  }
  if (run.status === "running") {
    return <p className="state running">Running. {run.message}</p>;
  }
  if (run.status === "failed") {
    return (
      <div className="state failed">
        <p className="state-title">{run.needs_mast ? "MAST required" : "Search failed"}</p>
        <p>{run.error}</p>
      </div>
    );
  }
  return (
    <p className="state done">
      Finished. {run.message}
      {run.candidate_ids && run.candidate_ids.length === 0 ? " The inbox was not changed." : ""}
    </p>
  );
}

function InboxView({ candidates, onOpen }) {
  return (
    <section>
      <h1>Inbox</h1>
      <p className="lede">SDE-pass stars. Known TOI versus new is a label, not the order. Sorted by SDE.</p>
      {candidates.length === 0 ? (
        <p className="empty">No candidates. A star appears here only after an SDE pass. This list stays empty when a search fails or nothing clears the gate.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>TIC</th>
              <th>Period</th>
              <th>Epoch</th>
              <th>Depth</th>
              <th>SDE</th>
              <th>Flags</th>
              <th>Label</th>
            </tr>
          </thead>
          <tbody>
            {candidates.map((row) => (
              <tr key={row.id}>
                <td>
                  <button type="button" className="link" onClick={() => onOpen(row.id)}>{row.tic_id}</button>
                </td>
                <td className="num">{fmt(row.period_days, 4)} d</td>
                <td className="num">{fmt(row.epoch_btjd, 4)}</td>
                <td className="num">{row.depth_ppm === null ? "—" : `${fmt(row.depth_ppm, 0)} ppm`}</td>
                <td className="num">{fmt(row.sde, 1)}</td>
                <td>{row.hard_flags.length ? row.hard_flags.map((flag) => FLAG_LABEL[flag] || flag).join(", ") : "None"}</td>
                <td>{row.label}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

function ReviewView({ detail, onExport, exportResult }) {
  if (!detail) {
    return (
      <section>
        <h1>Review</h1>
        <p className="empty">Select a candidate from the inbox.</p>
      </section>
    );
  }
  const flags = new Set(detail.hard_flags || []);
  return (
    <section>
      <h1>TIC {detail.tic_id}</h1>
      <p className="lede">
        {detail.label}
        {detail.status === "CANDIDATE_SDE" ? " · SDE pass, no classifier score" : ""}
      </p>
      <dl className="metrics">
        <div><dt>Period</dt><dd>{fmt(detail.period_days, 4)} d</dd></div>
        <div><dt>Epoch</dt><dd>{fmt(detail.epoch_btjd, 4)}</dd></div>
        <div><dt>Depth</dt><dd>{detail.depth_ppm === null ? "—" : `${fmt(detail.depth_ppm, 0)} ppm`}</dd></div>
        <div><dt>SDE</dt><dd>{fmt(detail.sde, 2)}</dd></div>
      </dl>
      <h2>Phase fold</h2>
      <PhaseChart curve={detail.phase_curve} />
      {detail.has_plot ? (
        <img className="plot" alt="Pipeline phase plot" src={`/api/candidates/${detail.id}/plot`} />
      ) : null}
      {detail.has_difference_image ? (
        <>
          <h2>Difference image</h2>
          <img className="plot" alt="Difference image" src={`/api/candidates/${detail.id}/difference-image`} />
        </>
      ) : null}
      <h2>Hard flags</h2>
      <table>
        <tbody>
          <FlagRow name="Odd/even" on={flags.has("odd_even")} value={fmt(detail.odd_even_ratio, 2)} />
          <FlagRow name="Secondary" on={flags.has("secondary")} value={fmt(detail.secondary_significance, 2)} />
          <FlagRow name="Centroid" on={flags.has("centroid")} value={fmt(detail.centroid_offset_px, 2)} />
          <FlagRow name="n transits" on={flags.has("n_transits")} value={detail.n_transits ?? "—"} />
        </tbody>
      </table>
      <div className="export">
        {detail.epoch_btjd === null ? (
          <p className="state failed">No epoch on this candidate. Export is refused so the packet cannot say t0 = 0.</p>
        ) : (
          <button type="button" onClick={onExport}>Export TFOP packet</button>
        )}
        {exportResult ? (
          <p className="state done">Wrote epoch {fmt(exportResult.epoch_btjd, 4)} BTJD.</p>
        ) : null}
      </div>
    </section>
  );
}

function FlagRow({ name, on, value }) {
  return (
    <tr>
      <th>{name}</th>
      <td className="num">{value}</td>
      <td>{on ? <span className="flag">Flag</span> : <span className="quiet">Clear</span>}</td>
    </tr>
  );
}
