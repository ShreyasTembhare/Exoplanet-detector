import { extent, fmt } from "./lib.js";

const W = 720;
const H = 220;
const PAD = { l: 64, r: 12, t: 12, b: 34 };

function ticks([lo, hi], n = 4) {
  return Array.from({ length: n + 1 }, (_, i) => lo + ((hi - lo) * i) / n);
}

function tickDigits([lo, hi]) {
  const span = Math.abs(hi - lo);
  if (span === 0) return 2;
  return Math.min(6, Math.max(0, 2 - Math.floor(Math.log10(span))));
}

// series: [{ x: [], y: [], kind: "line" | "points", tone: "ink" | "ember" | "moss" | "clay", label }]
export function Chart({ series, markers = [], xLabel, yLabel, label, logX = false, height = H }) {
  const usable = (series || []).filter((s) => s && s.x && s.x.length > 1);
  if (usable.length === 0) return <p className="quiet">No data saved for this panel.</p>;
  const tx = (v) => (logX ? Math.log10(v) : v);
  const xs = usable.flatMap((s) => s.x.map(tx));
  const ys = usable.flatMap((s) => s.y);
  const xr = extent(xs);
  const yr = extent(ys);
  const px = (v) => PAD.l + ((tx(v) - xr[0]) / (xr[1] - xr[0])) * (W - PAD.l - PAD.r);
  const py = (v) => PAD.t + ((yr[1] - v) / (yr[1] - yr[0])) * (height - PAD.t - PAD.b);
  const xd = tickDigits(xr);
  const yd = tickDigits(yr);
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={label}>
        <title>{label}</title>
        {ticks(yr).map((v) => (
          <g key={`y${v}`} className="grid">
            <line x1={PAD.l} x2={W - PAD.r} y1={py(v)} y2={py(v)} />
            <text x={PAD.l - 6} y={py(v) + 4} textAnchor="end">{fmt(v, yd)}</text>
          </g>
        ))}
        {ticks(xr).map((v, i, all) => {
          const value = logX ? 10 ** v : v;
          const x = PAD.l + ((v - xr[0]) / (xr[1] - xr[0])) * (W - PAD.l - PAD.r);
          const anchor = i === 0 ? "start" : i === all.length - 1 ? "end" : "middle";
          return (
            <text key={`x${v}`} className="tick" x={x} y={height - PAD.b + 16} textAnchor={anchor}>
              {fmt(value, logX ? 2 : xd)}
            </text>
          );
        })}
        {markers.map((m, i) => (
          <line key={`m${i}`} className="marker" x1={px(m)} x2={px(m)} y1={PAD.t} y2={height - PAD.b} />
        ))}
        {usable.map((s, i) => {
          const tone = s.tone || "ink";
          if (s.kind === "points") {
            return (
              <g key={i} className={`pts ${tone}`}>
                {s.x.map((x, j) => (Number.isFinite(s.y[j]) ? (
                  <rect key={j} x={px(x) - 0.75} y={py(s.y[j]) - 0.75} width="1.5" height="1.5" />
                ) : null))}
              </g>
            );
          }
          const d = s.x
            .map((x, j) => (Number.isFinite(s.y[j]) ? `${j === 0 ? "M" : "L"}${px(x).toFixed(1)},${py(s.y[j]).toFixed(1)}` : ""))
            .join(" ");
          return <path key={i} className={`ln ${tone}`} d={d} />;
        })}
        <text className="axis" x={W - PAD.r} y={height - 2} textAnchor="end">{xLabel}</text>
        <text className="axis" x={12} y={(height - PAD.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(height - PAD.b) / 2})`}>{yLabel}</text>
      </svg>
      {usable.length > 1 ? (
        <figcaption className="legend">
          {usable.map((s, i) => <span key={i} className={`key ${s.tone || "ink"}`}>{s.label}</span>)}
        </figcaption>
      ) : null}
    </figure>
  );
}

export function Bars({ rows, valueKey, label, tone = "ink" }) {
  if (!rows || rows.length === 0) return <p className="quiet">No data yet.</p>;
  const max = Math.max(1, ...rows.map((r) => r[valueKey] || 0));
  const w = W / rows.length;
  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${W} 120`} role="img" aria-label={label}>
        <title>{label}</title>
        {rows.map((r, i) => {
          const h = ((r[valueKey] || 0) / max) * 96;
          return (
            <rect key={i} className={`bar ${tone}`} x={i * w + 1} y={104 - h} width={Math.max(1, w - 2)} height={h}>
              <title>{`${r.hour}: ${r[valueKey] || 0}`}</title>
            </rect>
          );
        })}
        <text className="tick" x={0} y={118}>{rows[0].hour.slice(11, 16)}</text>
        <text className="tick" x={W} y={118} textAnchor="end">{rows[rows.length - 1].hour.slice(11, 16)} UTC</text>
        <text className="tick" x={4} y={12}>max {max}</text>
      </svg>
    </figure>
  );
}
