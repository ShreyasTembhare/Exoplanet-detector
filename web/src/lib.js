// Pure helpers shared by the views (unit-tested with Vitest).

export const TOKEN_KEY = "desk_token";

export function fmt(value, digits) {
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

export function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined || !Number.isFinite(Number(seconds))) return "—";
  const s = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  if (h >= 48) return `${Math.round(h / 24)} d`;
  if (h > 0) return `${h} h ${m} m`;
  if (m > 0) return `${m} m`;
  return `${s} s`;
}

export function statusLabel(status) {
  const raw = (status || "idle").toLowerCase();
  return ["running", "paused", "stopped", "failed", "done"].includes(raw) ? raw.toUpperCase() : "IDLE";
}

// Routes: #/live, #/queue, #/inbox?sort=sde&flags=none, #/candidate/<id>, #/stats
export function parseHash(hash) {
  const raw = String(hash || "").replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const parts = path.split("/").filter(Boolean).map(decodeURIComponent);
  const params = Object.fromEntries(new URLSearchParams(query));
  if (parts[0] === "candidate" && parts[1]) return { view: "review", id: parts[1], params };
  const views = ["live", "queue", "inbox", "review", "stats"];
  return { view: views.includes(parts[0]) ? parts[0] : "live", id: null, params };
}

export function formatHash(view, { id = null, params = {} } = {}) {
  if (view === "review" && id) return `#/candidate/${encodeURIComponent(id)}`;
  const query = new URLSearchParams(
    Object.entries(params).filter(([, value]) => value !== "" && value !== null && value !== undefined),
  ).toString();
  return `#/${view}${query ? `?${query}` : ""}`;
}

export const INBOX_DEFAULTS = {
  q: "", label: "", flags: "", min_sde: "", sort: "score", order: "desc", page: "0",
};

export function inboxQuery(filters, pageSize) {
  const merged = { ...INBOX_DEFAULTS, ...filters };
  const params = new URLSearchParams();
  for (const key of ["q", "label", "flags", "min_sde", "sort", "order"]) {
    if (merged[key] !== "" && merged[key] !== undefined && merged[key] !== null) params.set(key, merged[key]);
  }
  params.set("limit", String(pageSize));
  params.set("offset", String(Math.max(0, Number(merged.page) || 0) * pageSize));
  return params.toString();
}

// Merge newly streamed log lines, dropping duplicates and keeping the newest `cap`.
export function mergeLines(existing, incoming, cap = 1000) {
  if (!incoming || incoming.length === 0) return existing;
  const seen = new Set(existing.map((line) => line.id));
  const merged = existing.concat(incoming.filter((line) => line.id === undefined || !seen.has(line.id)));
  merged.sort((a, b) => (a.id ?? 0) - (b.id ?? 0));
  return merged.length > cap ? merged.slice(merged.length - cap) : merged;
}

export function withToken(url, token) {
  if (!token) return url;
  return `${url}${url.includes("?") ? "&" : "?"}token=${encodeURIComponent(token)}`;
}

export function demotionText(demotion) {
  if (!demotion || !Array.isArray(demotion.reasons) || demotion.reasons.length === 0) return "";
  return demotion.reasons.map((r) => `${r.code} ×${fmt(r.multiplier, 2)}`).join(", ");
}

export function extent(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of values) {
    if (Number.isFinite(v)) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  if (min === Infinity) return [0, 1];
  if (min === max) return [min - 0.5, max + 0.5];
  return [min, max];
}
