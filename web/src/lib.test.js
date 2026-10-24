import { describe, expect, it } from "vitest";
import {
  demotionText, extent, fmt, fmtDuration, formatHash, inboxQuery, mergeLines, parseHash, statusLabel, withToken,
} from "./lib.js";

describe("formatting", () => {
  it("formats numbers and blanks", () => {
    expect(fmt(1.23456, 2)).toBe("1.23");
    expect(fmt(null, 2)).toBe("—");
    expect(fmt("abc", 1)).toBe("—");
  });
  it("formats durations", () => {
    expect(fmtDuration(42)).toBe("42 s");
    expect(fmtDuration(3900)).toBe("1 h 5 m");
    expect(fmtDuration(null)).toBe("—");
  });
  it("labels statuses", () => {
    expect(statusLabel("running")).toBe("RUNNING");
    expect(statusLabel("weird")).toBe("IDLE");
  });
  it("describes demotion reasons", () => {
    expect(demotionText({ factor: 0.5, reasons: [{ code: "odd_even", multiplier: 0.5 }] })).toBe("odd_even ×0.50");
    expect(demotionText(null)).toBe("");
  });
  it("pads a flat extent", () => {
    expect(extent([2, 2])).toEqual([1.5, 2.5]);
    expect(extent([NaN])).toEqual([0, 1]);
  });
});

describe("routing", () => {
  it("round-trips candidate deep links", () => {
    const hash = formatHash("review", { id: "TIC_123_p3.14" });
    expect(hash).toBe("#/candidate/TIC_123_p3.14");
    expect(parseHash(hash)).toEqual({ view: "review", id: "TIC_123_p3.14", params: {} });
  });
  it("keeps inbox filters in the hash", () => {
    const hash = formatHash("inbox", { params: { sort: "sde", q: "", flags: "none" } });
    expect(hash).toBe("#/inbox?sort=sde&flags=none");
    expect(parseHash(hash).params).toEqual({ sort: "sde", flags: "none" });
  });
  it("falls back to live", () => {
    expect(parseHash("").view).toBe("live");
    expect(parseHash("#/nope").view).toBe("live");
  });
});

describe("inbox query", () => {
  it("paginates and drops empty filters", () => {
    const q = new URLSearchParams(inboxQuery({ page: "2", q: "" }, 50));
    expect(q.get("offset")).toBe("100");
    expect(q.get("limit")).toBe("50");
    expect(q.has("q")).toBe(false);
    expect(q.get("sort")).toBe("score");
  });
});

describe("log merging", () => {
  it("dedupes by id, sorts, and caps", () => {
    const a = [{ id: 1 }, { id: 2 }];
    const merged = mergeLines(a, [{ id: 2 }, { id: 3 }]);
    expect(merged.map((l) => l.id)).toEqual([1, 2, 3]);
    expect(mergeLines(merged, [{ id: 4 }], 2).map((l) => l.id)).toEqual([3, 4]);
    expect(mergeLines(a, [])).toBe(a);
  });
});

describe("token", () => {
  it("adds a query token only when set", () => {
    expect(withToken("/api/stream?after_id=3", "t k")).toBe("/api/stream?after_id=3&token=t%20k");
    expect(withToken("/api/x", "")).toBe("/api/x");
  });
});
