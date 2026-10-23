// @vitest-environment jsdom
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";

function respond(body, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

const routes = {
  "/api/health": { ok: true, checkpoint: false, mast: true, auth_required: false },
  "/api/queue": { queue: null },
  "/api/runs/current": { run: null },
  "/api/continuous": {
    status: "running", lines: [{ id: 1, ts: "2026-01-01T00:00:00Z", message: "Searching sector 3." }],
    progress: { status: "running", stars_done: 4, candidates_found: 1, work: { done: 2, pending: 1, leased: 1, failed: 0 } },
  },
  "/api/candidates": {
    total: 1,
    candidates: [{ id: "TIC_9_p1.00", tic_id: "9", score: 0.8, period_days: 1, sde: 12, hard_flags: [], label: "New", kind: "periodic" }],
  },
};

beforeEach(() => {
  window.EventSource = undefined;
  vi.stubGlobal("fetch", vi.fn((url) => {
    const path = String(url).split("?")[0];
    return respond(routes[path] || {});
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.location.hash = "";
});

describe("App", () => {
  it("shows live progress and the log", async () => {
    await act(async () => { render(<App />); });
    expect(await screen.findByText("Searching sector 3.")).toBeTruthy();
    expect(screen.getAllByText("RUNNING").length).toBeGreaterThan(0);
  });

  it("renders the inbox from a deep link", async () => {
    window.location.hash = "#/inbox?sort=sde";
    await act(async () => { render(<App />); });
    const link = await screen.findByRole("link", { name: "9" });
    expect(link.getAttribute("href")).toBe("#/candidate/TIC_9_p1.00");
  });

  it("asks for a token when the desk requires one", async () => {
    routes["/api/health"] = { ...routes["/api/health"], auth_required: true };
    await act(async () => { render(<App />); });
    expect(await screen.findByText("Desk token")).toBeTruthy();
    routes["/api/health"] = { ...routes["/api/health"], auth_required: false };
  });
});
