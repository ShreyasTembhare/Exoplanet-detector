import { useCallback, useEffect, useRef, useState } from "react";
import { AppShell } from "@astryxdesign/core/AppShell";
import { Badge } from "@astryxdesign/core/Badge";
import { Banner } from "@astryxdesign/core/Banner";
import { Button } from "@astryxdesign/core/Button";
import { Collapsible } from "@astryxdesign/core/Collapsible";
import { EmptyState } from "@astryxdesign/core/EmptyState";
import { FileInput } from "@astryxdesign/core/FileInput";
import { FormLayout } from "@astryxdesign/core/FormLayout";
import { Heading } from "@astryxdesign/core/Heading";
import { HStack } from "@astryxdesign/core/HStack";
import { Layout, LayoutContent, LayoutHeader, LayoutPanel } from "@astryxdesign/core/Layout";
import { List, ListItem } from "@astryxdesign/core/List";
import { MetadataList, MetadataListItem } from "@astryxdesign/core/MetadataList";
import { NumberInput } from "@astryxdesign/core/NumberInput";
import { Section } from "@astryxdesign/core/Section";
import { SideNav, SideNavItem } from "@astryxdesign/core/SideNav";
import { StackItem } from "@astryxdesign/core/Stack";
import { StatusDot } from "@astryxdesign/core/StatusDot";
import { Table, proportional } from "@astryxdesign/core/Table";
import { Text } from "@astryxdesign/core/Text";
import { TextArea } from "@astryxdesign/core/TextArea";
import { TextInput } from "@astryxdesign/core/TextInput";
import { TopNav, TopNavHeading } from "@astryxdesign/core/TopNav";
import { VStack } from "@astryxdesign/core/VStack";
import { useTheme } from "@astryxdesign/core/theme";

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

function statusLabel(status) {
  const raw = (status || "idle").toLowerCase();
  if (raw === "running") return "Running";
  if (raw === "paused") return "Paused";
  if (raw === "stopped") return "Stopped";
  if (raw === "failed") return "Failed";
  if (raw === "done") return "Done";
  return "Idle";
}

function statusBadgeVariant(status) {
  const raw = (status || "idle").toLowerCase();
  if (raw === "running") return "info";
  if (raw === "paused") return "warning";
  if (raw === "failed") return "error";
  if (raw === "done" || raw === "stopped") return "success";
  return "neutral";
}

const API_DOWN = "The desk API is not running on port 8765. From the repo root, run python -m tool.";

async function api(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new Error(API_DOWN);
  }
  const text = await response.text();
  let body = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = null;
    }
  }
  if (!response.ok) {
    const detail = body && typeof body.detail === "string" ? body.detail : null;
    if (detail) throw new Error(detail);
    if (response.status === 502 || response.status === 503 || response.status === 500) {
      throw new Error(API_DOWN);
    }
    throw new Error(response.statusText || "Request failed");
  }
  return body;
}

function flagsText(flags) {
  if (!flags || flags.length === 0) return "None";
  return flags.map((flag) => FLAG_LABEL[flag] || flag).join(", ");
}

function PhaseChart({ curve }) {
  const { token } = useTheme();
  const phase = (curve && curve.phase) || [];
  const flux = (curve && curve.flux) || [];
  if (phase.length < 2) {
    return <Text type="supporting">No phase curve was saved with this candidate.</Text>;
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
  const axis = token("--color-data-neutral");
  const series = token("--color-data-categorical-blue");
  const label = token("--color-text-secondary");
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Phase-folded light curve" width="100%" height="auto">
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke={axis} />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke={axis} />
      <path d={d} fill="none" stroke={series} strokeWidth="1.6" />
      <text x={pad.l} y={height - 8} fill={label} fontSize="11">0</text>
      <text x={width - pad.r - 8} y={height - 8} fill={label} fontSize="11">1</text>
      <text x={8} y={pad.t + 8} fill={label} fontSize="11">flux</text>
    </svg>
  );
}

function PlotImage({ alt, src }) {
  return (
    <Section variant="muted" padding={2}>
      <img alt={alt} src={src} width="100%" height="auto" style={{ display: "block", width: "100%", height: "auto" }} />
    </Section>
  );
}

function StatusBar({ continuous, candidates }) {
  const progress = (continuous && continuous.progress) || continuous || {};
  const raw = (progress.status || (continuous && continuous.status) || "idle").toLowerCase();
  const target = progress.current_target || continuous?.current_target || "—";
  const stars = progress.stars_done ?? continuous?.stars_done ?? 0;
  const found = progress.candidates_found ?? continuous?.candidates_found ?? candidates.length;
  const last = progress.last_log_line || continuous?.last_log_line || continuous?.message || "Waiting for the first log line.";
  const rate = progress.stars_per_hour;
  return (
    <LayoutHeader hasDivider>
      <HStack gap={3} vAlign="center" wrap="wrap">
        <Badge variant={statusBadgeVariant(raw)} label={statusLabel(raw)} />
        <Text type="supporting">{target}</Text>
        <Text type="supporting" hasTabularNumbers>{stars} stars</Text>
        <Text type="supporting" hasTabularNumbers>{found} candidates</Text>
        {continuous?.survey_pass && continuous?.mode === "sectors" ? (
          <Text type="supporting" hasTabularNumbers>
            pass {continuous.survey_pass}
            {continuous.sector_max ? ` · sector ${continuous.sector ?? "—"}/${continuous.sector_max}` : ""}
          </Text>
        ) : null}
        {rate != null && rate > 0 ? (
          <Text type="supporting" hasTabularNumbers>{fmt(rate, 1)}/h</Text>
        ) : null}
        <StackItem size="fill">
          <Text type="supporting" maxLines={1}>{last}</Text>
        </StackItem>
      </HStack>
    </LayoutHeader>
  );
}

export default function App() {
  const [step, setStep] = useState("live");
  const [health, setHealth] = useState(null);
  const [queue, setQueue] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [run, setRun] = useState(null);
  const [limit, setLimit] = useState(10);
  const [tableText, setTableText] = useState("");
  const [tablePath, setTablePath] = useState("");
  const [tableFile, setTableFile] = useState(null);
  const [sector, setSector] = useState(1);
  const [sde, setSde] = useState(7);
  const [curveText, setCurveText] = useState("");
  const [curvePath, setCurvePath] = useState("");
  const [curveFile, setCurveFile] = useState(null);
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

  const refreshContinuous = useCallback(async () => {
    const body = await api("/api/continuous");
    setContinuous(body);
    return body;
  }, []);

  useEffect(() => {
    api("/api/health").then(setHealth).catch((error) => setNotice(error.message));
    api("/api/queue").then((body) => setQueue(body.queue)).catch(() => {});
    api("/api/runs/current").then((body) => setRun(body.run)).catch(() => {});
    refreshContinuous().catch(() => {});
    refreshCandidates().catch((error) => setNotice(error.message));
  }, [refreshCandidates, refreshContinuous]);

  const continuousStatus = continuous ? continuous.status : "idle";

  useEffect(() => {
    const timer = setInterval(() => {
      refreshContinuous()
        .then(() => refreshCandidates().catch(() => {}))
        .catch((error) => setNotice(error.message));
      if (run && run.status === "running") {
        api("/api/runs/current")
          .then((body) => {
            setRun(body.run);
            if (body.run && body.run.status !== "running") refreshCandidates();
          })
          .catch(() => {});
      }
    }, 1200);
    return () => clearInterval(timer);
  }, [refreshContinuous, refreshCandidates, run]);

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

  async function onPickFile(file, setFile, setText) {
    setFile(file);
    if (!file) {
      setText("");
      return;
    }
    setText(await file.text());
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
      setStep("live");
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
      setStep("live");
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
    setStep("inbox");
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
  const showModel = checkpoint || candidates.some((row) => row.model_score != null || row.prob_planet != null);
  const searching = continuousStatus === "running";

  return (
    <AppShell
      height="fill"
      contentPadding={0}
      variant="elevated"
      banner={
        notice ? (
          <Banner
            status="error"
            title={notice}
            container="section"
            isDismissable
            onDismiss={() => setNotice("")}
          />
        ) : null
      }
      topNav={
        <TopNav heading={<TopNavHeading heading="Transit desk" />} label="App" />
      }
      sideNav={
        <SideNav>
          <SideNavItem
            label="Live"
            icon="clock"
            isSelected={step === "live"}
            onClick={() => setStep("live")}
            endContent={searching ? <StatusDot variant="success" isPulsing label="Searching" /> : null}
          />
          <SideNavItem
            label="Queue"
            icon="funnel"
            isSelected={step === "queue"}
            onClick={() => setStep("queue")}
          />
          <SideNavItem
            label="Inbox"
            icon="search"
            isSelected={step === "inbox"}
            onClick={() => setStep("inbox")}
            endContent={candidates.length > 0 ? <Badge variant="neutral" label={candidates.length} /> : null}
          />
        </SideNav>
      }
    >
      <Layout
        height="fill"
        header={<StatusBar continuous={continuous} candidates={candidates} />}
        content={
          step === "inbox" ? (
            <InboxView
              candidates={candidates}
              selected={selected}
              onOpen={openCandidate}
              showModel={showModel}
              detail={detail}
              onExport={onExport}
              exportResult={exportResult}
            />
          ) : (
            <LayoutContent padding={5} isScrollable={step !== "live"}>
              {step === "live" ? (
                <LiveView
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
                  curveFile={curveFile}
                  setCurveFile={setCurveFile}
                  onPickFile={onPickFile}
                  ticId={ticId}
                  setTicId={setTicId}
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
              {step === "queue" ? (
                <QueueView
                  queue={queue}
                  limit={limit}
                  setLimit={setLimit}
                  tableText={tableText}
                  setTableText={setTableText}
                  tablePath={tablePath}
                  setTablePath={setTablePath}
                  tableFile={tableFile}
                  setTableFile={setTableFile}
                  onPickFile={onPickFile}
                  onRank={onRank}
                  busy={busy}
                />
              ) : null}
            </LayoutContent>
          )
        }
      />
    </AppShell>
  );
}

function QueueView({
  queue, limit, setLimit, tableText, setTableText, tablePath, setTablePath,
  tableFile, setTableFile, onPickFile, onRank, busy,
}) {
  const empty = !queue || ((queue.primary || []).length === 0 && (queue.second_pass || []).length === 0);
  return (
    <VStack gap={5} maxWidth={880}>
      <VStack gap={1}>
        <Heading level={1}>Queue</Heading>
        <Text type="body" color="secondary">
          Rank who is next before any light curve is downloaded.
        </Text>
      </VStack>
      <form onSubmit={onRank}>
        <VStack gap={4}>
          <FormLayout direction="horizontal">
            <NumberInput
              label="Best N"
              value={limit}
              onChange={setLimit}
              min={1}
              isIntegerOnly
              isWheelEnabled={false}
            />
            <FileInput
              label="TIC table"
              value={tableFile}
              onChange={(file) => onPickFile(file, setTableFile, setTableText)}
              accept=".csv,.txt"
              mode="input"
              isOptional
            />
            <TextInput
              label="Path on this machine"
              value={tablePath}
              onChange={setTablePath}
              placeholder="/path/targets.csv"
              isOptional
            />
          </FormLayout>
          <Button type="submit" variant="primary" label="Rank" isLoading={busy} />
          <Collapsible trigger="Paste CSV" defaultIsOpen={false}>
            <TextArea
              label="TIC table"
              value={tableText}
              onChange={setTableText}
              placeholder="tic,radius,n_sectors,crowdsap,tmag,disposition"
            />
          </Collapsible>
        </VStack>
      </form>
      {empty ? (
        <EmptyState
          title="No target table yet"
          description="Rank a TIC file before download. The best N come first. Sky position is not a factor."
          headingLevel={2}
        />
      ) : (
        <VStack gap={5}>
          {queue.note ? <Text type="supporting">{queue.note}</Text> : null}
          <TargetTable title="Primary" rows={queue.primary} />
          <TargetTable title="Known hosts, masked second pass" rows={queue.second_pass} />
        </VStack>
      )}
    </VStack>
  );
}

function TargetTable({ title, rows }) {
  if (!rows || rows.length === 0) return null;
  return (
    <VStack gap={2}>
      <Heading level={2}>{title}</Heading>
      <Table
        data={rows}
        idKey="tic_id"
        density="compact"
        columns={[
          { key: "tic_id", header: "TIC", width: proportional(1, { minWidth: 80 }) },
          { key: "why", header: "Why this star is next", width: proportional(3, { minWidth: 160 }) },
          {
            key: "expected_snr",
            header: "SNR",
            align: "end",
            renderCell: (row) => (
              <Text type="code" hasTabularNumbers>{fmt(row.expected_snr, 1)}</Text>
            ),
          },
          {
            key: "disposition",
            header: "Disposition",
            renderCell: (row) => row.disposition || "—",
          },
        ]}
      />
    </VStack>
  );
}

function LiveView({
  sector, setSector, limit, setLimit, sde, setSde,
  curveText, setCurveText, curvePath, setCurvePath, curveFile, setCurveFile,
  onPickFile, ticId, setTicId, start, run, busy,
  curveDir, setCurveDir, continuous, keepSearching, pauseSearching, stopSearching,
}) {
  const logRef = useRef(null);
  const lines = (continuous && continuous.lines) || [];
  const status = continuous ? continuous.status : "idle";
  useEffect(() => {
    const node = logRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [lines.length]);
  const startLabel = status === "paused" || status === "stopped" ? "Resume" : "Start";
  const oneOffRunning = run && run.status === "running";
  return (
    <VStack gap={4} height="100%">
      <VStack gap={1}>
        <Heading level={1}>Live</Heading>
        <Text type="body" color="secondary">
          Start surveys TESS sectors from MAST, ranks the stars, skips any already searched, and keeps going after you close this page.
          A TIC table or local curves still take over when you set them.
          {continuous?.auto_start === false ? " Auto-start is off until Start or Resume." : ""}
        </Text>
      </VStack>
      <HStack gap={2} wrap="wrap">
        <Button
          variant="primary"
          label={startLabel}
          onClick={keepSearching}
          isDisabled={busy || status === "running" || oneOffRunning}
          isLoading={busy}
        />
        <Button label="Pause" onClick={pauseSearching} isDisabled={status !== "running"} />
        <Button
          label="Stop"
          onClick={stopSearching}
          isDisabled={status !== "running" && status !== "paused"}
        />
      </HStack>
      <StackItem size="fill" isScrollable ref={logRef}>
        {lines.length === 0 ? (
          <EmptyState
            title="The log is empty"
            description="When MAST is missing, the loop stays honest and invents no stars."
            headingLevel={2}
            isCompact
          />
        ) : (
          <List header={<Heading level={2}>Log</Heading>} density="compact" hasDividers>
            {lines.map((line, index) => (
              <ListItem
                key={`${line.ts}-${index}`}
                itemKey={`${line.ts}-${index}`}
                label={line.message}
                description={line.ts}
                align="start"
                startContent={
                  line.error ? (
                    <StatusDot variant="error" label="Error" />
                  ) : line.candidate ? (
                    <StatusDot variant="success" label="Candidate" />
                  ) : (
                    <StatusDot variant="neutral" label="Log" />
                  )
                }
              />
            ))}
          </List>
        )}
      </StackItem>
      <Collapsible trigger="Advanced" defaultIsOpen={false}>
        <VStack gap={5}>
          <FormLayout direction="horizontal">
            <NumberInput
              label="Sector"
              value={sector}
              onChange={setSector}
              isIntegerOnly
              isWheelEnabled={false}
            />
            <NumberInput
              label="Best N"
              value={limit}
              onChange={setLimit}
              min={1}
              isIntegerOnly
              isWheelEnabled={false}
            />
            <NumberInput
              label="SDE gate"
              value={sde}
              onChange={setSde}
              step={0.1}
              isWheelEnabled={false}
            />
          </FormLayout>
          <TextInput
            label="Local curve directory"
            value={curveDir}
            onChange={setCurveDir}
            placeholder="/path/curves"
            isOptional
          />
          <Text type="supporting">
            Training, autopilot, and SSL pretrain stay on python run.py.
          </Text>
          <VStack gap={3}>
            <Heading level={2}>One-off search</Heading>
            <Text type="supporting">
              A directory of time, flux CSVs is searched as given. Otherwise a ranked queue is walked, and if there is no queue the sectors advance.
            </Text>
            <Button
              label="Start sector search once"
              onClick={() => start("sector")}
              isDisabled={busy || status === "running" || oneOffRunning}
            />
            <RunState run={run} />
          </VStack>
          <VStack gap={3}>
            <Heading level={2}>Local light curve</Heading>
            <FormLayout>
              <TextInput label="TIC label" value={ticId} onChange={setTicId} />
              <FileInput
                label="Curve file"
                value={curveFile}
                onChange={(file) => onPickFile(file, setCurveFile, setCurveText)}
                accept=".csv,.txt"
                mode="input"
                isOptional
              />
              <TextInput
                label="Path on this machine"
                value={curvePath}
                onChange={setCurvePath}
                placeholder="/path/curve.csv"
                isOptional
              />
            </FormLayout>
            <Collapsible trigger="Paste curve CSV" defaultIsOpen={false}>
              <TextArea
                label="Curve"
                value={curveText}
                onChange={setCurveText}
                placeholder="time,flux"
              />
            </Collapsible>
            <Button
              label="Search this curve"
              onClick={() => start("local")}
              isDisabled={busy || status === "running" || oneOffRunning}
            />
          </VStack>
        </VStack>
      </Collapsible>
    </VStack>
  );
}

function RunState({ run }) {
  if (!run) {
    return <Text type="supporting">No one-off search has been started.</Text>;
  }
  if (run.status === "running") {
    return (
      <Banner status="info" title="Running" description={run.message} collapsible={false} />
    );
  }
  if (run.status === "failed") {
    return (
      <Banner
        status="error"
        title={run.needs_mast ? "MAST required" : "Search failed"}
        description={run.error}
        collapsible={false}
      />
    );
  }
  return (
    <Banner
      status="success"
      title="Finished"
      description={
        run.candidate_ids && run.candidate_ids.length === 0
          ? `${run.message} The inbox was not changed.`
          : run.message
      }
      collapsible={false}
    />
  );
}

function InboxView({ candidates, selected, onOpen, showModel, detail, onExport, exportResult }) {
  const columns = [
    {
      key: "tic_id",
      header: "TIC",
      width: proportional(1, { minWidth: 88 }),
      renderCell: (row) => (
        <Button label={String(row.tic_id)} variant="ghost" size="sm" onClick={() => onOpen(row.id)} />
      ),
    },
    {
      key: "period_days",
      header: "Period",
      align: "end",
      renderCell: (row) => (
        <Text type="code" hasTabularNumbers>{fmt(row.period_days, 4)} d</Text>
      ),
    },
    {
      key: "epoch_btjd",
      header: "Epoch",
      align: "end",
      renderCell: (row) => (
        <Text type="code" hasTabularNumbers>{fmt(row.epoch_btjd, 4)}</Text>
      ),
    },
    {
      key: "depth_ppm",
      header: "Depth",
      align: "end",
      renderCell: (row) => (
        <Text type="code" hasTabularNumbers>
          {row.depth_ppm === null ? "—" : `${fmt(row.depth_ppm, 0)} ppm`}
        </Text>
      ),
    },
    {
      key: "sde",
      header: "SDE",
      align: "end",
      renderCell: (row) => (
        <Text type="code" hasTabularNumbers>{fmt(row.sde, 1)}</Text>
      ),
    },
  ];
  if (showModel) {
    columns.push({
      key: "model_score",
      header: "Model",
      align: "end",
      renderCell: (row) => (
        <Text type="code" hasTabularNumbers>
          {row.model_score == null && row.prob_planet == null ? "—" : fmt(row.model_score ?? row.prob_planet, 3)}
        </Text>
      ),
    });
  }
  columns.push(
    {
      key: "hard_flags",
      header: "Flags",
      renderCell: (row) => flagsText(row.hard_flags),
    },
    { key: "label", header: "Label" },
  );

  return (
    <Layout
      height="fill"
      header={
        <LayoutHeader hasDivider>
          <Heading level={1}>Inbox</Heading>
        </LayoutHeader>
      }
      content={
        <LayoutContent padding={0} isScrollable>
          {candidates.length === 0 ? (
            <EmptyState
              title="No candidates yet"
              description="Continuous search has not found an SDE pass. This list stays empty when a search fails or nothing clears the gate."
              headingLevel={2}
            />
          ) : (
            <Table
              data={candidates}
              idKey="id"
              density="compact"
              hasHover
              columns={columns}
            />
          )}
        </LayoutContent>
      }
      end={
        <LayoutPanel width={380} hasDivider isScrollable label="Candidate" role="complementary">
          <ReviewPanel detail={selected ? detail : null} onExport={onExport} exportResult={exportResult} showModel={showModel} />
        </LayoutPanel>
      }
    />
  );
}

function ReviewPanel({ detail, onExport, exportResult, showModel }) {
  if (!detail) {
    return (
      <EmptyState
        title="Select a candidate"
        description="Open a TIC from the inbox to inspect the fold, flags, and export."
        headingLevel={2}
        isCompact
      />
    );
  }
  const flags = new Set(detail.hard_flags || []);
  return (
    <VStack gap={4}>
      <VStack gap={1}>
        <Heading level={2}>TIC {detail.tic_id}</Heading>
        <Text type="supporting">
          {detail.label}
          {detail.status === "CANDIDATE_SDE" ? " · SDE pass, no classifier score" : ""}
          {showModel && detail.model_score != null ? ` · model ${fmt(detail.model_score, 3)}` : ""}
        </Text>
      </VStack>
      <MetadataList columns={2} label={{ position: "top" }}>
        <MetadataListItem label="Period">{fmt(detail.period_days, 4)} d</MetadataListItem>
        <MetadataListItem label="Epoch">{fmt(detail.epoch_btjd, 4)}</MetadataListItem>
        <MetadataListItem label="Depth">
          {detail.depth_ppm === null ? "—" : `${fmt(detail.depth_ppm, 0)} ppm`}
        </MetadataListItem>
        <MetadataListItem label="SDE">{fmt(detail.sde, 2)}</MetadataListItem>
      </MetadataList>
      <VStack gap={2}>
        <Heading level={3}>Phase fold</Heading>
        <PhaseChart curve={detail.phase_curve} />
      </VStack>
      {detail.has_plot ? (
        <PlotImage alt="Pipeline phase plot" src={`/api/candidates/${detail.id}/plot`} />
      ) : null}
      {detail.has_difference_image ? (
        <VStack gap={2}>
          <Heading level={3}>Difference image</Heading>
          <PlotImage alt="Difference image" src={`/api/candidates/${detail.id}/difference-image`} />
        </VStack>
      ) : null}
      <VStack gap={2}>
        <Heading level={3}>Hard flags</Heading>
        <MetadataList>
          <MetadataListItem label="Odd/even">
            {fmt(detail.odd_even_ratio, 2)} · {flags.has("odd_even") ? "Flag" : "Clear"}
          </MetadataListItem>
          <MetadataListItem label="Secondary">
            {fmt(detail.secondary_significance, 2)} · {flags.has("secondary") ? "Flag" : "Clear"}
          </MetadataListItem>
          <MetadataListItem label="Centroid">
            {fmt(detail.centroid_offset_px, 2)} · {flags.has("centroid") ? "Flag" : "Clear"}
          </MetadataListItem>
          <MetadataListItem label="n transits">
            {detail.n_transits ?? "—"} · {flags.has("n_transits") ? "Flag" : "Clear"}
          </MetadataListItem>
        </MetadataList>
      </VStack>
      {detail.epoch_btjd === null ? (
        <Banner
          status="error"
          title="No epoch on this candidate"
          description="Export is refused so the packet cannot say t0 = 0."
          collapsible={false}
        />
      ) : (
        <Button label="Export TFOP packet" variant="primary" onClick={onExport} />
      )}
      {exportResult ? (
        <Banner
          status="success"
          title="Packet written"
          description={`Wrote epoch ${fmt(exportResult.epoch_btjd, 4)} BTJD.`}
          collapsible={false}
        />
      ) : null}
    </VStack>
  );
}
