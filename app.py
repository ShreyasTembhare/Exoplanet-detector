"""
TESS Exoplanet Detector — Streamlit Control Center.

Launch with:  streamlit run app.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TESS Exoplanet Detector",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

TASK_STATE_FILE = ROOT / ".streamlit_tasks.json"
CANDIDATE_DIR = ROOT / "candidates"
AUTOPILOT_STATE = ROOT / "autopilot_state.json"
DEFAULT_CHECKPOINT = "models/checkpoints/resnet1d.pt"

# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

def _load_task_state() -> dict:
    if TASK_STATE_FILE.exists():
        try:
            return json.loads(TASK_STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_task_state(state: dict):
    TASK_STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_pid_running(pid: int) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _start_background_task(name: str, cmd: list):
    log_path = ROOT / f".task_{name}.log"
    log_f = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=log_f, stderr=subprocess.STDOUT,
        cwd=str(ROOT), start_new_session=True,
    )
    state = _load_task_state()
    state[name] = {
        "pid": proc.pid,
        "started_at": datetime.utcnow().isoformat(),
        "cmd": " ".join(cmd),
        "log": str(log_path),
    }
    _save_task_state(state)
    return proc.pid


def _stop_background_task(name: str):
    state = _load_task_state()
    info = state.get(name, {})
    pid = info.get("pid")
    if pid and _is_pid_running(pid):
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    state.pop(name, None)
    _save_task_state(state)


def _task_status(name: str) -> dict:
    state = _load_task_state()
    info = state.get(name, {})
    pid = info.get("pid")
    running = _is_pid_running(pid) if pid else False
    log_path = info.get("log", "")
    log_tail = ""
    if log_path and os.path.exists(log_path):
        try:
            lines = Path(log_path).read_text().splitlines()
            log_tail = "\n".join(lines[-40:])
        except Exception:
            pass
    return {"running": running, "pid": pid, "info": info, "log_tail": log_tail}


# ---------------------------------------------------------------------------
# Sidebar: strategy profile + hardware
# ---------------------------------------------------------------------------

from strategy_profiles import PROFILES, DEFAULT_PROFILE, get_profile, profile_names

with st.sidebar:
    st.title("TESS Exoplanet Detector")
    st.caption("Control Center")
    st.divider()

    selected_profile = st.selectbox(
        "Strategy Profile",
        options=profile_names(),
        format_func=lambda k: PROFILES[k]["label"],
        index=profile_names().index(DEFAULT_PROFILE),
        help="Choose a niche detection strategy. Affects BLS period range, thresholds, and scoring.",
    )
    prof = get_profile(selected_profile)
    st.info(prof["description"])

    st.divider()
    st.subheader("Hardware")
    try:
        from device_util import get_device, get_jax_backend
        import torch
        dev = get_device()
        jax_be = get_jax_backend()
        st.text(f"PyTorch {torch.__version__}")
        st.text(f"Device: {dev}")
        st.text(f"JAX BLS: {jax_be}")
    except Exception as e:
        st.warning(f"Could not detect hardware: {e}")

    st.divider()
    st.subheader("Paths")
    candidate_dir = st.text_input("Candidate dir", value="candidates")
    checkpoint_path = st.text_input("Model checkpoint", value=DEFAULT_CHECKPOINT)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_scan, tab_hunt, tab_train, tab_autopilot, tab_candidates, tab_logs = st.tabs(
    ["Scan", "Hunt", "Train", "Autopilot", "Candidates", "Logs"]
)

# ========================== SCAN TAB ==========================
with tab_scan:
    st.header("Single Target Scan")
    st.caption("Analyse one TIC target through all pipeline phases.")

    col1, col2 = st.columns([2, 1])
    with col1:
        tic_id = st.text_input("TIC ID", value="TIC 441462736", help="e.g. TIC 441462736 or just 441462736")
    with col2:
        scan_sector = st.number_input("Sector (0 = auto)", min_value=0, max_value=100, value=0)

    col3, col4, col5 = st.columns(3)
    with col3:
        scan_pmin = st.number_input("Period min (d)", min_value=0.1, max_value=200.0,
                                     value=float(prof["period_min"]), step=0.1)
    with col4:
        scan_pmax = st.number_input("Period max (d)", min_value=0.5, max_value=500.0,
                                     value=float(prof["period_max"]), step=1.0)
    with col5:
        scan_nperiods = st.number_input("BLS grid size", min_value=1000, max_value=50000,
                                         value=5000, step=1000)

    scan_predict = st.checkbox("Run classifier prediction", value=True)

    if st.button("Run Scan", type="primary", use_container_width=True):
        with st.spinner("Running pipeline... this may take a few minutes."):
            try:
                from services import run_scan, ScanConfig
                cfg = ScanConfig(
                    tic_id=tic_id,
                    sector=scan_sector if scan_sector > 0 else None,
                    period_min=scan_pmin, period_max=scan_pmax,
                    nperiods=scan_nperiods, use_cache=True,
                    predict=scan_predict, checkpoint=checkpoint_path,
                )
                result = run_scan(cfg)
                st.session_state["scan_result"] = result
                st.success("Scan complete!")
            except Exception as e:
                st.error(f"Scan failed: {e}")

    if "scan_result" in st.session_state:
        res = st.session_state["scan_result"]
        st.subheader("Results")

        m1, m2, m3 = st.columns(3)
        m1.metric("Best Period", f"{res['best_period']:.4f} d")
        m2.metric("Epoch", f"{res['epoch']:.4f}")
        m3.metric("Cadences", f"{len(res['time']):,}")

        if "prediction" in res:
            pred = res["prediction"]
            m4, m5 = st.columns(2)
            label = "Planet" if pred["class"] == 1 else "False Positive"
            m4.metric("Classification", label)
            m5.metric("P(planet)", f"{pred['prob_planet']:.3f}")

        import numpy as np
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            axes[0].scatter(res["time"], res["flux"], s=0.3, alpha=0.5)
            axes[0].set_xlabel("Time (BTJD)")
            axes[0].set_ylabel("Flux")
            axes[0].set_title("Light Curve")

            axes[1].plot(res["periods"], res["power"], lw=0.5)
            axes[1].axvline(res["best_period"], color="red", ls="--", alpha=0.7, label=f"P={res['best_period']:.3f}")
            axes[1].set_xlabel("Period (d)")
            axes[1].set_ylabel("BLS Power")
            axes[1].set_title("BLS Periodogram")
            axes[1].legend(fontsize=8)

            phase = ((res["time"] - res["epoch"]) % res["best_period"]) / res["best_period"]
            axes[2].scatter(phase, res["flux"], s=0.3, alpha=0.5)
            axes[2].set_xlabel("Phase")
            axes[2].set_ylabel("Flux")
            axes[2].set_title("Phase-Folded")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not render plots: {e}")

# ========================== HUNT TAB ==========================
with tab_hunt:
    st.header("Sector Hunt")
    st.caption("Sweep all targets in a TESS sector and classify candidates.")

    col1, col2 = st.columns(2)
    with col1:
        hunt_sector = st.number_input("Sector", min_value=1, max_value=100, value=15, key="hunt_sector")
    with col2:
        hunt_limit = st.number_input("Target limit", min_value=1, max_value=100000, value=1000, key="hunt_limit")

    col3, col4 = st.columns(2)
    with col3:
        hunt_threshold = st.slider("AI probability threshold", 0.0, 1.0,
                                    value=float(prof["probability_threshold"]), step=0.05, key="hunt_thresh")
    with col4:
        hunt_bls = st.number_input("BLS power threshold", min_value=0.0, max_value=0.1,
                                    value=float(prof["bls_threshold"]), step=0.0001, format="%.4f", key="hunt_bls")

    hunt_status = _task_status("hunt")

    if hunt_status["running"]:
        st.warning(f"Hunt is running (PID {hunt_status['pid']})")
        if st.button("Stop Hunt", type="secondary"):
            _stop_background_task("hunt")
            st.rerun()

        log_file = f"processed_stars_s{hunt_sector:03d}.txt"
        if os.path.exists(log_file):
            try:
                import pandas as pd
                df = pd.read_csv(log_file)
                total = len(df)
                done = len(df[df["LAST_STAGE"] == "DONE"])
                candidates = len(df[df["STATUS"].str.startswith("CANDIDATE", na=False)])
                c1, c2, c3 = st.columns(3)
                c1.metric("Processed", f"{total:,}")
                c2.metric("Completed", f"{done:,}")
                c3.metric("Candidates", f"{candidates:,}")
            except Exception:
                pass

        if hunt_status["log_tail"]:
            with st.expander("Live Log Output", expanded=False):
                st.code(hunt_status["log_tail"], language="text")

        time.sleep(0.5)
        st.rerun()
    else:
        if st.button("Start Hunt", type="primary", use_container_width=True):
            cmd = [
                sys.executable, str(ROOT / "run.py"), "hunt",
                "--sector", str(hunt_sector),
                "--limit", str(hunt_limit),
                "--threshold", str(hunt_threshold),
                "--bls-threshold", str(hunt_bls),
                "--checkpoint", checkpoint_path,
                "--candidate-dir", candidate_dir,
                "--period-min", str(prof["period_min"]),
                "--period-max", str(prof["period_max"]),
                "--nperiods", str(prof["nperiods"]),
                "--strategy-profile", selected_profile,
            ]
            _start_background_task("hunt", cmd)
            st.rerun()

        log_file = f"processed_stars_s{hunt_sector:03d}.txt"
        if os.path.exists(log_file):
            st.subheader("Previous Run Results")
            try:
                import pandas as pd
                df = pd.read_csv(log_file)
                total = len(df)
                done = len(df[df["LAST_STAGE"] == "DONE"])
                candidates_df = df[df["STATUS"].str.startswith("CANDIDATE", na=False)]
                c1, c2, c3 = st.columns(3)
                c1.metric("Total Processed", f"{total:,}")
                c2.metric("Completed", f"{done:,}")
                c3.metric("Candidates Found", f"{len(candidates_df):,}")
                if not candidates_df.empty:
                    st.dataframe(candidates_df, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse log: {e}")

# ========================== TRAIN TAB ==========================
with tab_train:
    st.header("Train Classifier")
    st.caption("Train or fine-tune the ResNet-1D exoplanet classifier.")

    data_source = st.radio("Data source", ["auto (NASA Archive)", "Custom CSV"], horizontal=True)
    data_val = "auto" if "auto" in data_source else None
    if "Custom" in data_source:
        data_val = st.text_input("CSV path", value="")
        if not data_val:
            data_val = None

    col1, col2, col3 = st.columns(3)
    with col1:
        train_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=30, key="train_epochs")
    with col2:
        train_lr = st.number_input("Learning rate", min_value=1e-6, max_value=1e-1,
                                    value=1e-3, format="%.1e", key="train_lr")
    with col3:
        train_batch = st.number_input("Batch size", min_value=4, max_value=256, value=32, key="train_batch")

    col4, col5 = st.columns(2)
    with col4:
        train_patience = st.number_input("Early stop patience", min_value=1, max_value=50, value=7, key="train_pat")
    with col5:
        train_max_class = st.number_input("Max samples/class", min_value=10, max_value=2000, value=250, key="train_mpc")

    train_amp = st.checkbox("Mixed precision (AMP)", value=False)
    pretrained_path = st.text_input("Pretrained checkpoint (optional)", value="", key="pretrained")

    train_status = _task_status("train")

    if train_status["running"]:
        st.warning(f"Training is running (PID {train_status['pid']})")
        if st.button("Stop Training", type="secondary"):
            _stop_background_task("train")
            st.rerun()
        if train_status["log_tail"]:
            with st.expander("Training Log", expanded=True):
                st.code(train_status["log_tail"], language="text")
        time.sleep(1)
        st.rerun()
    else:
        if st.button("Start Training", type="primary", use_container_width=True):
            cmd = [
                sys.executable, str(ROOT / "run.py"), "train",
                "--epochs", str(train_epochs),
                "--batch-size", str(train_batch),
                "--lr", str(train_lr),
                "--patience", str(train_patience),
                "--max-per-class", str(train_max_class),
                "--out", checkpoint_path,
            ]
            if data_val:
                cmd.extend(["--data", data_val])
            if train_amp:
                cmd.append("--amp")
            if pretrained_path:
                cmd.extend(["--pretrained", pretrained_path])

            _start_background_task("train", cmd)
            st.rerun()

        metrics_path = Path(checkpoint_path).with_suffix(".metrics.json")
        if metrics_path.exists():
            st.subheader("Latest Model Metrics")
            try:
                metrics = json.loads(metrics_path.read_text())
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("F1 Score", f"{metrics.get('f1', 0):.4f}")
                c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                c3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                c4.metric("ROC-AUC", f"{metrics.get('roc_auc', 'N/A')}")
                cm = metrics.get("confusion_matrix", {})
                if cm:
                    st.caption(f"Confusion: TP={cm.get('tp',0)}  FP={cm.get('fp',0)}  FN={cm.get('fn',0)}  TN={cm.get('tn',0)}")
            except Exception as e:
                st.warning(f"Could not load metrics: {e}")

# ========================== AUTOPILOT TAB ==========================
with tab_autopilot:
    st.header("Autopilot Mode")
    st.caption("Fully autonomous multi-sector exoplanet hunting with TOI cross-matching.")

    col1, col2, col3 = st.columns(3)
    with col1:
        ap_start = st.number_input("Start sector", min_value=1, max_value=100, value=1, key="ap_start")
    with col2:
        ap_end = st.number_input("End sector", min_value=1, max_value=100, value=26, key="ap_end")
    with col3:
        ap_limit = st.number_input("Stars per sector", min_value=1, max_value=100000, value=10000, key="ap_limit")

    ap_status = _task_status("autopilot")

    if ap_status["running"]:
        st.warning(f"Autopilot is running (PID {ap_status['pid']})")
        if st.button("Stop Autopilot", type="secondary"):
            _stop_background_task("autopilot")
            st.rerun()

        if AUTOPILOT_STATE.exists():
            try:
                ap_state = json.loads(AUTOPILOT_STATE.read_text())
                completed = ap_state.get("completed_sectors", [])
                current = ap_state.get("current_sector")
                total_sectors = ap_end - ap_start + 1
                done_count = len([s for s in completed if ap_start <= s <= ap_end])

                c1, c2, c3 = st.columns(3)
                c1.metric("Sectors Completed", f"{done_count}/{total_sectors}")
                c2.metric("Current Sector", str(current) if current else "Idle")
                c3.metric("Progress", f"{done_count/max(total_sectors,1)*100:.0f}%")

                if total_sectors > 0:
                    st.progress(done_count / total_sectors)
            except Exception:
                pass

        if ap_status["log_tail"]:
            with st.expander("Autopilot Log", expanded=False):
                st.code(ap_status["log_tail"], language="text")

        time.sleep(2)
        st.rerun()
    else:
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Start Autopilot", type="primary", use_container_width=True):
                cmd = [
                    sys.executable, str(ROOT / "run.py"), "autopilot",
                    "--start-sector", str(ap_start),
                    "--end-sector", str(ap_end),
                    "--limit", str(ap_limit),
                    "--checkpoint", checkpoint_path,
                    "--candidate-dir", candidate_dir,
                    "--strategy-profile", selected_profile,
                    "--period-min", str(prof["period_min"]),
                    "--period-max", str(prof["period_max"]),
                    "--nperiods", str(prof["nperiods"]),
                ]
                _start_background_task("autopilot", cmd)
                st.rerun()

        if AUTOPILOT_STATE.exists():
            st.subheader("Previous Autopilot State")
            try:
                ap_state = json.loads(AUTOPILOT_STATE.read_text())
                completed = ap_state.get("completed_sectors", [])
                st.text(f"Completed sectors: {len(completed)}")
                if completed:
                    st.text(f"Sectors: {completed[:20]}{'...' if len(completed) > 20 else ''}")
            except Exception:
                pass

# ========================== CANDIDATES TAB ==========================
with tab_candidates:
    st.header("Candidate Browser")
    st.caption("Browse, rank, and triage all detected exoplanet candidates.")

    from services import load_candidates
    from strategy_profiles import compute_candidate_score

    candidates = load_candidates(candidate_dir)

    if not candidates:
        st.info("No candidates found yet. Run a Hunt or Autopilot to generate candidates.")
    else:
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            status_filter = st.multiselect(
                "Discovery status",
                options=["NEW_CANDIDATE", "KNOWN", "Unknown"],
                default=["NEW_CANDIDATE", "KNOWN", "Unknown"],
            )
        with filter_col2:
            min_prob = st.slider("Min P(planet)", 0.0, 1.0, 0.0, 0.05, key="cand_min_prob")
        with filter_col3:
            sort_by = st.selectbox("Sort by", ["Unified Score", "P(planet)", "BLS Power", "Period"])

        rows = []
        for c in candidates:
            status = c.get("discovery_status", "Unknown")
            if status not in status_filter:
                continue
            prob = c.get("prob_planet", 0.0)
            if prob < min_prob:
                continue

            bls_power = c.get("bls_max_power", 0.0)
            centroid = c.get("centroid_offset", 0.0)
            if centroid is None:
                centroid = 0.0
            score = compute_candidate_score(prob, bls_power, centroid, selected_profile)

            rows.append({
                "TIC": c.get("tic_id", "?"),
                "Period (d)": round(c.get("best_period", 0), 4),
                "P(planet)": round(prob, 3),
                "BLS Power": round(bls_power, 6),
                "Centroid": round(centroid, 3) if centroid == centroid else "N/A",
                "Score": score,
                "Status": status,
                "Profile": c.get("strategy_profile", "unknown"),
                "_plot_path": c.get("_plot_path", ""),
                "_json_path": c.get("_json_path", ""),
            })

        if not rows:
            st.info("No candidates match the current filters.")
        else:
            import pandas as pd

            df = pd.DataFrame(rows)

            sort_map = {
                "Unified Score": "Score",
                "P(planet)": "P(planet)",
                "BLS Power": "BLS Power",
                "Period": "Period (d)",
            }
            df = df.sort_values(sort_map[sort_by], ascending=False).reset_index(drop=True)

            st.subheader(f"Top Candidates ({len(df)} total)")

            new_count = len(df[df["Status"] == "NEW_CANDIDATE"])
            known_count = len(df[df["Status"] == "KNOWN"])
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Candidates", len(df))
            mc2.metric("New (not in TOI)", new_count)
            mc3.metric("Known TOIs", known_count)

            display_cols = ["TIC", "Period (d)", "P(planet)", "BLS Power", "Score", "Status"]
            st.dataframe(
                df[display_cols],
                use_container_width=True,
                hide_index=False,
                column_config={
                    "P(planet)": st.column_config.ProgressColumn(
                        "P(planet)", min_value=0, max_value=1, format="%.3f",
                    ),
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=1, format="%.4f",
                    ),
                },
            )

            st.divider()
            st.subheader("Candidate Detail")
            if len(df) > 0:
                detail_idx = st.number_input(
                    "Select candidate row index", min_value=0,
                    max_value=len(df) - 1, value=0, key="cand_detail_idx",
                )
                row = df.iloc[detail_idx]
                dc1, dc2 = st.columns([1, 2])
                with dc1:
                    st.markdown(f"**TIC:** {row['TIC']}")
                    st.markdown(f"**Period:** {row['Period (d)']} d")
                    st.markdown(f"**P(planet):** {row['P(planet)']}")
                    st.markdown(f"**BLS Power:** {row['BLS Power']}")
                    st.markdown(f"**Unified Score:** {row['Score']}")
                    st.markdown(f"**Status:** {row['Status']}")

                    json_path = row.get("_json_path", "")
                    if json_path and os.path.exists(json_path):
                        with st.expander("Evidence JSON"):
                            st.json(json.loads(Path(json_path).read_text()))

                with dc2:
                    plot_path = row.get("_plot_path", "")
                    if plot_path and os.path.exists(plot_path):
                        st.image(plot_path, caption=f"Phase-folded view — TIC {row['TIC']}")
                    else:
                        st.info("No plot available for this candidate.")

            st.divider()
            st.subheader("Export Discovery Packet")
            if st.button("Generate Discovery Packet (JSON)"):
                packet = []
                for _, r in df.iterrows():
                    packet.append({
                        "tic_id": r["TIC"],
                        "period_days": r["Period (d)"],
                        "prob_planet": r["P(planet)"],
                        "bls_power": r["BLS Power"],
                        "unified_score": r["Score"],
                        "discovery_status": r["Status"],
                        "strategy_profile": selected_profile,
                    })
                packet_json = json.dumps(packet, indent=2)
                st.download_button(
                    "Download Packet",
                    data=packet_json,
                    file_name=f"discovery_packet_{selected_profile}.json",
                    mime="application/json",
                )

# ========================== LOGS TAB ==========================
with tab_logs:
    st.header("Logs & Timing")

    log_choice = st.selectbox("Log source", [
        "Hunt sector log", "Hunter timings", "Autopilot state",
        "Task runner logs",
    ])

    if log_choice == "Hunt sector log":
        sector_num = st.number_input("Sector", min_value=1, max_value=100, value=15, key="log_sector")
        log_path = ROOT / f"processed_stars_s{sector_num:03d}.txt"
        if log_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(log_path)
                st.metric("Entries", len(df))
                status_counts = df["STATUS"].value_counts()
                st.bar_chart(status_counts)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.code(log_path.read_text()[-5000:], language="text")
        else:
            st.info(f"No log file found at {log_path}")

    elif log_choice == "Hunter timings":
        timings_path = ROOT / "hunter_timings.csv"
        if timings_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(timings_path)
                st.metric("Stars Timed", len(df))
                if "TOTAL_MS" in df.columns:
                    avg_ms = df["TOTAL_MS"].mean()
                    st.metric("Avg Time/Star", f"{avg_ms:.0f} ms")
                st.dataframe(df.tail(100), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse timings: {e}")
        else:
            st.info("No timing data yet. Run a hunt first.")

    elif log_choice == "Autopilot state":
        if AUTOPILOT_STATE.exists():
            st.json(json.loads(AUTOPILOT_STATE.read_text()))
        else:
            st.info("No autopilot state file found.")

    elif log_choice == "Task runner logs":
        for task_name in ["hunt", "train", "autopilot"]:
            task_log = ROOT / f".task_{task_name}.log"
            if task_log.exists():
                with st.expander(f"{task_name.title()} task log"):
                    content = task_log.read_text()
                    st.code(content[-5000:] if len(content) > 5000 else content, language="text")
