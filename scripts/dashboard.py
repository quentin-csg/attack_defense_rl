"""Streamlit training dashboard for the Attack & Defense RL project.

Launch with:
    streamlit run scripts/dashboard.py

Modes:
    Live   — auto-refreshes every N seconds to show metrics during training
    Replay — loads a saved JSONL file for post-training exploration
             + step slider to scrub through the training history

Layout:
    ┌─────────────────────────────────────────────────┐
    │  Attack & Defense RL — Training Dashboard       │
    ├──────────┬──────────┬──────────┬────────────────│
    │Timesteps │ Episodes │ Exfil %  │ Detection %    │  KPI row
    ├──────────┴──────────┴──────────┴────────────────│
    │  Training Curves    │   Cyber Metrics           │
    │  (Loss, Entropy)    │   (Exfil, Nodes, Susp.)   │
    ├─────────────────────┼───────────────────────────│
    │  Network Graph      │   Eval Checkpoints        │
    │  (heatmap)          │   (reward ± std)          │
    └─────────────────────┴───────────────────────────┘
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Allow `src/` imports when running from the project root.
# `streamlit run scripts/dashboard.py` sets cwd to the project root,
# so adding "." is enough.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dashboard.charts import (
    render_cyber_metrics,
    render_eval_curves,
    render_kpi_header,
    render_training_curves,
)
from src.dashboard.data_loader import (
    load_episode_metrics,
    load_evaluations,
    load_update_metrics,
)
from src.dashboard.network_view import render_network_graph

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Attack & Defense RL",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 1rem; }
        [data-testid="stMetricValue"] { font-size: 1.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.title("Controls")

mode = st.sidebar.radio("Mode", ["Live", "Replay"], index=0)

# --- Run selector ---
_logs_root = Path("logs")
_runs = sorted(
    [d.name for d in _logs_root.iterdir() if d.is_dir() and (d / "dashboard_metrics.jsonl").exists()],
    reverse=True,
) if _logs_root.exists() else []

if _runs:
    selected_run = st.sidebar.selectbox("Run", _runs, index=0)
    _run_dir = _logs_root / selected_run
    metrics_path = str(_run_dir / "dashboard_metrics.jsonl")
    eval_path = str(_run_dir / "evaluations.npz")
    st.sidebar.caption(f"Run: `{selected_run}`")
else:
    metrics_path = st.sidebar.text_input(
        "Metrics file",
        value="logs/dashboard_metrics.jsonl",
        help="Path to the JSONL file written by DashboardCallback during training.",
    )
    eval_path = st.sidebar.text_input(
        "Eval file",
        value="logs/evaluations.npz",
        help="Path to the evaluations.npz written by MaskableEvalCallback.",
    )

window = st.sidebar.slider(
    "Rolling window (episodes)",
    min_value=10,
    max_value=500,
    value=50,
    step=10,
    help="Number of recent episodes for computing rolling averages.",
)

refresh_interval = None
if mode == "Live":
    refresh_interval = st.sidebar.slider(
        "Refresh interval (s)",
        min_value=1,
        max_value=15,
        value=3,
        help="How often the dashboard reloads data from disk.",
    )


# ---------------------------------------------------------------------------
# Title
# ---------------------------------------------------------------------------

st.title("Attack & Defense RL — Training Dashboard")
if mode == "Live":
    st.caption(f"Live mode — refreshes every {refresh_interval}s")
else:
    st.caption("Replay mode — reading saved logs")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

ep_df = load_episode_metrics(metrics_path)
upd_df = load_update_metrics(metrics_path)
eval_df = load_evaluations(eval_path)

# ---------------------------------------------------------------------------
# Replay step slider — filter all data to the selected timestep
# ---------------------------------------------------------------------------

if mode == "Replay" and not ep_df.empty:
    max_ts = int(ep_df["timestep"].max())
    min_ts = int(ep_df["timestep"].min())

    selected_step = st.slider(
        "Replay — timestep",
        min_value=min_ts,
        max_value=max_ts,
        value=max_ts,
        step=max(1, (max_ts - min_ts) // 200),  # ~200 stops across the range
        format="%d steps",
        help="Scrub through training history — all charts update to reflect state at this timestep.",
    )

    # Filter episode and update dataframes up to selected_step
    ep_df = ep_df[ep_df["timestep"] <= selected_step].copy()
    upd_df = upd_df[upd_df["timestep"] <= selected_step].copy() if not upd_df.empty else upd_df
    eval_df = eval_df[eval_df["timestep"] <= selected_step].copy() if not eval_df.empty else eval_df

    n_ep_shown = len(ep_df)
    st.caption(f"Showing {n_ep_shown} episodes up to step {selected_step:,}")

# ---------------------------------------------------------------------------
# KPI header
# ---------------------------------------------------------------------------

render_kpi_header(ep_df)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 1: Training curves + Cyber metrics
# ---------------------------------------------------------------------------

col_left, col_right = st.columns(2, gap="large")

with col_left:
    render_training_curves(upd_df, window=window)

with col_right:
    render_cyber_metrics(ep_df, window=window)

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2: Network graph + Eval curves
# ---------------------------------------------------------------------------

col_graph, col_eval = st.columns(2, gap="large")

with col_graph:
    # Colour nodes by per-node suspicion from the last episode in the current view.
    suspicion_data: dict[int, float] | None = None
    if not ep_df.empty and "per_node_suspicion" in ep_df.columns:
        last_susp = ep_df["per_node_suspicion"].dropna()
        if not last_susp.empty and isinstance(last_susp.iloc[-1], dict):
            suspicion_data = {int(k): float(v) for k, v in last_susp.iloc[-1].items()}
    render_network_graph(suspicion_data=suspicion_data)

with col_eval:
    render_eval_curves(eval_df)

# ---------------------------------------------------------------------------
# Auto-refresh in live mode
# ---------------------------------------------------------------------------

if mode == "Live" and refresh_interval is not None:
    import time as _time
    jsonl_mtime = os.path.getmtime(metrics_path) if os.path.exists(metrics_path) else 0
    stale = (_time.time() - jsonl_mtime) > 30
    if stale:
        st.success("Training terminé — auto-refresh désactivé. Passe en mode Replay pour explorer les données.")
    else:
        _time.sleep(refresh_interval)
        st.rerun()
