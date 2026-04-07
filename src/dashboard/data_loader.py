from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def _parse_jsonl(path: str, _mtime: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the JSONL file and return (episode_df, update_df).

    Reads line-by-line to avoid loading the full file into memory.
    Drops ``per_node_suspicion`` from episode records (now stored in
    ``network_topology.json`` instead).  Update records are capped at
    MAX_UPDATE_ROWS to bound memory on legacy bloated files.
    """
    MAX_UPDATE_ROWS = 2_000

    episodes: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []
    update_step = 1  # will be recomputed once we know total lines

    # First pass: count update lines to decide sampling stride.
    # We do a lightweight count-only scan so we know whether to sample.
    n_updates_total = 0
    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                if '"type": "update"' in line:
                    n_updates_total += 1
    except OSError:
        return pd.DataFrame(), pd.DataFrame()

    if n_updates_total > MAX_UPDATE_ROWS:
        update_step = max(1, n_updates_total // MAX_UPDATE_ROWS)

    update_counter = 0
    try:
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record: dict[str, Any] = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = record.get("type")
                if t == "episode":
                    record.pop("per_node_suspicion", None)
                    episodes.append(record)
                elif t == "update":
                    if update_counter % update_step == 0:
                        updates.append(record)
                    update_counter += 1
    except OSError:
        pass

    return (
        pd.DataFrame(episodes) if episodes else pd.DataFrame(),
        pd.DataFrame(updates) if updates else pd.DataFrame(),
    )


def load_episode_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load episode-type rows from the JSONL file."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    ep_df, _ = _parse_jsonl(path, p.stat().st_mtime)
    return ep_df


def load_update_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load update-type rows (PPO training metrics) from the JSONL file."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    _, upd_df = _parse_jsonl(path, p.stat().st_mtime)
    return upd_df


def load_evaluations(path: str = "logs/evaluations.npz") -> pd.DataFrame:
    """Load the MaskableEvalCallback evaluations.npz file."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    try:
        data = np.load(p)
    except Exception:
        return pd.DataFrame()

    timesteps = data["timesteps"]
    results = data["results"]
    ep_lengths = data["ep_lengths"]

    return pd.DataFrame(
        {
            "timestep": timesteps,
            "mean_reward": results.mean(axis=1),
            "std_reward": results.std(axis=1),
            "mean_length": ep_lengths.mean(axis=1),
            "std_length": ep_lengths.std(axis=1),
        }
    )


def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling mean, returning the series as-is if too short."""
    return series.rolling(window=min(window, max(1, len(series))), min_periods=1).mean()


def downsample(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """Reduce a DataFrame to at most ``max_points`` rows by uniform sampling."""
    if len(df) <= max_points:
        return df
    indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[indices].reset_index(drop=True)
