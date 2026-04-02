from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def _load_metrics_cached(path: str, _mtime: float) -> pd.DataFrame:
    """Read the full JSONL file. ``_mtime`` is used only as a cache key."""
    records: list[dict[str, Any]] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()
    return pd.DataFrame(records) if records else pd.DataFrame()


def load_metrics(
    path: str = "logs/dashboard_metrics.jsonl",
    last_n: int | None = None,
) -> pd.DataFrame:
    """Load the JSONL metrics file into a DataFrame."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    mtime = p.stat().st_mtime
    df = _load_metrics_cached(path, mtime)

    if last_n is not None and not df.empty:
        df = df.tail(last_n).reset_index(drop=True)

    return df


def load_episode_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load only episode-type rows from the JSONL file."""
    df = load_metrics(path)
    if df.empty or "type" not in df.columns:
        return pd.DataFrame()
    ep = df[df["type"] == "episode"].copy()
    return ep.reset_index(drop=True)


def load_update_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load only update-type rows (PPO training metrics) from the JSONL file."""
    df = load_metrics(path)
    if df.empty or "type" not in df.columns:
        return pd.DataFrame()
    upd = df[df["type"] == "update"].copy()
    return upd.reset_index(drop=True)


def load_evaluations(path: str = "logs/evaluations.npz") -> pd.DataFrame:
    """Load the MaskableEvalCallback evaluations.npz file."""
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    try:
        data = np.load(p)
    except Exception:
        return pd.DataFrame()

    timesteps = data["timesteps"]         # shape (N,)
    results = data["results"]             # shape (N, n_eval_episodes)
    ep_lengths = data["ep_lengths"]       # shape (N, n_eval_episodes)

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
