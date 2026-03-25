"""Data loading utilities for the Streamlit training dashboard.

Reads metrics from two sources:
- ``logs/dashboard_metrics.jsonl`` — written by DashboardCallback during training
- ``logs/evaluations.npz`` — written by MaskableEvalCallback at each eval checkpoint
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_metrics(
    path: str = "logs/dashboard_metrics.jsonl",
    last_n: int | None = None,
) -> pd.DataFrame:
    """Load the JSONL metrics file into a DataFrame.

    Args:
        path: Path to the dashboard_metrics.jsonl file.
        last_n: If set, return only the last N rows (useful for live mode
                where we only want the recent window).

    Returns:
        DataFrame with columns depending on event type.
        All rows have ``type`` and ``timestep`` columns.
        Episode rows have cyber metric columns.
        Update rows have training metric columns.
        Returns an empty DataFrame if the file does not exist or is empty.
    """
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    try:
        lines = p.read_text(encoding="utf-8").splitlines()
        if last_n is not None:
            lines = lines[-last_n:]
        for line in lines:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    except (OSError, json.JSONDecodeError):
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def load_episode_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load only episode-type rows from the JSONL file.

    Returns:
        DataFrame with columns: timestep, wall_time, exfiltrated, detected,
        n_compromised, max_suspicion, episode_length, episode_reward.
        Empty DataFrame if no episode events exist.
    """
    df = load_metrics(path)
    if df.empty or "type" not in df.columns:
        return pd.DataFrame()
    ep = df[df["type"] == "episode"].copy()
    return ep.reset_index(drop=True)


def load_update_metrics(path: str = "logs/dashboard_metrics.jsonl") -> pd.DataFrame:
    """Load only update-type rows (PPO training metrics) from the JSONL file.

    Returns:
        DataFrame with columns: timestep, wall_time, entropy_loss,
        policy_gradient_loss, value_loss, approx_kl, clip_fraction,
        explained_variance, learning_rate.
        Empty DataFrame if no update events exist.
    """
    df = load_metrics(path)
    if df.empty or "type" not in df.columns:
        return pd.DataFrame()
    upd = df[df["type"] == "update"].copy()
    return upd.reset_index(drop=True)


def load_evaluations(path: str = "logs/evaluations.npz") -> pd.DataFrame:
    """Load the MaskableEvalCallback evaluations.npz file.

    Args:
        path: Path to evaluations.npz.

    Returns:
        DataFrame with columns: timestep, mean_reward, std_reward, mean_length, std_length.
        Empty DataFrame if the file does not exist.
    """
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
