"""Streamlit chart rendering for the training dashboard.

All functions use Streamlit-native charts (Altair under the hood) to stay
lightweight. Each function is self-contained and safe to call with an empty
DataFrame — it will show an informational message instead of crashing.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from src.dashboard.data_loader import downsample, rolling_mean

# ---------------------------------------------------------------------------
# KPI Header
# ---------------------------------------------------------------------------


def render_kpi_header(ep_df: pd.DataFrame) -> None:
    """Render a top row of st.metric() KPI chips.

    Args:
        ep_df: Episode-type DataFrame from load_episode_metrics().
    """
    col1, col2, col3, col4, col5 = st.columns(5)

    if ep_df.empty:
        col1.metric("Timesteps", "—")
        col2.metric("Episodes", "—")
        col3.metric("Exfil Rate", "—")
        col4.metric("Detection Rate", "—")
        col5.metric("Mean Reward", "—")
        return

    total_timesteps = int(ep_df["timestep"].max()) if "timestep" in ep_df.columns else 0
    total_episodes = len(ep_df)

    # Current window (last 50 episodes) vs previous window for delta
    window = 50
    recent = ep_df.tail(window)
    previous = ep_df.iloc[-2 * window : -window] if len(ep_df) >= 2 * window else None

    exfil_rate = recent["exfiltrated"].mean() if "exfiltrated" in recent.columns else 0.0
    detect_rate = recent["detected"].mean() if "detected" in recent.columns else 0.0
    mean_reward = recent["episode_reward"].mean() if "episode_reward" in recent.columns else 0.0

    exfil_delta = detect_delta = reward_delta = None
    if previous is not None and not previous.empty:
        exfil_delta = float(exfil_rate - previous["exfiltrated"].mean())
        detect_delta = float(detect_rate - previous["detected"].mean())
        reward_delta = float(mean_reward - previous["episode_reward"].mean())

    col1.metric("Timesteps", f"{total_timesteps:,}")
    col2.metric("Episodes", f"{total_episodes:,}")
    col3.metric(
        "Exfil Rate",
        f"{exfil_rate:.0%}",
        delta=f"{exfil_delta:+.1%}" if exfil_delta is not None else None,
    )
    col4.metric(
        "Detection Rate",
        f"{detect_rate:.0%}",
        delta=f"{detect_delta:+.1%}" if detect_delta is not None else None,
        delta_color="inverse",
    )
    col5.metric(
        "Mean Reward",
        f"{mean_reward:+.1f}",
        delta=f"{reward_delta:+.1f}" if reward_delta is not None else None,
    )


# ---------------------------------------------------------------------------
# Training Curves
# ---------------------------------------------------------------------------


def render_training_curves(upd_df: pd.DataFrame, window: int = 10) -> None:
    """Render PPO training metric curves (reward, losses, entropy).

    Args:
        upd_df: Update-type DataFrame from load_update_metrics().
        window: Rolling mean window size.
    """
    st.subheader("Training Curves")

    if upd_df.empty:
        st.info("No training update data yet. Start training to see metrics.")
        return

    df = downsample(upd_df.copy())
    x = "timestep"

    def _line_chart(col: str, label: str, color: str = "#4c9be8") -> None:
        if col not in df.columns:
            return
        plot_df = pd.DataFrame(
            {
                x: df[x],
                label: rolling_mean(df[col].astype(float), window),
            }
        )
        chart = (
            alt.Chart(plot_df)
            .mark_line(color=color, strokeWidth=2)
            .encode(
                x=alt.X(f"{x}:Q", title="Timesteps"),
                y=alt.Y(f"{label}:Q", title=label),
                tooltip=[x, label],
            )
            .properties(height=140)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _line_chart("entropy_loss", "Entropy", "#a8d8a8")
        _line_chart("value_loss", "Value Loss", "#f4a261")
    with c2:
        _line_chart("policy_gradient_loss", "Policy Loss", "#e07070")
        _line_chart("approx_kl", "Approx KL", "#c77dff")


# ---------------------------------------------------------------------------
# Cyber Metrics
# ---------------------------------------------------------------------------


def render_cyber_metrics(ep_df: pd.DataFrame, window: int = 50) -> None:
    """Render cyber domain metrics (exfil, detection, nodes, suspicion, length).

    Args:
        ep_df: Episode-type DataFrame from load_episode_metrics().
        window: Rolling mean window size.
    """
    st.subheader("Cyber Metrics")

    if ep_df.empty:
        st.info("No episode data yet. Start training to see metrics.")
        return

    df = downsample(ep_df.copy())
    x = "timestep"

    def _dual_line(col_a: str, label_a: str, col_b: str, label_b: str) -> None:
        cols_needed = [x, col_a, col_b]
        if not all(c in df.columns for c in cols_needed):
            return
        plot_df = pd.DataFrame(
            {
                x: df[x],
                label_a: rolling_mean(df[col_a].astype(float), window),
                label_b: rolling_mean(df[col_b].astype(float), window),
            }
        ).melt(x, var_name="metric", value_name="value")
        chart = (
            alt.Chart(plot_df)
            .mark_line(strokeWidth=2)
            .encode(
                x=alt.X(f"{x}:Q", title="Timesteps"),
                y=alt.Y("value:Q", title="Rate"),
                color=alt.Color("metric:N", scale=alt.Scale(range=["#4c9be8", "#e07070"])),
                tooltip=[x, "metric", "value"],
            )
            .properties(height=140)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    def _single_line(col: str, label: str, color: str = "#a8d8a8") -> None:
        if col not in df.columns:
            return
        plot_df = pd.DataFrame(
            {
                x: df[x],
                label: rolling_mean(df[col].astype(float), window),
            }
        )
        chart = (
            alt.Chart(plot_df)
            .mark_line(color=color, strokeWidth=2)
            .encode(
                x=alt.X(f"{x}:Q", title="Timesteps"),
                y=alt.Y(f"{label}:Q", title=label),
                tooltip=[x, label],
            )
            .properties(height=140)
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        _dual_line("exfiltrated", "Exfil Rate", "detected", "Detection Rate")
        _single_line("n_compromised", "Nodes Compromised", "#f4a261")
    with c2:
        _single_line("episode_reward", "Episode Reward", "#4c9be8")
        _single_line("max_suspicion", "Max Suspicion", "#c77dff")


# ---------------------------------------------------------------------------
# Eval Curves
# ---------------------------------------------------------------------------


def render_eval_curves(eval_df: pd.DataFrame) -> None:
    """Render evaluation checkpoint curves with reward ± std band.

    Args:
        eval_df: DataFrame from load_evaluations().
    """
    st.subheader("Evaluation Checkpoints")

    if eval_df.empty:
        st.info("No evaluation data yet (evaluations.npz not found).")
        return

    # Reward mean line
    df = eval_df.copy()
    df["reward_upper"] = df["mean_reward"] + df["std_reward"]
    df["reward_lower"] = df["mean_reward"] - df["std_reward"]

    band = (
        alt.Chart(df)
        .mark_area(opacity=0.2, color="#4c9be8")
        .encode(
            x=alt.X("timestep:Q", title="Timesteps"),
            y=alt.Y("reward_lower:Q", title="Reward"),
            y2="reward_upper:Q",
        )
    )
    line = (
        alt.Chart(df)
        .mark_line(color="#4c9be8", strokeWidth=2)
        .encode(
            x="timestep:Q",
            y="mean_reward:Q",
            tooltip=["timestep", "mean_reward", "std_reward"],
        )
    )
    st.altair_chart((band + line).properties(height=160).interactive(), use_container_width=True)

    # Episode length
    len_chart = (
        alt.Chart(df)
        .mark_line(color="#a8d8a8", strokeWidth=2)
        .encode(
            x=alt.X("timestep:Q", title="Timesteps"),
            y=alt.Y("mean_length:Q", title="Avg Episode Length"),
            tooltip=["timestep", "mean_length"],
        )
        .properties(height=140)
        .interactive()
    )
    st.altair_chart(len_chart, use_container_width=True)
