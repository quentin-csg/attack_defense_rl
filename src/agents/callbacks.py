"""Custom SB3 callbacks for Phase 3 Red Team training.

Provides domain-specific TensorBoard metrics beyond what SB3 logs by default:
- Exfiltration rate (primary objective success)
- Detection rate (primary failure mode)
- Nodes compromised, max suspicion, episode reward, episode length

Also provides DashboardCallback that writes metrics to a JSONL file for the
Streamlit training dashboard.
"""

from __future__ import annotations

import collections
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
)


class CyberMetricsCallback(BaseCallback):
    """Log cyber-domain metrics to TensorBoard at the end of each episode.

    Reads the terminal info dict from ``CyberEnv`` when an episode ends.
    When using DummyVecEnv (SB3 standard), the terminal info is stored
    under ``info["terminal_info"]`` because the env auto-resets. This
    callback reads from ``terminal_info`` to get the correct values.

    Maintains a rolling window (last 100 episodes) for each metric.

    Metrics logged under the ``cyber/`` namespace:
        - ``cyber/exfiltration_rate``  — fraction of episodes ending in exfil
        - ``cyber/detection_rate``     — fraction of episodes ending in detection
        - ``cyber/mean_nodes_compromised``
        - ``cyber/mean_max_suspicion``
        - ``cyber/mean_episode_length``
        - ``cyber/mean_episode_reward``
    """

    def __init__(self, window: int = 100, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._window = window
        self._exfiltrated: collections.deque[float] = collections.deque(maxlen=window)
        self._detected: collections.deque[float] = collections.deque(maxlen=window)
        self._nodes_compromised: collections.deque[float] = collections.deque(maxlen=window)
        self._max_suspicion: collections.deque[float] = collections.deque(maxlen=window)
        self._episode_lengths: collections.deque[float] = collections.deque(maxlen=window)
        self._episode_rewards: collections.deque[float] = collections.deque(maxlen=window)

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals.get("dones", [])
        infos: list[dict[str, Any]] = self.locals.get("infos", [])

        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue

            # DummyVecEnv auto-resets on episode end, so `info` already belongs
            # to the NEW episode. The terminal episode's info is stored under
            # info["terminal_info"]. Fall back to info itself for non-vec envs.
            terminal_info: dict[str, Any] = info.get("terminal_info", info)

            exfiltrated = bool(terminal_info.get("exfiltrated", False))
            n_compromised = float(terminal_info.get("n_compromised", 0))
            max_suspicion = float(terminal_info.get("max_suspicion", 0.0))
            episode_length = float(terminal_info.get("step", 0))
            episode_reward = float(terminal_info.get("episode_reward", 0.0))

            # Use the explicit "detected" flag exposed by CyberEnv (Fix 6).
            # Fall back to suspicion heuristic for backward compatibility.
            detected = bool(terminal_info.get("detected", not exfiltrated and max_suspicion >= 100.0))

            self._exfiltrated.append(float(exfiltrated))
            self._detected.append(float(detected))
            self._nodes_compromised.append(n_compromised)
            self._max_suspicion.append(max_suspicion)
            self._episode_lengths.append(episode_length)
            self._episode_rewards.append(episode_reward)

        if self._exfiltrated:
            self.logger.record("cyber/exfiltration_rate", np.mean(self._exfiltrated))
            self.logger.record("cyber/detection_rate", np.mean(self._detected))
            self.logger.record("cyber/mean_nodes_compromised", np.mean(self._nodes_compromised))
            self.logger.record("cyber/mean_max_suspicion", np.mean(self._max_suspicion))
            self.logger.record("cyber/mean_episode_length", np.mean(self._episode_lengths))
            self.logger.record("cyber/mean_episode_reward", np.mean(self._episode_rewards))

        return True  # do not stop training


class DashboardCallback(BaseCallback):
    """Write per-episode and per-update metrics to a JSONL file.

    The Streamlit dashboard reads this file for live and replay modes.
    Two event types are written:

    - ``{"type": "episode", "timestep": N, ...cyber metrics...}``
      Written at the end of each episode.
    - ``{"type": "update", "timestep": N, ...train metrics...}``
      Written after each PPO policy update (entropy, losses, KL, etc.).

    Args:
        log_path: Path to the JSONL output file. Opened in append mode so
                  successive training runs accumulate in the same file.
                  Use ``reset_on_start=True`` to truncate instead.
        reset_on_start: If True, truncate the file at training start.
    """

    def __init__(
        self,
        log_path: str = "logs/dashboard_metrics.jsonl",
        reset_on_start: bool = False,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._log_path = Path(log_path)
        self._topo_path = Path(log_path).parent / "network_topology.json"
        self._reset_on_start = reset_on_start
        self._file: Any = None
        self._last_update_timestep: int = -1

    def _on_training_start(self) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if self._reset_on_start else "a"
        self._file = self._log_path.open(mode, buffering=1)  # line-buffered

    def _on_step(self) -> bool:
        dones: np.ndarray = self.locals.get("dones", [])
        infos: list[dict[str, Any]] = self.locals.get("infos", [])

        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue

            terminal_info: dict[str, Any] = info.get("terminal_info", info)

            exfiltrated = bool(terminal_info.get("exfiltrated", False))
            detected = bool(
                terminal_info.get(
                    "detected", not exfiltrated and terminal_info.get("max_suspicion", 0) >= 100.0
                )
            )
            record: dict[str, Any] = {
                "type": "episode",
                "timestep": int(self.num_timesteps),
                "wall_time": time.time(),
                "exfiltrated": exfiltrated,
                "detected": detected,
                "n_compromised": float(terminal_info.get("n_compromised", 0)),
                "max_suspicion": float(terminal_info.get("max_suspicion", 0.0)),
                "episode_length": float(terminal_info.get("step", 0)),
                "episode_reward": float(terminal_info.get("episode_reward", 0.0)),
                "per_node_suspicion": {
                    str(k): float(v)
                    for k, v in terminal_info.get("per_node_suspicion", {}).items()
                },
            }
            self._write(record)

            # Write topology to a separate small file (overwrite each episode).
            # Kept out of the JSONL to avoid bloating it with redundant topology data.
            topo = terminal_info.get("network_topology")
            if topo is not None:
                try:
                    self._topo_path.write_text(json.dumps(topo), encoding="utf-8")
                except OSError:
                    pass

        # Capture train metrics from logger after each PPO update.
        # SB3 flushes name_to_value after dump() — check by timestep change.
        if self.num_timesteps != self._last_update_timestep:
            name_to_value: dict[str, Any] = getattr(self.model.logger, "name_to_value", {})
            train_keys = {
                "train/entropy_loss",
                "train/policy_gradient_loss",
                "train/value_loss",
                "train/approx_kl",
                "train/clip_fraction",
                "train/explained_variance",
                "train/learning_rate",
            }
            train_data = {k.replace("train/", ""): float(v) for k, v in name_to_value.items() if k in train_keys}
            if train_data:
                update_record: dict[str, Any] = {
                    "type": "update",
                    "timestep": int(self.num_timesteps),
                    "wall_time": time.time(),
                    **train_data,
                }
                self._write(update_record)
                self._last_update_timestep = self.num_timesteps

        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _write(self, record: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(record) + "\n")


def build_callback_list(
    eval_env: Any,
    log_dir: str | None,
    save_dir: str | None,
    save_freq: int,
    eval_freq: int,
    eval_episodes: int,
    dashboard_log_path: str = "logs/dashboard_metrics.jsonl",
    reset_dashboard: bool = False,
) -> CallbackList:
    """Compose the standard Phase 3 callback stack.

    Uses MaskableEvalCallback (sb3-contrib) instead of the standard
    EvalCallback so that action masks are correctly applied during
    evaluation rollouts.

    Args:
        eval_env: Separate evaluation environment (not used for training).
        log_dir: TensorBoard log directory. None to disable eval logging.
        save_dir: Directory where model checkpoints are saved. None to disable saving.
        save_freq: Save a checkpoint every this many env steps.
        eval_freq: Run evaluation every this many env steps.
        eval_episodes: Number of episodes per evaluation run.
        dashboard_log_path: Path for the JSONL dashboard metrics file.
        reset_dashboard: If True, truncate the JSONL file at training start.

    Returns:
        CallbackList containing CyberMetricsCallback + DashboardCallback +
        MaskableEvalCallback + CheckpointCallback.
    """
    cyber_metrics = CyberMetricsCallback(verbose=0)
    dashboard = DashboardCallback(log_path=dashboard_log_path, reset_on_start=reset_dashboard)

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    callbacks: list[Any] = [cyber_metrics, dashboard, eval_callback]

    if save_dir is not None:
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=save_dir,
                name_prefix="red_agent",
                verbose=0,
            )
        )

    return CallbackList(callbacks)
