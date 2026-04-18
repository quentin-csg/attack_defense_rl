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
    """Log cyber-domain metrics to TensorBoard at the end of each episode."""

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
    """Write per-episode and per-update metrics to a JSONL file."""

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
        self._last_n_updates: int = -1

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
            }
            self._write(record)

            # Write topology + per-node suspicion to a separate small file (overwrite each
            # episode).  Kept out of the JSONL to avoid bloating it with per-node vectors.
            topo = terminal_info.get("network_topology")
            per_node = {
                str(k): float(v)
                for k, v in terminal_info.get("per_node_suspicion", {}).items()
            }
            if topo is not None or per_node:
                try:
                    payload: dict[str, Any] = {}
                    if topo is not None:
                        payload["topology"] = topo
                    if per_node:
                        payload["per_node_suspicion"] = per_node
                    self._topo_path.write_text(json.dumps(payload), encoding="utf-8")
                except OSError:
                    pass

        # Capture train metrics from logger once per rollout.
        # Use num_timesteps // n_steps as the rollout counter — 1 write per
        # rollout regardless of n_epochs or SB3 version (n_updates increments
        # n_epochs times per rollout, which would produce n_epochs writes).
        n_steps: int = getattr(self.model, "n_steps", 2048)
        rollout_count: int = self.num_timesteps // n_steps if n_steps > 0 else -1
        if rollout_count != self._last_n_updates:
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
            self._last_n_updates = rollout_count

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
    """Compose the standard Phase 3 callback stack."""
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
