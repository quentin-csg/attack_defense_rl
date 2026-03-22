"""Custom SB3 callbacks for Phase 3 Red Team training.

Provides domain-specific TensorBoard metrics beyond what SB3 logs by default:
- Exfiltration rate (primary objective success)
- Detection rate (primary failure mode)
- Nodes compromised, max suspicion, episode reward, episode length
"""

from __future__ import annotations

import collections
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


def build_callback_list(
    eval_env: Any,
    log_dir: str | None,
    save_dir: str | None,
    save_freq: int,
    eval_freq: int,
    eval_episodes: int,
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

    Returns:
        CallbackList containing CyberMetricsCallback + MaskableEvalCallback +
        CheckpointCallback.
    """
    cyber_metrics = CyberMetricsCallback(verbose=0)

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

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=save_dir or "models/",
        name_prefix="red_agent",
        verbose=0,
    )

    return CallbackList([cyber_metrics, eval_callback, checkpoint_callback])
