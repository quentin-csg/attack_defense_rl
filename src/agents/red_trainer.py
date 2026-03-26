"""Red Team MaskablePPO training pipeline.

Entry point for Phase 3 training. Provides functions to create, train,
save, load, and evaluate the Red Team agent.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sb3_contrib import MaskablePPO

# Disable PyTorch distribution validation globally.
# With 700 masked actions (14 × 50), the masked softmax can produce probabilities
# that sum to 1.0 ± 1e-7 due to float32 precision, which fails PyTorch's Simplex
# constraint check. This is a known sb3-contrib issue — disabling validate_args
# removes the check without affecting training correctness.
torch.distributions.Distribution.set_default_validate_args(False)

from src.agents.callbacks import build_callback_list
from src.agents.wrappers import make_masked_env
from src.config import (
    DEFAULT_MAX_STEPS,
    RL_BATCH_SIZE,
    RL_CLIP_RANGE,
    RL_ENT_COEF,
    RL_EVAL_EPISODES,
    RL_EVAL_FREQ,
    RL_GAMMA,
    RL_LEARNING_RATE,
    RL_LOG_INTERVAL,
    RL_N_EPOCHS,
    RL_N_STEPS,
    RL_NET_ARCH,
    RL_SAVE_FREQ,
    RL_TOTAL_TIMESTEPS,
)

logger = logging.getLogger(__name__)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule: LR decays from initial_value to 0."""
    return lambda progress_remaining: initial_value * progress_remaining


def create_model(
    env: Any,
    log_dir: str | None = None,
    seed: int = 42,
) -> MaskablePPO:
    """Create a MaskablePPO model with Phase 3 hyperparameters.

    Args:
        env: ActionMasker-wrapped CyberEnv (or DummyVecEnv of masked envs).
        log_dir: TensorBoard log directory. None to disable logging.
        seed: Random seed for reproducibility.

    Returns:
        Configured MaskablePPO model (not yet trained).
    """
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=linear_schedule(RL_LEARNING_RATE),
        n_steps=RL_N_STEPS,
        batch_size=RL_BATCH_SIZE,
        n_epochs=RL_N_EPOCHS,
        ent_coef=RL_ENT_COEF,
        gamma=RL_GAMMA,
        clip_range=RL_CLIP_RANGE,
        policy_kwargs={"net_arch": list(RL_NET_ARCH)},
        tensorboard_log=log_dir,
        seed=seed,
        verbose=1,
        device="auto",
    )


def train(
    total_timesteps: int = RL_TOTAL_TIMESTEPS,
    seed: int = 42,
    log_dir: str | None = "logs/",
    save_dir: str | None = "models/",
    max_steps: int = DEFAULT_MAX_STEPS,
    eval_freq: int = RL_EVAL_FREQ,
    eval_episodes: int = RL_EVAL_EPISODES,
    save_freq: int = RL_SAVE_FREQ,
    dashboard_log_path: str | None = None,
    blue_team: object = None,
) -> MaskablePPO:
    """Run the full Red Team training pipeline.

    Creates environments, model, and callbacks, then calls model.learn().
    On completion or KeyboardInterrupt, saves the model before returning.

    Args:
        total_timesteps: Total environment steps for training.
        seed: Random seed (eval env uses seed + 1000).
        log_dir: TensorBoard log directory. None to disable TensorBoard.
        save_dir: Directory for model checkpoints and final model.
                  None to disable checkpoint saving.
        max_steps: Maximum steps per episode.
        eval_freq: Evaluate every this many env steps.
        eval_episodes: Number of episodes per evaluation.
        save_freq: Save a checkpoint every this many env steps.

    Returns:
        Trained MaskablePPO model (env is closed; attach a new env for
        further training).
    """
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    logger.info(
        "Starting training | timesteps=%d seed=%d log_dir=%s save_dir=%s",
        total_timesteps,
        seed,
        log_dir,
        save_dir,
    )

    train_env = make_masked_env(seed=seed, max_steps=max_steps, blue_team=blue_team)
    eval_env = make_masked_env(seed=seed + 1000, max_steps=max_steps, blue_team=blue_team)

    model: MaskablePPO | None = None  # guard against UnboundLocalError if create_model raises
    try:
        model = create_model(train_env, log_dir=log_dir, seed=seed)

        # Derive dashboard log path from log_dir if not explicitly provided
        dash_path = dashboard_log_path or (
            f"{log_dir}/dashboard_metrics.jsonl" if log_dir else "logs/dashboard_metrics.jsonl"
        )

        callbacks = build_callback_list(
            eval_env=eval_env,
            log_dir=log_dir,
            save_dir=save_dir,
            save_freq=save_freq,
            eval_freq=eval_freq,
            eval_episodes=eval_episodes,
            dashboard_log_path=dash_path,
            reset_dashboard=True,
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=RL_LOG_INTERVAL,
            reset_num_timesteps=True,
        )

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    finally:
        train_env.close()
        eval_env.close()

    if model is not None and save_dir is not None:
        final_path = str(Path(save_dir) / "red_agent_final")
        model.save(final_path)
        logger.info("Final model saved to %s.zip", final_path)

    return model


def load_model(path: str, env: Any = None) -> MaskablePPO:
    """Load a saved MaskablePPO model.

    Args:
        path: Path to the saved model (.zip extension optional).
        env: Environment to attach to the model. If None, the model
             can still be used for inference but not for further training.

    Raises:
        FileNotFoundError: If the model file does not exist.

    Returns:
        Loaded MaskablePPO model.
    """
    zip_path = Path(path) if str(path).endswith(".zip") else Path(str(path) + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Model file not found: {zip_path}")
    return MaskablePPO.load(path, env=env)


def evaluate(
    model: MaskablePPO,
    env: Any,
    n_episodes: int = 100,
    deterministic: bool = True,
) -> dict[str, float]:
    """Evaluate a trained model over N episodes.

    Args:
        model: Trained MaskablePPO model.
        env: ActionMasker-wrapped CyberEnv.
        n_episodes: Number of episodes to run.
        deterministic: If True, use the greedy policy; else sample.

    Returns:
        Dict with keys: exfiltration_rate, detection_rate,
        mean_episode_reward, mean_episode_length, mean_nodes_compromised,
        mean_max_suspicion.
    """
    exfiltrated_list: list[float] = []
    detected_list: list[float] = []
    rewards: list[float] = []
    lengths: list[float] = []
    nodes_compromised: list[float] = []
    max_suspicions: list[float] = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=deterministic, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated

        exfiltrated = bool(info.get("exfiltrated", False))
        # Use explicit detected flag if available (Fix 6); fall back to heuristic.
        detected = bool(info.get("detected", not exfiltrated and info.get("max_suspicion", 0) >= 100.0))

        exfiltrated_list.append(float(exfiltrated))
        detected_list.append(float(detected))
        rewards.append(float(info.get("episode_reward", 0.0)))
        lengths.append(float(info.get("step", 0)))
        nodes_compromised.append(float(info.get("n_compromised", 0)))
        max_suspicions.append(float(info.get("max_suspicion", 0.0)))

    return {
        "exfiltration_rate": float(np.mean(exfiltrated_list)),
        "detection_rate": float(np.mean(detected_list)),
        "mean_episode_reward": float(np.mean(rewards)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_nodes_compromised": float(np.mean(nodes_compromised)),
        "mean_max_suspicion": float(np.mean(max_suspicions)),
    }
