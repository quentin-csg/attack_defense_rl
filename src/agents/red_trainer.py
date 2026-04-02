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


def _make_eval_blue_team(blue_team: object, seed: int) -> object:
    """Return an independent Blue Team instance for an eval environment."""
    if blue_team is None:
        return None
    from src.agents.blue_scripted import ScriptedBlueTeam

    return ScriptedBlueTeam(seed=seed)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear learning rate schedule: LR decays from initial_value to 0."""
    return lambda progress_remaining: initial_value * progress_remaining


def create_model(
    env: Any,
    log_dir: str | None = None,
    seed: int = 42,
) -> MaskablePPO:
    """Create a MaskablePPO model with Phase 3 hyperparameters."""
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
    pcg_size: str | None = None,
    load_model_path: str | None = None,
) -> MaskablePPO:
    """Run the full Red Team training pipeline."""
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

    eval_blue = _make_eval_blue_team(blue_team, seed + 9999)
    if pcg_size is not None:
        from src.agents.wrappers import make_pcg_masked_env
        train_env = make_pcg_masked_env(size=pcg_size, seed=seed, max_steps=max_steps, blue_team=blue_team)
        eval_env = make_pcg_masked_env(size=pcg_size, seed=seed + 1000, max_steps=max_steps, blue_team=eval_blue)
    else:
        train_env = make_masked_env(seed=seed, max_steps=max_steps, blue_team=blue_team)
        eval_env = make_masked_env(seed=seed + 1000, max_steps=max_steps, blue_team=eval_blue)

    model: MaskablePPO | None = None  # guard against UnboundLocalError if create_model raises
    try:
        if load_model_path is not None:
            model = MaskablePPO.load(load_model_path, env=train_env, device="auto")
            model.tensorboard_log = log_dir
            logger.info("Loaded model from %s", load_model_path)
        else:
            model = create_model(train_env, log_dir=log_dir, seed=seed)

        # Derive dashboard log path from log_dir if not explicitly provided
        if not dashboard_log_path and not log_dir:
            Path("logs").mkdir(parents=True, exist_ok=True)
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


def train_curriculum(
    curriculum: Any,  # CurriculumManager — lazy import to avoid circular dep
    seed: int = 42,
    log_dir: str | None = "logs/",
    save_dir: str | None = "models/",
    blue_team: object = None,
    eval_freq: int = RL_EVAL_FREQ,
    eval_episodes: int = RL_EVAL_EPISODES,
    save_freq: int = RL_SAVE_FREQ,
) -> MaskablePPO:
    """Train the Red agent through a curriculum of increasing network sizes."""
    from src.agents.wrappers import make_pcg_masked_env
    from src.pcg.generator import generate_network

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    model: MaskablePPO | None = None
    world_count = 0

    try:
        while not curriculum.is_complete:
            net, meta = curriculum.generate_current_network()
            stage_name = curriculum.current_stage.size.value
            timesteps = curriculum.current_stage.timesteps_per_world

            logger.info(
                "Curriculum world %d | stage=%s | nodes=%d | hops=%d | steps=%d",
                world_count,
                stage_name,
                meta.n_nodes,
                meta.min_hops,
                timesteps,
            )

            # Build env with the fixed network for this world
            # (factory always returns the same net regardless of seed)
            _fixed_net = net

            def _factory(_s: int | None, _n=_fixed_net):
                from src.environment.network import Network
                # Reset node states for new episode but keep topology
                _n.reset_all_nodes()
                return _n

            from src.environment.cyber_env import CyberEnv
            from sb3_contrib.common.wrappers import ActionMasker
            from stable_baselines3.common.monitor import Monitor

            train_env = None
            eval_env = None
            try:
                base_env = CyberEnv(
                    network_factory=_factory,
                    max_steps=meta.recommended_max_steps,
                    seed=seed + world_count,
                    blue_team=blue_team,
                )
                train_env = ActionMasker(Monitor(base_env), action_mask_fn=lambda e: base_env.action_masks())

                # Eval env: PCG for the current stage so eval sees varied topologies.
                # Use a separate Blue Team instance to avoid shared RNG state with train_env.
                eval_blue = _make_eval_blue_team(blue_team, seed + world_count + 9999)
                eval_env = make_pcg_masked_env(
                    size=stage_name,
                    seed=seed + world_count + 9999,
                    max_steps=meta.recommended_max_steps,
                    blue_team=eval_blue,
                )

                if log_dir:
                    dash_path = f"{log_dir}/dashboard_metrics.jsonl"
                else:
                    Path("logs").mkdir(parents=True, exist_ok=True)
                    dash_path = "logs/dashboard_metrics.jsonl"

                if model is None:
                    model = create_model(train_env, log_dir=log_dir, seed=seed)
                else:
                    model.set_env(train_env)

                callbacks = build_callback_list(
                    eval_env=eval_env,
                    log_dir=log_dir,
                    save_dir=save_dir,
                    save_freq=save_freq,
                    eval_freq=eval_freq,
                    eval_episodes=eval_episodes,
                    dashboard_log_path=dash_path,
                    reset_dashboard=(world_count == 0),
                )

                model.learn(
                    total_timesteps=timesteps,
                    callback=callbacks,
                    log_interval=RL_LOG_INTERVAL,
                    reset_num_timesteps=(world_count == 0),
                )
            finally:
                if train_env is not None:
                    train_env.close()
                if eval_env is not None:
                    eval_env.close()

            curriculum.advance_world()
            world_count += 1

    except KeyboardInterrupt:
        logger.info("Curriculum training interrupted by user.")

    if model is not None and save_dir is not None:
        final_path = str(Path(save_dir) / "red_agent_curriculum_final")
        model.save(final_path)
        logger.info("Curriculum model saved to %s.zip", final_path)

    return model


def load_model(path: str, env: Any = None) -> MaskablePPO:
    """Load a saved MaskablePPO model."""
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
    """Evaluate a trained model over N episodes."""
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
