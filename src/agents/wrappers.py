"""SB3 wrappers for CyberEnv compatibility with MaskablePPO.

Provides factory functions that create properly wrapped environments
ready to be passed to MaskablePPO.learn().
"""

from __future__ import annotations

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.environment.cyber_env import CyberEnv


def make_masked_env(
    seed: int | None = None,
    max_steps: int = 200,
    render_mode: str | None = None,
    blue_team: object = None,
) -> ActionMasker:
    """Create a CyberEnv wrapped with Monitor + ActionMasker for MaskablePPO.

    Wrapping order: CyberEnv → Monitor → ActionMasker.
    Monitor is required for SB3 to log standard episode stats
    (rollout/ep_rew_mean, rollout/ep_len_mean) in TensorBoard.

    Args:
        seed: Random seed for reproducibility.
        max_steps: Maximum steps per episode.
        render_mode: Pygame render mode (None for headless training).
        blue_team: Optional ScriptedBlueTeam (or compatible) instance.
                   If None, no Blue Team acts (Phase 3 behaviour).

    Returns:
        ActionMasker-wrapped (Monitor-wrapped) CyberEnv.
    """
    base_env = CyberEnv(
        seed=seed, max_steps=max_steps, render_mode=render_mode, blue_team=blue_team
    )
    env = Monitor(base_env)
    return ActionMasker(env, action_mask_fn=lambda e: base_env.action_masks())


def make_vec_masked_env(
    n_envs: int = 1,
    seed: int = 42,
    max_steps: int = 200,
    blue_team: object = None,
) -> DummyVecEnv:
    """Create vectorized masked environments for parallel rollout collection.

    Each sub-environment has a different seed offset to ensure diversity.
    Uses DummyVecEnv (single process) — appropriate for CPU-only training.

    Args:
        n_envs: Number of parallel environments.
        seed: Base random seed (env i uses seed + i).
        max_steps: Maximum steps per episode.
        blue_team: Optional ScriptedBlueTeam (or compatible) instance.
                   If None, no Blue Team acts (Phase 3 behaviour).

    Returns:
        DummyVecEnv wrapping N ActionMasker-wrapped CyberEnv instances.
    """

    def _make_env(i: int):
        def _init() -> ActionMasker:
            return make_masked_env(seed=seed + i, max_steps=max_steps, blue_team=blue_team)

        return _init

    return DummyVecEnv([_make_env(i) for i in range(n_envs)])
