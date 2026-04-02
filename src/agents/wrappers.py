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

    def _make_env(i: int):
        def _init() -> ActionMasker:
            # Each sub-env gets its own ScriptedBlueTeam instance to avoid
            # shared mutable state across parallel envs (seed offset ensures variety).
            bt = None
            if blue_team is not None:
                from src.agents.blue_scripted import ScriptedBlueTeam
                from src.config import BLUE_ISOLATE_DURATION, PATROL_HOT_DURATION
                bt = ScriptedBlueTeam(
                    seed=seed + i,
                    isolate_duration=max(BLUE_ISOLATE_DURATION, max_steps // 25),
                    hot_duration=max(PATROL_HOT_DURATION, max_steps // 20),
                )
            return make_masked_env(seed=seed + i, max_steps=max_steps, blue_team=bt)

        return _init

    return DummyVecEnv([_make_env(i) for i in range(n_envs)])


def make_pcg_masked_env(
    size: str = "small",
    seed: int | None = None,
    max_steps: int | None = None,
    blue_team: object = None,
) -> ActionMasker:
    """Create a CyberEnv with PCG network generation"""

    from src.config import PCG_MAX_STEPS_LARGE, PCG_MAX_STEPS_MEDIUM, PCG_MAX_STEPS_SMALL
    from src.pcg.generator import NetworkSize, generate_network

    size_enum = NetworkSize(size)

    _size_steps = {
        NetworkSize.SMALL: PCG_MAX_STEPS_SMALL,
        NetworkSize.MEDIUM: PCG_MAX_STEPS_MEDIUM,
        NetworkSize.LARGE: PCG_MAX_STEPS_LARGE,
    }
    resolved_max_steps = max_steps if max_steps is not None else _size_steps[size_enum]

    def network_factory(episode_seed: int | None):
        net, _ = generate_network(size_enum, seed=episode_seed)
        return net

    base_env = CyberEnv(
        network_factory=network_factory,
        max_steps=resolved_max_steps,
        seed=seed,
        blue_team=blue_team,
    )
    env = Monitor(base_env)
    return ActionMasker(env, action_mask_fn=lambda e: base_env.action_masks())


def make_vec_pcg_masked_env(
    size: str = "small",
    n_envs: int = 1,
    seed: int = 42,
    max_steps: int | None = None,
    blue_team: object = None,
) -> DummyVecEnv:
    """Create a vectorised PCG environment for parallel rollout collection."""

    def _make_env(i: int):
        def _init() -> ActionMasker:
            # Each sub-env gets its own ScriptedBlueTeam instance.
            bt = None
            if blue_team is not None:
                from src.agents.blue_scripted import ScriptedBlueTeam
                from src.config import (
                    BLUE_ISOLATE_DURATION,
                    PATROL_HOT_DURATION,
                    PCG_MAX_STEPS_LARGE,
                    PCG_MAX_STEPS_MEDIUM,
                    PCG_MAX_STEPS_SMALL,
                )
                from src.pcg.generator import NetworkSize
                _resolved = max_steps if max_steps is not None else {
                    NetworkSize.SMALL: PCG_MAX_STEPS_SMALL,
                    NetworkSize.MEDIUM: PCG_MAX_STEPS_MEDIUM,
                    NetworkSize.LARGE: PCG_MAX_STEPS_LARGE,
                }[NetworkSize(size)]
                bt = ScriptedBlueTeam(
                    seed=seed + i,
                    isolate_duration=max(BLUE_ISOLATE_DURATION, _resolved // 25),
                    hot_duration=max(PATROL_HOT_DURATION, _resolved // 20),
                )
            return make_pcg_masked_env(
                size=size,
                seed=seed + i,
                max_steps=max_steps,
                blue_team=bt,
            )
        return _init

    return DummyVecEnv([_make_env(i) for i in range(n_envs)])
