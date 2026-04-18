"""Tests for Phase 3 environment wrappers."""

from __future__ import annotations

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

from src.agents.wrappers import make_masked_env, make_pcg_masked_env, make_vec_masked_env, make_vec_pcg_masked_env
from src.config import MAX_NODES
from src.environment.actions import N_ACTION_TYPES


class TestMakeMaskedEnv:
    def test_returns_action_masker(self) -> None:
        env = make_masked_env(seed=42, max_steps=50)
        assert isinstance(env, ActionMasker)
        env.close()

    def test_reset_returns_valid_obs(self) -> None:
        env = make_masked_env(seed=42, max_steps=50)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert "node_features" in obs
        assert obs["node_features"].dtype == np.float32
        env.close()

    def test_step_with_valid_action(self) -> None:
        env = make_masked_env(seed=42, max_steps=50)
        env.reset()
        mask = env.action_masks()
        valid = int(mask.nonzero()[0][0])
        obs, reward, terminated, truncated, info = env.step(valid)
        assert isinstance(reward, float)
        env.close()

    def test_action_masks_callable(self) -> None:
        env = make_masked_env(seed=42, max_steps=50)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (N_ACTION_TYPES * MAX_NODES,)
        assert mask.dtype == bool
        env.close()

    def test_mask_never_all_zero(self) -> None:
        """WAIT is always valid — mask must never be all-False."""
        env = make_masked_env(seed=42, max_steps=50)
        env.reset()
        for _ in range(10):
            mask = env.action_masks()
            assert mask.any(), "Action mask must not be all-False (WAIT must be valid)"
            valid = int(mask.nonzero()[0][0])
            obs, reward, terminated, truncated, info = env.step(valid)
            if terminated or truncated:
                break
        env.close()


class TestMakeVecMaskedEnv:
    def test_returns_dummy_vec_env(self) -> None:
        env = make_vec_masked_env(n_envs=2, seed=42, max_steps=50)
        assert isinstance(env, DummyVecEnv)
        env.close()

    def test_n_envs_respected(self) -> None:
        env = make_vec_masked_env(n_envs=3, seed=42, max_steps=50)
        assert env.num_envs == 3
        env.close()

    def test_vec_env_reset(self) -> None:
        env = make_vec_masked_env(n_envs=2, seed=42, max_steps=50)
        obs = env.reset()
        assert "node_features" in obs
        assert obs["node_features"].shape[0] == 2  # batch dim = n_envs
        env.close()

    def test_vec_masked_env_trains(self) -> None:
        """make_vec_masked_env must be usable with MaskablePPO.learn()."""
        from src.agents.red_trainer import create_model

        env = make_vec_masked_env(n_envs=2, seed=42, max_steps=50)
        model = create_model(env, log_dir=None, seed=42)
        model.learn(total_timesteps=256)  # must not raise
        env.close()


class TestMakePcgMaskedEnv:
    def test_returns_action_masker(self) -> None:
        env = make_pcg_masked_env(size="small", seed=0, max_steps=50)
        assert isinstance(env, ActionMasker)
        env.close()

    def test_mask_shape(self) -> None:
        env = make_pcg_masked_env(size="small", seed=0, max_steps=50)
        env.reset()
        mask = env.action_masks()
        assert mask.shape == (N_ACTION_TYPES * MAX_NODES,)
        assert mask.dtype == bool
        env.close()

    def test_mask_never_all_zero(self) -> None:
        env = make_pcg_masked_env(size="small", seed=0, max_steps=50)
        env.reset()
        mask = env.action_masks()
        assert mask.any(), "PCG env mask must not be all-False"
        env.close()


class TestVecPcgMaskedEnvMaskIndependence:
    def test_sub_envs_have_independent_masks(self) -> None:
        """Each sub-env in DummyVecEnv must have its own action mask closure.

        Different seeds → different topologies → different valid action sets.
        We verify that both envs produce valid (non-all-zero) masks and that
        calling action_masks() on one env does not corrupt the other.
        """
        vec_env = make_vec_pcg_masked_env(size="small", n_envs=2, seed=0, max_steps=50)
        vec_env.reset()

        masks_0 = vec_env.env_method("action_masks", indices=[0])[0]
        masks_1 = vec_env.env_method("action_masks", indices=[1])[0]

        assert masks_0.shape == (N_ACTION_TYPES * MAX_NODES,), "Sub-env 0 mask wrong shape"
        assert masks_1.shape == (N_ACTION_TYPES * MAX_NODES,), "Sub-env 1 mask wrong shape"
        assert masks_0.any(), "Sub-env 0 mask must not be all-False"
        assert masks_1.any(), "Sub-env 1 mask must not be all-False"

        # Step both envs and verify sub-env 1's mask is still valid afterwards.
        actions = np.array([int(masks_0.nonzero()[0][0]), int(masks_1.nonzero()[0][0])])
        vec_env.step(actions)

        masks_1_after = vec_env.env_method("action_masks", indices=[1])[0]
        assert masks_1_after.any(), "Sub-env 1 mask must not be all-False after a step"

        vec_env.close()
