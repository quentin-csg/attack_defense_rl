"""Phase 3 integration tests — end-to-end verification."""

from __future__ import annotations

import pytest
from gymnasium.utils.env_checker import check_env

from src.agents.red_trainer import create_model, evaluate
from src.agents.wrappers import make_masked_env
from src.environment.cyber_env import CyberEnv


class TestPhase1Regression:
    def test_check_env_still_passes(self) -> None:
        """Phase 1 compatibility must not regress after Phase 3 changes."""
        env = CyberEnv(seed=42)
        check_env(env, skip_render_check=True)
        env.close()

    def test_base_env_action_masks_still_work(self) -> None:
        env = CyberEnv(seed=42)
        env.reset()
        mask = env.action_masks()
        assert mask.any(), "Mask must not be all-False"
        env.close()


class TestFullEpisodeRandom:
    def test_random_episode_completes(self) -> None:
        """Full episode with random valid actions must terminate or truncate."""
        env = make_masked_env(seed=42, max_steps=100)
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            valid = mask.nonzero()[0]
            action = int(valid[0])
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps > 0
        env.close()


class TestTrainedModelEpisode:
    @pytest.fixture
    def short_model_and_env(self):
        env = make_masked_env(seed=42, max_steps=50)
        model = create_model(env, log_dir=None, seed=42)
        model.learn(total_timesteps=128)
        yield model, env
        env.close()

    def test_trained_model_full_episode(self, short_model_and_env) -> None:
        """Trained model must be able to run a full episode without error."""
        model, env = short_model_and_env
        obs, info = env.reset()
        done = False
        steps = 0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1
        assert steps > 0

    def test_predicted_actions_always_respect_mask(self, short_model_and_env) -> None:
        """Every action predicted by the model must be valid (mask=True)."""
        model, env = short_model_and_env
        obs, info = env.reset()
        done = False
        violations = 0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, deterministic=True, action_masks=mask)
            if not mask[int(action)]:
                violations += 1
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        assert violations == 0, f"{violations} invalid actions predicted"

    def test_evaluate_runs_without_error(self, short_model_and_env) -> None:
        model, env = short_model_and_env
        eval_env = make_masked_env(seed=99, max_steps=50)
        metrics = evaluate(model, eval_env, n_episodes=3)
        eval_env.close()
        assert isinstance(metrics, dict)
        assert len(metrics) == 6
