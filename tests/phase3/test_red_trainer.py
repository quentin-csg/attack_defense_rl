"""Tests for Phase 3 Red Team training pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from sb3_contrib import MaskablePPO

from src.agents.red_trainer import create_model, evaluate, load_model, train
from src.agents.wrappers import make_masked_env
from src.config import MAX_NODES
from src.environment.actions import N_ACTION_TYPES


@pytest.fixture
def env():
    e = make_masked_env(seed=42, max_steps=50)
    yield e
    e.close()


@pytest.fixture
def short_model(env):
    """MaskablePPO trained for 128 steps — fast smoke test fixture."""
    m = create_model(env, log_dir=None, seed=42)
    m.learn(total_timesteps=128)
    return m


class TestCreateModel:
    def test_returns_maskable_ppo(self, env) -> None:
        model = create_model(env, log_dir=None, seed=42)
        assert isinstance(model, MaskablePPO)

    def test_action_space_matches_env(self, env) -> None:
        model = create_model(env, log_dir=None, seed=42)
        assert model.action_space.n == N_ACTION_TYPES * MAX_NODES

    def test_policy_is_multi_input(self, env) -> None:
        model = create_model(env, log_dir=None, seed=42)
        assert "MultiInput" in type(model.policy).__name__


class TestModelPredict:
    def test_predict_returns_valid_action(self, short_model, env) -> None:
        obs, _ = env.reset()
        mask = env.action_masks()
        action, _ = short_model.predict(obs, deterministic=True, action_masks=mask)
        assert 0 <= int(action) < N_ACTION_TYPES * MAX_NODES

    def test_predict_respects_mask(self, short_model, env) -> None:
        """Predicted action must always be a valid (masked-in) action."""
        obs, _ = env.reset()
        for _ in range(20):
            mask = env.action_masks()
            action, _ = short_model.predict(obs, deterministic=True, action_masks=mask)
            assert mask[int(action)], f"Action {action} is masked out"
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break

    def test_predict_stochastic_is_valid(self, short_model, env) -> None:
        obs, _ = env.reset()
        mask = env.action_masks()
        action, _ = short_model.predict(obs, deterministic=False, action_masks=mask)
        assert mask[int(action)]


class TestTrainSmoke:
    def test_train_128_steps_no_error(self, env) -> None:
        model = create_model(env, log_dir=None, seed=42)
        model.learn(total_timesteps=128)  # must not raise

    def test_train_256_steps_no_error(self) -> None:
        env = make_masked_env(seed=42, max_steps=50)
        model = create_model(env, log_dir=None, seed=42)
        model.learn(total_timesteps=256)
        env.close()


class TestTrainFunction:
    def test_train_smoke(self) -> None:
        """train() must run end-to-end and return a MaskablePPO."""
        with tempfile.TemporaryDirectory() as tmp:
            model = train(
                total_timesteps=256,
                seed=42,
                log_dir=str(Path(tmp) / "logs"),
                save_dir=str(Path(tmp) / "models"),
                max_steps=50,
                eval_freq=128,
                eval_episodes=2,
                save_freq=128,
            )
        assert isinstance(model, MaskablePPO)

    def test_train_no_log_dir(self) -> None:
        """train() must not crash when log_dir=None."""
        with tempfile.TemporaryDirectory() as tmp:
            model = train(
                total_timesteps=256,
                seed=42,
                log_dir=None,
                save_dir=str(Path(tmp) / "models"),
                max_steps=50,
                eval_freq=128,
                eval_episodes=2,
                save_freq=128,
            )
        assert isinstance(model, MaskablePPO)

    def test_train_no_save_dir(self) -> None:
        """train() must not crash when save_dir=None."""
        model = train(
            total_timesteps=256,
            seed=42,
            log_dir=None,
            save_dir=None,
            max_steps=50,
            eval_freq=128,
            eval_episodes=2,
            save_freq=128,
        )
        assert isinstance(model, MaskablePPO)

    def test_train_saves_final_model(self) -> None:
        """train() must save red_agent_final.zip when save_dir is set."""
        with tempfile.TemporaryDirectory() as tmp:
            save_dir = str(Path(tmp) / "models")
            train(
                total_timesteps=256,
                seed=42,
                log_dir=None,
                save_dir=save_dir,
                max_steps=50,
                eval_freq=128,
                eval_episodes=2,
                save_freq=256,
            )
            assert (Path(save_dir) / "red_agent_final.zip").exists()


class TestSaveLoad:
    def test_save_load_roundtrip(self, short_model, env) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "test_model")
            short_model.save(path)
            assert Path(path + ".zip").exists()

            loaded = load_model(path, env=env)
            assert isinstance(loaded, MaskablePPO)

    def test_loaded_model_predicts_same_action(self, short_model, env) -> None:
        """Deterministic prediction must be identical after save/load."""
        obs, _ = env.reset()
        mask = env.action_masks()
        action_before, _ = short_model.predict(obs, deterministic=True, action_masks=mask)

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "model")
            short_model.save(path)
            loaded = load_model(path, env=env)

        action_after, _ = loaded.predict(obs, deterministic=True, action_masks=mask)
        assert int(action_before) == int(action_after)


class TestEvaluate:
    def test_returns_dict_with_expected_keys(self, short_model) -> None:
        eval_env = make_masked_env(seed=99, max_steps=50)
        metrics = evaluate(short_model, eval_env, n_episodes=3)
        eval_env.close()

        expected_keys = {
            "exfiltration_rate",
            "detection_rate",
            "mean_episode_reward",
            "mean_episode_length",
            "mean_nodes_compromised",
            "mean_max_suspicion",
        }
        assert set(metrics.keys()) == expected_keys

    def test_all_values_are_float(self, short_model) -> None:
        eval_env = make_masked_env(seed=99, max_steps=50)
        metrics = evaluate(short_model, eval_env, n_episodes=3)
        eval_env.close()
        for key, val in metrics.items():
            assert isinstance(val, float), f"{key} is not float: {type(val)}"

    def test_rates_are_in_0_1(self, short_model) -> None:
        eval_env = make_masked_env(seed=99, max_steps=50)
        metrics = evaluate(short_model, eval_env, n_episodes=5)
        eval_env.close()
        assert 0.0 <= metrics["exfiltration_rate"] <= 1.0
        assert 0.0 <= metrics["detection_rate"] <= 1.0
