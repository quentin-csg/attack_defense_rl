"""Tests for Phase 3 custom callbacks."""

from __future__ import annotations

from stable_baselines3.common.callbacks import CallbackList

from src.agents.callbacks import CyberMetricsCallback, build_callback_list
from src.agents.wrappers import make_masked_env


class TestCyberMetricsCallback:
    def test_init(self) -> None:
        cb = CyberMetricsCallback(window=50)
        assert len(cb._exfiltrated) == 0
        assert len(cb._detected) == 0

    def test_window_size_respected(self) -> None:
        cb = CyberMetricsCallback(window=10)
        assert cb._exfiltrated.maxlen == 10

    def test_deques_fill_on_episode_end(self, masked_env) -> None:
        """After a short training run, deques should be populated."""
        from sb3_contrib import MaskablePPO

        masked_env.reset()
        model = MaskablePPO(
            "MultiInputPolicy",
            masked_env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            seed=42,
            verbose=0,
            device="cpu",
        )
        cb = CyberMetricsCallback(window=100)
        model.learn(total_timesteps=256, callback=cb)
        # At least some episodes must have completed in 256 steps
        assert len(cb._episode_rewards) > 0

    def test_metrics_all_deques_populated(self, masked_env) -> None:
        """All metric deques should be populated with the same count."""
        from sb3_contrib import MaskablePPO

        masked_env.reset()
        model = MaskablePPO(
            "MultiInputPolicy",
            masked_env,
            n_steps=64,
            batch_size=32,
            n_epochs=1,
            seed=42,
            verbose=0,
            device="cpu",
        )
        cb = CyberMetricsCallback(window=100)
        model.learn(total_timesteps=256, callback=cb)
        n = len(cb._episode_rewards)
        # All deques must have the same number of entries
        assert len(cb._exfiltrated) == n
        assert len(cb._detected) == n
        assert len(cb._nodes_compromised) == n
        assert len(cb._max_suspicion) == n
        assert len(cb._episode_lengths) == n


class TestBuildCallbackList:
    def test_returns_callback_list(self, masked_env) -> None:
        eval_env = make_masked_env(seed=99, max_steps=50)
        cb_list = build_callback_list(
            eval_env=eval_env,
            log_dir=None,
            save_dir=None,
            save_freq=1000,
            eval_freq=500,
            eval_episodes=2,
        )
        assert isinstance(cb_list, CallbackList)
        eval_env.close()

    def test_callback_list_has_three_callbacks(self, masked_env) -> None:
        eval_env = make_masked_env(seed=99, max_steps=50)
        cb_list = build_callback_list(
            eval_env=eval_env,
            log_dir=None,
            save_dir=None,
            save_freq=1000,
            eval_freq=500,
            eval_episodes=2,
        )
        assert len(cb_list.callbacks) == 3  # CyberMetrics, Dashboard, Eval (no Checkpoint when save_dir=None)
        eval_env.close()
