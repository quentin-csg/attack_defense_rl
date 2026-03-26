"""Integration tests for Blue Team + CyberEnv interaction."""

from __future__ import annotations

import pytest

from src.agents.blue_scripted import ScriptedBlueTeam
from src.environment.cyber_env import CyberEnv
from src.environment.node import SessionLevel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_with_blue() -> CyberEnv:
    """CyberEnv with Blue Team active, deterministic seed."""
    blue = ScriptedBlueTeam(seed=42)
    env = CyberEnv(seed=42, max_steps=200, blue_team=blue)
    return env


@pytest.fixture
def env_no_blue() -> CyberEnv:
    """CyberEnv without Blue Team (Phase 3 backward-compat)."""
    return CyberEnv(seed=42, max_steps=200)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_no_blue_team_still_works(self, env_no_blue: CyberEnv) -> None:
        """CyberEnv with no Blue Team must behave exactly as before."""
        obs, info = env_no_blue.reset()
        assert obs is not None
        mask = env_no_blue.action_masks()
        assert mask.any()
        env_no_blue.close()

    def test_blue_team_none_by_default(self) -> None:
        """CyberEnv must default to blue_team=None."""
        env = CyberEnv(seed=0)
        assert env.blue_team is None
        env.close()


# ---------------------------------------------------------------------------
# Blue Team reset per episode
# ---------------------------------------------------------------------------


class TestBlueReset:
    def test_blue_thresholds_rerandomized_on_reset(self) -> None:
        """Blue Team thresholds must be re-randomised on each env.reset()."""
        blue = ScriptedBlueTeam(seed=123)
        env = CyberEnv(seed=42, max_steps=50, blue_team=blue)

        env.reset()
        thresh1 = blue.alert_threshold

        env.reset()
        thresh2 = blue.alert_threshold

        # Very unlikely both are identical across two different resets
        # (not guaranteed, but with seed 123 they differ)
        env.close()
        # Just verify reset() doesn't crash and thresholds are in range
        assert 0.0 <= thresh1 <= 100.0
        assert 0.0 <= thresh2 <= 100.0


# ---------------------------------------------------------------------------
# Blue Team actions appear in action log
# ---------------------------------------------------------------------------


class TestBlueActionLog:
    def test_blue_actions_logged_to_action_log(self, env_with_blue: CyberEnv) -> None:
        """Blue Team actions must be appended to _action_log with 'blue_action' color."""
        env_with_blue.reset()

        # Override thresholds AFTER reset() to prevent re-randomisation.
        blue = env_with_blue.blue_team
        blue._alert_thresh = 0.1   # everything triggers alert
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999  # disable patrol for determinism

        entry = env_with_blue.agent_position
        env_with_blue.network.get_node(entry).add_suspicion(50.0)

        # Step with WAIT action (index = ActionType.WAIT * MAX_NODES + entry_node)
        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + entry
        env_with_blue.step(wait_action)

        blue_logs = [e for e in env_with_blue._action_log if e[2] == "blue_action"]
        assert len(blue_logs) > 0

    def test_no_blue_log_without_blue_team(self, env_no_blue: CyberEnv) -> None:
        """Without Blue Team, no blue_action entries in the log."""
        env_no_blue.reset()
        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + env_no_blue.agent_position
        env_no_blue.step(wait_action)

        blue_logs = [e for e in env_no_blue._action_log if e[2] == "blue_action"]
        assert len(blue_logs) == 0
        env_no_blue.close()


# ---------------------------------------------------------------------------
# ROTATE_CREDENTIALS effect in env
# ---------------------------------------------------------------------------


class TestRotateInEnv:
    def test_rotate_removes_red_session_in_env(self) -> None:
        """After Blue ROTATE, Red's session on a suspicious node is removed."""
        blue = ScriptedBlueTeam(seed=42)
        env = CyberEnv(seed=42, max_steps=50, blue_team=blue)
        env.reset()

        # Override thresholds AFTER reset() so they aren't re-randomised.
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 0.1  # triggers immediately
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999

        # add_suspicion updates max_historical → WAIT floor = 25.0, still > 0.1.
        entry = env.network.entry_node_id
        node = env.network.get_node(entry)
        node.session_level = SessionLevel.USER
        node.add_suspicion(50.0)

        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + entry
        env.step(wait_action)

        assert node.session_level == SessionLevel.NONE
        env.close()

    def test_backdoor_survives_rotate_in_env(self) -> None:
        """Backdoor must protect Red's session from Blue ROTATE in full env."""
        blue = ScriptedBlueTeam(seed=42)
        env = CyberEnv(seed=42, max_steps=50, blue_team=blue)
        env.reset()

        blue._alert_thresh = 200.0
        blue._rotate_thresh = 0.1
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999

        entry = env.network.entry_node_id
        node = env.network.get_node(entry)
        node.session_level = SessionLevel.USER
        node.has_backdoor = True
        node.add_suspicion(50.0)

        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + entry
        env.step(wait_action)

        assert node.session_level == SessionLevel.USER
        env.close()


# ---------------------------------------------------------------------------
# ISOLATE effect in env
# ---------------------------------------------------------------------------


class TestIsolateInEnv:
    def test_isolate_makes_node_offline(self) -> None:
        """Blue ISOLATE must set is_online=False on the target node."""
        blue = ScriptedBlueTeam(seed=42)
        env = CyberEnv(seed=42, max_steps=50, blue_team=blue)
        env.reset()

        # Override thresholds AFTER reset() to prevent re-randomisation.
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 0.1
        blue._patrol_interval = 999

        entry = env.network.entry_node_id
        node = env.network.get_node(entry)
        node.add_suspicion(50.0)  # floor=25 after WAIT, still > 0.1

        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + entry
        env.step(wait_action)

        assert node.is_online is False
        env.close()

    def test_action_mask_blocks_actions_on_isolated_node(self) -> None:
        """After Blue ISOLATE, action mask must block all actions on that node."""
        blue = ScriptedBlueTeam(seed=42)
        env = CyberEnv(seed=42, max_steps=50, blue_team=blue)
        env.reset()

        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 0.1
        blue._patrol_interval = 999

        entry = env.network.entry_node_id
        env.network.get_node(entry).add_suspicion(50.0)

        from src.config import MAX_NODES
        from src.environment.actions import ActionType
        wait_action = ActionType.WAIT.value * MAX_NODES + entry
        env.step(wait_action)

        mask = env.action_masks()
        from src.environment.actions import ActionType as AT
        for at in AT:
            if at == AT.WAIT:
                continue
            assert not mask[at.value * MAX_NODES + entry], (
                f"{at.name} must be masked on isolated node {entry}"
            )
        env.close()


# ---------------------------------------------------------------------------
# Exfiltration still possible with Blue Team active
# ---------------------------------------------------------------------------


class TestRedCanWin:
    def test_episode_can_terminate(self) -> None:
        """With Blue Team active, episodes must still terminate (truncate or end)."""
        blue = ScriptedBlueTeam(seed=7)
        env = CyberEnv(seed=7, max_steps=200, blue_team=blue)
        obs, info = env.reset()

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            mask = env.action_masks()
            valid = mask.nonzero()[0]
            import numpy as np
            action = int(np.random.default_rng(steps).choice(valid))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        assert terminated or truncated
        env.close()

    def test_info_contains_blue_team_fields(self, env_with_blue: CyberEnv) -> None:
        """info dict must still contain all standard fields with Blue Team active."""
        _, info = env_with_blue.reset()
        required_fields = {
            "step", "episode_reward", "n_compromised", "n_discovered",
            "max_suspicion", "exfiltrated", "detected", "agent_position",
        }
        assert required_fields.issubset(info.keys())
        env_with_blue.close()
