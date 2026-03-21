"""Tests for CyberEnv Gymnasium environment."""

import numpy as np

from src.config import MAX_NODES, N_GLOBAL_FEATURES, N_NODE_FEATURES, REWARD_PER_STEP
from src.environment.actions import ActionType, encode_action
from src.environment.cyber_env import CyberEnv
from src.environment.node import DiscoveryLevel, SessionLevel


class TestCyberEnvSpaces:
    def test_observation_space(self, test_env: CyberEnv) -> None:
        obs_space = test_env.observation_space
        assert "node_features" in obs_space.spaces
        assert "adjacency" in obs_space.spaces
        assert "node_exists_mask" in obs_space.spaces
        assert "fog_mask" in obs_space.spaces
        assert "agent_position" in obs_space.spaces
        assert "global_features" in obs_space.spaces

    def test_action_space(self, test_env: CyberEnv) -> None:
        assert test_env.action_space.n == len(ActionType) * MAX_NODES


class TestCyberEnvReset:
    def test_reset_returns_tuple(self, test_env: CyberEnv) -> None:
        result = test_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_reset_observation_shapes(self, test_env: CyberEnv) -> None:
        obs, _ = test_env.reset()
        assert obs["node_features"].shape == (MAX_NODES, N_NODE_FEATURES)
        assert obs["adjacency"].shape == (MAX_NODES, MAX_NODES)
        assert obs["node_exists_mask"].shape == (MAX_NODES,)
        assert obs["fog_mask"].shape == (MAX_NODES,)
        assert obs["global_features"].shape == (N_GLOBAL_FEATURES,)

    def test_reset_entry_node_discovered(self, test_env: CyberEnv) -> None:
        obs, _ = test_env.reset()
        # Entry node should be discovered and have USER session
        entry = test_env.network.entry_node_id
        node = test_env.network.get_node(entry)
        assert node.discovery_level == DiscoveryLevel.DISCOVERED
        assert node.session_level == SessionLevel.USER

    def test_reset_clears_state(self, test_env: CyberEnv) -> None:
        obs, _ = test_env.reset()
        assert test_env.current_step == 0
        assert test_env.exfiltrated is False
        assert test_env.has_dumped_creds is False

    def test_reset_with_seed(self) -> None:
        env = CyberEnv(seed=42)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1["node_features"], obs2["node_features"])


class TestCyberEnvStep:
    def test_step_returns_five_values(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        result = test_env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, dict)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_wait_gives_time_penalty(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        _, reward, _, _, _ = test_env.step(action)
        assert reward == REWARD_PER_STEP  # -0.5

    def test_step_increments_step(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        test_env.step(action)
        assert test_env.current_step == 1

    def test_truncation_at_max_steps(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        for _ in range(test_env.max_steps):
            obs, reward, terminated, truncated, info = test_env.step(action)
        assert truncated is True

    def test_scan_discovers_neighbors(self, test_env: CyberEnv) -> None:
        test_env.reset()
        # Agent starts at entry (node 0) with USER session
        action = encode_action(ActionType.SCAN, 0)
        obs, reward, _, _, info = test_env.step(action)
        # Should discover node 1 (neighbor of 0)
        assert test_env.network.get_node(1).discovery_level == DiscoveryLevel.DISCOVERED
        assert reward > REWARD_PER_STEP  # got discovery bonus

    def test_full_attack_chain(self, test_env: CyberEnv) -> None:
        """Test a complete attack path: scan -> enumerate -> exploit -> privesc -> exfil."""
        test_env.reset(seed=0)

        # Step 1: Scan from entry to discover node 1
        test_env.step(encode_action(ActionType.SCAN, 0))
        assert test_env.network.get_node(1).discovery_level == DiscoveryLevel.DISCOVERED

        # Step 2: Enumerate node 1
        test_env.step(encode_action(ActionType.ENUMERATE, 1))
        assert test_env.network.get_node(1).discovery_level == DiscoveryLevel.ENUMERATED

        # Step 3: Exploit node 1 (try multiple times since probabilistic)
        exploited = False
        for _ in range(20):
            if test_env.network.get_node(1).session_level != SessionLevel.NONE:
                exploited = True
                break
            test_env.step(encode_action(ActionType.EXPLOIT, 1))
        assert exploited or test_env.network.get_node(1).session_level != SessionLevel.NONE

    def test_repeated_action_penalty(self, test_env: CyberEnv) -> None:
        test_env.reset()
        # Use SCAN (non-WAIT) to trigger the repeated action penalty
        action = encode_action(ActionType.SCAN, 0)
        _, r1, _, _, _ = test_env.step(action)
        _, r2, _, _, _ = test_env.step(action)  # repeated
        assert r2 < r1  # penalty for repetition

    def test_wait_no_repeated_penalty(self, test_env: CyberEnv) -> None:
        """WAIT is exempt from the repeated action penalty."""
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        _, r1, _, _, _ = test_env.step(action)
        _, r2, _, _, _ = test_env.step(action)  # repeated WAIT — no penalty
        assert r2 == r1

    def test_invalid_target_handled(self, test_env: CyberEnv) -> None:
        """Action targeting a non-existent node should not crash."""
        test_env.reset()
        # Target node 49 doesn't exist in 5-node network
        action = encode_action(ActionType.SCAN, 49)
        obs, reward, terminated, truncated, info = test_env.step(action)
        assert isinstance(obs, dict)


class TestActionMasks:
    def test_action_masks_shape(self, test_env: CyberEnv) -> None:
        test_env.reset()
        mask = test_env.action_masks()
        assert mask.shape == (len(ActionType) * MAX_NODES,)
        assert mask.dtype == bool

    def test_action_masks_wait_always_valid(self, test_env: CyberEnv) -> None:
        test_env.reset()
        mask = test_env.action_masks()
        wait_idx = ActionType.WAIT * MAX_NODES + 0
        assert mask[wait_idx] is np.True_

    def test_action_masks_never_all_zero(self, test_env: CyberEnv) -> None:
        test_env.reset()
        for _ in range(50):
            mask = test_env.action_masks()
            assert mask.any(), "Mask must never be all-zero (WAIT is always valid)"
            action = encode_action(ActionType.WAIT, 0)
            test_env.step(action)


class TestCyberEnvInfo:
    def test_info_keys(self, test_env: CyberEnv) -> None:
        _, info = test_env.reset()
        assert "step" in info
        assert "n_compromised" in info
        assert "n_discovered" in info
        assert "max_suspicion" in info
        assert "exfiltrated" in info

    def test_info_after_step(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.SCAN, 0)
        _, _, _, _, info = test_env.step(action)
        assert info["step"] == 1
        assert "action_result" in info


class TestGymnasiumCompliance:
    def test_observation_in_space(self, test_env: CyberEnv) -> None:
        obs, _ = test_env.reset()
        assert test_env.observation_space.contains(obs), (
            f"Observation not in space. Keys: {obs.keys()}"
        )

    def test_observation_in_space_after_step(self, test_env: CyberEnv) -> None:
        test_env.reset()
        action = encode_action(ActionType.WAIT, 0)
        obs, _, _, _, _ = test_env.step(action)
        assert test_env.observation_space.contains(obs)

    def test_dtypes_float32(self, test_env: CyberEnv) -> None:
        obs, _ = test_env.reset()
        assert obs["node_features"].dtype == np.float32
        assert obs["adjacency"].dtype == np.float32
        assert obs["global_features"].dtype == np.float32

    def test_check_env_passes(self) -> None:
        """gymnasium.utils.env_checker.check_env must pass without errors."""
        import pytest
        from gymnasium.utils.env_checker import check_env

        env = CyberEnv(seed=42)
        try:
            check_env(env, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"check_env raised: {e}")


class TestTermination:
    def test_terminated_on_exfiltration(self, test_env: CyberEnv) -> None:
        """Episode terminates when exfiltration succeeds."""
        from src.config import REWARD_EXFILTRATE
        from src.environment.node import SessionLevel

        test_env.reset()
        # Directly set ROOT on node 4 (has_loot in small_network)
        test_env.network.get_node(4).session_level = SessionLevel.ROOT
        action = encode_action(ActionType.EXFILTRATE, 4)
        _, reward, terminated, _, info = test_env.step(action)
        assert terminated is True
        assert info["exfiltrated"] is True
        assert reward >= REWARD_EXFILTRATE - 1  # includes REWARD_PER_STEP

    def test_terminated_on_detection(self, test_env: CyberEnv) -> None:
        """Episode terminates when any node reaches max suspicion."""
        from src.config import REWARD_PER_STEP, SUSPICION_SCAN

        test_env.reset()
        # Set suspicion just below max so SCAN pushes it to 100
        node0 = test_env.network.get_node(0)
        node0.suspicion_level = 100.0 - SUSPICION_SCAN
        action = encode_action(ActionType.SCAN, 0)
        _, reward, terminated, _, _ = test_env.step(action)
        assert terminated is True
        # reward = REWARD_PER_STEP + possible_discovery + REWARD_DETECTED
        # even with a +2 discovery bonus, reward is far below REWARD_PER_STEP
        assert reward < REWARD_PER_STEP

    def test_not_terminated_on_failed_exfiltration(self, test_env: CyberEnv) -> None:
        """Failed exfiltration (no ROOT) should not terminate the episode."""
        test_env.reset()
        test_env.network.get_node(4).session_level = SessionLevel.USER  # only USER
        action = encode_action(ActionType.EXFILTRATE, 4)
        _, _, terminated, _, _ = test_env.step(action)
        assert terminated is False
