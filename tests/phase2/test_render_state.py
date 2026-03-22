"""Tests for the RenderState and LogEntry dataclasses."""

from __future__ import annotations

import pytest

from src.environment.network import build_fixed_network
from src.visualization.render_state import LogEntry, RenderState


@pytest.fixture
def network():
    return build_fixed_network(seed=42)


class TestLogEntry:
    def test_construction(self) -> None:
        entry = LogEntry(step=5, text="[5] SCAN Node 0: SUCCESS", color_key="red_success")
        assert entry.step == 5
        assert entry.text == "[5] SCAN Node 0: SUCCESS"
        assert entry.color_key == "red_success"

    def test_all_color_keys_valid(self) -> None:
        valid_keys = ["red_success", "red_fail", "blue_action", "critical"]
        for key in valid_keys:
            e = LogEntry(step=1, text="test", color_key=key)
            assert e.color_key == key


class TestRenderState:
    def test_minimal_construction(self, network) -> None:
        state = RenderState(
            network=network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            n_discovered=1,
            total_nodes=network.num_nodes,
            max_suspicion=0.0,
            fog_percentage=0.0,
        )
        assert state.step == 0
        assert state.action_log == []
        assert state.attacker_path == []
        assert state.per_node_suspicion == {}

    def test_construction_with_all_fields(self, network) -> None:
        log = [LogEntry(step=1, text="test", color_key="red_success")]
        state = RenderState(
            network=network,
            agent_position=1,
            step=10,
            episode_reward=-5.0,
            n_compromised=2,
            n_discovered=4,
            total_nodes=network.num_nodes,
            max_suspicion=75.0,
            fog_percentage=50.0,
            action_log=log,
            last_action_type="SCAN",
            last_action_target=0,
            last_action_success=True,
            attacker_path=[0, 1],
            per_node_suspicion={0: 10.0, 1: 75.0},
        )
        assert state.step == 10
        assert state.last_action_type == "SCAN"
        assert state.max_suspicion == 75.0
        assert len(state.action_log) == 1
        assert state.attacker_path == [0, 1]
        assert state.per_node_suspicion[1] == 75.0

    def test_default_last_action_fields_are_none(self, network) -> None:
        state = RenderState(
            network=network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            n_discovered=0,
            total_nodes=network.num_nodes,
            max_suspicion=0.0,
            fog_percentage=100.0,
        )
        assert state.last_action_type is None
        assert state.last_action_target is None
        assert state.last_action_success is False

    def test_action_log_independent_lists(self, network) -> None:
        """Two RenderState instances must not share the same action_log list."""
        s1 = RenderState(
            network=network, agent_position=0, step=0, episode_reward=0.0,
            n_compromised=0, n_discovered=0, total_nodes=8, max_suspicion=0.0, fog_percentage=0.0,
        )
        s2 = RenderState(
            network=network, agent_position=0, step=0, episode_reward=0.0,
            n_compromised=0, n_discovered=0, total_nodes=8, max_suspicion=0.0, fog_percentage=0.0,
        )
        s1.action_log.append(LogEntry(step=1, text="x", color_key="red_success"))
        assert len(s2.action_log) == 0
