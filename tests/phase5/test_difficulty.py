"""Tests for PCG difficulty scoring and solvability checks (Phase 5)."""

from __future__ import annotations

import pytest

from src.environment.network import Network
from src.environment.node import Node, OsType, Service, SessionLevel
from src.pcg.difficulty import compute_difficulty, compute_max_steps, is_solvable
from src.pcg.generator import NetworkSize, generate_network


# ---------------------------------------------------------------------------
# compute_difficulty tests
# ---------------------------------------------------------------------------


class TestComputeDifficulty:
    def test_difficulty_non_negative(self) -> None:
        for size in NetworkSize:
            for seed in range(10):
                net, _ = generate_network(size, seed=seed)
                score = compute_difficulty(net, net.entry_node_id, net.target_node_id)
                assert score >= 0.0, f"{size.value} seed={seed}: difficulty={score} < 0"

    def test_large_harder_than_small_on_average(self) -> None:
        """Large networks should have a higher mean difficulty than Small ones."""
        small_scores = []
        large_scores = []
        for seed in range(20):
            s_net, _ = generate_network(NetworkSize.SMALL, seed=seed)
            l_net, _ = generate_network(NetworkSize.LARGE, seed=seed)
            small_scores.append(compute_difficulty(s_net, s_net.entry_node_id, s_net.target_node_id))
            large_scores.append(compute_difficulty(l_net, l_net.entry_node_id, l_net.target_node_id))
        assert sum(large_scores) / len(large_scores) > sum(small_scores) / len(small_scores), (
            "Large networks should be harder on average than Small ones"
        )

    def test_disconnected_graph_returns_high_score(self) -> None:
        """A network with no path entry→target must return a very high score."""
        net = Network()
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
        # No edge added — no path
        net.entry_node_id = 0
        net.target_node_id = 1
        score = compute_difficulty(net, 0, 1)
        assert score >= 100.0


# ---------------------------------------------------------------------------
# compute_max_steps tests
# ---------------------------------------------------------------------------


class TestComputeMaxSteps:
    def test_steps_within_bounds(self) -> None:
        for size in NetworkSize:
            for seed in range(10):
                net, _ = generate_network(size, seed=seed)
                steps = compute_max_steps(net, net.entry_node_id, net.target_node_id)
                assert 100 <= steps <= 400, f"{size.value}: steps={steps} out of [100,400]"

    def test_large_more_steps_than_small(self) -> None:
        """Large networks need more steps on average than Small ones."""
        small_steps = []
        large_steps = []
        for seed in range(20):
            s_net, _ = generate_network(NetworkSize.SMALL, seed=seed)
            l_net, _ = generate_network(NetworkSize.LARGE, seed=seed)
            small_steps.append(compute_max_steps(s_net, s_net.entry_node_id, s_net.target_node_id))
            large_steps.append(compute_max_steps(l_net, l_net.entry_node_id, l_net.target_node_id))
        assert sum(large_steps) / len(large_steps) > sum(small_steps) / len(small_steps)

    def test_disconnected_returns_max(self) -> None:
        net = Network()
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
        net.entry_node_id = 0
        net.target_node_id = 1
        assert compute_max_steps(net, 0, 1) == 400


# ---------------------------------------------------------------------------
# is_solvable tests
# ---------------------------------------------------------------------------


def _make_minimal_solvable() -> Network:
    """2-node solvable network: 0 (entry, RCE) → 1 (target, PRIVESC, loot)."""
    net = Network()
    net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
    net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
    net.add_edge(0, 1)
    net.entry_node_id = 0
    net.target_node_id = 1
    return net


class TestIsSolvable:
    def test_minimal_solvable_network(self) -> None:
        assert is_solvable(_make_minimal_solvable()) is True

    def test_all_generated_networks_solvable(self) -> None:
        """All sizes, 15 seeds — all pass is_solvable()."""
        for size in NetworkSize:
            for seed in range(15):
                net, _ = generate_network(size, seed=seed)
                assert is_solvable(net), f"{size.value} seed={seed}: not solvable"

    def test_rejects_disconnected_graph(self) -> None:
        net = Network()
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
        # No edge — no path
        net.entry_node_id = 0
        net.target_node_id = 1
        assert is_solvable(net) is False

    def test_rejects_target_without_loot(self) -> None:
        net = _make_minimal_solvable()
        net.get_node(net.target_node_id).has_loot = False
        assert is_solvable(net) is False

    def test_path_node_without_vuln_is_still_solvable(self) -> None:
        """A node on the path with no vuln is solvable via LATERAL_MOVE."""
        net = Network()
        # 0 (entry, rce) → 1 (no vuln, no creds) → 2 (target, privesc, loot)
        # Agent can CREDENTIAL_DUMP on 0 then LATERAL_MOVE to 1 then 2.
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], []))  # no vuln
        net.add_node(Node(2, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
        net.add_edge(0, 1)
        net.add_edge(1, 2)
        net.entry_node_id = 0
        net.target_node_id = 2
        assert is_solvable(net) is True

    def test_path_node_with_only_privesc_is_solvable(self) -> None:
        """A path node with only PRIVESC (no RCE) is still solvable via LATERAL_MOVE."""
        net = Network()
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["privesc_dirty_cow"]))
        net.add_node(Node(2, OsType.LINUX, [Service("ssh", 22)], ["privesc_sudo_baron"], has_loot=True))
        net.add_edge(0, 1)
        net.add_edge(1, 2)
        net.entry_node_id = 0
        net.target_node_id = 2
        assert is_solvable(net) is True

    def test_rejects_target_without_privesc(self) -> None:
        net = Network()
        net.add_node(Node(0, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"]))
        # Target has loot and RCE but no PRIVESC → can't reach ROOT
        net.add_node(Node(1, OsType.LINUX, [Service("ssh", 22)], ["rce_log4shell"], has_loot=True))
        net.add_edge(0, 1)
        net.entry_node_id = 0
        net.target_node_id = 1
        assert is_solvable(net) is False
