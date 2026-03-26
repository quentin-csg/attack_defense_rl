"""Integration tests — PCG + CyberEnv + wrappers (Phase 5)."""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.wrappers import make_pcg_masked_env
from src.config import MAX_NODES, N_NODE_FEATURES
from src.environment.cyber_env import CyberEnv
from src.environment.network import build_fixed_network
from src.pcg.generator import NetworkSize, generate_network


# ---------------------------------------------------------------------------
# CyberEnv + network_factory
# ---------------------------------------------------------------------------


def test_cyber_env_accepts_network_factory() -> None:
    """CyberEnv should accept network_factory without error."""
    def factory(seed):
        net, _ = generate_network(NetworkSize.SMALL, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, seed=42)
    obs, info = env.reset()
    assert obs is not None
    env.close()


def test_reset_generates_new_topology() -> None:
    """Each reset() with a factory must produce a different network topology."""
    def factory(seed):
        net, _ = generate_network(NetworkSize.SMALL, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, seed=0)
    edge_sets = []
    for _ in range(5):
        env.reset()
        edge_sets.append(frozenset(env.network.graph.edges()))
    env.close()
    # At least 2 distinct topologies across 5 resets
    assert len(set(edge_sets)) > 1, "All resets produced identical topologies"


def test_observation_shape_small_pcg() -> None:
    """Observation node_features must always be (MAX_NODES, N_NODE_FEATURES)."""
    def factory(seed):
        net, _ = generate_network(NetworkSize.SMALL, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, seed=42)
    obs, _ = env.reset()
    assert obs["node_features"].shape == (MAX_NODES, N_NODE_FEATURES)
    env.close()


def test_observation_shape_large_pcg() -> None:
    """Even for 60-node networks, observation must be padded to (64, 13)."""
    def factory(seed):
        net, _ = generate_network(NetworkSize.LARGE, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, seed=42, max_steps=350)
    obs, _ = env.reset()
    assert obs["node_features"].shape == (MAX_NODES, N_NODE_FEATURES)
    assert obs["adjacency"].shape == (MAX_NODES, MAX_NODES)
    assert obs["node_exists_mask"].shape == (MAX_NODES,)
    env.close()


def test_padding_slots_are_minus_one() -> None:
    """Padding rows in node_features must be filled with -1.0."""
    net, _ = generate_network(NetworkSize.SMALL, seed=42)
    env = CyberEnv(network=net, seed=42)
    obs, _ = env.reset()
    n_real = net.num_nodes
    # Rows beyond n_real should be all -1
    for row_idx in range(n_real, MAX_NODES):
        assert np.all(obs["node_features"][row_idx] == -1.0), (
            f"Row {row_idx} not padded with -1.0"
        )
    env.close()


# ---------------------------------------------------------------------------
# Action mask with PCG
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", ["small", "medium", "large"])
def test_action_mask_with_pcg_size(size: str) -> None:
    """Action mask must work for all PCG sizes without error."""
    size_enum = NetworkSize(size)

    def factory(seed):
        net, _ = generate_network(size_enum, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, max_steps=350, seed=42)
    env.reset()
    mask = env.action_masks()
    assert mask.dtype == bool
    assert mask.shape == (14 * MAX_NODES,)
    # WAIT must always be valid
    assert mask.any(), "Mask is all False — no valid actions"
    env.close()


# ---------------------------------------------------------------------------
# Full episode
# ---------------------------------------------------------------------------


def test_episode_runs_to_completion_small() -> None:
    """A random-action episode on a small PCG network must not crash."""
    def factory(seed):
        net, _ = generate_network(NetworkSize.SMALL, seed=seed)
        return net

    env = CyberEnv(network_factory=factory, max_steps=150, seed=42)
    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        mask = env.action_masks()
        valid = np.where(mask)[0]
        action = int(np.random.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
    assert steps > 0
    env.close()


# ---------------------------------------------------------------------------
# Blue Team with PCG
# ---------------------------------------------------------------------------


def test_blue_team_works_with_pcg_network() -> None:
    """ScriptedBlueTeam must function correctly on PCG-generated networks."""
    from src.agents.blue_scripted import ScriptedBlueTeam

    def factory(seed):
        net, _ = generate_network(NetworkSize.SMALL, seed=seed)
        return net

    blue = ScriptedBlueTeam(seed=99)
    env = CyberEnv(network_factory=factory, max_steps=150, seed=42, blue_team=blue)
    obs, _ = env.reset()
    mask = env.action_masks()
    valid = np.where(mask)[0]
    action = int(np.random.choice(valid))
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs is not None
    env.close()


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


def test_backward_compat_no_factory() -> None:
    """CyberEnv() without factory must use build_fixed_network as before."""
    env = CyberEnv(seed=42)
    obs, _ = env.reset()
    assert env.network.num_nodes == 8  # fixed network has 8 nodes
    env.close()


def test_backward_compat_network_param() -> None:
    """CyberEnv(network=net) must keep the same topology across resets."""
    net = build_fixed_network(seed=42)
    env = CyberEnv(network=net, seed=42)
    edges_before = frozenset(env.network.graph.edges())
    env.reset()
    edges_after = frozenset(env.network.graph.edges())
    assert edges_before == edges_after
    env.close()


# ---------------------------------------------------------------------------
# Wrapper factory
# ---------------------------------------------------------------------------


def test_make_pcg_masked_env_small() -> None:
    """make_pcg_masked_env should return a working ActionMasker env."""
    env = make_pcg_masked_env(size="small", seed=42)
    obs, _ = env.reset()
    assert obs is not None
    mask = env.action_masks()
    assert mask.any()
    env.close()


def test_make_pcg_masked_env_medium() -> None:
    env = make_pcg_masked_env(size="medium", seed=7)
    obs, _ = env.reset()
    assert obs is not None
    env.close()


def test_make_pcg_masked_env_large() -> None:
    env = make_pcg_masked_env(size="large", seed=13)
    obs, _ = env.reset()
    assert obs is not None
    env.close()
