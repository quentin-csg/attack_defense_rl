"""Shared fixtures for all test phases."""

from __future__ import annotations

import random

import pytest

from src.environment.cyber_env import CyberEnv
from src.environment.network import Network, build_fixed_network
from src.environment.node import (
    Node,
    OsType,
    Service,
)


@pytest.fixture
def seeded_rng() -> random.Random:
    """Deterministic random number generator."""
    return random.Random(42)


@pytest.fixture
def small_network() -> Network:
    """Fixed 5-node linear network for unit tests.

    Topology: 0 -- 1 -- 2 -- 3 -- 4
    Node 0 = entry, Node 4 = target (has_loot)
    """
    net = Network()
    nodes = [
        Node(0, OsType.NETWORK_DEVICE, [Service("ssh", 22)], ["rce_generic"]),
        Node(1, OsType.LINUX, [Service("ssh", 22), Service("http", 80)], ["sqli_basic"]),
        Node(
            2,
            OsType.WINDOWS,
            [Service("smb", 445)],
            ["rce_generic", "privesc_kernel"],
            has_weak_credentials=True,
        ),
        Node(3, OsType.LINUX, [Service("ssh", 22)], ["privesc_suid"]),
        Node(
            4,
            OsType.LINUX,
            [Service("ssh", 22), Service("nfs", 2049)],
            ["privesc_misconfig"],
            has_loot=True,
        ),
    ]
    for n in nodes:
        net.add_node(n)

    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        net.add_edge(a, b)

    net.entry_node_id = 0
    net.target_node_id = 4
    return net


@pytest.fixture
def medium_network() -> Network:
    """Fixed 8-node network for integration tests (same as build_fixed_network)."""
    return build_fixed_network(seed=42)


@pytest.fixture
def test_env(small_network: Network) -> CyberEnv:
    """CyberEnv configured with the small 5-node network."""
    return CyberEnv(network=small_network, max_steps=100, seed=42)
