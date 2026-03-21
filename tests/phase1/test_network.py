"""Tests for Network wrapper and fixed topology builder."""

import pytest

from src.config import MAX_NODES
from src.environment.network import Network, build_fixed_network
from src.environment.node import (
    DiscoveryLevel,
    Node,
    OsType,
    SessionLevel,
)


class TestNetwork:
    def test_add_node(self, small_network: Network) -> None:
        assert small_network.num_nodes == 5

    def test_add_edge(self, small_network: Network) -> None:
        assert small_network.is_adjacent(0, 1)
        assert small_network.is_adjacent(1, 2)
        assert not small_network.is_adjacent(0, 4)

    def test_get_neighbors(self, small_network: Network) -> None:
        neighbors = small_network.get_neighbors(1)
        assert set(neighbors) == {0, 2}

    def test_get_node(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        assert node.node_id == 0
        assert node.os_type == OsType.NETWORK_DEVICE

    def test_get_missing_node_raises(self, small_network: Network) -> None:
        with pytest.raises(KeyError):
            small_network.get_node(99)

    def test_node_id_exceeds_max(self) -> None:
        net = Network()
        with pytest.raises(ValueError, match="exceeds MAX_NODES"):
            net.add_node(Node(node_id=MAX_NODES + 1))

    def test_shortest_path(self, small_network: Network) -> None:
        length = small_network.shortest_path_length(0, 4)
        assert length == 4

    def test_shortest_path_no_path(self, small_network: Network) -> None:
        small_network.isolate_node(2)
        length = small_network.shortest_path_length(0, 4)
        assert length is None

    def test_isolate_and_restore(self, small_network: Network) -> None:
        small_network.isolate_node(2)
        assert not small_network.is_adjacent(1, 2)
        assert not small_network.is_adjacent(2, 3)
        assert not small_network.get_node(2).is_online

        small_network.restore_node(2)
        assert small_network.is_adjacent(1, 2)
        assert small_network.is_adjacent(2, 3)
        assert small_network.get_node(2).is_online

    def test_reset_all_nodes(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.session_level = SessionLevel.ROOT
        node.suspicion_level = 80.0
        node.discovery_level = DiscoveryLevel.ENUMERATED
        node.has_backdoor = True

        small_network.reset_all_nodes()

        assert node.session_level == SessionLevel.NONE
        assert node.suspicion_level == 0.0
        assert node.discovery_level == DiscoveryLevel.UNKNOWN
        assert node.has_backdoor is False


class TestBuildFixedNetwork:
    def test_fixed_network_size(self) -> None:
        net = build_fixed_network(seed=42)
        assert net.num_nodes == 8

    def test_fixed_network_connected(self) -> None:
        import networkx as nx

        net = build_fixed_network(seed=42)
        assert nx.is_connected(net.graph)

    def test_fixed_network_entry_target(self) -> None:
        net = build_fixed_network(seed=42)
        assert net.entry_node_id == 0
        assert net.target_node_id == 7
        assert net.get_node(7).has_loot is True

    def test_fixed_network_deterministic(self) -> None:
        net1 = build_fixed_network(seed=42)
        net2 = build_fixed_network(seed=42)
        assert net1.num_nodes == net2.num_nodes
        assert set(net1.graph.edges()) == set(net2.graph.edges())

    def test_path_entry_to_target(self) -> None:
        net = build_fixed_network(seed=42)
        length = net.shortest_path_length(net.entry_node_id, net.target_node_id)
        assert length is not None
        assert length > 0
