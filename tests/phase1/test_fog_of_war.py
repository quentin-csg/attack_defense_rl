"""Tests for Fog of War observation masking."""

import numpy as np

from src.config import MAX_NODES, N_NODE_FEATURES, PADDING_VALUE
from src.environment.fog_of_war import FogOfWar
from src.environment.network import Network
from src.environment.node import DiscoveryLevel, Node, OsType


class TestFogOfWar:
    def setup_method(self) -> None:
        self.fog = FogOfWar()

    def test_fog_mask_all_unknown(self, small_network: Network) -> None:
        """All nodes unknown -> fog mask is all zeros."""
        mask = self.fog.get_fog_mask(small_network.nodes)
        assert mask.shape == (MAX_NODES,)
        assert mask.sum() == 0

    def test_fog_mask_discovered(self, small_network: Network) -> None:
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        mask = self.fog.get_fog_mask(small_network.nodes)
        assert mask[0] == 1
        assert mask[1] == 1
        assert mask[2] == 0

    def test_node_exists_mask(self, small_network: Network) -> None:
        mask = self.fog.get_node_exists_mask(small_network.nodes)
        assert mask.shape == (MAX_NODES,)
        assert mask[:5].sum() == 5
        assert mask[5:].sum() == 0

    def test_encode_unknown_node(self) -> None:
        node = Node(0, OsType.LINUX)
        features = self.fog.encode_node_features(node)
        assert features.shape == (N_NODE_FEATURES,)
        assert np.all(features == PADDING_VALUE)

    def test_encode_discovered_node(self) -> None:
        node = Node(0, OsType.LINUX, discovery_level=DiscoveryLevel.DISCOVERED)
        features = self.fog.encode_node_features(node)
        # OS one-hot: Linux = [1, 0, 0]
        assert features[0] == 1.0
        assert features[1] == 0.0
        assert features[2] == 0.0
        # Vulns hidden (not enumerated)
        assert features[4] == 0.0
        # Loot hidden
        assert features[10] == 0.0

    def test_encode_enumerated_node_shows_vulns(self) -> None:
        node = Node(
            0,
            OsType.WINDOWS,
            vulnerabilities=["rce_generic", "sqli_basic"],
            discovery_level=DiscoveryLevel.ENUMERATED,
            has_loot=True,
        )
        features = self.fog.encode_node_features(node)
        # Windows = [0, 1, 0]
        assert features[1] == 1.0
        # Vulns visible
        assert features[4] > 0.0
        # Loot visible
        assert features[10] == 1.0

    def test_build_observation_shape(self, small_network: Network) -> None:
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED
        adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        obs = self.fog.build_observation(
            small_network.nodes,
            adj,
            current_step=0,
            max_steps=100,
            num_real_nodes=5,
            agent_position=0,
        )
        assert obs["node_features"].shape == (MAX_NODES, N_NODE_FEATURES)
        assert obs["adjacency"].shape == (MAX_NODES, MAX_NODES)
        assert obs["node_exists_mask"].shape == (MAX_NODES,)
        assert obs["fog_mask"].shape == (MAX_NODES,)
        assert obs["global_features"].shape == (3,)

    def test_fog_hides_adjacency(self, small_network: Network) -> None:
        """Edges to/from undiscovered nodes should be masked."""
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED
        # Node 1 is NOT discovered
        adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        adj[0, 1] = 1.0
        adj[1, 0] = 1.0

        obs = self.fog.build_observation(
            small_network.nodes,
            adj,
            current_step=0,
            max_steps=100,
            num_real_nodes=5,
            agent_position=0,
        )
        # Edge to undiscovered node 1 should be hidden
        assert obs["adjacency"][0, 1] == 0.0
        assert obs["adjacency"][1, 0] == 0.0

    def test_observation_dtypes(self, small_network: Network) -> None:
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED
        adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        obs = self.fog.build_observation(
            small_network.nodes,
            adj,
            current_step=0,
            max_steps=100,
            num_real_nodes=5,
            agent_position=0,
        )
        assert obs["node_features"].dtype == np.float32
        assert obs["adjacency"].dtype == np.float32
        assert obs["global_features"].dtype == np.float32
