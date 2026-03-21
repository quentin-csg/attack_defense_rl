"""Fog of War — partial observation mask for the Red Team agent."""

from __future__ import annotations

import numpy as np

from src.config import MAX_NODES, N_NODE_FEATURES, PADDING_VALUE
from src.environment.node import DiscoveryLevel, Node, OsType, SessionLevel


class FogOfWar:
    """Manages what the Red Team can and cannot see.

    The fog has two layers:
    - discovered: the node exists (IP known via SCAN)
    - enumerated: services & vulns are known (via ENUMERATE)

    Non-discovered nodes appear as padding (-1).
    Discovered-but-not-enumerated nodes have partial info (position, online, but
    no vuln/service details).
    """

    def get_fog_mask(self, nodes: dict[int, Node]) -> np.ndarray:
        """Return a binary mask of shape (MAX_NODES,).

        1 = node is discovered (visible), 0 = hidden.
        """
        mask = np.zeros(MAX_NODES, dtype=np.int8)
        for node_id, node in nodes.items():
            if node.discovery_level != DiscoveryLevel.UNKNOWN:
                mask[node_id] = 1
        return mask

    def get_node_exists_mask(self, nodes: dict[int, Node]) -> np.ndarray:
        """Return a binary mask of shape (MAX_NODES,).

        1 = node actually exists in the network, 0 = padding slot.
        """
        mask = np.zeros(MAX_NODES, dtype=np.int8)
        for node_id in nodes:
            mask[node_id] = 1
        return mask

    def encode_node_features(
        self,
        node: Node,
        max_services: int = 5,
        max_vulns: int = 5,
    ) -> np.ndarray:
        """Encode a single node into a feature vector.

        Returns an array of shape (N_NODE_FEATURES,) with float32 values.
        If the node is not discovered, returns all PADDING_VALUE.
        If discovered but not enumerated, vuln/service info is hidden (0).
        """
        features = np.full(N_NODE_FEATURES, PADDING_VALUE, dtype=np.float32)

        if node.discovery_level == DiscoveryLevel.UNKNOWN:
            return features

        # OS type one-hot (3 values)
        os_idx = {OsType.LINUX: 0, OsType.WINDOWS: 1, OsType.NETWORK_DEVICE: 2}
        features[0:3] = 0.0
        features[os_idx[node.os_type]] = 1.0

        # Basic info (always visible once discovered)
        features[3] = min(len(node.services) / max_services, 1.0)  # n_services norm
        features[5] = _session_to_float(node.session_level)
        features[6] = node.suspicion_level / 100.0  # suspicion norm
        features[7] = 1.0 if node.is_online else 0.0
        features[8] = 1.0 if node.has_backdoor else 0.0
        features[9] = 1.0 if node.has_tunnel else 0.0
        features[11] = 1.0 if node.is_under_surveillance else 0.0
        features[12] = _discovery_to_float(node.discovery_level)

        # Info only visible if enumerated
        if node.discovery_level == DiscoveryLevel.ENUMERATED:
            features[4] = min(len(node.vulnerabilities) / max_vulns, 1.0)  # n_vulns
            features[10] = 1.0 if node.has_loot else 0.0  # has_loot
        else:
            features[4] = 0.0  # unknown vulns
            features[10] = 0.0  # unknown loot

        return features

    def build_observation(
        self,
        nodes: dict[int, Node],
        adjacency: np.ndarray,
        current_step: int,
        max_steps: int,
        num_real_nodes: int,
        agent_position: int = 0,
    ) -> dict[str, np.ndarray]:
        """Build the full observation dict for the agent.

        Args:
            agent_position: Current Red Team position (tracked by CyberEnv).

        Returns a dict matching CyberEnv's observation_space.
        """
        node_features = np.full((MAX_NODES, N_NODE_FEATURES), PADDING_VALUE, dtype=np.float32)

        for node_id, node in nodes.items():
            node_features[node_id] = self.encode_node_features(node)

        fog_mask = self.get_fog_mask(nodes)
        node_exists_mask = self.get_node_exists_mask(nodes)

        # Apply fog: non-discovered nodes get full padding
        for i in range(MAX_NODES):
            if node_exists_mask[i] == 1 and fog_mask[i] == 0:
                node_features[i] = PADDING_VALUE

        # Adjacency: mask out fog (only show edges between discovered nodes)
        masked_adj = adjacency.copy()
        for i in range(MAX_NODES):
            if fog_mask[i] == 0:
                masked_adj[i, :] = 0.0
                masked_adj[:, i] = 0.0

        # Global features
        n_compromised = sum(1 for n in nodes.values() if n.session_level != SessionLevel.NONE)
        n_discovered = sum(1 for n in nodes.values() if n.discovery_level != DiscoveryLevel.UNKNOWN)
        global_features = np.array(
            [
                current_step / max(max_steps, 1),
                n_compromised / max(num_real_nodes, 1),
                n_discovered / max(num_real_nodes, 1),
            ],
            dtype=np.float32,
        )

        return {
            "node_features": node_features,
            "adjacency": masked_adj,
            "node_exists_mask": node_exists_mask,
            "fog_mask": fog_mask,
            "agent_position": np.int64(agent_position),
            "global_features": global_features,
        }


def _session_to_float(level: SessionLevel) -> float:
    """Convert SessionLevel to a normalized float."""
    return {SessionLevel.NONE: 0.0, SessionLevel.USER: 0.5, SessionLevel.ROOT: 1.0}[level]


def _discovery_to_float(level: DiscoveryLevel) -> float:
    """Convert DiscoveryLevel to a normalized float."""
    return {
        DiscoveryLevel.UNKNOWN: 0.0,
        DiscoveryLevel.DISCOVERED: 0.5,
        DiscoveryLevel.ENUMERATED: 1.0,
    }[level]
