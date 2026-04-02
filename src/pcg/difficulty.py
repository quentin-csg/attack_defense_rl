from __future__ import annotations

import networkx as nx

from src.config import (
    PCG_BASE_STEPS,
    PCG_STEPS_PER_HOP,
    PCG_STEPS_PER_NODE,
)
from src.environment.network import Network
from src.environment.vulnerability import VULN_REGISTRY, VulnCategory


def compute_difficulty(
    network: Network,
    entry: int,
    target: int,
) -> float:
    """Compute a difficulty score for the network."""
    if not nx.has_path(network.graph, entry, target):
        return 999.0  # unreachable — effectively infinite difficulty

    min_hops = nx.shortest_path_length(network.graph, entry, target)
    n_nodes = network.num_nodes

    path = nx.shortest_path(network.graph, entry, target)
    path_len = len(path)

    # Average number of vulns per node on the critical path
    path_vuln_density = sum(
        len(network.get_node(nid).vulnerabilities) for nid in path
    ) / path_len

    # Fraction of path nodes with weak credentials
    path_weak_creds = sum(
        1 for nid in path if network.get_node(nid).has_weak_credentials
    ) / path_len

    score = (
        min_hops * 3.0
        + n_nodes * 0.5
        - path_vuln_density * 5.0
        - path_weak_creds * 3.0
    )
    return max(0.0, score)


def compute_max_steps(
    network: Network,
    entry: int,
    target: int,
) -> int:
    """Compute the recommended max_steps budget for an episode."""
    if not nx.has_path(network.graph, entry, target):
        return 400  # unreachable — give maximum budget

    min_hops = nx.shortest_path_length(network.graph, entry, target)
    n_nodes = network.num_nodes
    steps = PCG_BASE_STEPS + min_hops * PCG_STEPS_PER_HOP + n_nodes * PCG_STEPS_PER_NODE
    return max(100, min(400, steps))


def is_solvable(network: Network) -> bool:
    """Verify a generated network has a valid, completable attack path."""
    entry = network.entry_node_id
    target = network.target_node_id

    # 1. Path must exist
    if not nx.has_path(network.graph, entry, target):
        return False

    # 2. Target must have the flag (has_loot == True)
    target_node = network.get_node(target)
    if not target_node.has_loot:
        return False

    return True
