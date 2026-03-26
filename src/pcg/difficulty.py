"""Difficulty scoring and solvability checks for PCG networks (Phase 5).

Functions:
    compute_difficulty(network, entry, target) -> float
        Score ≥ 0 — higher = harder for the Red agent.

    compute_max_steps(network, entry, target) -> int
        Episode budget scaled to network size and path length.

    is_solvable(network) -> bool
        Verifies a generated network has a valid, completable attack path.
"""

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
    """Compute a difficulty score for the network.

    Higher score = harder to complete.

    Formula:
        score = min_hops * 3.0
              + n_nodes * 0.5
              - path_vuln_density * 5.0   (more vulns = easier path)
              - path_weak_creds * 3.0     (weak creds = easier)

    Args:
        network: The generated network.
        entry: Entry node ID.
        target: Target node ID.

    Returns:
        Difficulty score ≥ 0.0.
    """
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
    """Compute the recommended max_steps budget for an episode.

    Based on PCG_BASE_STEPS + path length * PCG_STEPS_PER_HOP
    + network size * PCG_STEPS_PER_NODE, clamped to [100, 400].

    Args:
        network: The generated network.
        entry: Entry node ID.
        target: Target node ID.

    Returns:
        Recommended max_steps value (integer in [100, 400]).
    """
    if not nx.has_path(network.graph, entry, target):
        return 400  # unreachable — give maximum budget

    min_hops = nx.shortest_path_length(network.graph, entry, target)
    n_nodes = network.num_nodes
    steps = PCG_BASE_STEPS + min_hops * PCG_STEPS_PER_HOP + n_nodes * PCG_STEPS_PER_NODE
    return max(100, min(400, steps))


def is_solvable(network: Network) -> bool:
    """Verify a generated network has a valid, completable attack path.

    A network is solvable if ALL of the following hold:

    1. A path exists from entry_node_id to target_node_id.
    2. The target node has has_loot = True.
    3. The target node has at least one PRIVESC vulnerability (required
       to obtain ROOT → then EXFILTRATE).

    Note: No per-node vuln check on the path is required because the Red agent
    starts with USER session on the entry node and can traverse via
    CREDENTIAL_DUMP + LATERAL_MOVE on any connected path.

    Args:
        network: The network to validate.

    Returns:
        True if the network is solvable, False otherwise.
    """
    entry = network.entry_node_id
    target = network.target_node_id

    # 1. Path must exist
    if not nx.has_path(network.graph, entry, target):
        return False

    # 2. Target must have loot
    target_node = network.get_node(target)
    if not target_node.has_loot:
        return False

    # 3. Target must have a PRIVESC vuln (needed for ROOT → EXFILTRATE).
    #
    # Note: no per-node exploitable-vuln check is needed because the Red agent
    # starts with USER session on the entry node (given by reset()), and can
    # always use CREDENTIAL_DUMP + LATERAL_MOVE to traverse a connected path.
    # Requiring exploitable vulns on every path node would be too restrictive.
    has_privesc = any(
        VULN_REGISTRY[v].category == VulnCategory.PRIVESC
        for v in target_node.vulnerabilities
        if v in VULN_REGISTRY
    )
    if not has_privesc:
        return False

    return True
