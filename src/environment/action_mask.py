"""Action masking — compute which actions are valid at each step.

CORRECTION 1: Flat Discrete space, MaskablePPO requires a boolean mask.
WAIT is ALWAYS valid so the mask is never all-zero.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.config import CLEAN_LOGS_COOLDOWN, MAX_NODES
from src.environment.actions import N_ACTION_TYPES, ActionType
from src.environment.node import DiscoveryLevel, SessionLevel
from src.environment.vulnerability import VulnCategory, get_vuln

if TYPE_CHECKING:
    from src.environment.network import Network


def compute_action_mask(
    network: Network,
    current_step: int,
    agent_position: int,
    has_dumped_creds: bool = False,
) -> np.ndarray:
    """Compute a boolean mask of valid actions.

    Returns:
        np.ndarray of shape (N_ACTION_TYPES * MAX_NODES,) with dtype bool.
        True = action is valid, False = invalid.
    """
    mask = np.zeros(N_ACTION_TYPES * MAX_NODES, dtype=bool)

    for node_id, node in network.nodes.items():
        # --- SCAN: need session on the target (scan from a compromised node) ---
        if node.session_level != SessionLevel.NONE and node.is_online:
            mask[ActionType.SCAN * MAX_NODES + node_id] = True

        # --- ENUMERATE / ENUMERATE_AGGRESSIVE: node must be DISCOVERED (not yet ENUMERATED) ---
        # Re-enumerating an already-ENUMERATED node wastes a step and adds suspicion for nothing.
        if node.discovery_level == DiscoveryLevel.DISCOVERED and node.is_online:
            mask[ActionType.ENUMERATE * MAX_NODES + node_id] = True
            mask[ActionType.ENUMERATE_AGGRESSIVE * MAX_NODES + node_id] = True

        # --- EXPLOIT: node must be enumerated + have exploitable vuln + no session yet ---
        if (
            node.discovery_level == DiscoveryLevel.ENUMERATED
            and node.is_online
            and node.session_level == SessionLevel.NONE
        ):
            has_exploit_vuln = any(
                get_vuln(v).category
                in (VulnCategory.RCE, VulnCategory.SQLI, VulnCategory.URL_INJECTION)
                for v in node.vulnerabilities
            )
            if has_exploit_vuln:
                mask[ActionType.EXPLOIT * MAX_NODES + node_id] = True

        # --- BRUTE_FORCE: node must be ENUMERATED + have weak creds + no session ---
        # Requires ENUMERATED (not just DISCOVERED) to avoid leaking has_weak_credentials
        # through the fog of war: MaskablePPO sees the mask, so a DISCOVERED-only check
        # would let the agent infer weak creds without having scanned the node.
        if (
            node.discovery_level == DiscoveryLevel.ENUMERATED
            and node.is_online
            and node.session_level == SessionLevel.NONE
            and node.has_weak_credentials
        ):
            mask[ActionType.BRUTE_FORCE * MAX_NODES + node_id] = True

        # --- PRIVESC: need USER session + ENUMERATED (to know the vuln) + privesc vuln ---
        if (
            node.session_level == SessionLevel.USER
            and node.is_online
            and node.discovery_level == DiscoveryLevel.ENUMERATED
        ):
            has_privesc = any(
                get_vuln(v).category == VulnCategory.PRIVESC for v in node.vulnerabilities
            )
            if has_privesc:
                mask[ActionType.PRIVESC * MAX_NODES + node_id] = True

        # --- CREDENTIAL_DUMP: need at least USER session, creds not yet dumped ---
        # Once has_dumped_creds is True there is no benefit to dumping again —
        # masking prevents the agent from wasting steps and accumulating suspicion.
        if (
            not has_dumped_creds
            and node.session_level != SessionLevel.NONE
            and node.is_online
        ):
            mask[ActionType.CREDENTIAL_DUMP * MAX_NODES + node_id] = True

        # --- PIVOT: target must be DISCOVERED (not UNKNOWN to avoid FoW leak)
        #     and reachable from a compromised node via 2 hops ---
        if (
            node.discovery_level >= DiscoveryLevel.DISCOVERED
            and node.is_online
            and node.session_level == SessionLevel.NONE
        ):
            # Check if any compromised (and online) node can reach this one (2 hops)
            for comp_id, comp_node in network.nodes.items():
                if comp_node.session_level != SessionLevel.NONE and comp_node.is_online:
                    for neighbor_id in network.get_neighbors(comp_id):
                        if neighbor_id == node_id or node_id in network.get_neighbors(neighbor_id):
                            mask[ActionType.PIVOT * MAX_NODES + node_id] = True
                            break
                if mask[ActionType.PIVOT * MAX_NODES + node_id]:
                    break

        # --- LATERAL_MOVE: need dumped creds + target adjacent to compromised node ---
        if (
            has_dumped_creds
            and node.session_level == SessionLevel.NONE
            and node.is_online
            and node.discovery_level >= DiscoveryLevel.DISCOVERED
        ):
            # Check if adjacent to any compromised (and online) node
            for comp_id, comp_node in network.nodes.items():
                if (
                    comp_node.session_level != SessionLevel.NONE
                    and comp_node.is_online
                    and network.is_adjacent(comp_id, node_id)
                ):
                    mask[ActionType.LATERAL_MOVE * MAX_NODES + node_id] = True
                    break

        # --- INSTALL_BACKDOOR: need at least USER, no backdoor yet ---
        if node.session_level != SessionLevel.NONE and not node.has_backdoor and node.is_online:
            mask[ActionType.INSTALL_BACKDOOR * MAX_NODES + node_id] = True

        # --- EXFILTRATE: need ROOT + node has loot ---
        if node.session_level == SessionLevel.ROOT and node.has_loot and node.is_online:
            mask[ActionType.EXFILTRATE * MAX_NODES + node_id] = True

        # --- TUNNEL: need at least USER, no tunnel yet ---
        if node.session_level != SessionLevel.NONE and not node.has_tunnel and node.is_online:
            mask[ActionType.TUNNEL * MAX_NODES + node_id] = True

        # --- CLEAN_LOGS: need ROOT + cooldown respected ---
        if (
            node.session_level == SessionLevel.ROOT
            and node.is_online
            and (current_step - node.last_clean_logs_step) > CLEAN_LOGS_COOLDOWN
        ):
            mask[ActionType.CLEAN_LOGS * MAX_NODES + node_id] = True

    # --- WAIT: always valid on the agent's current position ---
    # Using agent_position (not just entry_node_id) ensures the mask is never all-zero
    # even after Blue Team ROTATE_CREDENTIALS strips all sessions from the entry node.
    mask[ActionType.WAIT * MAX_NODES + agent_position] = True

    return mask
