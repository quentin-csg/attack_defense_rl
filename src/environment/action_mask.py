"""Action masking — compute which actions are valid at each step.

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
    """
    mask = np.zeros(N_ACTION_TYPES * MAX_NODES, dtype=bool)

    for node_id, node in network.nodes.items():
        if node.session_level != SessionLevel.NONE and node.is_online:
            mask[ActionType.SCAN * MAX_NODES + node_id] = True

        # Re-enumerating ENUMERATED wastes a step; only allow on DISCOVERED.
        if node.discovery_level == DiscoveryLevel.DISCOVERED and node.is_online:
            mask[ActionType.ENUMERATE * MAX_NODES + node_id] = True
            mask[ActionType.ENUMERATE_AGGRESSIVE * MAX_NODES + node_id] = True

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

        # Requires ENUMERATED (not just DISCOVERED) to avoid leaking has_weak_credentials
        # through the mask — MaskablePPO observes it, so a weaker check leaks fog info.
        if (
            node.discovery_level == DiscoveryLevel.ENUMERATED
            and node.is_online
            and node.session_level == SessionLevel.NONE
            and node.has_weak_credentials
        ):
            mask[ActionType.BRUTE_FORCE * MAX_NODES + node_id] = True

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

        if (
            not has_dumped_creds
            and node.session_level != SessionLevel.NONE
            and node.is_online
        ):
            mask[ActionType.CREDENTIAL_DUMP * MAX_NODES + node_id] = True

        if (
            node.discovery_level >= DiscoveryLevel.DISCOVERED
            and node.is_online
            and node.session_level == SessionLevel.NONE
        ):
            for comp_id, comp_node in network.nodes.items():
                if comp_node.session_level != SessionLevel.NONE and comp_node.is_online:
                    for neighbor_id in network.get_neighbors(comp_id):
                        if neighbor_id == node_id or node_id in network.get_neighbors(neighbor_id):
                            mask[ActionType.PIVOT * MAX_NODES + node_id] = True
                            break
                if mask[ActionType.PIVOT * MAX_NODES + node_id]:
                    break

        if (
            has_dumped_creds
            and node.session_level == SessionLevel.NONE
            and node.is_online
            and node.discovery_level >= DiscoveryLevel.DISCOVERED
        ):
            for comp_id, comp_node in network.nodes.items():
                if (
                    comp_node.session_level != SessionLevel.NONE
                    and comp_node.is_online
                    and network.is_adjacent(comp_id, node_id)
                ):
                    mask[ActionType.LATERAL_MOVE * MAX_NODES + node_id] = True
                    break

        if node.session_level != SessionLevel.NONE and not node.has_backdoor and node.is_online:
            mask[ActionType.INSTALL_BACKDOOR * MAX_NODES + node_id] = True

        if node.session_level == SessionLevel.ROOT and node.has_loot and node.is_online:
            mask[ActionType.EXFILTRATE * MAX_NODES + node_id] = True

        if node.session_level != SessionLevel.NONE and not node.has_tunnel and node.is_online:
            mask[ActionType.TUNNEL * MAX_NODES + node_id] = True

        if (
            node.session_level == SessionLevel.ROOT
            and node.is_online
            and (current_step - node.last_clean_logs_step) > CLEAN_LOGS_COOLDOWN
        ):
            mask[ActionType.CLEAN_LOGS * MAX_NODES + node_id] = True

    target_node_id = network.target_node_id
    if target_node_id is not None and target_node_id < MAX_NODES:
        target_node = network.nodes.get(target_node_id)
        if (
            target_node is not None
            and agent_position == target_node_id
            and target_node.session_level != SessionLevel.NONE
            and target_node.is_online
            and target_node.has_loot
        ):
            mask[ActionType.LIST_FILES * MAX_NODES + target_node_id] = True

    # WAIT is always valid — using agent_position ensures the mask is never all-zero
    # even after ROTATE_CREDENTIALS strips all sessions from the entry node.
    wait_target = agent_position if agent_position < MAX_NODES else 0
    mask[ActionType.WAIT * MAX_NODES + wait_target] = True

    return mask
