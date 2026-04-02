"""Action types and execution logic for the Red Team agent."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from src.config import (
    BRUTE_FORCE_SUCCESS_RATE,
    CLEAN_LOGS_COOLDOWN,
    CLEAN_LOGS_SEQUENCE,
    EXPLOIT_CRASH_RATE,
    MAX_NODES,
    REWARD_EXFILTRATE,
    REWARD_NEW_NODE_COMPROMISED,
    REWARD_NEW_NODE_DISCOVERED,
    REWARD_ROOT_OBTAINED,
    SUSPICION_BRUTE_FORCE,
    SUSPICION_CREDENTIAL_DUMP,
    SUSPICION_ENUMERATE,
    SUSPICION_ENUMERATE_AGGRESSIVE,
    SUSPICION_EXFILTRATE,
    SUSPICION_LIST_FILES,
    SUSPICION_EXPLOIT_MIN,
    SUSPICION_INSTALL_BACKDOOR,
    SUSPICION_LATERAL_MOVE,
    SUSPICION_PIVOT,
    SUSPICION_PRIVESC,
    SUSPICION_SCAN,
    SUSPICION_TUNNEL,
    SUSPICION_WAIT_DECAY,
    TUNNEL_SUSPICION_DIVISOR,
)
from src.environment.node import DiscoveryLevel, Node, SessionLevel
from src.environment.vulnerability import VulnCategory, get_vuln

if TYPE_CHECKING:
    from src.environment.network import Network

logger = logging.getLogger(__name__)


class ActionType(IntEnum):
    """All possible Red Team action types."""

    SCAN = 0
    ENUMERATE = 1
    ENUMERATE_AGGRESSIVE = 2
    EXPLOIT = 3
    BRUTE_FORCE = 4
    PRIVESC = 5
    CREDENTIAL_DUMP = 6
    PIVOT = 7
    LATERAL_MOVE = 8
    INSTALL_BACKDOOR = 9
    EXFILTRATE = 10
    TUNNEL = 11
    CLEAN_LOGS = 12
    WAIT = 13
    LIST_FILES = 14  # win condition for PCG networks — replaces EXFILTRATE


N_ACTION_TYPES: int = len(ActionType)


@dataclass
class ActionResult:
    """Result of executing an action."""

    success: bool = False
    reward: float = 0.0
    suspicion_delta: float = 0.0
    info: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}


def decode_action(action: int) -> tuple[ActionType, int]:
    """Decode a flat discrete action into (action_type, target_node)."""
    return ActionType(action // MAX_NODES), action % MAX_NODES


def encode_action(action_type: ActionType, target_node: int) -> int:
    """Encode (action_type, target_node) into a flat discrete action."""
    return int(action_type) * MAX_NODES + target_node


def execute_action(
    action_type: ActionType,
    target_node_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_position: int,
    has_dumped_creds: bool = False,
) -> ActionResult:
    """Execute a Red Team action and return the result."""
    handlers: dict[ActionType, Any] = {
        ActionType.SCAN: _execute_scan,
        ActionType.ENUMERATE: _execute_enumerate,
        ActionType.ENUMERATE_AGGRESSIVE: _execute_enumerate_aggressive,
        ActionType.EXPLOIT: _execute_exploit,
        ActionType.BRUTE_FORCE: _execute_brute_force,
        ActionType.PRIVESC: _execute_privesc,
        ActionType.CREDENTIAL_DUMP: lambda tid, net, step, r, pos: _execute_credential_dump(
            tid, net, step, r, pos, has_dumped_creds
        ),
        ActionType.PIVOT: _execute_pivot,
        ActionType.LATERAL_MOVE: lambda tid, net, step, r, pos: _execute_lateral_move(
            tid, net, step, r, pos, has_dumped_creds
        ),
        ActionType.INSTALL_BACKDOOR: _execute_install_backdoor,
        ActionType.EXFILTRATE: _execute_exfiltrate,
        ActionType.LIST_FILES: _execute_list_files,
        ActionType.TUNNEL: _execute_tunnel,
        ActionType.CLEAN_LOGS: _execute_clean_logs,
        ActionType.WAIT: _execute_wait,
    }

    assert set(handlers) == set(ActionType), (  # noqa: S101
        f"Missing handlers for: {set(ActionType) - set(handlers)}"
    )

    return handlers[action_type](target_node_id, network, current_step, rng, agent_position)


def _apply_suspicion(node: Node, base_amount: float, network: Network) -> float:
    """Apply suspicion to a node, halved if a tunnel exists."""
    amount = base_amount / TUNNEL_SUSPICION_DIVISOR if node.has_tunnel else base_amount
    node.add_suspicion(amount)
    return amount


def _execute_scan(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """SCAN: discover adjacent nodes from a compromised node."""
    node = network.get_node(target_id)
    susp = _apply_suspicion(node, SUSPICION_SCAN, network)

    discovered = []
    reward = 0.0
    for nid in network.get_neighbors(target_id):
        neighbor = network.get_node(nid)
        if neighbor.discovery_level == DiscoveryLevel.UNKNOWN:
            neighbor.discovery_level = DiscoveryLevel.DISCOVERED
            discovered.append(nid)
            reward += REWARD_NEW_NODE_DISCOVERED

    return ActionResult(
        success=len(discovered) > 0,
        reward=reward,
        suspicion_delta=susp,
        info={"action": "SCAN", "target": target_id, "discovered": discovered},
    )


def _execute_enumerate(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """ENUMERATE: reveal services and vulns of a discovered node."""
    node = network.get_node(target_id)
    susp = _apply_suspicion(node, SUSPICION_ENUMERATE, network)
    was_already = node.discovery_level == DiscoveryLevel.ENUMERATED
    node.discovery_level = DiscoveryLevel.ENUMERATED
    return ActionResult(
        success=not was_already,
        reward=0.0,
        suspicion_delta=susp,
        info={"action": "ENUMERATE", "target": target_id, "was_already": was_already},
    )


def _execute_enumerate_aggressive(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """ENUMERATE_AGGRESSIVE: fast but noisy enumeration."""
    node = network.get_node(target_id)
    susp = _apply_suspicion(node, SUSPICION_ENUMERATE_AGGRESSIVE, network)
    was_already = node.discovery_level == DiscoveryLevel.ENUMERATED
    node.discovery_level = DiscoveryLevel.ENUMERATED
    return ActionResult(
        success=not was_already,
        reward=0.0,
        suspicion_delta=susp,
        info={"action": "ENUMERATE_AGGRESSIVE", "target": target_id},
    )


def _execute_exploit(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """EXPLOIT: attempt to gain USER session via a vulnerability."""
    node = network.get_node(target_id)

    exploit_vulns = [
        v
        for v in node.vulnerabilities
        if get_vuln(v).category in (VulnCategory.RCE, VulnCategory.SQLI, VulnCategory.URL_INJECTION)
    ]

    if not exploit_vulns:
        susp = _apply_suspicion(node, SUSPICION_EXPLOIT_MIN, network)
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "reason": "no_exploitable_vuln"},
        )

    vuln = get_vuln(exploit_vulns[0])
    susp = _apply_suspicion(node, vuln.suspicion_cost, network)

    roll = rng.random()
    fail_prob = max(0.0, 1.0 - vuln.success_rate - EXPLOIT_CRASH_RATE)
    if roll < EXPLOIT_CRASH_RATE:
        node.detectable_traces.add("EXPLOIT_CRASH")
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "outcome": "crash"},
        )
    elif roll < EXPLOIT_CRASH_RATE + fail_prob:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "outcome": "fail"},
        )
    else:
        was_none = node.session_level == SessionLevel.NONE
        if node.session_level == SessionLevel.NONE:
            node.session_level = SessionLevel.USER
        return ActionResult(
            success=True,
            reward=REWARD_NEW_NODE_COMPROMISED if was_none else 0.0,
            suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "outcome": "success", "vuln": vuln.name},
        )


def _execute_brute_force(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """BRUTE_FORCE: attempt login via weak credentials."""
    node = network.get_node(target_id)
    susp = _apply_suspicion(node, SUSPICION_BRUTE_FORCE, network)

    if not node.has_weak_credentials:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=susp,
            info={"action": "BRUTE_FORCE", "target": target_id, "reason": "no_weak_creds"},
        )

    if rng.random() < BRUTE_FORCE_SUCCESS_RATE:
        was_none = node.session_level == SessionLevel.NONE
        if node.session_level == SessionLevel.NONE:
            node.session_level = SessionLevel.USER
        return ActionResult(
            success=True,
            reward=REWARD_NEW_NODE_COMPROMISED if was_none else 0.0,
            suspicion_delta=susp,
            info={"action": "BRUTE_FORCE", "target": target_id, "outcome": "success"},
        )
    return ActionResult(
        success=False, reward=0.0, suspicion_delta=susp,
        info={"action": "BRUTE_FORCE", "target": target_id, "outcome": "fail"},
    )


def _execute_privesc(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """PRIVESC: escalate from USER to ROOT using a privesc vulnerability."""
    node = network.get_node(target_id)
    susp = _apply_suspicion(node, SUSPICION_PRIVESC, network)

    privesc_vulns = [v for v in node.vulnerabilities if get_vuln(v).category == VulnCategory.PRIVESC]
    if not privesc_vulns:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=susp,
            info={"action": "PRIVESC", "target": target_id, "reason": "no_privesc_vuln"},
        )

    vuln = get_vuln(privesc_vulns[0])
    if rng.random() < vuln.success_rate:
        node.session_level = SessionLevel.ROOT
        return ActionResult(
            success=True, reward=REWARD_ROOT_OBTAINED, suspicion_delta=susp,
            info={"action": "PRIVESC", "target": target_id, "outcome": "success"},
        )
    return ActionResult(
        success=False, reward=0.0, suspicion_delta=susp,
        info={"action": "PRIVESC", "target": target_id, "outcome": "fail"},
    )


def _execute_credential_dump(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
    has_dumped_creds: bool = False,
) -> ActionResult:
    """CREDENTIAL_DUMP: extract reusable credentials from a compromised node."""
    if has_dumped_creds:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "CREDENTIAL_DUMP", "target": target_id, "reason": "already_dumped"},
        )
    node = network.get_node(target_id)
    if node.session_level == SessionLevel.NONE:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "CREDENTIAL_DUMP", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_CREDENTIAL_DUMP, network)
    node.detectable_traces.add("CREDENTIAL_DUMP")
    return ActionResult(
        success=True, reward=0.0, suspicion_delta=susp,
        info={"action": "CREDENTIAL_DUMP", "target": target_id},
    )


def _execute_pivot(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """PIVOT: access a node via a compromised intermediary (2-hop max)."""
    node = network.get_node(target_id)

    if node.session_level != SessionLevel.NONE:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "PIVOT", "target": target_id, "reason": "already_compromised"},
        )

    reachable = False
    for comp_id, comp_node in network.nodes.items():
        if comp_node.session_level != SessionLevel.NONE and comp_node.is_online:
            for neighbor_id in network.get_neighbors(comp_id):
                if neighbor_id == target_id or target_id in network.get_neighbors(neighbor_id):
                    reachable = True
                    break
        if reachable:
            break

    if not reachable:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "PIVOT", "target": target_id, "reason": "not_reachable"},
        )

    susp = _apply_suspicion(node, SUSPICION_PIVOT, network)
    node.session_level = SessionLevel.USER
    node.discovery_level = DiscoveryLevel.ENUMERATED
    return ActionResult(
        success=True, reward=REWARD_NEW_NODE_COMPROMISED, suspicion_delta=susp,
        info={"action": "PIVOT", "target": target_id},
    )


def _execute_lateral_move(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
    has_dumped_creds: bool = False,
) -> ActionResult:
    """LATERAL_MOVE: use dumped credentials to access an adjacent node."""
    node = network.get_node(target_id)
    if not has_dumped_creds:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "LATERAL_MOVE", "target": target_id, "reason": "no_creds"},
        )
    adjacent_to_compromised = any(
        comp_node.session_level != SessionLevel.NONE
        and comp_node.is_online
        and network.is_adjacent(comp_id, target_id)
        for comp_id, comp_node in network.nodes.items()
    )
    if not adjacent_to_compromised:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "LATERAL_MOVE", "target": target_id, "reason": "not_adjacent"},
        )
    susp = _apply_suspicion(node, SUSPICION_LATERAL_MOVE, network)
    was_none = node.session_level == SessionLevel.NONE
    if node.session_level == SessionLevel.NONE:
        node.session_level = SessionLevel.USER
    return ActionResult(
        success=True,
        reward=REWARD_NEW_NODE_COMPROMISED if was_none else 0.0,
        suspicion_delta=susp,
        info={"action": "LATERAL_MOVE", "target": target_id},
    )


def _execute_install_backdoor(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """INSTALL_BACKDOOR: maintain persistent access on a node."""
    node = network.get_node(target_id)
    if node.session_level == SessionLevel.NONE:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "INSTALL_BACKDOOR", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_INSTALL_BACKDOOR, network)
    node.has_backdoor = True
    node.detectable_traces.add("INSTALL_BACKDOOR")
    return ActionResult(
        success=True, reward=0.0, suspicion_delta=susp,
        info={"action": "INSTALL_BACKDOOR", "target": target_id},
    )


def _execute_exfiltrate(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """EXFILTRATE: extract data from a node (legacy win condition). Requires ROOT."""
    node = network.get_node(target_id)
    if node.session_level != SessionLevel.ROOT:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "EXFILTRATE", "target": target_id, "reason": "no_root"},
        )
    susp = _apply_suspicion(node, SUSPICION_EXFILTRATE, network)
    node.detectable_traces.add("EXFILTRATE")
    return ActionResult(
        success=node.has_loot,
        reward=REWARD_EXFILTRATE if node.has_loot else 0.0,
        suspicion_delta=susp,
        info={"action": "EXFILTRATE", "target": target_id, "had_loot": node.has_loot},
    )


def _execute_list_files(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """LIST_FILES (ls): capture the flag on the target node (PCG win condition)."""
    node = network.get_node(target_id)
    if target_id != network.target_node_id:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "ls", "target": target_id, "reason": "not_target_node"},
        )
    if node.session_level == SessionLevel.NONE:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "ls", "target": target_id, "reason": "no_session"},
        )
    if not node.has_loot:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "ls", "target": target_id, "reason": "no_flag"},
        )
    susp = _apply_suspicion(node, SUSPICION_LIST_FILES, network)
    node.detectable_traces.add("ls")
    return ActionResult(
        success=True, reward=REWARD_EXFILTRATE, suspicion_delta=susp,
        info={"action": "ls (flag captured)", "target": target_id},
    )


def _execute_tunnel(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """TUNNEL: create an encrypted tunnel (halves future suspicion on this node)."""
    node = network.get_node(target_id)
    if node.session_level == SessionLevel.NONE:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "TUNNEL", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_TUNNEL, network)
    node.has_tunnel = True
    return ActionResult(
        success=True, reward=0.0, suspicion_delta=susp,
        info={"action": "TUNNEL", "target": target_id},
    )


def _execute_clean_logs(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """CLEAN_LOGS: erase traces. Requires ROOT. Diminishing returns + cooldown."""
    node = network.get_node(target_id)
    if node.session_level != SessionLevel.ROOT:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "CLEAN_LOGS", "target": target_id, "reason": "no_root"},
        )
    if current_step - node.last_clean_logs_step <= CLEAN_LOGS_COOLDOWN:
        return ActionResult(
            success=False, reward=0.0, suspicion_delta=0.0,
            info={"action": "CLEAN_LOGS", "target": target_id, "reason": "cooldown"},
        )
    idx = min(node.clean_logs_count, len(CLEAN_LOGS_SEQUENCE) - 1)
    suspicion_change = CLEAN_LOGS_SEQUENCE[idx]
    node.reduce_suspicion(abs(suspicion_change), bypass_floor=True)
    node.clean_logs_count += 1
    node.last_clean_logs_step = current_step
    node.detectable_traces.clear()
    return ActionResult(
        success=True, reward=0.0, suspicion_delta=suspicion_change,
        info={"action": "CLEAN_LOGS", "target": target_id, "reduction": suspicion_change},
    )


def _execute_wait(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """WAIT: do nothing. Suspicion decays on all nodes."""
    for node in network.nodes.values():
        node.reduce_suspicion(abs(SUSPICION_WAIT_DECAY))
    return ActionResult(
        success=True, reward=0.0, suspicion_delta=SUSPICION_WAIT_DECAY,
        info={"action": "WAIT"},
    )
