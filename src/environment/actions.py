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


# Single source of truth — adding a new ActionType member updates everything automatically
N_ACTION_TYPES: int = len(ActionType)


@dataclass
class ActionResult:
    """Result of executing an action.

    Attributes:
        success: Whether the action succeeded.
        reward: Immediate reward from this action.
        suspicion_delta: Change in suspicion on the target node.
        info: Additional info for logging/debugging.
        terminated: Whether the episode ends (e.g. full detection).
    """

    success: bool = False
    reward: float = 0.0
    suspicion_delta: float = 0.0
    info: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.info is None:
            self.info = {}


def decode_action(action: int) -> tuple[ActionType, int]:
    """Decode a flat discrete action into (action_type, target_node).

    Follows CORRECTION 1: action_type = action // MAX_NODES,
    target_node = action % MAX_NODES.
    """
    action_type = ActionType(action // MAX_NODES)
    target_node = action % MAX_NODES
    return action_type, target_node


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
    """Execute a Red Team action and return the result.

    Args:
        action_type: The type of action to perform.
        target_node_id: The target node ID.
        network: The current network state.
        current_step: Current episode step (for cooldown checks).
        rng: Random number generator (for deterministic tests).
        agent_position: Node ID where the agent currently is.
        has_dumped_creds: Whether creds have been dumped (for LATERAL_MOVE).

    Returns:
        ActionResult with success, reward, suspicion change, and info.
    """
    # LATERAL_MOVE needs has_dumped_creds; capture it via lambda for uniform dispatch
    handlers: dict[ActionType, Any] = {
        ActionType.SCAN: _execute_scan,
        ActionType.ENUMERATE: _execute_enumerate,
        ActionType.ENUMERATE_AGGRESSIVE: _execute_enumerate_aggressive,
        ActionType.EXPLOIT: _execute_exploit,
        ActionType.BRUTE_FORCE: _execute_brute_force,
        ActionType.PRIVESC: _execute_privesc,
        ActionType.CREDENTIAL_DUMP: _execute_credential_dump,
        ActionType.PIVOT: _execute_pivot,
        ActionType.LATERAL_MOVE: lambda tid, net, step, r, pos: _execute_lateral_move(
            tid, net, step, r, pos, has_dumped_creds
        ),
        ActionType.INSTALL_BACKDOOR: _execute_install_backdoor,
        ActionType.EXFILTRATE: _execute_exfiltrate,
        ActionType.TUNNEL: _execute_tunnel,
        ActionType.CLEAN_LOGS: _execute_clean_logs,
        ActionType.WAIT: _execute_wait,
    }

    # Coverage guard: every ActionType must have a handler entry
    assert set(handlers) == set(ActionType), (  # noqa: S101
        f"Missing handlers for: {set(ActionType) - set(handlers)}"
    )

    handler = handlers[action_type]
    return handler(target_node_id, network, current_step, rng, agent_position)


def _apply_suspicion(node: Node, base_amount: float, network: Network) -> float:
    """Apply suspicion to a node, halved if a tunnel exists on the path."""
    amount = base_amount
    if node.has_tunnel:
        amount /= TUNNEL_SUSPICION_DIVISOR
    node.add_suspicion(amount)
    return amount


# ---------------------------------------------------------------------------
# Individual action handlers
# ---------------------------------------------------------------------------


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
    neighbors = network.get_neighbors(target_id)
    reward = 0.0
    for nid in neighbors:
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
    """ENUMERATE: reveal services and vulns of a discovered node (slow, discrete)."""
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

    # Pick exploit suspicion based on vuln (use first available RCE/SQLI/URL vuln)
    exploit_vulns = [
        v
        for v in node.vulnerabilities
        if get_vuln(v).category in (VulnCategory.RCE, VulnCategory.SQLI, VulnCategory.URL_INJECTION)
    ]

    if not exploit_vulns:
        susp = _apply_suspicion(node, SUSPICION_EXPLOIT_MIN, network)
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "reason": "no_exploitable_vuln"},
        )

    vuln = get_vuln(exploit_vulns[0])
    susp_cost = vuln.suspicion_cost
    susp = _apply_suspicion(node, susp_cost, network)

    roll = rng.random()
    # Clamp fail_prob so success + crash + fail = 1 even if success_rate > 0.95
    fail_prob = max(0.0, 1.0 - vuln.success_rate - EXPLOIT_CRASH_RATE)
    if roll < EXPLOIT_CRASH_RATE:
        # Crash — leaves a detectable trace (service error in logs); is_online stays True
        # (the service may restart; the trace is what matters for Blue Team patrols)
        node.detectable_traces.add("EXPLOIT_CRASH")
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "outcome": "crash"},
        )
    elif roll < EXPLOIT_CRASH_RATE + fail_prob:
        # Fail silently
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "EXPLOIT", "target": target_id, "outcome": "fail"},
        )
    else:
        # Success
        was_none = node.session_level == SessionLevel.NONE
        if node.session_level == SessionLevel.NONE:
            node.session_level = SessionLevel.USER
        reward = REWARD_NEW_NODE_COMPROMISED if was_none else 0.0
        return ActionResult(
            success=True,
            reward=reward,
            suspicion_delta=susp,
            info={
                "action": "EXPLOIT",
                "target": target_id,
                "outcome": "success",
                "vuln": vuln.name,
            },
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
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "BRUTE_FORCE", "target": target_id, "reason": "no_weak_creds"},
        )

    if rng.random() < BRUTE_FORCE_SUCCESS_RATE:
        was_none = node.session_level == SessionLevel.NONE
        if node.session_level == SessionLevel.NONE:
            node.session_level = SessionLevel.USER
        reward = REWARD_NEW_NODE_COMPROMISED if was_none else 0.0
        return ActionResult(
            success=True,
            reward=reward,
            suspicion_delta=susp,
            info={"action": "BRUTE_FORCE", "target": target_id, "outcome": "success"},
        )
    else:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
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

    privesc_vulns = [
        v for v in node.vulnerabilities if get_vuln(v).category == VulnCategory.PRIVESC
    ]
    if not privesc_vulns:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "PRIVESC", "target": target_id, "reason": "no_privesc_vuln"},
        )

    vuln = get_vuln(privesc_vulns[0])
    if rng.random() < vuln.success_rate:
        node.session_level = SessionLevel.ROOT
        return ActionResult(
            success=True,
            reward=REWARD_ROOT_OBTAINED,
            suspicion_delta=susp,
            info={"action": "PRIVESC", "target": target_id, "outcome": "success"},
        )
    else:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=susp,
            info={"action": "PRIVESC", "target": target_id, "outcome": "fail"},
        )


def _execute_credential_dump(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """CREDENTIAL_DUMP: extract reusable credentials from a compromised node."""
    node = network.get_node(target_id)
    if node.session_level == SessionLevel.NONE:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "CREDENTIAL_DUMP", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_CREDENTIAL_DUMP, network)
    node.detectable_traces.add("CREDENTIAL_DUMP")

    return ActionResult(
        success=True,
        reward=0.0,
        suspicion_delta=susp,
        info={"action": "CREDENTIAL_DUMP", "target": target_id},
    )


def _execute_pivot(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """PIVOT: access a discovered non-adjacent node via a compromised intermediary.

    Grants a USER session on the target node and elevates discovery to ENUMERATED.
    Requires the target to be reachable within 2 hops from any compromised node.
    """
    node = network.get_node(target_id)

    # Guard: node must have no session (mask should enforce this, but defence-in-depth)
    if node.session_level != SessionLevel.NONE:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "PIVOT", "target": target_id, "reason": "already_compromised"},
        )

    # Defence-in-depth: verify reachability from a compromised node (2-hop max).
    # The action mask normally enforces this, but a direct handler call must also be safe.
    reachable = False
    for comp_id, comp_node in network.nodes.items():
        if comp_node.session_level != SessionLevel.NONE:
            neighbors = network.get_neighbors(comp_id)
            for neighbor_id in neighbors:
                if neighbor_id == target_id or target_id in network.get_neighbors(neighbor_id):
                    reachable = True
                    break
        if reachable:
            break

    if not reachable:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "PIVOT", "target": target_id, "reason": "not_reachable"},
        )

    susp = _apply_suspicion(node, SUSPICION_PIVOT, network)

    # Only award NEW_NODE_COMPROMISED reward if node was uncompromised
    was_none = node.session_level == SessionLevel.NONE  # always True here after the guard
    node.session_level = SessionLevel.USER
    node.discovery_level = DiscoveryLevel.ENUMERATED

    reward = REWARD_NEW_NODE_COMPROMISED if was_none else 0.0
    return ActionResult(
        success=True,
        reward=reward,
        suspicion_delta=susp,
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
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "LATERAL_MOVE", "target": target_id, "reason": "no_creds"},
        )
    susp = _apply_suspicion(node, SUSPICION_LATERAL_MOVE, network)

    was_none = node.session_level == SessionLevel.NONE
    if node.session_level == SessionLevel.NONE:
        node.session_level = SessionLevel.USER
    reward = REWARD_NEW_NODE_COMPROMISED if was_none else 0.0

    return ActionResult(
        success=True,
        reward=reward,
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
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "INSTALL_BACKDOOR", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_INSTALL_BACKDOOR, network)
    node.has_backdoor = True
    node.detectable_traces.add("INSTALL_BACKDOOR")

    return ActionResult(
        success=True,
        reward=0.0,
        suspicion_delta=susp,
        info={"action": "INSTALL_BACKDOOR", "target": target_id},
    )


def _execute_exfiltrate(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """EXFILTRATE: extract data from a node (main objective). Requires ROOT."""
    node = network.get_node(target_id)
    if node.session_level != SessionLevel.ROOT:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "EXFILTRATE", "target": target_id, "reason": "no_root"},
        )
    susp = _apply_suspicion(node, SUSPICION_EXFILTRATE, network)
    node.detectable_traces.add("EXFILTRATE")

    reward = REWARD_EXFILTRATE if node.has_loot else 0.0

    return ActionResult(
        success=node.has_loot,
        reward=reward,
        suspicion_delta=susp,
        info={"action": "EXFILTRATE", "target": target_id, "had_loot": node.has_loot},
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
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "TUNNEL", "target": target_id, "reason": "no_session"},
        )
    susp = _apply_suspicion(node, SUSPICION_TUNNEL, network)
    node.has_tunnel = True

    return ActionResult(
        success=True,
        reward=0.0,
        suspicion_delta=susp,
        info={"action": "TUNNEL", "target": target_id},
    )


def _execute_clean_logs(
    target_id: int,
    network: Network,
    current_step: int,
    rng: random.Random,
    agent_pos: int,
) -> ActionResult:
    """CLEAN_LOGS: erase traces on a node. Requires ROOT. Diminishing returns + cooldown."""
    node = network.get_node(target_id)
    if node.session_level != SessionLevel.ROOT:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "CLEAN_LOGS", "target": target_id, "reason": "no_root"},
        )

    # Cooldown check
    if current_step - node.last_clean_logs_step <= CLEAN_LOGS_COOLDOWN:
        return ActionResult(
            success=False,
            reward=0.0,
            suspicion_delta=0.0,
            info={"action": "CLEAN_LOGS", "target": target_id, "reason": "cooldown"},
        )

    # Diminishing returns
    idx = min(node.clean_logs_count, len(CLEAN_LOGS_SEQUENCE) - 1)
    suspicion_change = CLEAN_LOGS_SEQUENCE[idx]

    node.reduce_suspicion(abs(suspicion_change))
    node.clean_logs_count += 1
    node.last_clean_logs_step = current_step
    node.detectable_traces.clear()  # cleans all traces

    return ActionResult(
        success=True,
        reward=0.0,
        suspicion_delta=suspicion_change,
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
        success=True,
        reward=0.0,
        suspicion_delta=SUSPICION_WAIT_DECAY,
        info={"action": "WAIT"},
    )
