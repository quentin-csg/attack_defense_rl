"""Data transfer object between CyberEnv and the Pygame renderer.

RenderState is built by CyberEnv._build_render_state() at each render() call
and passed as a single argument to PygameRenderer.update() / get_frame().
This avoids the proliferation of individual keyword arguments as the dashboard
grows in complexity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.network import Network


@dataclass
class LogEntry:
    """A single timestamped entry in the action log panel.

    Attributes:
        step: Episode step at which the action occurred.
        text: Human-readable description of the action/event.
        color_key: Rendering category — one of:
            "red_success"  → green text (Red Team success)
            "red_fail"     → yellow text (Red Team failure)
            "blue_action"  → cyan text (Blue Team action)
            "critical"     → red text (detection, crash, etc.)
    """

    step: int
    text: str
    color_key: str  # "red_success" | "red_fail" | "blue_action" | "critical"


@dataclass
class RenderState:
    """Snapshot of all data the Pygame renderer needs to draw one frame.

    Built by CyberEnv._build_render_state() and consumed by PygameRenderer.
    All fields are read-only from the renderer's perspective.

    Attributes:
        network: Full network state (nodes, edges, sessions, suspicion).
        agent_position: Node ID where the Red Team agent currently sits.
        step: Current episode step counter.
        episode_reward: Cumulative reward for this episode.
        n_compromised: Number of nodes with a non-NONE session.
        n_discovered: Number of nodes that are not UNKNOWN.
        total_nodes: Total real nodes in the network (for x/y fractions).
        max_suspicion: Highest suspicion level across all nodes (0-100).
        fog_percentage: Fraction of the network still unknown (0-100).
        action_log: Ordered list of log entries (oldest first).
        last_action_type: Name of the most recent action (e.g. "EXPLOIT").
        last_action_target: Node ID targeted by the most recent action.
        last_action_success: Whether the most recent action succeeded.
        attacker_path: Ordered list of node IDs visited by the attacker
            since the start of the episode (for path highlighting).
        per_node_suspicion: Mapping node_id → current suspicion level (0-100).
    """

    network: Network
    agent_position: int
    step: int
    episode_reward: float
    n_compromised: int
    n_discovered: int
    total_nodes: int
    max_suspicion: float
    fog_percentage: float
    action_log: list[LogEntry] = field(default_factory=list)
    last_action_type: str | None = None
    last_action_target: int | None = None
    last_action_success: bool = False
    attacker_path: list[int] = field(default_factory=list)
    per_node_suspicion: dict[int, float] = field(default_factory=dict)
