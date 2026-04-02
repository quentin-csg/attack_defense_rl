from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.environment.network import Network


@dataclass
class LogEntry:
    """A single timestamped entry in the action log panel."""

    step: int
    text: str
    color_key: str  # "red_success" | "red_fail" | "blue_action" | "critical"


@dataclass
class RenderState:
    """Snapshot of all data the Pygame renderer needs to draw one frame."""

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
