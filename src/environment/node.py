from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto


class OsType(Enum):
    """Operating system types for network nodes."""
    LINUX = auto()
    WINDOWS = auto()
    NETWORK_DEVICE = auto()


class SessionLevel(IntEnum):
    """Red Team session level on a node (ordered, comparable)."""
    NONE = 0
    USER = 1
    ROOT = 2


class DiscoveryLevel(IntEnum):
    """How much the Red Team knows about a node (ordered, comparable)."""
    UNKNOWN = 0
    DISCOVERED = 1  # IP known via SCAN
    ENUMERATED = 2  # services & vulns known via ENUMERATE


@dataclass
class Service:
    """A service running on a node (e.g. SSH, HTTP, SMB)."""

    name: str
    port: int
    is_running: bool = True


@dataclass
class Node:
    """A single machine in the network."""

    node_id: int
    os_type: OsType = OsType.LINUX
    services: list[Service] = field(default_factory=list)
    vulnerabilities: list[str] = field(default_factory=list)
    is_online: bool = True
    suspicion_level: float = 0.0
    max_suspicion_historical: float = 0.0
    session_level: SessionLevel = SessionLevel.NONE
    discovery_level: DiscoveryLevel = DiscoveryLevel.UNKNOWN
    has_backdoor: bool = False
    has_tunnel: bool = False
    has_loot: bool = False
    has_weak_credentials: bool = False
    is_under_surveillance: bool = False
    clean_logs_count: int = 0
    last_clean_logs_step: int = -100
    detectable_traces: set[str] = field(default_factory=set)

    def add_suspicion(self, amount: float) -> None:
        """Add suspicion, applying surveillance multiplier if active."""
        from src.config import SURVEILLANCE_MULTIPLIER, SUSPICION_MAX

        effective = amount * (SURVEILLANCE_MULTIPLIER if self.is_under_surveillance else 1.0)
        self.suspicion_level = min(SUSPICION_MAX, self.suspicion_level + effective)
        if self.suspicion_level > self.max_suspicion_historical:
            self.max_suspicion_historical = self.suspicion_level

    def reduce_suspicion(self, amount: float, bypass_floor: bool = False) -> None:
        """Reduce suspicion by `amount` (positive value)."""
        from src.config import SUSPICION_MIN, WAIT_FLOOR_DIVISOR

        if bypass_floor:
            self.suspicion_level = max(SUSPICION_MIN, self.suspicion_level - amount)
        else:
            floor = self.max_suspicion_historical / WAIT_FLOOR_DIVISOR
            self.suspicion_level = max(
                max(SUSPICION_MIN, floor),
                self.suspicion_level - amount,
            )

    def reset_session(self) -> None:
        """Reset Red Team session (used by ROTATE_CREDENTIALS)."""
        if not self.has_backdoor:
            self.session_level = SessionLevel.NONE
            self.has_tunnel = False
