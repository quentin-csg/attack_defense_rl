"""Scripted Blue Team agent for Phase 4.

Rule-based defender that acts after each Red Team step. Uses noisy thresholds
(re-randomised on reset) so the Red agent cannot learn to exploit fixed timing.

Actions:
    ALERT            — marks a node under surveillance (suspicion x2 multiplier)
    ROTATE_CREDENTIALS — invalidates Red sessions on a node (unless backdoor)
    ISOLATE_NODE     — disconnects a node from the network
    PATROL           — stochastic trace detection (Poisson process, CORRECTION 3)

Usage:
    blue = ScriptedBlueTeam(seed=42)
    blue.reset()                                 # re-randomise thresholds
    actions = blue.act(network, current_step)    # returns list[BlueAction]
"""

from __future__ import annotations

import random
from typing import NamedTuple

from src.config import (
    BLUE_ALERT_NOISE,
    BLUE_ALERT_THRESHOLD,
    BLUE_ISOLATE_DURATION,
    BLUE_ISOLATE_NOISE,
    BLUE_ISOLATE_THRESHOLD,
    BLUE_ROTATE_COOLDOWN,
    BLUE_ROTATE_NOISE,
    BLUE_ROTATE_THRESHOLD,
    PATROL_DETECTION_SUSPICION,
    PATROL_INTERVAL,
)
from src.environment.network import Network


class BlueAction(NamedTuple):
    """Record of a single Blue Team action for logging and rendering."""

    action_type: str   # "ALERT", "ROTATE_CREDENTIALS", "ISOLATE_NODE", "PATROL"
    target_node_id: int
    details: str       # human-readable description for the action log


class ScriptedBlueTeam:
    """Rule-based Blue Team agent.

    Thresholds are randomised within a noise band on each ``reset()`` call so
    the Red agent cannot memorise exact trigger points.

    Args:
        seed: Random seed for threshold noise and patrol selection.
        patrol_interval: Mean number of steps between patrols (Poisson rate).
                         Configurable for Phase 6 dynamic handicap.
    """

    def __init__(self, seed: int = 42, patrol_interval: int = PATROL_INTERVAL) -> None:
        self._rng = random.Random(seed)
        self._patrol_interval = patrol_interval
        self._isolated_at: dict[int, int] = {}   # node_id -> step it was isolated
        self._last_rotate: dict[int, int] = {}   # node_id -> last step ROTATE fired
        self._randomize_thresholds()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Re-randomise noisy thresholds for a new episode."""
        self._isolated_at = {}
        self._last_rotate = {}
        self._randomize_thresholds()

    def act(self, network: Network, current_step: int) -> list[BlueAction]:
        """Execute one Blue Team step and return the list of actions taken.

        Order of operations:
        1. Stochastic patrol (Poisson process).
        2. Threshold checks per node — highest threshold first so ISOLATE
           takes priority over ROTATE, which takes priority over ALERT.

        Args:
            network: The current network state (modified in place).
            current_step: Current episode step (unused but available for
                          future extensions like cooldown tracking).

        Returns:
            List of BlueAction records in the order they were applied.
        """
        actions: list[BlueAction] = []

        # 0. Auto-restore isolated nodes whose duration has elapsed (A6)
        to_restore = [
            nid for nid, step in self._isolated_at.items()
            if current_step - step >= BLUE_ISOLATE_DURATION
        ]
        for nid in to_restore:
            network.restore_node(nid)
            del self._isolated_at[nid]
            actions.append(BlueAction(
                action_type="RESTORE_NODE",
                target_node_id=nid,
                details=f"Isolation expired after {BLUE_ISOLATE_DURATION} steps",
            ))

        # 1. Stochastic patrol (CORRECTION 3 — Poisson, not deterministic)
        if self._rng.random() < 1.0 / self._patrol_interval:
            patrol_action = self._patrol(network)
            if patrol_action is not None:
                actions.append(patrol_action)

        # 2. Threshold-based responses — check from highest to lowest priority
        #    so each node only triggers one action per step.
        for node_id, node in network.nodes.items():
            if not node.is_online:
                continue  # already isolated — no further action needed

            if node.suspicion_level >= self._isolate_thresh:
                actions.append(self._isolate(network, node_id, current_step))
            elif node.suspicion_level >= self._rotate_thresh:
                rotate_action = self._rotate(network, node_id, current_step)
                if rotate_action is not None:
                    actions.append(rotate_action)
            elif node.suspicion_level >= self._alert_thresh:
                actions.append(self._alert(network, node_id))

        return actions

    # ------------------------------------------------------------------
    # Threshold randomisation
    # ------------------------------------------------------------------

    def _randomize_thresholds(self) -> None:
        """Draw noisy thresholds within their configured bands."""
        self._alert_thresh = BLUE_ALERT_THRESHOLD + self._rng.uniform(
            -BLUE_ALERT_NOISE, BLUE_ALERT_NOISE
        )
        self._rotate_thresh = BLUE_ROTATE_THRESHOLD + self._rng.uniform(
            -BLUE_ROTATE_NOISE, BLUE_ROTATE_NOISE
        )
        self._isolate_thresh = BLUE_ISOLATE_THRESHOLD + self._rng.uniform(
            -BLUE_ISOLATE_NOISE, BLUE_ISOLATE_NOISE
        )
        # Clamp to valid suspicion range [0, 100]
        self._alert_thresh = max(0.0, min(100.0, self._alert_thresh))
        self._rotate_thresh = max(0.0, min(100.0, self._rotate_thresh))
        self._isolate_thresh = max(0.0, min(100.0, self._isolate_thresh))

    # ------------------------------------------------------------------
    # Action implementations
    # ------------------------------------------------------------------

    def _patrol(self, network: Network) -> BlueAction | None:
        """Inspect a random online node for detectable Red traces.

        If traces are found: add PATROL_DETECTION_SUSPICION and mark the
        node under surveillance (future Red actions = suspicion x2).
        """
        online_nodes = [nid for nid, n in network.nodes.items() if n.is_online]
        if not online_nodes:
            return None

        target_id = self._rng.choice(online_nodes)
        node = network.get_node(target_id)

        if node.detectable_traces:
            node.add_suspicion(PATROL_DETECTION_SUSPICION)
            node.is_under_surveillance = True
            traces_str = ", ".join(sorted(node.detectable_traces))
            return BlueAction(
                action_type="PATROL",
                target_node_id=target_id,
                details=f"Traces detected: {traces_str} (+{PATROL_DETECTION_SUSPICION:.0f} Susp)",
            )

        return BlueAction(
            action_type="PATROL",
            target_node_id=target_id,
            details="No traces found",
        )

    def _alert(self, network: Network, node_id: int) -> BlueAction:
        """Mark a node under enhanced surveillance."""
        node = network.get_node(node_id)
        node.is_under_surveillance = True
        return BlueAction(
            action_type="ALERT",
            target_node_id=node_id,
            details=f"Susp={node.suspicion_level:.0f} — node under surveillance",
        )

    def _rotate(self, network: Network, node_id: int, current_step: int) -> BlueAction | None:
        """Invalidate Red sessions on a node via credential rotation.

        Calls reset_session() which respects has_backdoor (if backdoor is
        installed, the session survives rotation).

        Returns None if ROTATE_COOLDOWN has not elapsed since the last rotation
        on this node (downgraded to no-action; ALERT was already applied).
        """
        if current_step - self._last_rotate.get(node_id, -BLUE_ROTATE_COOLDOWN - 1) < BLUE_ROTATE_COOLDOWN:
            return None  # cooldown active — skip rotation this step
        self._last_rotate[node_id] = current_step
        node = network.get_node(node_id)
        had_session = node.session_level.value > 0
        node.reset_session()
        protected = had_session and node.session_level.value > 0  # backdoor protected it
        detail = (
            f"Susp={node.suspicion_level:.0f} — session preserved (backdoor)"
            if protected
            else f"Susp={node.suspicion_level:.0f} — session invalidated"
        )
        return BlueAction(
            action_type="ROTATE_CREDENTIALS",
            target_node_id=node_id,
            details=detail,
        )

    def _isolate(self, network: Network, node_id: int, current_step: int) -> BlueAction:
        """Disconnect a node from the network (auto-restores after BLUE_ISOLATE_DURATION)."""
        node = network.get_node(node_id)
        network.isolate_node(node_id)
        self._isolated_at[node_id] = current_step
        return BlueAction(
            action_type="ISOLATE_NODE",
            target_node_id=node_id,
            details=f"Susp={node.suspicion_level:.0f} — node isolated",
        )

    # ------------------------------------------------------------------
    # Inspection helpers (for tests and debugging)
    # ------------------------------------------------------------------

    @property
    def alert_threshold(self) -> float:
        """Current noisy ALERT threshold."""
        return self._alert_thresh

    @property
    def rotate_threshold(self) -> float:
        """Current noisy ROTATE_CREDENTIALS threshold."""
        return self._rotate_thresh

    @property
    def isolate_threshold(self) -> float:
        """Current noisy ISOLATE_NODE threshold."""
        return self._isolate_thresh
