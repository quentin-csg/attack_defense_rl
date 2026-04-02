"""Scripted Blue Team agent — rule-based defender with adaptive patrol."""

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
    PATROL_HOT_DURATION,
    PATROL_HOT_WEIGHT,
    PATROL_INTERVAL,
    PATROL_NEIGHBOR_WEIGHT,
    PATROL_SUSPICION_DELTA_MIN,
)
from src.environment.network import Network


class BlueAction(NamedTuple):
    """Record of a single Blue Team action for logging and rendering."""

    action_type: str
    target_node_id: int
    details: str


class ScriptedBlueTeam:
    """Rule-based Blue Team agent.

    Thresholds are randomised within a noise band on each ``reset()`` so
    the Red agent cannot memorise exact trigger points.

    Patrol targeting is weighted by recent suspicion activity. Nodes where
    suspicion rose since the last step are flagged "hot" and patrolled more
    often. Hot status decays after ``hot_duration`` steps, so the blue team
    naturally follows the attacker as it moves through the network.

    Args:
        seed: Random seed for threshold noise and patrol selection.
        patrol_interval: Mean steps between patrols (Poisson rate).
        isolate_duration: Steps before an isolated node is auto-restored.
        hot_duration: Steps a node stays hot after its last suspicion rise.
            Callers should pass ``max(PATROL_HOT_DURATION, max_steps // 20)``.
    """

    def __init__(
        self,
        seed: int = 42,
        patrol_interval: int = PATROL_INTERVAL,
        isolate_duration: int | None = None,
        hot_duration: int | None = None,
    ) -> None:
        self._rng = random.Random(seed)
        self._patrol_interval = patrol_interval
        self._isolate_duration: int = (
            isolate_duration if isolate_duration is not None else BLUE_ISOLATE_DURATION
        )
        self._hot_duration: int = (
            hot_duration if hot_duration is not None else PATROL_HOT_DURATION
        )
        self._isolated_at: dict[int, int] = {}
        self._last_rotate: dict[int, int] = {}
        self._surveillance_logged: set[int] = set()
        self._prev_suspicion: dict[int, float] = {}
        self._hot_nodes: dict[int, int] = {}
        self._randomize_thresholds()

    def reset(self) -> None:
        """Re-randomise noisy thresholds for a new episode."""
        self._isolated_at = {}
        self._last_rotate = {}
        self._surveillance_logged = set()
        self._prev_suspicion = {}
        self._hot_nodes = {}
        self._randomize_thresholds()

    def act(self, network: Network, current_step: int) -> list[BlueAction]:
        """Execute one Blue Team step and return the list of actions taken."""
        actions: list[BlueAction] = []

        to_restore = [
            nid for nid, step in self._isolated_at.items()
            if current_step - step >= self._isolate_duration
        ]
        for nid in to_restore:
            network.restore_node(nid)
            node = network.get_node(nid)
            node.is_under_surveillance = False
            was_logged = nid in self._surveillance_logged
            self._surveillance_logged.discard(nid)
            del self._isolated_at[nid]
            detail = f"Isolation expired after {self._isolate_duration} steps"
            if was_logged:
                detail += " — surveillance lifted"
            actions.append(BlueAction(action_type="RESTORE_NODE", target_node_id=nid, details=detail))

        self._update_hot_nodes(network, current_step)

        online_nodes = [nid for nid, n in network.nodes.items() if n.is_online]
        patrol_scale = max(1, len(online_nodes) // 10)
        if self._rng.random() < patrol_scale / self._patrol_interval:
            patrol_action = self._patrol_from(network, online_nodes, current_step)
            if patrol_action is not None:
                actions.append(patrol_action)

        for node_id, node in network.nodes.items():
            if not node.is_online:
                continue
            if node.suspicion_level >= self._isolate_thresh:
                actions.append(self._isolate(network, node_id, current_step))
            elif node.suspicion_level >= self._rotate_thresh:
                rotate_action = self._rotate(network, node_id, current_step)
                if rotate_action is not None:
                    actions.append(rotate_action)
            elif node.suspicion_level >= self._alert_thresh:
                alert_action = self._alert(network, node_id)
                if alert_action is not None:
                    actions.append(alert_action)

        # Snapshot after all blue actions so next delta only captures red-caused rises.
        self._prev_suspicion = {nid: n.suspicion_level for nid, n in network.nodes.items()}

        return actions

    def _randomize_thresholds(self) -> None:
        self._alert_thresh = max(0.0, min(100.0,
            BLUE_ALERT_THRESHOLD + self._rng.uniform(-BLUE_ALERT_NOISE, BLUE_ALERT_NOISE)))
        self._rotate_thresh = max(0.0, min(100.0,
            BLUE_ROTATE_THRESHOLD + self._rng.uniform(-BLUE_ROTATE_NOISE, BLUE_ROTATE_NOISE)))
        self._isolate_thresh = max(0.0, min(100.0,
            BLUE_ISOLATE_THRESHOLD + self._rng.uniform(-BLUE_ISOLATE_NOISE, BLUE_ISOLATE_NOISE)))

    def _update_hot_nodes(self, network: Network, current_step: int) -> None:
        """Flag nodes where suspicion rose since the last step (red activity signal)."""
        for node_id, node in network.nodes.items():
            if not node.is_online:
                continue
            prev = self._prev_suspicion.get(node_id, 0.0)
            if node.suspicion_level - prev >= PATROL_SUSPICION_DELTA_MIN:
                self._hot_nodes[node_id] = current_step

    def _build_patrol_weights(
        self, online_nodes: list[int], network: Network, current_step: int
    ) -> list[float]:
        """Compute per-node patrol weights. Hot nodes 5×, their neighbors 2.5×, others 1.0."""
        self._hot_nodes = {
            nid: step
            for nid, step in self._hot_nodes.items()
            if current_step - step < self._hot_duration
        }
        if not self._hot_nodes:
            return [1.0] * len(online_nodes)
        hot_neighbors: set[int] = {
            nb
            for hot_nid in self._hot_nodes
            for nb in network.get_neighbors(hot_nid)
            if nb not in self._hot_nodes
        }
        return [
            PATROL_HOT_WEIGHT if nid in self._hot_nodes
            else PATROL_NEIGHBOR_WEIGHT if nid in hot_neighbors
            else 1.0
            for nid in online_nodes
        ]

    def _patrol_from(
        self, network: Network, online_nodes: list[int], current_step: int
    ) -> BlueAction | None:
        """Inspect a node for detectable Red traces, weighted toward hot nodes."""
        if not online_nodes:
            return None
        weights = self._build_patrol_weights(online_nodes, network, current_step)
        target_id = self._rng.choices(online_nodes, weights=weights, k=1)[0]
        node = network.get_node(target_id)
        if node.detectable_traces:
            node.add_suspicion(PATROL_DETECTION_SUSPICION)
            already_surveilled = node.is_under_surveillance
            node.is_under_surveillance = True
            self._surveillance_logged.add(target_id)
            traces_str = ", ".join(sorted(node.detectable_traces))
            suffix = "" if already_surveilled else " — now under surveillance"
            return BlueAction(
                action_type="PATROL",
                target_node_id=target_id,
                details=f"Traces detected: {traces_str} (+{PATROL_DETECTION_SUSPICION:.0f} Susp){suffix}",
            )
        return BlueAction(action_type="PATROL", target_node_id=target_id, details="No traces found")

    def _alert(self, network: Network, node_id: int) -> BlueAction | None:
        node = network.get_node(node_id)
        node.is_under_surveillance = True
        if node_id in self._surveillance_logged:
            return None
        self._surveillance_logged.add(node_id)
        return BlueAction(
            action_type="ALERT",
            target_node_id=node_id,
            details=f"Susp={node.suspicion_level:.0f} — node under surveillance",
        )

    def _rotate(self, network: Network, node_id: int, current_step: int) -> BlueAction | None:
        if current_step - self._last_rotate.get(node_id, -BLUE_ROTATE_COOLDOWN - 1) < BLUE_ROTATE_COOLDOWN:
            return None
        self._last_rotate[node_id] = current_step
        node = network.get_node(node_id)
        had_session = node.session_level.value > 0
        node.reset_session()
        protected = had_session and node.session_level.value > 0
        detail = (
            f"Susp={node.suspicion_level:.0f} — session preserved (backdoor)"
            if protected
            else f"Susp={node.suspicion_level:.0f} — session invalidated"
        )
        return BlueAction(action_type="ROTATE_CREDENTIALS", target_node_id=node_id, details=detail)

    def _isolate(self, network: Network, node_id: int, current_step: int) -> BlueAction:
        node = network.get_node(node_id)
        network.isolate_node(node_id)
        self._isolated_at[node_id] = current_step
        node.is_under_surveillance = True
        self._surveillance_logged.add(node_id)
        return BlueAction(
            action_type="ISOLATE_NODE",
            target_node_id=node_id,
            details=f"Susp={node.suspicion_level:.0f} — node isolated",
        )

    @property
    def alert_threshold(self) -> float:
        return self._alert_thresh

    @property
    def rotate_threshold(self) -> float:
        return self._rotate_thresh

    @property
    def isolate_threshold(self) -> float:
        return self._isolate_thresh

    @property
    def hot_nodes(self) -> dict[int, int]:
        return dict(self._hot_nodes)
