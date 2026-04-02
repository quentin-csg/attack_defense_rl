"""CyberEnv — Gymnasium environment for the attack/defense RL project."""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.config import (
    DEFAULT_MAX_STEPS,
    MAX_NODES,
    N_GLOBAL_FEATURES,
    N_NODE_FEATURES,
    REWARD_DETECTED,
    REWARD_DETECTED_RATIO,
    REWARD_EXFILTRATE,
    REWARD_EXFILTRATE_RATIO,
    REWARD_EXPLORATION_NODES,
    REWARD_NEW_NODE_COMPROMISED,
    REWARD_NEW_NODE_DISCOVERED,
    REWARD_PER_STEP,
    REWARD_REPEATED_ACTION,
    REWARD_ROOT_OBTAINED,
    SUSPICION_MAX,
)
from src.environment.action_mask import compute_action_mask
from src.environment.actions import (
    N_ACTION_TYPES,
    ActionResult,
    ActionType,
    decode_action,
    execute_action,
)
from src.environment.fog_of_war import FogOfWar
from src.environment.network import Network, build_fixed_network
from src.environment.node import DiscoveryLevel, SessionLevel

logger = logging.getLogger(__name__)

# Action types that move the agent to the target node on success.
_MOVEMENT_ACTIONS: frozenset[ActionType] = frozenset({
    ActionType.EXPLOIT,
    ActionType.BRUTE_FORCE,
    ActionType.LATERAL_MOVE,
    ActionType.PIVOT,
})


class CyberEnv(gym.Env):
    """Gymnasium environment for Red Team cyber attack simulation.

    Observation space (Dict):
        - node_features: Box(MAX_NODES, N_NODE_FEATURES)
        - adjacency: Box(MAX_NODES, MAX_NODES)
        - node_exists_mask: MultiBinary(MAX_NODES)
        - fog_mask: MultiBinary(MAX_NODES)
        - agent_position: Discrete(MAX_NODES)
        - global_features: Box(N_GLOBAL_FEATURES,)

    Action space: Discrete(N_ACTION_TYPES * MAX_NODES)
        Decoded as: action_type = action // MAX_NODES, target = action % MAX_NODES
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        network: Network | None = None,
        network_factory: Callable[[int | None], Network] | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        render_mode: str | None = None,
        blue_team: Any = None,
    ) -> None:
        """Initialise CyberEnv.

        Args:
            network: Fixed network (original behaviour).
            network_factory: Callable(seed) → Network. When provided, a new
                network is generated on each reset() (PCG mode). Takes
                precedence over ``network`` if both are supplied.
            max_steps: Maximum steps per episode.
            seed: Random seed.
            render_mode: ``"human"`` or ``"rgb_array"``.
            blue_team: Optional ScriptedBlueTeam. None = no defender.
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.blue_team = blue_team
        self._network_factory: Callable[[int | None], Network] | None = network_factory

        if network_factory is not None:
            self.network: Network = network_factory(seed)
        elif network is not None:
            self.network = network
        else:
            self.network = build_fixed_network(seed)

        self.fog = FogOfWar()
        self._base_adjacency = self._build_adjacency()

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(low=-1.0, high=1.0,
                    shape=(MAX_NODES, N_NODE_FEATURES), dtype=np.float32),
                "adjacency": spaces.Box(low=0.0, high=1.0,
                    shape=(MAX_NODES, MAX_NODES), dtype=np.float32),
                "node_exists_mask": spaces.MultiBinary(MAX_NODES),
                "fog_mask": spaces.MultiBinary(MAX_NODES),
                "agent_position": spaces.Discrete(MAX_NODES),
                "global_features": spaces.Box(low=0.0, high=1.0,
                    shape=(N_GLOBAL_FEATURES,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(N_ACTION_TYPES * MAX_NODES)

        # Dynamic rewards — scale with episode budget so ratio REWARD/cost > 1 always holds.
        self._exfiltrate_reward: float = max(
            REWARD_EXFILTRATE,
            REWARD_EXFILTRATE_RATIO * abs(REWARD_PER_STEP) * self.max_steps,
        )
        self._detected_reward: float = -max(
            abs(REWARD_DETECTED),
            REWARD_DETECTED_RATIO * self._exfiltrate_reward,
        )
        # Exploration scale: keeps total discovery budget ~constant across network sizes.
        self._exploration_scale: float = min(1.0, REWARD_EXPLORATION_NODES / self.network.num_nodes)

        self.current_step: int = 0
        self.agent_position: int = 0
        self.has_dumped_creds: bool = False
        self.last_action: int | None = None
        self.episode_reward: float = 0.0
        self.exfiltrated: bool = False
        self._detected: bool = False

        # Adjacency dirty flag: set when Blue Team isolates/restores a node.
        self._adjacency_dirty: bool = False
        self._last_isolated_set: frozenset[int] = frozenset()

        self._action_log: list[Any] = []
        self._attacker_path: list[int] = []
        self._last_action_type: str | None = None
        self._last_action_target: int | None = None
        self._last_action_success: bool = False
        self._renderer: Any = None

    def _build_adjacency(self) -> np.ndarray:
        """Build the adjacency matrix from the network graph."""
        adj = np.zeros((MAX_NODES, MAX_NODES), dtype=np.float32)
        for u, v in self.network.graph.edges():
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        return adj

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        if self._network_factory is not None:
            episode_seed = self._rng.randint(0, 2**31)
            self.network = self._network_factory(episode_seed)
            self._exploration_scale = min(1.0, REWARD_EXPLORATION_NODES / self.network.num_nodes)

        self.network.reset_all_nodes()
        self._base_adjacency = self._build_adjacency()

        if self.blue_team is not None:
            self.blue_team.reset()

        if self._renderer is not None:
            self._renderer.reset_layout()

        self.current_step = 0
        self.has_dumped_creds = False
        self.last_action = None
        self.episode_reward = 0.0
        self.exfiltrated = False
        self._detected = False
        self._adjacency_dirty = False
        self._last_isolated_set = frozenset()
        self._action_log = []
        self._last_action_type = None
        self._last_action_target = None
        self._last_action_success = False

        entry = self.network.entry_node_id
        self.agent_position = entry
        self._attacker_path = [entry]
        entry_node = self.network.get_node(entry)
        entry_node.discovery_level = DiscoveryLevel.DISCOVERED
        entry_node.session_level = SessionLevel.USER

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one action and return (obs, reward, terminated, truncated, info)."""
        self.current_step += 1
        reward = REWARD_PER_STEP

        action_type, target_node_id = decode_action(action)

        # WAIT is exempt from the repeated-action penalty.
        if action == self.last_action and action_type != ActionType.WAIT:
            reward += REWARD_REPEATED_ACTION
        self.last_action = action

        if self._exploration_scale < 1.0:
            _pre_disc = {nid: n.discovery_level for nid, n in self.network.nodes.items()}
            _pre_sess = {nid: n.session_level for nid, n in self.network.nodes.items()}
        else:
            _pre_disc = _pre_sess = None

        if target_node_id not in self.network.nodes:
            result = ActionResult(
                success=False, reward=0.0,
                info={"action": "INVALID", "target": target_node_id},
            )
        else:
            result = execute_action(
                action_type=action_type,
                target_node_id=target_node_id,
                network=self.network,
                current_step=self.current_step,
                rng=self._rng,
                agent_position=self.agent_position,
                has_dumped_creds=self.has_dumped_creds,
            )

        reward += result.reward

        if action_type in (ActionType.EXFILTRATE, ActionType.LIST_FILES) and result.success:
            reward += self._exfiltrate_reward - REWARD_EXFILTRATE

        if _pre_disc is not None:
            scale_adj = self._exploration_scale - 1.0
            for nid, node in self.network.nodes.items():
                if _pre_disc[nid] == DiscoveryLevel.UNKNOWN and node.discovery_level != DiscoveryLevel.UNKNOWN:
                    reward += REWARD_NEW_NODE_DISCOVERED * scale_adj
                if _pre_sess[nid] == SessionLevel.NONE and node.session_level != SessionLevel.NONE:  # type: ignore[index]
                    reward += REWARD_NEW_NODE_COMPROMISED * scale_adj
                if _pre_sess[nid] == SessionLevel.USER and node.session_level == SessionLevel.ROOT:  # type: ignore[index]
                    reward += REWARD_ROOT_OBTAINED * scale_adj

        if action_type == ActionType.CREDENTIAL_DUMP and result.success:
            self.has_dumped_creds = True

        if action_type in _MOVEMENT_ACTIONS and result.success and target_node_id in self.network.nodes:
            target = self.network.get_node(target_node_id)
            if target.session_level != SessionLevel.NONE:
                self.agent_position = target_node_id
                if target_node_id not in self._attacker_path:
                    self._attacker_path.append(target_node_id)

        self._last_action_type = action_type.name
        self._last_action_target = target_node_id
        self._last_action_success = result.success
        susp_delta = result.suspicion_delta
        susp_str = f" ({susp_delta:+.0f} Susp)" if susp_delta else ""
        status_str = "SUCCESS" if result.success else "FAIL"
        log_text = f"[{self.current_step}] {action_type.name} Node {target_node_id}: {status_str}{susp_str}"
        color_key = "red_success" if result.success else "red_fail"
        self._action_log.append((self.current_step, log_text, color_key))

        if action_type in (ActionType.EXFILTRATE, ActionType.LIST_FILES) and result.success:
            self.exfiltrated = True

        if self.blue_team is not None and not self.exfiltrated:
            blue_actions = self.blue_team.act(self.network, self.current_step)
            for ba in blue_actions:
                log_text = (
                    f"[{self.current_step}] BLUE {ba.action_type}"
                    f" Node {ba.target_node_id}: {ba.details}"
                )
                self._action_log.append((self.current_step, log_text, "blue_action"))
                if ba.action_type in ("ISOLATE_NODE", "RESTORE_NODE"):
                    self._adjacency_dirty = True
                if ba.action_type == "ROTATE_CREDENTIALS":
                    self.has_dumped_creds = False

        # Exfiltration wins over simultaneous detection — prevents ambiguous outcomes.
        detected = (
            not self.exfiltrated
            and any(n.suspicion_level >= SUSPICION_MAX for n in self.network.nodes.values())
        )
        if detected:
            reward += self._detected_reward
            self._detected = True

        self.episode_reward += reward
        terminated = self.exfiltrated or detected
        truncated = self.current_step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()
        info["action_result"] = result.info
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Return the boolean mask of valid actions (for MaskablePPO)."""
        return compute_action_mask(
            network=self.network,
            current_step=self.current_step,
            agent_position=self.agent_position,
            has_dumped_creds=self.has_dumped_creds,
        )

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Build the current observation."""
        # Rebuild adjacency when topology changed (Blue Team ISOLATE/RESTORE or direct API).
        current_isolated = frozenset(self.network.isolated_edges.keys())
        if self._adjacency_dirty or current_isolated != self._last_isolated_set:
            self._base_adjacency = self._build_adjacency()
            self._adjacency_dirty = False
            self._last_isolated_set = current_isolated
        return self.fog.build_observation(
            nodes=self.network.nodes,
            adjacency=self._base_adjacency,
            current_step=self.current_step,
            max_steps=self.max_steps,
            num_real_nodes=self.network.num_nodes,
            agent_position=self.agent_position,
        )

    def _get_info(self) -> dict[str, Any]:
        """Build the info dict."""
        n_compromised = sum(
            1 for n in self.network.nodes.values() if n.session_level != SessionLevel.NONE
        )
        n_discovered = sum(
            1 for n in self.network.nodes.values() if n.discovery_level != DiscoveryLevel.UNKNOWN
        )
        max_suspicion = max((n.suspicion_level for n in self.network.nodes.values()), default=0.0)
        return {
            "step": self.current_step,
            "episode_reward": self.episode_reward,
            "n_compromised": n_compromised,
            "n_discovered": n_discovered,
            "max_suspicion": max_suspicion,
            "exfiltrated": self.exfiltrated,
            "detected": self._detected,
            "agent_position": self.agent_position,
            "per_node_suspicion": {nid: n.suspicion_level for nid, n in self.network.nodes.items()},
            "network_topology": {
                "edges": list(self.network.graph.edges())
                + [e for edges in self.network.isolated_edges.values() for e in edges],
                "nodes": {
                    str(nid): {"os_type": n.os_type.name}
                    for nid, n in self.network.nodes.items()
                },
                "entry_node_id": self.network.entry_node_id,
                "target_node_id": self.network.target_node_id,
            },
        }

    def _build_render_state(self) -> Any:
        """Build a RenderState snapshot for the Pygame renderer.

        Lazy-imports RenderState to avoid pulling Pygame into training mode.
        """
        from src.visualization.render_state import LogEntry, RenderState

        info = self._get_info()
        n_total = self.network.num_nodes
        n_unknown = sum(
            1 for n in self.network.nodes.values() if n.discovery_level == DiscoveryLevel.UNKNOWN
        )
        fog_pct = (n_unknown / n_total * 100.0) if n_total > 0 else 0.0
        log_entries = [LogEntry(step=s, text=t, color_key=c) for s, t, c in self._action_log]
        per_node_susp = {nid: n.suspicion_level for nid, n in self.network.nodes.items()}

        return RenderState(
            network=self.network,
            agent_position=self.agent_position,
            step=info["step"],
            episode_reward=info["episode_reward"],
            n_compromised=info["n_compromised"],
            n_discovered=info["n_discovered"],
            total_nodes=n_total,
            max_suspicion=info["max_suspicion"],
            fog_percentage=fog_pct,
            action_log=log_entries,
            last_action_type=self._last_action_type,
            last_action_target=self._last_action_target,
            last_action_success=self._last_action_success,
            attacker_path=list(self._attacker_path),
            per_node_suspicion=per_node_susp,
        )

    def render(self) -> np.ndarray | None:
        """Render the current environment state."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from src.visualization.renderer import PygameRenderer
            self._renderer = PygameRenderer(headless=(self.render_mode == "rgb_array"))

        state = self._build_render_state()

        if self.render_mode == "human":
            self._renderer.update(state)
            return None
        elif self.render_mode == "rgb_array":
            return self._renderer.get_frame(state)
        return None

    @property
    def renderer_controls(self):
        """DashboardControls owned by the renderer (pause, speed). None if not yet created."""
        if self._renderer is None:
            return None
        return self._renderer.controls

    def close(self) -> None:
        """Clean up the renderer and Pygame resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
