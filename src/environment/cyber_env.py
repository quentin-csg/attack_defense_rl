"""CyberEnv — main Gymnasium environment for the attack/defense RL project.

Implements the full Red Team environment with:
- Flat Discrete action space with action masking (CORRECTION 1)
- Padded observations for variable network sizes (CORRECTION 2)
- Calibrated rewards (CORRECTION 4)
"""

from __future__ import annotations

import logging
import random
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
    REWARD_PER_STEP,
    REWARD_REPEATED_ACTION,
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


class CyberEnv(gym.Env):
    """Gymnasium environment for Red Team cyber attack simulation.

    The agent controls a Red Team attacker that must navigate a network,
    compromise nodes, and exfiltrate data while staying under the Blue Team's
    detection thresholds.

    Observation space (Dict):
        - node_features: Box(MAX_NODES, N_NODE_FEATURES) — per-node features
        - adjacency: Box(MAX_NODES, MAX_NODES) — adjacency matrix
        - node_exists_mask: MultiBinary(MAX_NODES) — which slots are real nodes
        - fog_mask: MultiBinary(MAX_NODES) — which nodes are discovered
        - agent_position: Discrete(MAX_NODES) — current Red Team position
        - global_features: Box(N_GLOBAL_FEATURES,) — step/compromised/discovered

    Action space: Discrete(N_ACTION_TYPES * MAX_NODES)
        Decoded as: action_type = action // MAX_NODES, target = action % MAX_NODES
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        network: Network | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self._seed = seed
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Network setup
        self._network_factory = network
        self.network: Network = network if network is not None else build_fixed_network(seed)

        # Fog of war
        self.fog = FogOfWar()

        # Precompute adjacency template
        self._base_adjacency = self._build_adjacency()

        # --- Spaces (defined once in __init__, not in reset) ---
        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(MAX_NODES, N_NODE_FEATURES),
                    dtype=np.float32,
                ),
                "adjacency": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(MAX_NODES, MAX_NODES),
                    dtype=np.float32,
                ),
                "node_exists_mask": spaces.MultiBinary(MAX_NODES),
                "fog_mask": spaces.MultiBinary(MAX_NODES),
                "agent_position": spaces.Discrete(MAX_NODES),
                "global_features": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(N_GLOBAL_FEATURES,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = spaces.Discrete(N_ACTION_TYPES * MAX_NODES)

        # Episode state
        self.current_step: int = 0
        self.agent_position: int = 0
        self.has_dumped_creds: bool = False
        self.last_action: int | None = None
        self.episode_reward: float = 0.0
        self.exfiltrated: bool = False
        self._detected: bool = False

        # Renderer (created lazily on first render() call)
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
        """Reset the environment for a new episode.

        Returns:
            (observation, info) tuple as required by Gymnasium >= 0.26.
        """
        super().reset(seed=seed)

        if seed is not None:
            self._rng = random.Random(seed)
            self._np_rng = np.random.default_rng(seed)

        # Reset network state
        self.network.reset_all_nodes()
        self._base_adjacency = self._build_adjacency()

        # Invalidate renderer layout so it is recomputed on next render() call.
        # Needed for Phase 5 (PCG) where topology changes between episodes.
        if self._renderer is not None:
            self._renderer.reset_layout()

        # Episode state
        self.current_step = 0
        self.has_dumped_creds = False
        self.last_action = None
        self.episode_reward = 0.0
        self.exfiltrated = False
        self._detected = False

        # Red Team starts at entry node with DISCOVERED status
        entry = self.network.entry_node_id
        self.agent_position = entry
        entry_node = self.network.get_node(entry)
        entry_node.discovery_level = DiscoveryLevel.DISCOVERED
        entry_node.session_level = SessionLevel.USER

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one action and return (obs, reward, terminated, truncated, info).

        Returns 5 values as required by Gymnasium.
        """
        self.current_step += 1
        reward = REWARD_PER_STEP  # time pressure

        # Decode action before repeat check so WAIT can be exempted
        action_type, target_node_id = decode_action(action)

        # Repeated action penalty (WAIT is exempt — waiting is strategically valid)
        if action == self.last_action and action_type != ActionType.WAIT:
            reward += REWARD_REPEATED_ACTION
        self.last_action = action

        # Validate target exists
        if target_node_id not in self.network.nodes:
            # Invalid target — treat as failed WAIT
            result = ActionResult(
                success=False,
                reward=0.0,
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

        # Track credential dumps
        if action_type == ActionType.CREDENTIAL_DUMP and result.success:
            self.has_dumped_creds = True

        # Update agent position (move to target if we have a session there)
        if target_node_id in self.network.nodes:
            target = self.network.get_node(target_node_id)
            if target.session_level != SessionLevel.NONE:
                self.agent_position = target_node_id

        # Check exfiltration success
        if action_type == ActionType.EXFILTRATE and result.success:
            self.exfiltrated = True

        # Check detection (any node suspicion >= max)
        detected = any(n.suspicion_level >= SUSPICION_MAX for n in self.network.nodes.values())
        if detected:
            reward += REWARD_DETECTED
            self._detected = True

        self.episode_reward += reward

        # Termination conditions
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
        }

    def render(self) -> np.ndarray | None:
        """Render the current environment state.

        Returns:
            np.ndarray of shape (H, W, 3) for render_mode="rgb_array", else None.
        """
        if self.render_mode is None:
            return None

        # Lazy import: avoids importing Pygame in training mode (render_mode=None).
        if self._renderer is None:
            from src.visualization.renderer import PygameRenderer

            # rgb_array mode renders to an offscreen buffer — no real window needed.
            self._renderer = PygameRenderer(headless=(self.render_mode == "rgb_array"))

        info = self._get_info()
        kwargs: dict[str, Any] = {
            "network": self.network,
            "agent_position": self.agent_position,
            "step": info["step"],
            "episode_reward": info["episode_reward"],
            "n_compromised": info["n_compromised"],
            "max_suspicion": info["max_suspicion"],
        }

        if self.render_mode == "human":
            self._renderer.update(**kwargs)
            return None
        elif self.render_mode == "rgb_array":
            return self._renderer.get_frame(**kwargs)
        return None

    def close(self) -> None:
        """Clean up the renderer and Pygame resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
