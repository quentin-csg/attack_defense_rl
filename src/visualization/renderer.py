"""Pygame window manager for the Attack & Defense RL visualization.

PygameRenderer owns the Pygame window lifecycle and orchestrates the
drawing calls. It is created lazily by CyberEnv on the first render() call,
so importing Pygame is avoided entirely during headless training.

Usage (via CyberEnv):
    env = CyberEnv(render_mode="human")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()   # creates window on first call, updates on subsequent calls
    env.close()    # cleans up Pygame
"""

from __future__ import annotations

import os

import numpy as np
import pygame

from src.environment.network import Network
from src.visualization import theme
from src.visualization.graph_view import (
    NodePositions,
    compute_layout,
    draw_agent_halo,
    draw_edges,
    draw_node_labels,
    draw_nodes,
    draw_special_markers,
    draw_stats_overlay,
    fog_node_ids,
)


def _load_font(size: int) -> pygame.font.Font:
    """Load a monospace font with a real fallback chain.

    Uses pygame.font.match_font() to check whether each candidate actually
    exists on this system before trying to load it. Falls back to Pygame's
    built-in default font if none of the candidates are available.
    """
    for name in theme.FONT_CANDIDATES:
        if pygame.font.match_font(name) is not None:
            return pygame.font.SysFont(name, size)
    return pygame.font.Font(None, size)  # built-in bitmap font


class PygameRenderer:
    """Manages the Pygame window and renders the network state each step.

    Attributes:
        is_open: True while the window has not been closed by the user.
    """

    def __init__(self, headless: bool = False) -> None:
        """Initialize Pygame and create the display surface.

        Args:
            headless: If True, use the dummy SDL video driver so no real window
                is created. Required for tests and CI environments.
        """
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption(theme.WINDOW_TITLE)

        self._screen: pygame.Surface = pygame.display.set_mode(
            (theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT)
        )
        self._clock: pygame.time.Clock = pygame.time.Clock()
        self._font_stats: pygame.font.Font = _load_font(theme.FONT_SIZE_STATS)
        self._font_label: pygame.font.Font = _load_font(theme.FONT_SIZE_LABEL)

        self._layout: NodePositions | None = None
        self._open: bool = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the window is still open (not closed by the user)."""
        return self._open

    def update(
        self,
        network: Network,
        agent_position: int,
        step: int,
        episode_reward: float,
        n_compromised: int,
        max_suspicion: float,
    ) -> None:
        """Redraw the frame and flip the display. Non-blocking.

        Also pumps pygame events so the window stays responsive.

        Args:
            network: The current network state.
            agent_position: Current Red Team node ID.
            step: Current episode step.
            episode_reward: Cumulative episode reward.
            n_compromised: Number of compromised nodes.
            max_suspicion: Highest suspicion level (0-100).
        """
        self._handle_events()
        if not self._open:
            return

        self._ensure_layout(network)
        self._render_frame(
            self._screen,
            network,
            agent_position,
            step,
            episode_reward,
            n_compromised,
            max_suspicion,
        )
        pygame.display.flip()
        self._clock.tick(theme.FPS)

    def get_frame(
        self,
        network: Network,
        agent_position: int,
        step: int,
        episode_reward: float,
        n_compromised: int,
        max_suspicion: float,
    ) -> np.ndarray:
        """Render to an offscreen surface and return as an RGB numpy array.

        Used for render_mode="rgb_array". Useful for headless tests and
        video recording.

        Returns:
            np.ndarray of shape (H, W, 3) and dtype uint8.
        """
        self._ensure_layout(network)

        offscreen = pygame.Surface((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT))
        self._render_frame(
            offscreen,
            network,
            agent_position,
            step,
            episode_reward,
            n_compromised,
            max_suspicion,
        )
        # surfarray.array3d returns (W, H, 3); transpose to (H, W, 3).
        raw = pygame.surfarray.array3d(offscreen)
        return np.transpose(raw, (1, 0, 2))

    def reset_layout(self) -> None:
        """Invalidate the cached layout so it is recomputed on the next render.

        Call this when the network topology changes (e.g., after PCG in Phase 5).
        Not needed in Phase 1 (fixed topology), but the hook is here.
        """
        self._layout = None

    def close(self) -> None:
        """Quit Pygame and release resources."""
        if pygame.get_init():
            pygame.quit()
        self._open = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_layout(self, network: Network) -> None:
        """Compute and cache the node layout if it has not been computed yet."""
        if self._layout is None:
            graph_area = pygame.Rect(
                theme.GRAPH_AREA_LEFT + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_TOP + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_WIDTH - 2 * theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_HEIGHT - 2 * theme.GRAPH_MARGIN,
            )
            self._layout = compute_layout(network.graph, graph_area)

    def _handle_events(self) -> None:
        """Process the Pygame event queue. Sets _open=False on QUIT."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._open = False

    def _render_frame(
        self,
        surface: pygame.Surface,
        network: Network,
        agent_position: int,
        step: int,
        episode_reward: float,
        n_compromised: int,
        max_suspicion: float,
    ) -> None:
        """Draw one complete frame onto the given surface (back-to-front)."""
        assert self._layout is not None  # _ensure_layout must be called first

        surface.fill(theme.BG_COLOR)
        fogged = fog_node_ids(network)

        draw_edges(surface, network.graph, self._layout, fogged)
        draw_nodes(surface, network, self._layout, fogged)
        draw_agent_halo(surface, agent_position, self._layout)
        draw_special_markers(
            surface,
            network.entry_node_id,
            network.target_node_id,
            self._layout,
        )
        draw_node_labels(surface, self._layout, self._font_label)
        draw_stats_overlay(
            surface,
            self._font_stats,
            step=step,
            episode_reward=episode_reward,
            n_compromised=n_compromised,
            total_nodes=network.num_nodes,
            max_suspicion=max_suspicion,
        )
