"""Tests for Phase 2 minimal visualization.

All tests run headless via SDL_VIDEODRIVER=dummy (set in conftest.py).
Tests verify behavior and correctness, not pixel-perfect output.
"""

import numpy as np
import pygame
import pytest

from src.environment.cyber_env import CyberEnv
from src.environment.network import build_fixed_network
from src.environment.node import DiscoveryLevel, Node, OsType, SessionLevel
from src.visualization import theme
from src.visualization.graph_view import (
    compute_layout,
    get_node_color,
)
from src.visualization.renderer import PygameRenderer

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_network():
    """The standard 8-node fixed network."""
    return build_fixed_network(seed=42)


@pytest.fixture
def renderer():
    """A headless PygameRenderer. Closed after the test."""
    r = PygameRenderer(headless=True)
    yield r
    r.close()


@pytest.fixture
def rgb_env():
    """CyberEnv with render_mode='rgb_array'. Closed after the test."""
    env = CyberEnv(seed=42, render_mode="rgb_array")
    env.reset()
    yield env
    env.close()


@pytest.fixture
def human_env():
    """CyberEnv with render_mode='human' (headless). Closed after the test."""
    env = CyberEnv(seed=42, render_mode="human")
    env.reset()
    yield env
    env.close()


# ---------------------------------------------------------------------------
# TestThemeConstants
# ---------------------------------------------------------------------------


class TestThemeConstants:
    def test_window_dimensions_positive(self) -> None:
        assert theme.WINDOW_WIDTH > 0
        assert theme.WINDOW_HEIGHT > 0

    def test_fps_positive(self) -> None:
        assert theme.FPS > 0

    def test_colors_are_rgb_tuples(self) -> None:
        color_attrs = [
            "BG_COLOR",
            "COLOR_NODE_OFFLINE",
            "COLOR_NODE_UNKNOWN",
            "COLOR_NODE_ROOT",
            "COLOR_NODE_USER",
            "COLOR_NODE_ENUMERATED",
            "COLOR_NODE_DISCOVERED",
            "COLOR_ENTRY_MARKER",
            "COLOR_TARGET_MARKER",
            "COLOR_AGENT_HALO",
            "COLOR_EDGE",
        ]
        for attr in color_attrs:
            color = getattr(theme, attr)
            assert isinstance(color, tuple), f"{attr} must be a tuple"
            assert len(color) >= 3, f"{attr} must have at least 3 channels"
            for channel in color[:3]:
                assert 0 <= channel <= 255, f"{attr} channel {channel} out of range"

    def test_node_sizes_positive(self) -> None:
        assert theme.NODE_RADIUS > 0
        assert theme.NODE_RADIUS_UNKNOWN > 0
        assert theme.AGENT_HALO_RADIUS > theme.NODE_RADIUS
        assert theme.SPECIAL_MARKER_RADIUS > theme.NODE_RADIUS

    def test_font_candidates_is_list(self) -> None:
        assert isinstance(theme.FONT_CANDIDATES, list)
        assert len(theme.FONT_CANDIDATES) > 0


# ---------------------------------------------------------------------------
# TestNodeColor
# ---------------------------------------------------------------------------


class TestNodeColor:
    def _node(
        self,
        discovery: DiscoveryLevel = DiscoveryLevel.UNKNOWN,
        session: SessionLevel = SessionLevel.NONE,
        is_online: bool = True,
    ) -> Node:
        node = Node(0, OsType.LINUX, discovery_level=discovery)
        node.session_level = session
        node.is_online = is_online
        return node

    def test_unknown_node_is_dim_gray(self) -> None:
        node = self._node(discovery=DiscoveryLevel.UNKNOWN)
        assert get_node_color(node) == theme.COLOR_NODE_UNKNOWN

    def test_discovered_none_is_yellow(self) -> None:
        node = self._node(discovery=DiscoveryLevel.DISCOVERED, session=SessionLevel.NONE)
        assert get_node_color(node) == theme.COLOR_NODE_DISCOVERED

    def test_enumerated_none_is_cyan(self) -> None:
        node = self._node(discovery=DiscoveryLevel.ENUMERATED, session=SessionLevel.NONE)
        assert get_node_color(node) == theme.COLOR_NODE_ENUMERATED

    def test_user_session_is_red(self) -> None:
        node = self._node(discovery=DiscoveryLevel.ENUMERATED, session=SessionLevel.USER)
        assert get_node_color(node) == theme.COLOR_NODE_USER

    def test_root_session_is_bright_red(self) -> None:
        node = self._node(discovery=DiscoveryLevel.ENUMERATED, session=SessionLevel.ROOT)
        assert get_node_color(node) == theme.COLOR_NODE_ROOT

    def test_offline_is_dark_gray(self) -> None:
        node = self._node(
            discovery=DiscoveryLevel.ENUMERATED,
            session=SessionLevel.NONE,
            is_online=False,
        )
        assert get_node_color(node) == theme.COLOR_NODE_OFFLINE

    def test_offline_overrides_root_session(self) -> None:
        """Offline priority > ROOT session."""
        node = self._node(
            discovery=DiscoveryLevel.ENUMERATED,
            session=SessionLevel.ROOT,
            is_online=False,
        )
        assert get_node_color(node) == theme.COLOR_NODE_OFFLINE

    def test_root_overrides_enumerated(self) -> None:
        """ROOT session priority > ENUMERATED with no session."""
        node = self._node(discovery=DiscoveryLevel.ENUMERATED, session=SessionLevel.ROOT)
        assert get_node_color(node) == theme.COLOR_NODE_ROOT


# ---------------------------------------------------------------------------
# TestComputeLayout
# ---------------------------------------------------------------------------


class TestComputeLayout:
    def test_all_real_nodes_have_positions(self, fixed_network, pygame_surface) -> None:
        area = pygame.Rect(100, 100, 800, 500)
        positions = compute_layout(fixed_network.graph, area)
        for node_id in fixed_network.nodes:
            assert node_id in positions, f"Node {node_id} missing from layout"

    def test_positions_within_area(self, fixed_network, pygame_surface) -> None:
        area = pygame.Rect(100, 100, 800, 500)
        positions = compute_layout(fixed_network.graph, area)
        for node_id, (px, py) in positions.items():
            assert area.left <= px <= area.right, f"Node {node_id} x={px} outside area"
            assert area.top <= py <= area.bottom, f"Node {node_id} y={py} outside area"

    def test_layout_is_deterministic(self, fixed_network, pygame_surface) -> None:
        area = pygame.Rect(100, 100, 800, 500)
        pos1 = compute_layout(fixed_network.graph, area)
        pos2 = compute_layout(fixed_network.graph, area)
        for node_id in fixed_network.nodes:
            assert pos1[node_id] == pos2[node_id], f"Node {node_id} position not deterministic"

    def test_empty_graph_returns_empty(self, pygame_surface) -> None:
        import networkx as nx

        area = pygame.Rect(0, 0, 800, 600)
        positions = compute_layout(nx.Graph(), area)
        assert positions == {}

    def test_single_node_placed_at_center(self, pygame_surface) -> None:
        """A single-node graph should be placed at the area center, not the corner."""
        import networkx as nx

        g = nx.Graph()
        g.add_node(0)
        area = pygame.Rect(100, 100, 800, 500)
        positions = compute_layout(g, area)
        assert 0 in positions
        px, py = positions[0]
        assert px == area.centerx
        assert py == area.centery

    def test_disconnected_graph_does_not_crash(self, pygame_surface) -> None:
        """kamada_kawai fails on disconnected graphs — must fall back to spring_layout."""
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from([0, 1, 2])
        g.add_edge(0, 1)
        # Node 2 is isolated — graph is disconnected
        area = pygame.Rect(0, 0, 800, 600)
        positions = compute_layout(g, area)
        assert set(positions.keys()) == {0, 1, 2}


# ---------------------------------------------------------------------------
# TestPygameRenderer
# ---------------------------------------------------------------------------


class TestPygameRenderer:
    def test_create_and_close(self) -> None:
        r = PygameRenderer(headless=True)
        assert r.is_open is True
        r.close()
        assert r.is_open is False

    def test_update_does_not_crash(self, renderer, fixed_network) -> None:
        renderer.update(
            network=fixed_network,
            agent_position=0,
            step=1,
            episode_reward=-0.5,
            n_compromised=1,
            max_suspicion=10.0,
        )

    def test_update_multiple_steps(self, renderer, fixed_network) -> None:
        """Calling update multiple times should not crash."""
        for step in range(5):
            renderer.update(
                network=fixed_network,
                agent_position=0,
                step=step,
                episode_reward=float(-step),
                n_compromised=0,
                max_suspicion=float(step * 3),
            )

    def test_get_frame_returns_ndarray(self, renderer, fixed_network) -> None:
        frame = renderer.get_frame(
            network=fixed_network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            max_suspicion=0.0,
        )
        assert isinstance(frame, np.ndarray)

    def test_get_frame_shape(self, renderer, fixed_network) -> None:
        frame = renderer.get_frame(
            network=fixed_network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            max_suspicion=0.0,
        )
        assert frame.shape == (theme.WINDOW_HEIGHT, theme.WINDOW_WIDTH, 3)

    def test_get_frame_dtype(self, renderer, fixed_network) -> None:
        frame = renderer.get_frame(
            network=fixed_network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            max_suspicion=0.0,
        )
        assert frame.dtype == np.uint8

    def test_reset_layout_triggers_recompute(self, renderer, fixed_network) -> None:
        """reset_layout() should invalidate the layout and cause recomputation."""
        renderer.update(
            network=fixed_network,
            agent_position=0,
            step=0,
            episode_reward=0.0,
            n_compromised=0,
            max_suspicion=0.0,
        )
        renderer.reset_layout()
        assert renderer._layout is None
        # Should recompute without crashing on next render
        renderer.update(
            network=fixed_network,
            agent_position=0,
            step=1,
            episode_reward=-0.5,
            n_compromised=0,
            max_suspicion=0.0,
        )
        assert renderer._layout is not None


# ---------------------------------------------------------------------------
# TestCyberEnvRender
# ---------------------------------------------------------------------------


class TestCyberEnvRender:
    def test_render_mode_none_returns_none(self) -> None:
        env = CyberEnv(seed=42, render_mode=None)
        env.reset()
        result = env.render()
        assert result is None
        env.close()

    def test_render_mode_none_no_renderer_created(self) -> None:
        env = CyberEnv(seed=42, render_mode=None)
        env.reset()
        env.render()
        assert env._renderer is None
        env.close()

    def test_render_mode_human_returns_none(self, human_env) -> None:
        result = human_env.render()
        assert result is None

    def test_render_mode_rgb_array_returns_ndarray(self, rgb_env) -> None:
        frame = rgb_env.render()
        assert isinstance(frame, np.ndarray)

    def test_render_mode_rgb_array_shape(self, rgb_env) -> None:
        frame = rgb_env.render()
        assert frame.shape == (theme.WINDOW_HEIGHT, theme.WINDOW_WIDTH, 3)

    def test_render_mode_rgb_array_dtype(self, rgb_env) -> None:
        frame = rgb_env.render()
        assert frame.dtype == np.uint8

    def test_close_cleans_up_renderer(self, rgb_env) -> None:
        rgb_env.render()
        assert rgb_env._renderer is not None
        rgb_env.close()
        assert rgb_env._renderer is None

    def test_render_after_several_steps(self, rgb_env) -> None:
        """Render should work correctly after multiple env steps."""
        from src.environment.actions import ActionType, encode_action

        for _ in range(10):
            rgb_env.step(encode_action(ActionType.WAIT, 0))
            frame = rgb_env.render()
            assert isinstance(frame, np.ndarray)

    def test_renderer_lazy_creation(self) -> None:
        """Renderer should not be created until render() is called."""
        env = CyberEnv(seed=42, render_mode="rgb_array")
        env.reset()
        assert env._renderer is None
        env.render()
        assert env._renderer is not None
        env.close()

    def test_render_before_reset_does_not_crash(self) -> None:
        """render() before reset() should not crash (renders default state)."""
        env = CyberEnv(seed=42, render_mode="rgb_array")
        # Deliberately do NOT call reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        env.close()

    def test_double_close_is_safe(self, rgb_env) -> None:
        """close() called twice must not raise."""
        rgb_env.render()
        rgb_env.close()
        rgb_env.close()  # second close — must not crash

    def test_close_then_render_recreates_renderer(self) -> None:
        """After close(), render() should recreate the renderer transparently."""
        env = CyberEnv(seed=42, render_mode="rgb_array")
        env.reset()
        env.render()
        env.close()
        assert env._renderer is None
        # render() should lazily create a new renderer
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert env._renderer is not None
        env.close()

    def test_render_after_node_compromised(self, rgb_env) -> None:
        """Render must work correctly after a node is compromised (USER session)."""
        from src.environment.node import SessionLevel

        rgb_env.network.get_node(1).session_level = SessionLevel.USER
        rgb_env.network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        frame = rgb_env.render()
        assert isinstance(frame, np.ndarray)

    def test_render_after_root_obtained(self, rgb_env) -> None:
        """Render must work correctly after a node reaches ROOT."""
        from src.environment.node import SessionLevel

        rgb_env.network.get_node(1).session_level = SessionLevel.ROOT
        rgb_env.network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        frame = rgb_env.render()
        assert isinstance(frame, np.ndarray)

    def test_render_after_node_isolated(self, rgb_env) -> None:
        """Render must work correctly after a node is isolated (offline)."""
        # Node 2 is adjacent to 0 — isolating it disconnects the graph sub-component.
        rgb_env.network.get_node(2).is_online = False
        frame = rgb_env.render()
        assert isinstance(frame, np.ndarray)

    def test_reset_invalidates_renderer_layout(self) -> None:
        """reset() must invalidate the renderer's cached layout."""
        env = CyberEnv(seed=42, render_mode="rgb_array")
        env.reset()
        env.render()  # computes layout
        assert env._renderer._layout is not None
        env.reset()   # should call reset_layout()
        assert env._renderer._layout is None
        env.close()

    def test_no_pygame_import_without_render(self) -> None:
        """CyberEnv with render_mode=None must not create a renderer."""
        env = CyberEnv(seed=42, render_mode=None)
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        assert env._renderer is None
        env.close()
