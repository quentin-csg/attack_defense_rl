"""Tests for ui_panels drawing functions (headless Pygame)."""

from __future__ import annotations

import pygame
import pytest

from src.environment.network import build_fixed_network
from src.visualization.render_state import LogEntry, RenderState
from src.visualization.ui_panels import (
    draw_action_log,
    draw_sidebar_panels,
    draw_stats_panel,
    draw_suspicion_bars,
    get_panel_header_rects,
)


def _make_state(network) -> RenderState:
    return RenderState(
        network=network,
        agent_position=0,
        step=42,
        episode_reward=+150.0,
        n_compromised=2,
        n_discovered=5,
        total_nodes=network.num_nodes,
        max_suspicion=65.0,
        fog_percentage=37.5,
        action_log=[
            LogEntry(step=1, text="[1] SCAN Node 0: SUCCESS", color_key="red_success"),
            LogEntry(step=2, text="[2] EXPLOIT Node 1: FAIL", color_key="red_fail"),
            LogEntry(step=3, text="[3] BLUE TEAM: ROTATE", color_key="blue_action"),
        ],
        last_action_type="EXPLOIT",
        last_action_target=1,
        last_action_success=False,
        attacker_path=[0, 1],
        per_node_suspicion={i: float(i * 10) for i in range(network.num_nodes)},
    )


@pytest.fixture
def network():
    return build_fixed_network(seed=42)


@pytest.fixture
def surface():
    pygame.init()
    surf = pygame.Surface((800, 600))
    yield surf
    pygame.quit()


@pytest.fixture
def font(surface):
    return pygame.font.Font(None, 16)


class TestDrawStatsPanel:
    def test_no_crash(self, surface, font, network) -> None:
        rect = pygame.Rect(0, 0, 220, 110)
        state = _make_state(network)
        draw_stats_panel(surface, rect, font, state)  # must not raise

    def test_surface_modified(self, surface, font, network) -> None:
        """Stats panel must actually draw something on the surface."""
        rect = pygame.Rect(0, 0, 220, 110)
        before = surface.copy()
        state = _make_state(network)
        draw_stats_panel(surface, rect, font, state)
        # At least one pixel should have changed
        arr_before = pygame.surfarray.array3d(before)
        arr_after = pygame.surfarray.array3d(surface)
        assert not (arr_before == arr_after).all()


class TestDrawSuspicionBars:
    def test_no_crash_empty(self, surface, font) -> None:
        rect = pygame.Rect(0, 400, 220, 180)
        draw_suspicion_bars(surface, rect, font, {})  # empty — must not raise

    def test_no_crash_with_data(self, surface, font) -> None:
        rect = pygame.Rect(0, 400, 220, 180)
        suspicion = {i: float(i * 10) for i in range(8)}
        draw_suspicion_bars(surface, rect, font, suspicion)  # must not raise

    def test_no_crash_max_suspicion(self, surface, font) -> None:
        rect = pygame.Rect(0, 400, 220, 180)
        suspicion = {0: 0.0, 1: 30.0, 2: 60.0, 3: 80.0, 4: 100.0}
        draw_suspicion_bars(surface, rect, font, suspicion)  # must not raise


class TestDrawActionLog:
    def test_no_crash_empty(self, surface, font) -> None:
        rect = pygame.Rect(580, 0, 220, 600)
        draw_action_log(surface, rect, font, [])  # must not raise

    def test_no_crash_with_entries(self, surface, font) -> None:
        rect = pygame.Rect(580, 0, 220, 600)
        entries = [
            LogEntry(step=i, text=f"[{i}] ACTION Node {i}: SUCCESS", color_key="red_success")
            for i in range(20)
        ]
        draw_action_log(surface, rect, font, entries)  # must not raise

    def test_no_crash_overflow(self, surface, font) -> None:
        """More entries than fit should not crash."""
        rect = pygame.Rect(580, 0, 220, 100)
        entries = [
            LogEntry(step=i, text=f"[{i}] ACTION: done", color_key="red_success")
            for i in range(200)
        ]
        draw_action_log(surface, rect, font, entries)  # must not raise


class TestDrawSidebarPanels:
    def test_no_crash_all_collapsed(self, surface, font, network) -> None:
        rect = pygame.Rect(0, 126, 204, 340)
        fonts = {"header": font, "body": font}
        panel_expanded = {"Scan": False, "Nodes": False, "Attacker": False, "Stats": False, "Haze": False}
        state = _make_state(network)
        draw_sidebar_panels(surface, rect, fonts, state, panel_expanded)  # must not raise

    def test_no_crash_all_expanded(self, surface, font, network) -> None:
        rect = pygame.Rect(0, 126, 204, 340)
        fonts = {"header": font, "body": font}
        panel_expanded = {"Scan": True, "Nodes": True, "Attacker": True, "Stats": True, "Haze": True}
        state = _make_state(network)
        draw_sidebar_panels(surface, rect, fonts, state, panel_expanded)  # must not raise


class TestGetPanelHeaderRects:
    def test_returns_dict_for_all_panels(self, surface, font) -> None:
        sidebar_rect = pygame.Rect(0, 126, 204, 340)
        rects = get_panel_header_rects(sidebar_rect, font)
        assert isinstance(rects, dict)
        for name in ["Scan", "Nodes", "Attacker", "Stats", "Haze"]:
            assert name in rects
            assert isinstance(rects[name], pygame.Rect)

    def test_rects_within_sidebar(self, surface, font) -> None:
        sidebar_rect = pygame.Rect(0, 126, 204, 340)
        rects = get_panel_header_rects(sidebar_rect, font)
        for name, rect in rects.items():
            assert rect.left >= sidebar_rect.left, f"{name} header too far left"
            assert rect.right <= sidebar_rect.right + 2, f"{name} header too far right"
