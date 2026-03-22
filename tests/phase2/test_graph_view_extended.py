"""Extended graph_view tests for Phase 2 complete features.

Tests node icons, attacker path, pulse effect, flash effect.
"""

from __future__ import annotations

import math

import pygame
import pytest

from src.environment.network import build_fixed_network
from src.environment.node import DiscoveryLevel, OsType, SessionLevel
from src.visualization.graph_view import (
    NodePositions,
    draw_attacker_path,
    draw_flash_effect,
    draw_node_icons,
    draw_pulse_effect,
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
def positions(network) -> NodePositions:
    area = pygame.Rect(50, 50, 700, 500)
    from src.visualization.graph_view import compute_layout
    return compute_layout(network.graph, area)


@pytest.fixture
def fogged() -> set[int]:
    return set()  # no fog for drawing tests


class TestDrawNodeIcons:
    def test_no_crash_all_os_types(self, surface, network, positions, fogged) -> None:
        """draw_node_icons must not crash for any OsType."""
        # Set all nodes to be discovered so icons are drawn
        for node in network.nodes.values():
            node.discovery_level = DiscoveryLevel.DISCOVERED
        draw_node_icons(surface, network, positions, fogged)  # must not raise

    def test_no_crash_with_fog(self, surface, network, positions) -> None:
        """Fogged nodes draw a small circle — must not crash."""
        all_fogged = set(network.nodes.keys())
        draw_node_icons(surface, network, positions, all_fogged)  # must not raise

    def test_windows_node_drawn(self, surface, positions, fogged) -> None:
        """A WINDOWS node should modify the surface (something is drawn)."""
        net = build_fixed_network(seed=42)
        # Force one node to WINDOWS + DISCOVERED
        node = next(iter(net.nodes.values()))
        node.os_type = OsType.WINDOWS
        node.discovery_level = DiscoveryLevel.DISCOVERED

        before = surface.copy()
        draw_node_icons(surface, net, positions, fogged)
        arr_before = pygame.surfarray.array3d(before)
        arr_after = pygame.surfarray.array3d(surface)
        assert not (arr_before == arr_after).all(), "Expected pixels to change"

    def test_linux_node_drawn(self, surface, positions, fogged) -> None:
        """A LINUX node should modify the surface."""
        net = build_fixed_network(seed=42)
        for node in net.nodes.values():
            node.os_type = OsType.LINUX
            node.discovery_level = DiscoveryLevel.DISCOVERED
        before = surface.copy()
        draw_node_icons(surface, net, positions, fogged)
        arr_before = pygame.surfarray.array3d(before)
        arr_after = pygame.surfarray.array3d(surface)
        assert not (arr_before == arr_after).all()

    def test_network_device_drawn(self, surface, positions, fogged) -> None:
        """A NETWORK_DEVICE node should modify the surface."""
        net = build_fixed_network(seed=42)
        for node in net.nodes.values():
            node.os_type = OsType.NETWORK_DEVICE
            node.discovery_level = DiscoveryLevel.DISCOVERED
        before = surface.copy()
        draw_node_icons(surface, net, positions, fogged)
        arr_before = pygame.surfarray.array3d(before)
        arr_after = pygame.surfarray.array3d(surface)
        assert not (arr_before == arr_after).all()


class TestDrawAttackerPath:
    def test_no_crash_empty_path(self, surface, positions) -> None:
        draw_attacker_path(surface, [], positions)  # must not raise

    def test_no_crash_single_node(self, surface, positions) -> None:
        draw_attacker_path(surface, [0], positions)  # single node — no edges drawn

    def test_no_crash_valid_path(self, surface, positions, network) -> None:
        node_ids = list(network.nodes.keys())[:3]
        draw_attacker_path(surface, node_ids, positions)  # must not raise

    def test_path_modifies_surface(self, surface, positions, network) -> None:
        """A 2-node path should draw a line and modify the surface."""
        node_ids = list(network.nodes.keys())[:2]
        before = surface.copy()
        draw_attacker_path(surface, node_ids, positions)
        arr_before = pygame.surfarray.array3d(before)
        arr_after = pygame.surfarray.array3d(surface)
        assert not (arr_before == arr_after).all()

    def test_invalid_node_ids_do_not_crash(self, surface, positions) -> None:
        draw_attacker_path(surface, [999, 1000], positions)  # not in positions


class TestDrawPulseEffect:
    def test_no_crash_no_root_nodes(self, surface, network, positions) -> None:
        """No ROOT nodes — nothing to draw, no crash."""
        draw_pulse_effect(surface, network, positions, anim_time=0.0)

    def test_no_crash_with_root_node(self, surface, network, positions) -> None:
        """A ROOT node should trigger pulse drawing without crash."""
        node = next(iter(network.nodes.values()))
        node.session_level = SessionLevel.ROOT
        node.discovery_level = DiscoveryLevel.ENUMERATED
        draw_pulse_effect(surface, network, positions, anim_time=0.0)
        draw_pulse_effect(surface, network, positions, anim_time=1.5)
        draw_pulse_effect(surface, network, positions, anim_time=math.pi)

    def test_pulse_varies_with_anim_time(self, surface, network, positions) -> None:
        """Different anim_time values should produce different pulse radii (smoke test)."""
        node = next(iter(network.nodes.values()))
        node.session_level = SessionLevel.ROOT
        # Just ensure no crash at several time points
        for t in [0.0, 0.5, 1.0, 1.5, 2.0, 10.0]:
            s = pygame.Surface((800, 600))
            draw_pulse_effect(s, network, positions, anim_time=t)


class TestDrawFlashEffect:
    def test_no_crash_empty(self, surface, positions) -> None:
        draw_flash_effect(surface, [], positions)  # must not raise

    def test_no_crash_with_events(self, surface, positions, network) -> None:
        node_id = next(iter(network.nodes.keys()))
        events = [(node_id, 0.4), (node_id, 0.1)]
        draw_flash_effect(surface, events, positions)  # must not raise

    def test_no_crash_expired_event(self, surface, positions, network) -> None:
        """time_remaining=0 should produce alpha=0 — no-op, no crash."""
        node_id = next(iter(network.nodes.keys()))
        events = [(node_id, 0.0)]
        draw_flash_effect(surface, events, positions)  # must not raise

    def test_invalid_node_no_crash(self, surface, positions) -> None:
        events = [(9999, 0.3)]
        draw_flash_effect(surface, events, positions)  # must not raise
