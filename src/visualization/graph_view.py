"""Graph drawing functions for the Pygame visualization.

Each function draws one layer onto a pygame.Surface (back-to-front order).
This module is stateless — it receives all data it needs as parameters.
The renderer calls these in order:
  edges → attacker_path → node_icons → pulse → flash → halo → markers → labels
"""

from __future__ import annotations

import math

import networkx as nx
import pygame

from src.environment.network import Network
from src.environment.node import DiscoveryLevel, Node, OsType, SessionLevel
from src.visualization import theme

# Type alias for computed pixel positions.
NodePositions = dict[int, tuple[int, int]]


def compute_layout(graph: nx.Graph, area: pygame.Rect) -> NodePositions:
    """Compute pixel positions for all nodes using kamada_kawai_layout.

    kamada_kawai is deterministic (no seed needed) and produces stable,
    aesthetically consistent layouts for small graphs.

    Args:
        graph: The NetworkX graph.
        area: The pygame.Rect defining the drawable area (with margins already applied).

    Returns:
        dict mapping node_id → (pixel_x, pixel_y).
    """
    if len(graph.nodes) == 0:
        return {}

    # Single node: place at the center of the area.
    if len(graph.nodes) == 1:
        node_id = next(iter(graph.nodes))
        return {node_id: (area.centerx, area.centery)}

    # kamada_kawai_layout is deterministic and stable for connected graphs.
    # For disconnected graphs (e.g., after Blue Team ISOLATE_NODE), it raises
    # NetworkXException — fall back to spring_layout with fixed seed.
    try:
        raw: dict[int, list[float]] = nx.kamada_kawai_layout(graph)
    except nx.NetworkXException:
        raw = nx.spring_layout(graph, seed=42)

    # raw values are in approximately [-1.0, 1.0]. Map to pixel coordinates.
    xs = [v[0] for v in raw.values()]
    ys = [v[1] for v in raw.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    positions: NodePositions = {}
    for node_id, (nx_x, nx_y) in raw.items():
        px = area.left + int((nx_x - x_min) / x_range * area.width)
        py = area.top + int((nx_y - y_min) / y_range * area.height)
        positions[node_id] = (px, py)

    return positions


def get_node_color(node: Node) -> tuple[int, int, int]:
    """Return the RGB color for a node based on its current state.

    Priority (highest first):
      1. Offline (isolated) → dark gray
      2. Unknown (fog of war) → dim gray
      3. ROOT session → bright red
      4. USER session → red
      5. ENUMERATED + no session → cyan
      6. DISCOVERED + no session → yellow
    """
    if not node.is_online:
        return theme.COLOR_NODE_OFFLINE
    if node.discovery_level == DiscoveryLevel.UNKNOWN:
        return theme.COLOR_NODE_UNKNOWN
    if node.session_level == SessionLevel.ROOT:
        return theme.COLOR_NODE_ROOT
    if node.session_level == SessionLevel.USER:
        return theme.COLOR_NODE_USER
    if node.discovery_level == DiscoveryLevel.ENUMERATED:
        return theme.COLOR_NODE_ENUMERATED
    # DISCOVERED + NONE (or anything else that reached here)
    return theme.COLOR_NODE_DISCOVERED


def fog_node_ids(network: Network) -> set[int]:
    """Return the set of node IDs that are currently unknown (fogged)."""
    return {
        nid for nid, node in network.nodes.items()
        if node.discovery_level == DiscoveryLevel.UNKNOWN
    }


def draw_edges(
    surface: pygame.Surface,
    graph: nx.Graph,
    positions: NodePositions,
    fogged: set[int],
) -> None:
    """Draw edges between nodes.

    Edges where both endpoints are known are drawn in full color.
    Edges where at least one endpoint is fogged are skipped entirely
    (the agent should not see connections to unknown nodes).

    Args:
        surface: Target surface to draw on.
        graph: NetworkX graph (provides edge list).
        positions: Pixel positions per node.
        fogged: Set of node IDs that are unknown (fog of war).
    """
    for u, v in graph.edges():
        if u not in positions or v not in positions:
            continue
        if u in fogged or v in fogged:
            continue
        pygame.draw.line(
            surface,
            theme.COLOR_EDGE,
            positions[u],
            positions[v],
            theme.EDGE_WIDTH,
        )


def draw_nodes(
    surface: pygame.Surface,
    network: Network,
    positions: NodePositions,
    fogged: set[int],
) -> None:
    """Draw all nodes as colored circles (minimal Phase 2 fallback).

    Preserved for backward compatibility. New code should use draw_node_icons().
    """
    for node_id, node in network.nodes.items():
        if node_id not in positions:
            continue
        pos = positions[node_id]
        color = get_node_color(node)
        if node_id in fogged:
            pygame.draw.circle(surface, color, pos, theme.NODE_RADIUS_UNKNOWN)
        else:
            pygame.draw.circle(surface, color, pos, theme.NODE_RADIUS)


def draw_node_icons(
    surface: pygame.Surface,
    network: Network,
    positions: NodePositions,
    fogged: set[int],
) -> None:
    """Draw all nodes as geometric icons based on OsType.

    Icon shapes:
      - WINDOWS (workstation): filled rectangle with a small stand (monitor)
      - LINUX (server):        two stacked rectangles (server rack)
      - NETWORK_DEVICE:        circle with small triangle arrows (router)

    Fogged nodes are drawn as small dim circles.

    Args:
        surface: Target surface.
        network: The network (provides Node objects for coloring).
        positions: Pixel positions per node.
        fogged: Set of node IDs that are unknown.
    """
    half = theme.ICON_SIZE // 2

    for node_id, node in network.nodes.items():
        if node_id not in positions:
            continue
        px, py = positions[node_id]
        color = get_node_color(node)

        if node_id in fogged:
            pygame.draw.circle(surface, color, (px, py), theme.NODE_RADIUS_UNKNOWN)
            continue

        if node.os_type == OsType.WINDOWS:
            # Monitor: rectangle body + narrow stand
            body_w, body_h = half + 4, half - 2
            pygame.draw.rect(surface, color, pygame.Rect(px - body_w // 2, py - body_h // 2, body_w, body_h))
            pygame.draw.rect(surface, color, pygame.Rect(px - 3, py + body_h // 2, 6, 4))  # stand

        elif node.os_type == OsType.LINUX:
            # Server rack: two stacked rectangles
            rack_w, rack_h = half + 2, (half - 2) // 2
            pygame.draw.rect(surface, color, pygame.Rect(px - rack_w // 2, py - rack_h - 1, rack_w, rack_h))
            pygame.draw.rect(surface, color, pygame.Rect(px - rack_w // 2, py + 1, rack_w, rack_h))
            # Divider line
            pygame.draw.line(surface, theme.BG_COLOR, (px - rack_w // 2, py), (px + rack_w // 2, py), 1)

        else:
            # NETWORK_DEVICE: circle with 4 small triangle arrows
            pygame.draw.circle(surface, color, (px, py), half - 2)
            arrow_len = 6
            for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
                ax = px + dx * (half + 2)
                ay = py + dy * (half + 2)
                bx = ax + dx * arrow_len
                by_ = ay + dy * arrow_len
                pygame.draw.line(surface, color, (ax, ay), (bx, by_), 2)


def draw_agent_halo(
    surface: pygame.Surface,
    agent_position: int,
    positions: NodePositions,
) -> None:
    """Draw concentric glow rings around the agent's current node.

    Draws 3 rings with decreasing alpha for a "halo lumineux" effect.

    Args:
        surface: Target surface.
        agent_position: Node ID where the Red Team agent currently is.
        positions: Pixel positions per node.
    """
    if agent_position not in positions:
        return
    pos = positions[agent_position]
    w = theme.HALO_RING_WIDTH
    for i, (extra_r, width) in enumerate([(0, w), (5, max(1, w - 1)), (10, max(1, w - 2))]):
        r = theme.AGENT_HALO_RADIUS + extra_r
        alpha = 200 - i * 60
        ring_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        color_with_alpha = (*theme.COLOR_AGENT_HALO, alpha)
        pygame.draw.circle(ring_surf, color_with_alpha, (r + 1, r + 1), r, width)
        surface.blit(ring_surf, (pos[0] - r - 1, pos[1] - r - 1))


def draw_attacker_path(
    surface: pygame.Surface,
    attacker_path: list[int],
    positions: NodePositions,
) -> None:
    """Draw thick blue edges along the attacker's visited path.

    Args:
        surface: Target surface.
        attacker_path: Ordered list of node IDs visited (oldest first).
        positions: Pixel positions per node.
    """
    if len(attacker_path) < 2:
        return
    for i in range(len(attacker_path) - 1):
        a, b = attacker_path[i], attacker_path[i + 1]
        if a in positions and b in positions:
            pygame.draw.line(
                surface,
                theme.ATTACKER_PATH_COLOR,
                positions[a],
                positions[b],
                theme.ATTACKER_PATH_WIDTH,
            )


def draw_pulse_effect(
    surface: pygame.Surface,
    network: Network,
    positions: NodePositions,
    anim_time: float,
) -> None:
    """Draw a pulsing ring on ROOT-session nodes.

    The ring radius oscillates between PULSE_MIN_RADIUS and PULSE_MAX_RADIUS
    using a sine wave driven by anim_time.

    Args:
        surface: Target surface.
        network: Network (to find ROOT nodes).
        positions: Pixel positions per node.
        anim_time: Accumulated animation time in seconds.
    """
    t = math.sin(anim_time * theme.PULSE_SPEED)
    r_range = theme.PULSE_MAX_RADIUS - theme.PULSE_MIN_RADIUS
    pulse_r = int(theme.PULSE_MIN_RADIUS + (t + 1) / 2 * r_range)

    for node_id, node in network.nodes.items():
        if node.session_level != SessionLevel.ROOT:
            continue
        if node_id not in positions:
            continue
        pos = positions[node_id]
        ring_surf = pygame.Surface((pulse_r * 2 + 4, pulse_r * 2 + 4), pygame.SRCALPHA)
        alpha = 180
        pygame.draw.circle(ring_surf, (*theme.COLOR_NODE_ROOT, alpha), (pulse_r + 2, pulse_r + 2), pulse_r, 2)
        surface.blit(ring_surf, (pos[0] - pulse_r - 2, pos[1] - pulse_r - 2))


def draw_flash_effect(
    surface: pygame.Surface,
    flash_events: list[tuple[int, float]],
    positions: NodePositions,
) -> None:
    """Draw expanding, fading flash circles for recent actions.

    Each flash event is a (node_id, time_remaining) tuple.
    time_remaining goes from FLASH_DURATION to 0 as the flash fades.

    Args:
        surface: Target surface.
        flash_events: List of (node_id, time_remaining) tuples.
        positions: Pixel positions per node.
    """
    for node_id, time_remaining in flash_events:
        if node_id not in positions:
            continue
        progress = 1.0 - (time_remaining / theme.FLASH_DURATION)  # 0 → 1
        max_r = theme.PULSE_MAX_RADIUS + 12
        r = int(theme.NODE_RADIUS + progress * (max_r - theme.NODE_RADIUS))
        alpha = int(200 * (1.0 - progress))
        if alpha <= 0 or r <= 0:
            continue
        pos = positions[node_id]
        flash_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(flash_surf, (255, 200, 50, alpha), (r + 1, r + 1), r, 2)
        surface.blit(flash_surf, (pos[0] - r - 1, pos[1] - r - 1))


def draw_special_markers(
    surface: pygame.Surface,
    entry_node_id: int,
    target_node_id: int,
    positions: NodePositions,
) -> None:
    """Draw rings around the entry (green) and target/loot (gold) nodes.

    Args:
        surface: Target surface.
        entry_node_id: The DMZ/entry node (Red Team start).
        target_node_id: The data center / exfiltration target.
        positions: Pixel positions per node.
    """
    if entry_node_id in positions:
        pygame.draw.circle(
            surface,
            theme.COLOR_ENTRY_MARKER,
            positions[entry_node_id],
            theme.SPECIAL_MARKER_RADIUS,
            theme.MARKER_RING_WIDTH,
        )
    # Target marker always drawn, even when entry == target (use a larger radius
    # so both rings are visible when they overlap).
    if target_node_id in positions:
        pygame.draw.circle(
            surface,
            theme.COLOR_TARGET_MARKER,
            positions[target_node_id],
            theme.SPECIAL_MARKER_RADIUS + (theme.MARKER_RING_WIDTH + 1 if target_node_id == entry_node_id else 0),
            theme.MARKER_RING_WIDTH,
        )


def draw_node_labels(
    surface: pygame.Surface,
    positions: NodePositions,
    font: pygame.font.Font,
) -> None:
    """Draw "Node X" labels below each node.

    Args:
        surface: Target surface.
        positions: Pixel positions per node.
        font: Pygame Font to use for rendering.
    """
    for node_id, (px, py) in positions.items():
        label = font.render(f"Node {node_id}", True, theme.COLOR_TEXT_LABEL)
        # Center label horizontally below the node circle
        lx = px - label.get_width() // 2
        ly = py + theme.NODE_RADIUS + 4
        surface.blit(label, (lx, ly))


def draw_stats_overlay(
    surface: pygame.Surface,
    font: pygame.font.Font,
    step: int,
    episode_reward: float,
    n_compromised: int,
    total_nodes: int,
    max_suspicion: float,
) -> None:
    """Draw a compact stats block in the top-left corner.

    Kept for backward compatibility. The full dashboard uses ui_panels.draw_stats_panel().

    Args:
        surface: Target surface.
        font: Pygame Font.
        step: Current episode step.
        episode_reward: Cumulative reward this episode.
        n_compromised: Number of compromised nodes.
        total_nodes: Total real nodes in the network.
        max_suspicion: Highest suspicion level across all nodes (0-100).
    """
    lines = [
        ("STEP",        f"{step}"),
        ("REWARD",      f"{episode_reward:+.1f}"),
        ("COMPROMISED", f"{n_compromised}/{total_nodes}"),
        ("SUSPICION",   f"{max_suspicion:.0f}%"),
    ]

    x = theme.STATS_X
    y = theme.STATS_Y
    for key, value in lines:
        key_surf = font.render(f"{key}: ", True, theme.COLOR_TEXT_KEY)
        val_surf = font.render(value, True, theme.COLOR_TEXT_VALUE)
        surface.blit(key_surf, (x, y))
        surface.blit(val_surf, (x + key_surf.get_width(), y))
        y += theme.STATS_LINE_HEIGHT
