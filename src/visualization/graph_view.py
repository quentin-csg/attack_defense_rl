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

def _get_node_id_font() -> pygame.font.Font | None:
    """Return a font for rendering node ID numbers on nodes.

    Returns None if pygame.font is not initialized (e.g. during some tests).
    A new font object is created each call — pygame Font objects are tied to
    a pygame session, so caching across pygame.quit()/init() cycles is unsafe.
    """
    if not pygame.font.get_init():
        return None
    return pygame.font.Font(None, theme.FONT_SIZE_NODE_ID + 4)


# ---------------------------------------------------------------------------
# Private drawing helpers
# ---------------------------------------------------------------------------


def _draw_fog_cloud(surface: pygame.Surface, px: int, py: int) -> None:
    """Draw a cloud-like gray blob for a fogged (unknown) node.

    Uses overlapping circles to form a cloud silhouette, replacing the old
    plain small circle.
    """
    # Central blob (darker)
    pygame.draw.circle(surface, theme.COLOR_FOG_CLOUD_DARK, (px, py), 12)
    # Surrounding blobs (lighter, forming cloud bumps)
    offsets = [(-9, -7, 9), (9, -7, 9), (0, -11, 8), (-13, 2, 8), (13, 2, 8)]
    for dx, dy, r in offsets:
        pygame.draw.circle(surface, theme.COLOR_FOG_CLOUD, (px + dx, py + dy), r)


def _draw_compromised_glow(
    surface: pygame.Surface,
    px: int,
    py: int,
    session_level: SessionLevel,
) -> None:
    """Draw a soft red aura behind a compromised node (USER or ROOT).

    Uses multiple concentric SRCALPHA circles for a diffuse glow effect.
    """
    if session_level == SessionLevel.ROOT:
        glow_r = theme.GLOW_RADIUS_ROOT
        base_color = theme.COLOR_GLOW_ROOT
        layers = 3
    else:
        glow_r = theme.GLOW_RADIUS_USER
        base_color = theme.COLOR_GLOW_USER
        layers = 2

    for i in range(layers):
        r = glow_r + i * 6
        alpha = base_color[3] - i * 15
        if alpha <= 0:
            continue
        glow_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*base_color[:3], alpha), (r + 1, r + 1), r)
        surface.blit(glow_surf, (px - r - 1, py - r - 1))


def _draw_node_bg(
    surface: pygame.Surface,
    px: int,
    py: int,
    border_color: tuple[int, int, int],
) -> None:
    """Draw the dark circular background behind a node icon.

    Args:
        surface: Target surface.
        px, py: Centre pixel coordinates.
        border_color: Thin 1px border in the node's state color.
    """
    pygame.draw.circle(surface, theme.COLOR_NODE_BG, (px, py), theme.NODE_BG_RADIUS)
    pygame.draw.circle(surface, border_color, (px, py), theme.NODE_BG_RADIUS, 1)


# ---------------------------------------------------------------------------
# Public layout and color helpers
# ---------------------------------------------------------------------------


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


def draw_background(surface: pygame.Surface) -> None:
    """Draw the background: vertical gradient + subtle radial glow in the graph centre.

    The gradient goes from BG_COLOR (top) to BG_COLOR_BOTTOM (bottom).
    A faint blue-purple glow is drawn over the graph area centre.

    Args:
        surface: Target surface (full window).
    """
    w = surface.get_width()
    h = surface.get_height()
    r0, g0, b0 = theme.BG_COLOR
    r1, g1, b1 = theme.BG_COLOR_BOTTOM

    # Vertical gradient — draw one horizontal scanline per row.
    for y in range(h):
        t = y / max(h - 1, 1)
        r = int(r0 + (r1 - r0) * t)
        g = int(g0 + (g1 - g0) * t)
        b = int(b0 + (b1 - b0) * t)
        pygame.draw.line(surface, (r, g, b), (0, y), (w, y))

    # Radial glow centred on the graph zone — very subtle, alpha ~25
    glow_cx = theme.LEFT_PANEL_WIDTH + (w - theme.LEFT_PANEL_WIDTH - theme.RIGHT_PANEL_WIDTH) // 2
    glow_cy = h // 2
    glow_r = min(w, h) // 2
    glow_surf = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    rg, gg, bg_ = theme.BG_GLOW_COLOR
    for step, alpha in [(glow_r, 25), (glow_r * 2 // 3, 15), (glow_r // 3, 8)]:
        pygame.draw.circle(glow_surf, (rg, gg, bg_, alpha), (glow_r, glow_r), step)
    surface.blit(glow_surf, (glow_cx - glow_r, glow_cy - glow_r))


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


# ---------------------------------------------------------------------------
# Public draw functions (called by renderer in back-to-front order)
# ---------------------------------------------------------------------------


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
      - WINDOWS (workstation): monitor with dark screen + stand + base
      - LINUX (server):        two stacked rack units with LED dots
      - NETWORK_DEVICE:        circle with antenna lines and signal rays

    Fogged nodes are drawn as gray cloud blobs.
    Each visible node has a dark circular background with a colored border,
    and its ID number rendered centered on the node.

    Args:
        surface: Target surface.
        network: The network (provides Node objects for coloring).
        positions: Pixel positions per node.
        fogged: Set of node IDs that are unknown.
    """
    font = _get_node_id_font()

    for node_id, node in network.nodes.items():
        if node_id not in positions:
            continue
        px, py = positions[node_id]
        color = get_node_color(node)

        # --- Fogged node: cloud blob ---
        if node_id in fogged:
            _draw_fog_cloud(surface, px, py)
            continue

        # --- Compromised glow (behind the node background) ---
        if node.session_level in (SessionLevel.USER, SessionLevel.ROOT):
            _draw_compromised_glow(surface, px, py, node.session_level)

        # --- Dark circular background with colored border ---
        _draw_node_bg(surface, px, py, color)

        # --- Icon depending on OS type ---
        if node.os_type == OsType.WINDOWS:
            # Modern flat monitor: thin bezel + wide screen + slim neck + wide foot
            body_w, body_h = 24, 15
            bx = px - body_w // 2
            by = py - body_h // 2 - 3  # shift up to leave room for neck+foot
            # Thin rounded bezel (1px border rect)
            pygame.draw.rect(surface, color, pygame.Rect(bx, by, body_w, body_h), border_radius=2)
            # Screen — fills almost entire bezel (1px margin)
            screen_rect = pygame.Rect(bx + 1, by + 1, body_w - 2, body_h - 2)
            pygame.draw.rect(surface, theme.COLOR_ICON_SCREEN, screen_rect, border_radius=1)
            # Slim power/status dot (top-right corner of bezel)
            pygame.draw.circle(surface, theme.COLOR_ICON_LED, (bx + body_w - 3, by + 3), 1)
            # Slim neck (center)
            pygame.draw.rect(surface, color, pygame.Rect(px - 1, by + body_h, 3, 4))
            # Wide flat foot (wider than neck, slim)
            pygame.draw.rect(surface, color, pygame.Rect(px - 8, by + body_h + 4, 16, 2), border_radius=1)

        elif node.os_type == OsType.LINUX:
            # Modern server rack: 3 slim units stacked, with port indicators
            rack_w, rack_h, gap = 22, 6, 2
            for i in range(3):
                uy = py - (rack_h + gap) * 1 + i * (rack_h + gap) - rack_h // 2
                # Unit body
                pygame.draw.rect(surface, color,
                                 pygame.Rect(px - rack_w // 2, uy, rack_w, rack_h), border_radius=1)
                # Dark ventilation area (left 60%)
                pygame.draw.rect(surface, theme.COLOR_ICON_SCREEN,
                                 pygame.Rect(px - rack_w // 2 + 1, uy + 1, rack_w * 3 // 5, rack_h - 2))
                # Status LED (right side, alternate green/dim)
                led_color = theme.COLOR_ICON_LED if i < 2 else (60, 60, 60)
                pygame.draw.circle(surface, led_color,
                                   (px + rack_w // 2 - 3, uy + rack_h // 2), 2)

        else:
            # NETWORK_DEVICE: shield (firewall) shape with lock symbol
            shield_pts = [
                (px, py - 13),       # top center
                (px + 11, py - 7),   # top right
                (px + 11, py + 2),   # mid right
                (px, py + 13),       # bottom tip
                (px - 11, py + 2),   # mid left
                (px - 11, py - 7),   # top left
            ]
            pygame.draw.polygon(surface, color, shield_pts)
            # Dark inner shield
            inner_pts = [
                (px, py - 10),
                (px + 8, py - 5),
                (px + 8, py + 2),
                (px, py + 9),
                (px - 8, py + 2),
                (px - 8, py - 5),
            ]
            pygame.draw.polygon(surface, theme.COLOR_ICON_SCREEN, inner_pts)
            # Lock body (small rounded rect)
            pygame.draw.rect(surface, color, pygame.Rect(px - 4, py - 1, 8, 6), border_radius=1)
            # Lock shackle (U arc above)
            pygame.draw.arc(surface, color,
                            pygame.Rect(px - 3, py - 6, 6, 7), 0, math.pi, 2)

        # --- Node ID number centered on the node ---
        if font is not None:
            id_surf = font.render(str(node_id), True, (255, 255, 255))
            id_x = px - id_surf.get_width() // 2
            id_y = py + 8
            surface.blit(id_surf, (id_x, id_y))


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
    """Draw a pulsing glow on ROOT-session nodes.

    Draws 2-3 concentric rings with decreasing alpha plus a soft filled
    background circle for a diffuse glow effect.

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

        # Soft filled glow behind (very low alpha)
        glow_r = pulse_r + 6
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*theme.COLOR_NODE_ROOT, 30), (glow_r + 1, glow_r + 1), glow_r)
        surface.blit(glow_surf, (pos[0] - glow_r - 1, pos[1] - glow_r - 1))

        # 2 concentric rings — inner brighter, outer dimmer
        for ring_offset, alpha in [(0, 180), (6, 90)]:
            r = pulse_r + ring_offset
            ring_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(ring_surf, (*theme.COLOR_NODE_ROOT, alpha), (r + 2, r + 2), r, 2)
            surface.blit(ring_surf, (pos[0] - r - 2, pos[1] - r - 2))


def draw_flash_effect(
    surface: pygame.Surface,
    flash_events: list[tuple[int, float]],
    positions: NodePositions,
) -> None:
    """Draw expanding, fading flash circles for recent actions.

    Each flash event is a (node_id, time_remaining) tuple.
    time_remaining goes from FLASH_DURATION to 0 as the flash fades.
    Two concentric rings are drawn: inner brighter, outer dimmer.

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

        # Inner ring (brighter)
        flash_surf = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(flash_surf, (255, 200, 50, alpha), (r + 1, r + 1), r, 2)
        surface.blit(flash_surf, (pos[0] - r - 1, pos[1] - r - 1))

        # Outer ring (dimmer, expands faster)
        r2 = r + 8
        alpha2 = max(0, alpha - 80)
        if alpha2 > 0:
            outer_surf = pygame.Surface((r2 * 2 + 2, r2 * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(outer_surf, (255, 160, 30, alpha2), (r2 + 1, r2 + 1), r2, 1)
            surface.blit(outer_surf, (pos[0] - r2 - 1, pos[1] - r2 - 1))


def draw_special_markers(
    surface: pygame.Surface,
    entry_node_id: int,
    target_node_id: int,
    positions: NodePositions,
    anim_time: float = 0.0,
) -> None:
    """Draw rings around the entry (green) and target/loot (gold) nodes.

    The target node gets a pulsing gold glow and a "TARGET" label so
    the user can immediately identify the exfiltration objective.

    Args:
        surface: Target surface.
        entry_node_id: The DMZ/entry node (Red Team start).
        target_node_id: The data center / exfiltration target.
        positions: Pixel positions per node.
        anim_time: Elapsed animation time (seconds) for the pulsing effect.
    """
    if entry_node_id in positions:
        pygame.draw.circle(
            surface,
            theme.COLOR_ENTRY_MARKER,
            positions[entry_node_id],
            theme.SPECIAL_MARKER_RADIUS,
            theme.MARKER_RING_WIDTH,
        )
    # Target marker: pulsing gold glow + ring + label
    if target_node_id in positions:
        pos = positions[target_node_id]
        base_r = theme.SPECIAL_MARKER_RADIUS + (
            theme.MARKER_RING_WIDTH + 1 if target_node_id == entry_node_id else 0
        )

        # Pulsing gold glow (similar to ROOT pulse but gold)
        t = math.sin(anim_time * theme.PULSE_SPEED * 0.7)  # slightly slower than ROOT
        glow_r = int(base_r + 4 + (t + 1) / 2 * 8)
        glow_surf = pygame.Surface((glow_r * 2 + 2, glow_r * 2 + 2), pygame.SRCALPHA)
        pygame.draw.circle(
            glow_surf, (*theme.COLOR_TARGET_MARKER, 35),
            (glow_r + 1, glow_r + 1), glow_r,
        )
        surface.blit(glow_surf, (pos[0] - glow_r - 1, pos[1] - glow_r - 1))

        # Solid gold ring
        pygame.draw.circle(surface, theme.COLOR_TARGET_MARKER, pos, base_r, theme.MARKER_RING_WIDTH)

        # "TARGET" label above the node
        font = _get_node_id_font()
        if font is not None:
            label = font.render("TARGET", True, theme.COLOR_TARGET_MARKER)
            lx = pos[0] - label.get_width() // 2
            ly = pos[1] - theme.NODE_BG_RADIUS - 20
            surface.blit(label, (lx, ly))


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
        # Center label horizontally below the node background circle
        lx = px - label.get_width() // 2
        ly = py + theme.NODE_BG_RADIUS + 6
        surface.blit(label, (lx, ly))


def draw_surveillance_shields(
    surface: pygame.Surface,
    network: Network,
    positions: NodePositions,
    fogged: set[int],
) -> None:
    """Draw a small blue shield icon on nodes currently under Blue Team surveillance.

    The shield is drawn at the top-right of the node background circle so it
    does not overlap with the node icon or the agent halo.

    Args:
        surface: Target surface.
        network: The network (to read is_under_surveillance per node).
        positions: Pixel positions per node.
        fogged: Nodes currently unknown — no shield shown for fogged nodes.
    """
    SHIELD_COLOR = (30, 160, 255)       # bright blue
    SHIELD_BORDER = (180, 230, 255)     # light blue outline

    for node_id, node in network.nodes.items():
        if not node.is_under_surveillance:
            continue
        if node_id not in positions or node_id in fogged:
            continue

        px, py = positions[node_id]
        # Place the shield at top-right corner of the node circle
        cx = px + theme.NODE_BG_RADIUS - 2
        cy = py - theme.NODE_BG_RADIUS + 2

        # Shield polygon (small, ~10px tall)
        half_w = 5
        pts = [
            (cx, cy - 5),       # top center
            (cx + half_w, cy - 2),
            (cx + half_w, cy + 2),
            (cx, cy + 5),       # bottom tip
            (cx - half_w, cy + 2),
            (cx - half_w, cy - 2),
        ]
        shield_surf = pygame.Surface((16, 16), pygame.SRCALPHA)
        # Offset all points so (cx, cy) maps to center of the mini surface
        local_pts = [(x - cx + 8, y - cy + 8) for x, y in pts]
        pygame.draw.polygon(shield_surf, SHIELD_COLOR, local_pts)
        pygame.draw.polygon(shield_surf, SHIELD_BORDER, local_pts, 1)
        # Small lock dot in center
        pygame.draw.circle(shield_surf, SHIELD_BORDER, (8, 8), 1)
        surface.blit(shield_surf, (cx - 8, cy - 8))


def find_node_at(
    positions: NodePositions,
    click_pos: tuple[int, int],
    radius: int | None = None,
) -> int | None:
    """Return the ID of the node closest to click_pos within radius, or None.

    Args:
        positions: Pixel positions per node (already offset-adjusted by caller).
        click_pos: Mouse click pixel coordinates.
        radius: Hit detection radius in pixels. Defaults to NODE_BG_RADIUS + 4.

    Returns:
        Node ID if a node is within radius of the click, else None.
    """
    hit_r = (radius if radius is not None else theme.NODE_BG_RADIUS + 4) ** 2
    cx, cy = click_pos
    for nid, (px, py) in positions.items():
        if (cx - px) ** 2 + (cy - py) ** 2 <= hit_r:
            return nid
    return None


def draw_selected_ring(
    surface: pygame.Surface,
    node_id: int,
    positions: NodePositions,
) -> None:
    """Draw a bright selection ring around the selected node.

    Args:
        surface: Target surface.
        node_id: The selected node's ID.
        positions: Pixel positions per node.
    """
    if node_id not in positions:
        return
    px, py = positions[node_id]
    r = theme.NODE_BG_RADIUS + 5
    sel_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(sel_surf, (255, 255, 255, 200), (r + 2, r + 2), r, 2)
    surface.blit(sel_surf, (px - r - 2, py - r - 2))


def draw_node_info_panel(
    surface: pygame.Surface,
    node: Node,
    node_id: int,
    px: int,
    py: int,
    font_header: pygame.font.Font,
    font_body: pygame.font.Font,
    surface_width: int,
) -> None:
    """Draw a floating info panel near the selected node.

    Placed to the right of the node by default; flips left if too close to the
    right edge.

    Args:
        surface: Target surface.
        node: The Node dataclass.
        node_id: Node identifier.
        px, py: Pixel coordinates of the node centre.
        font_header: Font for the panel title.
        font_body: Font for detail lines.
        surface_width: Full surface width (to flip panel side if needed).
    """
    # Build content lines
    os_names = {OsType.WINDOWS: "PC / Workstation", OsType.LINUX: "Server / Linux",
                OsType.NETWORK_DEVICE: "Firewall / Router"}
    session_names = {
        SessionLevel.NONE: ("NONE", (160, 160, 160)),
        SessionLevel.USER: ("USER", (220, 140, 0)),
        SessionLevel.ROOT: ("ROOT", (255, 34, 34)),
    }
    disc_names = {
        DiscoveryLevel.UNKNOWN: ("Unknown", (100, 100, 100)),
        DiscoveryLevel.DISCOVERED: ("Discovered", (220, 220, 0)),
        DiscoveryLevel.ENUMERATED: ("Enumerated", (0, 204, 204)),
    }
    sess_label, sess_color = session_names[node.session_level]
    disc_label, disc_color = disc_names[node.discovery_level]

    lines: list[tuple[str, tuple[int, int, int]]] = [
        (os_names.get(node.os_type, "Unknown"), (200, 200, 200)),
        (f"Session: {sess_label}", sess_color),
        (f"Discovery: {disc_label}", disc_color),
        (f"Suspicion: {node.suspicion_level:.0f}%", (220, 220, 0)),
        ("", (0, 0, 0)),  # spacer
    ]
    if node.vulnerabilities:
        lines.append(("Vulnerabilities:", (180, 100, 255)))
        for vuln in node.vulnerabilities[:4]:
            lines.append((f"  • {vuln}", (200, 160, 255)))
    else:
        lines.append(("No known vulns", (100, 100, 100)))
    lines.append(("", (0, 0, 0)))  # spacer
    flags: list[str] = []
    if node.has_backdoor:
        flags.append("Backdoor")
    if node.has_tunnel:
        flags.append("Tunnel")
    if node.has_loot:
        flags.append("Loot")
    if not node.is_online:
        flags.append("Offline")
    if flags:
        lines.append((", ".join(flags), (255, 140, 0)))

    # Panel geometry
    pad = 8
    line_h = 14
    panel_w = 170
    panel_h = pad * 2 + 20 + line_h * len(lines)

    # Place right or left depending on available space
    if px + theme.NODE_BG_RADIUS + 10 + panel_w < surface_width - theme.RIGHT_PANEL_WIDTH:
        bx = px + theme.NODE_BG_RADIUS + 10
    else:
        bx = px - theme.NODE_BG_RADIUS - 10 - panel_w
    by = py - panel_h // 2

    # Clamp vertically
    by = max(4, min(by, surface.get_height() - panel_h - 4))

    # Background + border
    bg_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    pygame.draw.rect(bg_surf, (10, 12, 30, 220), pygame.Rect(0, 0, panel_w, panel_h), border_radius=4)
    pygame.draw.rect(bg_surf, (0, 160, 140, 200), pygame.Rect(0, 0, panel_w, panel_h), 1, border_radius=4)
    surface.blit(bg_surf, (bx, by))

    # Title
    title_surf = font_header.render(f"Node {node_id}", True, (0, 220, 80))
    surface.blit(title_surf, (bx + pad, by + pad))

    # Detail lines
    iy = by + pad + 20
    for text, color in lines:
        if not text:
            iy += 4
            continue
        txt_surf = font_body.render(text, True, color)
        surface.blit(txt_surf, (bx + pad, iy))
        iy += line_h


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
