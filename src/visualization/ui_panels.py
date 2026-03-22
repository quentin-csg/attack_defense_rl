"""UI panel drawing functions for the Pygame dashboard.

All functions are stateless — they receive a pygame.Surface and data as
arguments, draw onto the surface, and return nothing. The renderer calls
them in order after the graph zone has been drawn.

Panel layout (left sidebar):
  - Stats panel        (top, always visible)
  - Collapsible panels (middle)
  - Suspicion bars     (bottom)

Right panel:
  - Action log
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pygame

from src.visualization import theme

if TYPE_CHECKING:
    from src.visualization.render_state import LogEntry, RenderState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_panel_box(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    font: pygame.font.Font,
) -> None:
    """Draw a dark box with a teal border and green title."""
    # Semi-transparent background via a temporary surface
    bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg.fill(theme.COLOR_PANEL_BG)
    surface.blit(bg, rect.topleft)
    pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER, rect, 1)
    # Title bar background line
    title_surf = font.render(title, True, theme.COLOR_PANEL_HEADER)
    surface.blit(title_surf, (rect.left + 6, rect.top + 4))


def _suspicion_color(level: float) -> tuple[int, int, int]:
    """Return bar color based on suspicion threshold."""
    if level < 30:
        return theme.COLOR_SUSPICION_LOW
    if level < 60:
        return theme.COLOR_SUSPICION_MED
    if level < 80:
        return theme.COLOR_SUSPICION_HIGH
    return theme.COLOR_SUSPICION_CRIT


def _log_color(color_key: str) -> tuple[int, int, int]:
    """Return text color for an action log entry."""
    return {
        "red_success": theme.COLOR_LOG_RED_SUCCESS,
        "red_fail": theme.COLOR_LOG_RED_FAIL,
        "blue_action": theme.COLOR_LOG_BLUE_ACTION,
        "critical": theme.COLOR_LOG_FAILURE,
    }.get(color_key, theme.COLOR_PANEL_BODY)


# ---------------------------------------------------------------------------
# Stats panel (top-left, always visible)
# ---------------------------------------------------------------------------

def draw_stats_panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    state: RenderState,
) -> None:
    """Draw the 4-line stats box (STEP, REWARD, COMPROMISED, SUSPICION).

    Args:
        surface: Target surface.
        rect: Bounding rect for the panel.
        font: Monospace font for stats text.
        state: Current render state.
    """
    _draw_panel_box(surface, rect, "", font)

    lines = [
        ("STEP", f"{state.step}"),
        ("REWARD", f"{state.episode_reward:+.1f}"),
        ("NODES", f"{state.n_compromised}/{state.total_nodes}"),
        ("SUSPICION", f"{state.max_suspicion:.0f}%"),
    ]

    x = rect.left + 8
    y = rect.top + 6
    for key, value in lines:
        key_surf = font.render(f"{key}: ", True, theme.COLOR_TEXT_KEY)
        val_surf = font.render(value, True, theme.COLOR_TEXT_VALUE)
        surface.blit(key_surf, (x, y))
        surface.blit(val_surf, (x + key_surf.get_width(), y))
        y += theme.STATS_LINE_HEIGHT


# ---------------------------------------------------------------------------
# Collapsible sidebar panels
# ---------------------------------------------------------------------------

_PANEL_NAMES: list[str] = ["Scan", "Nodes", "Attacker", "Stats", "Haze"]


def get_panel_header_rects(
    sidebar_rect: pygame.Rect,
    font: pygame.font.Font,
) -> dict[str, pygame.Rect]:
    """Return clickable rects for each panel header.

    Used by the renderer for mouse-click detection.

    Args:
        sidebar_rect: Bounding rect for the entire sidebar zone.
        font: Font (unused, kept for API consistency).

    Returns:
        dict mapping panel name → pygame.Rect of its header.
    """
    rects: dict[str, pygame.Rect] = {}
    y = sidebar_rect.top
    for name in _PANEL_NAMES:
        rects[name] = pygame.Rect(sidebar_rect.left, y, sidebar_rect.width, theme.PANEL_HEADER_HEIGHT)
        y += theme.PANEL_HEADER_HEIGHT
    return rects


def draw_sidebar_panels(
    surface: pygame.Surface,
    sidebar_rect: pygame.Rect,
    fonts: dict[str, pygame.font.Font],
    state: RenderState,
    panel_expanded: dict[str, bool],
) -> None:
    """Draw the 5 collapsible panels in the left sidebar.

    Args:
        surface: Target surface.
        sidebar_rect: Bounding rect for the sidebar zone.
        fonts: dict with keys "header" and "body".
        state: Current render state.
        panel_expanded: dict mapping panel name → bool (expanded/collapsed).
    """
    font_h = fonts["header"]
    font_b = fonts["body"]

    y = sidebar_rect.top
    for name in _PANEL_NAMES:
        header_rect = pygame.Rect(sidebar_rect.left, y, sidebar_rect.width, theme.PANEL_HEADER_HEIGHT)
        expanded = panel_expanded.get(name, False)

        # Header background
        header_bg = pygame.Surface((header_rect.width, header_rect.height), pygame.SRCALPHA)
        header_bg.fill((20, 40, 30, 180))
        surface.blit(header_bg, header_rect.topleft)
        pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER, header_rect, 1)

        arrow = "v" if expanded else ">"
        label_surf = font_h.render(f" {arrow} {name}", True, theme.COLOR_PANEL_HEADER)
        surface.blit(label_surf, (header_rect.left + 4, header_rect.top + 5))
        y += theme.PANEL_HEADER_HEIGHT

        if not expanded:
            continue

        # Content area (max 80px per panel to avoid overflow)
        content_height = min(80, sidebar_rect.bottom - y)
        if content_height <= 0:
            continue
        content_rect = pygame.Rect(sidebar_rect.left, y, sidebar_rect.width, content_height)
        content_bg = pygame.Surface((content_rect.width, content_rect.height), pygame.SRCALPHA)
        content_bg.fill((10, 20, 25, 160))
        surface.blit(content_bg, content_rect.topleft)
        pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER, content_rect, 1)

        lines = _get_panel_lines(name, state)
        cy = content_rect.top + 4
        for line in lines:
            if cy + 14 > content_rect.bottom:
                break
            txt = font_b.render(line, True, theme.COLOR_PANEL_BODY)
            surface.blit(txt, (content_rect.left + 6, cy))
            cy += 14

        y += content_height


def _get_panel_lines(name: str, state: RenderState) -> list[str]:
    """Return text lines for a given panel."""
    if name == "Scan":
        scan_entries = [e for e in state.action_log if "SCAN" in e.text]
        if scan_entries:
            last = scan_entries[-1]
            return [last.text[last.text.find("]") + 2:]]  # strip step prefix
        return ["No scan performed yet"]

    if name == "Nodes":
        lines = []
        for nid, node in state.network.nodes.items():
            from src.environment.node import DiscoveryLevel, SessionLevel
            if node.discovery_level == DiscoveryLevel.UNKNOWN:
                continue
            sess = node.session_level.name if node.session_level != SessionLevel.NONE else "-"
            lines.append(f"N{nid}: {node.discovery_level.name[:4]} / {sess[:4]}")
        return lines[:5] or ["No nodes discovered"]

    if name == "Attacker":
        from src.environment.node import SessionLevel
        node = state.network.nodes.get(state.agent_position)
        sess = node.session_level.name if node else "?"
        bd = sum(1 for n in state.network.nodes.values() if n.has_backdoor)
        tn = sum(1 for n in state.network.nodes.values() if n.has_tunnel)
        return [
            f"Pos: Node {state.agent_position}",
            f"Session: {sess}",
            f"Backdoors: {bd}  Tunnels: {tn}",
        ]

    if name == "Stats":
        return [
            f"Step: {state.step}",
            f"Reward: {state.episode_reward:+.1f}",
            f"Comp: {state.n_compromised}/{state.total_nodes}",
            f"Disc: {state.n_discovered}/{state.total_nodes}",
        ]

    if name == "Haze":
        fog = state.fog_percentage
        bar_w = max(0, min(40, int(fog / 100 * 40)))
        bar = "#" * bar_w + "." * (40 - bar_w)
        return [
            f"Fog: {fog:.0f}%",
            f"[{bar[:20]}]",
        ]

    return []


# ---------------------------------------------------------------------------
# Suspicion bars (bottom-left)
# ---------------------------------------------------------------------------

def draw_suspicion_bars(
    surface: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    per_node_suspicion: dict[int, float],
) -> None:
    """Draw a vertical bar chart of per-node suspicion levels.

    Args:
        surface: Target surface.
        rect: Bounding rect for the suspicion zone.
        font: Small monospace font for node ID labels.
        per_node_suspicion: Mapping node_id → suspicion level (0-100).
    """
    # Panel background
    bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg.fill(theme.COLOR_PANEL_BG)
    surface.blit(bg, rect.topleft)
    pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER, rect, 1)

    # Title
    title_font_surf = font.render("Suspicion Levels", True, theme.COLOR_PANEL_HEADER)
    surface.blit(title_font_surf, (rect.left + 4, rect.top + 4))

    if not per_node_suspicion:
        return

    node_ids = sorted(per_node_suspicion.keys())
    n = len(node_ids)
    if n == 0:
        return

    chart_top = rect.top + 22
    chart_bottom = rect.bottom - 16  # leave space for labels
    chart_height = max(1, chart_bottom - chart_top)

    bar_area_w = rect.width - 8
    bar_w = max(2, bar_area_w // n - 1)

    for i, nid in enumerate(node_ids):
        level = per_node_suspicion[nid]
        bar_h = int(level / 100.0 * chart_height)
        bar_h = max(1, bar_h)
        bx = rect.left + 4 + i * (bar_w + 1)
        by = chart_bottom - bar_h
        color = _suspicion_color(level)
        pygame.draw.rect(surface, color, pygame.Rect(bx, by, bar_w, bar_h))

        # Node ID label below bar
        label = font.render(str(nid), True, theme.COLOR_TEXT_LABEL)
        lx = bx + bar_w // 2 - label.get_width() // 2
        surface.blit(label, (lx, chart_bottom + 2))


# ---------------------------------------------------------------------------
# Action log (right panel)
# ---------------------------------------------------------------------------

def draw_action_log(
    surface: pygame.Surface,
    rect: pygame.Rect,
    font: pygame.font.Font,
    action_log: list[LogEntry],
) -> None:
    """Draw the scrollable action log in the right panel.

    Shows the most recent entries that fit within the rect height.
    Most recent entry is at the bottom.

    Args:
        surface: Target surface.
        rect: Bounding rect for the action log panel.
        font: Small monospace font.
        action_log: List of LogEntry objects (oldest first).
    """
    # Panel background
    bg = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    bg.fill(theme.COLOR_PANEL_BG)
    surface.blit(bg, rect.topleft)
    pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER, rect, 1)

    # Title
    title = font.render("Action Log", True, theme.COLOR_PANEL_HEADER)
    surface.blit(title, (rect.left + 6, rect.top + 4))

    line_h = 13
    usable_top = rect.top + 20
    usable_height = rect.height - 24
    max_lines = usable_height // line_h

    # Take the last N entries that fit
    visible = action_log[-max_lines:] if len(action_log) > max_lines else action_log
    # Draw from top
    y = usable_top
    for entry in visible:
        if y + line_h > rect.bottom - 4:
            break
        color = _log_color(entry.color_key)
        # Word-wrap: clip text to panel width
        max_chars = max(1, (rect.width - 12) // max(1, font.size("A")[0]))
        text = entry.text[:max_chars]
        txt_surf = font.render(text, True, color)
        surface.blit(txt_surf, (rect.left + 6, y))
        y += line_h
