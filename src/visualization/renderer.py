"""Pygame window manager for the Attack & Defense RL visualization.

PygameRenderer owns the Pygame window lifecycle and orchestrates the
drawing calls. Created lazily by CyberEnv on the first render() call.
"""

from __future__ import annotations

import copy
import os
from typing import TYPE_CHECKING

import numpy as np
import pygame

from src.visualization import theme
from src.visualization.controls import DashboardControls, handle_key_event
from src.visualization.graph_view import (
    NodePositions,
    compute_layout,
    draw_agent_halo,
    draw_attacker_path,
    draw_background,
    draw_edges,
    draw_flash_effect,
    draw_node_icons,
    draw_node_info_panel,
    draw_node_labels,
    draw_pulse_effect,
    draw_selected_ring,
    draw_special_markers,
    draw_surveillance_shields,
    find_node_at,
    fog_node_ids,
)
from src.visualization.ui_panels import (
    _PANEL_NAMES,
    draw_action_log,
    draw_sidebar_panels,
    draw_stats_panel,
    draw_suspicion_bars,
    get_panel_header_rects,
)

if TYPE_CHECKING:
    from src.visualization.render_state import RenderState


def _load_font(size: int) -> pygame.font.Font:
    """Load a monospace font with a fallback chain."""
    for name in theme.FONT_CANDIDATES:
        if pygame.font.match_font(name) is not None:
            return pygame.font.SysFont(name, size)
    return pygame.font.Font(None, size)


class PygameRenderer:
    """Manages the Pygame window and renders the network state each step."""

    def __init__(self, headless: bool = False) -> None:
        """Initialize Pygame and create the display surface.

        Args:
            headless: If True, use the dummy SDL video driver (tests / CI).
        """
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

        pygame.init()
        pygame.display.set_caption(theme.WINDOW_TITLE)
        self._screen: pygame.Surface = pygame.display.set_mode((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT))
        self._clock: pygame.time.Clock = pygame.time.Clock()

        self._font_stats: pygame.font.Font = _load_font(theme.FONT_SIZE_STATS)
        self._font_label: pygame.font.Font = _load_font(theme.FONT_SIZE_LABEL)
        self._font_panel_header: pygame.font.Font = _load_font(theme.FONT_SIZE_PANEL_HEADER)
        self._font_panel_body: pygame.font.Font = _load_font(theme.FONT_SIZE_PANEL_BODY)
        self._font_log: pygame.font.Font = _load_font(theme.FONT_SIZE_LOG)

        self._layout: NodePositions | None = None
        self._open: bool = True
        self._anim_time: float = 0.0
        self._flash_events: list[tuple[int, float]] = []
        self._panel_expanded: dict[str, bool] = {name: False for name in _PANEL_NAMES}
        self._panel_header_rects: dict[str, pygame.Rect] = {}
        self._controls: DashboardControls = DashboardControls()

        self._drag_offset: list[int] = [0, 0]
        self._drag_active: bool = False
        self._drag_last_pos: tuple[int, int] | None = None

        self._zoom_scale: float = 1.0
        self._graph_cx: int = (theme.LEFT_PANEL_WIDTH + theme.WINDOW_WIDTH - theme.RIGHT_PANEL_WIDTH) // 2
        self._graph_cy: int = theme.WINDOW_HEIGHT // 2

        self._node_drag_id: int | None = None
        self._node_drag_moved: bool = False
        self._node_drag_start: tuple[int, int] = (0, 0)

        self._log_scroll_offset: int = 0

        # _replay_idx == -1 means live; >= 0 means browsing history.
        self._state_history: list[RenderState] = []
        self._replay_idx: int = -1

        self._terminal_message: str = ""
        self._terminal_color: tuple[int, int, int] = (255, 255, 255)

        self._selected_node: int | None = None
        self._last_state: RenderState | None = None

        self._search_active: bool = False
        self._search_buffer: str = ""

        self._held_key: int | None = None
        self._hold_frames: int = 0

    @property
    def is_open(self) -> bool:
        """Whether the window is still open."""
        return self._open

    @property
    def controls(self) -> DashboardControls:
        """Dashboard controls (pause, speed multiplier)."""
        return self._controls

    def update(self, state: RenderState) -> None:
        """Redraw the frame and flip the display. Non-blocking."""
        self._handle_events()
        if not self._open:
            return
        render_state = self._state_history[self._replay_idx] if self.in_replay else state
        self._ensure_layout(render_state.network)
        self._tick_animation()
        self._maybe_add_flash(render_state)
        self._render_frame(self._screen, render_state)
        pygame.display.flip()
        self._clock.tick(theme.FPS)

    def get_frame(self, state: RenderState) -> np.ndarray:
        """Render to an offscreen surface and return as an RGB numpy array (H, W, 3)."""
        self._ensure_layout(state.network)
        self._tick_animation()
        self._maybe_add_flash(state)
        offscreen = pygame.Surface((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT))
        self._render_frame(offscreen, state)
        return np.transpose(pygame.surfarray.array3d(offscreen), (1, 0, 2))

    def reset_layout(self) -> None:
        """Invalidate the cached layout so it is recomputed on the next render."""
        self._layout = None

    def reset_history(self) -> None:
        """Clear the step-replay history."""
        self._state_history.clear()
        self._replay_idx = -1
        self._terminal_message = ""

    def set_terminal_message(self, message: str, color: tuple[int, int, int]) -> None:
        """Display a centered end-of-episode overlay on the next render."""
        self._terminal_message = message
        self._terminal_color = color

    def record_current_step(self) -> None:
        """Record the most recently rendered state for step-replay.

        Call exactly once per real env.step(), after env.render().
        Deep-copies the state so node attributes are frozen at this step.
        """
        if not self.in_replay and self._last_state is not None:
            self._state_history.append(copy.deepcopy(self._last_state))

    @property
    def in_replay(self) -> bool:
        """True when the user is browsing historical steps."""
        return self._replay_idx >= 0

    def close(self) -> None:
        """Quit Pygame and release resources."""
        if pygame.get_init():
            pygame.quit()
        self._open = False

    def _ensure_layout(self, network) -> None:
        if self._layout is None:
            graph_area = pygame.Rect(
                theme.GRAPH_AREA_LEFT + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_TOP + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_WIDTH - 2 * theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_HEIGHT - 2 * theme.GRAPH_MARGIN,
            )
            self._layout = compute_layout(network.graph, graph_area)

    def _tick_animation(self) -> None:
        dt = 1.0 / theme.FPS
        self._anim_time += dt
        self._flash_events = [(nid, t - dt) for nid, t in self._flash_events if t - dt > 0]

    def _maybe_add_flash(self, state: RenderState) -> None:
        if (
            state.last_action_success
            and state.last_action_target is not None
            and state.last_action_target in state.network.nodes
        ):
            flashing_ids = {nid for nid, _ in self._flash_events}
            if state.last_action_target not in flashing_ids:
                self._flash_events.append((state.last_action_target, theme.FLASH_DURATION))

    def _effective_layout(self) -> NodePositions:
        """Return layout positions with zoom + pan applied."""
        if self._layout is None:
            return {}
        ox, oy = self._drag_offset
        scale = self._zoom_scale
        cx, cy = self._graph_cx, self._graph_cy
        return {
            nid: (int(cx + (px - cx) * scale + ox), int(cy + (py - cy) * scale + oy))
            for nid, (px, py) in self._layout.items()
        }

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._open = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_left_click(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                if self._node_drag_id is not None and not self._node_drag_moved:
                    hit = self._node_drag_id
                    self._selected_node = None if self._selected_node == hit else hit
                self._node_drag_id = None
                self._node_drag_moved = False
                self._drag_active = False
                self._drag_last_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if self._node_drag_id is not None and self._layout is not None:
                    sx, sy = self._node_drag_start
                    if not self._node_drag_moved:
                        if abs(event.pos[0] - sx) + abs(event.pos[1] - sy) > 4:
                            self._node_drag_moved = True
                    if self._node_drag_moved:
                        ox, oy = self._drag_offset
                        scale = self._zoom_scale
                        cx, cy = self._graph_cx, self._graph_cy
                        self._layout[self._node_drag_id] = (
                            cx + (event.pos[0] - cx - ox) / scale,
                            cy + (event.pos[1] - cy - oy) / scale,
                        )
                elif self._drag_active and self._drag_last_pos is not None:
                    dx = event.pos[0] - self._drag_last_pos[0]
                    dy = event.pos[1] - self._drag_last_pos[1]
                    self._drag_offset[0] += dx
                    self._drag_offset[1] += dy
                    self._drag_last_pos = event.pos
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx >= theme.WINDOW_WIDTH - theme.RIGHT_PANEL_WIDTH:
                    self._log_scroll_offset = max(0, self._log_scroll_offset + event.y * 3)
                elif mx > theme.LEFT_PANEL_WIDTH:
                    old_scale = self._zoom_scale
                    factor = 1.1 if event.y > 0 else (1 / 1.1)
                    self._zoom_scale = max(0.2, min(4.0, old_scale * factor))
                    ratio = self._zoom_scale / old_scale
                    ox, oy = self._drag_offset
                    cx, cy = self._graph_cx, self._graph_cy
                    self._drag_offset[0] = int((mx - cx) * (1 - ratio) + ox * ratio)
                    self._drag_offset[1] = int((my - cy) * (1 - ratio) + oy * ratio)
            elif event.type == pygame.TEXTINPUT:
                if self._search_active and event.text.isdigit() and len(self._search_buffer) < 5:
                    self._search_buffer += event.text
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.KEYUP:
                if event.key == self._held_key:
                    self._held_key = None
                    self._hold_frames = 0
        self._tick_held_key()

    def _handle_left_click(self, pos: tuple[int, int]) -> None:
        for name, rect in self._panel_header_rects.items():
            if rect.collidepoint(pos):
                self._panel_expanded[name] = not self._panel_expanded[name]
                return
        eff = self._effective_layout()
        hit = find_node_at(eff, pos)
        if hit is not None:
            self._node_drag_id = hit
            self._node_drag_moved = False
            self._node_drag_start = pos
            return
        if pos[0] > theme.LEFT_PANEL_WIDTH and pos[0] < theme.WINDOW_WIDTH - theme.RIGHT_PANEL_WIDTH:
            self._selected_node = None
            self._drag_active = True
            self._drag_last_pos = pos

    def _handle_key(self, key: int) -> None:
        if self._search_active:
            self._handle_search_key(key)
            return
        if key == pygame.K_ESCAPE:
            self._open = False
        elif key == pygame.K_LEFT:
            self._held_key = pygame.K_LEFT
            self._hold_frames = 0
            if self._state_history:
                self._controls.paused = True
                if self._replay_idx == -1:
                    self._replay_idx = max(0, len(self._state_history) - 2)
                else:
                    self._replay_idx = max(0, self._replay_idx - 1)
        elif key == pygame.K_RIGHT:
            self._held_key = pygame.K_RIGHT
            self._hold_frames = 0
            if self.in_replay:
                next_idx = self._replay_idx + 1
                if next_idx >= len(self._state_history):
                    self._replay_idx = -1
                    self._controls.paused = False
                    self._held_key = None
                else:
                    self._replay_idx = next_idx
        elif key == pygame.K_g:
            if self._state_history:
                self._search_active = True
                self._search_buffer = ""
        elif key == pygame.K_z:
            self._zoom_scale = 1.0
            self._drag_offset = [0, 0]
        else:
            event = pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            handle_key_event(event, self._controls)

    def _handle_search_key(self, key: int) -> None:
        if key == pygame.K_ESCAPE:
            self._search_active = False
            self._search_buffer = ""
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self._search_buffer:
                self._jump_to_step(int(self._search_buffer))
            self._search_active = False
            self._search_buffer = ""
        elif key == pygame.K_BACKSPACE:
            self._search_buffer = self._search_buffer[:-1]

    def _jump_to_step(self, step_num: int) -> None:
        if not self._state_history:
            return
        best_idx = min(range(len(self._state_history)),
                       key=lambda i: abs(self._state_history[i].step - step_num))
        self._replay_idx = best_idx
        self._controls.paused = True

    def _tick_held_key(self) -> None:
        """Auto-advance replay when LEFT/RIGHT is held (progressive acceleration)."""
        if self._held_key is None or not self._state_history:
            return
        self._hold_frames += 1
        if self._hold_frames < 5:
            return
        extra = 1 if self._hold_frames < 20 else (3 if self._hold_frames < 40 else 6)
        for _ in range(extra):
            if self._held_key == pygame.K_LEFT:
                if self._replay_idx == -1:
                    self._controls.paused = True
                    self._replay_idx = max(0, len(self._state_history) - 2)
                else:
                    self._replay_idx = max(0, self._replay_idx - 1)
            elif self._held_key == pygame.K_RIGHT and self.in_replay:
                next_idx = self._replay_idx + 1
                if next_idx >= len(self._state_history):
                    self._replay_idx = -1
                    self._controls.paused = False
                    self._held_key = None
                    break
                else:
                    self._replay_idx = next_idx

    def _render_frame(self, surface: pygame.Surface, state: RenderState) -> None:
        """Draw one complete frame onto the given surface (back-to-front)."""
        assert self._layout is not None  # noqa: S101
        self._last_state = state
        layout = self._effective_layout()

        draw_background(surface)
        fogged = fog_node_ids(state.network)
        # In replay: show fogged nodes as neutral icons rather than cloud blobs.
        icons_fogged = set() if self.in_replay else fogged

        draw_edges(surface, state.network.graph, layout, fogged)
        draw_attacker_path(surface, state.attacker_path, layout)
        draw_node_icons(surface, state.network, layout, icons_fogged)
        draw_surveillance_shields(surface, state.network, layout, fogged)
        draw_pulse_effect(surface, state.network, layout, self._anim_time)
        draw_flash_effect(surface, self._flash_events, layout)
        draw_agent_halo(surface, state.agent_position, layout)
        draw_special_markers(surface, state.network.entry_node_id, state.network.target_node_id,
                             layout, anim_time=self._anim_time)
        draw_node_labels(surface, layout, self._font_label)

        if self._selected_node is not None and self._selected_node in state.network.nodes:
            draw_selected_ring(surface, self._selected_node, layout)
            if self._selected_node in layout:
                px, py = layout[self._selected_node]
                draw_node_info_panel(surface, state.network.nodes[self._selected_node],
                                     self._selected_node, px, py,
                                     self._font_panel_header, self._font_panel_body, surface.get_width())

        draw_stats_panel(surface, pygame.Rect(*theme.STATS_PANEL_RECT), self._font_stats, state)
        fonts = {"header": self._font_panel_header, "body": self._font_panel_body}
        draw_sidebar_panels(surface, pygame.Rect(*theme.SIDEBAR_RECT), fonts, state, self._panel_expanded)
        self._panel_header_rects = get_panel_header_rects(pygame.Rect(*theme.SIDEBAR_RECT), self._font_panel_header)
        draw_suspicion_bars(surface, pygame.Rect(*theme.SUSPICION_RECT), self._font_log, state.per_node_suspicion)
        draw_action_log(surface, pygame.Rect(*theme.ACTION_LOG_RECT), self._font_log,
                        state.action_log, self._log_scroll_offset)

        if self._search_active:
            hint_text = f"Go to step: {self._search_buffer}_    [Enter=jump  Esc=cancel]"
            hint_color = (255, 220, 60)
        elif self.in_replay:
            step_num = self._state_history[self._replay_idx].step
            total = len(self._state_history)
            hint_text = (f"REPLAY  step {step_num}  [{self._replay_idx + 1}/{total}]  "
                         f"← prev  → next/resume  G=jump  ESC=quit")
            hint_color = theme.COLOR_TARGET_MARKER
        else:
            hint_text = "ESC=quit  drag=pan  scroll=zoom  Z=reset  click=info  SPACE=pause  +/-=speed  ←/→=replay  G=jump"
            hint_color = theme.COLOR_HINT
        hint = self._font_log.render(hint_text, True, hint_color)
        surface.blit(hint, (theme.GRAPH_AREA_LEFT + 10, theme.WINDOW_HEIGHT - 16))

        if self._search_active:
            font_search = _load_font(22)
            label = f"Go to step:  {self._search_buffer}\u2588"
            label_surf = font_search.render(label, True, (255, 220, 60))
            sub_surf = self._font_log.render("Enter = jump   Esc = cancel", True, theme.COLOR_HINT)
            box_w = max(label_surf.get_width(), sub_surf.get_width()) + 36
            box_h = label_surf.get_height() + sub_surf.get_height() + 22
            cx = theme.GRAPH_AREA_LEFT + theme.GRAPH_AREA_WIDTH // 2
            cy = theme.GRAPH_AREA_HEIGHT // 2 - 40
            box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            box.fill((10, 12, 40, 230))
            surface.blit(box, (cx - box_w // 2, cy - box_h // 2))
            pygame.draw.rect(surface, theme.COLOR_PANEL_BORDER,
                             pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h), 1)
            surface.blit(label_surf, (cx - label_surf.get_width() // 2, cy - box_h // 2 + 8))
            surface.blit(sub_surf, (cx - sub_surf.get_width() // 2,
                                    cy - box_h // 2 + 8 + label_surf.get_height() + 6))
