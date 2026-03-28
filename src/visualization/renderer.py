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

import copy
import os

# Type-only import to avoid circular at runtime
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

        # Fonts
        self._font_stats: pygame.font.Font = _load_font(theme.FONT_SIZE_STATS)
        self._font_label: pygame.font.Font = _load_font(theme.FONT_SIZE_LABEL)
        self._font_panel_header: pygame.font.Font = _load_font(theme.FONT_SIZE_PANEL_HEADER)
        self._font_panel_body: pygame.font.Font = _load_font(theme.FONT_SIZE_PANEL_BODY)
        self._font_log: pygame.font.Font = _load_font(theme.FONT_SIZE_LOG)

        self._layout: NodePositions | None = None
        self._open: bool = True

        # Animation state
        self._anim_time: float = 0.0
        self._flash_events: list[tuple[int, float]] = []  # (node_id, time_remaining)

        # Panel state (all collapsed by default)
        self._panel_expanded: dict[str, bool] = {name: False for name in _PANEL_NAMES}

        # Pre-computed panel rects (set on first render)
        self._panel_header_rects: dict[str, pygame.Rect] = {}

        # Dashboard controls (pause, speed) — owned by the renderer so that
        # it is the single consumer of the Pygame event queue.
        self._controls: DashboardControls = DashboardControls()

        # Graph pan state
        self._drag_offset: list[int] = [0, 0]       # [dx, dy] offset applied to layout
        self._drag_active: bool = False
        self._drag_last_pos: tuple[int, int] | None = None

        # Graph zoom state (mouse-wheel zoom around cursor position)
        self._zoom_scale: float = 1.0
        # Graph centre in screen space (pivot for zoom); updated in _effective_layout
        self._graph_cx: int = (theme.LEFT_PANEL_WIDTH + theme.WINDOW_WIDTH - theme.RIGHT_PANEL_WIDTH) // 2
        self._graph_cy: int = theme.WINDOW_HEIGHT // 2

        # Node drag state (drag a single node to reposition it)
        self._node_drag_id: int | None = None        # node being dragged, or None
        self._node_drag_moved: bool = False          # True once mouse moved > threshold
        self._node_drag_start: tuple[int, int] = (0, 0)  # initial mouse pos

        # Action log scroll (0 = bottom/latest, positive = scrolled up)
        self._log_scroll_offset: int = 0

        # Step replay history: stores a RenderState snapshot per env step.
        # _replay_idx == -1 means "live" (show the latest state).
        # _replay_idx >= 0 means we're browsing a historical step.
        self._state_history: list[RenderState] = []
        self._replay_idx: int = -1

        # End-of-episode overlay (set via set_terminal_message, cleared on reset_history)
        self._terminal_message: str = ""
        self._terminal_color: tuple[int, int, int] = (255, 255, 255)

        # Selected node (click to inspect)
        self._selected_node: int | None = None
        # Snapshot of the network state for drawing the info panel
        self._last_state: RenderState | None = None

        # Step search bar (press / to activate)
        self._search_active: bool = False
        self._search_buffer: str = ""

        # Held arrow key state for replay acceleration
        # _held_key is K_LEFT or K_RIGHT while the key is physically pressed.
        # _hold_frames counts frames since the key was first pressed; after a
        # threshold the replay auto-advances each frame (with increasing step size).
        self._held_key: int | None = None
        self._hold_frames: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """Whether the window is still open (not closed by the user)."""
        return self._open

    @property
    def controls(self) -> DashboardControls:
        """Dashboard controls (pause, speed multiplier)."""
        return self._controls

    def update(self, state: RenderState) -> None:
        """Redraw the frame and flip the display. Non-blocking.

        Also pumps pygame events so the window stays responsive.
        Stores the state in history for step-replay (LEFT/RIGHT arrows).

        Args:
            state: Full render state snapshot from CyberEnv.
        """
        self._handle_events()
        if not self._open:
            return

        # In replay mode, render from the historical snapshot; otherwise use live state.
        # History recording is done explicitly via record_current_step() — NOT here —
        # so that pause-loop renders and post-episode renders don't inflate the count.
        render_state = self._state_history[self._replay_idx] if self.in_replay else state

        self._ensure_layout(render_state.network)
        self._tick_animation()
        self._maybe_add_flash(render_state)
        self._render_frame(self._screen, render_state)
        pygame.display.flip()
        self._clock.tick(theme.FPS)

    def get_frame(self, state: RenderState) -> np.ndarray:
        """Render to an offscreen surface and return as an RGB numpy array.

        Used for render_mode="rgb_array". Useful for headless tests and
        video recording.

        Args:
            state: Full render state snapshot from CyberEnv.

        Returns:
            np.ndarray of shape (H, W, 3) and dtype uint8.
        """
        self._ensure_layout(state.network)
        self._tick_animation()
        self._maybe_add_flash(state)

        offscreen = pygame.Surface((theme.WINDOW_WIDTH, theme.WINDOW_HEIGHT))
        self._render_frame(offscreen, state)
        # surfarray.array3d returns (W, H, 3); transpose to (H, W, 3).
        raw = pygame.surfarray.array3d(offscreen)
        return np.transpose(raw, (1, 0, 2))

    def reset_layout(self) -> None:
        """Invalidate the cached layout so it is recomputed on the next render.

        Call this when the network topology changes (e.g., after PCG in Phase 5).
        Not needed in Phase 1 (fixed topology), but the hook is here.
        """
        self._layout = None

    def reset_history(self) -> None:
        """Clear the step-replay history (call at the start of each new episode)."""
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
        Pause-loop renders and post-episode renders must NOT call this.
        _last_state is always set by _render_frame(), so this is safe after
        any env.render() call.

        Deep-copies the state so that network node attributes captured at this
        step are not overwritten by later mutations of the live Network object.
        """
        if not self.in_replay and self._last_state is not None:
            self._state_history.append(copy.deepcopy(self._last_state))

    @property
    def in_replay(self) -> bool:
        """True when the user is browsing historical steps (replay mode)."""
        return self._replay_idx >= 0

    def close(self) -> None:
        """Quit Pygame and release resources."""
        if pygame.get_init():
            pygame.quit()
        self._open = False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_layout(self, network) -> None:
        """Compute and cache the node layout if it has not been computed yet."""
        if self._layout is None:
            graph_area = pygame.Rect(
                theme.GRAPH_AREA_LEFT + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_TOP + theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_WIDTH - 2 * theme.GRAPH_MARGIN,
                theme.GRAPH_AREA_HEIGHT - 2 * theme.GRAPH_MARGIN,
            )
            self._layout = compute_layout(network.graph, graph_area)

    def _tick_animation(self) -> None:
        """Advance animation timers by one frame."""
        dt = 1.0 / theme.FPS
        self._anim_time += dt
        self._flash_events = [
            (nid, t - dt) for nid, t in self._flash_events if t - dt > 0
        ]

    def _maybe_add_flash(self, state: RenderState) -> None:
        """Add a flash event when a new action has been registered."""
        if (
            state.last_action_success
            and state.last_action_target is not None
            and state.last_action_target in state.network.nodes
        ):
            # Only add if not already flashing this node
            flashing_ids = {nid for nid, _ in self._flash_events}
            if state.last_action_target not in flashing_ids:
                self._flash_events.append((state.last_action_target, theme.FLASH_DURATION))

    def _effective_layout(self) -> NodePositions:
        """Return layout positions with zoom + pan applied.

        Zoom is centred on the graph area centre (_graph_cx, _graph_cy).
        Pan offset (_drag_offset) is applied in screen space after zoom.
        Formula: screen = center + (layout - center) * scale + offset
        """
        if self._layout is None:
            return {}
        ox, oy = self._drag_offset
        scale = self._zoom_scale
        cx, cy = self._graph_cx, self._graph_cy
        return {
            nid: (
                int(cx + (px - cx) * scale + ox),
                int(cy + (py - cy) * scale + oy),
            )
            for nid, (px, py) in self._layout.items()
        }

    def _handle_events(self) -> None:
        """Process the Pygame event queue."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._open = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_left_click(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                # If node was pressed but not moved → treat as a click (select/deselect)
                if self._node_drag_id is not None and not self._node_drag_moved:
                    hit = self._node_drag_id
                    self._selected_node = None if self._selected_node == hit else hit
                self._node_drag_id = None
                self._node_drag_moved = False
                self._drag_active = False
                self._drag_last_pos = None
            elif event.type == pygame.MOUSEMOTION:
                if self._node_drag_id is not None and self._layout is not None:
                    # Threshold before drag activates (avoids jitter on click)
                    sx, sy = self._node_drag_start
                    if not self._node_drag_moved:
                        if abs(event.pos[0] - sx) + abs(event.pos[1] - sy) > 4:
                            self._node_drag_moved = True
                    if self._node_drag_moved:
                        ox, oy = self._drag_offset
                        scale = self._zoom_scale
                        cx, cy = self._graph_cx, self._graph_cy
                        # Invert zoom+pan to get layout-space position
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
                    # Scroll action log (wheel up = older entries)
                    self._log_scroll_offset = max(0, self._log_scroll_offset + event.y * 3)
                elif mx > theme.LEFT_PANEL_WIDTH:
                    # Zoom graph around the cursor position
                    old_scale = self._zoom_scale
                    factor = 1.1 if event.y > 0 else (1 / 1.1)
                    self._zoom_scale = max(0.2, min(4.0, old_scale * factor))
                    new_scale = self._zoom_scale
                    # Adjust drag offset so the layout point under the cursor stays fixed
                    ratio = new_scale / old_scale
                    ox, oy = self._drag_offset
                    cx, cy = self._graph_cx, self._graph_cy
                    self._drag_offset[0] = int((mx - cx) * (1 - ratio) + ox * ratio)
                    self._drag_offset[1] = int((my - cy) * (1 - ratio) + oy * ratio)
            elif event.type == pygame.TEXTINPUT:
                # Digit input for the step-search bar (layout-independent)
                if self._search_active and event.text.isdigit() and len(self._search_buffer) < 5:
                    self._search_buffer += event.text
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
            elif event.type == pygame.KEYUP:
                if event.key == self._held_key:
                    self._held_key = None
                    self._hold_frames = 0
        # After all events: tick held-key replay acceleration (once per frame)
        self._tick_held_key()

    def _handle_left_click(self, pos: tuple[int, int]) -> None:
        """Handle a left-click: panel header toggle, node select, or pan start."""
        # 1. Check sidebar panel headers (highest priority)
        for name, rect in self._panel_header_rects.items():
            if rect.collidepoint(pos):
                self._panel_expanded[name] = not self._panel_expanded[name]
                return

        # 2. Check node hit (in graph zone only) — start node drag; selection
        #    is committed on MOUSEUP only if the mouse did not move (click vs drag).
        eff = self._effective_layout()
        hit = find_node_at(eff, pos)
        if hit is not None:
            self._node_drag_id = hit
            self._node_drag_moved = False
            self._node_drag_start = pos
            return

        # 3. Click on background → start pan drag (only in graph zone)
        if pos[0] > theme.LEFT_PANEL_WIDTH and pos[0] < theme.WINDOW_WIDTH - theme.RIGHT_PANEL_WIDTH:
            self._selected_node = None  # deselect on background click
            self._drag_active = True
            self._drag_last_pos = pos

    def _handle_key(self, key: int) -> None:
        """Handle keyboard shortcuts (ESC + DashboardControls keys + LEFT/RIGHT replay)."""
        # When the search bar is active, all input is routed there first.
        if self._search_active:
            self._handle_search_key(key)
            return

        if key == pygame.K_ESCAPE:
            self._open = False
        elif key == pygame.K_LEFT:
            # Immediate step back + begin tracking held key for acceleration
            self._held_key = pygame.K_LEFT
            self._hold_frames = 0
            if self._state_history:
                self._controls.paused = True
                if self._replay_idx == -1:
                    self._replay_idx = max(0, len(self._state_history) - 2)
                else:
                    self._replay_idx = max(0, self._replay_idx - 1)
        elif key == pygame.K_RIGHT:
            # Immediate step forward + begin tracking held key for acceleration
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
            # Activate step-search bar (only when there is history to search)
            if self._state_history:
                self._search_active = True
                self._search_buffer = ""
        elif key == pygame.K_z:
            # Reset zoom and pan to default view
            self._zoom_scale = 1.0
            self._drag_offset = [0, 0]
        else:
            # Delegate SPACE/+/-/R to DashboardControls
            event = pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})
            handle_key_event(event, self._controls)

    # --- Search bar input ---

    def _handle_search_key(self, key: int) -> None:
        """Handle a KEYDOWN event while the step-search bar is active.

        Digit characters are handled separately via TEXTINPUT events in
        _handle_events (layout-independent). Only control keys are handled here.
        """
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
        """Jump replay to the history entry whose .step value is closest to step_num."""
        if not self._state_history:
            return
        best_idx = min(
            range(len(self._state_history)),
            key=lambda i: abs(self._state_history[i].step - step_num),
        )
        self._replay_idx = best_idx
        self._controls.paused = True

    # --- Held-key replay acceleration ---

    def _tick_held_key(self) -> None:
        """Auto-advance replay when LEFT/RIGHT is held, with progressive acceleration.

        Called once per frame after all events have been processed.
        Frame thresholds at 10 FPS (each frame ≈ 100 ms):
          frames 1-4   (~0-400 ms) : initial press already handled, no auto-advance
          frames 5-19  (~0.5-2 s)  : 1 extra step per frame  → ~10 steps/s
          frames 20-39 (~2-4 s)    : 3 extra steps per frame → ~30 steps/s
          frames ≥ 40  (>4 s)      : 6 extra steps per frame → ~60 steps/s
        """
        if self._held_key is None or not self._state_history:
            return
        self._hold_frames += 1
        if self._hold_frames < 5:
            return  # still in the natural initial-press phase

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
        assert self._layout is not None  # _ensure_layout must be called first

        self._last_state = state
        layout = self._effective_layout()

        draw_background(surface)

        fogged = fog_node_ids(state.network)
        # In replay: draw fog nodes as neutral grey icons instead of cloud blobs —
        # the cloud shape indicates "live unknown" but in replay we want a visible
        # intermediate shape. Edges still respect fog (no edges to unknown nodes).
        icons_fogged = set() if self.in_replay else fogged

        # --- Graph zone (centre) ---
        draw_edges(surface, state.network.graph, layout, fogged)
        draw_attacker_path(surface, state.attacker_path, layout)
        draw_node_icons(surface, state.network, layout, icons_fogged)
        draw_surveillance_shields(surface, state.network, layout, fogged)
        draw_pulse_effect(surface, state.network, layout, self._anim_time)
        draw_flash_effect(surface, self._flash_events, layout)
        draw_agent_halo(surface, state.agent_position, layout)
        draw_special_markers(
            surface,
            state.network.entry_node_id,
            state.network.target_node_id,
            layout,
            anim_time=self._anim_time,
        )
        draw_node_labels(surface, layout, self._font_label)

        # --- Selected node ring + info panel ---
        if self._selected_node is not None and self._selected_node in state.network.nodes:
            draw_selected_ring(surface, self._selected_node, layout)
            if self._selected_node in layout:
                px, py = layout[self._selected_node]
                draw_node_info_panel(
                    surface,
                    state.network.nodes[self._selected_node],
                    self._selected_node,
                    px, py,
                    self._font_panel_header,
                    self._font_panel_body,
                    surface.get_width(),
                )

        # --- Stats panel (top-left) ---
        stats_rect = pygame.Rect(*theme.STATS_PANEL_RECT)
        draw_stats_panel(surface, stats_rect, self._font_stats, state)

        # --- Sidebar collapsible panels ---
        sidebar_rect = pygame.Rect(*theme.SIDEBAR_RECT)
        fonts = {"header": self._font_panel_header, "body": self._font_panel_body}
        draw_sidebar_panels(surface, sidebar_rect, fonts, state, self._panel_expanded)
        # Update clickable header rects (in case sidebar layout changed)
        self._panel_header_rects = get_panel_header_rects(sidebar_rect, self._font_panel_header)

        # --- Suspicion bars (bottom-left) ---
        susp_rect = pygame.Rect(*theme.SUSPICION_RECT)
        draw_suspicion_bars(surface, susp_rect, self._font_log, state.per_node_suspicion)

        # --- Action log (right panel) ---
        log_rect = pygame.Rect(*theme.ACTION_LOG_RECT)
        draw_action_log(surface, log_rect, self._font_log, state.action_log, self._log_scroll_offset)

        # --- Controls hint (bottom centre) ---
        if self._search_active:
            hint_text = f"Go to step: {self._search_buffer}_    [Enter=jump  Esc=cancel]"
            hint_color = (255, 220, 60)  # yellow — search mode
        elif self.in_replay:
            step_num = self._state_history[self._replay_idx].step
            total = len(self._state_history)
            hint_text = (
                f"REPLAY  step {step_num}  [{self._replay_idx + 1}/{total}]  "
                f"← prev  → next/resume  G=jump  ESC=quit"
            )
            hint_color = theme.COLOR_TARGET_MARKER  # gold, visible during replay
        else:
            hint_text = (
                "ESC=quit  drag=pan  scroll=zoom  Z=reset  click=info  SPACE=pause  +/-=speed  ←/→=replay  G=jump"
            )
            hint_color = theme.COLOR_HINT
        hint = self._font_log.render(hint_text, True, hint_color)
        surface.blit(hint, (theme.GRAPH_AREA_LEFT + 10, theme.WINDOW_HEIGHT - 16))

        # --- Step-search bar overlay (centred on graph area, above terminal banner) ---
        if self._search_active:
            font_search = _load_font(22)
            label = f"Go to step:  {self._search_buffer}\u2588"  # block cursor
            label_surf = font_search.render(label, True, (255, 220, 60))
            sub_surf = self._font_log.render(
                "Enter = jump   Esc = cancel", True, theme.COLOR_HINT
            )
            box_w = max(label_surf.get_width(), sub_surf.get_width()) + 36
            box_h = label_surf.get_height() + sub_surf.get_height() + 22
            cx = theme.GRAPH_AREA_LEFT + theme.GRAPH_AREA_WIDTH // 2
            cy = theme.GRAPH_AREA_HEIGHT // 2 - 40
            box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            box.fill((10, 12, 40, 230))
            surface.blit(box, (cx - box_w // 2, cy - box_h // 2))
            pygame.draw.rect(
                surface, theme.COLOR_PANEL_BORDER,
                pygame.Rect(cx - box_w // 2, cy - box_h // 2, box_w, box_h), 1,
            )
            surface.blit(label_surf, (cx - label_surf.get_width() // 2, cy - box_h // 2 + 8))
            surface.blit(
                sub_surf,
                (cx - sub_surf.get_width() // 2, cy - box_h // 2 + 8 + label_surf.get_height() + 6),
            )


