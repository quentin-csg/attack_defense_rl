"""Visual constants for the Pygame dashboard.

Pure data module — no Pygame imports. Colors are plain RGB(A) tuples.
All visualization code reads constants from here; nothing is hardcoded elsewhere.
"""

# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------
WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_TITLE: str = "Attack & Defense RL"
FPS: int = 10

# ---------------------------------------------------------------------------
# Background gradient
# ---------------------------------------------------------------------------
BG_COLOR: tuple[int, int, int] = (26, 26, 46)       # #1a1a2e — dark navy (top)
BG_COLOR_BOTTOM: tuple[int, int, int] = (10, 10, 28) # slightly darker at bottom
# Subtle radial glow in the graph centre (vignette effect)
BG_GLOW_COLOR: tuple[int, int, int] = (30, 35, 70)  # blue-purple tint for centre glow

# ---------------------------------------------------------------------------
# Node fill colors  (applied in priority order — see graph_view.get_node_color)
# ---------------------------------------------------------------------------
# 1. Node is offline (isolated by Blue Team)
COLOR_NODE_OFFLINE: tuple[int, int, int] = (40, 40, 40)

# 2. Node is unknown (fog of war — UNKNOWN discovery level)
COLOR_NODE_UNKNOWN: tuple[int, int, int] = (60, 60, 60)

# 3. Active session states
COLOR_NODE_ROOT: tuple[int, int, int] = (255, 34, 34)   # bright red
COLOR_NODE_USER: tuple[int, int, int] = (204, 0, 0)     # red

# 4. Known but not compromised
COLOR_NODE_ENUMERATED: tuple[int, int, int] = (0, 204, 204)   # cyan
COLOR_NODE_DISCOVERED: tuple[int, int, int] = (204, 204, 0)   # yellow

# ---------------------------------------------------------------------------
# Special node markers (drawn as concentric rings around the node)
# ---------------------------------------------------------------------------
COLOR_ENTRY_MARKER: tuple[int, int, int] = (0, 255, 100)   # green ring for entry node
COLOR_TARGET_MARKER: tuple[int, int, int] = (255, 215, 0)  # gold ring for loot/target node
COLOR_AGENT_HALO: tuple[int, int, int] = (220, 220, 255)   # pale white ring for agent pos

# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------
COLOR_EDGE: tuple[int, int, int] = (0, 160, 140)  # teal — edges between discovered nodes
# Edges to fogged nodes are not drawn (skipped entirely in draw_edges).

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------
COLOR_TEXT_KEY: tuple[int, int, int] = (0, 220, 80)      # green — stat label
COLOR_TEXT_VALUE: tuple[int, int, int] = (255, 255, 255) # white — stat value
COLOR_TEXT_LABEL: tuple[int, int, int] = (160, 160, 160) # gray — node labels

# ---------------------------------------------------------------------------
# Sizes
# ---------------------------------------------------------------------------
NODE_RADIUS: int = 24
NODE_RADIUS_UNKNOWN: int = 18    # fog cloud approximate radius
AGENT_HALO_RADIUS: int = 34      # radius of the agent halo ring
SPECIAL_MARKER_RADIUS: int = 32  # radius of entry/target rings
EDGE_WIDTH: int = 1
MARKER_RING_WIDTH: int = 2       # thickness of special marker rings
HALO_RING_WIDTH: int = 3         # thickness of each agent halo ring

# Node background circle (dark circle behind each icon)
NODE_BG_RADIUS: int = 24
COLOR_NODE_BG: tuple[int, int, int] = (30, 30, 50)

# Node ID number rendered on the node
FONT_SIZE_NODE_ID: int = 16

# Fog cloud colours
COLOR_FOG_CLOUD: tuple[int, int, int] = (80, 80, 90)
COLOR_FOG_CLOUD_DARK: tuple[int, int, int] = (55, 55, 65)

# Compromised glow (RGBA — drawn on SRCALPHA surface)
GLOW_RADIUS_USER: int = 34
GLOW_RADIUS_ROOT: int = 40
COLOR_GLOW_USER: tuple[int, int, int, int] = (204, 0, 0, 50)
COLOR_GLOW_ROOT: tuple[int, int, int, int] = (255, 34, 34, 70)

# Icon internal detail colours
COLOR_ICON_SCREEN: tuple[int, int, int] = (20, 20, 35)    # dark screen on monitors
COLOR_ICON_LED: tuple[int, int, int] = (0, 255, 100)       # green LED dots on server racks
COLOR_ICON_ANTENNA: tuple[int, int, int] = (140, 140, 160) # antenna lines on routers

# ---------------------------------------------------------------------------
# Layout — panel widths
# ---------------------------------------------------------------------------
LEFT_PANEL_WIDTH: int = 220    # left sidebar (stats box + collapsible panels + suspicion bars)
RIGHT_PANEL_WIDTH: int = 250   # right action log panel

# The graph is rendered in the centre zone between the two sidebars.
GRAPH_AREA_LEFT: int = LEFT_PANEL_WIDTH
GRAPH_AREA_TOP: int = 0
GRAPH_AREA_WIDTH: int = WINDOW_WIDTH - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH  # 810px
GRAPH_AREA_HEIGHT: int = WINDOW_HEIGHT
GRAPH_MARGIN: int = 55  # pixels between graph edge nodes and the graph zone border

# ---------------------------------------------------------------------------
# Stats panel (top-left, inside the left sidebar)
# ---------------------------------------------------------------------------
STATS_PANEL_RECT: tuple[int, int, int, int] = (8, 8, LEFT_PANEL_WIDTH - 16, 110)
STATS_X: int = 15       # kept for backward compat with draw_stats_overlay
STATS_Y: int = 15
STATS_LINE_HEIGHT: int = 24

# ---------------------------------------------------------------------------
# Sidebar collapsible panels (below stats panel, inside the left sidebar)
# ---------------------------------------------------------------------------
SIDEBAR_RECT: tuple[int, int, int, int] = (8, 126, LEFT_PANEL_WIDTH - 16, 340)
PANEL_HEADER_HEIGHT: int = 26  # height of each panel header row

# ---------------------------------------------------------------------------
# Suspicion bars (bottom-left, inside the left sidebar)
# ---------------------------------------------------------------------------
SUSPICION_RECT: tuple[int, int, int, int] = (8, 474, LEFT_PANEL_WIDTH - 16, 238)

# ---------------------------------------------------------------------------
# Action log (right panel)
# ---------------------------------------------------------------------------
ACTION_LOG_RECT: tuple[int, int, int, int] = (
    WINDOW_WIDTH - RIGHT_PANEL_WIDTH + 4,
    8,
    RIGHT_PANEL_WIDTH - 12,
    WINDOW_HEIGHT - 16,
)

# ---------------------------------------------------------------------------
# Panel styling
# ---------------------------------------------------------------------------
COLOR_PANEL_BG: tuple[int, int, int, int] = (15, 15, 35, 200)   # RGBA semi-transparent
COLOR_PANEL_BORDER: tuple[int, int, int] = (0, 160, 140)         # teal border
COLOR_PANEL_HEADER: tuple[int, int, int] = (0, 220, 80)          # green header text
COLOR_PANEL_BODY: tuple[int, int, int] = (180, 180, 180)         # gray body text

# ---------------------------------------------------------------------------
# Suspicion bar colours (by threshold)
# ---------------------------------------------------------------------------
COLOR_SUSPICION_LOW: tuple[int, int, int] = (0, 200, 80)     # green   0-30
COLOR_SUSPICION_MED: tuple[int, int, int] = (220, 220, 0)    # yellow  30-60
COLOR_SUSPICION_HIGH: tuple[int, int, int] = (255, 140, 0)   # orange  60-80
COLOR_SUSPICION_CRIT: tuple[int, int, int] = (255, 30, 30)   # red     80-100

# ---------------------------------------------------------------------------
# Action log colours
# ---------------------------------------------------------------------------
COLOR_LOG_RED_SUCCESS: tuple[int, int, int] = (0, 220, 80)   # green  — Red Team success
COLOR_LOG_RED_FAIL: tuple[int, int, int] = (180, 180, 0)     # yellow — Red Team fail
COLOR_LOG_BLUE_ACTION: tuple[int, int, int] = (0, 180, 220)  # cyan   — Blue Team action
COLOR_LOG_FAILURE: tuple[int, int, int] = (255, 60, 60)      # red    — critical failures

# ---------------------------------------------------------------------------
# Animations
# ---------------------------------------------------------------------------
PULSE_SPEED: float = 3.0        # radians/second for ROOT node pulse
PULSE_MIN_RADIUS: int = 26
PULSE_MAX_RADIUS: int = 38
FLASH_DURATION: float = 0.5     # seconds for exploit/blue-action flash
ATTACKER_PATH_COLOR: tuple[int, int, int] = (60, 100, 255)   # blue attacker path edges
ATTACKER_PATH_WIDTH: int = 4

# ---------------------------------------------------------------------------
# Node icons
# ---------------------------------------------------------------------------
ICON_SIZE: int = 38   # bounding box for geometric node icons

# ---------------------------------------------------------------------------
# Controls hint
# ---------------------------------------------------------------------------
COLOR_HINT: tuple[int, int, int] = (80, 80, 100)  # dim text for controls hint

# ---------------------------------------------------------------------------
# Stats overlay (top-left) — kept for backward compat
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
# Pygame tries these in order; falls back to the built-in default if none found.
FONT_CANDIDATES: list[str] = ["jetbrainsmono", "consolas", "couriernew", "monospace"]
FONT_SIZE_STATS: int = 18
FONT_SIZE_LABEL: int = 13
FONT_SIZE_PANEL_HEADER: int = 14
FONT_SIZE_PANEL_BODY: int = 12
FONT_SIZE_LOG: int = 11
