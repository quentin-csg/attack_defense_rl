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
# Background
# ---------------------------------------------------------------------------
BG_COLOR: tuple[int, int, int] = (26, 26, 46)  # #1a1a2e — dark navy

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
NODE_RADIUS: int = 18
NODE_RADIUS_UNKNOWN: int = 10    # smaller silhouette for fogged nodes
AGENT_HALO_RADIUS: int = 27      # radius of the agent halo ring
SPECIAL_MARKER_RADIUS: int = 24  # radius of entry/target rings
EDGE_WIDTH: int = 2
MARKER_RING_WIDTH: int = 2       # thickness of special marker rings
HALO_RING_WIDTH: int = 3

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# The graph is rendered inside a rect that covers most of the window.
GRAPH_AREA_LEFT: int = 0
GRAPH_AREA_TOP: int = 0
GRAPH_AREA_WIDTH: int = WINDOW_WIDTH
GRAPH_AREA_HEIGHT: int = WINDOW_HEIGHT
GRAPH_MARGIN: int = 110  # pixels between graph edge nodes and window border

# ---------------------------------------------------------------------------
# Stats overlay (top-left)
# ---------------------------------------------------------------------------
STATS_X: int = 15
STATS_Y: int = 15
STATS_LINE_HEIGHT: int = 24
STATS_PADDING: int = 10

# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
# Pygame tries these in order; falls back to the built-in default if none found.
FONT_CANDIDATES: list[str] = ["jetbrainsmono", "consolas", "couriernew", "monospace"]
FONT_SIZE_STATS: int = 18
FONT_SIZE_LABEL: int = 13
