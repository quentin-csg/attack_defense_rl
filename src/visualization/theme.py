"""Visual constants for the Pygame dashboard.

Pure data module — no Pygame imports. Colors are plain RGB(A) tuples.
"""

WINDOW_WIDTH: int = 1280
WINDOW_HEIGHT: int = 720
WINDOW_TITLE: str = "Attack & Defense RL"
FPS: int = 10

BG_COLOR: tuple[int, int, int] = (26, 26, 46)
BG_COLOR_BOTTOM: tuple[int, int, int] = (10, 10, 28)
BG_GLOW_COLOR: tuple[int, int, int] = (30, 35, 70)

COLOR_NODE_OFFLINE: tuple[int, int, int] = (40, 40, 40)
COLOR_NODE_UNKNOWN: tuple[int, int, int] = (60, 60, 60)
COLOR_NODE_ROOT: tuple[int, int, int] = (255, 34, 34)
COLOR_NODE_USER: tuple[int, int, int] = (204, 0, 0)
COLOR_NODE_ENUMERATED: tuple[int, int, int] = (0, 204, 204)
COLOR_NODE_DISCOVERED: tuple[int, int, int] = (204, 204, 0)

COLOR_ENTRY_MARKER: tuple[int, int, int] = (0, 255, 100)
COLOR_TARGET_MARKER: tuple[int, int, int] = (255, 215, 0)
COLOR_AGENT_HALO: tuple[int, int, int] = (220, 220, 255)

COLOR_EDGE: tuple[int, int, int] = (0, 160, 140)

COLOR_TEXT_KEY: tuple[int, int, int] = (0, 220, 80)
COLOR_TEXT_VALUE: tuple[int, int, int] = (255, 255, 255)
COLOR_TEXT_LABEL: tuple[int, int, int] = (160, 160, 160)

NODE_RADIUS: int = 24
NODE_RADIUS_UNKNOWN: int = 18
AGENT_HALO_RADIUS: int = 34
SPECIAL_MARKER_RADIUS: int = 32
EDGE_WIDTH: int = 1
MARKER_RING_WIDTH: int = 2
HALO_RING_WIDTH: int = 3

NODE_BG_RADIUS: int = 24
COLOR_NODE_BG: tuple[int, int, int] = (30, 30, 50)
FONT_SIZE_NODE_ID: int = 16

COLOR_FOG_CLOUD: tuple[int, int, int] = (80, 80, 90)
COLOR_FOG_CLOUD_DARK: tuple[int, int, int] = (55, 55, 65)

GLOW_RADIUS_USER: int = 34
GLOW_RADIUS_ROOT: int = 40
COLOR_GLOW_USER: tuple[int, int, int, int] = (204, 0, 0, 50)
COLOR_GLOW_ROOT: tuple[int, int, int, int] = (255, 34, 34, 70)

COLOR_ICON_SCREEN: tuple[int, int, int] = (20, 20, 35)
COLOR_ICON_LED: tuple[int, int, int] = (0, 255, 100)
COLOR_ICON_ANTENNA: tuple[int, int, int] = (140, 140, 160)

LEFT_PANEL_WIDTH: int = 220
RIGHT_PANEL_WIDTH: int = 380
GRAPH_AREA_LEFT: int = LEFT_PANEL_WIDTH
GRAPH_AREA_TOP: int = 0
GRAPH_AREA_WIDTH: int = WINDOW_WIDTH - LEFT_PANEL_WIDTH - RIGHT_PANEL_WIDTH
GRAPH_AREA_HEIGHT: int = WINDOW_HEIGHT
GRAPH_MARGIN: int = 55

STATS_PANEL_RECT: tuple[int, int, int, int] = (8, 8, LEFT_PANEL_WIDTH - 16, 110)
STATS_X: int = 15
STATS_Y: int = 15
STATS_LINE_HEIGHT: int = 24

SIDEBAR_RECT: tuple[int, int, int, int] = (8, 126, LEFT_PANEL_WIDTH - 16, 340)
PANEL_HEADER_HEIGHT: int = 26

SUSPICION_RECT: tuple[int, int, int, int] = (8, 474, LEFT_PANEL_WIDTH - 16, 238)

ACTION_LOG_RECT: tuple[int, int, int, int] = (
    WINDOW_WIDTH - RIGHT_PANEL_WIDTH + 4,
    8,
    RIGHT_PANEL_WIDTH - 12,
    WINDOW_HEIGHT - 16,
)

COLOR_PANEL_BG: tuple[int, int, int, int] = (15, 15, 35, 200)
COLOR_PANEL_BORDER: tuple[int, int, int] = (0, 160, 140)
COLOR_PANEL_HEADER: tuple[int, int, int] = (0, 220, 80)
COLOR_PANEL_BODY: tuple[int, int, int] = (180, 180, 180)

COLOR_SUSPICION_LOW: tuple[int, int, int] = (0, 200, 80)
COLOR_SUSPICION_MED: tuple[int, int, int] = (220, 220, 0)
COLOR_SUSPICION_HIGH: tuple[int, int, int] = (255, 140, 0)
COLOR_SUSPICION_CRIT: tuple[int, int, int] = (255, 30, 30)

COLOR_LOG_RED_SUCCESS: tuple[int, int, int] = (0, 220, 80)
COLOR_LOG_RED_FAIL: tuple[int, int, int] = (180, 180, 0)
COLOR_LOG_BLUE_ACTION: tuple[int, int, int] = (0, 180, 220)
COLOR_LOG_FAILURE: tuple[int, int, int] = (255, 60, 60)

PULSE_SPEED: float = 3.0
PULSE_MIN_RADIUS: int = 26
PULSE_MAX_RADIUS: int = 38
FLASH_DURATION: float = 0.5
ATTACKER_PATH_COLOR: tuple[int, int, int] = (60, 100, 255)
ATTACKER_PATH_WIDTH: int = 4

ICON_SIZE: int = 38
COLOR_HINT: tuple[int, int, int] = (80, 80, 100)

FONT_CANDIDATES: list[str] = ["jetbrainsmono", "consolas", "couriernew", "monospace"]
FONT_SIZE_STATS: int = 18
FONT_SIZE_LABEL: int = 13
FONT_SIZE_PANEL_HEADER: int = 14
FONT_SIZE_PANEL_BODY: int = 12
FONT_SIZE_LOG: int = 11
