from __future__ import annotations

from dataclasses import dataclass, field

import pygame

# Speed presets cycled through with + / - keys
_SPEED_PRESETS: list[float] = [0.25, 0.5, 1.0, 2.0, 4.0]


@dataclass
class DashboardControls:
    """Mutable UI state for the visualisation script."""

    paused: bool = False
    speed_multiplier: float = 1.0
    restart_requested: bool = False
    _speed_idx: int = field(default=2, repr=False)  # index of 1.0 in _SPEED_PRESETS

    def toggle_pause(self) -> None:
        """Toggle pause on/off."""
        self.paused = not self.paused

    def speed_up(self) -> None:
        """Increase speed by one preset step."""
        self._speed_idx = min(self._speed_idx + 1, len(_SPEED_PRESETS) - 1)
        self.speed_multiplier = _SPEED_PRESETS[self._speed_idx]

    def speed_down(self) -> None:
        """Decrease speed by one preset step."""
        self._speed_idx = max(self._speed_idx - 1, 0)
        self.speed_multiplier = _SPEED_PRESETS[self._speed_idx]

    def reset(self) -> None:
        """Reset speed to default (1.0) and unpause."""
        self._speed_idx = 2
        self.speed_multiplier = _SPEED_PRESETS[self._speed_idx]
        self.paused = False


def handle_key_event(event: pygame.event.Event, controls: DashboardControls) -> bool:
    """Process a KEYDOWN event and update controls accordingly."""
    if event.type != pygame.KEYDOWN:
        return False

    key = event.key
    if key == pygame.K_SPACE:
        controls.toggle_pause()
        return True
    if key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS, pygame.K_UP):
        controls.speed_up()
        return True
    if key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_DOWN):
        controls.speed_down()
        return True
    if key == pygame.K_r:
        controls.restart_requested = True
        return True
    return False
