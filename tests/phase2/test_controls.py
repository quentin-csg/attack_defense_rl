"""Tests for DashboardControls and handle_key_event."""

from __future__ import annotations

import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from src.visualization.controls import _SPEED_PRESETS, DashboardControls, handle_key_event


@pytest.fixture(autouse=True)
def init_pygame():
    pygame.init()
    yield
    pygame.quit()


class TestDashboardControls:
    def test_default_state(self) -> None:
        c = DashboardControls()
        assert c.paused is False
        assert c.speed_multiplier == 1.0

    def test_toggle_pause(self) -> None:
        c = DashboardControls()
        c.toggle_pause()
        assert c.paused is True
        c.toggle_pause()
        assert c.paused is False

    def test_speed_up(self) -> None:
        c = DashboardControls()
        original = c.speed_multiplier
        c.speed_up()
        assert c.speed_multiplier > original

    def test_speed_down(self) -> None:
        c = DashboardControls()
        original = c.speed_multiplier
        c.speed_down()
        assert c.speed_multiplier < original

    def test_speed_up_capped_at_max(self) -> None:
        c = DashboardControls()
        for _ in range(20):
            c.speed_up()
        assert c.speed_multiplier == max(_SPEED_PRESETS)

    def test_speed_down_capped_at_min(self) -> None:
        c = DashboardControls()
        for _ in range(20):
            c.speed_down()
        assert c.speed_multiplier == min(_SPEED_PRESETS)

    def test_reset_restores_defaults(self) -> None:
        c = DashboardControls()
        c.toggle_pause()
        c.speed_up()
        c.reset()
        assert c.paused is False
        assert c.speed_multiplier == 1.0


class TestHandleKeyEvent:
    def _keydown(self, key: int) -> pygame.event.Event:
        return pygame.event.Event(pygame.KEYDOWN, {"key": key, "mod": 0, "unicode": ""})

    def test_space_toggles_pause(self) -> None:
        c = DashboardControls()
        consumed = handle_key_event(self._keydown(pygame.K_SPACE), c)
        assert consumed is True
        assert c.paused is True

    def test_plus_speeds_up(self) -> None:
        c = DashboardControls()
        original = c.speed_multiplier
        consumed = handle_key_event(self._keydown(pygame.K_PLUS), c)
        assert consumed is True
        assert c.speed_multiplier > original

    def test_minus_slows_down(self) -> None:
        c = DashboardControls()
        original = c.speed_multiplier
        consumed = handle_key_event(self._keydown(pygame.K_MINUS), c)
        assert consumed is True
        assert c.speed_multiplier < original

    def test_r_resets(self) -> None:
        c = DashboardControls()
        c.speed_up()
        c.toggle_pause()
        consumed = handle_key_event(self._keydown(pygame.K_r), c)
        assert consumed is True
        assert c.paused is False
        assert c.speed_multiplier == 1.0

    def test_unrelated_key_not_consumed(self) -> None:
        c = DashboardControls()
        consumed = handle_key_event(self._keydown(pygame.K_a), c)
        assert consumed is False

    def test_non_keydown_event_not_consumed(self) -> None:
        c = DashboardControls()
        event = pygame.event.Event(pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (0, 0)})
        consumed = handle_key_event(event, c)
        assert consumed is False
