"""Phase 2 test configuration — headless Pygame setup.

SDL_VIDEODRIVER and SDL_AUDIODRIVER must be set BEFORE any pygame import,
so they are set at module level here. pytest loads conftest.py before
collecting test modules, guaranteeing the env vars are in place.
"""

import os

import pygame
import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")


@pytest.fixture
def pygame_surface():
    """Initialize Pygame, yield a 800x600 surface, then quit cleanly.

    Avoids the fragile inline pygame.init() / pygame.quit() pattern that
    leaves Pygame in an inconsistent state when a test fails mid-execution.
    """
    pygame.init()
    surface = pygame.Surface((800, 600))
    yield surface
    pygame.quit()
