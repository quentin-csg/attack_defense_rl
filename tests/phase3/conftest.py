"""Phase 3 test fixtures."""

from __future__ import annotations

import pytest

from src.agents.wrappers import make_masked_env


@pytest.fixture
def masked_env():
    """CyberEnv wrapped with ActionMasker, seeded, short episodes."""
    env = make_masked_env(seed=42, max_steps=50)
    yield env
    env.close()
