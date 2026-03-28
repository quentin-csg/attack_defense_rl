"""Curriculum learning manager for Phase 5.

Manages progressive training across three difficulty stages:
    Stage 1 (Small)  → 5 worlds × 100k timesteps
    Stage 2 (Medium) → 5 worlds × 150k timesteps
    Stage 3 (Large)  → 5 worlds × 200k timesteps

Each "world" is a fixed generated topology that the agent trains on for
a set number of timesteps before a new topology is generated (forcing
the agent to generalise rather than memorise).

Usage:
    curriculum = CurriculumManager(seed=42)
    while not curriculum.is_complete:
        net, meta = curriculum.generate_current_network()
        # ... train on net for meta.recommended_max_steps ...
        curriculum.advance_world()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.config import (
    CURRICULUM_TIMESTEPS_LARGE,
    CURRICULUM_TIMESTEPS_MEDIUM,
    CURRICULUM_TIMESTEPS_SMALL,
    CURRICULUM_WORLDS_PER_STAGE,
)
from src.pcg.generator import NetworkMeta, NetworkSize, generate_network
from src.environment.network import Network


@dataclass
class CurriculumStage:
    """One stage in the curriculum (e.g. all Small networks)."""

    size: NetworkSize
    max_worlds: int           # number of different topologies at this stage
    timesteps_per_world: int  # training budget per topology


@dataclass
class CurriculumManager:
    """Manages progression through curriculum stages.

    Args:
        stages: Ordered list of curriculum stages (default: small→medium→large).
        seed: Base random seed; world i at stage j uses seed + offset.
    """

    stages: list[CurriculumStage] = field(default_factory=list)
    seed: int = 42

    # Internal state
    current_stage_idx: int = field(default=0, init=False)
    current_world_idx: int = field(default=0, init=False)
    _world_seed_offset: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = [
                CurriculumStage(
                    size=NetworkSize.SMALL,
                    max_worlds=CURRICULUM_WORLDS_PER_STAGE,
                    timesteps_per_world=CURRICULUM_TIMESTEPS_SMALL,
                ),
                CurriculumStage(
                    size=NetworkSize.MEDIUM,
                    max_worlds=CURRICULUM_WORLDS_PER_STAGE,
                    timesteps_per_world=CURRICULUM_TIMESTEPS_MEDIUM,
                ),
                CurriculumStage(
                    size=NetworkSize.LARGE,
                    max_worlds=CURRICULUM_WORLDS_PER_STAGE,
                    timesteps_per_world=CURRICULUM_TIMESTEPS_LARGE,
                ),
            ]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_stage(self) -> CurriculumStage:
        """The active curriculum stage."""
        return self.stages[self.current_stage_idx]

    @property
    def is_complete(self) -> bool:
        """True once the last world of the last stage has been trained."""
        return (
            self.current_stage_idx >= len(self.stages) - 1
            and self.current_world_idx >= self.current_stage.max_worlds
        )

    @property
    def total_timesteps(self) -> int:
        """Total timesteps in the full curriculum (all stages × worlds)."""
        return sum(s.max_worlds * s.timesteps_per_world for s in self.stages)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_current_network(self) -> tuple[Network, NetworkMeta]:
        """Generate the network for the current curriculum position.

        Returns a new network each call (useful for testing), but the
        training loop should call this once per world, not per episode.
        """
        world_seed = self.seed + self._world_seed_offset
        return generate_network(self.current_stage.size, seed=world_seed)

    # ------------------------------------------------------------------
    # Advancement
    # ------------------------------------------------------------------

    def advance_world(self) -> bool:
        """Move to the next world within the current stage.

        If the current stage is exhausted, automatically advances to the
        next stage.

        Returns:
            True if the stage changed (moved to a harder stage), False otherwise.
        """
        if self.is_complete:
            return False

        self.current_world_idx += 1
        self._world_seed_offset += 1

        if self.current_world_idx >= self.current_stage.max_worlds:
            return self.advance_stage()
        return False

    def advance_stage(self) -> bool:
        """Move to the next curriculum stage.

        Returns:
            True if the stage changed, False if already on the last stage.
        """
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            self.current_world_idx = 0
            return True
        # Already on last stage — do NOT reset world_idx so is_complete can detect it
        return False

    def reset(self) -> None:
        """Reset to the beginning of the curriculum."""
        self.current_stage_idx = 0
        self.current_world_idx = 0
        self._world_seed_offset = 0
