"""Tests for the PCG CurriculumManager (Phase 5)."""

from __future__ import annotations

from src.pcg.curriculum import CurriculumManager, CurriculumStage
from src.pcg.difficulty import is_solvable
from src.pcg.generator import NetworkSize


# ---------------------------------------------------------------------------
# Basic stage progression
# ---------------------------------------------------------------------------


def test_default_stages_are_small_medium_large() -> None:
    cm = CurriculumManager(seed=42)
    assert len(cm.stages) == 3
    assert cm.stages[0].size == NetworkSize.SMALL
    assert cm.stages[1].size == NetworkSize.MEDIUM
    assert cm.stages[2].size == NetworkSize.LARGE


def test_initial_state() -> None:
    cm = CurriculumManager(seed=42)
    assert cm.current_stage_idx == 0
    assert cm.current_world_idx == 0
    assert cm.current_stage.size == NetworkSize.SMALL
    assert not cm.is_complete


def test_world_advancement_within_stage() -> None:
    cm = CurriculumManager(seed=42)
    max_worlds = cm.stages[0].max_worlds

    for i in range(max_worlds - 1):
        changed = cm.advance_world()
        assert not changed, f"Stage changed unexpectedly at world {i+1}"
        assert cm.current_world_idx == i + 1
        assert cm.current_stage_idx == 0


def test_stage_progression_small_to_medium() -> None:
    cm = CurriculumManager(seed=42)
    max_worlds = cm.stages[0].max_worlds

    for _ in range(max_worlds):
        cm.advance_world()

    assert cm.current_stage_idx == 1
    assert cm.current_stage.size == NetworkSize.MEDIUM
    assert cm.current_world_idx == 0


def test_stage_progression_through_all_stages() -> None:
    cm = CurriculumManager(seed=42)
    total_worlds = sum(s.max_worlds for s in cm.stages)

    for i in range(total_worlds - 1):
        cm.advance_world()

    # Should be on the last stage, last world
    assert cm.current_stage_idx == len(cm.stages) - 1
    assert not cm.is_complete  # last world not yet completed

    cm.advance_world()
    assert cm.is_complete


def test_curriculum_completion_flag() -> None:
    cm = CurriculumManager(seed=0)
    total_worlds = sum(s.max_worlds for s in cm.stages)
    for _ in range(total_worlds):
        cm.advance_world()
    assert cm.is_complete


def test_reset_clears_state() -> None:
    cm = CurriculumManager(seed=42)
    for _ in range(7):
        cm.advance_world()
    cm.reset()
    assert cm.current_stage_idx == 0
    assert cm.current_world_idx == 0
    assert not cm.is_complete


def test_generates_valid_networks() -> None:
    """Each world in the curriculum generates a solvable network."""
    cm = CurriculumManager(seed=99)
    for _ in range(3):  # check first 3 worlds (small only for speed)
        net, meta = cm.generate_current_network()
        assert is_solvable(net), f"World at stage {cm.current_stage.size.value} not solvable"
        cm.advance_world()


def test_different_worlds_different_topologies() -> None:
    """Consecutive worlds in the same stage produce different networks."""
    cm = CurriculumManager(seed=42)
    edge_sets: list[frozenset] = []
    max_worlds = cm.stages[0].max_worlds

    for _ in range(max_worlds):
        net, _ = cm.generate_current_network()
        edge_sets.append(frozenset(net.graph.edges()))
        if cm.current_stage_idx == 0:
            cm.advance_world()
            if cm.current_stage_idx > 0:
                break

    assert len(set(edge_sets)) > 1, "All worlds produced identical topologies"


def test_custom_stages() -> None:
    """CurriculumManager accepts custom stage configurations."""
    custom_stages = [
        CurriculumStage(NetworkSize.SMALL, max_worlds=2, timesteps_per_world=10_000),
        CurriculumStage(NetworkSize.LARGE, max_worlds=2, timesteps_per_world=20_000),
    ]
    cm = CurriculumManager(stages=custom_stages, seed=7)
    assert cm.stages[0].size == NetworkSize.SMALL
    assert cm.stages[1].size == NetworkSize.LARGE

    # Advance through custom curriculum
    for _ in range(4):  # 2+2 worlds
        cm.advance_world()
    assert cm.is_complete
