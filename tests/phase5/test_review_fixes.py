"""Regression tests for bugs fixed during pre-Phase 6 review.

Covers:
- C2: _distribute_nodes never produces sum(counts) > n_nodes
- C3: _pick_vulns never crashes on empty valid_pool
- M3: advance_world() is a no-op after curriculum completion
- visualize.py arg parsing (no --max-steps) does not crash
- evaluate.py supports --pcg and --blue-team flags
"""

from __future__ import annotations

import random

import pytest


# ---------------------------------------------------------------------------
# C2: _distribute_nodes correctness
# ---------------------------------------------------------------------------


class TestDistributeNodes:
    def _counts(self, n_nodes: int, n_zones: int, seed: int = 0) -> list[int]:
        from src.pcg.generator import _distribute_nodes, Zone

        rng = random.Random(seed)
        # Build a fake zone sequence of length n_zones
        zones = [Zone.DMZ] + [Zone.CORPORATE] * (n_zones - 2) + [Zone.DATACENTER]
        return _distribute_nodes(n_nodes, zones, rng)

    def test_sum_equals_n_nodes_small_network(self) -> None:
        """Total node count must equal n_nodes regardless of zone configuration."""
        for seed in range(20):
            counts = self._counts(n_nodes=10, n_zones=4, seed=seed)
            assert sum(counts) == 10, f"seed={seed}: sum={sum(counts)} != 10, counts={counts}"

    def test_sum_equals_n_nodes_tight_fit(self) -> None:
        """n_nodes = n_zones * 2 — very tight distribution."""
        for seed in range(20):
            counts = self._counts(n_nodes=8, n_zones=4, seed=seed)
            assert sum(counts) == 8, f"seed={seed}: sum={sum(counts)} != 8, counts={counts}"

    def test_sum_equals_n_nodes_extreme(self) -> None:
        """6 nodes across 5 zones — worst-case tight fit."""
        for seed in range(20):
            counts = self._counts(n_nodes=6, n_zones=5, seed=seed)
            assert sum(counts) == 6, f"seed={seed}: sum={sum(counts)} != 6, counts={counts}"

    def test_all_counts_positive(self) -> None:
        """All zone counts must be >= 1 (zones can't have 0 nodes)."""
        for n_nodes in range(8, 16):
            for n_zones in range(2, 5):
                for seed in range(5):
                    counts = self._counts(n_nodes=n_nodes, n_zones=n_zones, seed=seed)
                    assert all(c >= 1 for c in counts), (
                        f"n_nodes={n_nodes}, n_zones={n_zones}, seed={seed}: "
                        f"zero-count zone in {counts}"
                    )

    def test_two_zone_network(self) -> None:
        """DMZ + DC only (n_zones=2) must distribute all nodes."""
        from src.pcg.generator import _distribute_nodes, Zone
        rng = random.Random(42)
        counts = _distribute_nodes(8, [Zone.DMZ, Zone.DATACENTER], rng)
        assert sum(counts) == 8
        assert len(counts) == 2


# ---------------------------------------------------------------------------
# C3: _pick_vulns safety
# ---------------------------------------------------------------------------


class TestPickVulns:
    def test_no_crash_empty_pool(self) -> None:
        """_pick_vulns must not crash when the registry has no matching vulns."""
        from src.pcg.generator import Zone, _pick_vulns
        from src.environment.vulnerability import VULN_REGISTRY

        # Temporarily empty the registry to simulate missing vulns
        backup = dict(VULN_REGISTRY)
        VULN_REGISTRY.clear()
        try:
            rng = random.Random(42)
            # Target returns [] (no vulns) — no crash even with empty registry
            result = _pick_vulns(Zone.DATACENTER, is_target=True, rng=rng)
            assert result == []
        finally:
            VULN_REGISTRY.update(backup)

    def test_target_has_no_vulns(self) -> None:
        """Target nodes have no vulnerabilities — win via LIST_FILES (ls)."""
        from src.pcg.generator import Zone, _pick_vulns

        rng = random.Random(42)
        for zone in Zone:
            result = _pick_vulns(zone, is_target=True, rng=rng)
            assert result == [], f"Expected [] for target in zone={zone}, got {result}"

    def test_no_crash_100_generations(self) -> None:
        """_pick_vulns must never crash across 100 random calls per zone."""
        from src.pcg.generator import Zone, _pick_vulns

        for seed in range(100):
            rng = random.Random(seed)
            for zone in Zone:
                # non-target: at least 1 vuln
                result = _pick_vulns(zone, is_target=False, rng=rng)
                assert isinstance(result, list)
                assert len(result) >= 1
                # target: no vulns
                result_target = _pick_vulns(zone, is_target=True, rng=rng)
                assert result_target == []


# ---------------------------------------------------------------------------
# M3: CurriculumManager.advance_world() guard after completion
# ---------------------------------------------------------------------------


class TestCurriculumCompletionGuard:
    def test_advance_world_after_completion_is_noop(self) -> None:
        """advance_world() after is_complete must not change state."""
        from src.pcg.curriculum import CurriculumManager, CurriculumStage

        # Tiny curriculum: 1 stage, 1 world
        cm = CurriculumManager(
            stages=[CurriculumStage(size="small", max_worlds=1, timesteps_per_world=100)],
            seed=42,
        )
        # Exhaust the curriculum
        cm.advance_world()
        assert cm.is_complete

        # Record state
        stage_before = cm.current_stage_idx
        world_before = cm.current_world_idx
        offset_before = cm._world_seed_offset

        # Calling again must be a no-op
        result = cm.advance_world()
        assert result is False
        assert cm.current_stage_idx == stage_before
        assert cm.current_world_idx == world_before
        assert cm._world_seed_offset == offset_before

    def test_advance_world_multiple_times_after_completion(self) -> None:
        """Repeated calls after completion must all be no-ops."""
        from src.pcg.curriculum import CurriculumManager, CurriculumStage

        cm = CurriculumManager(
            stages=[CurriculumStage(size="small", max_worlds=2, timesteps_per_world=100)],
            seed=0,
        )
        cm.advance_world()
        cm.advance_world()
        assert cm.is_complete

        for _ in range(5):
            cm.advance_world()

        assert cm.is_complete
        assert cm.current_world_idx == 2  # stayed at completion boundary


# ---------------------------------------------------------------------------
# evaluate.py argument parsing (no crash without model on disk)
# ---------------------------------------------------------------------------


class TestEvaluateArgParsing:
    def test_parses_pcg_flag(self) -> None:
        """evaluate.py parse_args must accept --pcg small."""
        import sys
        from scripts.evaluate import parse_args  # type: ignore[import]

        old_argv = sys.argv
        sys.argv = ["evaluate.py", "dummy_model.zip", "--pcg", "small"]
        try:
            args = parse_args()
            assert args.pcg == "small"
            assert args.blue_team is False
        finally:
            sys.argv = old_argv

    def test_parses_blue_team_flag(self) -> None:
        """evaluate.py parse_args must accept --blue-team."""
        import sys
        from scripts.evaluate import parse_args  # type: ignore[import]

        old_argv = sys.argv
        sys.argv = ["evaluate.py", "dummy_model.zip", "--blue-team"]
        try:
            args = parse_args()
            assert args.blue_team is True
            assert args.pcg is None
        finally:
            sys.argv = old_argv

    def test_parses_combined_flags(self) -> None:
        """evaluate.py parse_args must accept --pcg medium --blue-team together."""
        import sys
        from scripts.evaluate import parse_args  # type: ignore[import]

        old_argv = sys.argv
        sys.argv = ["evaluate.py", "dummy.zip", "--pcg", "medium", "--blue-team"]
        try:
            args = parse_args()
            assert args.pcg == "medium"
            assert args.blue_team is True
        finally:
            sys.argv = old_argv


# ---------------------------------------------------------------------------
# visualize.py: max_steps resolved locally (no None crash)
# ---------------------------------------------------------------------------


class TestVisualizeArgParsing:
    def test_max_steps_defaults_to_none(self) -> None:
        """visualize.py --max-steps defaults to None (resolved per mode)."""
        import sys
        from scripts.visualize import parse_args  # type: ignore[import]

        old_argv = sys.argv
        sys.argv = ["visualize.py"]
        try:
            args = parse_args()
            assert args.max_steps is None
        finally:
            sys.argv = old_argv

    def test_max_steps_pcg_resolves_to_size_default(self) -> None:
        """When --pcg is given without --max-steps, max_steps resolves from config."""
        from src.config import PCG_MAX_STEPS_SMALL

        import sys
        from scripts.visualize import parse_args  # type: ignore[import]

        old_argv = sys.argv
        sys.argv = ["visualize.py", "--pcg", "small"]
        try:
            args = parse_args()
            # Simulate the resolution logic from main()
            from src.config import PCG_MAX_STEPS_SMALL, PCG_MAX_STEPS_MEDIUM, PCG_MAX_STEPS_LARGE
            from src.pcg.generator import NetworkSize
            size_enum = NetworkSize(args.pcg)
            _pcg_max_steps = {
                NetworkSize.SMALL: PCG_MAX_STEPS_SMALL,
                NetworkSize.MEDIUM: PCG_MAX_STEPS_MEDIUM,
                NetworkSize.LARGE: PCG_MAX_STEPS_LARGE,
            }
            max_steps = args.max_steps or _pcg_max_steps[size_enum]
            assert max_steps == PCG_MAX_STEPS_SMALL
            # Crucially: max_steps is an int, not None
            assert isinstance(max_steps, int)
        finally:
            sys.argv = old_argv
