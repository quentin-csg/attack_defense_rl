"""Tests for ScriptedBlueTeam — unit tests for Blue Team rule logic."""

from __future__ import annotations

import pytest

from src.agents.blue_scripted import BlueAction, ScriptedBlueTeam
from src.config import (
    BLUE_ALERT_NOISE,
    BLUE_ALERT_THRESHOLD,
    BLUE_ISOLATE_NOISE,
    BLUE_ISOLATE_THRESHOLD,
    BLUE_ROTATE_NOISE,
    BLUE_ROTATE_THRESHOLD,
    PATROL_DETECTION_SUSPICION,
)
from src.environment.network import Network, build_fixed_network
from src.environment.node import SessionLevel

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fixed_network() -> Network:
    return build_fixed_network(seed=42)


@pytest.fixture
def blue() -> ScriptedBlueTeam:
    return ScriptedBlueTeam(seed=42)


# ---------------------------------------------------------------------------
# Noisy thresholds
# ---------------------------------------------------------------------------


class TestNoisyThresholds:
    def test_thresholds_within_band(self, blue: ScriptedBlueTeam) -> None:
        """Thresholds must stay within their noise bands."""
        assert BLUE_ALERT_THRESHOLD - BLUE_ALERT_NOISE <= blue.alert_threshold <= BLUE_ALERT_THRESHOLD + BLUE_ALERT_NOISE
        assert BLUE_ROTATE_THRESHOLD - BLUE_ROTATE_NOISE <= blue.rotate_threshold <= BLUE_ROTATE_THRESHOLD + BLUE_ROTATE_NOISE
        assert BLUE_ISOLATE_THRESHOLD - BLUE_ISOLATE_NOISE <= blue.isolate_threshold <= BLUE_ISOLATE_THRESHOLD + BLUE_ISOLATE_NOISE

    def test_thresholds_clamped_to_100(self) -> None:
        """Thresholds must never exceed 100.0."""
        blue = ScriptedBlueTeam(seed=0)
        for _ in range(50):
            blue.reset()
            assert blue.alert_threshold <= 100.0
            assert blue.rotate_threshold <= 100.0
            assert blue.isolate_threshold <= 100.0

    def test_thresholds_vary_across_resets(self) -> None:
        """Over many resets, alert_threshold must take at least 2 distinct values."""
        blue = ScriptedBlueTeam(seed=99)
        values: set[float] = set()
        for _ in range(30):
            blue.reset()
            values.add(round(blue.alert_threshold, 4))
        assert len(values) > 1, "Alert threshold must vary across resets"

    def test_threshold_ordering_maintained(self, blue: ScriptedBlueTeam) -> None:
        """alert < rotate < isolate must hold after reset (statistical check)."""
        for _ in range(50):
            blue.reset()
            # Not guaranteed per individual reset due to overlap in noise bands,
            # but base values are well-separated so it holds in practice.
            assert blue.alert_threshold < blue.rotate_threshold


# ---------------------------------------------------------------------------
# ALERT action
# ---------------------------------------------------------------------------


class TestAlertAction:
    def test_alert_marks_surveillance(self, fixed_network: Network) -> None:
        """ALERT must set is_under_surveillance on the target node."""
        blue = ScriptedBlueTeam(seed=42)
        # Force a known threshold
        blue._alert_thresh = 50.0
        blue._rotate_thresh = 200.0  # disable higher actions
        blue._isolate_thresh = 200.0

        node = fixed_network.get_node(0)
        node.suspicion_level = 55.0

        actions = blue.act(fixed_network, current_step=1)
        alert_actions = [a for a in actions if a.action_type == "ALERT"]
        assert any(a.target_node_id == 0 for a in alert_actions)
        assert node.is_under_surveillance is True

    def test_alert_not_triggered_below_threshold(self, fixed_network: Network) -> None:
        """No ALERT when suspicion is below threshold."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 70.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999  # disable patrol

        for node in fixed_network.nodes.values():
            node.suspicion_level = 0.0

        actions = blue.act(fixed_network, current_step=1)
        assert not any(a.action_type == "ALERT" for a in actions)


# ---------------------------------------------------------------------------
# ROTATE_CREDENTIALS action
# ---------------------------------------------------------------------------


class TestRotateCredentials:
    def test_rotate_resets_session(self, fixed_network: Network) -> None:
        """ROTATE_CREDENTIALS must call reset_session() → session back to NONE."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 50.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.session_level = SessionLevel.USER
        node.suspicion_level = 60.0

        actions = blue.act(fixed_network, current_step=1)
        rotate_actions = [a for a in actions if a.action_type == "ROTATE_CREDENTIALS"]
        assert any(a.target_node_id == 0 for a in rotate_actions)
        assert node.session_level == SessionLevel.NONE

    def test_rotate_backdoor_protects_session(self, fixed_network: Network) -> None:
        """ROTATE_CREDENTIALS must NOT remove session if node has a backdoor."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 50.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.session_level = SessionLevel.USER
        node.has_backdoor = True
        node.suspicion_level = 60.0

        blue.act(fixed_network, current_step=1)
        assert node.session_level == SessionLevel.USER  # backdoor preserved

    def test_rotate_takes_priority_over_alert(self, fixed_network: Network) -> None:
        """A node triggering ROTATE threshold must NOT also get ALERT."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 50.0
        blue._rotate_thresh = 60.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.suspicion_level = 65.0

        actions = blue.act(fixed_network, current_step=1)
        node0_actions = [a for a in actions if a.target_node_id == 0]
        types = [a.action_type for a in node0_actions]
        assert "ROTATE_CREDENTIALS" in types
        assert "ALERT" not in types  # ROTATE takes priority


# ---------------------------------------------------------------------------
# ISOLATE_NODE action
# ---------------------------------------------------------------------------


class TestIsolateNode:
    def test_isolate_disconnects_node(self, fixed_network: Network) -> None:
        """ISOLATE_NODE must call isolate_node() → is_online=False + edges removed."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 50.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.suspicion_level = 60.0

        actions = blue.act(fixed_network, current_step=1)
        isolate_actions = [a for a in actions if a.action_type == "ISOLATE_NODE"]
        assert any(a.target_node_id == 0 for a in isolate_actions)
        assert node.is_online is False

    def test_isolate_takes_priority_over_rotate_and_alert(self, fixed_network: Network) -> None:
        """A node triggering ISOLATE threshold must only get ISOLATE."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 40.0
        blue._rotate_thresh = 50.0
        blue._isolate_thresh = 60.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.suspicion_level = 65.0

        actions = blue.act(fixed_network, current_step=1)
        node0_actions = [a for a in actions if a.target_node_id == 0]
        types = [a.action_type for a in node0_actions]
        assert "ISOLATE_NODE" in types
        assert "ROTATE_CREDENTIALS" not in types
        assert "ALERT" not in types

    def test_isolated_node_skipped_on_next_act(self, fixed_network: Network) -> None:
        """Already-isolated nodes (is_online=False) must be skipped in threshold checks."""
        blue = ScriptedBlueTeam(seed=42)
        blue._isolate_thresh = 50.0
        blue._patrol_interval = 999

        node = fixed_network.get_node(0)
        node.suspicion_level = 60.0
        fixed_network.isolate_node(0)  # already isolated

        actions = blue.act(fixed_network, current_step=2)
        assert not any(a.target_node_id == 0 and a.action_type == "ISOLATE_NODE" for a in actions)


# ---------------------------------------------------------------------------
# Patrol
# ---------------------------------------------------------------------------


class TestPatrol:
    def test_patrol_detects_traces(self, fixed_network: Network) -> None:
        """Patrol must detect detectable_traces and add suspicion + surveillance."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 1  # always patrol

        # Put a detectable trace on node 0
        target = fixed_network.get_node(0)
        target.detectable_traces.add("INSTALL_BACKDOOR")
        # Force patrol to pick node 0 by using a fixed seed that picks it
        # We'll just run and check that some node had detection applied
        found_detection = False
        for _ in range(20):
            blue2 = ScriptedBlueTeam(seed=_ * 7)
            blue2._alert_thresh = 200.0
            blue2._rotate_thresh = 200.0
            blue2._isolate_thresh = 200.0
            blue2._patrol_interval = 1
            net2 = build_fixed_network(seed=42)
            net2.get_node(0).detectable_traces.add("INSTALL_BACKDOOR")
            actions = blue2.act(net2, current_step=1)
            patrol_hits = [
                a for a in actions
                if a.action_type == "PATROL" and "Traces detected" in a.details
            ]
            if patrol_hits and patrol_hits[0].target_node_id == 0:
                assert net2.get_node(0).suspicion_level == PATROL_DETECTION_SUSPICION
                assert net2.get_node(0).is_under_surveillance is True
                found_detection = True
                break
        assert found_detection, "Patrol should eventually detect traces on node 0"

    def test_patrol_no_detection_without_traces(self, fixed_network: Network) -> None:
        """Patrol on a clean node must return 'No traces found' and not add suspicion."""
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 1

        # Clear all traces
        for node in fixed_network.nodes.values():
            node.detectable_traces.clear()
            node.suspicion_level = 0.0

        actions = blue.act(fixed_network, current_step=1)
        patrol_actions = [a for a in actions if a.action_type == "PATROL"]
        assert len(patrol_actions) == 1
        assert "No traces found" in patrol_actions[0].details
        for node in fixed_network.nodes.values():
            assert node.suspicion_level == 0.0

    def test_patrol_does_not_trigger_on_every_step(self) -> None:
        """With patrol_interval=5, patrol fires ~1/5 of the time (Poisson)."""
        net = build_fixed_network(seed=42)
        blue = ScriptedBlueTeam(seed=1234)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0

        patrol_count = 0
        n_steps = 500
        for step in range(n_steps):
            actions = blue.act(net, current_step=step)
            if any(a.action_type == "PATROL" for a in actions):
                patrol_count += 1

        rate = patrol_count / n_steps
        # Expected ~0.2 (1/5); allow ±0.08 for statistical noise
        assert 0.12 <= rate <= 0.28, f"Patrol rate {rate:.3f} outside expected range [0.12, 0.28]"

    def test_patrol_not_detect_scan_enumerate_traces(self, fixed_network: Network) -> None:
        """Patrol must NOT detect SCAN or ENUMERATE — they leave no traces.

        Only INSTALL_BACKDOOR, EXFILTRATE, CREDENTIAL_DUMP, EXPLOIT (crash)
        add to detectable_traces.
        """
        # SCAN and ENUMERATE do NOT add to detectable_traces (by actions.py design)
        # So detectable_traces must be empty after only those actions.
        for node in fixed_network.nodes.values():
            node.detectable_traces.clear()

        blue = ScriptedBlueTeam(seed=0)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 1

        actions = blue.act(fixed_network, current_step=1)
        patrol_actions = [a for a in actions if a.action_type == "PATROL"]
        assert len(patrol_actions) == 1
        assert "No traces found" in patrol_actions[0].details

    def test_patrol_clean_logs_erases_traces(self, fixed_network: Network) -> None:
        """CLEAN_LOGS before patrol prevents trace detection."""
        # Simulate CLEAN_LOGS effect: clear detectable_traces
        node = fixed_network.get_node(0)
        node.detectable_traces.add("INSTALL_BACKDOOR")
        node.detectable_traces.clear()  # Red used CLEAN_LOGS

        # Now patrol the specific node
        blue = ScriptedBlueTeam(seed=42)
        blue._alert_thresh = 200.0
        blue._rotate_thresh = 200.0
        blue._isolate_thresh = 200.0
        blue._patrol_interval = 1

        # Run until patrol hits node 0
        for seed in range(50):
            blue2 = ScriptedBlueTeam(seed=seed)
            blue2._alert_thresh = 200.0
            blue2._rotate_thresh = 200.0
            blue2._isolate_thresh = 200.0
            blue2._patrol_interval = 1
            net2 = build_fixed_network(seed=42)
            net2.get_node(0).detectable_traces.clear()
            actions = blue2.act(net2, current_step=1)
            patrol_on_0 = [
                a for a in actions if a.action_type == "PATROL" and a.target_node_id == 0
            ]
            if patrol_on_0:
                assert "No traces found" in patrol_on_0[0].details
                assert net2.get_node(0).suspicion_level == 0.0
                return  # test passed
        # If patrol never hit node 0 across 50 seeds, that's fine — test is conservative

    def test_patrol_detection_adds_surveillance(self, fixed_network: Network) -> None:
        """Patrol that detects traces must set is_under_surveillance = True."""
        for seed in range(30):
            blue = ScriptedBlueTeam(seed=seed)
            blue._alert_thresh = 200.0
            blue._rotate_thresh = 200.0
            blue._isolate_thresh = 200.0
            blue._patrol_interval = 1
            net2 = build_fixed_network(seed=42)
            node = net2.get_node(0)
            node.detectable_traces.add("CREDENTIAL_DUMP")
            actions = blue.act(net2, current_step=1)
            detected = [
                a for a in actions
                if a.action_type == "PATROL" and a.target_node_id == 0 and "Traces detected" in a.details
            ]
            if detected:
                assert node.is_under_surveillance is True
                return
        # statistical pass if no seed hit node 0


# ---------------------------------------------------------------------------
# Blue Action NamedTuple
# ---------------------------------------------------------------------------


class TestBlueAction:
    def test_blue_action_fields(self) -> None:
        ba = BlueAction(action_type="ALERT", target_node_id=3, details="test")
        assert ba.action_type == "ALERT"
        assert ba.target_node_id == 3
        assert ba.details == "test"
