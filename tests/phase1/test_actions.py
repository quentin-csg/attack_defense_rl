"""Tests for action types, encoding/decoding, and execution logic."""

import random

from src.config import (
    MAX_NODES,
    REWARD_EXFILTRATE,
    REWARD_NEW_NODE_COMPROMISED,
    REWARD_NEW_NODE_DISCOVERED,
    REWARD_ROOT_OBTAINED,
)
from src.environment.actions import (
    ActionType,
    decode_action,
    encode_action,
    execute_action,
)
from src.environment.network import Network
from src.environment.node import DiscoveryLevel, SessionLevel


class TestActionEncoding:
    def test_encode_decode_roundtrip(self) -> None:
        for at in ActionType:
            for node in [0, 1, 10, MAX_NODES - 1]:
                encoded = encode_action(at, node)
                decoded_type, decoded_node = decode_action(encoded)
                assert decoded_type == at
                assert decoded_node == node

    def test_action_space_size(self) -> None:
        total = len(ActionType) * MAX_NODES
        assert total == 14 * 50  # 700


class TestScan:
    def test_scan_discovers_neighbors(self, small_network: Network) -> None:
        rng = random.Random(42)
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED

        result = execute_action(ActionType.SCAN, 0, small_network, 1, rng, 0)
        assert result.success is True
        assert result.reward == REWARD_NEW_NODE_DISCOVERED  # 1 neighbor (node 1)
        assert small_network.get_node(1).discovery_level == DiscoveryLevel.DISCOVERED

    def test_scan_adds_suspicion(self, small_network: Network) -> None:
        rng = random.Random(42)
        small_network.get_node(0).session_level = SessionLevel.USER
        execute_action(ActionType.SCAN, 0, small_network, 1, rng, 0)
        assert small_network.get_node(0).suspicion_level > 0.0

    def test_scan_already_discovered(self, small_network: Network) -> None:
        rng = random.Random(42)
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED
        result = execute_action(ActionType.SCAN, 0, small_network, 1, rng, 0)
        assert result.success is False
        assert result.reward == 0.0


class TestEnumerate:
    def test_enumerate_reveals_vulns(self, small_network: Network) -> None:
        rng = random.Random(42)
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED
        result = execute_action(ActionType.ENUMERATE, 1, small_network, 1, rng, 0)
        assert result.success is True
        assert small_network.get_node(1).discovery_level == DiscoveryLevel.ENUMERATED

    def test_enumerate_aggressive_more_suspicion(self, small_network: Network) -> None:
        rng = random.Random(42)
        n1 = small_network.get_node(1)
        n1.discovery_level = DiscoveryLevel.DISCOVERED
        execute_action(ActionType.ENUMERATE_AGGRESSIVE, 1, small_network, 1, rng, 0)
        susp_aggressive = n1.suspicion_level

        # Compare with normal enumerate on a fresh node
        n2 = small_network.get_node(2)
        n2.discovery_level = DiscoveryLevel.DISCOVERED
        execute_action(ActionType.ENUMERATE, 2, small_network, 1, rng, 0)
        susp_normal = n2.suspicion_level

        assert susp_aggressive > susp_normal


class TestExploit:
    def test_exploit_success(self, small_network: Network) -> None:
        """With a fixed seed, exploit should succeed (80% base rate)."""
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        # Try multiple seeds to find one that succeeds
        for seed in range(100):
            rng = random.Random(seed)
            small_network.get_node(1).session_level = SessionLevel.NONE
            result = execute_action(ActionType.EXPLOIT, 1, small_network, 1, rng, 0)
            if result.success:
                assert small_network.get_node(1).session_level == SessionLevel.USER
                assert result.reward == REWARD_NEW_NODE_COMPROMISED
                return
        raise AssertionError("No seed produced a successful exploit in 100 tries")

    def test_exploit_probabilities(self, small_network: Network) -> None:
        """Over many trials, exploit success rate should be roughly correct."""
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        successes = 0
        trials = 1000
        for seed in range(trials):
            rng = random.Random(seed)
            small_network.get_node(1).session_level = SessionLevel.NONE
            result = execute_action(ActionType.EXPLOIT, 1, small_network, 1, rng, 0)
            if result.success:
                successes += 1
        rate = successes / trials
        # sqli_basic has 0.85 success, crash is 0.05 -> effective ~0.80
        assert 0.65 < rate < 0.95, f"Exploit rate {rate} outside expected range"

    def test_exploit_no_vuln(self, small_network: Network) -> None:
        """Node with no exploitable vuln should fail."""
        node = small_network.get_node(3)  # only has privesc_suid
        node.discovery_level = DiscoveryLevel.ENUMERATED
        rng = random.Random(42)
        result = execute_action(ActionType.EXPLOIT, 3, small_network, 1, rng, 0)
        assert result.success is False


class TestBruteForce:
    def test_brute_force_with_weak_creds_eventually_succeeds(self, small_network: Network) -> None:
        """Node 2 has weak_credentials — brute force should succeed within 100 seeds."""
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        for seed in range(100):
            rng = random.Random(seed)
            small_network.get_node(2).session_level = SessionLevel.NONE
            result = execute_action(ActionType.BRUTE_FORCE, 2, small_network, 1, rng, 0)
            if result.success:
                assert small_network.get_node(2).session_level == SessionLevel.USER
                return
        raise AssertionError("No seed produced brute force success in 100 tries")

    def test_brute_force_no_weak_creds(self, small_network: Network) -> None:
        small_network.get_node(0).discovery_level = DiscoveryLevel.DISCOVERED
        rng = random.Random(42)
        result = execute_action(ActionType.BRUTE_FORCE, 0, small_network, 1, rng, 0)
        assert result.success is False


class TestPrivesc:
    def test_privesc_success(self, small_network: Network) -> None:
        small_network.get_node(2).session_level = SessionLevel.USER
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        # Try seeds to find success
        for seed in range(100):
            rng = random.Random(seed)
            small_network.get_node(2).session_level = SessionLevel.USER
            result = execute_action(ActionType.PRIVESC, 2, small_network, 1, rng, 0)
            if result.success:
                assert small_network.get_node(2).session_level == SessionLevel.ROOT
                assert result.reward == REWARD_ROOT_OBTAINED
                return
        raise AssertionError("No seed produced successful privesc")

    def test_privesc_no_vuln(self, small_network: Network) -> None:
        small_network.get_node(1).session_level = SessionLevel.USER
        rng = random.Random(42)
        result = execute_action(ActionType.PRIVESC, 1, small_network, 1, rng, 0)
        assert result.success is False


class TestExfiltrate:
    def test_exfiltrate_with_loot(self, small_network: Network) -> None:
        small_network.get_node(4).session_level = SessionLevel.ROOT
        rng = random.Random(42)
        result = execute_action(ActionType.EXFILTRATE, 4, small_network, 1, rng, 0)
        assert result.success is True
        assert result.reward == REWARD_EXFILTRATE

    def test_exfiltrate_no_loot(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.ROOT
        rng = random.Random(42)
        result = execute_action(ActionType.EXFILTRATE, 0, small_network, 1, rng, 0)
        assert result.success is False
        assert result.reward == 0.0


class TestCleanLogs:
    def test_clean_logs_reduces_suspicion(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.session_level = SessionLevel.ROOT
        node.add_suspicion(50.0)
        rng = random.Random(42)
        result = execute_action(ActionType.CLEAN_LOGS, 0, small_network, 10, rng, 0)
        assert result.success is True
        assert node.suspicion_level < 50.0

    def test_clean_logs_diminishing_returns(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.session_level = SessionLevel.ROOT
        rng = random.Random(42)

        reductions = []
        for i in range(4):
            node.add_suspicion(50.0)
            before = node.suspicion_level
            execute_action(ActionType.CLEAN_LOGS, 0, small_network, i * 3, rng, 0)
            after = node.suspicion_level
            reductions.append(before - after)

        # Each reduction should be less than the previous
        for i in range(1, len(reductions)):
            assert reductions[i] <= reductions[i - 1]

    def test_clean_logs_cooldown(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.session_level = SessionLevel.ROOT
        node.add_suspicion(50.0)
        rng = random.Random(42)

        # First clean at step 10
        result1 = execute_action(ActionType.CLEAN_LOGS, 0, small_network, 10, rng, 0)
        assert result1.success is True

        # Second clean at step 11 (too soon, cooldown = 1)
        result2 = execute_action(ActionType.CLEAN_LOGS, 0, small_network, 11, rng, 0)
        assert result2.success is False

        # Third clean at step 12 (cooldown passed)
        result3 = execute_action(ActionType.CLEAN_LOGS, 0, small_network, 12, rng, 0)
        assert result3.success is True

    def test_clean_logs_clears_traces(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.session_level = SessionLevel.ROOT
        node.detectable_traces.add("INSTALL_BACKDOOR")
        rng = random.Random(42)
        execute_action(ActionType.CLEAN_LOGS, 0, small_network, 10, rng, 0)
        assert len(node.detectable_traces) == 0


class TestWait:
    def test_wait_reduces_suspicion(self, small_network: Network) -> None:
        node = small_network.get_node(0)
        node.add_suspicion(50.0)
        rng = random.Random(42)
        execute_action(ActionType.WAIT, 0, small_network, 1, rng, 0)
        assert node.suspicion_level < 50.0

    def test_wait_floor(self, small_network: Network) -> None:
        """WAIT should not reduce suspicion below max_historical / 2."""
        node = small_network.get_node(0)
        node.add_suspicion(60.0)  # max_historical = 60
        rng = random.Random(42)
        # Many WAITs
        for step in range(50):
            execute_action(ActionType.WAIT, 0, small_network, step, rng, 0)
        assert node.suspicion_level >= 30.0  # floor = 60/2


class TestTunnel:
    def test_tunnel_halves_suspicion(self, small_network: Network) -> None:
        rng = random.Random(42)
        node = small_network.get_node(0)
        node.session_level = SessionLevel.USER
        node.discovery_level = DiscoveryLevel.DISCOVERED

        # Install tunnel
        execute_action(ActionType.TUNNEL, 0, small_network, 1, rng, 0)
        assert node.has_tunnel is True

        # Now actions on this node should generate less suspicion
        susp_before = node.suspicion_level
        execute_action(ActionType.SCAN, 0, small_network, 2, rng, 0)
        susp_with_tunnel = node.suspicion_level - susp_before

        # Compare with a node without tunnel
        node2 = small_network.get_node(1)
        node2.session_level = SessionLevel.USER
        susp_before2 = node2.suspicion_level
        execute_action(ActionType.SCAN, 1, small_network, 3, rng, 0)
        susp_without_tunnel = node2.suspicion_level - susp_before2

        assert susp_with_tunnel < susp_without_tunnel


class TestInstallBackdoor:
    def test_install_backdoor(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.USER
        rng = random.Random(42)
        result = execute_action(ActionType.INSTALL_BACKDOOR, 0, small_network, 1, rng, 0)
        assert result.success is True
        assert small_network.get_node(0).has_backdoor is True
        assert "INSTALL_BACKDOOR" in small_network.get_node(0).detectable_traces


class TestCredentialDump:
    def test_credential_dump(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.USER
        rng = random.Random(42)
        result = execute_action(ActionType.CREDENTIAL_DUMP, 0, small_network, 1, rng, 0)
        assert result.success is True
        assert "CREDENTIAL_DUMP" in small_network.get_node(0).detectable_traces


class TestPivot:
    def test_pivot_accesses_discovered_node(self, small_network: Network) -> None:
        """PIVOT targets DISCOVERED (not UNKNOWN) nodes to avoid FoW leaks."""
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        rng = random.Random(42)
        # Node 2 is discovered, adjacent to 1 which is adjacent to 0 (compromised)
        result = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, 0)
        assert result.success is True


class TestLateralMove:
    def test_lateral_move(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED
        rng = random.Random(42)
        result = execute_action(
            ActionType.LATERAL_MOVE, 1, small_network, 1, rng, 0, has_dumped_creds=True
        )
        assert result.success is True
        assert small_network.get_node(1).session_level == SessionLevel.USER

    def test_lateral_move_no_creds_fails(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED
        rng = random.Random(42)
        result = execute_action(
            ActionType.LATERAL_MOVE, 1, small_network, 1, rng, 0, has_dumped_creds=False
        )
        assert result.success is False


class TestExploitCrash:
    def test_exploit_crash_adds_trace(self, small_network: Network) -> None:
        """Exploit crash outcome should add EXPLOIT_CRASH to detectable_traces."""
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        # Find a seed that triggers crash (roll < 0.05)
        for seed in range(1000):
            rng = random.Random(seed)
            small_network.get_node(1).session_level = SessionLevel.NONE
            small_network.get_node(1).detectable_traces.clear()
            result = execute_action(ActionType.EXPLOIT, 1, small_network, 1, rng, 0)
            if result.info.get("outcome") == "crash":
                assert "EXPLOIT_CRASH" in small_network.get_node(1).detectable_traces
                return
        raise AssertionError("No crash outcome in 1000 seeds")


class TestExfiltrateRequiresRoot:
    def test_exfiltrate_with_user_fails(self, small_network: Network) -> None:
        """EXFILTRATE requires ROOT — USER session should fail."""
        small_network.get_node(4).session_level = SessionLevel.USER
        rng = random.Random(42)
        result = execute_action(ActionType.EXFILTRATE, 4, small_network, 1, rng, 0)
        assert result.success is False
        assert result.reward == 0.0


class TestHandlerDispatchCoverage:
    def test_all_action_types_have_handlers(self, small_network: Network) -> None:
        """execute_action must dispatch every ActionType without KeyError."""
        from src.environment.actions import N_ACTION_TYPES

        assert len(ActionType) == N_ACTION_TYPES
        # Calling execute_action for each type should not raise KeyError
        # (the coverage assert inside execute_action verifies the dict is complete)
        rng = random.Random(42)
        small_network.get_node(0).session_level = SessionLevel.USER
        for at in ActionType:
            try:
                execute_action(at, 0, small_network, 1, rng, 0)
            except KeyError as e:
                raise AssertionError(f"No handler for {at}: {e}") from e


class TestNActionTypesConsistency:
    def test_n_action_types_matches_enum(self) -> None:
        from src.config import MAX_NODES
        from src.environment.actions import N_ACTION_TYPES

        assert len(ActionType) == N_ACTION_TYPES
        assert len(ActionType) * MAX_NODES == N_ACTION_TYPES * MAX_NODES


class TestPivotBugFix:
    """Tests for Bug 2 fix: PIVOT must grant a USER session on the target node."""

    def test_pivot_grants_user_session(self, small_network: Network) -> None:
        """PIVOT must set session_level=USER on the target node."""
        rng = random.Random(42)
        # Set node 2 as DISCOVERED with no session
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(2).session_level = SessionLevel.NONE
        # Agent at node 0, node 1 has a session (intermediary)
        small_network.get_node(1).session_level = SessionLevel.USER
        result = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert result.success is True
        assert small_network.get_node(2).session_level == SessionLevel.USER

    def test_pivot_sets_discovery_to_enumerated(self, small_network: Network) -> None:
        """PIVOT must elevate the target node's discovery to ENUMERATED."""
        rng = random.Random(42)
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(2).session_level = SessionLevel.NONE
        small_network.get_node(1).session_level = SessionLevel.USER
        execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert small_network.get_node(2).discovery_level == DiscoveryLevel.ENUMERATED

    def test_pivot_reward_is_positive(self, small_network: Network) -> None:
        """PIVOT must return a positive reward (REWARD_NEW_NODE_COMPROMISED)."""
        rng = random.Random(42)
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(2).session_level = SessionLevel.NONE
        small_network.get_node(1).session_level = SessionLevel.USER
        result = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert result.reward == REWARD_NEW_NODE_COMPROMISED


class TestPivotGuards:
    """Tests for the defence-in-depth guards added to _execute_pivot."""

    def test_pivot_fails_if_already_compromised(self, small_network: Network) -> None:
        """PIVOT on an already-compromised node must return success=False and reward=0."""
        rng = random.Random(42)
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(2).session_level = SessionLevel.USER  # already compromised
        result = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert result.success is False
        assert result.reward == 0.0

    def test_pivot_does_not_downgrade_root(self, small_network: Network) -> None:
        """PIVOT on a ROOT node must fail — not downgrade the session to USER."""
        rng = random.Random(42)
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        small_network.get_node(2).session_level = SessionLevel.ROOT
        result = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert result.success is False
        # Session must stay ROOT, not be downgraded
        assert small_network.get_node(2).session_level == SessionLevel.ROOT

    def test_pivot_no_duplicate_reward(self, small_network: Network) -> None:
        """Two consecutive PIVOTs on the same node: second must give 0 reward."""
        rng = random.Random(42)
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        small_network.get_node(2).session_level = SessionLevel.NONE
        small_network.get_node(1).session_level = SessionLevel.USER
        # First PIVOT succeeds
        r1 = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert r1.success is True
        assert r1.reward == REWARD_NEW_NODE_COMPROMISED
        # Second PIVOT on same node fails (node now has session)
        r2 = execute_action(ActionType.PIVOT, 2, small_network, 1, rng, agent_position=0)
        assert r2.success is False
        assert r2.reward == 0.0
