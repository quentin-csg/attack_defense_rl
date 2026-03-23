"""Tests for action masking logic."""

import numpy as np

from src.config import MAX_NODES
from src.environment.action_mask import N_ACTION_TYPES, compute_action_mask
from src.environment.actions import ActionType
from src.environment.network import Network
from src.environment.node import DiscoveryLevel, SessionLevel


class TestActionMask:
    def test_mask_shape(self, small_network: Network) -> None:
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask.shape == (N_ACTION_TYPES * MAX_NODES,)
        assert mask.dtype == bool

    def test_wait_always_valid(self, small_network: Network) -> None:
        """WAIT is ALWAYS valid — mask must never be all-zero."""
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        wait_idx = ActionType.WAIT * MAX_NODES + 0
        assert mask[wait_idx] is np.True_
        assert mask.any()

    def test_scan_requires_session(self, small_network: Network) -> None:
        """SCAN only valid on nodes where Red has a session."""
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        # No sessions yet -> no SCAN valid (except WAIT)
        scan_mask = mask[ActionType.SCAN * MAX_NODES : (ActionType.SCAN + 1) * MAX_NODES]
        assert not scan_mask.any()

        # Give session on node 0
        small_network.get_node(0).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.SCAN * MAX_NODES + 0] is np.True_

    def test_enumerate_requires_discovered(self, small_network: Network) -> None:
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.ENUMERATE * MAX_NODES + 1] is np.True_
        assert mask[ActionType.ENUMERATE * MAX_NODES + 2] is np.False_  # not discovered

    def test_exploit_requires_enumerated_and_vuln(self, small_network: Network) -> None:
        # Node 1 has sqli_basic (exploitable)
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.EXPLOIT * MAX_NODES + 1] is np.True_

    def test_exploit_blocked_if_already_compromised(self, small_network: Network) -> None:
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        small_network.get_node(1).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.EXPLOIT * MAX_NODES + 1] is np.False_

    def test_privesc_requires_user_enumerated_and_vuln(self, small_network: Network) -> None:
        # Node 2 has privesc_kernel; must also be ENUMERATED
        small_network.get_node(2).session_level = SessionLevel.USER
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PRIVESC * MAX_NODES + 2] is np.True_

    def test_privesc_blocked_if_not_enumerated(self, small_network: Network) -> None:
        # USER session but only DISCOVERED (not ENUMERATED) — can't know the vuln
        small_network.get_node(2).session_level = SessionLevel.USER
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PRIVESC * MAX_NODES + 2] is np.False_

    def test_exfiltrate_requires_root_and_loot(self, small_network: Network) -> None:
        # Node 4 has loot
        small_network.get_node(4).session_level = SessionLevel.ROOT
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.EXFILTRATE * MAX_NODES + 4] is np.True_
        # Node without loot
        assert mask[ActionType.EXFILTRATE * MAX_NODES + 0] is np.False_

    def test_clean_logs_requires_root(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.ROOT
        mask = compute_action_mask(small_network, current_step=10, agent_position=0)
        assert mask[ActionType.CLEAN_LOGS * MAX_NODES + 0] is np.True_

    def test_lateral_move_requires_dumped_creds(self, small_network: Network) -> None:
        small_network.get_node(0).session_level = SessionLevel.USER
        small_network.get_node(1).discovery_level = DiscoveryLevel.DISCOVERED

        # Without creds
        mask = compute_action_mask(
            small_network, current_step=0, agent_position=0, has_dumped_creds=False
        )
        assert mask[ActionType.LATERAL_MOVE * MAX_NODES + 1] is np.False_

        # With creds
        mask = compute_action_mask(
            small_network, current_step=0, agent_position=0, has_dumped_creds=True
        )
        assert mask[ActionType.LATERAL_MOVE * MAX_NODES + 1] is np.True_

    def test_offline_node_blocks_all(self, small_network: Network) -> None:
        """Offline nodes should not allow any action."""
        small_network.get_node(2).is_online = False
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        for at in ActionType:
            if at == ActionType.WAIT:
                continue
            assert mask[at * MAX_NODES + 2] is np.False_

    def test_padding_slots_all_false(self, small_network: Network) -> None:
        """Nodes 5-49 are padding (don't exist) and must all be masked False."""
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        for node_id in range(5, MAX_NODES):
            for at in ActionType:
                assert not mask[at * MAX_NODES + node_id], (
                    f"Padding slot {node_id} should be False for {at}"
                )

    def test_wait_valid_at_agent_position(self, small_network: Network) -> None:
        """WAIT must be valid at the agent's current position."""
        # agent_position=0, which is also entry_node_id in small_network
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.WAIT * MAX_NODES + 0] is np.True_

    def test_wait_valid_on_any_agent_position(self, small_network: Network) -> None:
        """WAIT must be valid wherever the agent currently is — not just the entry node.

        Critical for Phase 4: if Blue Team ROTATE_CREDENTIALS strips all sessions,
        the mask must still have at least one valid action (WAIT) at the agent's position.
        """
        for agent_pos in range(5):  # all real nodes in small_network
            mask = compute_action_mask(small_network, current_step=0, agent_position=agent_pos)
            assert mask[ActionType.WAIT * MAX_NODES + agent_pos] is np.True_, (
                f"WAIT must be valid at agent_position={agent_pos}"
            )
            assert mask.any(), f"Mask must never be all-zero (agent_position={agent_pos})"

    def test_install_backdoor_requires_session(self, small_network: Network) -> None:
        """INSTALL_BACKDOOR only valid on nodes with at least USER session."""
        # No session: blocked
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.INSTALL_BACKDOOR * MAX_NODES + 0] is np.False_

        # With session: allowed (and no backdoor yet)
        small_network.get_node(0).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.INSTALL_BACKDOOR * MAX_NODES + 0] is np.True_

        # Already has backdoor: blocked
        small_network.get_node(0).has_backdoor = True
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.INSTALL_BACKDOOR * MAX_NODES + 0] is np.False_

    def test_tunnel_requires_session_no_tunnel(self, small_network: Network) -> None:
        """TUNNEL only valid on nodes with session and no existing tunnel."""
        # No session: blocked
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.TUNNEL * MAX_NODES + 0] is np.False_

        # With session, no tunnel: allowed
        small_network.get_node(0).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.TUNNEL * MAX_NODES + 0] is np.True_

        # Already has tunnel: blocked
        small_network.get_node(0).has_tunnel = True
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.TUNNEL * MAX_NODES + 0] is np.False_

    def test_credential_dump_requires_session(self, small_network: Network) -> None:
        """CREDENTIAL_DUMP only valid on nodes with at least USER session."""
        # No session: blocked
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.CREDENTIAL_DUMP * MAX_NODES + 0] is np.False_

        # With USER session: allowed
        small_network.get_node(0).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.CREDENTIAL_DUMP * MAX_NODES + 0] is np.True_

    def test_pivot_requires_discovered_and_no_session(self, small_network: Network) -> None:
        """PIVOT target must be DISCOVERED and have no session yet."""
        # UNKNOWN node: blocked
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PIVOT * MAX_NODES + 2] is np.False_

        # DISCOVERED but no compromised intermediary: still blocked
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PIVOT * MAX_NODES + 2] is np.False_

        # DISCOVERED + compromised intermediary reachable: allowed
        small_network.get_node(1).session_level = SessionLevel.USER  # intermediary
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PIVOT * MAX_NODES + 2] is np.True_

        # Already has session: blocked
        small_network.get_node(2).session_level = SessionLevel.USER
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.PIVOT * MAX_NODES + 2] is np.False_

    def test_brute_force_requires_weak_credentials(self, small_network: Network) -> None:
        """BRUTE_FORCE only valid on nodes with weak_credentials and no session."""
        # Node 2 has weak credentials; node 1 does not
        small_network.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        # Node 2 has has_weak_credentials=True (from conftest) -> valid
        assert mask[ActionType.BRUTE_FORCE * MAX_NODES + 2] is np.True_
        # Node 1 does not -> blocked
        assert mask[ActionType.BRUTE_FORCE * MAX_NODES + 1] is np.False_

    def test_brute_force_requires_enumerated_not_just_discovered(
        self, small_network: Network
    ) -> None:
        """BRUTE_FORCE must be blocked on DISCOVERED (non-ENUMERATED) nodes even if they
        have weak credentials — otherwise the agent could infer weak_creds via the mask
        before enumerating (fog-of-war leak)."""
        # Node 2 has weak_credentials=True but only DISCOVERED (not ENUMERATED)
        small_network.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.BRUTE_FORCE * MAX_NODES + 2] is np.False_

        # Once ENUMERATED, the action is allowed
        small_network.get_node(2).discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.BRUTE_FORCE * MAX_NODES + 2] is np.True_

    def test_enumerate_blocked_if_already_enumerated(self, small_network: Network) -> None:
        """Re-enumerating an already-ENUMERATED node is wasteful and must be blocked."""
        node = small_network.get_node(1)
        node.discovery_level = DiscoveryLevel.ENUMERATED
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        assert mask[ActionType.ENUMERATE * MAX_NODES + 1] is np.False_
        assert mask[ActionType.ENUMERATE_AGGRESSIVE * MAX_NODES + 1] is np.False_

    def test_credential_dump_blocked_if_already_dumped(self, small_network: Network) -> None:
        """Once creds have been dumped (has_dumped_creds=True), CREDENTIAL_DUMP must be
        masked on all nodes — there is no benefit to dumping again."""
        small_network.get_node(0).session_level = SessionLevel.USER
        # Before dump: allowed
        mask = compute_action_mask(
            small_network, current_step=0, agent_position=0, has_dumped_creds=False
        )
        assert mask[ActionType.CREDENTIAL_DUMP * MAX_NODES + 0] is np.True_

        # After dump: blocked everywhere
        mask = compute_action_mask(
            small_network, current_step=0, agent_position=0, has_dumped_creds=True
        )
        cred_dump_slice = mask[
            ActionType.CREDENTIAL_DUMP * MAX_NODES : (ActionType.CREDENTIAL_DUMP + 1) * MAX_NODES
        ]
        assert not cred_dump_slice.any()
