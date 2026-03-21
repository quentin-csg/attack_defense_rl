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

    def test_wait_uses_entry_node_id(self, small_network: Network) -> None:
        """WAIT must be valid on entry_node_id regardless of which node it is."""
        mask = compute_action_mask(small_network, current_step=0, agent_position=0)
        entry = small_network.entry_node_id
        assert mask[ActionType.WAIT * MAX_NODES + entry] is np.True_
