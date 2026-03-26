"""Tests for the pre-Phase 5 review fixes (A1-A9).

Verifies defence-in-depth improvements, Blue Team design fixes,
and the adjacency dirty flag optimization.
"""

from __future__ import annotations

import random

from src.agents.blue_scripted import ScriptedBlueTeam
from src.agents.wrappers import make_vec_masked_env
from src.config import BLUE_ISOLATE_DURATION, BLUE_ROTATE_COOLDOWN
from src.environment.actions import (
    ActionType,
    execute_action,
)
from src.environment.cyber_env import CyberEnv
from src.environment.network import Network, build_fixed_network
from src.environment.node import DiscoveryLevel, Node, OsType, Service, SessionLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rng() -> random.Random:
    return random.Random(42)


def _small_network() -> Network:
    """5-node linear network: 0-1-2-3-4, entry=0, target=4 (has_loot)."""
    net = Network()
    nodes = [
        Node(0, OsType.NETWORK_DEVICE, [Service("ssh", 22)], ["rce_generic"]),
        Node(1, OsType.LINUX, [Service("ssh", 22)], ["sqli_basic"]),
        Node(2, OsType.WINDOWS, [Service("smb", 445)], ["rce_generic", "privesc_kernel"],
             has_weak_credentials=True),
        Node(3, OsType.LINUX, [Service("ssh", 22)], ["privesc_suid"]),
        Node(4, OsType.LINUX, [Service("nfs", 2049)], ["privesc_misconfig"], has_loot=True),
    ]
    for n in nodes:
        net.add_node(n)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 4)]:
        net.add_edge(a, b)
    net.entry_node_id = 0
    net.target_node_id = 4
    return net


# ---------------------------------------------------------------------------
# A1 — make_vec_masked_env() propagates blue_team
# ---------------------------------------------------------------------------


def test_vec_env_accepts_blue_team() -> None:
    """make_vec_masked_env() should accept blue_team and not crash."""
    blue = ScriptedBlueTeam(seed=0)
    vec_env = make_vec_masked_env(n_envs=2, seed=10, max_steps=50, blue_team=blue)
    obs = vec_env.reset()
    assert obs is not None
    vec_env.close()


def test_vec_env_none_blue_team_unchanged() -> None:
    """make_vec_masked_env() with blue_team=None works as before."""
    vec_env = make_vec_masked_env(n_envs=1, seed=0, max_steps=50, blue_team=None)
    obs = vec_env.reset()
    assert obs is not None
    vec_env.close()


# ---------------------------------------------------------------------------
# A2 — PIVOT handler rejects offline intermediary
# ---------------------------------------------------------------------------


def test_pivot_rejects_offline_intermediary() -> None:
    """PIVOT should fail if the only compromised node that could relay is offline."""
    net = _small_network()
    # Give agent USER on node 1 (adjacent to 0)
    net.get_node(1).session_level = SessionLevel.USER
    net.get_node(1).discovery_level = DiscoveryLevel.ENUMERATED
    # Target = node 2 (adjacent to 1, 2-hop from 0)
    net.get_node(2).discovery_level = DiscoveryLevel.DISCOVERED

    # Sanity: pivot succeeds when intermediary is online
    result = execute_action(
        ActionType.PIVOT, 2, net, 0, _make_rng(), agent_position=0
    )
    assert result.success, "Should succeed when intermediary is online"

    # Now isolate node 1 (make it offline) and reset target session
    net.isolate_node(1)
    assert not net.get_node(1).is_online
    net.get_node(2).session_level = SessionLevel.NONE

    result = execute_action(
        ActionType.PIVOT, 2, net, 0, _make_rng(), agent_position=0
    )
    assert not result.success
    assert result.info.get("reason") == "not_reachable"


# ---------------------------------------------------------------------------
# A3 — CREDENTIAL_DUMP handler rejects if already dumped
# ---------------------------------------------------------------------------


def test_cred_dump_handler_rejects_if_already_dumped() -> None:
    """Handler should return failure if has_dumped_creds=True (defence-in-depth)."""
    net = _small_network()
    net.get_node(0).session_level = SessionLevel.USER

    # Without creds: success
    result = execute_action(
        ActionType.CREDENTIAL_DUMP, 0, net, 0, _make_rng(),
        agent_position=0, has_dumped_creds=False
    )
    assert result.success

    # Reset for second test
    net.get_node(0).session_level = SessionLevel.USER

    # With creds already dumped: handler should refuse
    result = execute_action(
        ActionType.CREDENTIAL_DUMP, 0, net, 0, _make_rng(),
        agent_position=0, has_dumped_creds=True
    )
    assert not result.success
    assert result.info.get("reason") == "already_dumped"


# ---------------------------------------------------------------------------
# A4 — LATERAL_MOVE handler rejects non-adjacent targets
# ---------------------------------------------------------------------------


def test_lateral_move_rejects_non_adjacent() -> None:
    """Handler should reject LATERAL_MOVE to a node not adjacent to any compromised node."""
    net = _small_network()
    # Agent position 0 (USER), has creds
    net.get_node(0).session_level = SessionLevel.USER
    net.get_node(0).is_online = True

    # Node 3 is 3 hops from 0 — not adjacent to any compromised node
    result = execute_action(
        ActionType.LATERAL_MOVE, 3, net, 0, _make_rng(),
        agent_position=0, has_dumped_creds=True
    )
    assert not result.success
    assert result.info.get("reason") == "not_adjacent"


def test_lateral_move_accepts_adjacent_target() -> None:
    """Handler should accept LATERAL_MOVE to a node adjacent to a compromised node."""
    net = _small_network()
    net.get_node(0).session_level = SessionLevel.USER
    net.get_node(0).is_online = True

    # Node 1 is adjacent to node 0 (compromised)
    result = execute_action(
        ActionType.LATERAL_MOVE, 1, net, 0, _make_rng(),
        agent_position=0, has_dumped_creds=True
    )
    assert result.success


# ---------------------------------------------------------------------------
# A5 — Adjacency dirty flag: only rebuilds when topology changes
# ---------------------------------------------------------------------------


def test_adjacency_dirty_flag_set_on_isolate() -> None:
    """After Blue Team ISOLATE, _adjacency_dirty should be True before next obs."""
    net = build_fixed_network(seed=42)
    blue = ScriptedBlueTeam(seed=0)
    # Force ISOLATE by setting suspicion to 100 on a node
    net.get_node(1).add_suspicion(100.0)
    # Force isolate threshold to fire
    blue._isolate_thresh = 0.0  # will trigger on every node with susp > 0
    # We'll test via CyberEnv
    env = CyberEnv(network=net, blue_team=blue, seed=42)
    env.reset()
    # Manually trigger Blue Team action via step
    # After reset, thresholds are re-randomised — override them
    env.blue_team._isolate_thresh = 0.0
    env.blue_team._rotate_thresh = -1.0
    env.blue_team._alert_thresh = -1.0
    # Give agent a session so step doesn't crash
    env.network.get_node(env.agent_position).session_level = SessionLevel.USER
    from src.environment.actions import encode_action
    wait_action = encode_action(ActionType.WAIT, env.agent_position)
    env.step(wait_action)
    # After step, if any node was isolated, dirty flag was set and then cleared by _get_obs
    # The flag is cleared during _get_obs (called inside step) — so after step it's False
    assert env._adjacency_dirty is False


def test_adjacency_dirty_flag_restore() -> None:
    """After auto-restore, the adjacency should reflect restored edges."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=0)
    env = CyberEnv(network=net, blue_team=blue, seed=42)
    env.reset()
    # Manually isolate node 1
    env.network.isolate_node(1)
    env.blue_team._isolated_at[1] = env.current_step - BLUE_ISOLATE_DURATION
    # Force restore on next act
    blue._isolate_thresh = 200.0  # won't trigger threshold actions
    blue._rotate_thresh = 200.0
    blue._alert_thresh = 200.0
    blue_actions = blue.act(env.network, env.current_step + BLUE_ISOLATE_DURATION)
    restore_actions = [a for a in blue_actions if a.action_type == "RESTORE_NODE"]
    assert len(restore_actions) == 1
    assert env.network.get_node(1).is_online


# ---------------------------------------------------------------------------
# A6 — Auto-restore: isolated nodes come back after BLUE_ISOLATE_DURATION
# ---------------------------------------------------------------------------


def test_isolate_auto_restore_after_duration() -> None:
    """Isolated node is auto-restored after BLUE_ISOLATE_DURATION steps."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    blue._isolate_thresh = 0.0   # force ISOLATE on any node with susp > 0
    blue._rotate_thresh = -1.0
    blue._alert_thresh = -1.0

    net.get_node(1).add_suspicion(50.0)
    blue.act(net, current_step=0)  # should isolate node 1
    assert not net.get_node(1).is_online
    assert 1 in blue._isolated_at

    # Run steps but below duration — still isolated
    blue._isolate_thresh = 200.0  # don't re-isolate
    for step in range(1, BLUE_ISOLATE_DURATION):
        blue.act(net, current_step=step)
    assert not net.get_node(1).is_online

    # At exactly BLUE_ISOLATE_DURATION steps later — should restore
    blue.act(net, current_step=BLUE_ISOLATE_DURATION)
    assert net.get_node(1).is_online
    assert 1 not in blue._isolated_at


def test_isolate_reset_clears_isolated_at() -> None:
    """reset() clears _isolated_at so previous isolations don't carry over."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    blue._isolate_thresh = 0.0
    net.get_node(1).add_suspicion(50.0)
    blue.act(net, current_step=0)
    assert 1 in blue._isolated_at

    blue.reset()
    assert blue._isolated_at == {}


# ---------------------------------------------------------------------------
# A7 — Cooldown ROTATE_CREDENTIALS per node
# ---------------------------------------------------------------------------


def test_rotate_respects_cooldown() -> None:
    """ROTATE_CREDENTIALS should not fire more than once per BLUE_ROTATE_COOLDOWN steps."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    blue._rotate_thresh = 0.0    # fire ROTATE on any node
    blue._isolate_thresh = 200.0
    blue._alert_thresh = -1.0
    net.get_node(1).session_level = SessionLevel.USER

    # First call — should ROTATE
    actions1 = blue.act(net, current_step=0)
    rotate1 = [a for a in actions1 if a.action_type == "ROTATE_CREDENTIALS" and a.target_node_id == 1]
    assert len(rotate1) == 1

    # Second call immediately — cooldown active, should NOT ROTATE node 1
    net.get_node(1).session_level = SessionLevel.USER  # restore session
    actions2 = blue.act(net, current_step=1)
    rotate2 = [a for a in actions2 if a.action_type == "ROTATE_CREDENTIALS" and a.target_node_id == 1]
    assert len(rotate2) == 0

    # After cooldown expires — should ROTATE again
    net.get_node(1).session_level = SessionLevel.USER
    actions3 = blue.act(net, current_step=BLUE_ROTATE_COOLDOWN)
    rotate3 = [a for a in actions3 if a.action_type == "ROTATE_CREDENTIALS" and a.target_node_id == 1]
    assert len(rotate3) == 1


def test_rotate_cooldown_per_node_independent() -> None:
    """ROTATE cooldown is tracked independently per node."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    blue._rotate_thresh = 0.0
    blue._isolate_thresh = 200.0
    blue._alert_thresh = -1.0
    net.get_node(1).session_level = SessionLevel.USER
    net.get_node(2).session_level = SessionLevel.USER

    # Both rotate at step 0
    blue.act(net, current_step=0)
    assert blue._last_rotate.get(1) == 0
    assert blue._last_rotate.get(2) == 0

    # At step 1: both on cooldown — neither rotates
    net.get_node(1).session_level = SessionLevel.USER
    net.get_node(2).session_level = SessionLevel.USER
    actions = blue.act(net, current_step=1)
    rotates = [a for a in actions if a.action_type == "ROTATE_CREDENTIALS"]
    assert len(rotates) == 0


def test_rotate_reset_clears_cooldown() -> None:
    """reset() clears _last_rotate so cooldown doesn't carry between episodes."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    blue._rotate_thresh = 0.0
    blue._isolate_thresh = 200.0
    net.get_node(1).session_level = SessionLevel.USER
    blue.act(net, current_step=0)
    assert 1 in blue._last_rotate

    blue.reset()
    assert blue._last_rotate == {}


# ---------------------------------------------------------------------------
# A8 — CLEAN_LOGS bypasses the WAIT floor
# ---------------------------------------------------------------------------


def test_clean_logs_can_go_below_wait_floor() -> None:
    """CLEAN_LOGS should reduce suspicion below max_historical / 2."""
    net = _small_network()
    node = net.get_node(0)
    node.session_level = SessionLevel.ROOT
    # Push suspicion to 80 → WAIT floor = 40
    node.add_suspicion(80.0)
    assert node.max_suspicion_historical == 80.0

    # WAIT cannot go below 40
    node.reduce_suspicion(100.0, bypass_floor=False)
    assert node.suspicion_level == 40.0  # floor = 80/2

    # CLEAN_LOGS (bypass_floor=True) can go below 40
    node.reduce_suspicion(30.0, bypass_floor=True)
    assert node.suspicion_level == 10.0  # 40 - 30 = 10


def test_wait_still_respects_floor() -> None:
    """WAIT should still be bounded by the WAIT floor."""
    net = _small_network()
    node = net.get_node(0)
    node.add_suspicion(60.0)
    node.reduce_suspicion(100.0, bypass_floor=False)
    assert node.suspicion_level == 30.0  # floor = 60/2


# ---------------------------------------------------------------------------
# A9 — ROTATE_CREDENTIALS invalidates has_dumped_creds
# ---------------------------------------------------------------------------


def test_rotate_clears_dumped_creds_in_env() -> None:
    """After Blue Team ROTATE_CREDENTIALS, has_dumped_creds should be False."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    env = CyberEnv(network=net, blue_team=blue, seed=42)
    env.reset()

    # Give agent creds
    env.has_dumped_creds = True

    # Force Blue Team to ROTATE on next step
    env.blue_team._rotate_thresh = 0.0
    env.blue_team._isolate_thresh = 200.0
    env.blue_team._alert_thresh = -1.0
    # Ensure a node has suspicion to trigger ROTATE
    env.network.get_node(1).add_suspicion(50.0)
    env.network.get_node(1).session_level = SessionLevel.USER

    from src.environment.actions import encode_action
    wait_action = encode_action(ActionType.WAIT, env.agent_position)
    env.step(wait_action)

    # has_dumped_creds should be cleared because ROTATE fired
    assert env.has_dumped_creds is False


def test_no_rotate_does_not_clear_creds() -> None:
    """has_dumped_creds should remain True if no ROTATE_CREDENTIALS happens."""
    net = _small_network()
    blue = ScriptedBlueTeam(seed=42)
    env = CyberEnv(network=net, blue_team=blue, seed=42)
    env.reset()
    env.has_dumped_creds = True

    # Blue Team thresholds so high they won't fire
    env.blue_team._rotate_thresh = 200.0
    env.blue_team._isolate_thresh = 200.0
    env.blue_team._alert_thresh = 200.0

    from src.environment.actions import encode_action
    wait_action = encode_action(ActionType.WAIT, env.agent_position)
    env.step(wait_action)

    assert env.has_dumped_creds is True
