"""PCG network generator for Phase 5.

Generates randomised enterprise-topology networks in three sizes:
  - Small  : 10-15 nodes, 2-3 subnets
  - Medium : 25-30 nodes, 4-5 subnets
  - Large  : 50-60 nodes, 7-8 subnets

Each generated network follows a realistic zone structure:
  DMZ → CORPORATE (workstations) → SERVER (internal) → DATACENTER (target)

Usage:
    from src.pcg.generator import NetworkSize, generate_network

    net, meta = generate_network(NetworkSize.SMALL, seed=42)
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)
from enum import Enum
from typing import NamedTuple

import networkx as nx

from src.config import (
    MAX_NODES,
    PCG_LARGE_NODES,
    PCG_LARGE_SUBNETS,
    PCG_MEDIUM_NODES,
    PCG_MEDIUM_SUBNETS,
    PCG_SMALL_NODES,
    PCG_SMALL_SUBNETS,
)
from src.environment.network import Network
from src.environment.node import DiscoveryLevel, Node, OsType, Service, SessionLevel
from src.environment.vulnerability import VULN_REGISTRY, VulnCategory


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class NetworkSize(Enum):
    """Procedural network size preset."""

    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Zone(Enum):
    """Logical zone within the enterprise network."""

    DMZ = "dmz"
    CORPORATE = "corporate"
    SERVER = "server"
    DATACENTER = "datacenter"


@dataclass(frozen=True)
class NetworkMeta:
    """Metadata about a generated network (read-only)."""

    size: NetworkSize
    n_nodes: int
    n_subnets: int
    min_hops: int               # shortest path entry_node_id → target_node_id
    difficulty_score: float
    recommended_max_steps: int
    zone_map: dict[int, Zone]   # node_id → Zone


# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

# OS probability weights per zone: (LINUX, WINDOWS, NETWORK_DEVICE)
_OS_WEIGHTS: dict[Zone, tuple[float, float, float]] = {
    Zone.DMZ:        (0.55, 0.10, 0.35),
    Zone.CORPORATE:  (0.30, 0.60, 0.10),
    Zone.SERVER:     (0.70, 0.20, 0.10),
    Zone.DATACENTER: (0.80, 0.20, 0.00),
}

# Vulnerability pools per zone (drawn from VULN_REGISTRY)
_ZONE_VULN_POOLS: dict[Zone, list[str]] = {
    Zone.DMZ: [
        "rce_log4shell", "rce_shellshock", "ssrf_metadata",
        "sqli_union_bypass", "sqli_blind_time",
    ],
    Zone.CORPORATE: [
        "rce_eternal_blue", "privesc_print_nightmare",
        "sqli_union_bypass", "rce_log4shell",
    ],
    Zone.SERVER: [
        "privesc_docker_escape", "rce_log4shell", "sqli_blind_time",
        "sqli_union_bypass", "ssrf_metadata",
    ],
    Zone.DATACENTER: [
        "privesc_sudo_baron", "privesc_dirty_cow",
        "privesc_kernel", "privesc_suid",
    ],
}

# Service pools by OS type (name, port)
_SERVICES_BY_OS: dict[OsType, list[tuple[str, int]]] = {
    OsType.LINUX: [
        ("ssh", 22), ("http", 80), ("https", 443),
        ("ftp", 21), ("smtp", 25), ("nfs", 2049),
    ],
    OsType.WINDOWS: [
        ("smb", 445), ("rdp", 3389), ("http", 80),
        ("https", 443), ("winrm", 5985),
    ],
    OsType.NETWORK_DEVICE: [
        ("ssh", 22), ("telnet", 23), ("snmp", 161), ("https", 443),
    ],
}

# Weak credentials probability per zone
_WEAK_CREDS_PROB: dict[Zone, float] = {
    Zone.DMZ:        0.10,
    Zone.CORPORATE:  0.25,
    Zone.SERVER:     0.08,
    Zone.DATACENTER: 0.05,
}

# Loot probability for non-target nodes per zone
_LOOT_PROB: dict[Zone, float] = {
    Zone.DMZ:        0.00,
    Zone.CORPORATE:  0.05,
    Zone.SERVER:     0.15,
    Zone.DATACENTER: 1.00,  # all DC nodes carry loot (high-value zone)
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_network(
    size: NetworkSize,
    seed: int | None = None,
) -> tuple[Network, NetworkMeta]:
    """Generate a procedural enterprise network.

    Retries up to 10 times if the generated topology fails validation,
    then falls back to a minimal guaranteed-solvable network.

    Args:
        size: Desired network size preset.
        seed: Random seed for reproducibility. None = non-deterministic.

    Returns:
        (network, meta) tuple where meta contains difficulty info.
    """
    rng = random.Random(seed)

    for attempt in range(10):
        attempt_seed = None if seed is None else seed + attempt
        try:
            net, meta = _generate_attempt(size, random.Random(attempt_seed))
            # Import here to avoid circular dep at module load time
            from src.pcg.difficulty import is_solvable
            if is_solvable(net):
                return net, meta
        except Exception:  # noqa: BLE001 — retry on any generation failure
            pass

    # All 10 attempts failed — log diagnostic info and use fallback
    logger.warning(
        "generate_network: all 10 attempts failed for size=%s seed=%s — using fallback linear chain",
        size.value,
        seed,
    )
    return _generate_fallback(size, rng)


# ---------------------------------------------------------------------------
# Internal generation logic
# ---------------------------------------------------------------------------


def _generate_attempt(
    size: NetworkSize,
    rng: random.Random,
) -> tuple[Network, NetworkMeta]:
    """Single generation attempt. May raise or return an unsolvable network."""
    node_range, subnet_range = _size_params(size)
    n_nodes = rng.randint(*node_range)
    n_subnets = rng.randint(*subnet_range)

    # --- Step 1: Design zone layout ---
    zone_sequence = _build_zone_sequence(n_subnets, rng)
    node_counts = _distribute_nodes(n_nodes, zone_sequence, rng)

    # --- Step 2: Assign node IDs and zones ---
    zone_map: dict[int, Zone] = {}
    node_id_counter = 0
    subnet_node_ids: list[list[int]] = []

    for zone, count in zip(zone_sequence, node_counts):
        ids = list(range(node_id_counter, node_id_counter + count))
        node_id_counter += count
        subnet_node_ids.append(ids)
        for nid in ids:
            zone_map[nid] = zone

    total_nodes = node_id_counter
    assert total_nodes <= MAX_NODES, f"Generated {total_nodes} nodes > MAX_NODES={MAX_NODES}"

    # --- Step 3: Build intra-zone topology ---
    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    for ids in subnet_node_ids:
        _connect_subnet(G, ids, rng)

    # --- Step 4: Connect zones in sequence ---
    _connect_zones_sequential(G, subnet_node_ids, rng)

    # Cross-zone shortcut edges removed — sequential + sparse intra-zone is enough

    # Verify connectivity
    if not nx.is_connected(G):
        raise ValueError("Generated graph is not connected")

    # --- Step 5: Select entry and target ---
    dmz_ids = subnet_node_ids[0]       # first zone = DMZ
    dc_ids = subnet_node_ids[-1]        # last zone = DATACENTER
    entry_id = rng.choice(dmz_ids)
    # Target = DATACENTER node farthest from the entry (maximise path length)
    bfs_lengths = nx.single_source_shortest_path_length(G, entry_id)
    target_id = max(dc_ids, key=lambda n: bfs_lengths.get(n, 0))

    # --- Step 6: Assign node properties ---
    nodes: list[Node] = []
    for nid in range(total_nodes):
        zone = zone_map[nid]
        node = _make_node(nid, zone, nid == target_id, rng)
        nodes.append(node)

    # --- Step 7: Build Network object ---
    net = Network()
    for node in nodes:
        net.add_node(node)
    for u, v in G.edges():
        net.add_edge(u, v)
    net.entry_node_id = entry_id
    net.target_node_id = target_id

    # --- Step 8: Compute metadata ---
    from src.pcg.difficulty import compute_difficulty, compute_max_steps
    min_hops = nx.shortest_path_length(net.graph, entry_id, target_id)
    diff = compute_difficulty(net, entry_id, target_id)
    max_steps = compute_max_steps(net, entry_id, target_id)

    meta = NetworkMeta(
        size=size,
        n_nodes=total_nodes,
        n_subnets=n_subnets,
        min_hops=min_hops,
        difficulty_score=diff,
        recommended_max_steps=max_steps,
        zone_map=zone_map,
    )
    return net, meta


# ---------------------------------------------------------------------------
# Zone layout helpers
# ---------------------------------------------------------------------------


def _size_params(size: NetworkSize) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return (node_range, subnet_range) for the given size."""
    if size == NetworkSize.SMALL:
        return PCG_SMALL_NODES, PCG_SMALL_SUBNETS
    if size == NetworkSize.MEDIUM:
        return PCG_MEDIUM_NODES, PCG_MEDIUM_SUBNETS
    return PCG_LARGE_NODES, PCG_LARGE_SUBNETS


def _build_zone_sequence(n_subnets: int, rng: random.Random) -> list[Zone]:
    """Build the ordered list of zones for each subnet.

    Always: DMZ first, DATACENTER last.
    Middle subnets: CORPORATE and SERVER alternated (more CORPORATE for variety).
    """
    n_middle = n_subnets - 2
    if n_middle <= 0:
        # Minimum: DMZ + DATACENTER
        return [Zone.DMZ, Zone.DATACENTER]

    middle_zones: list[Zone] = []
    for i in range(n_middle):
        # Alternate: CORPORATE for even indices, SERVER for every 3rd
        if n_middle == 1:
            middle_zones.append(Zone.CORPORATE)
        elif i % 3 == 2:
            middle_zones.append(Zone.SERVER)
        else:
            middle_zones.append(Zone.CORPORATE)

    # Ensure at least 1 SERVER zone for medium/large
    if Zone.SERVER not in middle_zones and n_middle >= 2:
        middle_zones[-1] = Zone.SERVER

    return [Zone.DMZ] + middle_zones + [Zone.DATACENTER]


def _distribute_nodes(
    n_nodes: int,
    zone_sequence: list[Zone],
    rng: random.Random,
) -> list[int]:
    """Distribute n_nodes across zones. DMZ and DC get 2-4 each, rest split evenly."""
    n_zones = len(zone_sequence)

    # DMZ and DC: 2-4 nodes each, but capped so middle zones always get ≥1 each
    n_middle_zones = n_zones - 2
    # Reserve at least 1 node per middle zone
    max_border = max(2, (n_nodes - max(0, n_middle_zones)) // 2)
    dmz_count = rng.randint(2, min(4, max_border))
    dc_count = rng.randint(2, min(4, max_border))
    remaining = n_nodes - dmz_count - dc_count

    if remaining <= 0:
        # Extreme edge case (very small network, many zones) — collapse to DMZ + DC only
        # Trim border counts to fit all nodes
        half = n_nodes // 2
        return [half, n_nodes - half]

    if n_middle_zones == 0:
        # Only DMZ + DC: give all remaining to DC
        return [dmz_count, dc_count + remaining]

    # Distribute remaining nodes roughly evenly among middle zones with jitter
    base = remaining // n_middle_zones
    extra = remaining % n_middle_zones
    middle_counts = [base + (1 if i < extra else 0) for i in range(n_middle_zones)]

    # Add ±1 jitter per zone (keep total intact)
    for i in range(len(middle_counts) - 1):
        delta = rng.randint(-1, 1)
        if middle_counts[i] + delta >= 2 and middle_counts[i + 1] - delta >= 2:
            middle_counts[i] += delta
            middle_counts[i + 1] -= delta

    return [dmz_count] + middle_counts + [dc_count]


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _connect_subnet(G: nx.Graph, ids: list[int], rng: random.Random) -> None:
    """Add intra-subnet edges using Barabási-Albert or simple chain."""
    n = len(ids)
    if n == 1:
        return  # single node — no edges needed
    if n <= 3:
        # Simple chain (or triangle for n=3)
        for i in range(n - 1):
            G.add_edge(ids[i], ids[i + 1])
        if n == 3:
            G.add_edge(ids[0], ids[2])  # close triangle
        return

    # Barabási-Albert with m=1 (spanning tree — sparser, more realistic)
    m = 1
    ba = nx.barabasi_albert_graph(n, m, seed=rng.randint(0, 2**31))
    mapping = {i: ids[i] for i in range(n)}
    for u, v in ba.edges():
        G.add_edge(mapping[u], mapping[v])


def _connect_zones_sequential(
    G: nx.Graph,
    subnet_node_ids: list[list[int]],
    rng: random.Random,
) -> None:
    """Connect adjacent zones with 1-2 gateway edges."""
    for i in range(len(subnet_node_ids) - 1):
        src_ids = subnet_node_ids[i]
        dst_ids = subnet_node_ids[i + 1]
        # Pick a gateway node in each zone (prefer low-degree for variety)
        gw_src = rng.choice(src_ids)
        gw_dst = rng.choice(dst_ids)
        G.add_edge(gw_src, gw_dst)
        # Second gateway edge for redundancy (~25% chance — keep graph sparse)
        if len(src_ids) >= 2 and len(dst_ids) >= 2 and rng.random() < 0.25:
            alt_src = rng.choice([x for x in src_ids if x != gw_src])
            alt_dst = rng.choice([x for x in dst_ids if x != gw_dst])
            G.add_edge(alt_src, alt_dst)


def _add_cross_zone_edges(
    G: nx.Graph,
    subnet_node_ids: list[list[int]],
    rng: random.Random,
    n_extra: int = 1,
) -> None:
    """Add cross-zone shortcut edges (skip one zone) for network variety."""
    n_subnets = len(subnet_node_ids)
    added = 0
    attempts = 0
    while added < n_extra and attempts < 20:
        attempts += 1
        i = rng.randint(0, n_subnets - 3)
        j = i + 2  # skip one zone
        src = rng.choice(subnet_node_ids[i])
        dst = rng.choice(subnet_node_ids[j])
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)
            added += 1


# ---------------------------------------------------------------------------
# Node property assignment
# ---------------------------------------------------------------------------


def _make_node(
    node_id: int,
    zone: Zone,
    is_target: bool,
    rng: random.Random,
) -> Node:
    """Create a Node with zone-appropriate properties."""
    os_type = _pick_os(zone, rng)
    services = _pick_services(os_type, rng)
    vulns = _pick_vulns(zone, is_target, rng)
    # Target node: no weak credentials (can't brute-force), always has flag.txt.
    # Non-target nodes never have loot — only the flag node triggers the win condition.
    has_weak_creds = False if is_target else rng.random() < _WEAK_CREDS_PROB[zone]
    has_loot = is_target  # only the target (flag node) has loot

    return Node(
        node_id=node_id,
        os_type=os_type,
        services=services,
        vulnerabilities=vulns,
        has_weak_credentials=has_weak_creds,
        has_loot=has_loot,
    )


def _pick_os(zone: Zone, rng: random.Random) -> OsType:
    weights = _OS_WEIGHTS[zone]
    types = [OsType.LINUX, OsType.WINDOWS, OsType.NETWORK_DEVICE]
    return rng.choices(types, weights=weights, k=1)[0]


def _pick_services(os_type: OsType, rng: random.Random) -> list[Service]:
    pool = _SERVICES_BY_OS[os_type]
    n = rng.randint(1, min(3, len(pool)))
    chosen = rng.sample(pool, n)
    return [Service(name=name, port=port) for name, port in chosen]


def _pick_vulns(zone: Zone, is_target: bool, rng: random.Random) -> list[str]:
    """Pick vulnerabilities from the zone pool.

    Target node has NO vulnerabilities — it only contains flag.txt.
    The agent must reach it via LATERAL_MOVE and then run LIST_FILES (ls).
    Non-target nodes get 1-3 vulns from the zone pool.
    """
    if is_target:
        return []  # no exploitable vulns — must reach via lateral movement

    pool = _ZONE_VULN_POOLS[zone]
    valid_pool = [v for v in pool if v in VULN_REGISTRY]
    n_vulns = rng.randint(1, min(3, len(valid_pool))) if valid_pool else 1
    if not valid_pool:
        return ["rce_generic"]
    return rng.sample(valid_pool, n_vulns)


# ---------------------------------------------------------------------------
# Fallback network (guaranteed solvable, minimal structure)
# ---------------------------------------------------------------------------


def _generate_fallback(size: NetworkSize, rng: random.Random) -> tuple[Network, NetworkMeta]:
    """Minimal guaranteed-solvable network for the given size.

    Used when all 10 retry attempts fail. Builds a linear chain:
        DMZ_entry → Corp₁ → ... → DC_target
    Each node on the path has at least one exploitable vuln and the target
    has a PRIVESC vuln.
    """
    node_range, _ = _size_params(size)
    n_nodes = node_range[0]  # use minimum for fallback

    # Build a simple chain: 0=entry(DMZ), ..., n-1=target(DC)
    zone_assignment = [Zone.DMZ] + [Zone.CORPORATE] * (n_nodes - 2) + [Zone.DATACENTER]
    zone_map: dict[int, Zone] = {i: zone_assignment[i] for i in range(n_nodes)}

    net = Network()
    for nid in range(n_nodes):
        zone = zone_map[nid]
        is_target = nid == n_nodes - 1
        # Ensure every node on the chain has an exploitable vuln
        if zone == Zone.DATACENTER:
            vulns = ["privesc_sudo_baron"]
        elif zone == Zone.CORPORATE:
            vulns = ["rce_eternal_blue"]
        else:
            vulns = ["rce_log4shell"]
        # Target always has loot; intermediate non-DMZ nodes have a small chance
        has_loot = is_target or (zone == Zone.CORPORATE and rng.random() < 0.3)
        node = Node(
            node_id=nid,
            os_type=OsType.LINUX,
            services=[Service("ssh", 22)],
            vulnerabilities=vulns,
            has_loot=has_loot,
        )
        net.add_node(node)
    for i in range(n_nodes - 1):
        net.add_edge(i, i + 1)
    net.entry_node_id = 0
    net.target_node_id = n_nodes - 1

    from src.pcg.difficulty import compute_difficulty, compute_max_steps
    diff = compute_difficulty(net, 0, n_nodes - 1)
    steps = compute_max_steps(net, 0, n_nodes - 1)

    meta = NetworkMeta(
        size=size,
        n_nodes=n_nodes,
        n_subnets=2,
        min_hops=n_nodes - 1,
        difficulty_score=diff,
        recommended_max_steps=steps,
        zone_map=zone_map,
    )
    return net, meta
