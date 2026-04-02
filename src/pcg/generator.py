from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum

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
from src.environment.node import Node, OsType, Service
from src.environment.vulnerability import VULN_REGISTRY

logger = logging.getLogger(__name__)


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
    min_hops: int
    difficulty_score: float
    recommended_max_steps: int
    zone_map: dict[int, Zone]


# OS probability weights per zone: (LINUX, WINDOWS, NETWORK_DEVICE)
_OS_WEIGHTS: dict[Zone, tuple[float, float, float]] = {
    Zone.DMZ:        (0.55, 0.10, 0.35),
    Zone.CORPORATE:  (0.30, 0.60, 0.10),
    Zone.SERVER:     (0.70, 0.20, 0.10),
    Zone.DATACENTER: (0.80, 0.20, 0.00),
}

_ZONE_VULN_POOLS: dict[Zone, list[str]] = {
    Zone.DMZ: ["rce_log4shell", "rce_shellshock", "ssrf_metadata", "sqli_union_bypass", "sqli_blind_time"],
    Zone.CORPORATE: ["rce_eternal_blue", "privesc_print_nightmare", "sqli_union_bypass", "rce_log4shell"],
    Zone.SERVER: ["privesc_docker_escape", "rce_log4shell", "sqli_blind_time", "sqli_union_bypass", "ssrf_metadata"],
    Zone.DATACENTER: ["privesc_sudo_baron", "privesc_dirty_cow", "privesc_kernel", "privesc_suid"],
}

_SERVICES_BY_OS: dict[OsType, list[tuple[str, int]]] = {
    OsType.LINUX:          [("ssh", 22), ("http", 80), ("https", 443), ("ftp", 21), ("smtp", 25), ("nfs", 2049)],
    OsType.WINDOWS:        [("smb", 445), ("rdp", 3389), ("http", 80), ("https", 443), ("winrm", 5985)],
    OsType.NETWORK_DEVICE: [("ssh", 22), ("telnet", 23), ("snmp", 161), ("https", 443)],
}

_WEAK_CREDS_PROB: dict[Zone, float] = {
    Zone.DMZ: 0.10, Zone.CORPORATE: 0.25, Zone.SERVER: 0.08, Zone.DATACENTER: 0.05,
}

_LOOT_PROB: dict[Zone, float] = {
    Zone.DMZ: 0.00, Zone.CORPORATE: 0.05, Zone.SERVER: 0.15, Zone.DATACENTER: 1.00,
}


def generate_network(
    size: NetworkSize,
    seed: int | None = None,
) -> tuple[Network, NetworkMeta]:
    """Generate a procedural enterprise network.

    Retries up to 10 times if validation fails, then falls back to a
    minimal guaranteed-solvable linear chain.
    """
    rng = random.Random(seed)

    for attempt in range(10):
        attempt_seed = None if seed is None else seed + attempt
        try:
            net, meta = _generate_attempt(size, random.Random(attempt_seed))
            from src.pcg.difficulty import is_solvable
            if is_solvable(net):
                return net, meta
        except Exception:  # noqa: BLE001
            pass

    logger.warning(
        "generate_network: all 10 attempts failed for size=%s seed=%s — using fallback linear chain",
        size.value, seed,
    )
    return _generate_fallback(size, rng)


def _generate_attempt(size: NetworkSize, rng: random.Random) -> tuple[Network, NetworkMeta]:
    """Single generation attempt. May raise or return an unsolvable network."""
    node_range, subnet_range = _size_params(size)
    n_nodes = rng.randint(*node_range)
    n_subnets = rng.randint(*subnet_range)

    zone_sequence = _build_zone_sequence(n_subnets, rng)
    node_counts = _distribute_nodes(n_nodes, zone_sequence, rng)

    zone_map: dict[int, Zone] = {}
    node_id_counter = 0
    subnet_node_ids: list[list[int]] = []

    for zone, count in zip(zone_sequence, node_counts, strict=False):
        ids = list(range(node_id_counter, node_id_counter + count))
        node_id_counter += count
        subnet_node_ids.append(ids)
        for nid in ids:
            zone_map[nid] = zone

    total_nodes = node_id_counter
    assert total_nodes <= MAX_NODES, f"Generated {total_nodes} nodes > MAX_NODES={MAX_NODES}"

    G = nx.Graph()
    G.add_nodes_from(range(total_nodes))
    for ids in subnet_node_ids:
        _connect_subnet(G, ids, rng)
    _connect_zones_sequential(G, subnet_node_ids, rng)

    if not nx.is_connected(G):
        raise ValueError("Generated graph is not connected")

    dmz_ids = subnet_node_ids[0]
    dc_ids = subnet_node_ids[-1]
    entry_id = rng.choice(dmz_ids)
    bfs_lengths = nx.single_source_shortest_path_length(G, entry_id)
    target_id = max(dc_ids, key=lambda n: bfs_lengths.get(n, 0))

    nodes: list[Node] = []
    for nid in range(total_nodes):
        zone = zone_map[nid]
        node = _make_node(nid, zone, nid == target_id, rng, total_nodes=total_nodes)
        nodes.append(node)

    net = Network()
    for node in nodes:
        net.add_node(node)
    for u, v in G.edges():
        net.add_edge(u, v)
    net.entry_node_id = entry_id
    net.target_node_id = target_id

    from src.pcg.difficulty import compute_difficulty, compute_max_steps
    min_hops = nx.shortest_path_length(net.graph, entry_id, target_id)
    diff = compute_difficulty(net, entry_id, target_id)
    max_steps = compute_max_steps(net, entry_id, target_id)

    meta = NetworkMeta(
        size=size, n_nodes=total_nodes, n_subnets=n_subnets,
        min_hops=min_hops, difficulty_score=diff,
        recommended_max_steps=max_steps, zone_map=zone_map,
    )
    return net, meta


def _size_params(size: NetworkSize) -> tuple[tuple[int, int], tuple[int, int]]:
    if size == NetworkSize.SMALL:
        return PCG_SMALL_NODES, PCG_SMALL_SUBNETS
    if size == NetworkSize.MEDIUM:
        return PCG_MEDIUM_NODES, PCG_MEDIUM_SUBNETS
    return PCG_LARGE_NODES, PCG_LARGE_SUBNETS


def _build_zone_sequence(n_subnets: int, rng: random.Random) -> list[Zone]:
    """Build the ordered list of zones: DMZ first, DATACENTER last."""
    n_middle = n_subnets - 2
    if n_middle <= 0:
        return [Zone.DMZ, Zone.DATACENTER]

    middle_zones: list[Zone] = []
    for i in range(n_middle):
        if n_middle == 1:
            middle_zones.append(Zone.CORPORATE)
        elif i % 3 == 2:
            middle_zones.append(Zone.SERVER)
        else:
            middle_zones.append(Zone.CORPORATE)

    if Zone.SERVER not in middle_zones and n_middle >= 2:
        middle_zones[-1] = Zone.SERVER

    return [Zone.DMZ] + middle_zones + [Zone.DATACENTER]


def _distribute_nodes(n_nodes: int, zone_sequence: list[Zone], rng: random.Random) -> list[int]:
    """Distribute n_nodes across zones. DMZ and DC get 2-4 each, rest split evenly."""
    n_zones = len(zone_sequence)
    n_middle_zones = n_zones - 2
    max_border = max(2, (n_nodes - max(0, n_middle_zones)) // 2)
    dmz_count = rng.randint(2, min(4, max_border))
    dc_count = rng.randint(2, min(4, max_border))
    remaining = n_nodes - dmz_count - dc_count

    if remaining <= 0:
        half = n_nodes // 2
        return [half, n_nodes - half]

    if n_middle_zones == 0:
        return [dmz_count, dc_count + remaining]

    base = remaining // n_middle_zones
    extra = remaining % n_middle_zones
    middle_counts = [base + (1 if i < extra else 0) for i in range(n_middle_zones)]

    for i in range(len(middle_counts) - 1):
        delta = rng.randint(-1, 1)
        if middle_counts[i] + delta >= 2 and middle_counts[i + 1] - delta >= 2:
            middle_counts[i] += delta
            middle_counts[i + 1] -= delta

    return [dmz_count] + middle_counts + [dc_count]


def _connect_subnet(G: nx.Graph, ids: list[int], rng: random.Random) -> None:
    """Add intra-subnet edges using Barabási-Albert or simple chain."""
    n = len(ids)
    if n == 1:
        return
    if n <= 3:
        for i in range(n - 1):
            G.add_edge(ids[i], ids[i + 1])
        if n == 3:
            G.add_edge(ids[0], ids[2])
        return

    ba = nx.barabasi_albert_graph(n, 1, seed=rng.randint(0, 2**31))
    mapping = {i: ids[i] for i in range(n)}
    for u, v in ba.edges():
        G.add_edge(mapping[u], mapping[v])

    # Extra edges for large subnets to increase lateral movement diversity.
    extra = max(0, (n - 5) // 5)
    candidates = [(ids[i], ids[j]) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(candidates)
    added = 0
    for u, v in candidates:
        if added >= extra:
            break
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            added += 1


def _connect_zones_sequential(
    G: nx.Graph, subnet_node_ids: list[list[int]], rng: random.Random
) -> None:
    """Connect adjacent zones with 1-2 gateway edges."""
    for i in range(len(subnet_node_ids) - 1):
        src_ids = subnet_node_ids[i]
        dst_ids = subnet_node_ids[i + 1]
        gw_src = rng.choice(src_ids)
        gw_dst = rng.choice(dst_ids)
        G.add_edge(gw_src, gw_dst)
        # Second gateway edge probability scales with zone sizes.
        prob_2nd = min(0.75, (len(src_ids) + len(dst_ids)) / 20.0)
        if len(src_ids) >= 2 and len(dst_ids) >= 2 and rng.random() < prob_2nd:
            alt_src = rng.choice([x for x in src_ids if x != gw_src])
            alt_dst = rng.choice([x for x in dst_ids if x != gw_dst])
            G.add_edge(alt_src, alt_dst)


def _add_cross_zone_edges(
    G: nx.Graph, subnet_node_ids: list[list[int]], rng: random.Random, n_extra: int = 1
) -> None:
    """Add cross-zone shortcut edges (skip one zone) for network variety."""
    n_subnets = len(subnet_node_ids)
    added = 0
    attempts = 0
    while added < n_extra and attempts < 20:
        attempts += 1
        i = rng.randint(0, n_subnets - 3)
        src = rng.choice(subnet_node_ids[i])
        dst = rng.choice(subnet_node_ids[i + 2])
        if not G.has_edge(src, dst):
            G.add_edge(src, dst)
            added += 1


def _make_node(
    node_id: int, zone: Zone, is_target: bool, rng: random.Random, total_nodes: int = 20
) -> Node:
    """Create a Node with zone-appropriate properties."""
    os_type = _pick_os(zone, rng)
    services = _pick_services(os_type, rng)
    vulns = _pick_vulns(zone, is_target, rng)
    # Scale weak-cred probability so absolute count stays ~constant across sizes.
    _WEAK_CREDS_SCALE_NODES: int = 20
    weak_creds_prob = _WEAK_CREDS_PROB[zone] * min(1.0, _WEAK_CREDS_SCALE_NODES / total_nodes)
    has_weak_creds = False if is_target else rng.random() < weak_creds_prob
    return Node(
        node_id=node_id,
        os_type=os_type,
        services=services,
        vulnerabilities=vulns,
        has_weak_credentials=has_weak_creds,
        has_loot=is_target,
    )


def _pick_os(zone: Zone, rng: random.Random) -> OsType:
    weights = _OS_WEIGHTS[zone]
    return rng.choices([OsType.LINUX, OsType.WINDOWS, OsType.NETWORK_DEVICE], weights=weights, k=1)[0]


def _pick_services(os_type: OsType, rng: random.Random) -> list[Service]:
    pool = _SERVICES_BY_OS[os_type]
    n = rng.randint(1, min(3, len(pool)))
    return [Service(name=name, port=port) for name, port in rng.sample(pool, n)]


def _pick_vulns(zone: Zone, is_target: bool, rng: random.Random) -> list[str]:
    """Pick vulnerabilities from the zone pool. Target node has none — only flag.txt."""
    if is_target:
        return []
    pool = _ZONE_VULN_POOLS[zone]
    valid_pool = [v for v in pool if v in VULN_REGISTRY]
    if not valid_pool:
        return ["rce_generic"]
    return rng.sample(valid_pool, rng.randint(1, min(3, len(valid_pool))))


def _generate_fallback(size: NetworkSize, rng: random.Random) -> tuple[Network, NetworkMeta]:
    """Minimal guaranteed-solvable linear chain. Used when all 10 retry attempts fail."""
    node_range, _ = _size_params(size)
    n_nodes = node_range[0]

    zone_assignment = [Zone.DMZ] + [Zone.CORPORATE] * (n_nodes - 2) + [Zone.DATACENTER]
    zone_map: dict[int, Zone] = {i: zone_assignment[i] for i in range(n_nodes)}

    net = Network()
    for nid in range(n_nodes):
        zone = zone_map[nid]
        is_target = nid == n_nodes - 1
        if zone == Zone.DATACENTER:
            vulns = ["privesc_sudo_baron"]
        elif zone == Zone.CORPORATE:
            vulns = ["rce_eternal_blue"]
        else:
            vulns = ["rce_log4shell"]
        has_loot = is_target or (zone == Zone.CORPORATE and rng.random() < 0.3)
        node = Node(
            node_id=nid, os_type=OsType.LINUX,
            services=[Service("ssh", 22)],
            vulnerabilities=vulns, has_loot=has_loot,
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
        size=size, n_nodes=n_nodes, n_subnets=2,
        min_hops=n_nodes - 1, difficulty_score=diff,
        recommended_max_steps=steps, zone_map=zone_map,
    )
    return net, meta
