"""Network wrapper around NetworkX — manages the graph topology and nodes."""

from __future__ import annotations

import networkx as nx

from src.config import MAX_NODES
from src.environment.node import (
    DiscoveryLevel,
    Node,
    OsType,
    Service,
    SessionLevel,
)


class Network:
    """Wrapper around a NetworkX graph that holds Node objects.

    Attributes:
        graph: The underlying NetworkX graph.
        nodes: Mapping of node_id -> Node.
        entry_node_id: The node where the Red Team starts (DMZ entry point).
        target_node_id: The node containing the exfiltration target (data center).
        isolated_edges: Edges temporarily removed by ISOLATE_NODE.
    """

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self.nodes: dict[int, Node] = {}
        self.entry_node_id: int = 0
        self.target_node_id: int = 0
        self.isolated_edges: dict[int, list[tuple[int, int]]] = {}

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)

    def add_node(self, node: Node) -> None:
        """Add a node to the network."""
        if node.node_id >= MAX_NODES:
            raise ValueError(f"node_id {node.node_id} exceeds MAX_NODES ({MAX_NODES})")
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id)

    def add_edge(self, node_a: int, node_b: int) -> None:
        """Add a bidirectional edge between two nodes."""
        if node_a not in self.nodes or node_b not in self.nodes:
            raise ValueError(f"Both nodes must exist: {node_a}, {node_b}")
        self.graph.add_edge(node_a, node_b)

    def get_neighbors(self, node_id: int) -> list[int]:
        """Return the list of neighbor node IDs."""
        return list(self.graph.neighbors(node_id))

    def is_adjacent(self, node_a: int, node_b: int) -> bool:
        """Check if two nodes are directly connected."""
        return self.graph.has_edge(node_a, node_b)

    def isolate_node(self, node_id: int) -> None:
        """Temporarily remove all edges of a node (Blue Team ISOLATE).

        Stores removed edges so they can be restored later.
        """
        edges = list(self.graph.edges(node_id))
        self.isolated_edges[node_id] = edges
        self.graph.remove_edges_from(edges)
        self.nodes[node_id].is_online = False

    def restore_node(self, node_id: int) -> None:
        """Restore previously isolated node's edges."""
        if node_id in self.isolated_edges:
            self.graph.add_edges_from(self.isolated_edges[node_id])
            del self.isolated_edges[node_id]
            self.nodes[node_id].is_online = True

    def get_node(self, node_id: int) -> Node:
        """Get a node by ID.

        Raises:
            KeyError: If node_id does not exist.
        """
        return self.nodes[node_id]

    def shortest_path_length(self, source: int, target: int) -> int | None:
        """Return shortest path length, or None if no path exists."""
        try:
            return nx.shortest_path_length(self.graph, source, target)
        except nx.NetworkXNoPath:
            return None

    def reset_all_nodes(self) -> None:
        """Reset all node states for a new episode."""
        for node in self.nodes.values():
            node.suspicion_level = 0.0
            node.max_suspicion_historical = 0.0
            node.session_level = SessionLevel.NONE
            node.discovery_level = DiscoveryLevel.UNKNOWN
            node.has_backdoor = False
            node.has_tunnel = False
            node.is_online = True
            node.is_under_surveillance = False
            node.clean_logs_count = 0
            node.last_clean_logs_step = -100
            node.detectable_traces = set()

        # Restore any isolated nodes
        for node_id in list(self.isolated_edges.keys()):
            self.restore_node(node_id)


def build_fixed_network(seed: int | None = None) -> Network:
    """Build a fixed 8-node enterprise network for Phase 1 development.

    Topology (logical zones):

        [Internet / Attacker]
                |
        Node 0: WEB SERVER (LINUX) — public-facing entry point
                |
        Node 1: FIREWALL (NETWORK_DEVICE) — DMZ perimeter
              /       \\
        Node 2        Node 3: CORE ROUTER (NETWORK_DEVICE)
        Mail Server     /     |     \\
        (LINUX)      Node 4  Node 5  Node 6
        DMZ          PC HR   PC Dev  App Server
                     (WIN)   (WIN)   (LINUX)
                                         |
                                     Node 7: DATABASE (LINUX)
                                     (Data Center — TARGET)

    Zones:
        Internet/DMZ : 0 (Web Server — internet-facing)
        DMZ          : 1 (Firewall), 2 (Mail Server)
        Corp LAN     : 3 (Core Router), 4 (PC HR), 5 (PC Dev)
        Server VLAN  : 6 (App Server)
        Data Center  : 7 (Database — has_loot)

    Attack paths:
        Web  : 0 (Log4Shell) → 1 (Shellshock) → 3 → 6 (Log4Shell) → 7 (Baron Samedit) → exfil
        Mail : 0 → 1 → 2 (brute weak_creds)   → 1 → 3 → 4 (EternalBlue) → ... → 7
        Dev  : 0 → 1 → 3 (SSRF) → 5 (EternalBlue + PrintNightmare) → ... → 7
    """
    net = Network()

    # Node definitions: (id, os_type, services, vulns, has_loot, has_weak_creds)
    node_specs: list[tuple[int, OsType, list[Service], list[str], bool, bool]] = [
        # --- Internet-facing (entry point) ---
        (
            0,
            OsType.LINUX,
            [Service("ssh", 22), Service("http", 80), Service("https", 443)],
            ["rce_log4shell", "sqli_union_bypass"],
            False,
            False,
        ),
        # --- DMZ ---
        (
            1,
            OsType.NETWORK_DEVICE,
            [Service("ssh", 22), Service("https", 443)],
            ["rce_shellshock"],
            False,
            False,
        ),
        (
            2,
            OsType.LINUX,
            [Service("ssh", 22), Service("smtp", 25), Service("imap", 993)],
            ["sqli_blind_time"],
            False,
            True,   # weak creds on mail server (common in real life)
        ),
        # --- Corp LAN ---
        (
            3,
            OsType.NETWORK_DEVICE,
            [Service("ssh", 22), Service("snmp", 161)],
            ["ssrf_metadata"],
            False,
            False,
        ),
        (
            4,
            OsType.WINDOWS,
            [Service("smb", 445), Service("rdp", 3389)],
            ["rce_eternal_blue"],
            False,
            True,   # HR workstation — weak credentials
        ),
        (
            5,
            OsType.WINDOWS,
            [Service("smb", 445), Service("rdp", 3389), Service("http", 8080)],
            ["rce_eternal_blue", "privesc_print_nightmare"],
            False,
            False,
        ),
        # --- Server VLAN ---
        (
            6,
            OsType.LINUX,
            [Service("ssh", 22), Service("http", 8443), Service("mysql", 3306)],
            ["rce_log4shell", "privesc_docker_escape"],
            False,
            False,
        ),
        # --- Data Center ---
        (
            7,
            OsType.LINUX,
            [Service("ssh", 22), Service("nfs", 2049), Service("postgres", 5432)],
            ["privesc_sudo_baron"],
            True,   # exfiltration target
            False,
        ),
    ]

    for nid, os_type, services, vulns, has_loot, has_weak in node_specs:
        node = Node(
            node_id=nid,
            os_type=os_type,
            services=services,
            vulnerabilities=vulns,
            has_loot=has_loot,
            has_weak_credentials=has_weak,
        )
        net.add_node(node)

    # Edges — enterprise logical connections
    edges = [
        (0, 1),  # Web Server → Firewall (internet → DMZ)
        (1, 2),  # Firewall → Mail Server (DMZ)
        (1, 3),  # Firewall → Core Router (DMZ → Corp LAN)
        (3, 4),  # Core Router → PC HR
        (3, 5),  # Core Router → PC Dev
        (3, 6),  # Core Router → App Server (Corp LAN → Server VLAN)
        (4, 5),  # Workstations on same LAN segment
        (6, 7),  # App Server → Database (Server VLAN → Data Center)
    ]
    for a, b in edges:
        net.add_edge(a, b)

    net.entry_node_id = 0
    net.target_node_id = 7

    return net
