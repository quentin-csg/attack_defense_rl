"""Tests for the PCG network generator (Phase 5)."""

from __future__ import annotations

import pytest

from src.config import MAX_NODES, PCG_LARGE_NODES, PCG_MEDIUM_NODES, PCG_SMALL_NODES
from src.pcg.difficulty import is_solvable
from src.pcg.generator import NetworkMeta, NetworkSize, Zone, generate_network


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gen(size: NetworkSize, seed: int = 42) -> tuple:
    return generate_network(size, seed=seed)


# ---------------------------------------------------------------------------
# Node count tests
# ---------------------------------------------------------------------------


class TestNodeCounts:
    def test_small_node_count(self) -> None:
        net, meta = _gen(NetworkSize.SMALL)
        lo, hi = PCG_SMALL_NODES
        assert lo <= meta.n_nodes <= hi, f"Small: expected [{lo},{hi}], got {meta.n_nodes}"

    def test_medium_node_count(self) -> None:
        net, meta = _gen(NetworkSize.MEDIUM)
        lo, hi = PCG_MEDIUM_NODES
        assert lo <= meta.n_nodes <= hi, f"Medium: expected [{lo},{hi}], got {meta.n_nodes}"

    def test_large_node_count(self) -> None:
        net, meta = _gen(NetworkSize.LARGE)
        lo, hi = PCG_LARGE_NODES
        assert lo <= meta.n_nodes <= hi, f"Large: expected [{lo},{hi}], got {meta.n_nodes}"

    def test_all_node_ids_under_max_nodes(self) -> None:
        for size in NetworkSize:
            net, _ = _gen(size, seed=7)
            for nid in net.nodes:
                assert nid < MAX_NODES, f"Node ID {nid} >= MAX_NODES={MAX_NODES}"


# ---------------------------------------------------------------------------
# Subnet count tests
# ---------------------------------------------------------------------------


class TestSubnetCounts:
    def test_small_subnet_count(self) -> None:
        _, meta = _gen(NetworkSize.SMALL)
        assert 2 <= meta.n_subnets <= 3, f"Small subnets: expected [2,3], got {meta.n_subnets}"

    def test_medium_subnet_count(self) -> None:
        _, meta = _gen(NetworkSize.MEDIUM)
        assert 4 <= meta.n_subnets <= 5, f"Medium subnets: expected [4,5], got {meta.n_subnets}"

    def test_large_subnet_count(self) -> None:
        _, meta = _gen(NetworkSize.LARGE)
        assert 7 <= meta.n_subnets <= 8, f"Large subnets: expected [7,8], got {meta.n_subnets}"


# ---------------------------------------------------------------------------
# Connectivity & path tests
# ---------------------------------------------------------------------------


class TestConnectivity:
    def test_always_connected_small(self) -> None:
        """50 different Small seeds — all connected."""
        import networkx as nx
        for seed in range(50):
            net, _ = generate_network(NetworkSize.SMALL, seed=seed)
            assert nx.is_connected(net.graph), f"Small seed={seed}: graph not connected"

    def test_always_connected_medium(self) -> None:
        import networkx as nx
        for seed in range(20):
            net, _ = generate_network(NetworkSize.MEDIUM, seed=seed)
            assert nx.is_connected(net.graph), f"Medium seed={seed}: graph not connected"

    def test_always_connected_large(self) -> None:
        import networkx as nx
        for seed in range(10):
            net, _ = generate_network(NetworkSize.LARGE, seed=seed)
            assert nx.is_connected(net.graph), f"Large seed={seed}: graph not connected"

    def test_path_entry_to_target_exists(self) -> None:
        """All sizes, multiple seeds: entry → target path always exists."""
        import networkx as nx
        for size in NetworkSize:
            for seed in range(20):
                net, _ = generate_network(size, seed=seed)
                assert nx.has_path(net.graph, net.entry_node_id, net.target_node_id), (
                    f"{size.value} seed={seed}: no path entry→target"
                )


# ---------------------------------------------------------------------------
# Zone tests
# ---------------------------------------------------------------------------


class TestZones:
    def test_entry_in_dmz(self) -> None:
        for seed in range(20):
            net, meta = generate_network(NetworkSize.SMALL, seed=seed)
            assert meta.zone_map[net.entry_node_id] == Zone.DMZ, (
                f"seed={seed}: entry node not in DMZ"
            )

    def test_target_in_datacenter(self) -> None:
        for seed in range(20):
            net, meta = generate_network(NetworkSize.SMALL, seed=seed)
            assert meta.zone_map[net.target_node_id] == Zone.DATACENTER, (
                f"seed={seed}: target node not in DATACENTER"
            )

    def test_dmz_and_dc_zones_always_present(self) -> None:
        for size in NetworkSize:
            _, meta = generate_network(size, seed=42)
            zones_in_map = set(meta.zone_map.values())
            assert Zone.DMZ in zones_in_map
            assert Zone.DATACENTER in zones_in_map


# ---------------------------------------------------------------------------
# Target / loot tests
# ---------------------------------------------------------------------------


class TestTarget:
    def test_target_has_loot(self) -> None:
        for size in NetworkSize:
            for seed in range(15):
                net, _ = generate_network(size, seed=seed)
                assert net.get_node(net.target_node_id).has_loot, (
                    f"{size.value} seed={seed}: target missing has_loot"
                )

    def test_target_has_no_vulns(self) -> None:
        # With LIST_FILES win condition, target has no exploitable vulns — only flag.txt
        for seed in range(20):
            net, _ = generate_network(NetworkSize.SMALL, seed=seed)
            target = net.get_node(net.target_node_id)
            assert target.vulnerabilities == [], (
                f"seed={seed}: target should have no vulns, got {target.vulnerabilities}"
            )
            assert target.has_weak_credentials is False, (
                f"seed={seed}: target should have no weak credentials"
            )


# ---------------------------------------------------------------------------
# Vulnerability distribution tests
# ---------------------------------------------------------------------------


class TestVulnDistribution:
    def test_nodes_have_vulns(self) -> None:
        """Every non-target node must have ≥1 vulnerability.
        The target node intentionally has no vulns (win via LIST_FILES / ls).
        """
        net, _ = generate_network(NetworkSize.MEDIUM, seed=42)
        for nid, node in net.nodes.items():
            if nid == net.target_node_id:
                continue  # target has no vulns by design
            assert len(node.vulnerabilities) >= 1, f"Node {nid} has no vulnerabilities"

    def test_vulns_in_registry(self) -> None:
        """All assigned vuln names must exist in VULN_REGISTRY."""
        from src.environment.vulnerability import VULN_REGISTRY
        net, _ = generate_network(NetworkSize.MEDIUM, seed=42)
        for nid, node in net.nodes.items():
            for v in node.vulnerabilities:
                assert v in VULN_REGISTRY, f"Node {nid}: unknown vuln '{v}'"


# ---------------------------------------------------------------------------
# Reproducibility / determinism tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_seed_determinism(self) -> None:
        """Same seed → same node count, same edges, same entry/target."""
        net1, meta1 = generate_network(NetworkSize.MEDIUM, seed=123)
        net2, meta2 = generate_network(NetworkSize.MEDIUM, seed=123)
        assert meta1.n_nodes == meta2.n_nodes
        assert net1.entry_node_id == net2.entry_node_id
        assert net1.target_node_id == net2.target_node_id
        assert set(net1.graph.edges()) == set(net2.graph.edges())

    def test_different_seeds_different_networks(self) -> None:
        """Different seeds must produce at least some distinct topologies."""
        edge_sets = set()
        for seed in range(10):
            net, _ = generate_network(NetworkSize.MEDIUM, seed=seed)
            edge_sets.add(frozenset(net.graph.edges()))
        assert len(edge_sets) > 1, "All 10 seeds produced identical topologies"


# ---------------------------------------------------------------------------
# Solvability test
# ---------------------------------------------------------------------------


class TestSolvability:
    def test_generated_networks_are_solvable(self) -> None:
        """All generated networks must pass is_solvable()."""
        for size in NetworkSize:
            for seed in range(15):
                net, _ = generate_network(size, seed=seed)
                assert is_solvable(net), (
                    f"{size.value} seed={seed}: network is not solvable"
                )
