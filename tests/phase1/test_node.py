"""Tests for Node dataclass and related enums."""

from src.environment.node import (
    DiscoveryLevel,
    Node,
    OsType,
    Service,
    SessionLevel,
)


class TestNode:
    def test_node_creation(self) -> None:
        node = Node(node_id=0, os_type=OsType.LINUX)
        assert node.node_id == 0
        assert node.os_type == OsType.LINUX
        assert node.is_online is True
        assert node.suspicion_level == 0.0
        assert node.session_level == SessionLevel.NONE
        assert node.discovery_level == DiscoveryLevel.UNKNOWN

    def test_node_with_services(self) -> None:
        services = [Service("ssh", 22), Service("http", 80)]
        node = Node(node_id=1, services=services)
        assert len(node.services) == 2
        assert node.services[0].name == "ssh"
        assert node.services[0].port == 22

    def test_add_suspicion(self) -> None:
        node = Node(node_id=0)
        node.add_suspicion(30.0)
        assert node.suspicion_level == 30.0
        assert node.max_suspicion_historical == 30.0

    def test_suspicion_capped_at_max(self) -> None:
        node = Node(node_id=0)
        node.add_suspicion(150.0)
        assert node.suspicion_level == 100.0

    def test_suspicion_surveillance_multiplier(self) -> None:
        node = Node(node_id=0, is_under_surveillance=True)
        node.add_suspicion(10.0)
        assert node.suspicion_level == 20.0  # x2 multiplier

    def test_reduce_suspicion(self) -> None:
        node = Node(node_id=0)
        node.add_suspicion(50.0)
        node.reduce_suspicion(20.0)  # positive = amount to reduce
        assert node.suspicion_level == 30.0

    def test_reduce_suspicion_floor(self) -> None:
        """Suspicion should not drop below max_historical / 2."""
        node = Node(node_id=0)
        node.add_suspicion(80.0)  # max_historical = 80
        node.reduce_suspicion(60.0)
        # Floor = 80 / 2 = 40
        assert node.suspicion_level == 40.0

    def test_reset_session_without_backdoor(self) -> None:
        node = Node(node_id=0, session_level=SessionLevel.USER)
        node.reset_session()
        assert node.session_level == SessionLevel.NONE

    def test_reset_session_with_backdoor(self) -> None:
        node = Node(node_id=0, session_level=SessionLevel.USER, has_backdoor=True)
        node.reset_session()
        assert node.session_level == SessionLevel.USER  # backdoor protects


class TestEnums:
    def test_os_types(self) -> None:
        assert len(OsType) == 3

    def test_session_levels(self) -> None:
        assert SessionLevel.NONE.value == 0
        assert SessionLevel.USER.value == 1
        assert SessionLevel.ROOT.value == 2

    def test_reset_session_clears_tunnel(self) -> None:
        """reset_session() must also clear has_tunnel — a tunnel without a session
        is invalid and would grant unfair suspicion discounts after credential rotation."""
        node = Node(node_id=0, os_type=OsType.LINUX)
        node.session_level = SessionLevel.USER
        node.has_tunnel = True

        node.reset_session()

        assert node.session_level == SessionLevel.NONE
        assert node.has_tunnel is False

    def test_reset_session_preserves_tunnel_with_backdoor(self) -> None:
        """A backdoor protects both session and tunnel from credential rotation."""
        node = Node(node_id=0, os_type=OsType.LINUX)
        node.session_level = SessionLevel.USER
        node.has_backdoor = True
        node.has_tunnel = True

        node.reset_session()

        assert node.session_level == SessionLevel.USER  # backdoor preserved session
        assert node.has_tunnel is True                  # backdoor preserved tunnel

    def test_discovery_levels(self) -> None:
        assert DiscoveryLevel.UNKNOWN.value == 0
        assert DiscoveryLevel.DISCOVERED.value == 1
        assert DiscoveryLevel.ENUMERATED.value == 2

    def test_session_ordering(self) -> None:
        assert SessionLevel.NONE.value < SessionLevel.USER.value < SessionLevel.ROOT.value
