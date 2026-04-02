from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.environment.network import Network, build_fixed_network
from src.environment.node import Node, OsType, SessionLevel


# Suspicion colour thresholds (matches Pygame theme)
def _suspicion_color(suspicion: float) -> str:
    """Map suspicion level (0-100) to a hex colour string."""
    if suspicion >= 80:
        return "#e05252"   # red
    if suspicion >= 60:
        return "#f4a261"   # orange
    if suspicion >= 30:
        return "#f9e45a"   # yellow
    return "#4caf7d"       # green


def _session_color(session: SessionLevel) -> str:
    """Fallback colour when suspicion data is unavailable."""
    match session:
        case SessionLevel.ROOT:
            return "#ff4444"
        case SessionLevel.USER:
            return "#ff8800"
        case _:
            return "#4c9be8"


_OS_SYMBOLS = {
    OsType.LINUX: "circle",
    OsType.WINDOWS: "square",
    OsType.NETWORK_DEVICE: "diamond",
}

_NODE_LABELS = {
    0: "Web Server",
    1: "Firewall",
    2: "Mail Server",
    3: "Core Router",
    4: "PC HR",
    5: "PC Dev",
    6: "App Server",
    7: "Database",
}


def _network_from_topology(topo: dict) -> Network:
    """Reconstruct a minimal Network object from serialised topology data."""
    net = Network()
    for str_nid, attrs in topo.get("nodes", {}).items():
        nid = int(str_nid)
        os_type = OsType[attrs.get("os_type", "LINUX")]
        node = Node(node_id=nid, os_type=os_type, services=[], vulnerabilities=[])
        net.nodes[nid] = node
        net.graph.add_node(nid)

    for edge in topo.get("edges", []):
        u, v = int(edge[0]), int(edge[1])
        if u in net.nodes and v in net.nodes:
            net.graph.add_edge(u, v)

    net.entry_node_id = topo.get("entry_node_id", 0)
    net.target_node_id = topo.get("target_node_id", 0)
    return net


def render_network_graph(
    network: Network | None = None,
    suspicion_data: dict[int, float] | None = None,
    topology_data: dict | None = None,
    seed: int = 42,
) -> None:
    st.subheader("Network Graph")

    if network is None:
        if topology_data is not None:
            network = _network_from_topology(topology_data)
        else:
            network = build_fixed_network(seed=seed)

    entry_id = getattr(network, "entry_node_id", None)
    target_id = getattr(network, "target_node_id", None)

    # Compute layout positions
    pos: dict[int, tuple[float, float]] = nx.spring_layout(network.graph, seed=seed)

    # Build edge traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for u, v in network.graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1.5, color="#555577"),
        hoverinfo="none",
        showlegend=False,
    )

    # Build node traces per node so we can use different symbols and border colours
    node_traces: list[go.Scatter] = []
    for node_id, node in network.nodes.items():
        x, y = pos[node_id]

        if suspicion_data is not None:
            color = _suspicion_color(suspicion_data.get(node_id, 0.0))
        else:
            color = _session_color(node.session_level)

        # Border colour: green for entry, red for target, dark otherwise
        if node_id == entry_id:
            border_color = "#00ff88"
            border_width = 3
        elif node_id == target_id:
            border_color = "#ff4444"
            border_width = 3
        else:
            border_color = "#1a1a2e"
            border_width = 2

        susp_val = suspicion_data.get(node_id, 0.0) if suspicion_data else node.suspicion_level
        label = _NODE_LABELS.get(node_id, f"Node {node_id}")

        role = ""
        if node_id == entry_id:
            role = " [ENTRY]"
        elif node_id == target_id:
            role = " [TARGET]"

        hover = (
            f"<b>{label}{role}</b><br>"
            f"OS: {node.os_type.name}<br>"
            f"Suspicion: {susp_val:.0f}%<br>"
            f"Session: {node.session_level.name}<br>"
            f"Online: {node.is_online}"
        )

        symbol = _OS_SYMBOLS.get(node.os_type, "circle")
        node_trace = go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(
                size=22,
                color=color,
                symbol=symbol,
                line=dict(width=border_width, color=border_color),
            ),
            text=[label],
            textposition="bottom center",
            textfont=dict(color="#e0e0e0", size=10),
            hovertext=[hover],
            hoverinfo="text",
            showlegend=False,
        )
        node_traces.append(node_trace)

    fig = go.Figure(
        data=[edge_trace, *node_traces],
        layout=go.Layout(
            paper_bgcolor="#1a1a2e",
            plot_bgcolor="#1a1a2e",
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=340,
        ),
    )

    # Legend for node colours + special borders
    legend_html = (
        "<div style='display:flex;gap:16px;font-size:12px;color:#ccc;margin-bottom:4px'>"
        "<span style='color:#4caf7d'>● Low suspicion</span>"
        "<span style='color:#f9e45a'>● Medium</span>"
        "<span style='color:#f4a261'>● High</span>"
        "<span style='color:#e05252'>● Critical</span>"
        "<span style='color:#00ff88'>◎ Entry</span>"
        "<span style='color:#ff4444'>◎ Target</span>"
        "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
