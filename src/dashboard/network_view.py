"""NetworkX graph rendering for the Streamlit training dashboard.

Renders the fixed 8-node enterprise network as an interactive Plotly figure,
with nodes coloured by suspicion level (green → yellow → red scale).
"""

from __future__ import annotations

import networkx as nx
import plotly.graph_objects as go
import streamlit as st

from src.environment.network import Network, build_fixed_network
from src.environment.node import OsType, SessionLevel


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


def render_network_graph(
    network: Network | None = None,
    suspicion_data: dict[int, float] | None = None,
    seed: int = 42,
) -> None:
    """Render the network graph as an interactive Plotly chart in Streamlit.

    Args:
        network: The Network to visualise. If None, builds the default fixed network.
        suspicion_data: Optional dict mapping node_id → suspicion level (0-100).
                        When provided, nodes are coloured by suspicion.
                        When None, nodes are coloured by session level.
        seed: Layout seed for reproducible spring_layout positions.
    """
    st.subheader("Network Graph")

    if network is None:
        network = build_fixed_network(seed=seed)

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

    # Build node traces per OS type so we can use different symbols
    node_traces: list[go.Scatter] = []
    for node_id, node in network.nodes.items():
        x, y = pos[node_id]

        if suspicion_data is not None:
            color = _suspicion_color(suspicion_data.get(node_id, 0.0))
        else:
            color = _session_color(node.session_level)

        susp_val = suspicion_data.get(node_id, 0.0) if suspicion_data else node.suspicion_level
        label = _NODE_LABELS.get(node_id, f"Node {node_id}")
        hover = (
            f"<b>{label}</b><br>"
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
                line=dict(width=2, color="#1a1a2e"),
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

    # Legend for node colours
    legend_html = (
        "<div style='display:flex;gap:16px;font-size:12px;color:#ccc;margin-bottom:4px'>"
        "<span style='color:#4caf7d'>● Low suspicion</span>"
        "<span style='color:#f9e45a'>● Medium</span>"
        "<span style='color:#f4a261'>● High</span>"
        "<span style='color:#e05252'>● Critical</span>"
        "</div>"
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True)
