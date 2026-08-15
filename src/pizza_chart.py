"""Opta-style percentile 'pizza' chart, built with Plotly's Barpolar so it
stays interactive (hover tooltips) inside the live Streamlit dashboard.

Percentiles are computed against a same-position reference population
(see player_comparison_data.compute_percentiles), matching the framing used
by Opta Analyst's own charts: each wedge shows where a player-season ranks
relative to their positional peers, not a raw stat value.
"""
import plotly.graph_objects as go

from player_comparison_data import CATEGORIES, CATEGORY_GROUP

GROUP_COLORS = {
    "Attacking": "#C4614A",   # rust
    "Creative": "#79AE99",    # sage
    "Defensive": "#E9A13B",   # marigold
}
INK = "#0D141D"
BONE = "#ECE7DB"
BONE_DIM = "#9FA8B3"
LINE = "#2B3A4C"


def build_pizza_chart(player_row, title: str) -> go.Figure:
    """player_row: a Series with '{col}_pct' and raw '{col}' values for
    every key in CATEGORIES (i.e. one row from compute_percentiles())."""
    cols = list(CATEGORIES.keys())
    labels = [CATEGORIES[c] for c in cols]
    percentiles = [player_row[f"{c}_pct"] for c in cols]
    raw_values = [player_row[c] for c in cols]
    colors = [GROUP_COLORS[CATEGORY_GROUP[c]] for c in cols]
    hover = [
        f"{lbl}<br>{val:.2f} per 90<br>{pct:.0f}th percentile"
        for lbl, val, pct in zip(labels, raw_values, percentiles)
    ]

    n = len(cols)
    theta = [i * 360 / n for i in range(n)]

    fig = go.Figure(go.Barpolar(
        r=percentiles,
        theta=theta,
        width=[360 / n * 0.92] * n,
        marker_color=colors,
        marker_line_color=INK,
        marker_line_width=2,
        opacity=0.9,
        text=hover,
        hovertemplate="%{text}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(color=BONE, size=14), x=0.5, xanchor="center"),
        polar=dict(
            bgcolor=INK,
            radialaxis=dict(range=[0, 100], showticklabels=False, ticks="", gridcolor=LINE, linecolor=LINE),
            angularaxis=dict(
                tickmode="array", tickvals=theta, ticktext=labels,
                gridcolor=LINE, linecolor=LINE, tickfont=dict(color=BONE_DIM, size=10),
            ),
        ),
        paper_bgcolor=INK, plot_bgcolor=INK,
        showlegend=False, margin=dict(l=40, r=40, t=65, b=30), height=445,
    )
    return fig
