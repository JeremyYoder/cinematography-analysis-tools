"""
Interactive Plotly visualizations for cinematography analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List, Dict, Optional
from .timeline import SHOT_COLORS, SHOT_TYPE_LABELS

def decimate_data(df: pd.DataFrame, limit: int = 2000) -> pd.DataFrame:
    """Downsample data for visualization if it exceeds the limit to prevent crashes."""
    if len(df) <= limit:
        return df
    
    # Use a simple strided sampling for the UI
    step = len(df) // limit
    return df.iloc[::step].copy()

def create_interactive_timeline(timeline_data: List[Dict], total_duration: float):
    """Create a Plotly Gantt-style timeline for shot breakdown."""
    if not timeline_data:
        return go.Figure()

    df = pd.DataFrame(timeline_data)
    df = decimate_data(df, limit=1500)  # Lower limit for bars as they are heavier
    df["label"] = df["shot_type"].map(SHOT_TYPE_LABELS)
    
    # Create the figure
    fig = go.Figure()

    for _, row in df.iterrows():
        color = SHOT_COLORS.get(row["shot_type"], "#888888")
        fig.add_trace(go.Bar(
            x=[row["duration"]],
            y=["Shots"],
            base=row["start_time"],
            orientation="h",
            marker=dict(color=color, line=dict(color="#0f172a", width=1)),
            name=row["shot_type"],
            hovertemplate=(
                f"<b>{row['label']} ({row['shot_type']})</b><br>" +
                f"Time: {row['start_time']:.2f}s - {row['start_time']+row['duration']:.2f}s<br>" +
                f"Duration: {row['duration']:.2f}s<br>" +
                f"Confidence: {row.get('confidence', 0):.1f}%" +
                "<extra></extra>"
            ),
            showlegend=False
        ))

    fig.update_layout(
        barmode="stack",
        height=220,
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis=dict(title="Time (seconds)", color="#94a3b8", gridcolor="#334155", range=[0, total_duration]),
        yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Shot Breakdown Timeline", font=dict(color="#f8fafc", size=16)),
        hovermode="closest"
    )

    return fig

def create_rhythm_chart(rhythm_data: List[Dict]):
    """Create a line chart showing cinematic rhythm (shot durations)."""
    if not rhythm_data:
        return go.Figure()

    df = pd.DataFrame(rhythm_data)
    df = decimate_data(df, limit=2000)
    
    fig = px.line(
        df, x="start_time", y="duration",
        labels={"start_time": "Time (s)", "duration": "Shot Length (s)"},
        template="plotly_dark",
        markers=True
    )
    
    fig.update_traces(
        line=dict(color="#8b5cf6", width=3),
        fill="tozeroy",
        hovertemplate="Time: %{x:.1f}s<br>Shot Length: %{y:.2f}s<extra></extra>"
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis=dict(color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#94a3b8", gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.5)",
        title=dict(text="Cinematic Rhythm (Cutting Rate)", font=dict(color="#f8fafc", size=16))
    )
    
    return fig

def create_distribution_bar(distribution: Dict):
    """Create a labeled horizontal bar chart for shot types."""
    if not distribution:
        return go.Figure()

    labels = []
    values = []
    colors = []
    hover_texts = []

    for st, data in distribution.items():
        labels.append(f"{st} ({data['percentage']}%)")
        values.append(data["percentage"])
        colors.append(SHOT_COLORS.get(st, "#888"))
        hover_texts.append(f"{st}: {data['count']} shots, {data['total_duration']}s total")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        text=labels,
        textposition="inside",
        insidetextanchor="middle",
        textfont=dict(color="white", size=12),
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(title="Percentage (%)", color="#94a3b8", gridcolor="#334155"),
        yaxis=dict(color="#94a3b8", autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        title=dict(text="Shot Type Distribution", font=dict(color="#f8fafc", size=16))
    )

    return fig

def create_scope_gauge(wide_perc: float):
    """Create a gauge for Wide-vs-Tight balance."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=wide_perc,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Cinematic Scope (Wide %)", 'font': {'color': "#f8fafc", 'size': 16}},
        number={'font': {'color': "#8b5cf6", 'size': 32}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94a3b8"},
            'bar': {'color': "#8b5cf6"},
            'bgcolor': "#1e293b",
            'borderwidth': 2,
            'bordercolor': "#334155",
            'steps': [
                {'range': [0, 40], 'color': '#334155'},
                {'range': [40, 60], 'color': '#475569'},
                {'range': [60, 100], 'color': '#334155'}
            ],
            'threshold': {
                'line': {'color': "#f87171", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig
