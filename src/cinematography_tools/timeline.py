"""
Shot timeline analysis and visualization.

Combines shot-type classifications with timestamps to produce
a cinematic shot breakdown timeline with visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from . import SHOT_TYPES, SHOT_TYPE_LABELS

# Color scheme for shot types (perceptually distinct)
SHOT_COLORS = {
    "LS": "#4A90D9",   # Blue — wide/expansive
    "FS": "#50C878",   # Green — full body
    "MS": "#F4C542",   # Gold — medium
    "CS": "#E8725C",   # Coral — close
    "ECS": "#9B59B6",  # Purple — extreme close
}


def build_timeline(
    frames: List[Path],
    predictions: pd.DataFrame,
    sample_rate: float,
    video_duration: float,
) -> pd.DataFrame:
    """Build a shot timeline from frame predictions.

    Args:
        frames: List of frame file paths.
        predictions: DataFrame with shot-type predictions.
        sample_rate: Frames per second used during extraction.
        video_duration: Total video duration in seconds.

    Returns:
        DataFrame with columns: start_time, end_time, shot_type,
        confidence, duration.
    """
    if predictions.empty:
        return pd.DataFrame(columns=[
            "start_time", "end_time", "shot_type", "confidence", "duration"
        ])

    # Assign timestamps to each prediction
    predictions = predictions.copy()
    predictions["timestamp"] = [i / sample_rate for i in range(len(predictions))]

    # Merge consecutive frames with the same shot type into segments
    segments = []
    current_type = predictions.iloc[0]["shot-type"]
    current_start = 0.0
    confidences = [predictions.iloc[0]["prediction"]]

    for _, row in predictions.iloc[1:].iterrows():
        if row["shot-type"] != current_type:
            # Shot change detected
            segments.append({
                "start_time": round(current_start, 2),
                "end_time": round(row["timestamp"], 2),
                "shot_type": current_type,
                "confidence": round(np.mean(confidences), 2),
                "duration": round(row["timestamp"] - current_start, 2),
            })
            current_type = row["shot-type"]
            current_start = row["timestamp"]
            confidences = [row["prediction"]]
        else:
            confidences.append(row["prediction"])

    # Add final segment
    segments.append({
        "start_time": round(current_start, 2),
        "end_time": round(video_duration, 2),
        "shot_type": current_type,
        "confidence": round(np.mean(confidences), 2),
        "duration": round(video_duration - current_start, 2),
    })

    return pd.DataFrame(segments)


def visualize_timeline(
    timeline: pd.DataFrame,
    output_path: Path,
    title: str = "Shot Type Timeline",
    video_duration: Optional[float] = None,
    palettes: Optional[List[Dict]] = None,
):
    """Generate a visual shot timeline chart.

    Creates a horizontal bar chart showing shot types over time,
    with optional color palette strips below.

    Args:
        timeline: Timeline DataFrame from build_timeline().
        output_path: Path to save the visualization PNG.
        title: Chart title.
        video_duration: Total video duration for x-axis scaling.
        palettes: Optional color palette data for annotation.
    """
    if timeline.empty:
        print("No timeline data to visualize.")
        return

    duration = video_duration or timeline["end_time"].max()

    # Create figure with timeline + optional palette strip
    n_rows = 2 if palettes else 1
    height_ratios = [3, 1] if palettes else [1]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(max(14, duration / 10), 3 + (1 if palettes else 0)),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    fig.patch.set_facecolor("#1a1a2e")

    # ── Shot type timeline ──
    ax = axes[0, 0]
    ax.set_facecolor("#16213e")

    for _, row in timeline.iterrows():
        color = SHOT_COLORS.get(row["shot_type"], "#888888")
        ax.barh(
            0, row["duration"], left=row["start_time"],
            height=0.6, color=color, edgecolor="#1a1a2e", linewidth=0.5,
        )
        # Add label if segment is wide enough
        if row["duration"] > duration * 0.03:
            ax.text(
                row["start_time"] + row["duration"] / 2, 0,
                row["shot_type"],
                ha="center", va="center",
                fontsize=8, fontweight="bold", color="white",
                fontfamily="monospace",
            )

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", color="#e0e0e0", fontsize=10)
    ax.set_title(title, color="#e0e0e0", fontsize=14, fontweight="bold", pad=12)
    ax.tick_params(colors="#e0e0e0", labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#444444")

    # Legend
    legend_patches = [
        mpatches.Patch(
            color=SHOT_COLORS[st],
            label=f"{st} — {SHOT_TYPE_LABELS.get(st, st)}",
        )
        for st in SHOT_TYPES if st in timeline["shot_type"].values
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=7,
        facecolor="#16213e",
        edgecolor="#444444",
        labelcolor="#e0e0e0",
        ncol=len(legend_patches),
    )

    # ── Color palette strip (if provided) ──
    if palettes and n_rows > 1:
        ax2 = axes[1, 0]
        ax2.set_facecolor("#16213e")
        ax2.set_yticks([])
        ax2.set_xlim(0, duration)

        # Map palette entries to timeline segments
        n_frames = len(palettes)
        for i, pal_data in enumerate(palettes):
            t = i / n_frames * duration
            w = duration / n_frames
            palette = pal_data.get("palette", [])
            if palette:
                n_colors = len(palette)
                for j, color_info in enumerate(palette):
                    y = j / n_colors
                    h = 1.0 / n_colors
                    hex_color = color_info.get("hex", "#000000")
                    ax2.add_patch(plt.Rectangle(
                        (t, y), w, h,
                        facecolor=hex_color,
                        edgecolor="none",
                    ))

        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Color Palette", color="#e0e0e0", fontsize=9)
        ax2.tick_params(colors="#e0e0e0", labelsize=8)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["bottom"].set_color("#444444")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  ✅ Timeline saved to {output_path}")


def generate_summary(
    timeline: pd.DataFrame,
    output_path: Path,
    palettes: Optional[List[Dict]] = None,
):
    """Generate a summary statistics file for the video analysis.

    Args:
        timeline: Timeline DataFrame.
        output_path: Path to save the summary JSON.
        palettes: Optional color palette data.
    """
    if timeline.empty:
        summary = {"error": "No shots detected"}
    else:
        total_duration = timeline["end_time"].max() - timeline["start_time"].min()

        # Shot type distribution
        distribution = {}
        for st in SHOT_TYPES:
            st_rows = timeline[timeline["shot_type"] == st]
            if not st_rows.empty:
                total_time = st_rows["duration"].sum()
                distribution[st] = {
                    "label": SHOT_TYPE_LABELS.get(st, st),
                    "count": int(len(st_rows)),
                    "total_duration": round(total_time, 2),
                    "percentage": round(total_time / total_duration * 100, 2),
                    "avg_duration": round(st_rows["duration"].mean(), 2),
                }

        summary = {
            "total_duration": round(total_duration, 2),
            "total_shots": len(timeline),
            "shot_distribution": distribution,
            "avg_shot_duration": round(timeline["duration"].mean(), 2),
            "shortest_shot": round(timeline["duration"].min(), 2),
            "longest_shot": round(timeline["duration"].max(), 2),
        }

        # Add overall color palette if available
        if palettes:
            # Aggregate all palette colors across the video
            all_colors = []
            for pal in palettes:
                for c in pal.get("palette", []):
                    all_colors.append(c["hex"])
            if all_colors:
                # Top 10 most frequent colors
                from collections import Counter
                color_counts = Counter(all_colors).most_common(10)
                summary["dominant_colors"] = [
                    {"hex": hex_val, "frequency": count}
                    for hex_val, count in color_counts
                ]

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved to {output_path}")

    return summary
