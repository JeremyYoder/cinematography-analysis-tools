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
    current_type = predictions.iloc[0]["shot_type"]
    current_start = 0.0
    confidences = [predictions.iloc[0]["confidence"]]

    for _, row in predictions.iloc[1:].iterrows():
        if row["shot_type"] != current_type:
            # Shot change detected
            segments.append({
                "start_time": round(current_start, 2),
                "end_time": round(row["timestamp"], 2),
                "shot_type": current_type,
                "confidence": round(np.mean(confidences), 2),
                "duration": round(row["timestamp"] - current_start, 2),
            })
            current_type = row["shot_type"]
            current_start = row["timestamp"]
            confidences = [row["confidence"]]
        else:
            confidences.append(row["confidence"])

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
    title: str = "Cinematic Shot Breakdown",
    video_duration: Optional[float] = None,
    palettes: Optional[List[Dict]] = None,
):
    """Generate a high-definition visual shot timeline chart."""
    if timeline.empty:
        return

    duration = video_duration or timeline["end_time"].max()

    # Create figure with high-definition proportions
    n_rows = 2 if palettes else 1
    height_ratios = [4, 1.2] if palettes else [1]
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(16, 4.5 if palettes else 3),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )
    fig.patch.set_facecolor("#0f172a") # Darker slate background

    # ── Shot type timeline ──
    ax = axes[0, 0]
    ax.set_facecolor("#1e293b")

    for _, row in timeline.iterrows():
        color = SHOT_COLORS.get(row["shot_type"], "#888888")
        ax.barh(
            0, row["duration"], left=row["start_time"],
            height=0.8, color=color, edgecolor="#0f172a", linewidth=1,
        )
        if row["duration"] > duration * 0.02:
            ax.text(
                row["start_time"] + row["duration"] / 2, 0,
                row["shot_type"],
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                alpha=0.9
            )

    ax.set_xlim(0, duration)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel("Time (seconds)", color="#94a3b8", fontsize=11, labelpad=8)
    ax.set_title(title, color="#f8fafc", fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(colors="#94a3b8", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color("#334155")
    
    # Premium Legend
    legend_patches = [
        mpatches.Patch(color=SHOT_COLORS[st], label=f"{st} — {SHOT_TYPE_LABELS.get(st, st)}")
        for st in SHOT_TYPES if st in timeline["shot_type"].values
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        fontsize=9,
        facecolor="#1e293b",
        edgecolor="#334155",
        labelcolor="#f1f5f9",
        ncol=len(legend_patches),
        bbox_to_anchor=(1.0, 1.15)
    )

    # ── Color gamut strip ──
    if palettes and n_rows > 1:
        ax2 = axes[1, 0]
        ax2.set_facecolor("#1e293b")
        ax2.set_yticks([])
        ax2.set_xlim(0, duration)

        n_frames = len(palettes)
        for i, pal_data in enumerate(palettes):
            t = pal_data.get("timestamp", i / n_frames * duration)
            w = (duration / n_frames) * 1.5 # Slight overlap for smoothness
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
        ax2.set_xlabel("Global Color Palette", color="#94a3b8", fontsize=10)
        ax2.tick_params(colors="#94a3b8", labelsize=10)
        for spine in ax2.spines.values():
            spine.set_color("#334155")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
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
            "rhythm": timeline[["start_time", "duration", "shot_type"]].to_dict("records"),
        }

        # Cinematic Scale (Wide vs Tight)
        wide_count = sum(distribution.get(st, {}).get("count", 0) for st in ["LS", "FS"])
        tight_count = sum(distribution.get(st, {}).get("count", 0) for st in ["MS", "CS", "ECS"])
        summary["cinematic_scale"] = {
            "wide_count": wide_count,
            "tight_count": tight_count,
            "wide_percentage": round(wide_count / max(1, len(timeline)) * 100, 1),
        }

        # Add overall color palette if available
        if palettes:
            all_colors = []
            lab_b_values = []
            for pal in palettes:
                for c in pal.get("palette", []):
                    all_colors.append(c["hex"])
                    if "lab" in c:
                        lab_b_values.append(c["lab"][2]) # b* channel

            if all_colors:
                from collections import Counter
                color_counts = Counter(all_colors).most_common(12)
                summary["dominant_colors"] = [
                    {"hex": hex_val, "frequency": count}
                    for hex_val, count in color_counts
                ]

            if lab_b_values:
                avg_b = sum(lab_b_values) / len(lab_b_values)
                summary["color_mood"] = {
                    "avg_b": round(avg_b, 2),
                    "label": "Warm / Golden" if avg_b > 2 else "Cool / Melancholic" if avg_b < -2 else "Neutral / Balanced",
                    "warmth_score": round(min(max((avg_b + 20) / 40 * 100, 0), 100), 1) # Simple 0-100 score
                }

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✅ Summary saved to {output_path}")

    return summary
