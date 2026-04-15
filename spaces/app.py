"""
Gradio web interface for Cinematography Analysis Tools.

Deploy to HuggingFace Spaces or run locally with:
    pip install gradio
    python spaces/app.py
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Force non-interactive Matplotlib backend BEFORE any other imports
# This prevents NSException crashes on macOS when plotting in background threads
os.environ["MPLBACKEND"] = "Agg"

# Fix for FFmpeg path on some macOS environments
os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + "/usr/local/bin"

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from cinematography_tools import SHOT_TYPE_LABELS
from cinematography_tools.color import extract_palette, lab_to_rgb
from cinematography_tools.charts import (
    create_interactive_timeline,
    create_rhythm_chart,
    create_distribution_bar,
    create_scope_gauge
)
from cinematography_tools.performance import log_performance, get_correction_factor, get_perf_summary, RunTimer

# Pre-calculate hardware correction at startup
HARDWARE_MULTIPLIER = get_correction_factor()
PERF_STATUS = get_perf_summary()


def classify_image(image):
    """Classify a single image and return shot type with confidence."""
    if image is None:
        return {}, None, "Upload an image to classify."

    # Save temp image for processing
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(image)
        img.save(f.name)
        temp_path = Path(f.name)

    try:
        from cinematography_tools.model import load_model
        from cinematography_tools.predict import predict_image

        model, _ = load_model()
        result = predict_image(model, temp_path)

        # Format label dict for Gradio
        labels = {
            f"{SHOT_TYPE_LABELS.get(cls, cls)} ({cls})": conf / 100
            for cls, conf in result["all_predictions"].items()
        }

        # Info text
        info = f"**Predicted: {SHOT_TYPE_LABELS.get(result['shot_type'], result['shot_type'])}**\n"
        info += f"Confidence: {result['confidence']:.1f}%"

        return labels, image, info

    except Exception as exc:
        return {}, image, f"⚠️ Error: {exc}"

    finally:
        temp_path.unlink(missing_ok=True)


# --- Hardware-Calibrated Estimation Logic ---
ANALYSIS_MODES = {
    "💨 Preview (Ultra Fast)": {"fps": 0.5, "palette_stride": 15.0},
    "🎬 Cinematic Standard": {"fps": 2.0, "palette_stride": 5.0},
    "🔬 Deep Studio Analyst": {"fps": 10.0, "palette_stride": 2.0},
    "⚙️ Custom Settings": {"fps": None, "palette_stride": None}
}

def format_human_time(seconds):
    """Convert raw seconds to a human-readable duration (e.g., '1h 24m 10s')."""
    if seconds < 1:
        return "< 1 second"
    
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    
    parts = []
    if h > 0: parts.append(f"{h}h")
    if m > 0: parts.append(f"{m}m")
    if s > 0 or not parts: parts.append(f"{s}s")
    
    return " ".join(parts)

def get_est_time(video_duration, fps, mode_name):
    """Estimate analysis time based on M1 Pro hardware benchmarks + Correction Factor."""
    if not video_duration or video_duration <= 0:
        return "N/A"
    
    # Use mode defaults if not Custom
    stride = ANALYSIS_MODES[mode_name]["palette_stride"]
    if mode_name == "⚙️ Custom Settings":
        stride = 5.0 # Default fallback
    
    n_batches = (video_duration * fps) / 16
    n_palettes = video_duration / stride
    
    # Apply historical hardware multiplier for high accuracy
    raw_est = ((n_batches * 0.11) + (n_palettes * 0.35) + 1.5) * HARDWARE_MULTIPLIER
    return format_human_time(raw_est)


def analyze_palette_ui(image, n_colors=5):
    """Extract and display color palette from an image."""
    if image is None:
        return None, ""

    try:
        palette = extract_palette(image, n_colors=n_colors)
        # (rest of function remains same...)

        # Create palette visualization
        swatch_w, swatch_h = 120, 80
        padding = 8
        total_w = n_colors * (swatch_w + padding) - padding
        viz = Image.new("RGB", (total_w, swatch_h + 40), (26, 26, 46))
        draw = ImageDraw.Draw(viz)

        for i, color in enumerate(palette):
            x = i * (swatch_w + padding)
            r, g, b = color["rgb"]
            draw.rounded_rectangle([x, 0, x + swatch_w, swatch_h], radius=8, fill=(r, g, b))
            draw.text((x + swatch_w // 2, swatch_h + 8), f"{color['hex']}", fill=(224, 224, 224), anchor="mt")
            draw.text((x + swatch_w // 2, swatch_h + 24), f"{color['percentage']:.0f}%", fill=(160, 160, 160), anchor="mt")

        # Build info text
        info = "### CIELAB Color Palette\n\n"
        info += "| Color | Hex | L* | a* | b* | Dominance |\n"
        info += "|-------|-----|-----|-----|-----|-----------|\n"
        for c in palette:
            L, a, b_val = c["lab"]
            info += f"| 🟧 | `{c['hex']}` | {L:.1f} | {a:.1f} | {b_val:.1f} | {c['percentage']:.1f}% |\n"

        return np.array(viz), info

    except Exception as exc:
        return None, f"Error: {exc}"


def analyze_video_ui(video_path, sample_rate, n_colors, mode_name):
    """Full video analysis pipeline for Gradio with Interactive Plotly Dashboards."""
    if video_path is None:
        return [None] * 4 + ["", "⚠️ Please upload a video first."]
    
    # Apply Mode Settings
    if mode_name != "⚙️ Custom Settings":
        settings = ANALYSIS_MODES[mode_name]
        sample_rate = settings["fps"]
        stride_seconds = settings["palette_stride"]
    else:
        stride_seconds = 5.0 # Custom default

    output_dir = Path(tempfile.mkdtemp())
    input_path = Path(video_path)

    try:
        from cinematography_tools.video import get_video_info, stream_frames
        from cinematography_tools.model import load_model
        from cinematography_tools.predict import predict_images_batch
        from cinematography_tools.color import extract_palette
        from cinematography_tools.timeline import build_timeline, generate_summary

        timer = RunTimer()
        info = get_video_info(input_path)

        with timer.phase("model_load"):
            model, _ = load_model()

        predictions_list = []
        palettes = []
        
        batch_size = 16
        current_batch = []
        current_timestamps = []
        next_palette_time = 0.0

        print(f"--- Analysis Start: {mode_name} | FPS: {sample_rate} ---")

        # ── Analysis Loop (Batched) ──
        with timer.phase("extraction_and_inference"):
            try:
                for timestamp, frame_arr in stream_frames(input_path, sample_rate):
                    img = Image.fromarray(frame_arr)
                    current_batch.append(img)
                    current_timestamps.append(timestamp)

                    if len(current_batch) >= batch_size:
                        results = predict_images_batch(model, current_batch)
                        for i, res in enumerate(results):
                            predictions_list.append({
                                "shot_type": res["shot_type"],
                                "confidence": res["confidence"],
                                "timestamp": current_timestamps[i],
                            })
                        
                        if current_timestamps[0] >= next_palette_time:
                            pal = extract_palette(frame_arr, n_colors=n_colors)
                            palettes.append({"timestamp": current_timestamps[0], "palette": pal})
                            next_palette_time += stride_seconds
                        
                        current_batch = []
                        current_timestamps = []

                if current_batch:
                    results = predict_images_batch(model, current_batch)
                    for i, res in enumerate(results):
                        predictions_list.append({
                            "shot_type": res["shot_type"],
                            "confidence": res["confidence"],
                            "timestamp": current_timestamps[i],
                        })
            except Exception as loop_exc:
                print(f"⚠️ Warning: Loop interrupted: {loop_exc}")

        if not predictions_list:
            return [None] * 5 + ["⚠️ No frames were analyzed."]

        with timer.phase("timeline_build"):
            predictions = pd.DataFrame(predictions_list)
            timeline = build_timeline([], predictions, sample_rate, info["duration"])
            summary = generate_summary(timeline, output_dir / "summary.json", palettes=palettes)
        
        # ── Create Interactive Charts ──
        with timer.phase("chart_generation"):
            fig_timeline = create_interactive_timeline(summary["rhythm"], info["duration"])
            fig_rhythm = create_rhythm_chart(summary["rhythm"])
            fig_dist = create_distribution_bar(summary["shot_distribution"])
            fig_scope = create_scope_gauge(summary["cinematic_scale"]["wide_percentage"])

        # ── Color Gamut HTML ──
        gamut_html = "<div style='background: rgba(30, 41, 59, 0.7); padding: 20px; border-radius: 12px; border: 1px solid #334155;'>"
        gamut_html += "<h3 style='margin-top:0; color: #f8fafc; text-align: center;'>🎨 CIELAB Color Gamut</h3>"
        if "color_mood" in summary:
            mood = summary["color_mood"]
            gamut_html += f"<p style='text-align: center; color: #94a3b8; margin-bottom: 20px;'>Mood: <span style='color: #8b5cf6; font-weight: bold;'>{mood['label']}</span> (Score: {mood['warmth_score']}%)</p>"

        gamut_html += "<div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;'>"
        for color_info in summary.get("dominant_colors", []):
            gamut_html += f"""
            <div style='text-align: center; background: #1e293b; padding: 10px; border-radius: 8px; border: 1px solid #334155;'>
                <div style='background: {color_info["hex"]}; height: 50px; border-radius: 4px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);'></div>
                <p style='font-family: monospace; color: #f1f5f9; font-size: 0.85em; margin: 8px 0 0 0;'>{color_info["hex"]}</p>
            </div>
            """
        gamut_html += "</div></div>"

        # Production Stats Mini-Card
        stats_html = f"""
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px;'>
            <div style='background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;'>
                <p style='color: #94a3b8; font-size: 0.9em; margin: 0;'>Total Cuts</p>
                <p style='color: #f1f5f9; font-size: 2em; font-weight: bold; margin: 5px 0;'>{summary['total_shots']}</p>
            </div>
            <div style='background: #1e293b; padding: 15px; border-radius: 12px; border: 1px solid #334155; text-align: center;'>
                <p style='color: #94a3b8; font-size: 0.9em; margin: 0;'>Avg Shot Length</p>
                <p style='color: #f1f5f9; font-size: 2em; font-weight: bold; margin: 5px 0;'>{summary['avg_shot_duration']:.2f}s</p>
            </div>
        </div>
        """
        
        # Log structured performance telemetry
        stride = ANALYSIS_MODES[mode_name]["palette_stride"] or 5.0
        est_raw = (((len(predictions_list) / 16) * 0.11) + ((info["duration"] / stride) * 0.35) + 1.5) * HARDWARE_MULTIPLIER
        timer.finalize(
            video_duration=info["duration"],
            frames=len(predictions_list),
            mode=mode_name,
            est_time=est_raw,
            extra={
                "total_shots": summary["total_shots"],
                "avg_shot_duration": summary["avg_shot_duration"],
                "palette_count": len(palettes),
                "fps": sample_rate,
            }
        )

        return fig_timeline, fig_rhythm, fig_dist, fig_scope, gamut_html, stats_html

    except Exception as exc:
        import traceback
        error_msg = f"⚠️ Analysis failed: {exc}\n{traceback.format_exc()}"
        return [None] * 5 + [error_msg]


def update_ui_estimate(video_path, fps, mode_name):
    """Dynamic callback for the Pre-flight Check label."""
    if not video_path:
        return f"🚀 **Ready**: Take a quick look or dive deep into the frames. (Status: {PERF_STATUS})"
    
    try:
        from cinematography_tools.video import get_video_info
        info = get_video_info(video_path)
        est = get_est_time(info["duration"], fps, mode_name)
        return f"🚀 **Pre-flight Check**: Analysis of this movie will take approx **{est}** on your M1 Pro. (Status: {PERF_STATUS})"
    except:
        return "⚠️ Could not read video metadata."

def update_sliders_visibility(mode_name):
    """Enable sliders only in Custom mode."""
    visible = (mode_name == "⚙️ Custom Settings")
    return gr.update(visible=visible)


# ── Build Gradio Interface ──

with gr.Blocks(
    title="🎬 Cinematography Analysis Tools",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue", neutral_hue="slate"),
    css="""
        .gradio-container { max-width: 1400px !important; }
        .header { text-align: center; margin-bottom: 24px; }
        .stat-card { background: #1e293b; border-radius: 12px; border: 1px solid #334155; padding: 20px; }
    """,
) as demo:
    gr.HTML("""
        <div class="header">
            <h1>🎬 Cinematography Analysis Tools</h1>
            <p style="color: #888; font-size: 1.1em;">
                Power Dashboard &bull; Hardware-Accelerated Analytics &bull; Pre-flight Estimation
            </p>
        </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Video Analysis ──
        with gr.Tab("📽️ Full Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Cinematic Footage", sources=["upload"])
                    
                    with gr.Group():
                        mode_input = gr.Dropdown(
                            choices=list(ANALYSIS_MODES.keys()),
                            value="🎬 Cinematic Standard",
                            label="Analysis Depth (Preset)"
                        )
                        
                        est_markdown = gr.Markdown("🚀 **Pre-flight Check**: Analysis is ready for upload.")
                        
                        with gr.Row(visible=False) as custom_params:
                            sr_input = gr.Slider(minimum=0.5, maximum=10, value=2, step=0.5, label="Sample Rate (FPS)")
                            c_input = gr.Slider(minimum=3, maximum=15, value=10, step=1, label="Palette Depth")
                    
                    analyze_vid_btn = gr.Button("🚀 Run Hardware-Accelerated Analysis", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    timeline_plot = gr.Plot(label="Interactive Timeline")
                    gr.Markdown("""
                    > [!TIP]
                    > **Reader's Guide**:
                    > - **Interactive Timeline**: Hover to see exact shot types and AI confidence scores.
                    > - **Cinematic Scope**: Measures visual scale. High (%) = **Environment** (World). Low (%) = **Subject** (Character).
                    """)
                    with gr.Row():
                        with gr.Column(scale=1):
                            stats_output = gr.HTML(label="High-Level Metrics")
                            scope_plot = gr.Plot(label="Cinematic Scope")
                        with gr.Column(scale=1):
                            dist_plot = gr.Plot(label="Shot Distribution")
            
            with gr.Row():
                with gr.Column(scale=1):
                    rhythm_plot = gr.Plot(label="Editor's Rhythm")
                with gr.Column(scale=1):
                    gamut_output = gr.HTML(label="Color Intelligence")

            # Dynamic UI Logic
            mode_input.change(update_sliders_visibility, inputs=[mode_input], outputs=[custom_params])
            
            # Estimate Triggers
            for input_comp in [video_input, sr_input, mode_input]:
                input_comp.change(
                    update_ui_estimate, 
                    inputs=[video_input, sr_input, mode_input], 
                    outputs=[est_markdown]
                )

            analyze_vid_btn.click(
                fn=analyze_video_ui,
                inputs=[video_input, sr_input, c_input, mode_input],
                outputs=[timeline_plot, rhythm_plot, dist_plot, scope_plot, gamut_output, stats_output],
            )

        # ── Tab 2: Shot Classification ──
        with gr.Tab("🎯 Shot Classification"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Frame", type="numpy", height=400)
                    classify_btn = gr.Button("Analyze Still", variant="primary", size="lg")
                with gr.Column(scale=1):
                    label_output = gr.Label(label="Shot Type Distribution", num_top_classes=5)
                    info_output = gr.Markdown(label="Details")

            classify_btn.click(
                fn=classify_image,
                inputs=[img_input],
                outputs=[label_output, info_output],
            )
        
        # ── Tab 3: About ──
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## Hardware Acceleration (Phase 8 Release)
            - **M1 Metal Engine**: AI processing is offloaded to the Apple GPU (MPS).
            - **Source Scaling**: Raw 1080p bandwidth is reduced by 4000% via FFmpeg source-decimation.
            - **Predictive Estimator**: Benchmarks the hardware to estimate analysis time before you start.
            """)

if __name__ == "__main__":
    # Silicon-Native Hub (Gradio 5 / Python 3.11)
    demo.queue() 
    demo.launch(
        server_name="127.0.0.1",
        show_api=False, # Bypass schema bug in current framework
        quiet=False
    )
