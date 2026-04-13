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
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from cinematography_tools import SHOT_TYPE_LABELS
from cinematography_tools.color import extract_palette, lab_to_rgb


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


def analyze_palette_ui(image, n_colors=5):
    """Extract and display color palette from an image."""
    if image is None:
        return None, ""

    try:
        palette = extract_palette(image, n_colors=n_colors)

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


def analyze_video_ui(video_path, sample_rate, n_colors):
    """Full video analysis pipeline for Gradio."""
    if video_path is None:
        return None, "Upload a video file to begin analysis."

    output_dir = Path(tempfile.mkdtemp())
    input_path = Path(video_path)

    try:
        # We reuse the logic from cli.py but slightly wrapped for Gradio
        from cinematography_tools.video import get_video_info, stream_frames
        from cinematography_tools.model import load_model
        from cinematography_tools.predict import predict_image
        from cinematography_tools.color import extract_palette
        from cinematography_tools.timeline import build_timeline, visualize_timeline, generate_summary

        info = get_video_info(input_path)
        model, _ = load_model()

        predictions_list = []
        palettes = []
        frames_extracted = 0
        
        # Calculate palette interval
        expected_frames = int(info["duration"] * sample_rate)
        palette_interval = max(1, expected_frames // 50)

        for timestamp, frame_arr in stream_frames(input_path, sample_rate):
            img = Image.fromarray(frame_arr)
            result = predict_image(model, img)
            predictions_list.append({
                "shot-type": result["shot_type"],
                "prediction": result["confidence"],
                "timestamp": timestamp,
            })
            
            if frames_extracted % palette_interval == 0:
                pal = extract_palette(frame_arr, n_colors=n_colors)
                palettes.append({"timestamp": timestamp, "palette": pal})
                
            frames_extracted += 1

        if not predictions_list:
            return None, "No frames were analyzed."

        predictions = pd.DataFrame(predictions_list)
        timeline = build_timeline([], predictions, sample_rate, info["duration"])
        
        timeline_img_path = output_dir / "timeline.png"
        visualize_timeline(
            timeline, timeline_img_path, 
            title=f"Shot Breakdown — {input_path.name}",
            video_duration=info["duration"],
            palettes=palettes if len(palettes) > 5 else None
        )

        summary = generate_summary(timeline, output_dir / "summary.json", palettes=palettes)
        
        summary_text = f"### Analysis Complete\n\n"
        summary_text += f"- **Total Shots:** {summary['total_shots']}\n"
        summary_text += f"- **Avg Shot Duration:** {summary['avg_shot_duration']:.2f}s\n"
        summary_text += f"- **Dominant Colors:** " + ", ".join([c['hex'] for c in summary.get('dominant_colors', [])[:5]])

        return str(timeline_img_path), summary_text

    except Exception as exc:
        return None, f"⚠️ Analysis failed: {exc}"


# ── Build Gradio Interface ──

with gr.Blocks(
    title="🎬 Cinematography Analysis Tools",
    theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue", neutral_hue="slate"),
    css=".gradio-container { max-width: 1000px !important; } .header { text-align: center; margin-bottom: 24px; }",
) as demo:
    gr.HTML("""
        <div class="header">
            <h1>🎬 Cinematography Analysis Tools</h1>
            <p style="color: #888; font-size: 1.1em;">
                Native PyTorch Modernization &bull; Zero-IO Video Ingestion &bull; CIELAB Color Analysis
            </p>
        </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Video Analysis ──
        with gr.TabItem("📽️ Full Video Analysis"):
            gr.Markdown("Analyze a complete movie file to generate a shot-type timeline and color distribution.")
            with gr.Row():
                with gr.Column(scale=1):
                    video_input = gr.Video(label="Upload Video")
                    with gr.Row():
                        sr_input = gr.Slider(minimum=0.5, maximum=10, value=2, step=0.5, label="Sample Rate (FPS)")
                        c_input = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Palette Colors")
                    analyze_vid_btn = gr.Button("Run Full Analysis", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    timeline_output = gr.Image(label="Shot Visual Timeline")
                    summary_output = gr.Markdown(label="Video Stats")

            analyze_vid_btn.click(
                fn=analyze_video_ui,
                inputs=[video_input, sr_input, c_input],
                outputs=[timeline_output, summary_output],
            )

        # ── Tab 2: Shot Classification ──
        with gr.TabItem("🎯 Shot Classification"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Upload Frame", type="numpy", height=400)
                    classify_btn = gr.Button("Classify Shot Type", variant="primary", size="lg")

                with gr.Column(scale=1):
                    label_output = gr.Label(label="Shot Type Prediction", num_top_classes=5)
                    info_output = gr.Markdown(label="Details")

            classify_btn.click(
                fn=classify_image,
                inputs=[img_input],
                outputs=[label_output, gr.Image(visible=False), info_output],
            )

        # ── Tab 3: Color Palette ──
        with gr.TabItem("🎨 Color Palette"):
            with gr.Row():
                with gr.Column(scale=1):
                    pal_input = gr.Image(label="Upload Frame", type="numpy", height=400)
                    pal_colors = gr.Slider(minimum=3, maximum=10, value=5, step=1, label="Number of Colors")
                    palette_btn = gr.Button("Extract Palette", variant="primary", size="lg")

                with gr.Column(scale=1):
                    palette_viz = gr.Image(label="Color Palette", height=160)
                    palette_info = gr.Markdown(label="Palette Details")

            palette_btn.click(
                fn=analyze_palette_ui,
                inputs=[pal_input, pal_colors],
                outputs=[palette_viz, palette_info],
            )

        # ── Tab 4: About ──
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## Shot Types
            | Shot Type | Description |
            |-----------|-------------|
            | **LS** — Long Shot | Wide establishing shots showing full environment |
            | **FS** — Full Shot | Subject visible from head to toe |
            | **MS** — Medium Shot | Subject from waist up |
            | **CS** — Close Shot | Subject's face and shoulders |
            | **ECS** — Extreme Close Shot | Tight framing on a detail |

            ## Optimization
            This environment uses the **V2 Modernized Engine**:
            - **Native PyTorch**: Removed legacy FastAI v1 dependencies.
            - **Zero-IO**: FFmpeg frames are streamed directly into memory via `stdout` pipes.
            - **CIELAB**: Color analysis uses perceptually uniform clustering for superior accuracy.
            """)

if __name__ == "__main__":
    demo.launch()
