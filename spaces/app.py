"""
Gradio web interface for Cinematography Analysis Tools.

Deploy to HuggingFace Spaces or run locally with:
    pip install gradio
    python spaces/app.py
"""

import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw

from cinematography_tools import SHOT_TYPE_LABELS
from cinematography_tools.color import extract_palette, lab_to_rgb


def classify_image(image):
    """Classify a single image and return shot type with confidence.

    Returns label dict and heatmap overlay (placeholder without model).
    """
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
        return {}, image, f"⚠️ Model not available: {exc}\n\nUpload the model weights to enable classification."

    finally:
        temp_path.unlink(missing_ok=True)


def analyze_palette(image, n_colors=5):
    """Extract and display color palette from an image."""
    if image is None:
        return None, ""

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        img = Image.fromarray(image)
        img.save(f.name)
        temp_path = Path(f.name)

    try:
        palette = extract_palette(temp_path, n_colors=n_colors)

        # Create palette visualization
        swatch_w, swatch_h = 120, 80
        padding = 8
        total_w = n_colors * (swatch_w + padding) - padding
        viz = Image.new("RGB", (total_w, swatch_h + 40), (26, 26, 46))
        draw = ImageDraw.Draw(viz)

        for i, color in enumerate(palette):
            x = i * (swatch_w + padding)
            r, g, b = color["rgb"]
            draw.rounded_rectangle(
                [x, 0, x + swatch_w, swatch_h],
                radius=8, fill=(r, g, b),
            )
            draw.text(
                (x + swatch_w // 2, swatch_h + 8),
                f"{color['hex']}",
                fill=(224, 224, 224),
                anchor="mt",
            )
            draw.text(
                (x + swatch_w // 2, swatch_h + 24),
                f"{color['percentage']:.0f}%",
                fill=(160, 160, 160),
                anchor="mt",
            )

        # Build info text
        info = "### CIELAB Color Palette\n\n"
        info += "| Color | Hex | L\\* | a\\* | b\\* | Dominance |\n"
        info += "|-------|-----|-----|-----|-----|-----------|\n"
        for c in palette:
            L, a, b = c["lab"]
            info += f"| 🟧 | `{c['hex']}` | {L:.1f} | {a:.1f} | {b:.1f} | {c['percentage']:.1f}% |\n"

        return np.array(viz), info

    except Exception as exc:
        return None, f"Error: {exc}"

    finally:
        temp_path.unlink(missing_ok=True)


# ── Build Gradio Interface ──

with gr.Blocks(
    title="🎬 Cinematography Shot Type Classifier",
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 960px !important; }
    .header { text-align: center; margin-bottom: 16px; }
    """,
) as demo:
    gr.HTML("""
        <div class="header">
            <h1>🎬 Cinematography Analysis Tools</h1>
            <p style="color: #888; font-size: 1.1em;">
                Classify cinematic shot types with a pretrained ResNet-50 &bull;
                Extract CIELAB color palettes
            </p>
        </div>
    """)

    with gr.Tabs():
        # ── Tab 1: Shot Classification ──
        with gr.TabItem("🎯 Shot Classification"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=400,
                    )
                    classify_btn = gr.Button("Classify Shot Type", variant="primary", size="lg")

                with gr.Column(scale=1):
                    label_output = gr.Label(label="Shot Type Prediction", num_top_classes=5)
                    info_output = gr.Markdown(label="Details")

            classify_btn.click(
                fn=classify_image,
                inputs=[img_input],
                outputs=[label_output, gr.Image(visible=False), info_output],
            )

            gr.Examples(
                examples=[],  # Add example image paths here
                inputs=[img_input],
            )

        # ── Tab 2: Color Palette ──
        with gr.TabItem("🎨 Color Palette"):
            with gr.Row():
                with gr.Column(scale=1):
                    pal_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=400,
                    )
                    n_colors = gr.Slider(
                        minimum=3, maximum=10, value=5, step=1,
                        label="Number of Colors",
                    )
                    palette_btn = gr.Button("Extract Palette", variant="primary", size="lg")

                with gr.Column(scale=1):
                    palette_viz = gr.Image(label="Color Palette", height=160)
                    palette_info = gr.Markdown(label="Palette Details")

            palette_btn.click(
                fn=analyze_palette,
                inputs=[pal_input, n_colors],
                outputs=[palette_viz, palette_info],
            )

        # ── Tab 3: About ──
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## Shot Types

            This model recognizes **5 cinematic shot types**:

            | Shot Type | Description |
            |-----------|-------------|
            | **LS** — Long Shot | Wide establishing shots showing full environment |
            | **FS** — Full Shot | Subject visible from head to toe |
            | **MS** — Medium Shot | Subject from waist up |
            | **CS** — Close Shot | Subject's face and shoulders |
            | **ECS** — Extreme Close Shot | Tight framing on a detail |

            ## Color Analysis

            Color palettes are extracted using **K-means clustering in CIELAB color space**.
            CIELAB is used because Euclidean distances in CIELAB space correspond more
            closely to perceived color differences than in RGB or HSV.

            ## CLI Usage

            ```bash
            pip install cinematography-analysis-tools
            shot-classify predict --input ./frames/
            shot-classify analyze-video --input movie.mp4
            shot-classify palette --input photo.jpg
            ```

            [GitHub Repository](https://github.com/JeremyYoder/cinematography-analysis-tools)
            """)


if __name__ == "__main__":
    demo.launch()
