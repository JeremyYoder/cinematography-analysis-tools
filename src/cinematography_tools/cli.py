"""
Command-line interface for cinematography-analysis-tools.

Usage::

    shot-classify predict  --input ./frames/ --output ./results/preds.csv
    shot-classify heatmap  --input ./frames/ --output ./heatmaps/ --alpha 0.8
    shot-classify analyze-video --input movie.mp4 --output ./analysis/ --sample-rate 2
    shot-classify palette  --input ./frames/ --output palette.json --colors 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def cmd_predict(args):
    """Run shot-type prediction on a directory of images."""
    from .predict import run_predictions

    run_predictions(
        path_base=args.model_path,
        path_img=args.input,
        path_preds=args.output,
    )


def cmd_heatmap(args):
    """Generate activation heatmaps."""
    from .heatmap import generate_heatmaps

    generate_heatmaps(
        path_base=args.model_path,
        path_img=args.input,
        path_hms=args.output,
        alpha=args.alpha,
    )


def cmd_analyze_video(args):
    """Analyze a video file sequentially using zero-IO streaming."""
    import pandas as pd
    from PIL import Image
    from .color import extract_palette
    from .predict import predict_image
    from .timeline import build_timeline, generate_summary, visualize_timeline
    from .utils import ensure_directory
    from .video import stream_frames, get_video_info

    video_path = Path(args.input)
    output_dir = ensure_directory(Path(args.output))

    print(f"🎬 Streaming Analysis: {video_path.name}")
    print("=" * 60)

    info = get_video_info(video_path)
    print(f"  Video: {info['width']}x{info['height']} @ {info['fps']:.1f}fps")
    print(f"  Duration: {info['duration']:.1f}s | Codec: {info['codec']}")
    print()

    from .model import load_model
    path_base_obj = Path(args.model_path) if args.model_path else None
    model, _ = load_model(path_base_obj)

    # Initialize storage
    predictions_list = []
    palettes = []
    frames_extracted = 0

    # Number of frames to trigger a palette extraction (roughly ~50 palettes total)
    expected_frames = int(info["duration"] * args.sample_rate)
    palette_interval = max(1, expected_frames // 50)

    print("Step 1/2: In-Memory Streaming & Classification...")
    for timestamp, frame_arr in stream_frames(video_path, args.sample_rate, args.max_frames):
        # frame_arr is rgb24 numpy array. Convert to PIL for predict pipeline
        img = Image.fromarray(frame_arr)
        
        # Predict shot type
        result = predict_image(model, img)
        predictions_list.append({
            "shot-type": result["shot_type"],
            "prediction": result["confidence"],
            "timestamp": timestamp,
        })
        
        # Color extraction
        if frames_extracted % palette_interval == 0:
            pal = extract_palette(frame_arr, n_colors=args.colors)
            palettes.append({
                "timestamp": timestamp,
                "palette": pal,
            })
            
        frames_extracted += 1

    if not predictions_list:
        print("No frames extracted. Exiting.")
        return

    # Convert predictions to DataFrame
    predictions = pd.DataFrame(predictions_list)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    
    import json
    with open(output_dir / "palettes.json", "w") as f:
        json.dump(palettes, f, indent=2)

    print(f"  ✅ Extracted {frames_extracted} frames directly into memory")

    print("\nStep 2/2: Building timeline...")
    # build_timeline in timeline.py expects the dataframe to just have 'prediction' and 'shot-type' 
    # but we provided timestamp directly. Let's pass empty frames list.
    timeline = build_timeline([], predictions, args.sample_rate, info["duration"])
    timeline.to_csv(output_dir / "timeline.csv", index=False)
    print(f"  ✅ Timeline: {len(timeline)} shots detected")

    visualize_timeline(
        timeline,
        output_dir / "timeline.png",
        title=f"Shot Timeline — {video_path.stem}",
        video_duration=info["duration"],
        palettes=palettes if len(palettes) > 5 else None,
    )

    summary = generate_summary(
        timeline,
        output_dir / "summary.json",
        palettes=palettes,
    )

    print()
    print("=" * 60)
    print(f"🎬 Analysis Complete: {video_path.name}")
    print(f"   Total duration: {summary.get('total_duration', 0):.1f}s")
    print(f"   Total shots: {summary.get('total_shots', 0)}")
    print(f"   Avg shot duration: {summary.get('avg_shot_duration', 0):.1f}s")
    print()

    dist = summary.get("shot_distribution", {})
    if dist:
        print("   Shot Distribution:")
        for st, info_dict in sorted(dist.items(), key=lambda x: -x[1]["percentage"]):
            bar = "█" * int(info_dict["percentage"] / 2.5)
            print(f"   {st:>3s}  {bar:<40s} {info_dict['percentage']:5.1f}%  ({info_dict['count']} shots)")

    print()
    print(f"   Output: {output_dir}")
    print(f"   ├── predictions.csv")
    print(f"   ├── timeline.csv")
    print(f"   ├── timeline.png")
    print(f"   ├── palettes.json")
    print(f"   └── summary.json")


def cmd_palette(args):
    """Extract color palettes from images."""
    import json

    from .color import extract_palette, extract_palettes_batch
    from .utils import discover_images

    input_path = Path(args.input)

    if input_path.is_file():
        palette = extract_palette(input_path, n_colors=args.colors)
        result = {"file": str(input_path), "palette": palette}
    else:
        images = discover_images(input_path)
        if not images:
            print(f"No images found in {input_path}")
            return
        print(f"Extracting palettes from {len(images)} images...")
        result = extract_palettes_batch(images, n_colors=args.colors)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"✅ Palettes saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="shot-classify",
        description="🎬 Cinematography Analysis Tools — Classify shot types, generate heatmaps, and analyze videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  shot-classify predict --input ./frames/
  shot-classify heatmap --input ./frames/ --alpha 0.8
  shot-classify analyze-video --input movie.mp4 --output ./analysis/
  shot-classify palette --input photo.jpg --colors 8
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── predict ──
    p_pred = subparsers.add_parser("predict", help="Classify images by shot type")
    p_pred.add_argument("--input", "-i", required=True, help="Directory of images to classify")
    p_pred.add_argument("--output", "-o", default=None, help="Output directory for predictions CSV")
    p_pred.add_argument("--model-path", "-m", default=None, help="Path to model directory (auto-downloads if not set)")
    p_pred.set_defaults(func=cmd_predict)

    # ── heatmap ──
    p_heat = subparsers.add_parser("heatmap", help="Generate activation heatmaps")
    p_heat.add_argument("--input", "-i", required=True, help="Directory of images")
    p_heat.add_argument("--output", "-o", default=None, help="Output directory for heatmaps")
    p_heat.add_argument("--alpha", "-a", type=float, default=0.5, help="Heatmap transparency (default: 0.5)")
    p_heat.add_argument("--model-path", "-m", default=None, help="Path to model directory")
    p_heat.set_defaults(func=cmd_heatmap)

    # ── analyze-video ──
    p_video = subparsers.add_parser("analyze-video", help="Full video analysis pipeline")
    p_video.add_argument("--input", "-i", required=True, help="Path to video file")
    p_video.add_argument("--output", "-o", default="./analysis", help="Output directory (default: ./analysis)")
    p_video.add_argument("--sample-rate", "-r", type=float, default=2.0, help="Frames per second to sample (default: 2)")
    p_video.add_argument("--max-frames", type=int, default=None, help="Maximum frames to extract")
    p_video.add_argument("--colors", "-c", type=int, default=5, help="Colors per palette (default: 5)")
    p_video.add_argument("--model-path", "-m", default=None, help="Path to model directory")
    p_video.set_defaults(func=cmd_analyze_video)

    # ── palette ──
    p_pal = subparsers.add_parser("palette", help="Extract color palettes (CIELAB)")
    p_pal.add_argument("--input", "-i", required=True, help="Image file or directory")
    p_pal.add_argument("--output", "-o", default=None, help="Output JSON file")
    p_pal.add_argument("--colors", "-c", type=int, default=5, help="Number of colors (default: 5)")
    p_pal.set_defaults(func=cmd_palette)

    # ── serve ──
    def cmd_serve(args):
        from .api import serve
        serve(host=args.host, port=args.port)

    p_serve = subparsers.add_parser("serve", help="Boot the highly scalable native FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1", help="Host IP address")
    p_serve.add_argument("--port", "-p", type=int, default=8000, help="Port to run on")
    p_serve.set_defaults(func=cmd_serve)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
