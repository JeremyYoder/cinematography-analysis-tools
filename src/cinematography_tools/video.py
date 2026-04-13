"""
Video frame extraction using FFmpeg.

Extracts frames from video files at a configurable sample rate
using FFmpeg subprocess calls. No OpenCV dependency.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from .utils import ensure_directory


def check_ffmpeg() -> bool:
    """Check if FFmpeg is available on the system."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True, timeout=10,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_video_info(video_path: Path) -> Dict:
    """Get video metadata using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: duration, fps, width, height, codec.
    """
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(video_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            f"ffprobe failed for {video_path}. Is FFmpeg installed?\n"
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        ) from exc

    # Find the video stream
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break

    if video_stream is None:
        raise ValueError(f"No video stream found in {video_path}")

    # Parse frame rate (can be "24000/1001" format)
    fps_str = video_stream.get("r_frame_rate", "24/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 24.0
    else:
        fps = float(fps_str)

    duration = float(data.get("format", {}).get("duration", 0))

    return {
        "duration": duration,
        "fps": fps,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "codec": video_stream.get("codec_name", "unknown"),
        "total_frames": int(duration * fps) if duration > 0 else 0,
    }


def extract_frames(
    video_path: Path,
    output_dir: Path,
    sample_rate: float = 2.0,
    max_frames: Optional[int] = None,
    quality: int = 2,
) -> List[Path]:
    """Extract frames from a video using FFmpeg.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save extracted frames.
        sample_rate: Frames per second to extract (e.g., 2.0 = 2 fps).
        max_frames: Maximum number of frames to extract.
        quality: JPEG quality (2 = highest, 31 = lowest).

    Returns:
        Sorted list of extracted frame file paths.
    """
    if not check_ffmpeg():
        raise RuntimeError(
            "FFmpeg is required but not found.\n"
            "Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Linux)"
        )

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = ensure_directory(output_dir)

    # Build FFmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", f"fps={sample_rate}",
        "-q:v", str(quality),
        "-vsync", "vfr",
    ]

    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])

    # Output pattern
    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd.append(output_pattern)

    # Get video info for progress reporting
    info = get_video_info(video_path)
    expected_frames = int(info["duration"] * sample_rate) if info["duration"] > 0 else 0

    print(f"Extracting frames from {video_path.name}")
    print(f"  Duration: {info['duration']:.1f}s | Resolution: {info['width']}x{info['height']}")
    print(f"  Sample rate: {sample_rate} fps → ~{expected_frames} frames expected")

    try:
        subprocess.run(
            cmd,
            capture_output=True, check=True,
            timeout=max(300, info["duration"] * 2),
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg frame extraction failed:\n{exc.stderr.decode()}"
        ) from exc

    # Collect extracted frames
    frames = sorted(output_dir.glob("frame_*.jpg"))
    print(f"  ✅ Extracted {len(frames)} frames")

    return frames


def extract_frame_at_time(
    video_path: Path,
    timestamp: float,
    output_path: Path,
) -> Path:
    """Extract a single frame at a specific timestamp.

    Args:
        video_path: Path to the input video.
        timestamp: Time in seconds.
        output_path: Path to save the extracted frame.

    Returns:
        Path to the extracted frame.
    """
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(timestamp),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]

    subprocess.run(cmd, capture_output=True, check=True, timeout=30)
    return output_path
