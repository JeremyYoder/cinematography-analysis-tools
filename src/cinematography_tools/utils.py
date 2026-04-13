"""
Shared utility functions for image discovery, path handling, and I/O.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def discover_images(
    directory: Path,
    recursive: bool = True,
) -> List[Path]:
    """Find all image files in a directory.

    Args:
        directory: Path to search for images.
        recursive: If True, search subdirectories recursively.

    Returns:
        Sorted list of image file paths.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")

    if recursive:
        files = [
            f for f in directory.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]
    else:
        files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

    return sorted(files)


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist.

    Args:
        path: Directory path to create.

    Returns:
        The path that was created or already existed.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_path_exists(path: str | Path, label: str = "Path") -> Path:
    """Validate that a path exists and return it as a Path object.

    Args:
        path: Path to validate.
        label: Human-readable label for error messages.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return p
