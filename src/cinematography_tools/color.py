"""
Color palette extraction in CIELAB color space.

Uses K-means clustering on CIELAB-converted pixels to extract
perceptually accurate dominant color palettes from images.
CIELAB is used because Euclidean distances in CIELAB correspond
more closely to perceived color differences than in RGB or HSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB values to CIELAB color space.

    Uses the D65 illuminant standard. Input RGB values should be
    in [0, 255] range (uint8).

    Args:
        rgb: Array of shape (N, 3) with RGB values in [0, 255].

    Returns:
        Array of shape (N, 3) with L*, a*, b* values.
    """
    # Normalize to [0, 1]
    rgb_norm = rgb.astype(np.float64) / 255.0

    # Apply sRGB gamma correction (linearize)
    mask = rgb_norm > 0.04045
    rgb_linear = np.where(mask, ((rgb_norm + 0.055) / 1.055) ** 2.4, rgb_norm / 12.92)

    # Convert to XYZ (D65 illuminant)
    # sRGB to XYZ matrix
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = rgb_linear @ M.T

    # D65 reference white point
    xyz_ref = np.array([0.95047, 1.00000, 1.08883])
    xyz_norm = xyz / xyz_ref

    # XYZ to CIELAB
    epsilon = 0.008856
    kappa = 903.3

    mask = xyz_norm > epsilon
    f = np.where(mask, np.cbrt(xyz_norm), (kappa * xyz_norm + 16.0) / 116.0)

    L = 116.0 * f[:, 1] - 16.0
    a = 500.0 * (f[:, 0] - f[:, 1])
    b = 200.0 * (f[:, 1] - f[:, 2])

    return np.column_stack([L, a, b])


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIELAB values back to RGB.

    Args:
        lab: Array of shape (N, 3) with L*, a*, b* values.

    Returns:
        Array of shape (N, 3) with RGB values in [0, 255] as uint8.
    """
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    # CIELAB to XYZ
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    epsilon = 0.008856
    kappa = 903.3

    xr = np.where(fx ** 3 > epsilon, fx ** 3, (116.0 * fx - 16.0) / kappa)
    yr = np.where(L > kappa * epsilon, ((L + 16.0) / 116.0) ** 3, L / kappa)
    zr = np.where(fz ** 3 > epsilon, fz ** 3, (116.0 * fz - 16.0) / kappa)

    # D65 reference white
    xyz = np.column_stack([xr, yr, zr]) * np.array([0.95047, 1.00000, 1.08883])

    # XYZ to linear sRGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ])
    rgb_linear = xyz @ M_inv.T

    # Apply sRGB gamma
    rgb_linear = np.clip(rgb_linear, 0, 1)
    mask = rgb_linear > 0.0031308
    rgb = np.where(mask, 1.055 * (rgb_linear ** (1.0 / 2.4)) - 0.055, 12.92 * rgb_linear)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def extract_palette(
    image_path: Path,
    n_colors: int = 5,
    sample_size: int = 10000,
    max_iterations: int = 50,
) -> List[Dict]:
    """Extract a dominant color palette from an image using K-means in CIELAB space.

    Args:
        image_path: Path to the image file.
        n_colors: Number of dominant colors to extract.
        sample_size: Number of pixels to sample for clustering.
        max_iterations: Maximum K-means iterations.

    Returns:
        List of dicts, each with: rgb, lab, hex, percentage.
        Sorted by dominance (most dominant first).
    """
    from sklearn.cluster import KMeans

    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img).reshape(-1, 3)

    # Subsample for performance
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]

    # Convert to CIELAB for perceptually uniform clustering
    lab_pixels = rgb_to_lab(pixels)

    # K-means clustering in CIELAB space
    kmeans = KMeans(n_clusters=n_colors, max_iter=max_iterations, n_init=3, random_state=42)
    labels = kmeans.fit_predict(lab_pixels)
    centers_lab = kmeans.cluster_centers_

    # Calculate cluster sizes (dominance)
    unique, counts = np.unique(labels, return_counts=True)
    percentages = counts / len(labels) * 100

    # Convert centers back to RGB for display
    centers_rgb = lab_to_rgb(centers_lab)

    # Build palette sorted by dominance
    palette = []
    sort_idx = np.argsort(-percentages)
    for i in sort_idx:
        r, g, b = int(centers_rgb[i, 0]), int(centers_rgb[i, 1]), int(centers_rgb[i, 2])
        L, a, b_val = float(centers_lab[i, 0]), float(centers_lab[i, 1]), float(centers_lab[i, 2])
        palette.append({
            "rgb": (r, g, b),
            "lab": (round(L, 2), round(a, 2), round(b_val, 2)),
            "hex": f"#{r:02x}{g:02x}{b:02x}",
            "percentage": round(float(percentages[i]), 2),
        })

    return palette


def extract_palettes_batch(
    image_paths: List[Path],
    n_colors: int = 5,
) -> List[Dict]:
    """Extract color palettes for multiple images.

    Args:
        image_paths: List of image file paths.
        n_colors: Number of dominant colors per image.

    Returns:
        List of dicts with: file, palette, timestamp (if available).
    """
    results = []
    for path in image_paths:
        try:
            palette = extract_palette(path, n_colors=n_colors)
            results.append({
                "file": str(path.name),
                "palette": palette,
            })
        except Exception as exc:
            print(f"  Warning: Could not extract palette from {path.name}: {exc}")
            results.append({
                "file": str(path.name),
                "palette": [],
                "error": str(exc),
            })

    return results
