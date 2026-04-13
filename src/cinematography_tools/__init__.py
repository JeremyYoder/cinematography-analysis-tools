"""
Cinematography Analysis Tools

A package for classifying cinematic shot types, generating heatmaps,
and extracting CIELAB color palettes from film footage.

Usage:
    # CLI
    shot-classify predict --input ./frames/

    # As a library
    from cinematography_tools import predict, heatmap, video
"""

__version__ = "1.0.0"

SHOT_TYPES = ["CS", "ECS", "FS", "LS", "MS"]

SHOT_TYPE_LABELS = {
    "CS": "Close Shot",
    "ECS": "Extreme Close Shot",
    "FS": "Full Shot",
    "LS": "Long Shot",
    "MS": "Medium Shot",
}
