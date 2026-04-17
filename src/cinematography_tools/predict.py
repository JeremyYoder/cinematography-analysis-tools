"""
Shot-type prediction for still images natively using PyTorch.

Classifies images into one of five cinematic shot types:
LS (Long/Far Shot), FS (Full Shot), MS (Medium Shot),
CS (Close Shot), ECS (Extreme Close Shot).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image

from .utils import discover_images, ensure_directory
from . import SHOT_TYPES

warnings.filterwarnings("ignore", ".*default behavior*")

HIERARCHY_MAP = {"LS": 0, "FS": 1, "MS": 2, "CS": 3, "ECS": 4}
PRECALCULATED_HIERARCHY = [-HIERARCHY_MAP.get(cls, 999) for cls in SHOT_TYPES]


def predict_image(model: torch.nn.Module, image: Path | str | Image.Image, device: torch.device | None = None, tfms=None) -> Dict:
    """Predict the shot type of a single image using pure PyTorch."""
    if device is None:
        device = next(model.parameters()).device
    if tfms is None:
        from .transforms import get_inference_transforms
        tfms = get_inference_transforms()
    
    if isinstance(image, (Path, str)):
        img = Image.open(str(image)).convert("RGB")
        file_name = str(image)
    else:
        img = image.convert("RGB")
        file_name = "in-memory"
        
    tensor = tfms(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    all_preds = [(SHOT_TYPES[i], float(probs[i]) * 100) for i in range(len(SHOT_TYPES))]
    # ⚡ Bolt: Use O(1) array lookups instead of dictionary .get() inside max() key function to eliminate PyTorch inference loop overhead
    best_idx = max(range(len(SHOT_TYPES)), key=lambda i: (probs[i], PRECALCULATED_HIERARCHY[i]))
    best = (SHOT_TYPES[best_idx], float(probs[best_idx]) * 100)

    return {
        "shot_type": best[0],
        "confidence": best[1],
        "all_predictions": {cls: conf for cls, conf in all_preds},
        "file": file_name,
    }


def predict_images_batch(model: torch.nn.Module, images: List[Image.Image]) -> List[Dict]:
    """Predict shot types for a batch of images in parallel.

    This function utilizes the hardware much more effectively for video analysis
    by processing multiple frames in a single (N, 3, 224, 224) tensor.

    Args:
        model: PyTorch ResNet50 classifier.
        images: List of opened PIL Images.

    Returns:
        List of prediction dicts (shot_type, confidence).
    """
    if not images:
        return []

    from .transforms import get_inference_transforms
    device = next(model.parameters()).device
    tfms = get_inference_transforms()

    # Stack all images into a single batch tensor
    batch_tensor = torch.stack([tfms(img.convert("RGB")) for img in images]).to(device)

    with torch.no_grad():
        logits = model(batch_tensor)
        probs_batch = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()

    batch_results = []
    for probs in probs_batch:
        # ⚡ Bolt: Use O(1) array lookups instead of dictionary .get() inside max() key function to eliminate PyTorch inference loop overhead
        best_idx = max(range(len(SHOT_TYPES)), key=lambda i: (probs[i], PRECALCULATED_HIERARCHY[i]))
        best = (SHOT_TYPES[best_idx], float(probs[best_idx]) * 100)
        batch_results.append({
            "shot_type": best[0],
            "confidence": best[1],
        })

    return batch_results


def predict_batch(
    model: torch.nn.Module,
    image_dir: Path,
    output_path: Optional[Path] = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """Predict shot types for all images in a directory.

    Args:
        model: PyTorch ResNet50 classifier.
        image_dir: Directory containing images.
        output_path: Optional path to save predictions CSV.
        recursive: Search subdirectories recursively.

    Returns:
        DataFrame with columns: shot-type, prediction, shot.
    """
    image_dir = Path(image_dir)
    files = discover_images(image_dir, recursive=recursive)

    if not files:
        print(f"No valid image files found in {image_dir}")
        return pd.DataFrame(columns=["shot-type", "prediction", "shot"])

    print(f"Found {len(files)} images to process.")

    # ⚡ Bolt: Hoist invariant operations outside the processing loop
    device = next(model.parameters()).device
    from .transforms import get_inference_transforms
    tfms = get_inference_transforms()

    results = []
    for idx, file in enumerate(files):
        print(f"Processing image {idx + 1}/{len(files)}...")
        result = predict_image(model, file, device=device, tfms=tfms)
        results.append({
            "shot-type": result["shot_type"],
            "prediction": result["confidence"],
            "shot": str(file.relative_to(image_dir)),
        })

    df = pd.DataFrame(results)

    if output_path is not None:
        output_path = Path(output_path)
        ensure_directory(output_path.parent)
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return df


def run_predictions(
    path_base: Optional[str] = None,
    path_img: str = ".",
    path_preds: Optional[str] = None,
) -> pd.DataFrame:
    """High-level function to run predictions end-to-end.

    Loads the native PyTorch model, processes images, and saves results.

    Args:
        path_base: Path to the shot-type-classifier directory (legacy).
            If None, auto-downloads the model.
        path_img: Path to the image directory.
        path_preds: Optional path to save predictions CSV.

    Returns:
        DataFrame of predictions.
    """
    from .model import load_model

    path_base_obj = Path(path_base) if path_base else None
    
    # We now get a pure PyTorch model back, discarding the 'data' tuple
    model, _ = load_model(path_base_obj)

    output_path = Path(path_preds) / "preds.csv" if path_preds else Path(path_img) / "preds.csv"

    return predict_batch(model, Path(path_img), output_path=output_path)
