"""
Activation heatmap generation for the shot-type classifier natively using PyTorch.

Produces spatial heatmaps showing which regions of an image
most influenced the ResNet-50's spatial feature layers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
from matplotlib.ticker import NullLocator

from .utils import discover_images, ensure_directory
from . import SHOT_TYPES


class ActivationPurger:
    """A simple PyTorch Forward Hook to capture intermediate feature maps."""
    def __init__(self, target_layer: torch.nn.Module):
        self.hook = target_layer.register_forward_hook(self.hook_fn)
        self.features = None

    def hook_fn(self, module, input, output):
        self.features = output.detach().cpu()
        
    def remove(self):
        self.hook.remove()


def show_heatmap(
    img: Image.Image,
    hm: torch.Tensor,
    path: Path,
    predicted_class: str,
    idx: int,
    ax: plt.Axes,
    interpolation="spline16",
    alpha=0.5,
):
    """Render and save a heatmap overlay on an image."""
    ax.clear()
    ax.set_axis_off()
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())

    # Draw original image
    ax.imshow(img)
    
    # Resize spatial activation tensor (C, H, W) -> (H, W)
    # The original was squished to 666x375
    hm_numpy = hm.numpy()

    # Draw Heatmap
    ax.imshow(
        hm_numpy, alpha=alpha, extent=(0, img.width, img.height, 0),
        interpolation=interpolation, cmap="YlOrRd",
    )

    fname = f"{predicted_class}_{str(idx + 1)}_heatmap.png"
    plt.savefig(path / fname, bbox_inches="tight", pad_inches=0, dpi=800)


def save_img(img: Image.Image, path: Path, predicted_class: str, idx: int, ax: plt.Axes):
    """Save an original image alongside its heatmap."""
    ax.clear()
    ax.imshow(img)
    ax.set_axis_off()
    ax.xaxis.set_major_locator(NullLocator())
    ax.yaxis.set_major_locator(NullLocator())

    fname = f"{predicted_class}_{str(idx + 1)}.png"
    plt.savefig(path / fname, bbox_inches="tight", pad_inches=0, dpi=800)


def generate_heatmaps(
    path_base: Optional[str] = None,
    path_img: str = ".",
    path_hms: Optional[str] = None,
    alpha: float = 0.5,
):
    """Generate activation heatmaps for all images in a directory.

    This is the main entry point for heatmap generation.

    Args:
        path_base: Path to the shot-type-classifier directory (legacy).
            If None, auto-downloads the model.
        path_img: Path to the image directory.
        path_hms: Optional path to save heatmaps. Defaults to path_img.
        alpha: Heatmap transparency (0.0 = invisible, 1.0 = opaque).
    """
    from .model import load_model
    from .transforms import get_inference_transforms

    path_img = Path(path_img)
    path_hms = Path(path_hms) if path_hms else path_img

    files = discover_images(path_img, recursive=True)

    if not files:
        print(f"No valid image files found in {path_img}")
        return

    # Load pure PyTorch model
    path_base_obj = Path(path_base) if path_base else None
    model, _ = load_model(path_base_obj)
    device = next(model.parameters()).device
    
    ensure_directory(path_hms)
    
    # We hook the backbone body (model[0])
    hook = ActivationPurger(model[0])

    tfms = get_inference_transforms()
    fig, ax = plt.subplots(figsize=(5, 3))

    try:
        for idx, file in enumerate(files):
            print(f"# {idx + 1} / {len(files)}")
            
            img = Image.open(str(file)).convert("RGB")
            
            # Legacy logic squished images exactly to 666x375 for visual heatmapping in matplotlib
            img_resized = img.resize((666, 375), Image.Resampling.BILINEAR)

            tensor = tfms(img).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(tensor)
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
                pred_idx = probs.argmax().item()
                predicted_class = SHOT_TYPES[pred_idx]

            # hook.features has shape (1, 2048, H, W). We take the mean across 2048 feature channels (dim 1)
            # The result is (1, H, W) -> we squeeze it to (H, W)
            acts = hook.features.squeeze(0)
            avg_acts = acts.mean(dim=0)

            save_img(img_resized, path_hms, predicted_class, idx, ax)
            show_heatmap(
                img_resized, avg_acts, path_hms, predicted_class, idx, ax,
                interpolation="spline16", alpha=alpha,
            )

    finally:
        hook.remove()
        plt.close(fig)
