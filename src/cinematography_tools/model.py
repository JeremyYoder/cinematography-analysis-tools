"""
Model loading, auto-download, and caching.

Downloads the pretrained pure PyTorch ResNet-50 shot-type classifier weights
from GitHub Releases on first use and caches them locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

CACHE_DIR = Path.home() / ".cache" / "cinematography-tools"

# Note: The user MUST push resnet50_cinematography.pth to their Github Release!
MODEL_FILENAME = "resnet50_cinematography.pth"
GITHUB_RELEASE_URL = (
    f"https://github.com/JeremyYoder/cinematography-analysis-tools/releases/download/v1.0.0/{MODEL_FILENAME}"
)

# Wait! We need to fallback to the old pickle if the user hasn't uploaded the pure state dict yet,
# but we stripped FastAI, meaning we CAN'T load the old pickle!
# Thus, they must use the new file.


class AdaptiveConcatPool2d(nn.Module):
    """FastAI's custom Concat Pool"""
    def __init__(self, size=1):
        super().__init__()
        self.ap = nn.AdaptiveAvgPool2d(size)
        self.mp = nn.AdaptiveMaxPool2d(size)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def build_cinematography_resnet() -> nn.Module:
    """Builds the ResNet50 architecture simulating the FastAI output head."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Remove the standard ResNet avgpool and fc layers
    body = nn.Sequential(*list(model.children())[:-2])
    
    # FastAI's custom head
    head = nn.Sequential(
        AdaptiveConcatPool2d(1),
        Flatten(),
        nn.BatchNorm1d(4096, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.25, inplace=False),
        nn.Linear(in_features=4096, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=5, bias=True)
    )
    
    return nn.Sequential(body, head)


def get_cache_dir() -> Path:
    """Return (and create) the model cache directory."""
    cache = Path(os.environ.get("SHOT_CLASSIFIER_CACHE", str(CACHE_DIR)))
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def download_model(url: str = GITHUB_RELEASE_URL, dest: Path | None = None) -> Path:
    """Download model weights if not already cached."""
    import requests

    if dest is None:
        dest = get_cache_dir()
    else:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)

    model_path = dest / MODEL_FILENAME

    if model_path.exists():
        return model_path

    print(f"Downloading pure PyTorch weights to {model_path}...")
    print(f"  Source: {url}")

    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Failed to download model from {url}.\n"
            f"Ensure {MODEL_FILENAME} is uploaded to the Github Release.\n"
            f"Error: {exc}"
        ) from exc

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(model_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                pct = downloaded / total * 100
                bar = "█" * int(pct // 2.5) + "░" * (40 - int(pct // 2.5))
                sys.stdout.write(f"\r  [{bar}] {pct:5.1f}%")
                sys.stdout.flush()

    print(f"\n  ✅ Downloaded {downloaded / 1024 / 1024:.1f} MB")
    return model_path


def load_model(
    path_base: Path | None = None,
    device: str = "auto",
) -> torch.nn.Module:
    """Load the pretrained ResNet-50 shot-type classifier natively."""
    
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            
    device_obj = torch.device(device)

    # Automatically look for the pure pytorch weight file
    if path_base is not None:
        path_base = Path(path_base)
        legacy_model = path_base / "models" / MODEL_FILENAME
        local_model = path_base / MODEL_FILENAME
        
        if local_model.exists():
            model_path = local_model
        elif legacy_model.exists():
            model_path = legacy_model
        else:
            model_path = download_model()
    else:
        # Check if they have extract_model scratch output locally
        if Path(MODEL_FILENAME).exists():
            model_path = Path(MODEL_FILENAME)
        else:
            model_path = download_model()

    model = build_cinematography_resnet()
    
    print(f"Loading state dict from {model_path} into model...")
    state_dict = torch.load(model_path, map_location=device_obj, weights_only=True)
    model.load_state_dict(state_dict)
    
    model = model.to(device_obj)
    model.eval()

    return model, None  # None returned for data compatibility format if needed
