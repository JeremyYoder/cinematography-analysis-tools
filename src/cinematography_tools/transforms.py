"""
Data augmentation and preprocessing transforms.

Uses standard torchvision.transforms instead of legacy fastai.
"""

from __future__ import annotations

import torchvision.transforms as T

# Standard ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_inference_transforms(size: int = 224) -> T.Compose:
    """Get the standard preprocessing transforms for model inference.

    Args:
        size: Target image size.

    Returns:
        A torchvision Compose object.
    """
    return T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def xtra_tfms(base_size: int = 75 * 5) -> list:
    """Legacy shim: returns custom training transformations.
    (Converted to torchvision equivalents where possible)
    """
    box_dim = int(base_size / 4)
    return [
        T.RandomErasing(p=0.8, scale=(0.02, 0.33), value='random'),
        T.RandomPerspective(distortion_scale=0.2, p=0.5),
        T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    ]


def get_tfms(*args, **kwargs):
    """Legacy shim: returns a tuple of (training_transforms, validation_transforms)."""
    train_tfms = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        *xtra_tfms(),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    valid_tfms = get_inference_transforms(224)
    
    return train_tfms, valid_tfms
