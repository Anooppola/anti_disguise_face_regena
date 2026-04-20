"""
preprocessing.py - Image preprocessing utilities
"""

import logging
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ─── Single image pipeline (for inference) ─────────────────────────────────────

_INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

_TO_PIL = transforms.ToPILImage()


def preprocess_image(image: Union[Image.Image, str], image_size: int = 256) -> torch.Tensor:
    """
    Convert a PIL image (or path) to a normalized tensor [-1, 1].
    Returns: (1, 3, image_size, image_size)
    """
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image or path, got {type(image)}")

    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return tfm(image).unsqueeze(0)   # (1, 3, H, W)


def postprocess_tensor(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (1, 3, H, W) or (3, H, W) tensor in [-1, 1] to a PIL Image.
    """
    t = tensor.squeeze(0).detach().cpu()
    t = (t * 0.5 + 0.5).clamp(0, 1)   # → [0, 1]
    return _TO_PIL(t)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a (1, 3, H, W) or (3, H, W) tensor in [-1, 1] to a uint8 numpy array (H, W, 3).
    """
    img = postprocess_tensor(tensor)
    return np.array(img)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """De-normalize from [-1, 1] to [0, 1]."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)
