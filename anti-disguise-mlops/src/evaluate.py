"""
evaluate.py - Evaluation metrics: PSNR, SSIM, and GAN Discriminator metrics
"""

import logging
import math
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


# ─── Pixel-level metrics ───────────────────────────────────────────────────────

def compute_psnr(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    PSNR between real and fake tensors in [-1, 1].
    Higher is better (dB).
    """
    real_np = ((real.detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
    fake_np = ((fake.detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)

    psnr_values = []
    for r, f in zip(real_np, fake_np):
        # r,f shape: (3, H, W) → (H, W, 3)
        r_hwc = r.transpose(1, 2, 0)
        f_hwc = f.transpose(1, 2, 0)
        psnr_values.append(skimage_psnr(r_hwc, f_hwc, data_range=255))

    return float(np.mean(psnr_values))


def compute_ssim(real: torch.Tensor, fake: torch.Tensor) -> float:
    """
    SSIM between real and fake tensors in [-1, 1].
    Range [0, 1]; higher is better.
    """
    real_np = ((real.detach().cpu().numpy() * 0.5 + 0.5)).clip(0, 1)
    fake_np = ((fake.detach().cpu().numpy() * 0.5 + 0.5)).clip(0, 1)

    ssim_values = []
    for r, f in zip(real_np, fake_np):
        r_hwc = r.transpose(1, 2, 0)
        f_hwc = f.transpose(1, 2, 0)
        ssim_values.append(
            skimage_ssim(r_hwc, f_hwc, channel_axis=2, data_range=1.0)
        )
    return float(np.mean(ssim_values))


# ─── Discriminator-level metrics ───────────────────────────────────────────────

def compute_discriminator_metrics(
    real_preds: List[float],
    fake_preds: List[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Precision, Recall, F1 from discriminator sigmoid output.
    real_preds: list of D(x, y)  predictions  (sigmoid)
    fake_preds: list of D(x, G(x)) predictions (sigmoid)
    """
    y_true = [1] * len(real_preds) + [0] * len(fake_preds)
    y_pred = [1 if p >= threshold else 0 for p in real_preds + fake_preds]

    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1_score":  f1_score(y_true, y_pred, zero_division=0),
    }


# ─── Full evaluation loop ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    generator,
    discriminator,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run a full evaluation pass over the validation set.
    Returns a dict of averaged metrics.
    """
    generator.eval()
    discriminator.eval()

    psnr_list, ssim_list = [], []
    real_preds_all, fake_preds_all = [], []

    for masked, unmasked in val_loader:
        masked   = masked.to(device)
        unmasked = unmasked.to(device)

        fake = generator(masked)

        # Pixel metrics
        psnr_list.append(compute_psnr(unmasked, fake))
        ssim_list.append(compute_ssim(unmasked, fake))

        # Discriminator predictions (flatten spatial dims, apply sigmoid)
        real_pred = torch.sigmoid(discriminator(masked, unmasked)).mean(dim=[1, 2, 3])
        fake_pred = torch.sigmoid(discriminator(masked, fake)).mean(dim=[1, 2, 3])
        real_preds_all.extend(real_pred.cpu().tolist())
        fake_preds_all.extend(fake_pred.cpu().tolist())

    disc_metrics = compute_discriminator_metrics(real_preds_all, fake_preds_all)

    results = {
        "psnr":      float(np.mean(psnr_list)),
        "ssim":      float(np.mean(ssim_list)),
        **disc_metrics,
    }

    logger.info(
        "Eval → PSNR: %.2f | SSIM: %.4f | F1: %.4f",
        results["psnr"], results["ssim"], results["f1_score"]
    )
    return results
