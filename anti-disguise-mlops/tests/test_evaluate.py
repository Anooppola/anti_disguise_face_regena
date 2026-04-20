"""
tests/test_evaluate.py - Tests for PSNR, SSIM, and discriminator metrics
"""

import torch
import pytest
from src.evaluate import compute_psnr, compute_ssim, compute_discriminator_metrics


def make_batch(seed=0):
    torch.manual_seed(seed)
    return torch.rand(2, 3, 256, 256) * 2 - 1  # [-1, 1]


class TestPixelMetrics:
    def test_psnr_identical_images(self):
        x = make_batch(0)
        psnr = compute_psnr(x, x)
        assert psnr > 40, f"PSNR of identical images should be very high, got {psnr:.2f}"

    def test_psnr_different_images(self):
        x = make_batch(0)
        y = make_batch(1)
        psnr = compute_psnr(x, y)
        assert 0 < psnr < 40

    def test_ssim_identical_images(self):
        x = make_batch(0)
        ssim = compute_ssim(x, x)
        assert ssim > 0.99, f"SSIM of identical images should be ~1.0, got {ssim:.4f}"

    def test_ssim_different_images(self):
        x = make_batch(0)
        y = make_batch(1)
        ssim = compute_ssim(x, y)
        assert 0 < ssim < 1.0


class TestDiscriminatorMetrics:
    def test_perfect_discriminator(self):
        real_preds = [1.0, 0.9, 0.8]
        fake_preds = [0.0, 0.1, 0.2]
        metrics = compute_discriminator_metrics(real_preds, fake_preds)
        assert metrics["precision"] == 1.0
        assert metrics["recall"]    == 1.0
        assert metrics["f1_score"]  == 1.0

    def test_metrics_keys_present(self):
        metrics = compute_discriminator_metrics([0.7, 0.8], [0.3, 0.2])
        assert set(metrics.keys()) == {"precision", "recall", "f1_score"}
