"""
tests/test_model.py - Unit tests for Generator and Discriminator
"""

import torch
import pytest
from src.model import Generator, Discriminator, VGGPerceptualLoss


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="module")
def dummy_batch(device):
    return torch.randn(1, 3, 256, 256, device=device)


class TestGenerator:
    def test_output_shape(self, device, dummy_batch):
        G = Generator().to(device)
        G.eval()
        with torch.no_grad():
            out = G(dummy_batch)
        assert out.shape == (1, 3, 256, 256), f"Unexpected shape: {out.shape}"

    def test_output_range(self, device, dummy_batch):
        G = Generator().to(device)
        G.eval()
        with torch.no_grad():
            out = G(dummy_batch)
        assert out.min() >= -1.01 and out.max() <= 1.01, \
            "Output should be in [-1, 1] (Tanh)"

    def test_gradient_flow(self, device):
        G = Generator().to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        out = G(x)
        loss = out.mean()
        loss.backward()
        # At least one parameter should have a gradient
        assert any(p.grad is not None for p in G.parameters())


class TestDiscriminator:
    def test_output_shape(self, device, dummy_batch):
        D = Discriminator().to(device)
        D.eval()
        with torch.no_grad():
            out = D(dummy_batch, dummy_batch)
        # PatchGAN output for 256×256 input → (1, 1, 30, 30)
        assert out.shape == (1, 1, 30, 30), f"Unexpected shape: {out.shape}"

    def test_gradient_flow(self, device):
        D = Discriminator().to(device)
        x = torch.randn(1, 3, 256, 256, device=device)
        out = D(x, x)
        loss = out.mean()
        loss.backward()
        assert any(p.grad is not None for p in D.parameters())


class TestVGGPerceptualLoss:
    def test_loss_positive(self, device, dummy_batch):
        vgg = VGGPerceptualLoss(device)
        gen  = torch.randn_like(dummy_batch)
        real = torch.randn_like(dummy_batch)
        loss = vgg(gen, real)
        assert loss.item() >= 0

    def test_same_input_zero_loss(self, device, dummy_batch):
        vgg = VGGPerceptualLoss(device)
        loss = vgg(dummy_batch, dummy_batch)
        assert loss.item() < 1e-5
